# Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Tuple, Union

import torch
import os
import transformer_engine.pytorch.cpp_extensions as ext
import transformer_engine.pytorch.fp8 as fp8
import transformer_engine_torch as tex
from torch.nn.parameter import Parameter

from transformer_engine.pytorch.attention import (
    _flash_attn_version,
    _flash_attn_version_required,
    _flash_attn_max_version,
    _flash_attn_2_plus,
    _flash_attn_2_1_plus,
    _flash_attn_2_3_plus,
    _flash_attn_2_4_plus,
    _flash_attn_2_4_1_plus,
    _flash_attn_2_5_7_plus,
)
if _flash_attn_version >= _flash_attn_version_required:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as _flash_attn_forward
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as _flash_attn_backward

from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_qkvpacked,
)
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import requires_grad, supports_fp8_transposes

from fast_dropout_lib import dropout_add_fwd, dropout_bwd

_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = False
_2X_ACC_WGRAD = False

META_QKV = tex.FP8FwdTensors.GEMM1_OUTPUT
META_O = tex.FP8FwdTensors.GEMM2_INPUT
META_DO = tex.FP8BwdTensors.GRAD_INPUT2
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1
META_S = tex.FP8FwdTensors.GEMM2_OUTPUT
META_DP = tex.FP8BwdTensors.GRAD_INPUT1

use_flash_attn = os.getenv('USE_FLASH_ATTENTION', '0') == '1'

class _MHA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_weight_fp8: torch.Tensor,
        qkv_bias: torch.Tensor,
        proj_weight: torch.Tensor,
        proj_weight_fp8: torch.Tensor,
        proj_bias: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_attention_heads: int,
        p_dropout: float,
        hidden_dropout_prob: float,
        max_s: int,
        set_zero: bool,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        is_training: bool,
        ntokens: Any,
    ) -> torch.Tensor:
        assert inp.dim() == 2
        # Make sure input dimensions are compatible
        in_features = qkv_weight.shape[-1]
        h = num_attention_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1
        if b < 4 and b > 1:
            max_s = 512

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        inputmat_t = None
        if supports_fp8_transposes():
            inputmat = ext.cast_to_fp8(
                inp,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
            )
        else:
            inputmat, inputmat_t = ext.fp8_cast_transpose_fused(
                inp,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
            )

        M, Z, philox_unpacked, S_dmask = None, None, None, None
        softmax_lse, rng_state = None, None
        softmax_scale = None
        qkv_out, qkv_out_fp16 = None, None
        if not use_flash_attn:
            # FMHA
            qkv_out, _ = ext.fp8_gemm(
                qkv_weight_fp8._data,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                inputmat,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                torch.uint8,
                workspace,
                bias=qkv_bias,
                use_bias=True,
                out_index=META_QKV,
                fp8_meta_tensor=fp8_meta["scaling_fwd"],
                use_split_accumulator=_2X_ACC_FPROP,
                D_dtype=fp8_dtype_forward,
            )
            qkv_out = qkv_out.view(-1, 3, h, d)
            context_, aux_ctx_tensors = fused_attn_fwd_qkvpacked(
                is_training,
                max_s,
                cu_seqlens,
                qkv_out,
                fp8_dtype_forward,
                FusedAttnBackend["FP8"],
                attn_bias=None,
                d_scale_qkv=fp8_meta["scaling_fwd"].scale_inv,
                d_scale_qkv_offset=META_QKV,
                q_scale_s=fp8_meta["scaling_fwd"].scale,
                q_scale_s_offset=META_S,
                d_scale_s=fp8_meta["scaling_fwd"].scale_inv,
                d_scale_s_offset=META_S,
                q_scale_o=fp8_meta["scaling_fwd"].scale,
                q_scale_o_offset=META_O,
                amax_s=fp8_meta["scaling_fwd"].amax_history,
                amax_s_offset=META_S,
                amax_o=fp8_meta["scaling_fwd"].amax_history,
                amax_o_offset=META_O,
                attn_scale=None,
                dropout=p_dropout if is_training else 0,
                fast_zero_fill=set_zero,
                qkv_layout="t3hd",
                attn_bias_type="no_bias",
                attn_mask_type="padding",
                rng_gen=None,
            )
            M, Z, philox_unpacked = None, None, None
            if is_training:
                M, Z, philox_unpacked = aux_ctx_tensors
            context = context_.view(-1, in_features)

        else:
            qkv_out_fp16, _ = ext.fp8_gemm(
                qkv_weight_fp8._data,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                inputmat,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                torch.float16,
                workspace,
                bias=qkv_bias,
                use_bias=True,
                use_split_accumulator=_2X_ACC_FPROP,
            )
            qkv_out_fp16 = qkv_out_fp16.view(-1, 3, h, d)
            softmax_scale = d ** (-0.5)
            # out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state
            out = torch.empty((qkv_out_fp16.size(0), h, d), dtype=torch.float16, device=qkv_out_fp16.device)
            tex.mha_fill (out, cu_seqlens[-1])
            out, _, _, _, out_padded, softmax_lse, _, rng_state = _flash_attn_forward(
                qkv_out_fp16[:, 0],
                qkv_out_fp16[:, 1],
                qkv_out_fp16[:, 2],
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                p_dropout if is_training else 0,
                softmax_scale,
                causal=False,
                window_size=(-1, -1),
                alibi_slopes=None,
                return_softmax=False,
                zero_tensors=False,
                out=out,
            )
            context = out.view(-1, in_features)
            context = ext.cast_to_fp8(
                context,
                fp8_meta["scaling_fwd"],
                META_O,
                fp8_dtype_forward,
            )
            context_ = out_padded

        if is_training:
            acc = False
            out = torch.empty_like(inp)
        else:
            acc = True
            out = inp
            mask = None
        proj_out, _ = ext.fp8_gemm(
            proj_weight_fp8._data,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM2_WEIGHT,
            fp8_dtype_forward,
            context,
            fp8_meta["scaling_fwd"].scale_inv,
            META_O,
            fp8_dtype_forward,
            torch.float16,
            workspace,
            out = out,
            accumulate = acc,
            bias=proj_bias,
            use_bias=True,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        if is_training:
            proj_out, mask = dropout_add_fwd (proj_out, inp, hidden_dropout_prob)

        ctx.save_for_backward(
            inputmat if supports_fp8_transposes() else inputmat_t,
            qkv_weight_fp8,
            workspace,
            qkv_out, qkv_out_fp16,
            S_dmask, M, Z, philox_unpacked,
            softmax_lse, rng_state,
            cu_seqlens,
            context_,
            context,
            proj_weight_fp8,
            mask,
            fp8_meta["scaling_fwd"].scale.clone(),
            fp8_meta["scaling_fwd"].scale_inv.clone(),
        )
        ctx.fp8_meta = fp8_meta
        ctx.p_dropout = p_dropout
        ctx.hidden_dropout_prob = hidden_dropout_prob
        ctx.max_s = max_s
        ctx.softmax_scale = softmax_scale
        ctx.set_zero = set_zero
        ctx.hidden_size = in_features
        ctx.num_attention_heads = num_attention_heads
        ctx.ntokens = ntokens
        ctx.reduce_and_update_bwd_fp8_tensors = False
        if requires_grad(inp, qkv_weight, qkv_bias):
            ctx.reduce_and_update_bwd_fp8_tensors = (
                ctx.reduce_and_update_bwd_fp8_tensors
                or fp8.FP8GlobalStateManager.is_first_fp8_module()
            )

        return proj_out

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with torch.cuda.nvtx.range("_MHA_backward"):
            (
                inputmat,
                qkv_weight_fp8,
                workspace,
                qkv_out, qkv_out_fp16,
                S_dmask, M, Z, philox_unpacked,
                softmax_lse, rng_state,
                cu_seqlens,
                context,
                context_fp8,
                proj_weight_fp8,
                mask,
                fwd_scales,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )
            proj_grad_output = dropout_bwd(grad_output, mask, ctx.hidden_dropout_prob)
            if supports_fp8_transposes():
                proj_bgrad, proj_grad_output_c = tex.fp8_cast_dbias(
                    proj_grad_output,
                    ctx.fp8_meta["scaling_bwd"].scale,
                    ctx.fp8_meta["scaling_bwd"].amax_history,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    fp8_dtype_backward,
                    [-1, -1, 1],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                )
                wgrad_input = context_fp8.view(grad_output.shape)
                wgrad_layout = "NT"
                proj_grad_output = proj_grad_output_c
                dgrad_weight_input = proj_weight_fp8._data
                dgrad_layout = "NN"
            else:
                proj_bgrad, proj_grad_output_c, proj_grad_output_t = ext.fp8_cast_transpose_bgrad_fused(
                    proj_grad_output,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                )
                wgrad_input = tex.fp8_transpose(context_fp8.view(grad_output.shape), fp8_dtype_forward)
                wgrad_layout = "TN"
                proj_grad_output = proj_grad_output_t
                dgrad_weight_input = proj_weight_fp8.transpose_2d()
                dgrad_layout = "TN"

            # PROJ WGRAD
            proj_wgrad, _ = ext.fp8_gemm(
                wgrad_input,
                fwd_scale_inverses,
                META_O,
                fp8_dtype_forward,
                proj_grad_output,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                tex.FP8BwdTensors.GRAD_OUTPUT2,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                layout=wgrad_layout,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
            if not use_flash_attn:
                proj_dgrad, _ = ext.fp8_gemm(
                    dgrad_weight_input,
                    fwd_scale_inverses,
                    tex.FP8FwdTensors.GEMM2_WEIGHT,
                    fp8_dtype_forward,
                    proj_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                    torch.uint8,
                    workspace,
                    layout=dgrad_layout,
                    out_index=META_DO,
                    fp8_meta_tensor=ctx.fp8_meta["scaling_bwd"],
                    D_dtype=fp8_dtype_backward,
                    use_split_accumulator=_2X_ACC_DGRAD,
                )

                dqkv_fp8, *rest = fused_attn_bwd_qkvpacked(
                    ctx.max_s,
                    cu_seqlens,
                    qkv_out,
                    context,
                    proj_dgrad.view_as(context),
                    fp8_dtype_forward,
                    fp8_dtype_backward,
                    [M, Z, philox_unpacked],
                    FusedAttnBackend["FP8"],
                    d_scale_qkv=fwd_scale_inverses[META_QKV],  # d_scale_qkv,
                    d_scale_s=fwd_scale_inverses[META_S],  # d_scale_s,
                    d_scale_o=fwd_scale_inverses[META_O],  # d_scale_o,
                    d_scale_do=ctx.fp8_meta["scaling_bwd"].scale_inv[META_DO],  # d_scale_do
                    d_scale_dp=ctx.fp8_meta["scaling_bwd"].scale_inv[META_DP],  # d_scale_dp
                    q_scale_s=fwd_scales[META_S],  # q_scale_s
                    q_scale_dp=ctx.fp8_meta["scaling_bwd"].scale[META_DP],  # q_scale_ds
                    q_scale_dqkv=ctx.fp8_meta["scaling_bwd"].scale[
                        META_DQKV
                    ],  # q_scale_dqkv
                    amax_dp=ctx.fp8_meta["scaling_bwd"].amax_history[0][META_DP],  # amax_ds
                    amax_dqkv=ctx.fp8_meta["scaling_bwd"].amax_history[0][
                        META_DQKV
                    ],  # amax_dqkv
                    attn_scale=None,
                    dropout=ctx.p_dropout,
                    fast_zero_fill=ctx.set_zero,
                    qkv_layout="t3hd",
                    attn_bias_type="no_bias",
                    attn_mask_type="padding",
                )
                dqkv_grad_output_c = dqkv_fp8.view(-1, 3 * ctx.hidden_size)
                qkv_bgrad, dqkv_grad_output_t = ext.fp8_transpose_bgrad_fused(
                    dqkv_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"],
                    META_DQKV,
                    fp8_dtype_backward,
                    torch.float16,
                )

            else:
                proj_dgrad, _ = ext.fp8_gemm(
                    dgrad_weight_input,
                    fwd_scale_inverses,
                    tex.FP8FwdTensors.GEMM2_WEIGHT,
                    fp8_dtype_forward,
                    proj_grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                    torch.float16,
                    workspace,
                    layout=dgrad_layout,
                    use_split_accumulator=_2X_ACC_DGRAD,
                )
                dqkv = torch.empty_like(qkv_out_fp16)
                tex.mha_fill (dqkv, cu_seqlens[-1])
                _flash_attn_backward(
                    proj_dgrad.view(-1,16,64),
                    qkv_out_fp16[:, 0],
                    qkv_out_fp16[:, 1],
                    qkv_out_fp16[:, 2],
                    context,
                    softmax_lse,
                    dqkv[:, 0],
                    dqkv[:, 1],
                    dqkv[:, 2],
                    cu_seqlens,
                    cu_seqlens,
                    ctx.max_s,
                    ctx.max_s,
                    ctx.p_dropout,
                    ctx.softmax_scale,
                    causal=False,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    deterministic=False,
                    rng_state=rng_state,
                    zero_tensors=False,
                )
                qkv_bgrad, dqkv_grad_output_c = tex.fp8_cast_dbias(
                    dqkv.view(-1, 3 * ctx.hidden_size),
                    ctx.fp8_meta["scaling_bwd"].scale,
                    ctx.fp8_meta["scaling_bwd"].amax_history,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    fp8_dtype_backward,
                    [-1, -1, 1],
                    META_DQKV,
                    META_DQKV,
                    META_DQKV,
                )
            if supports_fp8_transposes():
                dgrad_weight_input = qkv_weight_fp8._data
                dqkv_grad_output = dqkv_grad_output_c
            else:
                dgrad_weight_input = qkv_weight_fp8.transpose_2d()
                dqkv_grad_output = dqkv_grad_output_t

                
            ####################################################################################
            # QKV DGRAD
            qkv_dgrad, _ = ext.fp8_gemm(
                dgrad_weight_input,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                out=grad_output,
                accumulate=True,
                layout=dgrad_layout,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            # QKV WGRAD
            qkv_wgrad, _ = ext.fp8_gemm(
                inputmat,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                dqkv_grad_output,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                layout=wgrad_layout,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            fp8.FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            qkv_dgrad,
            qkv_wgrad,
            None,
            qkv_bgrad,
            proj_wgrad,
            None,
            proj_bgrad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP8_MHA(TransformerEngineBaseModule):
    def __init__(self, config, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.h = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.d = self.hidden_size // self.h
        self.set_zero = True
        assert self.d * self.h == self.hidden_size, "Invalid hidden size/num_heads"

        self.qkv_weight = Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.qkv_bias = Parameter(
            torch.empty(
                self.hidden_size * 3,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.proj_weight = Parameter(
            torch.empty(
                self.hidden_size,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.proj_bias = Parameter(
            torch.empty(
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        with torch.no_grad():
            self.qkv_bias.zero_()
            self.qkv_weight.fill_(1.0)
            self.proj_bias.zero_()
            self.proj_weight.fill_(1.0)
        # workspace for cublasLt
        self.workspace = torch.empty(
            _CUBLASLT_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )

    def forward(
        self, inp: torch.Tensor, cu_seqlens, max_s, ntokens=None
    ) -> torch.Tensor:
        with self.prepare_forward(inp, True, num_gemms=4) as inp:
            weight1_fp8 = self.get_fp8_workspace(
                tensor=self.qkv_weight,
                fp8_meta_forward=True,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                cache_name="weight_qkv",
                update_workspace=True,
                skip_update_flag=None,
            )
            weight2_fp8 = self.get_fp8_workspace(
                tensor=self.proj_weight,
                fp8_meta_forward=True,
                fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
                cache_name="weight_proj",
                update_workspace=True,
                skip_update_flag=None,
            )
            out = _MHA.apply(
                inp,
                self.qkv_weight,
                weight1_fp8,
                self.qkv_bias,
                self.proj_weight,
                weight2_fp8,
                self.proj_bias,
                cu_seqlens,
                self.h,
                self.p_dropout,
                self.hidden_dropout_prob,
                max_s,
                self.set_zero,
                self.fp8_meta,
                self.workspace,
                self.training,
                ntokens,
            )

        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        print(
            "FP8_MHA is_first_microbatch None? {}".format(is_first_microbatch is None)
        )
        """Needs override."""


class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_weight_fp8: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_weight_fp8: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        activation_dtype: torch.dtype,
        is_training: bool,
        hidden_dropout_prob: float,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        ln_out_return, mu, rsigma = tex.layernorm_fwd(
            inp, ln_weight, ln_bias, eps, 0, False
        )
        ln_out_total = ln_out_return
        if supports_fp8_transposes():
            ln_out_total = ext.cast_to_fp8(
                ln_out_total,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
            )
        else:
            ln_out_total, ln_out_total_t = ext.fp8_cast_transpose_fused(
                ln_out_total,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
            )

        gelu_out, fc1_out = ext.fp8_gemm(
            fc1_weight_fp8._data,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            ln_out_total,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            torch.uint8,
            workspace,
            gelu=True,
            out_index=tex.FP8FwdTensors.GEMM2_INPUT,
            fp8_meta_tensor=fp8_meta["scaling_fwd"],
            bias=fc1_bias,
            use_bias=True,
            use_split_accumulator=_2X_ACC_FPROP,
            D_dtype=fp8_dtype_forward,
        )

        if is_training:
            acc = False
            out = torch.empty_like(ln_out_return)
        else:
            acc = True
            out = ln_out_return
            mask = None
        fc2_out, _ = ext.fp8_gemm(
            fc2_weight_fp8._data,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM2_WEIGHT,
            fp8_dtype_forward,
            gelu_out,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM2_INPUT,
            fp8_dtype_forward,
            torch.float16,
            workspace,
            out = out,
            accumulate = acc,
            bias=fc2_bias,
            use_bias=True,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        if is_training:
            fc2_out, mask = dropout_add_fwd (fc2_out, ln_out_return,
                                             hidden_dropout_prob)

        ctx.save_for_backward(
            inp,
            ln_weight,
            mu,
            rsigma,
            ln_out_total if supports_fp8_transposes() else ln_out_total_t,
            fc1_out,
            gelu_out,
            fc1_weight_fp8,
            fc2_weight_fp8,
            fc1_bias,
            fc2_bias,
            mask,
            workspace,
            fp8_meta["scaling_fwd"].scale_inv.clone(),
        )
        ctx.activation_dtype = activation_dtype
        ctx.hidden_dropout_prob = hidden_dropout_prob
        ctx.fp8_meta = fp8_meta
        ctx.inp_shape = inp.shape
        ctx.reduce_and_update_bwd_fp8_tensors = False
        if requires_grad(inp, fc1_weight, fc1_bias):
            ctx.reduce_and_update_bwd_fp8_tensors = (
                ctx.reduce_and_update_bwd_fp8_tensors
                or fp8.FP8GlobalStateManager.is_first_fp8_module()
            )

        fc2_out = fc2_out.view(-1, *inp.shape[1:-1], fc2_out.shape[-1])

        return fc2_out

    @staticmethod
    def backward(
        ctx, grad_output:torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with torch.cuda.nvtx.range("_LayerNormMLP_backward"):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out_total,
                fc1_out,
                gelu_out,
                fc1_weight_fp8,
                fc2_weight_fp8,
                fc1_bias,
                fc2_bias,
                mask,
                workspace,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            grad_output_mat = grad_output.view((-1, grad_output.shape[-1]))

            fp8_dtype_forward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            fc2_grad_output = dropout_bwd(grad_output_mat, mask, ctx.hidden_dropout_prob)
            if supports_fp8_transposes():
                fc2_bias_grad, grad_output_c = tex.fp8_cast_dbias(
                    fc2_grad_output,
                    ctx.fp8_meta["scaling_bwd"].scale,
                    ctx.fp8_meta["scaling_bwd"].amax_history,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    fp8_dtype_backward,
                    [-1, -1, 1],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                )
                dgrad_weight_input = fc2_weight_fp8._data
                dgrad_layout = "NN"
                wgrad_input = gelu_out
                wgrad_layout = "NT"
                grad_output = grad_output_c
            else:
                fc2_bias_grad, grad_output_c, grad_output_t = ext.fp8_cast_transpose_bgrad_fused(
                    fc2_grad_output,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                )

                dgrad_weight_input = fc2_weight_fp8.transpose_2d()
                dgrad_layout = "TN"
                wgrad_input = tex.fp8_transpose(gelu_out, fp8_dtype_forward)
                wgrad_layout = "TN"
                grad_output = grad_output_t

            # FC2 DGRAD
            fc2_dgrad, _ = ext.fp8_gemm(
                dgrad_weight_input,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                fp8_dtype_forward,
                grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                layout=dgrad_layout,
                use_split_accumulator=_2X_ACC_DGRAD,
            )

            # FC2 WGRAD
            fc2_wgrad, _ = ext.fp8_gemm(
                wgrad_input,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM2_INPUT,
                fp8_dtype_forward,
                grad_output,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                layout=wgrad_layout,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

            if supports_fp8_transposes():
                fc1_bias_grad, dgelu = tex.fp8_cast_dbias_dgelu(
                    fc2_dgrad,
                    fc1_out,
                    ctx.fp8_meta["scaling_bwd"].scale,
                    ctx.fp8_meta["scaling_bwd"].amax_history,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    fp8_dtype_backward,
                    [-1, -1, 1],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                )
                dgrad_weight_input = fc1_weight_fp8._data
                fc1_grad_output = dgelu
            else:
                fc1_bias_grad, dgelu, dgelu_t = ext.fp8_cast_transpose_bgrad_dgelu_fused(
                    fc2_dgrad,
                    fc1_out,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                )

                dgrad_weight_input = fc1_weight_fp8.transpose_2d()
                fc1_grad_output = dgelu_t

            # FC1 DGRAD
            fc1_dgrad, _ = ext.fp8_gemm(
                dgrad_weight_input,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dgelu,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                tex.FP8BwdTensors.GRAD_OUTPUT2,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                out=grad_output_mat,
                accumulate=True,
                layout=dgrad_layout,
                use_split_accumulator=_2X_ACC_DGRAD,
            )

            # FC1 WGRAD
            fc1_wgrad, _ = ext.fp8_gemm(
                ln_out_total,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                fc1_grad_output,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                tex.FP8BwdTensors.GRAD_OUTPUT2,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                layout=wgrad_layout,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

            d_ln_out = fc1_dgrad
            dxmat, dgamma, dbeta = tex.layernorm_bwd(
                d_ln_out, inputmat, mu, rsigma, ln_weight, 0, False
            )
        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            fp8.FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dxmat.view(ctx.inp_shape),
            dgamma,
            dbeta,
            fc1_wgrad,
            None,
            fc1_bias_grad,
            fc2_wgrad,
            None,
            fc2_bias_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LayerNormMLP(TransformerEngineBaseModule):
    """
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.

    Optimization parameters
    -----------------------
    params_dtype : torch.dtype, default = `torch.float32`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    """

    def __init__(
        self,
        config,
        eps: float = 1e-5,
        params_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # LN init
        self.eps = eps
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.hidden_size = config.hidden_size
        self.layer_norm_weight = Parameter(
            torch.empty(
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.layer_norm_bias = Parameter(
            torch.empty(
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        # FC1 init
        self.fc1_weight = Parameter(
            torch.empty(
                config.intermediate_size,
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        self.fc1_bias = Parameter(
            torch.empty(
                config.intermediate_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        # FC2 init
        self.fc2_weight = Parameter(
            torch.empty(
                config.hidden_size,
                config.intermediate_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        self.fc2_bias = Parameter(
            torch.empty(
                config.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )

        with torch.no_grad():
            self.layer_norm_bias.zero_()
            self.layer_norm_weight.fill_(1.0)
            self.fc1_bias.zero_()
            self.fc1_weight.fill_(1.0)
            self.fc2_bias.zero_()
            self.fc2_weight.fill_(1.0)

        # workspace for cublasLt
        self.workspace = torch.empty(
            _CUBLASLT_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )

    def forward(
        self, inp: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        """

        with self.prepare_forward(inp, True, num_gemms=2) as inp:
            fc1_fp8 = self.get_fp8_workspace(
                tensor=self.fc1_weight,
                fp8_meta_forward=True,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                cache_name="weight_fc1",
                update_workspace=True,
                skip_update_flag=None,
            )
            fc2_fp8 = self.get_fp8_workspace(
                tensor=self.fc2_weight,
                fp8_meta_forward=True,
                fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
                cache_name="weight_fc2",
                update_workspace=True,
                skip_update_flag=None,
            )

            out = _LayerNormMLP.apply(
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                self.fc1_weight,
                fc1_fp8,
                self.fc1_bias,
                self.fc2_weight,
                fc2_fp8,
                self.fc2_bias,
                self.eps,
                self.fp8_meta,
                self.workspace,
                self.activation_dtype,
                self.training,
                self.hidden_dropout_prob,
            )

        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        pass
