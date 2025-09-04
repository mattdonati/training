# Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import ctypes
import random
from types import MethodType
import atexit
import logging_utils
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import apex.contrib.torchsched
from apex.contrib.torchsched.passes import pre_grad_custom_pass

import torch
import torch._dynamo
import torch.distributed
from checkpoint_tools import MultiprocessCheckpointIO
from megatron.core import parallel_state
from mlperf_logging.mllog import constants
from nemo.collections.multimodal.models.text_to_image.stable_diffusion.ldm.ddpm import (
    MegatronLatentDiffusion,
)
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.core.optim.distributed_adam import create_distributed_pgs
from nemo.utils import logging
from nemo.utils.model_utils import inject_model_parallel_rank
from callbacks import SDCallback, mllogger
from nemo.utils.exp_manager import exp_manager
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.jit import no_torch_dynamo
from omegaconf.omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning import seed_everything
from lightning.pytorch.plugins.io import TorchCheckpointIO
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from lightning.pytorch.plugins import ClusterEnvironment
from lightning.pytorch.plugins.io.checkpoint_plugin import CheckpointIO
from lightning.fabric.utilities.distributed import _destroy_dist_connection
from lightning.fabric.plugins.environments import TorchElasticEnvironment


def l2_promote():
    _libcudart = ctypes.CDLL("libcudart.so")

    # Check what's the device limit for current device, should be 64 by default
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))

    # Set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    result = _libcudart.cudaDeviceSetLimit(
        ctypes.c_int(0x05),
        ctypes.c_int(128),
    )

    # Get the device limit again, should be 128
    result = _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    logging.info("L2 promotion: %d B", pValue[0])
    return result


@hydra_runner()
def main(cfg) -> None:
    if logging_utils.get_rank() == 0:
        mllogger.start(key=constants.INIT_START)

    if cfg.model.get("inductor", False):
        # Disable dynamic shape
        torch._dynamo.config.dynamic_shapes = False
        torch._dynamo.config.automatic_dynamic_shapes = False
        # Increase cache size
        torch._dynamo.config.cache_size_limit = 128
        # Disable tracing TE DPA entirely
        DotProductAttention.forward = no_torch_dynamo(recursive=True)(
            DotProductAttention.forward
        )
        if cfg.model.get("use_cudnn_layer_norm", False):
            torch._inductor.config.pre_grad_custom_pass = \
                pre_grad_custom_pass
        if cfg.model.get("use_torch_sched", False):
            apex.contrib.torchsched.set_default_backend("torchsched")

    # Promote L2 fetch to 128 bytes
    l2_promote()

    seed = random.SystemRandom().randint(0, 2**32 - 1)
    if logging_utils.get_rank() == 0:
        mllogger.event(key=constants.SEED, value=seed)
    seed_everything(seed, verbose=False)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    megatron_amp_O2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = cfg.model.optim.get("name") == "distributed_fused_adam"

    torch.backends.cuda.matmul.allow_tf32 = True

    plugins = []

    class NLPDDPStrategySaveOnAllNodes(NLPDDPStrategy):
        """
        Override strategy's save_checkpoint method.

        We want to save the model checkpoint on each local_rank=0,
        in order to avoid copying them to shared storage later.

        In order to be able to keep using the Nemo automatic checkpointing
        and do this, we need to make this change at the strategy level.
        """
        def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: ClusterEnvironment = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        no_ddp_communication_hook: bool = False,
        nccl_communicator_config_path: Optional[str] = None,
        sharp: bool = False,
        dist_ckpt_parallel_save: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
        ) -> None:
            super().__init__(parallel_devices,
                             cluster_environment,
                             checkpoint_io,
                             no_ddp_communication_hook,
                             nccl_communicator_config_path,
                             sharp,
                             dist_ckpt_parallel_save,
                             **kwargs)

            # will be true only for 0
            self.is_local_zero = not int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

        # https://github.com/NVIDIA/NeMo/blob/89e8699926402ebf5fb04a2e004798eb64ab7740/nemo/collections/nlp/parts/nlp_overrides.py#L381
        def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
        ) -> None:
            super().save_checkpoint(checkpoint, filepath, storage_options=storage_options)

            # if use distributed checkpointing, saving is called by all ranks and megatron decides which to save on
            if not self.use_distributed_checkpointing:
                filepath = inject_model_parallel_rank(filepath)

                # super has already saved on global zero, so save on all other local zero
                if self.is_local_zero and not self.is_global_zero:
                    self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)

    strategy = NLPDDPStrategySaveOnAllNodes(
        # we don't use DDP for async grad allreduce
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    if cfg.model.precision in [16, "bf16", "16-mixed", "bf16-mixed"]:
        scaler = None
        if cfg.model.precision in [16, "16-mixed"]:
            scaler = GradScaler(
                init_scale=cfg.model.get("native_amp_init_scale", 65536.0),
                growth_interval=cfg.model.get(
                    "native_amp_growth_interval",
                    1000,
                ),
                hysteresis=cfg.model.get("hysteresis", 2),
            )
        if megatron_amp_O2 and not with_distributed_adam:
            plugins.append(
                MegatronHalfPrecisionPlugin(
                    precision=cfg.model.precision, device="cuda", scaler=scaler
                )
            )
        else:
            plugins.append(
                PipelineMixedPrecisionPlugin(
                    precision=cfg.model.precision, device="cuda", scaler=scaler
                )
            )

    if cfg.get("cluster_type", None) == "BCP":
        plugins.append(TorchElasticEnvironment())


    # Sample inputs for warmup
    assert cfg.model.first_stage_key == "images_moments" and \
        cfg.model.cond_stage_key == "clip_encoded", \
        "Expect images_moments and clip_encoded to warmup."
    n = cfg.model.micro_batch_size
    c = cfg.model.channels
    h = cfg.model.image_size
    d = cfg.model.unet_config.context_dim
    x = torch.randn((n, 2*c, h, h), dtype=torch.float32, device="cpu")
    cc = torch.randn((n, 77, d), dtype=torch.float32, device="cpu")
    inputs = {"images_moments": x, "clip_encoded": cc}

    callbacks = []
    cb = SDCallback(
        capture_iteration=cfg.model.capture_cudagraph_iters,
        warmup_iterations=cfg.model.capture_cudagraph_iters+1,
        warmup_inputs=inputs,
        logger=mllogger,
        train_log_interval=cfg.exp_manager.checkpoint_callback_params.every_n_train_steps,
        global_batch_size=cfg.model.global_batch_size,
        log_tracked_stats=True,
        tracked_stats_data=["throughput", "train_step_time", "max_memory_usage"],
    )
    cb.save_full_cfg(cfg)
    callbacks.append(cb)

    async_checkpoint_io = cfg.model.get("async_checkpoint_io", False)
    if async_checkpoint_io:
        # TODO: this is an assert to hint that FP8 has conficts with async checkpointing.
        assert cfg.model.unet_config.get("use_te_fp8", False) == False, \
            "Async checkpoint IO has conflict with FP8 extra_state data."
        checkpoint_io = MultiprocessCheckpointIO(
            checkpoint_io=TorchCheckpointIO(),
        )
        plugins.append(checkpoint_io)

    trainer = Trainer(
        plugins=plugins,
        strategy=strategy,
        callbacks=callbacks,
        enable_progress_bar=False,
        **cfg.trainer,
    )

    exp_manager(trainer, cfg.exp_manager)

    """
    # Update resume from checkpoint found by exp_manager
    if cfg.model.get("resume_from_checkpoint") is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = \
            trainer._checkpoint_connector.resume_from_checkpoint_fit_path
    logging.info(
        "Resuming training from checkpoint: %s",
        resume_from_checkpoint,
    )
    """

    trainer._checkpoint_connector = _CheckpointConnector(trainer)

    # Re-order communicator
    def get_setup_distributed_func(cfg):
        def setup_distributed_func(
            self, global_rank: int = None, world_size: int = None
        ) -> None:
            self._orig_setup_distributed(global_rank, world_size)

            group = parallel_state.get_data_parallel_group()
            if cfg.model.optim.get("name") == "distributed_fused_adam":
                distribute_within_nodes = cfg.model.optim.get("distribute_within_nodes", False)
                distributed_size = cfg.model.optim.get("distributed_size", None)
                if distribute_within_nodes:
                    if distributed_size is not None and distributed_size != torch.cuda.device_count():
                        raise ValueError("Inconsistent distributed_size value")
                    distributed_size = torch.cuda.device_count()
                if distributed_size is not None:
                    dist_pg_infos = create_distributed_pgs(distributed_size=distributed_size)
                    if dist_pg_infos:
                        group = dist_pg_infos['redundant_process_group']

            dummy = torch.randn(64, device="cuda", dtype=torch.float16)
            logging.info(
                "Warmup allreduce with communicator at %x, size %d",
                id(group),
                group.size(),
            )
            for _ in range(20):
                torch.distributed.all_reduce(dummy, group=group)

            # Prevent following communicators to lock the tree
            os.environ["NCCL_SHARP_DISABLE"] = "1"
            os.environ["NCCL_COLLNET_ENABLE"] = "0"

        return setup_distributed_func

    # Re-order communicator
    setup_distributed = get_setup_distributed_func(cfg)
    trainer.strategy._orig_setup_distributed = trainer.strategy.setup_distributed
    trainer.strategy.setup_distributed = MethodType(setup_distributed, trainer.strategy)

    model = MegatronLatentDiffusion(cfg.model, trainer)
    if async_checkpoint_io:
        checkpoint_io.setup(model.state_dict())

    # Put on a side stream to meet the CUDA graph requirements
    with torch.cuda.stream(torch.cuda.Stream()):
        trainer.fit(model)

    if async_checkpoint_io:
        # Since we created checkpoint in a new process, we wait to make sure the
        # last checkpoint is saved
        checkpoint_io.teardown()

    trainer.strategy.barrier()
    atexit.unregister(_destroy_dist_connection)


if __name__ == "__main__":
    main()
