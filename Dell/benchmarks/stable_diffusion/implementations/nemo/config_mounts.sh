_cont_mounts="${LOGDIR}:/results"

if [[ "${USE_SYNTHETIC_DATA:-0}" -eq 0 ]]; then
    _cont_mounts="${_cont_mounts},${DATADIR}:/datasets,${COCODIR}:/coco2014/,${CHECKPOINT_CLIP}:/checkpoints/clip,${CHECKPOINT_FID}:/checkpoints/inception,${CHECKPOINT_SD}:/checkpoints/sd"
    mounts_to_verify="DATADIR:/datasets COCO_TSV:/coco2014/val2014_30k.tsv COCO_STATS:/coco2014/val2014_512x512_30k_stats.npz CHECKPOINT_CLIP:/checkpoints/clip/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/blobs/9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4 CHECKPOINT_FID:/checkpoints/inception/pt_inception-2015-12-05-6726825d.pth CHECKPOINT_SD:/checkpoints/sd/512-base-ema.ckpt"
else
    if [[ "${VERIFY_MOUNTS:-0}" -eq 1 ]]; then
        echo "Overriding VERIFY_MOUNTS to 0 because USE_SYNTHETIC_DATA is 1"
        export VERIFY_MOUNTS=0
    fi
fi

if [[ -n "${APILOG_DIR:-}" ]]; then
    APILOG_DIR=${APILOG_DIR}/${MODEL_FRAMEWORK}/${MODEL_NAME}/${DGXSYSTEM}
    mkdir -p ${APILOG_DIR}
    _cont_mounts="${_cont_mounts},${APILOG_DIR}:/logs"

    # Create JSON file for cuDNN
    JSON_MODEL_NAME="MLPERF_${MODEL_NAME}_${MODEL_FRAMEWORK}_train"
    JSON_README_LINK="${README_PREFIX}/${MODEL_NAME}/${MODEL_FRAMEWORK}/README.md"
    JSON_FMT='{model_name: $mn, readme_link: $rl, configs: {($dt): [$bs]}, sweep: {($dt): [$bs]}}'
    JSON_OUTPUT="${JSON_MODEL_NAME}.cudnn.json"
    jq -n --indent 4 --arg mn "${JSON_MODEL_NAME}" --arg rl "${JSON_README_LINK}" --arg dt "${APILOG_PRECISION}" --arg bs "${BATCHSIZE}" "$JSON_FMT" > "${APILOG_DIR}/${JSON_OUTPUT}"
fi

if [[ "${JET:-0}" -eq 1 ]]; then
    _cont_mounts="${_cont_mounts},${JET_DIR}:/root/.jet"
fi