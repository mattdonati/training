
_cont_mounts="${LOGDIR}:/results"
cont_mounts_to_verify="'/datasets/open-images-v6'"
if [ "${USE_SYNTHETIC_DATA:-0}" -eq 0 ]; then
    _cont_mounts="${DATADIR}:/datasets/open-images-v6,${_cont_mounts},${BACKBONE_DIR}:/root/.cache/torch"
else
    if [[ "${VERIFY_MOUNTS:-0}" -eq 1 ]]; then
        echo "Overriding VERIFY_MOUNTS to 0 because USE_SYNTHETIC_DATA is 1"
        export VERIFY_MOUNTS=0
    fi
fi

if [ "${EXTRA_MOUNTS:-}" != "" ]; then
    _cont_mounts="${_cont_mounts},${EXTRA_MOUNTS}"
fi

# API Logging
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