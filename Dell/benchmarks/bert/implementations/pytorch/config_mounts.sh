
_cont_mounts="\
${DATADIR_PHASE2}:/workspace/data_phase2,\
${CHECKPOINTDIR_PHASE1}:/workspace/phase1,\
${EVALDIR}:/workspace/evaldata,\
${LOGDIR}:/results"
_cont_mounts="${_cont_mounts}"
cont_mounts_to_verify="'/workspace/data_phase2', '/workspace/phase1', '/workspace/evaldata'"

if [[ "${NVTX_FLAG:-0}" -gt 0 ]] && [[ -d "${NVMLPERF_NSIGHT_LOCATION}" ]]; then
    _cont_mounts="${_cont_mounts},${NVMLPERF_NSIGHT_LOCATION}:/nsight"
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