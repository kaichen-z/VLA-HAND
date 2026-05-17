#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/chonghej/scratch/chonghej}"
PYTHON_BIN="${PYTHON_BIN:-/scratch/chonghej/conda_envs/vitra/bin/python}"
OPENTOUCH_DIR="${OPENTOUCH_DIR:-${REPO_ROOT}/thirdparty/OpenTouch-MIT-opentouch}"
OPENTOUCH_REPO="${OPENTOUCH_REPO:-https://github.com/OpenTouch-MIT/opentouch.git}"
RAW_ROOT="${RAW_ROOT:-/home/chonghej/scratch/chonghej/vla_touch/datasets/opentouch_raw}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/datasets/opentouch_official_retrieval_hf}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/opentouch_official_encoder_train}"
SMOKE_RAW_ROOT="${SMOKE_RAW_ROOT:-${RUN_ROOT}/smoke_raw}"
SMOKE_DATASET_ROOT="${SMOKE_DATASET_ROOT:-${RUN_ROOT}/smoke_hf_dataset}"
MODEL_NAME="${MODEL_NAME:-OpenTouch-DINOv3-B16-AllModalities}"
VISUAL_MODEL_NAME="${VISUAL_MODEL_NAME:-google/vit-base-patch16-224-in21k}"
TASK_TYPE="${TASK_TYPE:-vp2t}"
STAGE="${STAGE:-smoke}"
GPU="${GPU:-auto}"
IMAGE_WIDTH="${IMAGE_WIDTH:-224}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-224}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-20}"
STRIDE="${STRIDE:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-300}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
WORKERS="${WORKERS:-8}"
SMOKE_WORKERS="${SMOKE_WORKERS:-2}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-amp_bf16}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
VAL_FREQUENCY="${VAL_FREQUENCY:-5}"
REPORT_TO="${REPORT_TO:-tensorboard}"
NAME="${NAME:-opentouch_all_modalities_vp2t}"
PRESSURE_METHOD="${PRESSURE_METHOD:-none}"
PRESSURE_MAX="${PRESSURE_MAX:-3072.0}"
PRESSURE_INTERVALS="${PRESSURE_INTERVALS:-5}"
TMPDIR="${TMPDIR:-${SCRATCH_ROOT}/tmp}"
HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/hf_datasets_cache}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/xdg_cache}"

setup_runtime_dirs() {
  mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}"
  export TMPDIR HF_HOME HF_DATASETS_CACHE XDG_CACHE_HOME
}

choose_gpu() {
  if [[ "${GPU}" != "auto" ]]; then
    echo "${GPU}"
    return
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); score=$2+200*$3; if (NR==1 || score<best) {best=score; gpu=$1}} END {print gpu}'
}

ensure_opentouch_repo() {
  if [[ ! -d "${OPENTOUCH_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${OPENTOUCH_DIR}")"
    git clone --depth 1 "${OPENTOUCH_REPO}" "${OPENTOUCH_DIR}"
  fi
}

write_model_config() {
  mkdir -p "${OPENTOUCH_DIR}/src/opentouch/model_configs"
  cat > "${OPENTOUCH_DIR}/src/opentouch/model_configs/${MODEL_NAME}.json" <<'JSON'
{
    "embed_dim": 64,
    "visual_cfg": {
        "model_name": "__VISUAL_MODEL_NAME__",
        "freeze_backbone": true,
        "time_pool": "mean"
    },
    "tactile_cfg": {
        "encoder_type": "cnnet"
    },
    "pose_cfg": {
        "normalize_mode": "simple"
    },
    "enabled_modalities": ["visual", "tactile", "pose"],
    "fusion_method": "concat",
    "normalize": true
}
JSON
  sed -i "s#__VISUAL_MODEL_NAME__#${VISUAL_MODEL_NAME}#g" "${OPENTOUCH_DIR}/src/opentouch/model_configs/${MODEL_NAME}.json"
}

preflight() {
  test -x "${PYTHON_BIN}"
  test -d "${RAW_ROOT}"
  export PYTHONPATH="${OPENTOUCH_DIR}/src:${PYTHONPATH:-}"
  "${PYTHON_BIN}" - <<'PY'
import importlib
for name in ["torch", "torchvision", "datasets", "transformers", "h5py", "cv2", "PIL", "opentouch", "opentouch_train"]:
    importlib.import_module(name)
print("preflight imports ok")
PY
}

run_with_datasets_localfs_patch() {
  local mode="$1"
  shift
  "${PYTHON_BIN}" - "${mode}" "$@" <<'PY'
import runpy
import sys

import datasets.arrow_dataset
import datasets.builder
import datasets.filesystems
import datasets.load


def is_remote_filesystem_compat(fs):
    protocol = getattr(fs, "protocol", None)
    if isinstance(protocol, (tuple, list, set)):
        return not any(item in {"file", "local"} for item in protocol)
    return protocol not in {None, "file", "local"}


for module in (
    datasets.arrow_dataset,
    datasets.builder,
    datasets.filesystems,
    datasets.load,
):
    module.is_remote_filesystem = is_remote_filesystem_compat

mode = sys.argv[1]
if mode == "script":
    script = sys.argv[2]
    sys.argv = [script] + sys.argv[3:]
    runpy.run_path(script, run_name="__main__")
elif mode == "module":
    module = sys.argv[2]
    sys.argv = [module] + sys.argv[3:]
    runpy.run_module(module, run_name="__main__")
elif mode == "check_dataset":
    from datasets import load_from_disk

    dataset_path = sys.argv[2]
    dataset = load_from_disk(dataset_path)
    print(f"dataset_path={dataset_path}")
    print(f"num_rows={len(dataset)}")
    print(f"columns={dataset.column_names}")
else:
    raise SystemExit(f"Unknown patched runner mode: {mode}")
PY
}

build_dataset() {
  local input_dir="$1"
  local output_dir="$2"
  if [[ -f "${output_dir}/dataset_info.json" || -f "${output_dir}/state.json" ]]; then
    echo "Dataset already exists: ${output_dir}"
    return
  fi
  mkdir -p "$(dirname "${output_dir}")"
  export PYTHONPATH="${OPENTOUCH_DIR}:${OPENTOUCH_DIR}/src:${PYTHONPATH:-}"
  run_with_datasets_localfs_patch script "${OPENTOUCH_DIR}/build_retrieval_data.py" \
    --input-dir "${input_dir}" \
    --output-dir "${output_dir}" \
    --image-size "${IMAGE_WIDTH}" "${IMAGE_HEIGHT}" \
    --pressure-max "${PRESSURE_MAX}" \
    --pressure-intervals "${PRESSURE_INTERVALS}" \
    --pressure-method "${PRESSURE_METHOD}" \
    --num-workers "${WORKERS}" \
    --process-batch-size 512
}

check_dataset_load() {
  local dataset_dir="$1"
  export PYTHONPATH="${OPENTOUCH_DIR}:${OPENTOUCH_DIR}/src:${PYTHONPATH:-}"
  run_with_datasets_localfs_patch check_dataset "${dataset_dir}"
}

prepare_smoke_raw() {
  mkdir -p "${SMOKE_RAW_ROOT}"
  local first_hdf5
  first_hdf5="$(find "${RAW_ROOT}" -maxdepth 1 -type f -name '*.hdf5' | sort | head -1)"
  if [[ -z "${first_hdf5}" ]]; then
    echo "No HDF5 files found in ${RAW_ROOT}" >&2
    exit 1
  fi
  ln -sfn "${first_hdf5}" "${SMOKE_RAW_ROOT}/$(basename "${first_hdf5}")"
}

train_retrieval() {
  local train_data="$1"
  local run_name="$2"
  local epochs="$3"
  local batch_size="$4"
  local workers="$5"
  local selected_gpu="$6"

  mkdir -p "${RUN_ROOT}/logs"
  export PYTHONPATH="${OPENTOUCH_DIR}/src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${selected_gpu}"
  cd "${OPENTOUCH_DIR}"
  run_with_datasets_localfs_patch module opentouch_train.main \
    --train-data "${train_data}" \
    --model "${MODEL_NAME}" \
    --task-type "${TASK_TYPE}" \
    --batch-size "${batch_size}" \
    --lr "${LR}" \
    --epochs "${epochs}" \
    --precision "${PRECISION}" \
    --workers "${workers}" \
    --sequence-length "${SEQUENCE_LENGTH}" \
    --stride "${STRIDE}" \
    --report-to "${REPORT_TO}" \
    --save-frequency "${SAVE_FREQUENCY}" \
    --val-frequency "${VAL_FREQUENCY}" \
    --logs "${RUN_ROOT}/logs" \
    --name "${run_name}" \
    --save-most-recent
}

main() {
  setup_runtime_dirs
  mkdir -p "${RUN_ROOT}"
  ensure_opentouch_repo
  write_model_config
  preflight
  local selected_gpu
  selected_gpu="$(choose_gpu)"
  echo "Using GPU ${selected_gpu}"

  case "${STAGE}" in
    preflight)
      ;;
    build)
      build_dataset "${RAW_ROOT}" "${DATASET_ROOT}"
      ;;
    check)
      check_dataset_load "${DATASET_ROOT}"
      ;;
    smoke)
      prepare_smoke_raw
      WORKERS="${SMOKE_WORKERS}" build_dataset "${SMOKE_RAW_ROOT}" "${SMOKE_DATASET_ROOT}"
      train_retrieval "${SMOKE_DATASET_ROOT}" "${NAME}_smoke" "${SMOKE_EPOCHS}" "${SMOKE_BATCH_SIZE}" "${SMOKE_WORKERS}" "${selected_gpu}"
      ;;
    full)
      build_dataset "${RAW_ROOT}" "${DATASET_ROOT}"
      train_retrieval "${DATASET_ROOT}" "${NAME}" "${EPOCHS}" "${BATCH_SIZE}" "${WORKERS}" "${selected_gpu}"
      ;;
    *)
      echo "Unknown STAGE=${STAGE}; expected preflight, build, check, smoke, or full" >&2
      exit 2
      ;;
  esac
}

main "$@"
