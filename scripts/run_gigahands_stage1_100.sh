#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPUS="${GPUS:-0,1,5}"
RUN_STEPS="${RUN_STEPS:-100}"
PER_GPU_BATCH="${PER_GPU_BATCH:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
WANDB_MODE="${WANDB_MODE:-offline}"
DATA_ROOT="${DATA_ROOT:-/home/chonghej/GigaHands/dataset/vitra_gigahands_demo}"
BASE_CONFIG="${BASE_CONFIG:-vitra/configs/human_pretrain_gigahands_demo.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/chonghej/vitra_runs/gigahands_demo_100}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
NPROC="${NPROC:-${#GPU_LIST[@]}}"
TOTAL_BATCH="${TOTAL_BATCH:-$((PER_GPU_BATCH * NPROC))}"
SAVE_STEPS="${SAVE_STEPS:-${RUN_STEPS}}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-0}"
TASK_NAME="${TASK_NAME:-gigahands_demo_stage1_100steps}"
GENERATED_CONFIG="${GENERATED_CONFIG:-${RUN_ROOT}/configs/human_pretrain_gigahands_100steps.json}"

mkdir -p "$(dirname "${GENERATED_CONFIG}")"

echo "Using repo: ${REPO_ROOT}"
echo "Using GPUs: ${GPUS}"
echo "Processes: ${NPROC}"
echo "Per-GPU batch: ${PER_GPU_BATCH}"
echo "Total batch: ${TOTAL_BATCH}"
echo "Run steps: ${RUN_STEPS}"
echo "Save steps: ${SAVE_STEPS}"
echo "Save checkpoint: ${SAVE_CHECKPOINT}"
echo "Data root: ${DATA_ROOT}"
echo "Generated config: ${GENERATED_CONFIG}"

python - <<PY
import json
from pathlib import Path

base_config = Path("${BASE_CONFIG}")
generated_config = Path("${GENERATED_CONFIG}")
run_root = Path("${RUN_ROOT}")

with base_config.open() as f:
    cfg = json.load(f)

cfg["task_name"] = "${TASK_NAME}"
cfg["batch_size"] = int("${PER_GPU_BATCH}")
cfg["total_batch_size"] = int("${TOTAL_BATCH}")
cfg["save_steps"] = int("${SAVE_STEPS}")
cfg["save_checkpoint"] = bool(int("${SAVE_CHECKPOINT}"))
cfg["epoch_save_interval"] = 100000
cfg["resume"] = False
cfg["output_root"] = str(run_root / "checkpoints")
cfg["log_root"] = str(run_root / "logs")
cfg["cache_root"] = str(run_root / "cache")

cfg["trainer"]["max_steps"] = int("${RUN_STEPS}")
cfg["trainer"]["lr_scheduler_type"] = "backbone-freeze-warmup"
cfg["trainer"]["llm_freeze_step"] = 5000

cfg["train_dataset"]["data_root_dir"] = "${DATA_ROOT}"
cfg["train_dataset"]["data_mix"] = "gigahands_demo_only"
cfg["train_dataset"]["num_workers"] = int("${NUM_WORKERS}")
cfg["train_dataset"]["prefetch_factor"] = None
cfg["train_dataset"]["augmentation"] = False
cfg["train_dataset"]["normalization"] = True
cfg["train_dataset"]["state_mask_prob"] = 0.0

generated_config.parent.mkdir(parents=True, exist_ok=True)
with generated_config.open("w") as f:
    json.dump(cfg, f, indent=2)
    f.write("\\n")

print(json.dumps({
    "task_name": cfg["task_name"],
    "output_root": cfg["output_root"],
    "data_root_dir": cfg["train_dataset"]["data_root_dir"],
    "data_mix": cfg["train_dataset"]["data_mix"],
    "batch_size": cfg["batch_size"],
    "total_batch_size": cfg["total_batch_size"],
    "max_steps": cfg["trainer"]["max_steps"],
    "save_steps": cfg["save_steps"],
    "save_checkpoint": cfg["save_checkpoint"],
}, indent=2))
PY

echo "GPU memory before launch:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1, not launching training."
    echo "Command:"
    echo "CUDA_VISIBLE_DEVICES=${GPUS} WANDB_MODE=${WANDB_MODE} PYTHONPATH=. torchrun --nproc_per_node=${NPROC} --standalone scripts/train.py --config ${GENERATED_CONFIG} --data_mix gigahands_demo_only --batch_size ${PER_GPU_BATCH} --total_batch_size ${TOTAL_BATCH} --max_steps ${RUN_STEPS} --num_workers ${NUM_WORKERS}"
    exit 0
fi

CUDA_VISIBLE_DEVICES="${GPUS}" \
WANDB_MODE="${WANDB_MODE}" \
PYTHONPATH=. \
torchrun --nproc_per_node="${NPROC}" --standalone scripts/train.py \
    --config "${GENERATED_CONFIG}" \
    --data_mix gigahands_demo_only \
    --batch_size "${PER_GPU_BATCH}" \
    --total_batch_size "${TOTAL_BATCH}" \
    --max_steps "${RUN_STEPS}" \
    --num_workers "${NUM_WORKERS}"
