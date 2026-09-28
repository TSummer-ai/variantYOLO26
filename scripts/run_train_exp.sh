#!/usr/bin/env bash
# 通用消融训练脚本（VisDrone）
# 用法: bash run_train_exp.sh <实验名> [额外的 yolo 参数 ...]
#   env 可覆盖: MODEL / EPOCHS / IMGSZ / BATCH / WORKERS / YOLO_SRC(代码目录, 默认主 checkout)
# 例:  MODEL=weights/yolo26n.pt EPOCHS=100 bash run_train_exp.sh b_sar --sar_w 1.0
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
NAME="${1:?需要实验名}"; shift

MODEL="${MODEL:-$YOLO_ROOT/weights/yolo26n.pt}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
YOLO_SRC="${YOLO_SRC:-$YOLO_ROOT/ultralytics}"

# 允许指向别的代码目录（feature worktree），确保 import 命中该目录
export PYTHONPATH="$YOLO_SRC${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

echo "=== exp=$NAME model=$MODEL epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH src=$YOLO_SRC ==="
yolo detect train \
    model="$MODEL" \
    data="$YOLO_ROOT/datasets/visdrone.yaml" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device=0 \
    workers="$WORKERS" \
    cache=False \
    seed="${SEED:-0}" \
    project="$YOLO_ROOT/runs/visdrone" \
    name="$NAME" \
    exist_ok=True \
    plots=True \
    val=True \
    "$@"
echo "=== TRAIN DONE ($NAME) ==="
