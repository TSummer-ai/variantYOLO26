#!/usr/bin/env bash
# VisDrone2019-DET 基线：COCO 预训练 yolo26n 微调（100 epoch, 640, batch 16）
# 必须在放宽沙箱权限下运行（CUDA 需要写 /proc/self/task/<tid>/comm）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

yolo detect train \
    model="$YOLO_ROOT/weights/yolo26n.pt" \
    data="$YOLO_ROOT/datasets/visdrone.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=8 \
    cache=False \
    seed=0 \
    project="$YOLO_ROOT/runs/visdrone" \
    name=base_n_640 \
    exist_ok=True \
    plots=True \
    val=True
echo "=== TRAIN DONE ==="
