#!/usr/bin/env bash
# 复现 YOLO26n 官方 COCO val AP（默认 one-to-many 头 + NMS，官方表：mAP50-95 = 40.9）
# 必须在放宽沙箱权限下运行（CUDA 需要写 /proc/self/task/<tid>/comm）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
mkdir -p "$YOLO_ROOT/workdir" "$YOLO_ROOT/runs"
cd "$YOLO_ROOT/workdir"

yolo val \
    model="$YOLO_ROOT/weights/yolo26n.pt" \
    data="$YOLO_ROOT/datasets/coco-val.yaml" \
    imgsz=640 \
    batch=16 \
    device=0 \
    plots=True \
    project="$YOLO_ROOT/runs" \
    name=yolo26n_coco_val \
    exist_ok=True
echo "=== VAL DONE ==="
