#!/usr/bin/env bash
# P2 + 960 输入：表示侧继续加深
# 判定线（预先登记）：相对 P2@640（md300: 0.1980 / md1000: 0.2030）
#   最终 mAP50-95 >= 0.203  -> 值得写进最终模型
#   < 0.198                -> 说明 960 在 P2 之上没有额外收益
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

echo "=== P2@960 开始 $(date +%F' '%T) ==="
yolo detect train \
    model="$YOLO_ROOT/ultralytics/ultralytics/cfg/models/26/yolo26-p2.yaml" \
    pretrained="$YOLO_ROOT/weights/yolo26n.pt" \
    data="$YOLO_ROOT/datasets/visdrone.yaml" \
    epochs=100 \
    imgsz=960 \
    batch=4 \
    device=0 \
    workers=8 \
    cache=False \
    seed=0 \
    project="$YOLO_ROOT/runs/visdrone" \
    name=p2_960 \
    exist_ok=True \
    plots=True \
    val=True
echo "=== P2@960 结束 $(date +%F' '%T) ==="
