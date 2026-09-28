#!/usr/bin/env bash
# 创新点 B 第二组：把尺度权重加到 CIoU 项（sar_iou_w），两个配置各 100 epoch
#   1) CIoU 单独加权：sar_w=0, sar_iou_w=1
#   2) 两项都加权：  sar_w=1, sar_iou_w=1
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
ROOT=${YOLO_ROOT}
export YOLO_SRC="$ROOT/wt-exp"
export MODEL="$ROOT/weights/yolo26n.pt"
export EPOCHS=100
export IMGSZ=640
export BATCH=16
export SEED=0

echo "=== SAR2 消融开始 $(date +%F' '%T)（代码 $YOLO_SRC）==="
run_one() {
    local NAME=$1 SW=$2 SIW=$3
    echo "=== [$(date +%T)] 开始 $NAME (sar_w=$SW, sar_iou_w=$SIW) ==="
    bash "$ROOT/run_train_exp.sh" "$NAME" sar_w="$SW" sar_iou_w="$SIW" sar_s0=16.0 sar_cap=4.0 \
        > "$ROOT/train_${NAME}.log" 2>&1
    echo "=== [$(date +%T)] $NAME 结束 rc=$? ==="
}
run_one b_sar_iouw10 0.0 1.0
run_one b_sar_w10iouw10 1.0 1.0
echo "=== SAR2_ABLATION_DONE $(date +%F' '%T) ==="
