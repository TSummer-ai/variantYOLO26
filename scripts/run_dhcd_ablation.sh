#!/usr/bin/env bash
# 创新点 A（DHCD 双头一致性蒸馏）消融链：两个权重，各 100 epoch，其余与基线一致
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
ROOT=${YOLO_ROOT}
export YOLO_SRC="$ROOT/wt-dhcd"
export MODEL="$ROOT/weights/yolo26n.pt"
export EPOCHS=100
export IMGSZ=640
export BATCH=16
export SEED=0

echo "=== DHCD 消融开始 $(date +%F' '%T)（代码 $YOLO_SRC）==="
for W in 1.0 2.0; do
    NAME="a_dhcd_w${W/./}"
    echo "=== [$(date +%T)] 开始 $NAME (dhcd_w=$W) ==="
    bash "$ROOT/run_train_exp.sh" "$NAME" dhcd_w="$W" > "$ROOT/train_${NAME}.log" 2>&1
    echo "=== [$(date +%T)] $NAME 结束 rc=$? ==="
done
echo "=== DHCD_ABLATION_DONE $(date +%F' '%T) ==="
