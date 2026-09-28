#!/usr/bin/env bash
# P2 头实验：yolo26n-p2（4 个检测层 P2/P3/P4/P5），COCO 预训练权重部分加载
# 用 epochs=100 而不是 30 —— 这样 LR/mosaic 调度与基线完全一致，
# 在第 30 个 epoch 做"同 epoch 对照"，趋势好就让它继续跑完，不好就杀掉。省一次重启。
#
# 预先登记的判定线（基线同 epoch 值，来自 runs/visdrone/base_n_640/results.csv）：
#   baseline epoch 30: mAP50=0.30018  mAP50-95=0.16169
#   P2  epoch 30 >= 0.167  -> 继续跑满        (约 +0.5 AP，值得继续)
#   P2  epoch 30 <  0.165  -> 立即停，回退     (无明显收益)
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

echo "=== P2 实验开始 $(date +%F' '%T) ==="
echo "判定线: epoch30 >= 0.167 继续 / < 0.165 停（基线 epoch30 = 0.16169）"
yolo detect train \
    model="$YOLO_ROOT/ultralytics/ultralytics/cfg/models/26/yolo26-p2.yaml" \
    pretrained="$YOLO_ROOT/weights/yolo26n.pt" \
    data="$YOLO_ROOT/datasets/visdrone.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=0 \
    workers=8 \
    cache=False \
    seed=0 \
    project="$YOLO_ROOT/runs/visdrone" \
    name=p2_n_640 \
    exist_ok=True \
    plots=True \
    val=True
echo "=== P2 实验结束 $(date +%F' '%T) ==="
