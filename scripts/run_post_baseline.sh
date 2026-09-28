#!/usr/bin/env bash
# 基线训练结束后的自动后处理：
#   1) 等训练进程退出
#   2) val（默认 one2many + NMS）
#   3) val（nms=False, one2one / e2e）
#   4) 错误分析（分尺寸召回 / 分类别召回 / 混淆 / 两头差距）
# 必须在放宽沙箱权限下运行（CUDA）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
cd "$YOLO_ROOT/workdir"

W="$YOLO_ROOT/runs/visdrone/base_n_640/weights/best.pt"
LOG="$YOLO_ROOT"

echo "=== [1/4] 等待基线训练结束 ==="
while pgrep -f "[y]olo detect train" > /dev/null; do sleep 30; done
echo "训练进程已退出: $(date +%T)"
grep -c "TRAIN DONE" "$YOLO_ROOT/train_base_n_640.log" || true
ls -la "$W"

echo "=== [2/4] val: 默认头 + NMS ==="
yolo val model="$W" data="$YOLO_ROOT/datasets/visdrone.yaml" imgsz=640 batch=16 device=0 \
    split=val plots=True project="$YOLO_ROOT/runs/visdrone" name=val_nms exist_ok=True \
    > "$LOG/val_visdrone_nms.log" 2>&1
echo "  -> $LOG/val_visdrone_nms.log"

echo "=== [3/4] val: nms=False (e2e / one2one) ==="
yolo val model="$W" data="$YOLO_ROOT/datasets/visdrone.yaml" imgsz=640 batch=16 device=0 \
    nms=False split=val plots=True project="$YOLO_ROOT/runs/visdrone" name=val_e2e exist_ok=True \
    > "$LOG/val_visdrone_e2e.log" 2>&1
echo "  -> $LOG/val_visdrone_e2e.log"

echo "=== [4/4] 错误分析 ==="
python "$YOLO_ROOT/analyze_visdrone.py" --weights "$W" --imgsz 640 --device 0 \
    > "$LOG/analysis_visdrone.log" 2>&1
echo "  -> $LOG/analysis_visdrone.log"

echo "=== POST DONE ==="
