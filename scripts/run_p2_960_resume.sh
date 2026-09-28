#!/usr/bin/env bash
# 续训 P2@960（从 last.pt，epoch ~30 继续），带自动重试。
# 崩溃原因：cuDNN CUDNN_STATUS_EXECUTION_FAILED，显存 89% 占用，大概率是碎片/压力。
# 对策：① PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ② 失败自动续训重试
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

CKPT="$YOLO_ROOT/runs/visdrone/p2_960/weights/last.pt"
echo "=== P2@960 续训开始 $(date +%F' '%T)（alloc: expandable_segments）==="

for attempt in $(seq 1 6); do
    echo "--- 第 $attempt 次尝试 $(date +%T) ---"
    yolo detect train model="$CKPT" resume=True
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "=== P2@960 正常结束 $(date +%F' '%T) ==="
        break
    fi
    echo "--- 第 $attempt 次异常退出 rc=$rc，30 秒后续训重试 ---"
    sleep 30
done
