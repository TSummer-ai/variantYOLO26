#!/usr/bin/env bash
# 全量 oracle 分解（两个头），结果写到 runs/visdrone/oracle/
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
W="$YOLO_ROOT/runs/visdrone/base_n_640/weights/best.pt"
for H in o2m e2e; do
  echo "########## head=$H $(date +%T) ##########"
  python "$YOLO_ROOT/oracle_decompose.py" --weights "$W" --head $H --device 0 \
      --out "$YOLO_ROOT/runs/visdrone/oracle"
done
echo "=== ORACLE_DONE $(date +%T) ==="
