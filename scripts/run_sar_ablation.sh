#!/usr/bin/env bash
# 创新点 B（SAR 尺度自适应回归）消融链：3 个强度，各 100 epoch，其余设置与基线完全一致
# 用法: bash run_sar_ablation.sh            （后台跑，日志 train_b_sar_w*.log）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
ROOT=${YOLO_ROOT}
export YOLO_SRC="$ROOT/wt-exp"          # 使用 feat/sar-loss 分支代码
export MODEL="$ROOT/weights/yolo26n.pt"
export EPOCHS=100
export IMGSZ=640
export BATCH=16
export SEED=0

echo "=== SAR 消融开始 $(date +%F' '%T) ==="
echo "代码目录: $YOLO_SRC"
for W in 1.0 0.5 2.0; do
    NAME="b_sar_w${W/./}"
    echo "=== [$(date +%T)] 开始 $NAME (sar_w=$W, sar_s0=16.0, sar_cap=4.0) ==="
    bash "$ROOT/run_train_exp.sh" "$NAME" sar_w="$W" sar_s0=16.0 sar_cap=4.0 \
        > "$ROOT/train_${NAME}.log" 2>&1
    rc=$?
    echo "=== [$(date +%T)] $NAME 结束 rc=$rc ==="
done

echo "=== 汇总 ==="
python - "$ROOT" <<'PYEOF'
import sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for name, w in (("base_n_640", 0.0), ("b_sar_w10", 1.0), ("b_sar_w05", 0.5), ("b_sar_w20", 2.0)):
    p = root / "runs/visdrone" / name / "results.csv"
    if not p.exists():
        rows.append((name, w, "-", "-", "-"))
        continue
    best = max(([float(x) for x in l.split(",")[7:9]] for l in p.read_text().splitlines()[1:]), default=None)
    rows.append((name, w, f"{best[0]:.4f}" if best else "-", f"{best[1]:.4f}" if best else "-", "-"))
print(f"{'实验':16s} {'sar_w':>6s} {'best mAP50':>11s} {'best mAP50-95':>14s}")
for n, w, a, b, _ in rows:
    print(f"{n:16s} {w:6.1f} {a:>11s} {b:>14s}")
PYEOF
echo "=== SAR_ABLATION_DONE $(date +%F' '%T) ==="
