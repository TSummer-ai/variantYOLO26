#!/usr/bin/env bash
# P2 训练结束后自动做：两套 val + oracle 复核 + 分尺寸召回 + 对比表
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
ROOT="$YOLO_ROOT"
cd "$ROOT/workdir"

echo "=== 等待 P2 训练结束 ==="
while ! grep -q "=== P2 实验结束" "$ROOT/p2_n_640.log" 2>/dev/null; do sleep 30; done
echo "训练已结束: $(date +%T)"
W="$ROOT/runs/visdrone/p2_n_640/weights/best.pt"
ls -la "$W"

echo "=== [1/5] val 默认头 ==="
yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=640 batch=8 device=0 \
    split=val plots=True project="$ROOT/runs/visdrone" name=val_p2_nms exist_ok=True \
    > "$ROOT/val_p2_nms.log" 2>&1

echo "=== [2/5] val e2e (nms=False) ==="
yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=640 batch=8 device=0 nms=False \
    split=val plots=True project="$ROOT/runs/visdrone" name=val_p2_e2e exist_ok=True \
    > "$ROOT/val_p2_e2e.log" 2>&1

echo "=== [3/5] oracle 复核（两个头）==="
for H in o2m e2e; do
  python "$ROOT/oracle_decompose.py" --weights "$W" --head $H --device 0 \
      --out "$ROOT/runs/visdrone/oracle_p2" >> "$ROOT/oracle_p2.log" 2>&1
done

echo "=== [4/5] 分尺寸召回 ==="
python "$ROOT/analyze_visdrone.py" --weights "$W" --imgsz 640 --device 0 --conf 0.25 \
    --out "$ROOT/runs/visdrone/analysis_p2" > "$ROOT/analysis_p2.log" 2>&1

echo "=== [5/5] 对比表 ==="
python - "$ROOT" <<'PYEOF' | tee "$ROOT/runs/visdrone/P2_COMPARISON.md"
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def best(name):
    p = root / "runs/visdrone" / name / "results.csv"
    if not p.exists(): return 0.0, 0, 0.0
    rows = list(csv.DictReader(open(p)))
    r = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
    return float(r["metrics/mAP50-95(B)"]), int(r["epoch"]), float(r["metrics/mAP50(B)"])
def val(prefix):
    p = root / prefix
    if not p.exists(): return None
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text()).replace("\r", "\n")
    m = [l for l in txt.splitlines() if re.match(r"^\s+all\s+\d+", l)]
    return [float(x) for x in m[-1].split()[3:7]] if m else None
runs = [("base_n_640", "val_visdrone", "baseline（P3/P4/P5）"),
        ("p2_n_640",   "val_p2",       "P2（P2/P3/P4/P5）")]
print("# P2 头 vs 基线（VisDrone val, 100 epoch @640）\n")
print("| 实验 | 训练 best mAP50-95 (@epoch) | NMS mAP50 | NMS mAP50-95 | e2e mAP50 | e2e mAP50-95 |")
print("|---|---|---|---|---|---|")
for name, prefix, label in runs:
    b, e, b50 = best(name)
    nms, e2e = val(root / f"{prefix}_nms.log"), val(root / f"{prefix}_e2e.log")
    f = lambda v, i: f"{v[i]:.4f}" if v else "—"
    print(f"| {label} | {b:.4f} (ep{e}) | {f(nms,2)} | {f(nms,3)} | {f(e2e,2)} | {f(e2e,3)} |")
PYEOF

echo "=== P2_POST_DONE $(date +%T) ==="
