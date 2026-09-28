#!/usr/bin/env bash
# P2@960 收尾：两套 val（960）+ oracle 机制复核 + 分尺寸召回 + 三方对比表
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT="$YOLO_ROOT"
cd "$ROOT/workdir"
W="$ROOT/runs/visdrone/p2_960/weights/best.pt"

echo "=== [1/5] val 默认头 @960 ==="
yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=960 batch=4 device=0 \
    split=val plots=True project="$ROOT/runs/visdrone" name=val_p2_960_nms exist_ok=True \
    > "$ROOT/val_p2_960_nms.log" 2>&1

echo "=== [2/5] val e2e @960 ==="
yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=960 batch=4 device=0 nms=False \
    split=val plots=True project="$ROOT/runs/visdrone" name=val_p2_960_e2e exist_ok=True \
    > "$ROOT/val_p2_960_e2e.log" 2>&1

echo "=== [3/5] max_det=1000 复测 ==="
yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=960 batch=4 device=0 max_det=1000 \
    split=val plots=False project="$ROOT/runs/visdrone" name=val_p2_960_md1000 exist_ok=True \
    > "$ROOT/val_p2_960_md1000.log" 2>&1

echo "=== [4/5] oracle 机制复核 @960 ==="
for H in o2m e2e; do
  python "$ROOT/oracle_decompose.py" --weights "$W" --head $H --device 0 --imgsz 960 \
      --out "$ROOT/runs/visdrone/oracle_p2_960" >> "$ROOT/oracle_p2_960.log" 2>&1
done

echo "=== [5/5] 三方对比表 ==="
python - "$ROOT" <<'PYEOF' | tee "$ROOT/runs/visdrone/FINAL_COMPARISON.md"
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def best(name):
    p = root / "runs/visdrone" / name / "results.csv"
    if not p.exists(): return 0.0, 0
    rows = list(csv.DictReader(open(p)))
    r = max(rows, key=lambda x: float(x["metrics/mAP50-95(B)"]))
    return float(r["metrics/mAP50-95(B)"]), int(r["epoch"])
def val(prefix):
    p = root / prefix
    if not p.exists(): return None
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text()).replace("\r", "\n")
    m = [l for l in txt.splitlines() if re.match(r"^\s+all\s+\d+", l)]
    return [float(x) for x in m[-1].split()[3:7]] if m else None
runs = [("base_n_640", 640, "val_visdrone", "baseline @640"),
        ("p2_n_640", 640, "val_p2", "P2 @640"),
        ("p2_960", 960, "val_p2_960", "P2 @960")]
print("# 最终对比（VisDrone val, yolo val 内部指标）\n")
print("| 模型 | imgsz | 训练 best mAP50-95 | NMS mAP50 | NMS mAP50-95 | e2e mAP50 | e2e mAP50-95 |")
print("|---|---|---|---|---|---|---|")
for name, imgsz, prefix, label in runs:
    b, e = best(name)
    nms, e2e = val(root / f"{prefix}_nms.log"), val(root / f"{prefix}_e2e.log")
    f = lambda v, i: f"{v[i]:.4f}" if v else "—"
    print(f"| {label} | {imgsz} | {b:.4f} (ep{e}) | {f(nms,2)} | {f(nms,3)} | {f(e2e,2)} | {f(e2e,3)} |")
print("\n注：@960 的推理算力是 @640 的 2.25 倍（17.3 vs 7.7 GFLOPs），属不同算力点。")
PYEOF
echo "=== P2_960_POST_DONE $(date +%T) ==="
