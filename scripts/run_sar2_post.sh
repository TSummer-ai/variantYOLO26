#!/usr/bin/env bash
# SAR2 消融链结束后：每个变体跑两套 val，汇总 SAR2 对比表
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
ROOT="$YOLO_ROOT"
cd "$ROOT/workdir"

echo "=== 等待 SAR2 消融链结束 ==="
while ! grep -q "SAR2_ABLATION_DONE" "$ROOT/sar2_ablation.log" 2>/dev/null; do sleep 30; done
echo "消融链已结束: $(date +%T)"

for NAME in b_sar_iouw10 b_sar_w10iouw10; do
    W="$ROOT/runs/visdrone/$NAME/weights/best.pt"
    [ -f "$W" ] || { echo "跳过 $NAME（无 best.pt）"; continue; }
    echo "=== [$NAME] val 默认头 ==="
    yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=640 batch=16 device=0 \
        split=val plots=True project="$ROOT/runs/visdrone" name="val_${NAME}_nms" exist_ok=True \
        > "$ROOT/val_${NAME}_nms.log" 2>&1
    echo "=== [$NAME] val e2e (nms=False) ==="
    yolo val model="$W" data="$ROOT/datasets/visdrone.yaml" imgsz=640 batch=16 device=0 nms=False \
        split=val plots=True project="$ROOT/runs/visdrone" name="val_${NAME}_e2e" exist_ok=True \
        > "$ROOT/val_${NAME}_e2e.log" 2>&1
done

echo "=== 汇总 ==="
python - "$ROOT" <<'PYEOF' | tee "$ROOT/runs/visdrone/SAR2_COMPARISON.md"
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
runs = [
    ("base_n_640",        "val_visdrone",            "baseline (sar_w=0)"),
    ("b_sar_w05",         "val_b_sar_w05",           "SAR L1 w=0.5"),
    ("b_sar_w10",         "val_b_sar_w10",           "SAR L1 w=1.0"),
    ("b_sar_w20",         "val_b_sar_w20",           "SAR L1 w=2.0"),
    ("b_sar_iouw10",      "val_b_sar_iouw10",        "SAR CIoU w=1.0 (L1 off)"),
    ("b_sar_w10iouw10",   "val_b_sar_w10iouw10",     "SAR L1+CIoU w=1.0"),
]
def val_line(p):
    if not p.exists():
        return None
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text()).replace("\r", "\n")
    m = [l for l in txt.splitlines() if re.match(r"^\s+all\s+\d+", l)]
    return [float(x) for x in m[-1].split()[3:7]] if m else None
print("# SAR 消融结果 v2（VisDrone val, 100 epoch @640）\n")
print("| 实验 | 训练 best mAP50-95 | NMS mAP50 | NMS mAP50-95 | e2e mAP50 | e2e mAP50-95 |")
print("|---|---|---|---|---|---|")
for name, prefix, label in runs:
    csv = root / "runs/visdrone" / name / "results.csv"
    best = 0.0
    if csv.exists():
        for l in csv.read_text().splitlines()[1:]:
            best = max(best, float(l.split(",")[8]))
    nms = val_line(root / f"{prefix}_nms.log")
    e2e = val_line(root / f"{prefix}_e2e.log")
    f = lambda v, i: f"{v[i]:.4f}" if v else "—"
    print(f"| {label} | {best:.4f} | {f(nms,2)} | {f(nms,3)} | {f(e2e,2)} | {f(e2e,3)} |")
PYEOF

echo "=== SAR2_POST_DONE $(date +%T) ==="
