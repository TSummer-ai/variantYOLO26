#!/usr/bin/env bash
# DHCD 消融链结束后：两套 val + 分尺寸召回分析 + 汇总表（重点看 e2e 头）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -uo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
ROOT="$YOLO_ROOT"
cd "$ROOT/workdir"

echo "=== 等待 DHCD 消融链结束 ==="
while ! grep -q "DHCD_ABLATION_DONE" "$ROOT/dhcd_ablation.log" 2>/dev/null; do sleep 30; done
echo "消融链已结束: $(date +%T)"

for NAME in a_dhcd_w10 a_dhcd_w20; do
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
    echo "=== [$NAME] 分尺寸召回分析（NMS + e2e）==="
    python "$ROOT/analyze_visdrone.py" --weights "$W" --imgsz 640 --device 0 --conf 0.25 \
        --out "$ROOT/runs/visdrone/analysis_${NAME}" > "$ROOT/analysis_${NAME}.log" 2>&1
done

echo "=== 汇总 ==="
python - "$ROOT" <<'PYEOF' | tee "$ROOT/runs/visdrone/DHCD_COMPARISON.md"
import csv, re, sys
from pathlib import Path
root = Path(sys.argv[1])
runs = [
    ("base_n_640", "val_visdrone", "baseline（无蒸馏）"),
    ("a_dhcd_w10", "val_a_dhcd_w10", "DHCD dhcd_w=1.0"),
    ("a_dhcd_w20", "val_a_dhcd_w20", "DHCD dhcd_w=2.0"),
]
def val_line(p):
    if not p.exists():
        return None
    txt = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text()).replace("\r", "\n")
    m = [l for l in txt.splitlines() if re.match(r"^\s+all\s+\d+", l)]
    return [float(x) for x in m[-1].split()[3:7]] if m else None
print("# DHCD（创新点 A）消融结果（VisDrone val, 100 epoch @640）\n")
print("| 实验 | 训练 best mAP50-95 | NMS mAP50 | NMS mAP50-95 | e2e mAP50 | e2e mAP50-95 | e2e−NMS(50-95) |")
print("|---|---|---|---|---|---|---|")
for name, prefix, label in runs:
    csv = root / "runs/visdrone" / name / "results.csv"
    best = 0.0
    if csv.exists():
        rows = list(csv.DictReader(open(csv)))
        for r in rows:  # 按列名取值——带 dhcd_loss 的 run 列会右移，按索引取会读错成 mAP50
            best = max(best, float(r["metrics/mAP50-95(B)"]))
    nms, e2e = val_line(root / f"{prefix}_nms.log"), val_line(root / f"{prefix}_e2e.log")
    f = lambda v, i: f"{v[i]:.4f}" if v else "—"
    gap = f"{e2e[3] - nms[3]:+.4f}" if (nms and e2e) else "—"
    print(f"| {label} | {best:.4f} | {f(nms,2)} | {f(nms,3)} | {f(e2e,2)} | {f(e2e,3)} | {gap} |")
print("\n> 目标：DHCD 应缩小 `e2e−NMS` 的差距（基线为 −0.0050），并抬升 e2e 头在小目标上的召回。\n")
for name, _, label in runs[1:]:
    p = root / "runs/visdrone" / f"analysis_{name}" / "summary.md"
    if p.exists():
        print(f"## {label} 分尺寸召回\n")
        body = p.read_text().split("## 1.")[-1].split("## 2.")[0]
        print("```\n" + body.strip() + "\n```\n")
PYEOF

echo "=== DHCD_POST_DONE $(date +%T) ==="
