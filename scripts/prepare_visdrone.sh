#!/usr/bin/env bash
# VisDrone2019-DET -> YOLO 格式（工作区内可写；图片用软链，标签生成到工作区）
# 注意：用户的 COCO json 里 category_id 是 0-based，而 ultralytics convert_coco 内部按 (id-1) 取类，
#       所以先整体 +1 再转换，cls 才会落在 0..9。
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh

VD=/home/wang/DeepSeek/datasets/VisDrone2019
OUT="$YOLO_ROOT/datasets/visdrone"

echo "=== [1/4] 图片软链 + 列表 ==="
mkdir -p "$OUT/images"
ln -sfn "$VD/VisDrone2019-DET-train/images" "$OUT/images/train"
ln -sfn "$VD/VisDrone2019-DET-val/images" "$OUT/images/val"
ls "$OUT/images/train" | sed 's|^|./images/train/|' > "$OUT/train.txt"
ls "$OUT/images/val" | sed 's|^|./images/val/|' > "$OUT/val.txt"
echo "train 图片: $(wc -l < "$OUT/train.txt")  val 图片: $(wc -l < "$OUT/val.txt")"

echo "=== [2/4] category_id 0-based -> 1-based ==="
rm -rf "$YOLO_ROOT/.tmp_visdrone_ann" "$YOLO_ROOT/.tmp_visdrone_convert"
mkdir -p "$YOLO_ROOT/.tmp_visdrone_ann"
python - <<'PYEOF'
import json
VD = "/home/wang/DeepSeek/datasets/VisDrone2019/annotations"
OUT = "${YOLO_ROOT}/.tmp_visdrone_ann"
for split in ("train", "val"):
    d = json.load(open(f"{VD}/instances_{split}.json"))
    for c in d["categories"]:
        c["id"] += 1
    for a in d["annotations"]:  # 关键：标注里的 category_id 也必须一起 +1，否则 cls 会变成 -1..8
        a["category_id"] += 1
    # 图片名 -> 保持原样；只为让 convert_coco 的 (id-1) 得到 0..9
    json.dump(d, open(f"{OUT}/instances_{split}.json", "w"))
    print(f"  {split}: categories -> {[c['id'] for c in d['categories']]}")
PYEOF

echo "=== [3/4] convert_coco ==="
python - <<'PYEOF'
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir="${YOLO_ROOT}/.tmp_visdrone_ann",
    save_dir="${YOLO_ROOT}/.tmp_visdrone_convert",
    use_segments=False,
    use_keypoints=False,
    cls91to80=False,
)
PYEOF

mkdir -p "$OUT/labels"
rm -rf "$OUT/labels/train" "$OUT/labels/val"
mv "$YOLO_ROOT/.tmp_visdrone_convert/labels/train" "$OUT/labels/train"
mv "$YOLO_ROOT/.tmp_visdrone_convert/labels/val" "$OUT/labels/val"

echo "=== [4/4] data yaml ==="
cat > "$YOLO_ROOT/datasets/visdrone.yaml" <<'YAMLEOF'
# VisDrone2019-DET（10 类，YOLO 格式；图片软链到 /home/wang/DeepSeek/datasets/VisDrone2019）
path: ${YOLO_ROOT}/datasets/visdrone
train: train.txt
val: val.txt
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
YAMLEOF

python - <<'PYEOF'
import json
from pathlib import Path
OUT = Path("${YOLO_ROOT}/datasets/visdrone")
VD = "/home/wang/DeepSeek/datasets/VisDrone2019/annotations"
for split in ("train", "val"):
    d = json.load(open(f"{VD}/instances_{split}.json"))
    anns = len(d["annotations"])
    labs = list((OUT / "labels" / split).glob("*.txt"))
    lines = sum(1 for f in labs for _ in open(f))
    cls_hist = {}
    for f in labs:
        for ln in open(f):
            c = int(ln.split()[0]); cls_hist[c] = cls_hist.get(c, 0) + 1
    print(f"{split}: 图片 {len(d['images'])} | json 标注 {anns} | label 文件 {len(labs)} | label 行 {lines} | 类别分布 {dict(sorted(cls_hist.items()))}")
PYEOF
echo "=== DONE ==="
