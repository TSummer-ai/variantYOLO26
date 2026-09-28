#!/usr/bin/env bash
# 用 Ultralytics 自带 convert_coco 从官方 instances_val2017.json 生成 YOLO 标签
# （等价于官方 coco2017labels-segments.zip 里的 labels/val2017）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
source ${YOLO_ROOT}/activate_yolo.sh
DS="$YOLO_ROOT/datasets/coco"

# 只保留 val 的 instances 标注，避免 convert_coco 误转 keypoints/captions/train
rm -rf "$YOLO_ROOT/.tmp_coco_ann_val" "$YOLO_ROOT/.tmp_coco_convert"
mkdir -p "$YOLO_ROOT/.tmp_coco_ann_val"
cp -f "$DS/annotations/instances_val2017.json" "$YOLO_ROOT/.tmp_coco_ann_val/"

python - <<'PYEOF'
from ultralytics.data.converter import convert_coco
convert_coco(
    labels_dir="${YOLO_ROOT}/.tmp_coco_ann_val",
    save_dir="${YOLO_ROOT}/.tmp_coco_convert",
    use_segments=True,
    use_keypoints=False,
    cls91to80=True,
)
PYEOF

mkdir -p "$DS/labels"
rm -rf "$DS/labels/val2017"
mv "$YOLO_ROOT/.tmp_coco_convert/labels/val2017" "$DS/labels/val2017"

# 官方 val2017.txt 形式：相对 dataset root 的 ./images/val2017/xxx.jpg
ls "$DS/images/val2017" | sed 's|^|./images/val2017/|' > "$DS/val2017.txt"

# 生成 val 专用 data yaml（names 与官方 coco.yaml 完全一致）
python - <<'PYEOF'
import yaml
src = yaml.safe_load(open("${YOLO_ROOT}/ultralytics/ultralytics/cfg/datasets/coco.yaml"))
out = {
    "path": "${YOLO_ROOT}/datasets/coco",
    "train": "val2017.txt",
    "val": "val2017.txt",
    "names": src["names"],
}
with open("${YOLO_ROOT}/datasets/coco-val.yaml", "w") as f:
    f.write("# COCO val2017 (仅 val split, 用于复现官方 AP)\n")
    yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, default_flow_style=None)
print("wrote coco-val.yaml with", len(out["names"]), "classes")
PYEOF

echo "label files      : $(ls "$DS/labels/val2017" | wc -l)"
echo "image files      : $(wc -l < "$DS/val2017.txt")"
echo "label box lines  : $(cat "$DS/labels/val2017"/*.txt | wc -l)"
echo "=== DONE ==="
