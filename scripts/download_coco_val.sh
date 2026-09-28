#!/usr/bin/env bash
# 从 ModelScope 镜像下载 COCO val2017 官方数据（815MB 图片 + 252MB 标注）
# --- 路径自举：可用 YOLO_ROOT 覆盖，默认取仓库根目录 ---
YOLO_ROOT="${YOLO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export YOLO_ROOT

set -euo pipefail
ROOT=${YOLO_ROOT}
DS="$ROOT/datasets/coco"
mkdir -p "$DS"
BASE="https://modelscope.cn/api/v1/datasets/codexie/coco_val2017/repo?Revision=master&FilePath="

echo "=== [1/3] val2017.zip (815MB) ==="
curl -L --retry 5 --retry-delay 3 -C - -o "$DS/val2017.zip" "${BASE}val2017.zip"
echo "=== [2/3] annotations_trainval2017.zip (252MB) ==="
curl -L --retry 5 --retry-delay 3 -C - -o "$DS/annotations_trainval2017.zip" "${BASE}annotations_trainval2017.zip"
echo "=== [3/3] unzip ==="
mkdir -p "$DS/images"
unzip -q -o "$DS/val2017.zip" -d "$DS/images"
unzip -q -o "$DS/annotations_trainval2017.zip" -d "$DS"
echo "=== sizes ==="
echo "val images: $(ls "$DS/images/val2017" | wc -l)"
echo "jsons: $(ls "$DS/annotations"/*.json | wc -l)"
du -sh "$DS"
echo "=== DONE ==="
