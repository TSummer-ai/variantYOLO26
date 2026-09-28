#!/usr/bin/env bash
# 用法： source activate_yolo.sh
# 作用： 激活 venv，并把 ultralytics / torch / matplotlib 缓存目录指到仓库内
#        （共享机器/沙箱上默认的 ~/.config、~/.cache 常不可写）
#
# 说明： 本脚本会把 YOLO_ROOT 强制设为「本脚本所在目录」——所有 scripts/ 里的脚本
#        都以它作为路径根。若你此前已 export 过别的 YOLO_ROOT，会收到一条警告。

_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${YOLO_ROOT:-}" ] && [ "$YOLO_ROOT" != "$_self" ]; then
    echo "[activate_yolo.sh] 警告：YOLO_ROOT 原为 $YOLO_ROOT，现改为本仓库目录 $_self" >&2
fi
YOLO_ROOT="$_self"
export YOLO_ROOT

export YOLO_CONFIG_DIR="$YOLO_ROOT/.config"          # ultralytics 会再拼一层 Ultralytics/
export TORCH_HOME="$YOLO_ROOT/.cache/torch"
export MPLCONFIGDIR="$YOLO_ROOT/.cache/matplotlib"
export PIP_CACHE_DIR="$YOLO_ROOT/.pip-cache"
export PATH="$YOLO_ROOT/.venv/bin:$PATH"
# 想用仓库里的 ultralytics 源码（而非 pip 装的版本）时设置这个变量
if [ -n "${YOLO_ULTRALYTICS_SRC:-}" ]; then
    export PYTHONPATH="$YOLO_ULTRALYTICS_SRC${PYTHONPATH:+:$PYTHONPATH}"
fi
mkdir -p "$YOLO_CONFIG_DIR/Ultralytics" "$TORCH_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"

echo "[activate_yolo.sh] YOLO_ROOT=$YOLO_ROOT"
