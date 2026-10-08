#!/usr/bin/env bash
# 任务1：baseline@640 补种子，用于测"种子方差噪声底"（原主结果均为 seed=0 单种子）。
# 与 runs/visdrone/base_n_640/args.yaml 完全一致，仅改 seed 与输出目录。
#
#   bash run_seed_variance.sh smoke   # 2 epoch 冒烟（约 3 分钟）
#   bash run_seed_variance.sh 1 2     # 依次跑 seed=1、seed=2（各 100 epoch，约 1.8 h）
set -uo pipefail
source /home/wang/DeepSeek/YOLO/activate_yolo.sh
mkdir -p "$YOLO_ROOT/workdir"
cd "$YOLO_ROOT/workdir"

MODE="${1:-smoke}"

run_one () {
    local seed="$1" epochs="$2" name="$3"
    echo "=== [$(date +%H:%M:%S)] TRAIN seed=${seed} epochs=${epochs} -> ${name} ==="
    yolo detect train \
        model="$YOLO_ROOT/weights/yolo26n.pt" \
        data="$YOLO_ROOT/datasets/visdrone.yaml" \
        epochs="${epochs}" \
        imgsz=640 \
        batch=16 \
        device=0 \
        workers=8 \
        cache=False \
        seed="${seed}" \
        deterministic=True \
        project="$YOLO_ROOT/runs/visdrone" \
        name="${name}" \
        exist_ok=True \
        plots=False \
        val=True
    echo "=== [$(date +%H:%M:%S)] DONE ${name} (exit=$?) ==="
}

if [ "$MODE" = "smoke" ]; then
    run_one 1 2 smoke_seed1
else
    for seed in "$@"; do
        run_one "${seed}" 100 "base_n_640_seed${seed}"
    done
fi
echo "=== ALL DONE $(date +%H:%M:%S) ==="
