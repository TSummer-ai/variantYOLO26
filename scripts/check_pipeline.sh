#!/usr/bin/env bash
# 训练与流水线状态精确检查（避免 grep 匹配到等待脚本自身的命令行）
#
# 背景：等待脚本的命令行里含有 "probe_seed_variance.py"、"finalize_ablations_doc.py" 等字样，
#       用 `ps | grep -c "python probe_..."` 会把等待脚本自己算进去，造成"提前启动分析"的误判。
#       本脚本改为按 **解释器 + 脚本作为 argv[1]** 精确匹配。
set -uo pipefail
cd /home/wang/DeepSeek/YOLO

TRAIN_PID=$(pgrep -f "^bash run_seed_variance.sh" | head -1 || true)
echo "=== 训练 ==="
if [ -n "${TRAIN_PID}" ]; then
    echo "  运行中 (PID ${TRAIN_PID})"
else
    echo "  已结束（无 run_seed_variance.sh 进程）"
fi

for d in base_n_640 base_n_640_seed1 base_n_640_seed2; do
    f="runs/visdrone/$d/results.csv"
    if [ -f "$f" ]; then
        ep=$(( $(wc -l < "$f") - 1 ))
        t=$(tail -1 "$f" | cut -d, -f2)
        best=$(cut -d, -f9 "$f" | tail -n +2 | sort -g | tail -1)
        sp=$(python3 -c "print(f'{$t/max($ep,1):.1f}')")
        printf "  %-20s %3d/100 ep  %6.1fs/ep  best mAP50-95=%.4f" "$d" "$ep" "$sp" "$best"
        [ "$ep" -ge 100 ] && echo "  [完成]" || echo ""
    else
        printf "  %-20s 未开始\n" "$d"
    fi
done

echo "=== 流水线 ==="
W=$(pgrep -f "while kill -0" | head -1 || true)
[ -n "${W}" ] && echo "  等待脚本运行中 (PID ${W})" || echo "  无等待脚本"
# 精确：解释器可执行名 + 脚本名作为第一个参数
RUNNING=$(ps -eo pid,comm,args --no-headers | awk '$2 ~ /^python/ {
    n=split($0,a," ");
    for(i=1;i<=n;i++) if (a[i] ~ /\.py$/) { print a[i]; break }
}' | grep -E "probe_seed_variance|probe_resolution_by_size|bench_batched_slicing|probe_wbf_nmodels|finalize_ablations_doc" | sort -u)
if [ -n "${RUNNING}" ]; then
    echo "  真正在跑的分析脚本:"; echo "${RUNNING}" | sed 's/^/    /'
else
    echo "  无分析脚本在跑（符合预期）"
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "  查询失败"
