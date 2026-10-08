#!/usr/bin/env python
"""种子方差"提前预览"：用同轮次训练曲线估噪声底量级。

注意：这是**训练期逐轮曲线**的比较，不是最终 best-of-100，因此只用于预览量级；
正式的种子方差结论由 `probe_seed_variance.py`（对最终 best.pt 做 val）给出。

用法: python probe_seed_curve_diff.py
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

ROOT = Path("/home/wang/DeepSeek/YOLO")
RUNS = ROOT / "runs/visdrone"


def curve(p: Path):
    d = {}
    if not p.exists():
        return d
    for row in csv.DictReader(open(p)):
        try:
            d[int(row["epoch"])] = float(row["metrics/mAP50-95(B)"])
        except Exception:
            pass
    return d


def main():
    seeds = {0: "base_n_640", 1: "base_n_640_seed1", 2: "base_n_640_seed2"}
    curves = {s: curve(RUNS / n / "results.csv") for s, n in seeds.items()}
    for s, c in curves.items():
        print(f"seed{s} ({seeds[s]}): {len(c)} epoch" + (f", 最优 {max(c.values()):.4f} (ep{max(c, key=c.get)})" if c else ""))

    done = [s for s, c in curves.items() if c]
    if len(done) < 2:
        print("\n不足两个种子，无法比较")
        return
    ref = 0
    for s in done:
        if s == ref:
            continue
        common = sorted(set(curves[ref]) & set(curves[s]))
        if not common:
            continue
        d = [curves[s][e] - curves[ref][e] for e in common]
        print(f"\nseed{ref} vs seed{s}（可比 {len(common)} 轮）:")
        print(f"  逐轮差值 均值 {st.mean(d):+.4f}  标准差 {st.stdev(d):.4f}  极差 {max(d)-min(d):.4f}")
        print(f"  同轮次最优 seed{ref}={max(curves[ref][e] for e in common):.4f}  "
              f"seed{s}={max(curves[s][e] for e in common):.4f}")
        print(f"  ⇒ 噪声底量级预览（1σ）≈ ±{st.stdev(d):.4f}")
    print("\n⚠️ 预览口径：训练期逐轮曲线，非最终 best-of-100；"
          "正式结论见 probe_seed_variance.py 的输出。")


if __name__ == "__main__":
    main()
