#!/usr/bin/env python
"""精度-速度折线图（论文/参赛用）。

数据全部实测：forward latency 来自 bench_models.py（RTX 4060 Laptop, FP32, batch 1, 100 次平均），
mAP 来自 yolo val（VisDrone val, 548 图）。所有配置都是 100 epoch 同设定训练。

产出 figures/fig_tradeoff.png（双面板：vs 延迟 / vs GFLOPs）
"""

from __future__ import annotations

import os

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)  # 仓库里 figures/ 默认不存在
_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_CJK).exists():
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False

# 名称: (GFLOPs, 前向ms, mAP50-95@md300, mAP50-95@md1000, 是否帕累托点, 颜色, 说明)
PTS = {
    "baseline @640\n(640 训/640 推)":      (5.9,  4.57, 0.1820, 0.1860, True,  "#7f7f7f", ""),
    "P2-pruned\n(P2 neck + 三层头)":        (6.6,  5.40, 0.1888, None,   True,  "#2ca02c", ""),
    "P2 @640":                              (7.7,  5.88, 0.1980, 0.2030, True,  "#1f77b4", ""),
    "baseline @960 推理\n(640 训/960 推)":  (13.6, 5.96, 0.2240, None,   True,  "#ff7f0e", "不重训也涨"),
    "P2 @960\n(960 训/960 推)":             (17.6, 8.96, 0.2630, 0.2690, True,  "#d62728", "最终配置"),
}


def panel(ax, xkey, xlabel, logx=False):
    pts = sorted(PTS.items(), key=lambda kv: kv[1][{("lat"): 1, ("gflops"): 0}[xkey]])
    xs = [p[1][{("lat"): 1, ("gflops"): 0}[xkey]] for p in pts]
    ys = [p[1][2] for p in pts]
    ax.plot(xs, ys, "-", color="#444444", alpha=0.55, linewidth=1.6, zorder=1)
    for name, (gf, lat, ap, ap_md, pareto, color, note) in pts:
        x = lat if xkey == "lat" else gf
        ax.scatter([x], [ap], s=150, color=color, zorder=3, edgecolor="white", linewidth=1.2)
        if ap_md:  # max_det=1000 免费增益
            ax.scatter([x], [ap_md], s=70, facecolor="none", edgecolor=color, linewidth=1.6, zorder=3)
            ax.plot([x, x], [ap, ap_md], "-", color=color, alpha=0.5, linewidth=1)
        dy = -0.016 if "baseline @640" in name or "pruned" in name or "P2 @640" in name else 0.006
        ax.annotate(f"{name}\n{ap:.3f}" + (f" → {ap_md:.3f}" if ap_md else "") + (f"\n({note})" if note else ""),
                    (x, ap), textcoords="offset points", xytext=(10, dy * 100), fontsize=7.5, va="center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mAP50-95 (VisDrone val)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.165, 0.290)
    if logx:
        ax.set_xscale("log")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    panel(axes[0], "lat", "单图前向延迟 (ms, RTX 4060 Laptop, FP32, batch 1) — 越左越好")
    axes[0].set_title("(a) 精度 - 延迟")
    panel(axes[1], "gflops", "计算量 GFLOPs (640 输入基线 = 5.9) — 越左越好")
    axes[1].set_title("(b) 精度 - 计算量")
    fig.suptitle("VisDrone 小目标检测：精度-速度权衡（实心=同分辨率训练推理，空心=max_det=1000）", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_tradeoff.png", dpi=140)
    print("saved", OUT / "fig_tradeoff.png")

    f, ax = plt.subplots(figsize=(9, 6))
    panel(ax, "lat", "单图前向延迟 (ms, RTX 4060 Laptop, FP32) — 越左越好")
    ax.set_title("精度 - 速度折线图（VisDrone val, YOLO26n, 100 epoch）")
    f.tight_layout()
    f.savefig(OUT / "fig_tradeoff_latency.png", dpi=150)
    print("saved", OUT / "fig_tradeoff_latency.png")


if __name__ == "__main__":
    main()
