#!/usr/bin/env python
"""生成论文/参赛用的汇总图表（纯读日志，不需要 GPU）。

产出 figures/ 下：
  fig1_size_missrate.png  分尺寸漏检率：base@640 / P2@640 / P2@960（机制图）
  fig2_oracle.png         oracle 四象限分解（三配置）
  fig3_pareto.png         精度-延迟帕累托
  fig4_gt_size.png        GT 尺寸分布 @640 vs @960
  fig_all.png             四联图
"""

from __future__ import annotations

import os

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（系统自带 Noto Sans CJK）
from matplotlib import font_manager
_CJK = next((f for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
             if Path(f).exists()), None)
if _CJK:
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False
from PIL import Image

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "results/figures"
OUT.mkdir(parents=True, exist_ok=True)
BUCKETS = ["0-8", "8-16", "16-32", "32-64", ">=64"]

# 实测的工作点（bench_models.py + val 日志）
POINTS = {
    "baseline@640": dict(gflops=5.9, lat=4.57, e2e=6.26, ap=0.1820, ap50=0.3280, color="#888888"),
    "P2@640":       dict(gflops=7.7, lat=5.88, e2e=7.01, ap=0.1980, ap50=0.3490, color="#1f77b4"),
    "P2@960":       dict(gflops=17.6, lat=8.96, e2e=8.73, ap=0.2630, ap50=0.4400, color="#d62728"),
}
ORACLE_LOGS = {"baseline@640": "results/oracle_logs/oracle.log",
               "P2@640": "results/oracle_logs/oracle_p2.log",
               "P2@960": "results/oracle_logs/oracle_p2_960.log"}


def parse_oracle(path: Path):
    """返回 (漏检率 dict, oracle AP dict)，取 head=o2m 段。"""
    if not path.exists():
        return {}, {}
    lines = path.read_text(errors="ignore").splitlines()
    # "head=o2m" 会出现两次（段标记 + GT 行），必须按行定位，不能 split
    starts = [i for i, l in enumerate(lines) if "head=o2m" in l]
    ends = [i for i, l in enumerate(lines) if "head=e2e" in l]
    assert starts, f"{path} 里没有 head=o2m"
    end = next((i for i in ends if i > starts[0]), len(lines))
    sec = "\n".join(lines[starts[0]:end])
    miss = {}
    for line in sec.splitlines():
        m = re.match(r"\s*(0-8|8-16|16-32|32-64|>=64)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            miss[m.group(1)] = float(m.group(4))
    aps = {}
    for line in sec.splitlines():
        m = re.match(r"(real|loc-oracle|cls-oracle|fn-oracle|fp-oracle)（[^）]*）\s+AP50-95=([\d.]+)", line)
        if m:
            aps[m.group(1)] = float(m.group(2))
    return miss, aps


def fig_size_missrate(ax):
    vals = {}
    for name, log in ORACLE_LOGS.items():
        miss, _ = parse_oracle(ROOT / log)
        vals[name] = [miss.get(b, np.nan) for b in BUCKETS]
    x = np.arange(len(BUCKETS))
    w = 0.26
    for i, (name, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1) * w, v, w, label=name, color=POINTS[name]["color"])
        for xi, vi in zip(x + (i - 1) * w, v):
            ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} px" for b in BUCKETS])
    ax.set_ylabel("miss rate (IoU 0.5, conf 0.001)")
    ax.set_xlabel("GT size sqrt(area), original image")
    ax.set_title("(a) Miss rate by object size — 改善随目标变小单调增大")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)


def fig_oracle(ax):
    keys = ["loc-oracle", "cls-oracle", "fn-oracle", "fp-oracle"]
    labels = ["+定位完美", "+分类完美", "+不漏检", "+无误检"]
    x = np.arange(len(keys))
    w = 0.26
    for i, (name, log) in enumerate(ORACLE_LOGS.items()):
        _, aps = parse_oracle(ROOT / log)
        base = aps.get("real", 0)
        gains = [aps.get(k, 0) - base for k in keys]
        ax.bar(x + (i - 1) * w, gains, w, label=f"{name} (real={base:.3f})", color=POINTS[name]["color"])
        for xi, gi in zip(x + (i - 1) * w, gains):
            ax.text(xi, gi + 0.008, f"{gi:+.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP50-95 gain (COCO protocol)")
    ax.set_title("(b) Oracle 分解：漏检始终是最大项")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def fig_pareto(ax):
    for name, p in POINTS.items():
        ax.scatter(p["lat"], p["ap"], s=110, color=p["color"], zorder=3)
        ax.annotate(f"{name}\n{p['ap']:.3f} AP / {p['lat']:.1f} ms", (p["lat"], p["ap"]),
                    textcoords="offset points", xytext=(8, -6), fontsize=8)
    xs = [p["lat"] for p in POINTS.values()]
    ys = [p["ap"] for p in POINTS.values()]
    ax.plot(xs, ys, "--", color="gray", alpha=0.6, zorder=1)
    ax.set_xlabel("forward latency (ms/img, RTX 4060 Laptop, FP32)")
    ax.set_ylabel("mAP50-95 (val)")
    ax.set_title("(c) 精度-延迟帕累托")
    ax.grid(alpha=0.3)
    ax.set_xlim(3.8, 10.0)


def fig_gt_size(ax):
    if not (ROOT / "datasets/visdrone/val.txt").exists():
        ax.text(0.5, 0.5, "需要 VisDrone 数据集\n（准备方式见 README）", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return
    from collections import Counter
    ds = ROOT / "datasets/visdrone"
    sizes = {}
    for line in open(ds / "val.txt"):
        ip = ds / line.strip()[2:]
        with Image.open(ip) as im:
            w, h = im.size
        lp = ds / "labels/val" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        for ln in open(lp):
            _, _, _, bw, bh = ln.split()
            sizes.setdefault("orig", []).append(np.sqrt(float(bw) * w * float(bh) * h))
            for tag, imgsz in (("640", 640), ("960", 960)):
                s = min(imgsz / w, imgsz / h)
                sizes.setdefault(tag, []).append(np.sqrt(float(bw) * w * float(bh) * h) * s)
    bins = [0, 8, 12, 16, 24, 32, 64, 1e9]
    labels = ["0-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
    x = np.arange(len(labels))
    w = 0.38
    for i, (tag, lab) in enumerate((("640", "letterbox @640"), ("960", "letterbox @960"))):
        arr = np.array(sizes[tag])
        pct = [np.mean((arr >= lo) & (arr < hi)) * 100 for lo, hi in zip(bins[:-1], bins[1:])]
        ax.bar(x + (i - 0.5) * w, pct, w, label=lab, color=["#1f77b4", "#d62728"][i])
        for xi, p in zip(x + (i - 0.5) * w, pct):
            if p > 1:
                ax.text(xi, p + 0.6, f"{p:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of GT")
    ax.set_xlabel("object size at network input (px)")
    ax.set_title("(d) 分辨率把目标'变大'：<8px 占比 31.1% → 13.4%")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig_size_missrate(axes[0, 0])
    fig_oracle(axes[0, 1])
    fig_pareto(axes[1, 0])
    fig_gt_size(axes[1, 1])
    fig.suptitle("VisDrone 小目标检测：误差分解 → 表示侧优化（YOLO26n, 100 epoch）", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "fig_all.png", dpi=140)
    print("saved", OUT / "fig_all.png")

    for name, fn in (("fig1_size_missrate", fig_size_missrate), ("fig2_oracle", fig_oracle),
                     ("fig3_pareto", fig_pareto), ("fig4_gt_size", fig_gt_size)):
        f, ax = plt.subplots(figsize=(7, 5))
        fn(ax)
        f.tight_layout()
        f.savefig(OUT / f"{name}.png", dpi=150)
        print("saved", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
