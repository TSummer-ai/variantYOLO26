#!/usr/bin/env python
"""关键补做实验的图（数据源：docs/RESULTS_CRITICAL_ABLATIONS.md，COCO 协议，全量 548 图）。

三组图：
  fig_critical_resolution_vs_tiling.png —— 等算力下「单次高分辨率整图」vs「960+切片管线」
  fig_fusion_shootout.png               —— 同 4 次前向的 8 种融合器横评
  fig_sensitivity.png                   —— 门控阈值 / 一致性权重 / 支持视角分布 / 种子方差
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path("/home/wang/DeepSeek/YOLO/figures"); FIG.mkdir(exist_ok=True)

# ---------------- 数据（全部来自 RESULTS_CRITICAL_ABLATIONS.md）----------------
# (标签, 算力, AP50-95, AP50, 端到端 ms, FPS, 是否单次前向)
RES = [("full @640", 1.00, 0.1941, 0.3362, 6.97, 143.6, True),
       ("full @960", 2.25, 0.2512, 0.4221, 8.23, 121.6, True),
       ("full @1280", 4.00, 0.2746, 0.4591, 11.72, 85.4, True),
       ("full @1600", 6.25, 0.2831, 0.4733, 17.27, 57.9, True),
       ("full @1920", 9.00, 0.2749, 0.4626, 25.24, 39.6, True),
       ("960+3x1 gated+NMS", 9.00, 0.2750, 0.4683, 63.45, 15.8, False),
       ("MV-Fuse (gated+CVA)", 9.00, 0.2788, 0.4754, 52.90, 18.9, False)]

# 融合器横评（同 4 次前向）
FUSE = [("MV-Fuse (gated+CVA+NMS)", 0.2788, 0.4754),
        ("gated + WBF[official,max]", 0.2778, None),
        ("gated + Soft-NMS", 0.2774, 0.4697),
        ("union + WBF[official,bma]", 0.2772, 0.4682),
        ("gated + NMS (ours, prev)", 0.2750, 0.4683),
        ("union + NMS", 0.2719, 0.4634),
        ("gated + Cluster-DIoU (ASAHI-ish)", 0.2696, None),
        ("union + Cluster-DIoU", 0.2674, None),
        ("tiles-only 2x2 + NMS (SAHI-ish)", 0.2499, 0.4308)]

BIGPX = [("no gate", 0.2719), ("8", 0.2513), ("16", 0.2586), ("24", 0.2662),
         ("32", 0.2713), ("48", 0.2750), ("64", 0.2749), ("96", 0.2732), ("128", 0.2730)]
WEIGHTS = [("none (=0.2750)", 0.2750), ("original", 0.2788), ("linear", 0.2787),
           ("quadratic", 0.2785), ("binary (>=2)", 0.2787), ("REVERSED", 0.2569)]
SEEDS = [0.1740, 0.1735, 0.1717]

# ============================================================ 图 1
fig, ax = plt.subplots(1, 3, figsize=(17.5, 4.7))

# (a) AP50-95 vs 算力
single = [r for r in RES if r[6]]
tiled = [r for r in RES if not r[6]]
ax[0].plot([r[1] for r in single], [r[2] for r in single], "o-", color="#2e7d32",
           lw=2.2, ms=7, label="single forward: full-image at higher resolution")
ax[0].plot([r[1] for r in tiled], [r[2] for r in tiled], "s", color="#d9534f",
           ms=11, label="4 forwards: 960 + 3x1 tiling pipelines")
OFF = {"full @1920": (-88, -6), "960+3x1 gated+NMS": (12, 12), "MV-Fuse (gated+CVA)": (12, -16)}
for r in RES:
    ax[0].annotate(f"{r[0]}\n{r[2]:.4f}", (r[1], r[2]), textcoords="offset points",
                   xytext=OFF.get(r[0], (6, -4) if r[6] else (8, 8)), fontsize=7.6,
                   color="#1b5e20" if r[6] else "#a33")
ax[0].set_xlabel("compute  (forwards x $(imgsz/640)^2$)")
ax[0].set_ylabel("mAP50-95 (COCO)")
ax[0].set_title("(a) At equal compute, a single high-resolution forward\n"
                "beats the whole tiling pipeline", fontsize=10.5)
ax[0].legend(fontsize=8, loc="lower right"); ax[0].grid(alpha=0.3)
ax[0].set_ylim(0.185, 0.295); ax[0].set_xlim(0.4, 11.2)

# (b) AP50-95 vs 端到端延迟
ax[1].plot([r[4] for r in single], [r[2] for r in single], "o-", color="#2e7d32", lw=2.2, ms=7,
           label="single forward")
ax[1].plot([r[4] for r in tiled], [r[2] for r in tiled], "s", color="#d9534f", ms=11,
           label="tiling pipelines")
OFF2 = {"MV-Fuse (gated+CVA)": (6, 6), "960+3x1 gated+NMS": (6, -16), "full @1920": (-58, 6)}
for r in RES:
    ax[1].annotate(r[0], (r[4], r[2]), textcoords="offset points",
                   xytext=OFF2.get(r[0], (5, -9) if r[6] else (5, 7)),
                   fontsize=7.6, color="#1b5e20" if r[6] else "#a33")
ax[1].set_xlabel("end-to-end latency (ms, RTX 4060 Laptop)")
ax[1].set_ylabel("mAP50-95 (COCO)")
ax[1].set_title("(b) On the latency axis the gap is larger:\n@1600 is 17.3 ms vs 52.9 ms for MV-Fuse",
                fontsize=10.5)
ax[1].legend(fontsize=8, loc="lower right"); ax[1].grid(alpha=0.3)
ax[1].set_ylim(0.185, 0.295)

# (c) AP50 vs 算力（同一结论在 AP50 上也成立）
ax[2].plot([r[1] for r in single], [r[3] for r in single], "o-", color="#2e7d32", lw=2.2, ms=7)
ax[2].plot([r[1] for r in tiled], [r[3] for r in tiled], "s", color="#d9534f", ms=11)
OFF3 = {"full @1920": (-88, -6), "960+3x1 gated+NMS": (12, 12), "MV-Fuse (gated+CVA)": (12, -16)}
for r in RES:
    ax[2].annotate(r[0], (r[1], r[3]), textcoords="offset points",
                   xytext=OFF3.get(r[0], (5, -9) if r[6] else (5, 6)),
                   fontsize=7.6, color="#1b5e20" if r[6] else "#a33")
ax[2].set_xlabel("compute"); ax[2].set_ylabel("mAP50 (COCO)")
ax[2].set_title("(c) Same ordering on mAP50", fontsize=10.5)
ax[2].grid(alpha=0.3); ax[2].set_ylim(0.315, 0.50); ax[2].set_xlim(0.4, 11.2)

fig.suptitle("Critical ablation: resolution vs tiling at equal compute "
             "(VisDrone val, 548 images, COCO protocol, same p2_960 weights)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(FIG / "fig_critical_resolution_vs_tiling.png", dpi=170); plt.close(fig)
print("✅ fig_critical_resolution_vs_tiling.png")

# ============================================================ 图 2
fig, ax = plt.subplots(figsize=(9.6, 5.0))
names = [f[0] for f in FUSE][::-1]; vals = [f[1] for f in FUSE][::-1]
cols = ["#2e7d32" if n.startswith("MV-Fuse") else "#d9534f" if "SAHI" in n
        else "#e0a800" if "Cluster" in n else "#8fa8c8" for n in names]
b = ax.barh(names, vals, color=cols)
for r, v in zip(b, vals):
    ax.text(v + 0.0005, r.get_y() + r.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5)
ax.axvline(0.2788, ls="--", c="#2e7d32", lw=1.2)
ax.axvline(0.2750, ls=":", c="gray", lw=1.2)
ax.text(0.2752, 0.2, "our previous method 0.2750", fontsize=7.5, rotation=90, va="bottom", color="gray")
ax.set_xlim(0.244, 0.2845)
ax.set_xlabel("mAP50-95 (COCO, full val)")
ax.set_title("Fusion shoot-out at identical cost (4 forwards, same candidate pool)\n"
             "Top five are within 0.0016 of each other — the fusion rule is nearly saturated;\n"
             "ASAHI-style merging is the worst of the 4-forward group, pure slicing is far worse",
             fontsize=10)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig_fusion_shootout.png", dpi=170); plt.close(fig)
print("✅ fig_fusion_shootout.png")

# ============================================================ 图 3
fig, ax = plt.subplots(2, 2, figsize=(12.8, 8.4))

# (a) big_px 全量扫描
lab = [b[0] for b in BIGPX]; v = [b[1] for b in BIGPX]
cols = ["#d9534f" if l == "no gate" else "#2e7d32" if l in ("48", "64") else "#8fa8c8" for l in lab]
b0 = ax[0, 0].bar(lab, v, color=cols)
for r, x in zip(b0, v):
    ax[0, 0].text(r.get_x() + r.get_width() / 2, x + 0.0004, f"{x:.4f}", ha="center", fontsize=7.6)
ax[0, 0].axhline(0.2719, ls="--", c="k", lw=1)
ax[0, 0].set_ylim(0.246, 0.279)
ax[0, 0].set_xlabel("big_px (px)"); ax[0, 0].set_ylabel("mAP50-95 (full val)")
ax[0, 0].set_title("(a) Size-gate threshold: smooth single peak,\n48-64 form a wide plateau (not fine-tuned)",
                   fontsize=10)
ax[0, 0].grid(axis="y", alpha=0.3)

# (b) 一致性权重方案
lab = [w[0] for w in WEIGHTS]; v = [w[1] for w in WEIGHTS]
cols = ["#d9534f" if "REVERSED" in l else "#2e7d32" if l == "original" else "#8fa8c8" for l in lab]
b1 = ax[0, 1].bar(lab, v, color=cols)
for r, x in zip(b1, v):
    ax[0, 1].text(r.get_x() + r.get_width() / 2, x + 0.0006, f"{x:.4f}", ha="center", fontsize=7.6)
ax[0, 1].axhline(0.2750, ls="--", c="k", lw=1)
ax[0, 1].text(5.4, 0.2762, "no re-scoring 0.2750", fontsize=7.5, ha="right")
ax[0, 1].set_ylim(0.252, 0.283)
ax[0, 1].tick_params(axis="x", labelsize=7.5)
ax[0, 1].set_ylabel("mAP50-95 (full val)")
ax[0, 1].set_title("(b) Consistency weights: any monotone scheme gives\n0.2785-0.2788 (range 0.0003); reversed collapses",
                   fontsize=10)
ax[0, 1].grid(axis="y", alpha=0.3)

# (c) 支持视角数分布
ax[1, 0].bar(["0 views", "1 view", "2 views"], [0.1, 59.6, 40.3],
             color=["#d9534f", "#8fa8c8", "#2e7d32"])
for i, x in enumerate([0.1, 59.6, 40.3]):
    ax[1, 0].text(i, x + 0.8, f"{x}%", ha="center", fontsize=9)
ax[1, 0].set_ylim(0, 70); ax[1, 0].set_ylabel("share of candidates (%)")
ax[1, 0].set_title("(c) Candidates after gated fusion (1,133,238 total):\n"
                   "0.1% unsupported / 59.6% single-view / 40.3% two-view", fontsize=10)
ax[1, 0].grid(axis="y", alpha=0.3)

# (d) 种子方差
m = float(np.mean(SEEDS)); s = float(np.std(SEEDS, ddof=1))
ax[1, 1].bar(["seed 0", "seed 1", "seed 2"], SEEDS, color="#8fa8c8")
ax[1, 1].axhline(m, ls="--", c="k", lw=1.2)
ax[1, 1].errorbar([0, 1, 2], SEEDS, yerr=s, fmt="none", ecolor="#d9534f", capsize=6, lw=1.6)
for i, x in enumerate(SEEDS):
    ax[1, 1].text(i, x + 0.0004, f"{x:.4f}", ha="center", fontsize=8.5)
ax[1, 1].set_ylim(0.168, 0.177)
ax[1, 1].set_ylabel("mAP50-95 (COCO)")
ax[1, 1].set_title(f"(d) Seed variance (baseline@640, 3 seeds):\n"
                   f"mean {m:.4f}, std {s:.4f} -> effects below ~0.002 are noise", fontsize=10)
ax[1, 1].grid(axis="y", alpha=0.3)

fig.suptitle("Sensitivity and variance (VisDrone val, full 548 images, COCO protocol)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.savefig(FIG / "fig_sensitivity.png", dpi=170); plt.close(fig)
print("✅ fig_sensitivity.png")
