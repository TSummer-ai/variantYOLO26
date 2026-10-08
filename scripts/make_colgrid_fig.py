#!/usr/bin/env python
"""列切分 vs 方形网格：letterbox 宽度受限的几何证据 + AP 证据（4 面板）。

数据源：RESULTS_SLICING.md（VisDrone val；3x1/2x2 为全量 548 图，2x1/4x1 为 120 图子集）
公平性处理：绝对值混用不同子集不可比 => 主口径用 ΔAP（相对各自协议的基线）。

几何（W=1360, H=765, imgsz=960, overlap=0.2, 单行 ny=1）：
  切片宽 tw = W*(1+0.2*(nx-1))/nx ；缩放 = min(960/tw, 960/765)
  nx=1: tw=1360 -> min(0.706, 1.255)=0.706  (整图)
  nx=2: tw= 816 -> min(1.176, 1.255)=1.176
  nx=3: tw= 544 -> min(1.765, 1.255)=1.255  <- 高度成为限制
  nx>=3: 缩放锁定在 1.255，放大倍数不再增加
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path("/home/wang/DeepSeek/YOLO/figures")
FIG.mkdir(exist_ok=True)

W, H, IMG = 1360.0, 765.0, 960.0
OV = 0.2
BASE_FULL, BASE_SUB = 0.2512, 0.2472

# ---- 几何：nx = 1..5，单行 ----
NX = np.array([1, 2, 3, 4, 5])
tw = W * (1 + OV * (NX - 1)) / NX
width_lim = IMG / tw
height_lim = np.full_like(width_lim, IMG / H)
eff = np.minimum(width_lim, height_lim)
mag = eff / (IMG / W)                      # 相对整图的放大倍数

# ---- AP 数据 ----
# (标签, nx, ny, 前向次数, AP50-95, 协议)
AP = [("full image", 1, 1, 1, 0.2512, "full"),
      ("2x1", 2, 1, 3, 0.2641, "subset"),
      ("3x1", 3, 1, 4, 0.2750, "full"),
      ("2x2", 2, 2, 5, 0.2703, "full"),
      ("4x1", 4, 1, 5, 0.2631, "subset")]
MAG_OF = {1: 1.00, 2: 1.667, 3: 1.778, 4: 1.778}

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
axes = axes.ravel()

# ================= (a) 机制：宽度限制 vs 高度限制 =================
ax = axes[0]
ax.plot(NX, width_lim, "o--", color="#4c78a8", label="width limit  $960/t_w$")
ax.plot(NX, height_lim, "s--", color="#e45756", label="height limit  $960/H=1.255$")
ax.plot(NX, eff, "-", color="k", lw=2.4, label="effective $\\min(\\cdot)$")
ax.axvspan(0.8, 2.0, color="#4c78a8", alpha=0.10)
ax.axvspan(2.0, 5.4, color="#e45756", alpha=0.10)
ax.text(1.40, 0.42, "width-\nlimited", ha="center", fontsize=8.5, color="#2b5b8c")
ax.text(3.85, 0.42, "height-limited\n(no further gain)", ha="center", fontsize=8.5, color="#a33")
for x, e, m in zip(NX, eff, mag):
    ax.annotate(f"{m:.2f}x", (x, e), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(NX); ax.set_xlim(0.7, 5.4); ax.set_ylim(0.3, 3.15)
ax.set_xlabel("number of columns $n_x$ (single row)")
ax.set_ylabel("letterbox scale of a tile")
ax.set_title("(a) Mechanism: tiling gains scale only while width-limited", fontsize=10.5)
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)

# ================= (b) 放大倍数 vs 布局 =================
ax = axes[1]
labels = ["full\n1x1", "2x1", "3x1", "2x2", "4x1"]
mags = [1.00, 1.667, 1.778, 1.667, 1.778]
cols = ["#9aa5b1", "#8fa8c8", "#2e7d32", "#8fa8c8", "#d9534f"]
b = ax.bar(labels, mags, color=cols)
for r, v in zip(b, mags):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}x", ha="center", fontsize=9)
ax.axhline(1.778, ls="--", c="k", lw=1.2)
ax.text(0.03, 1.735, "height-limit ceiling 1.78x", fontsize=8.5,
        transform=ax.get_yaxis_transform())
ax.set_ylim(0.85, 2.0); ax.set_ylabel("effective magnification vs full image")
ax.set_title("(b) 2x2 = 2x1 (both width-limited); 3x1 = 4x1 = 1.78x", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

# ================= (c) ΔAP vs 布局 =================
ax = axes[2]
d = [(l, f, ap - (BASE_FULL if p == "full" else BASE_SUB), p) for l, _, _, f, ap, p in AP]
xs = np.arange(len(d)); vals = [x[2] * 100 for x in d]
cols = ["#9aa5b1" if x[0] == "full image" else ("#2e7d32" if x[0] == "3x1" else "#8fa8c8") for x in d]
b = ax.bar(xs, vals, color=cols, hatch=["", "//", "", "", "//"], edgecolor="k", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f"{x[0]}\n{x[1]} fwd" for x in d], fontsize=9)
for i, v in enumerate(vals):
    ax.text(i, v + 0.05, f"{v:+.2f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 2.75)
ax.set_ylabel(r"$\Delta$ mAP50-95 vs its own baseline (points)")
ax.set_title("(c) 3x1: +2.38 @4 fwd  >  2x2: +1.91 @5 fwd  >  4x1: +1.59\n"
             "(hatched = 120-image subset; others = full val)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

# ================= (d) 放大倍数 vs ΔAP =================
ax = axes[3]
for l, nx, ny, f, ap, p in AP:
    if l == "full image":
        continue
    x = MAG_OF[nx]; y = (ap - (BASE_FULL if p == "full" else BASE_SUB)) * 100
    ax.scatter(x, y, s=110 + f * 60,
               color="#2e7d32" if l == "3x1" else ("#d9534f" if l == "4x1" else "#8fa8c8"),
               alpha=0.85, edgecolors="k", zorder=3)
    ax.annotate(f"{l} ({f} fwd)", (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9)
ax.axvline(1.778, ls=":", c="k", lw=1.2)
ax.text(1.762, 2.62, "magnification ceiling", rotation=90, fontsize=8.5, va="top", ha="right")
ax.set_xlim(1.60, 1.90); ax.set_ylim(1.2, 2.8)
ax.set_xlabel("effective magnification")
ax.set_ylabel(r"$\Delta$ mAP50-95 (points)")
ax.set_title("(d) Past the ceiling: 3x1 and 4x1 share 1.78x, but 4x1 is worse\n"
             "-> extra columns add compute/duplicates, not magnification", fontsize=10.5)
ax.grid(alpha=0.3)

fig.suptitle("Why column-splitting beats square grids on wide (1360x765) VisDrone images "
             "(imgsz=960, overlap=0.2, single row)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = FIG / "fig_column_vs_grid.png"
fig.savefig(out, dpi=170); plt.close(fig)
print(f"已生成 {out}")
