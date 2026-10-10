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

# ================= (c) 绝对 AP50-95（两条基线，协议可见）=================
ax = axes[2]
xs = np.arange(len(AP)); vals_abs = [x[4] for x in AP]
cols = ["#9aa5b1" if x[0] == "full image" else ("#2e7d32" if x[0] == "3x1" else "#8fa8c8") for x in AP]
hatches = ["", "//", "", "", "//"]          # 斜纹 = 120 图子集
b = ax.bar(xs, vals_abs, color=cols, hatch=hatches, edgecolor="k", linewidth=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f"{x[0]}\n{x[3]} fwd" for x in AP], fontsize=9)
for i, (x, v) in enumerate(zip(AP, vals_abs)):
    delta = (v - (BASE_FULL if x[5] == "full" else BASE_SUB)) * 100
    ax.text(i, v + 0.0012, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i, v + 0.0002, f"({delta:+.2f})", ha="center", fontsize=8, color="#444")
ax.axhline(BASE_FULL, ls="--", c="#1f4e79", lw=1.2)
ax.axhline(BASE_SUB, ls=":", c="#a33", lw=1.4)
ax.text(0.985, BASE_FULL + 0.0005, f"baseline full val {BASE_FULL:.4f}", fontsize=8,
        color="#1f4e79", ha="right", transform=ax.get_yaxis_transform())
ax.text(0.985, BASE_SUB + 0.0005, f"baseline 120-img subset {BASE_SUB:.4f}", fontsize=8,
        color="#a33", ha="right", transform=ax.get_yaxis_transform())
ax.set_ylim(0.244, 0.2815)
ax.set_ylabel("mAP50-95 (absolute)")
ax.set_title("(c) Absolute AP: 3x1 = 0.2750 / 2x2 = 0.2703 / 4x1 = 0.2631\n"
             "(hatched = 120-image subset; use the matching dashed/dotted baseline)", fontsize=10.5)
ax.grid(axis="y", alpha=0.3)

# ================= (d) 放大倍数 vs ΔAP =================
ax = axes[3]
for l, nx, ny, f, ap, p in AP:
    if l == "full image":
        continue
    x = MAG_OF[nx]; y = ap
    ax.scatter(x, y, s=110 + f * 60,
               color="#2e7d32" if l == "3x1" else ("#d9534f" if l == "4x1" else "#8fa8c8"),
               alpha=0.85, edgecolors="k", zorder=3)
    ax.annotate(f"{l} ({f} fwd)", (x, y), textcoords="offset points", xytext=(8, 7), fontsize=9)
ax.axvline(1.778, ls=":", c="k", lw=1.2)
ax.text(1.764, 0.2775, "magnification ceiling", rotation=90, fontsize=8.5, va="top", ha="right")
ax.set_xlim(1.60, 1.90); ax.set_ylim(0.2585, 0.2805)
ax.set_xlabel("effective magnification")
ax.set_ylabel("mAP50-95 (absolute)")
ax.set_title("(d) Past the ceiling: 3x1 and 4x1 share 1.78x, but 4x1 is worse\n"
             "-> extra columns add compute/duplicates, not magnification", fontsize=10.5)
ax.grid(alpha=0.3)

fig.suptitle("Why column-splitting beats square grids on wide (1360x765) VisDrone images "
             "(imgsz=960, overlap=0.2, single row)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = FIG / "列切分与方形网格对比.png"
fig.savefig(out, dpi=170); plt.close(fig)
print(f"已生成 {out}")
