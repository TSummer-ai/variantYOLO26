#!/usr/bin/env python
"""切片参数寻优图（数据源：RESULTS_SLICING.md 第四节，120 图子集，COCO 协议）。

诚实性设计：明确区分「单变量干净对比」与「多变量混淆组」。
  - big_px：干净（其余固定 2x2 / 0.3 / 0.6）
  - merge_iou：干净（其余固定 2x2 / 0.3 / big_px=48）
  - grid 与 overlap：现有数据中同时变了 grid+overlap+big_px，只能作为"联合配置"呈现
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = Path("/home/wang/DeepSeek/YOLO/figures")
FIG.mkdir(exist_ok=True)

BASE = 0.2472          # 同一 120 图子集上的"仅整图"基线
# ---- 干净单变量：big_px（2x2, overlap 0.3, merge 0.6）----
BIGPX = [(r"48", 0.2633), (r"32", 0.2593), (r"16", 0.2522), ("no gate", 0.2552)]
# ---- 干净单变量：merge_iou（2x2, 0.3, big_px=48）----
MERGE = [("0.6", 0.2633), ("0.5", 0.2626)]
# ---- 联合配置排名（含混淆组）----
ALL = [
    ("2x2 / 0.3 /\nbig48 / iou0.6", 0.2633, True),
    ("2x2 / 0.3 /\nbig48 / iou0.5", 0.2626, True),
    ("2x2 / 0.3 /\nbig32 / iou0.6", 0.2593, True),
    ("3x3 / 0.2 /\nbig32 / iou0.6", 0.2575, False),
    ("2x2 / 0.3 /\nno gate / iou0.6", 0.2552, True),
    ("2x2 / 0.3 /\nbig16 / iou0.6", 0.2522, True),
]

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

# ---- (a) big_px ----
ax = axes[0]
labels = [b[0] for b in BIGPX]; vals = [b[1] for b in BIGPX]
colors = ["#2e7d32" if v == max(vals) else "#8fa8c8" for v in vals]
colors[3] = "#d9534f"                       # no gate 用红色强调
b = ax.bar(labels, vals, color=colors)
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)
ax.axhline(BASE, ls="--", c="k", lw=1.2)
ax.text(0.02, BASE + 0.0004, f"full image only ({BASE:.4f})", fontsize=8, transform=ax.get_yaxis_transform())
ax.set_ylim(0.244, 0.2665)
ax.set_ylabel("mAP50-95 (120-image subset)")
ax.set_xlabel("big_px (px)")
ax.set_title("(a) Size gate threshold — clean single-variable sweep", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# ---- (b) merge_iou ----
ax = axes[1]
labels = [m[0] for m in MERGE]; vals = [m[1] for m in MERGE]
b = ax.bar(labels, vals, color=["#2e7d32", "#8fa8c8"])
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=8)
ax.axhline(BASE, ls="--", c="k", lw=1.2)
ax.set_ylim(0.244, 0.2665)
ax.set_xlabel("merge NMS IoU")
ax.set_title("(b) Merge IoU — clean single-variable sweep", fontsize=10)
ax.grid(axis="y", alpha=0.3)

# ---- (c) 全部配置排名（含混淆组）----
ax = axes[2]
labels = [a[0] for a in ALL]; vals = [a[1] for a in ALL]; clean = [a[2] for a in ALL]
b = ax.barh(range(len(vals))[::-1], vals,
            color=["#2e7d32" if c else "#e0a800" for c in clean])
for i, (v, c) in enumerate(zip(vals, clean)):
    ax.text(v + 0.0003, len(vals) - 1 - i, f"{v:.4f}" + ("" if c else "  (confounded)"),
            va="center", fontsize=8)
ax.axvline(BASE, ls="--", c="k", lw=1.2)
ax.set_yticks(range(len(vals))[::-1]); ax.set_yticklabels(labels, fontsize=7.5)
ax.set_xlim(0.244, 0.2685)
ax.set_xlabel("mAP50-95")
ax.set_title("(c) All configs — yellow rows change >1 variable\n(grid+overlap+big_px together); do not attribute separately",
             fontsize=10)
ax.grid(axis="x", alpha=0.3)

fig.suptitle("Tiled-inference hyperparameter sweep (VisDrone val subset, COCO protocol) — "
             "monotone, same-direction evidence", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = FIG / "fig_slicing_tuning.png"
fig.savefig(out, dpi=170); plt.close(fig)
print(f"已生成 {out}")

# 同时导出 CSV 供表格引用
csv = FIG.parent / "results" / "slicing_tuning.csv"
csv.parent.mkdir(exist_ok=True)
with open(csv, "w") as f:
    f.write("grid,overlap,big_px,merge_iou,AP50-95,delta_vs_full_image,clean_single_variable\n")
    f.write(f"-, -, -, -, {BASE:.4f}, 0.0000, baseline(full image)\n")
    for g, o, bx, mi, v, c in [("2", "0.3", "48", "0.6", 0.2633, 1), ("2", "0.3", "48", "0.5", 0.2626, 1),
                               ("2", "0.3", "32", "0.6", 0.2593, 1), ("3", "0.2", "32", "0.6", 0.2575, 0),
                               ("2", "0.3", "none", "0.6", 0.2552, 1), ("2", "0.3", "16", "0.6", 0.2522, 1)]:
        f.write(f"{g},{o},{bx},{mi},{v:.4f},{v-BASE:+.4f},{'yes' if c else 'no'}\n")
print(f"已生成 {csv}")
