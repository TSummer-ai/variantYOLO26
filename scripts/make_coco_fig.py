#!/usr/bin/env python
"""COCO val2017 跨数据集泛化图（数据源 exp_coco.json，COCO 协议）。

(a) 变体 B：官方 COCO 预训练 YOLO26n 在 COCO val2017 上，整图 vs MV-Fuse vs 朴素并集
    —— 说明方法跨数据集不损害（+1.41 AP50），且尺寸门控是安全阀（无门控 −11.05 AP50-95）
(b) 变体 A：VisDrone 训练模型迁到 COCO（6 类映射）—— 说明检测器跨域迁移能力有限（诚实下界）
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path("/home/wang/DeepSeek/YOLO/figures"); FIG.mkdir(exist_ok=True)

# ---- 变体 B（COCO 模型 @ COCO val2017，5000 图）----
B = [("full image\n(1 forward)", 0.3958, 0.5495, "#9aa5b1"),
     ("+ size gate + CVA\n(MV-Fuse, 4 fwd)", 0.3986, 0.5636, "#2e7d32"),
     ("plain union\n(no gate, 4 fwd)", 0.2853, 0.4341, "#d9534f")]
# ---- 变体 A（VisDrone 模型 @ COCO，6 类映射，5000 图）----
A_VIS = 0.2788   # VisDrone val 同 6 类的表现（P2@960 + MV-Fuse）
A_COCO = 0.0354
A_VIS50, A_COCO50 = 0.4754, 0.0665

fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.9))

# ---------- (a) ----------
x = np.arange(len(B))
b = ax[0].bar(x, [v[1] for v in B], 0.5, color=[v[3] for v in B])
for r, v in zip(b, B):
    ax[0].text(r.get_x() + r.get_width() / 2, v[1] + 0.004, f"{v[1]:.4f}", ha="center",
               fontsize=10, fontweight="bold")
    ax[0].text(r.get_x() + r.get_width() / 2, 0.02, f"mAP50\n{v[2]:.4f}", ha="center",
               fontsize=8.5, color="white", fontweight="bold")
ax[0].annotate("", xy=(2, 0.4400), xytext=(1, 0.4400),
               arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=1.5))
ax[0].text(1.5, 0.4470, "+0.0028", ha="center", fontsize=9, color="#2e7d32", fontweight="bold")
ax[0].annotate("", xy=(2, 0.3450), xytext=(1, 0.3450),
               arrowprops=dict(arrowstyle="<->", color="#d9534f", lw=1.5))
ax[0].text(1.5, 0.3520, "−0.1133 without the gate", ha="center", fontsize=9,
           color="#d9534f", fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels([v[0] for v in B], fontsize=9)
ax[0].set_ylim(0, 0.65); ax[0].set_ylabel("mAP50-95 (COCO val2017)")
ax[0].set_title("(a) The method transfers to a different dataset without harm\n"
                "and the size gate is what keeps it safe", fontsize=10.5)
ax[0].grid(axis="y", alpha=0.3)

# ---------- (b) ----------
bars = ax[1].bar(["VisDrone val\n(6 shared classes)", "COCO val2017\n(6-class transfer)"],
                 [A_VIS, A_COCO], 0.45, color=["#2e7d32", "#d9534f"])
for r, v, v50 in zip(bars, [A_VIS, A_COCO], [A_VIS50, A_COCO50]):
    ax[1].text(r.get_x() + r.get_width() / 2, v + 0.008, f"mAP50-95 {v:.4f}", ha="center",
               fontsize=9.5, fontweight="bold")
    ax[1].text(r.get_x() + r.get_width() / 2, v + 0.045, f"mAP50 {v50:.4f}", ha="center", fontsize=8.5)
ax[1].annotate("", xy=(1, A_COCO), xytext=(0, A_VIS),
               arrowprops=dict(arrowstyle="->", color="gray", lw=1.4, ls="--"))
ax[1].text(0.5, 0.20, "domain + scale + semantics gap\n"
                      "851,345 boxes on 5,000 images (170/img)",
           ha="center", fontsize=9, color="#555")
ax[1].set_ylim(0, 0.42); ax[1].set_ylabel("mAP50-95 (COCO protocol)")
ax[1].set_title("(b) Detector cross-domain transfer is weak\n"
                "(reported as an honest lower bound, not a generalization claim)", fontsize=10.5)
ax[1].grid(axis="y", alpha=0.3)

fig.suptitle("Cross-dataset generalization on COCO val2017 "
             "(5,000 images, COCO protocol, faster-coco-eval)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = FIG / "fig_coco_generalization.png"
fig.savefig(out, dpi=170); plt.close(fig)
print(f"✅ {out}")
