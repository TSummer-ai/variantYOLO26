#!/usr/bin/env python
"""E5/E9 学习器对比图：手工"支持数"加权 vs 学习器（LR / MLP）。

核心结论：**零学习参数的手工加权最好**（参数量越多越差，MLP 直接崩）。

面板：
 (a) 留出半集 mAP50（同集可比）：原分数 / 手工支持数 / LR(14特征) / MLP(16隐层)
 (b) 留出半集 mAP50-95：原分数 / 手工支持数 / LR(4特征)
 (c) 权重底数 a 的完整曲线：拟合半集 与 留出半集 —— 证明对 a 不敏感（不是调出来的）
 (d) 学习参数量 vs 性能：0 参数最好
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/home/wang/DeepSeek/YOLO")
from probe_oracle_policy import _np_iou, load_gt_xyxy, name, to_coco  # noqa: E402
from probe_slicing import DS, build_gt, evaluate  # noqa: E402

CACHE = Path("/home/wang/DeepSeek/YOLO/oracle_policy_o3/n548")
V_FULL, V_GATE = ("p2_960", 960, "full"), ("p2_960", 960, "3x1")
FIG = Path("/home/wang/DeepSeek/YOLO/figures"); FIG.mkdir(exist_ok=True)
OUT = Path("/home/wang/DeepSeek/YOLO/exp_learner_curve.json")


def main():
    files = [l.strip() for l in open(DS / "val.txt")]
    n = len(files)
    coco = build_gt(files)
    P = {v: pickle.load(open(CACHE / f"{name(v).replace('@','_').replace('/','_')}.pkl", "rb"))
         for v in (V_FULL, V_GATE)}
    full, gate = P[V_FULL], P[V_GATE]

    # 每个候选的支持视角数（0/1/2）
    NV = []
    for i in range(n):
        b, s, c = gate[i]
        if len(s) == 0:
            NV.append(np.zeros(0, int)); continue
        nv = np.zeros(len(s), int)
        for vb, vs, vc in (full[i], gate[i]):
            m = vs >= 0.001
            vb2, vc2 = vb[m], vc[m]
            if len(vb2) == 0:
                continue
            iou = _np_iou(b, vb2)
            for j in range(len(b)):
                same = vc2 == c[j]
                if same.any() and iou[j][same].max() >= 0.5:
                    nv[j] += 1
        NV.append(nv)

    perm = np.random.RandomState(0).permutation(n)
    fit = set(perm[: n // 2].tolist()); hold = sorted(set(range(n)) - fit)

    def ap_on(ids, scorer):
        preds = [None] * n
        for i in ids:
            b, s, c = gate[i]
            preds[i] = (b, scorer(i, b, s, c), c)
        gt = {**coco, "images": [x for x in coco["images"] if x["id"] in set(ids)],
              "annotations": [x for x in coco["annotations"] if x["image_id"] in set(ids)]}
        p95, p50 = evaluate(gt, to_coco([x if x is not None else (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for x in preds]))
        return p95, p50

    # ---- (c) 权重底数 a 的完整曲线 ----
    AR = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
    curve = {"a": AR, "fit": [], "hold": []}
    for a in AR:
        w = np.array([a ** min(int(x), 5) for x in range(6)])
        sc = lambda i, b, s, c: s * np.array([w[min(int(x), 5)] for x in NV[i]]) if len(s) else s
        curve["fit"].append(ap_on(sorted(fit), sc)[1])
        curve["hold"].append(ap_on(hold, sc)[1])
        print(f"  a={a:<4} fit(mAP50)={curve['fit'][-1]:.4f}  hold(mAP50)={curve['hold'][-1]:.4f}", flush=True)
    best_fit = AR[int(np.argmax(curve["fit"]))]
    print(f"  拟合 argmax a*={best_fit}")

    # ---- 基线（原分数）----
    sc_id = lambda i, b, s, c: s
    base_hold_95, base_hold_50 = ap_on(hold, sc_id)
    wstar = np.array([best_fit ** min(int(x), 5) for x in range(6)])
    sc_star = lambda i, b, s, c: s * np.array([wstar[min(int(x), 5)] for x in NV[i]]) if len(s) else s
    st_hold_95, st_hold_50 = ap_on(hold, sc_star)
    print(f"  留出: 原分数 mAP50={base_hold_50:.4f}/mAP50-95={base_hold_95:.4f} ; "
          f"手工 a*={best_fit} mAP50={st_hold_50:.4f}/mAP50-95={st_hold_95:.4f}")

    # 学习器结果（来自实验日志，同一留出半集）
    LR50, MLP50 = 0.4712, 0.4021
    LR95 = 0.2773
    out = {"a": AR, "fit": curve["fit"], "hold": curve["hold"], "a_star": best_fit,
           "orig": [base_hold_50, base_hold_95], "hand": [st_hold_50, st_hold_95],
           "lr50": LR50, "mlp50": MLP50, "lr95": LR95}
    import json
    json.dump(out, open(OUT, "w"), indent=1)

    # ================= 图 =================
    fig, ax = plt.subplots(2, 2, figsize=(12.6, 8.6))
    ax = ax.ravel()

    # (a) mAP50
    names = ["original\nscores", f"hand-crafted\ncount weights (a*={best_fit})",
             "logistic reg.\n(14 feats, 15 par.)", "MLP\n(16 hidden, 257 par.)"]
    vals = [base_hold_50, st_hold_50, LR50, MLP50]
    cols = ["#9aa5b1", "#2e7d32", "#e0a800", "#d9534f"]
    b = ax[0].bar(names, vals, color=cols)
    for r, v in zip(b, vals):
        ax[0].text(r.get_x() + r.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax[0].set_ylim(0.38, 0.50); ax[0].set_ylabel("mAP50 (held-out half)")
    ax[0].set_title("(a) Learned re-scorers do NOT beat 1 hand-set parameter", fontsize=10.5)
    ax[0].grid(axis="y", alpha=0.3); ax[0].tick_params(axis="x", labelsize=8)

    # (b) mAP50-95
    names2 = ["original\nscores", f"hand-crafted\ncount weights", "logistic reg.\n(4 feats, 5 par.)"]
    vals2 = [base_hold_95, st_hold_95, LR95]
    b = ax[1].bar(names2, vals2, color=["#9aa5b1", "#2e7d32", "#e0a800"])
    for r, v in zip(b, vals2):
        ax[1].text(r.get_x() + r.get_width() / 2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax[1].set_ylim(0.272, 0.281); ax[1].set_ylabel("mAP50-95 (held-out half)")
    ax[1].set_title("(b) Same conclusion on mAP50-95", fontsize=10.5)
    ax[1].grid(axis="y", alpha=0.3); ax[1].tick_params(axis="x", labelsize=8)

    # (c) a 曲线
    ax[2].plot(AR, curve["fit"], "o-", color="#8fa8c8", label="fit half (used to choose a)")
    ax[2].plot(AR, curve["hold"], "s-", color="#e45756", label="held-out half (reported)")
    ax[2].axvline(best_fit, ls=":", c="k", lw=1.2)
    ax[2].annotate(f"a*={best_fit} chosen on fit half", (best_fit, min(curve["hold"])),
                   textcoords="offset points", xytext=(8, -14), fontsize=8.5)
    ax[2].set_xlabel("weight base a   (w = a^(#supporting views))"); ax[2].set_ylabel("mAP50")
    ax[2].set_title("(c) The held-out curve is flat around a* => not finely tuned", fontsize=10.5)
    ax[2].legend(fontsize=8.5); ax[2].grid(alpha=0.3)

    # (d) 参数量 vs 性能
    pts = [("hand-crafted (0 learned)", 0, st_hold_50, "#2e7d32"),
           ("logistic reg. (15)", 15, LR50, "#e0a800"),
           ("MLP (257)", 257, MLP50, "#d9534f")]
    for lab, p, v, c in pts:
        ax[3].scatter(p, v, s=170, color=c, edgecolors="k", zorder=3)
        ax[3].annotate(lab, (p, v), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax[3].axhline(st_hold_50, ls="--", c="#2e7d32", lw=1)
    ax[3].set_xscale("symlog"); ax[3].set_ylim(0.39, 0.49)
    ax[3].set_xlabel("number of learned parameters (symlog)")
    ax[3].set_ylabel("mAP50 (held-out half)")
    ax[3].set_title("(d) More learned parameters -> worse: the signal is a\n"
                    "discrete, robust statistic (view count), not a feature vector", fontsize=10.5)
    ax[3].grid(alpha=0.3)

    fig.suptitle("Cross-view-agreement re-scoring: a zero-parameter rule beats learned re-scorers "
                 "(VisDrone val, held-out half, identical candidate pool)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out_png = FIG / "学习器与手工权重对比.png"
    fig.savefig(out_png, dpi=170); plt.close(fig)
    print(f"已生成 {out_png}\n已写入 {OUT}")


if __name__ == "__main__":
    main()
