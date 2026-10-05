#!/usr/bin/env python
"""MV-Fuse：多视角一致性重打分 —— 视角子集搜索 + 学习化打分器（留出法验证）。

已实测的机制：
  - 候选被越多视角支持，TP 率越高（1 视角 0.4% → 4 视角 8.5%）
  - 分离度随目标变小而变强（0–8px: TP 96.5% vs FP 72.8%，+23.7 点）
  - 手工强共识权重把四视角并集从 0.2726 提到 0.2789（超过 per-image oracle 0.2768）

本脚本：
  A. 视角子集扫描（算力 vs 精度）
  B. 手工权重方案
  C. 学习化打分器（逻辑回归，特征 = 分数/支持数/尺寸/最大一致 IoU），**在留出的一半图上评测**
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_oracle_policy import _np_iou, load_gt_xyxy, name, to_coco  # noqa: E402
from probe_slicing import DS, build_gt, evaluate, merge_nms  # noqa: E402

ALLV = [("p2_960", 960, "full"), ("p2_960", 960, "2x1"), ("p2_960", 960, "3x1"), ("p2_960", 960, "2x2")]
NFWD = {"full": 1, "2x1": 3, "3x1": 4, "2x2": 5}


def build_pool(P, gts, i, nsys):
    """返回 (boxes, scores, classes, nviews, maxiouw, is_tp, size)。"""
    allb = np.concatenate([P[v][i][0] for v in nsys]) if nsys else np.zeros((0, 4))
    alls = np.concatenate([P[v][i][1] for v in nsys]) if nsys else np.zeros(0)
    allc = np.concatenate([P[v][i][2] for v in nsys]) if nsys else np.zeros(0, int)
    if len(alls) == 0:
        return None
    b, s, c = merge_nms(allb, alls, allc, iou_thr=0.6)
    if len(s) == 0:
        return None
    nv = np.zeros(len(s), int)
    mx = np.zeros(len(s))
    for v in nsys:
        vb, vs, vc = P[v][i]
        m = vs >= 0.001
        vb2, vc2 = vb[m], vc[m]
        if len(vb2) == 0:
            continue
        iou = _np_iou(b, vb2)
        for j in range(len(b)):
            same = vc2 == c[j]
            if same.any():
                best = iou[j][same].max()
                if best >= 0.5:
                    nv[j] += 1
                    mx[j] = max(mx[j], best)
    gb, gc, _, _ = gts[i]
    sz = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))
    tp = np.zeros(len(s), bool)
    if len(gb):
        iou = _np_iou(gb, b)
        for j in range(len(s)):
            same = gc == c[j]
            tp[j] = bool(same.any() and iou[same, j].max() >= 0.5)
    return b, s, c, nv, mx, tp, sz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/wang/DeepSeek/YOLO/oracle_policy_o3/n548")
    ap.add_argument("--stage", default="pool", choices=["pool", "report"])
    a = ap.parse_args()
    CACHE = Path(a.cache)
    files = [l.strip() for l in open(DS / "val.txt")]
    gts = [load_gt_xyxy(l) for l in files]
    coco = build_gt(files)
    P = {v: pickle.load(open(CACHE / f"{name(v).replace('@','_').replace('/','_')}.pkl", "rb")) for v in ALLV}

    SUBSETS = [("full", [0]), ("full+2x1", [0, 1]), ("full+3x1", [0, 2]),
               ("full+2x1+3x1", [0, 1, 2]), ("full+2x1+3x1+2x2", [0, 1, 2, 3])]
    pool_file = CACHE / "mvfuse_pools.pkl"
    if a.stage == "pool":
        pools = {}
        for sname, idxs in SUBSETS:
            nsys = [ALLV[k] for k in idxs]
            rec = []
            for i in range(len(files)):
                r = build_pool(P, gts, i, nsys)
                if r is not None:
                    rec.append(r)
                if i % 150 == 0:
                    print(f"  [{sname}] {i}/{len(files)}", flush=True)
            pools[sname] = rec
            print(f"  ✅ {sname}: {sum(len(r[0]) for r in rec)} 候选")
        pickle.dump(pools, open(pool_file, "wb"))
        print("  ✅ 候选池缓存完成")
        return

    pools = pickle.load(open(pool_file, "rb"))
    n = len(files)
    rng = np.random.RandomState(0)
    perm = rng.permutation(n)
    fit_imgs = set(perm[: n // 2].tolist())
    hold_imgs = sorted(set(range(n)) - fit_imgs)

    def feats_of(rec):
        F, Y, imgof = [], [], []
        for i, r in enumerate(rec):
            b, s, c, nv, mx, tp, sz = r
            for j in range(len(s)):
                F.append([np.log(s[j] + 1e-6), nv[j], np.log(sz[j] + 1e-3), mx[j]])
                Y.append(1.0 if tp[j] else 0.0)
                imgof.append(i)
        return np.array(F), np.array(Y), np.array(imgof)

    def eval_half(build):
        preds = [None] * n
        for i, r in enumerate(rec):
            if i not in set(hold_imgs):
                continue
            b, s, c, nv, mx, tp, sz = r
            preds[i] = (b, build(s, c, nv, mx, sz), c)
        gt_sub = _restrict(coco, hold_imgs)
        return evaluate(gt_sub, to_coco([x if x is not None else (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for x in preds]))

    W = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
    print(f"\n=== 视角子集 × 打分方案（全部在**留出半集**上评测，274 图）===")
    print(f"{'视角子集':>18}{'前向数':>7}{'原分数':>16}{'强共识':>16}{'学习器(留出)':>16}")
    rows_out = []
    for sname, idxs in SUBSETS:
        rec = pools[sname]
        F, Y, imgof = feats_of(rec)
        nvf = max(len(idxs), 1)
        F[:, 1] = F[:, 1] / nvf
        isfit = np.array([im in fit_imgs for im in imgof])
        mu, sd = F[isfit].mean(0), F[isfit].std(0) + 1e-9
        Xf = np.concatenate([np.ones((len(F), 1)), (F - mu) / sd], 1)
        w = np.zeros(Xf.shape[1])
        for _ in range(300):
            pr = 1 / (1 + np.exp(-np.clip(Xf[isfit] @ w, -30, 30)))
            w -= 2.0 * (Xf[isfit].T @ (pr - Y[isfit]) / max(isfit.sum(), 1) + 1e-4 * w)
        def sc_orig(s, c, nv, mx, sz): return s
        def sc_cons(s, c, nv, mx, sz):
            return s * np.array([W[min(int(k), len(W) - 1)] for k in nv])
        def sc_lr(s, c, nv, mx, sz):
            Fc = np.stack([np.log(s + 1e-6), nv / nvf, np.log(sz + 1e-3), mx], 1)
            logit = np.concatenate([np.ones((len(Fc), 1)), (Fc - mu) / sd], 1) @ w
            p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
            return s * p
        o = eval_half(sc_orig); cc = eval_half(sc_cons); ll = eval_half(sc_lr)
        rows_out.append((sname, sum(NFWD[ALLV[k][2]] for k in idxs), o, cc, ll))
        print(f"{sname:>18}{sum(NFWD[ALLV[k][2]] for k in idxs):>7}{o[0]:>9.4f}/{o[1]:<6.4f}{cc[0]:>9.4f}/{cc[1]:<6.4f}{ll[0]:>9.4f}/{ll[1]:<6.4f}")
    best = max(rows_out, key=lambda r: max(r[2][0], r[3][0], r[4][0]))
    print(f"\n最佳单点: {best[0]} (前向 {best[1]})  原分数 {best[2][0]:.4f} / 强共识 {best[3][0]:.4f} / 学习器 {best[4][0]:.4f}")
    print("注：以上均为留出半集（274 图）结果，三者同集可比。全量集参考：960整图 0.2512；3x1尺寸门控 0.2750；4视角并集+强共识 0.2789")


def _restrict(coco, idxs):
    keep = set(idxs)
    imgs = [im for im in coco["images"] if im["id"] in keep]
    anns = [x for x in coco["annotations"] if x["image_id"] in keep]
    return {**coco, "images": imgs, "annotations": anns}


if __name__ == "__main__":
    main()
