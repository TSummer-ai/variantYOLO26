#!/usr/bin/env python
"""任务4：融合方式强基线对照（同前向数）。

问题：尺寸感知切片融合的 +2.38 AP，究竟来自"按尺寸门控信息源"，
      还是只是"换了个更强的框合并器"？本脚本在同一候选池上对比 5 种合并器：

  NMS           通用贪心 NMS（SAHI 默认）
  Soft-NMS      Gaussian 衰减（原论文 sigma=0.5）
  DIoU-NMS      抑制判据换成 DIoU（同样的候选，更弱抑制）
  Cluster-DIoU  先按 DIoU 聚类再加权平均（ASAHI 的合并思路）
  WBF           加权框融合（weighted box fusion）

两条候选池：union（整图+切片直接并集）与 gated（>=48px 只信整图）。
外加本项目的尺寸门控 + 跨视角一致性重打分作为参考臂。

缓存：整图@960、3x1@960、2x2@960 的逐图预测存到 npz，之后重跑不再需要 GPU。

用法:
  python probe_fusion_baselines.py --weights runs/visdrone/p2_960/weights/best.pt --limit 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))
BIG_PX = 48.0
OVERLAP = 0.3
W = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]


# --------------------------------------------------------------------------
# 基础几何
# --------------------------------------------------------------------------
def iou_matrix(a, b):
    """a: (N,4), b: (M,4) -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    ar = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    br = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    return inter / (ar[:, None] + br[None, :] - inter + 1e-9)


def diou_matrix(a, b):
    """DIoU = IoU - d^2 / c^2（中心距平方 / 最小闭包对角线平方）"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iou = iou_matrix(a, b)
    ca = (a[:, :2] + a[:, 2:]) / 2.0
    cb = (b[:, :2] + b[:, 2:]) / 2.0
    d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)
    lt = np.minimum(a[:, None, :2], b[None, :, :2])
    rb = np.maximum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    c2 = wh[..., 0] ** 2 + wh[..., 1] ** 2 + 1e-9
    return iou - d2 / c2


# --------------------------------------------------------------------------
# 合并器（全部按类别分别处理）
# --------------------------------------------------------------------------
def _by_class(b, s, c):
    out = {}
    for cl in np.unique(c):
        m = c == cl
        out[int(cl)] = (b[m], s[m])
    return out


def m_nms(b, s, c, thr=0.6):
    return merge_nms(b, s, c, thr)


def m_greedy(b, s, c, thr, sim="iou"):
    """通用贪心抑制：sim 为 'iou' 或 'diou'。"""
    if len(b) == 0:
        return b, s, c
    keep_b, keep_s, keep_c = [], [], []
    for cl, (bb, ss) in _by_class(b, s, c).items():
        order = np.argsort(-ss)
        bb, ss = bb[order], ss[order]
        sim_fn = iou_matrix if sim == "iou" else diou_matrix
        M = sim_fn(bb, bb)
        alive = np.ones(len(bb), bool)
        for i in range(len(bb)):
            if not alive[i]:
                continue
            alive[i + 1:] &= ~(M[i, i + 1:] > thr)
        keep_b.append(bb[alive]); keep_s.append(ss[alive])
        keep_c.append(np.full(int(alive.sum()), cl, int))
    return (np.concatenate(keep_b), np.concatenate(keep_s), np.concatenate(keep_c))


def m_softnms(b, s, c, thr=0.6, sigma=0.5):
    """Gaussian Soft-NMS（Bodla et al.）。"""
    if len(b) == 0:
        return b, s, c
    kb, ks, kc = [], [], []
    for cl, (bb, ss) in _by_class(b, s, c).items():
        bb = bb.copy(); ss = ss.copy()
        M = iou_matrix(bb, bb)
        idx = np.arange(len(bb))
        out_b, out_s = [], []
        while len(idx):
            j = int(np.argmax(ss[idx]))
            gi = idx[j]
            out_b.append(bb[gi]); out_s.append(ss[gi])
            idx = np.delete(idx, j)
            if len(idx) == 0:
                break
            ov = M[gi, idx]
            ss[idx] = ss[idx] * np.exp(-(ov ** 2) / sigma)
        kb.append(np.array(out_b)); ks.append(np.array(out_s))
        kc.append(np.full(len(out_b), cl, int))
    return (np.concatenate(kb), np.concatenate(ks), np.concatenate(kc))


def m_cluster_diou(b, s, c, thr=0.6):
    """Cluster-DIoU：按 DIoU>thr 做连通聚类，簇内按分数加权平均坐标（ASAHI 式合并）。"""
    if len(b) == 0:
        return b, s, c
    fb, fs, fc = [], [], []
    for cl, (bb, ss) in _by_class(b, s, c).items():
        n = len(bb)
        if n == 0:
            continue
        adj = (diou_matrix(bb, bb) > thr)
        adj = np.triu(adj | adj.T, 1)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in np.argwhere(adj):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        for _, g in groups.items():
            g = np.array(g)
            w = ss[g]
            fb.append((bb[g] * w[:, None]).sum(0) / w.sum())
            fs.append(float(w.max()))
            fc.append(cl)
    return (np.array(fb).reshape(-1, 4), np.array(fs), np.array(fc, int))


def m_wbf(b, s, c, iou_thr=0.6, skip_thr=0.0, n_models=2):
    """Weighted Boxes Fusion（Solovyev et al.），单模型多视角版本。

    坐标 = Σ(score_i * box_i)/Σ(score_i)；分数 = Σ(score_i^2)/Σ(score_i) * min(N,n)/n_models。
    """
    if len(b) == 0:
        return b, s, c
    fb, fs, fc = [], [], []
    for cl, (bb, ss) in _by_class(b, s, c).items():
        order = np.argsort(-ss)
        bb, ss = bb[order], ss[order]
        clusters = []          # list of [ [boxes], [scores] ]
        reps = np.zeros((0, 4))
        for i in range(len(bb)):
            if ss[i] < skip_thr:
                continue
            if len(reps):
                ious = iou_matrix(bb[i:i + 1], reps)[0]
                k = int(np.argmax(ious))
                if ious[k] > iou_thr:
                    clusters[k][0].append(bb[i]); clusters[k][1].append(ss[i])
                    continue
            clusters.append([[bb[i]], [ss[i]]])
            reps = np.concatenate([reps, bb[i:i + 1]])
        for clu in clusters:
            B = np.array(clu[0]); S = np.array(clu[1])
            nb = (B * S[:, None]).sum(0) / S.sum()
            sc = float((S * S).sum() / S.sum()) * min(len(S), n_models) / n_models
            fb.append(nb); fs.append(sc); fc.append(cl)
    return (np.array(fb).reshape(-1, 4), np.array(fs), np.array(fc, int))


# --------------------------------------------------------------------------
# 候选池构建
# --------------------------------------------------------------------------
def gated_pool(bf, sf, cf, bt, st, ct, big_px=BIG_PX):
    if len(bt):
        tsz = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
        kt = tsz < big_px
    else:
        kt = np.zeros(0, bool)
    if len(bf) == 0 and not kt.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    b = np.concatenate([bf, bt[kt]])
    s = np.concatenate([sf, st[kt]])
    c = np.concatenate([cf, ct[kt]])
    return b, s, c


def union_pool(bf, sf, cf, bt, st, ct):
    if len(bf) == 0:
        return bt, st, ct
    if len(bt) == 0:
        return bf, sf, cf
    return np.concatenate([bf, bt]), np.concatenate([sf, st]), np.concatenate([cf, ct])


def nviews(pool_b, pool_c, views):
    """每个候选被几个视角支持（同类且 IoU>=0.5）。"""
    nv = np.zeros(len(pool_b), int)
    for vb, vs, vc in views:
        if len(vb) == 0:
            continue
        m = vs >= 0.001
        vb2, vc2 = vb[m], vc[m]
        if len(vb2) == 0:
            continue
        iou = iou_matrix(pool_b, vb2)
        for j in range(len(pool_b)):
            same = vc2 == pool_c[j]
            if same.any() and iou[j][same].max() >= 0.5:
                nv[j] += 1
    return nv


# --------------------------------------------------------------------------
def to_preds(b, s, c, i, out):
    for j in range(len(s)):
        out.append({"image_id": i, "category_id": int(c[j]) + 1,
                    "bbox": [float(b[j, 0]), float(b[j, 1]),
                             float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                    "score": float(s[j])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_960/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=str(ROOT / "fusion_cache_3x1_2x2.npz"))
    ap.add_argument("--out", default=str(ROOT / "fusion_baselines.json"))
    a = ap.parse_args()

    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    n = len(files)
    print(f"val 图数 = {n}", flush=True)
    coco_gt = build_gt(files)

    cache_path = Path(a.cache)
    if cache_path.exists():
        print(f"复用缓存 {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        cache = list(z["cache"])
    else:
        from ultralytics import YOLO
        model = YOLO(a.weights)
        cache = []
        t0 = time.perf_counter()
        for i, line in enumerate(files):
            ip = DS / line[2:]
            fb, fs, fc = predict_full(model, ip, a.imgsz, 0.001, 0.7, a.device)
            t1b, t1s, t1c = predict_sliced(model, ip, a.imgsz, 0.001, 0.7, a.device, (3, 1), OVERLAP)
            t2b, t2s, t2c = predict_sliced(model, ip, a.imgsz, 0.001, 0.7, a.device, 2, OVERLAP)
            # 视角内先做跨块 NMS，与 probe_mvfuse.py 保持一致
            t1b, t1s, t1c = merge_nms(t1b, t1s, t1c, 0.6)
            t2b, t2s, t2c = merge_nms(t2b, t2s, t2c, 0.6)
            cache.append((fb, fs, fc, t1b, t1s, t1c, t2b, t2s, t2c))
            if i % 50 == 0:
                print(f"  cache {i}/{n}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        np.savez_compressed(cache_path, cache=np.array(cache, dtype=object))
        print(f"缓存已写入 {cache_path}", flush=True)

    results = {}

    def run(tag, fn):
        preds = []
        for i, item in enumerate(cache):
            b, s, c = fn(item)
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        results[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds)}
        print(f"  {tag:<34} AP50-95={p95:.4f}  AP50={p50:.4f}  ({len(preds)} 框)", flush=True)

    fb0, fs0, fc0, t1b, t1s, t1c, t2b, t2s, t2c = cache[0]

    print("\n=== 参考臂 ===", flush=True)
    run("full@960 (1 前向)", lambda it: (it[0], it[1], it[2]))
    run("tiles 3x1 + NMS (3 前向)", lambda it: m_nms(it[3], it[4], it[5], 0.6))
    run("tiles 2x2 + NMS (4 前向, SAHI 式)", lambda it: m_nms(it[6], it[7], it[8], 0.6))

    print("\n=== union 候选池：只换合并器（4 前向）===", flush=True)
    u = lambda it: union_pool(*it[:6])  # noqa: E731
    run("union + NMS", lambda it: m_nms(*u(it), 0.6))
    run("union + Soft-NMS(gauss .5)", lambda it: m_softnms(*u(it), 0.6, 0.5))
    run("union + DIoU-NMS", lambda it: m_greedy(*u(it), 0.6, "diou"))
    run("union + Cluster-DIoU", lambda it: m_cluster_diou(*u(it), 0.6))
    run("union + WBF", lambda it: m_wbf(*u(it), 0.6))

    print("\n=== gated 候选池（尺寸门控，>=48px 只信整图）：只换合并器（4 前向）===", flush=True)
    g = lambda it: gated_pool(*it[:6])  # noqa: E731
    run("gated + NMS（此前方法）", lambda it: m_nms(*g(it), 0.6))
    run("gated + Soft-NMS(gauss .5)", lambda it: m_softnms(*g(it), 0.6, 0.5))
    run("gated + DIoU-NMS", lambda it: m_greedy(*g(it), 0.6, "diou"))
    run("gated + Cluster-DIoU", lambda it: m_cluster_diou(*g(it), 0.6))
    run("gated + WBF", lambda it: m_wbf(*g(it), 0.6))

    print("\n=== 本方法：尺寸门控 + 跨视角一致性重打分 ===", flush=True)

    def ours(it, merge="nms"):
        gb, gs, gc = g(it)
        views = ((it[0], it[1], it[2]), (it[3], it[4], it[5]))
        if merge == "nms":
            gb, gs, gc = m_nms(gb, gs, gc, 0.6)
        else:
            gb, gs, gc = m_wbf(gb, gs, gc, 0.6)
        nv = nviews(gb, gc, views)
        w = np.array([W[min(int(x), len(W) - 1)] for x in nv])
        return gb, gs * w, gc

    run("gated + NMS + 一致性（MV-Fuse）", lambda it: ours(it, "nms"))
    run("gated + WBF + 一致性", lambda it: ours(it, "wbf"))

    results["_meta"] = {"weights": a.weights, "imgsz": a.imgsz, "n_img": n,
                        "note": "视角内均先做跨块 NMS(0.6)；union/gated 为候选池定义"}
    json.dump(results, open(a.out, "w"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 74)
    base = results["full@960 (1 前向)"]["ap50_95"]
    gate = results["gated + NMS（此前方法）"]["ap50_95"]
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"{k:<36}{v['ap50_95']:>9.4f}  (vs 整图 {v['ap50_95']-base:+.4f})")
    print("=" * 74)
    print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
