#!/usr/bin/env python
"""P1 补充消融（纯 CPU，复用 fusion_cache_full.npz）：

  A. 尺寸门控阈值 big_px 的全量 val 敏感性扫描
     —— 原报告的寻优只在 120 图子集上做过（big_px ∈ {16,32,48}）。
  B. MV-Fuse 的权重敏感性 / "零参数"核查
     —— 原报告的 W=[0.3,0.6,1.0,1.5,2.0,2.5] 为手工设定未调参；本脚本扫描多组 W。

用法: python probe_gate_weight_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import (DS, build_gt, evaluate, iou_matrix, m_nms,  # noqa: E402
                                    m_softnms, to_preds, union_pool)

ROOT = Path("/home/wang/DeepSeek/YOLO")
OVERLAP = 0.3


def gated_pool_px(bf, sf, cf, bt, st, ct, big_px):
    if len(bt):
        tsz = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
        kt = tsz < big_px
    else:
        kt = np.zeros(0, bool)
    if len(bf) == 0 and not kt.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return (np.concatenate([bf, bt[kt]]), np.concatenate([sf, st[kt]]),
            np.concatenate([cf, ct[kt]]))


def nviews(pool_b, pool_c, views):
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


def main():
    cache_path = ROOT / "fusion_cache_full.npz"
    z = np.load(cache_path, allow_pickle=True)
    data = list(z["cache"])
    files = [l.strip() for l in open(DS / "val.txt")]
    print(f"缓存 {cache_path}  图数={len(data)}", flush=True)
    coco_gt = build_gt(files)
    res = {}

    def run(tag, fn):
        preds = []
        for i, it in enumerate(data):
            b, s, c = fn(it)
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4)}
        print(f"  {tag:<34} AP50-95={p95:.4f}  AP50={p50:.4f}", flush=True)
        return p95

    print("\n=== A. 尺寸门控阈值 big_px（全量 548 图，4 前向）===", flush=True)
    run("union（无门控）+ NMS", lambda it: m_nms(*union_pool(*it[:6]), 0.6))
    for px in (8, 16, 24, 32, 48, 64, 96, 128, 10 ** 9):
        tag = f"gated(big_px={px}) + NMS" if px < 10 ** 9 else "gated(全部保留=并集) + NMS"
        run(tag, lambda it, px=px: m_nms(*gated_pool_px(*it[:6], px), 0.6))
    for px in (24, 32, 48, 64, 96):
        run(f"gated(big_px={px}) + Soft-NMS",
            lambda it, px=px: m_softnms(*gated_pool_px(*it[:6], px), 0.6, 0.5))

    print("\n=== B. MV-Fuse 权重敏感性（gated big_px=48 + NMS 融合）===", flush=True)
    Wsets = {
        "W=[1,1,1,1,1,1]（不重打分）": [1, 1, 1, 1, 1, 1],
        "W=原设定[0.3,0.6,1.0,1.5,2.0,2.5]": [0.3, 0.6, 1.0, 1.5, 2.0, 2.5],
        "W=线性[1,2,3,4,5,6]": [1, 2, 3, 4, 5, 6],
        "W=平方[1,4,9,16,25,36]": [1, 4, 9, 16, 25, 36],
        "W=仅二值（支持=2 才 ×1.5）": None,
        "W=反向[2.5,2.0,1.5,1.0,0.6,0.3]": [2.5, 2.0, 1.5, 1.0, 0.6, 0.3],
    }

    def mv(it, W):
        gb, gs, gc = gated_pool_px(*it[:6], 48.0)
        views = ((it[0], it[1], it[2]), (it[3], it[4], it[5]))
        gb, gs, gc = m_nms(gb, gs, gc, 0.6)
        nv = nviews(gb, gc, views)
        if W is None:
            w = np.where(nv >= 2, 1.5, 1.0)
        else:
            w = np.array([W[min(int(x), len(W) - 1)] for x in nv])
        return gb, gs * w, gc

    for tag, W in Wsets.items():
        run(tag, lambda it, W=W: mv(it, W))
        # 记录支持度分布（只在第一次）
    nv_all = []
    for it in data[:200]:
        gb, gs, gc = gated_pool_px(*it[:6], 48.0)
        gb, gs, gc = m_nms(gb, gs, gc, 0.6)
        nv_all.append(nviews(gb, gc, ((it[0], it[1], it[2]), (it[3], it[4], it[5]))))
    nv = np.concatenate(nv_all)
    dist = {int(k): round(float(v) * 100, 1) for k, v in
            zip(*np.unique(nv, return_counts=True))}
    res["_support_dist_200img_pct"] = dist
    print(f"\n  支持视角数分布（前 200 图，%）: {dist}", flush=True)

    json.dump(res, open(ROOT / "gate_weight_sensitivity.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n已写入 {ROOT/'gate_weight_sensitivity.json'}")


if __name__ == "__main__":
    main()
