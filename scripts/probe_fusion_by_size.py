#!/usr/bin/env python
"""任务4c：各融合方案的分尺寸召回（纯 CPU，复用 fusion_cache_full.npz）。

目的：任务4 发现 union+WBF(0.2826) > MV-Fuse(0.2788) > gated+NMS(0.2750)，
      但"为什么"未知。本脚本按目标尺寸拆开看召回，回答两个问题：
        Q1 WBF 的增益是集中在极小目标，还是全面的？
        Q2 尺寸门控为何在 WBF 上变成 −2.0 AP？

用法: python probe_fusion_by_size.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import (DS, build_gt, evaluate, gated_pool, m_nms,  # noqa: E402
                                    m_softnms, to_preds, union_pool, nviews, W)
from probe_slicing import BUCKETS, recall_by_bucket  # noqa: E402
from probe_wbf_mechanism import wbf_variant  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")


def main():
    z = np.load(ROOT / "fusion_cache_full.npz", allow_pickle=True)
    data = list(z["cache"])
    files = [l.strip() for l in open(DS / "val.txt")]
    print(f"缓存图数 = {len(data)}", flush=True)
    coco_gt = build_gt(files)

    variants = {
        "整图@960": lambda it: (it[0], it[1], it[2]),
        "仅切片3x1+NMS": lambda it: m_nms(it[3], it[4], it[5], 0.6),
        "union+NMS": lambda it: m_nms(*union_pool(*it[:6]), 0.6),
        "union+WBF(原式)": lambda it: wbf_variant(*union_pool(*it[:6]), 0.6, "orig"),
        "union+WBF(max)": lambda it: wbf_variant(*union_pool(*it[:6]), 0.6, "max"),
        "gated+NMS": lambda it: m_nms(*gated_pool(*it[:6]), 0.6),
        "gated+SoftNMS": lambda it: m_softnms(*gated_pool(*it[:6]), 0.6, 0.5),
        "gated+WBF(原式)": lambda it: wbf_variant(*gated_pool(*it[:6]), 0.6, "orig"),
        "MV-Fuse(门控+一致性)": None,
    }

    def mv(it):
        gb, gs, gc = gated_pool(*it[:6])
        views = ((it[0], it[1], it[2]), (it[3], it[4], it[5]))
        gb, gs, gc = m_nms(gb, gs, gc, 0.6)
        nv = nviews(gb, gc, views)
        w = np.array([W[min(int(x), len(W) - 1)] for x in nv])
        return gb, gs * w, gc

    variants["MV-Fuse(门控+一致性)"] = mv

    res = {}
    per_image_all = {}
    for tag, fn in variants.items():
        preds, per_image = [], []
        for i, it in enumerate(data):
            b, s, c = fn(it)
            per_image.append((b, s, c))
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds)}
        per_image_all[tag] = per_image
        print(f"  {tag:<24} AP50-95={p95:.4f}  boxes={len(preds)}", flush=True)

    print("\n分尺寸召回（conf=0.001, IoU=0.5）...", flush=True)
    rec = {t: recall_by_bucket(files, per_image_all[t]) for t in variants}
    names = [b[0] for b in BUCKETS]
    tbl = {}
    for nm in names:
        tbl[nm] = {"gt": rec[list(variants)[0]][nm][0],
                   "recall": {t: round(rec[t][nm][1] / rec[t][nm][0], 4) if rec[t][nm][0] else None
                              for t in variants}}
    res["_by_size"] = tbl

    hdr = "".join(f"{t:>22}" for t in variants)
    print(f"\n{'尺寸':>8}{'GT数':>8}{hdr}")
    for nm in names:
        if not tbl[nm]["gt"]:
            continue
        vals = "".join(f"{tbl[nm]['recall'][t]:>22.4f}" for t in variants)
        print(f"{nm:>8}{tbl[nm]['gt']:>8}{vals}")

    json.dump(res, open(ROOT / "fusion_by_size.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n已写入 {ROOT/'fusion_by_size.json'}")


if __name__ == "__main__":
    main()
