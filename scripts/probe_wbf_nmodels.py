#!/usr/bin/env python
"""任务4d：WBF 的 n_models 敏感性（纯 CPU，复用缓存）。

Task 4b 用了 WBF 的标准公式
    fused = Σ(s_i^2)/Σ(s_i) * min(N, n_models)/n_models
并取 n_models=2。审稿人会问"n_models=2 是不是挑出来的"。
本脚本把 n_models 从 1 扫到 6，划出完整敏感性曲线。

注意：n_models=1 时孤簇保持原分数、多成员簇取加权平均（反而偏向孤簇）；
      n_models 越大，孤簇被压得越狠（多视角加成越强）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import (DS, build_gt, evaluate, gated_pool, m_nms,  # noqa: E402
                                    to_preds, union_pool)
from probe_wbf_mechanism import wbf_variant  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")


def main():
    z = np.load(ROOT / "fusion_cache_full.npz", allow_pickle=True)
    data = list(z["cache"])
    files = [l.strip() for l in open(DS / "val.txt")]
    coco_gt = build_gt(files)
    print(f"图数 {len(data)}", flush=True)

    res = {}

    def run(tag, fn):
        preds = []
        for i, it in enumerate(data):
            b, s, c = fn(it)
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4)}
        print(f"  {tag:<40} AP50-95={p95:.4f}  AP50={p50:.4f}", flush=True)

    print("\n=== union 池 + WBF，n_models 扫描 ===", flush=True)
    run("union + NMS（参考）", lambda it: m_nms(*union_pool(*it[:6]), 0.6))
    for nm in (1, 2, 3, 4, 6, 10):
        run(f"union + WBF(n_models={nm})",
            lambda it, nm=nm: wbf_variant(*union_pool(*it[:6]), 0.6, "orig", n_models=nm))

    print("\n=== gated 池 + WBF，n_models 扫描 ===", flush=True)
    run("gated + NMS（参考）", lambda it: m_nms(*gated_pool(*it[:6]), 0.6))
    for nm in (1, 2, 3):
        run(f"gated + WBF(n_models={nm})",
            lambda it, nm=nm: wbf_variant(*gated_pool(*it[:6]), 0.6, "orig", n_models=nm))

    json.dump(res, open(ROOT / "wbf_nmodels.json", "w"), ensure_ascii=False, indent=1)
    print(f"\n已写入 {ROOT/'wbf_nmodels.json'}")


if __name__ == "__main__":
    main()
