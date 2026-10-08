#!/usr/bin/env python
"""任务4b：WBF 增益来源核查（纯 CPU，复用已缓存预测）。

Task 4 发现 union + WBF（0.2826）反超 gated + NMS + 一致性（MV-Fuse, 0.2788）。
需要判定：增益来自"框坐标加权平均"，还是来自 WBF 的置信度公式
    fused_score = Σ(s_i^2)/Σ(s_i) * min(N, n_models)/n_models
其中 N=1 的孤簇分数被压到 s/2，而 N>=2 的簇保持原量级
—— 这等价于"按多视角支持度提升分数"，即与 MV-Fuse 的一致性重打分同源。

对照：
  wbf          原公式（含多视角加成）
  wbf_max      簇内取 max 分数（去掉多视角加成，保留坐标融合）
  wbf_mean     簇内取均值分数
  nms          普通 NMS
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import (DS, BIG_PX, build_gt, evaluate, gated_pool,  # noqa: E402
                                    iou_matrix, m_nms, union_pool, to_preds)

ROOT = Path("/home/wang/DeepSeek/YOLO")


def wbf_variant(b, s, c, iou_thr=0.6, score_mode="orig", n_models=2):
    """score_mode: orig | max | mean"""
    if len(b) == 0:
        return b, s, c
    fb, fs, fc = [], [], []
    for cl in np.unique(c):
        m = c == cl
        bb, ss = b[m], s[m]
        order = np.argsort(-ss)
        bb, ss = bb[order], ss[order]
        clusters, reps = [], np.zeros((0, 4))
        for i in range(len(bb)):
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
            if score_mode == "orig":
                sc = float((S * S).sum() / S.sum()) * min(len(S), n_models) / n_models
            elif score_mode == "max":
                sc = float(S.max())
            else:
                sc = float(S.mean())
            fb.append(nb); fs.append(sc); fc.append(int(cl))
    return np.array(fb).reshape(-1, 4), np.array(fs), np.array(fc, int)


def main():
    cache = ROOT / "fusion_cache_full.npz"
    files = [l.strip() for l in open(DS / "val.txt")]
    z = np.load(cache, allow_pickle=True)
    data = list(z["cache"])
    print(f"缓存载入 {cache}  图数={len(data)}", flush=True)
    coco_gt = build_gt(files)

    res = {}

    def run(tag, fn):
        preds = []
        for i, it in enumerate(data):
            bb, ss, cc = fn(it)
            to_preds(bb, ss, cc, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds)}
        print(f"  {tag:<28} AP50-95={p95:.4f}  AP50={p50:.4f}  ({len(preds)} 框)", flush=True)

    u = lambda it: union_pool(*it[:6])  # noqa: E731
    g = lambda it: gated_pool(*it[:6])  # noqa: E731

    print("\n=== WBF 增益来源核查（union 池，4 前向）===", flush=True)
    run("union + NMS", lambda it: m_nms(*u(it), 0.6))
    run("union + WBF(原公式)", lambda it: wbf_variant(*u(it), 0.6, "orig"))
    run("union + WBF(簇内 max)", lambda it: wbf_variant(*u(it), 0.6, "max"))
    run("union + WBF(簇内 mean)", lambda it: wbf_variant(*u(it), 0.6, "mean"))
    print("\n=== gated 池 ===\n", flush=True)
    run("gated + WBF(原公式)", lambda it: wbf_variant(*g(it), 0.6, "orig"))
    run("gated + WBF(簇内 max)", lambda it: wbf_variant(*g(it), 0.6, "max"))

    json.dump(res, open(ROOT / "wbf_mechanism.json", "w"), ensure_ascii=False, indent=1)
    print("\n" + "=" * 58)
    for k, v in res.items():
        print(f"{k:<30}{v['ap50_95']:>9.4f}")
    print("=" * 58)
    print(f"已写入 {ROOT/'wbf_mechanism.json'}")


if __name__ == "__main__":
    main()
