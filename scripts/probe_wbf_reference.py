#!/usr/bin/env python
"""任务4e：用官方 WBF（ZFTurbo/ensemble_boxes v1.0.9）交叉验证我的实现。

背景：Task 4b/§2.5 的核心负结论是"WBF 的置信度公式已隐含多视角一致性信号"，
      而我用的是自己重写的简化 WBF。若官方实现给出同样的三点模式
      （avg 差 / max 中 / box_and_model_avg 好），该结论即与实现无关。

官方实现的 conf_type ∈ {avg, max, box_and_model_avg, absent_model_aware_avg}，
正好对应我实现的 {mean, max, orig, —}。

用法:
  python probe_wbf_reference.py --limit 50      # 先估速
  python probe_wbf_reference.py --limit 0       # 全量
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import (DS, build_gt, evaluate, gated_pool, m_nms,  # noqa: E402
                                    to_preds, union_pool)
from probe_wbf_mechanism import wbf_variant  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")
REF_PATH = ROOT / "third_party/ensemble_boxes/ensemble_boxes_wbf.py"


def load_ref():
    spec = importlib.util.spec_from_file_location("wbf_ref", REF_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.weighted_boxes_fusion


def norm(b, W, H):
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    x = b.copy().astype(np.float32)
    x[:, [0, 2]] = np.clip(x[:, [0, 2]] / W, 0, 1)
    x[:, [1, 3]] = np.clip(x[:, [1, 3]] / H, 0, 1)
    return x


def denorm(b, W, H):
    if len(b) == 0:
        return np.zeros((0, 4), np.float32)
    x = b.copy().astype(np.float32)
    x[:, [0, 2]] *= W
    x[:, [1, 3]] *= H
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--iou-thr", type=float, default=0.6)
    ap.add_argument("--out", default=str(ROOT / "wbf_reference.json"))
    a = ap.parse_args()

    wbf_ref = load_ref()
    z = np.load(ROOT / "fusion_cache_full.npz", allow_pickle=True)
    data = list(z["cache"])
    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        data, files = data[:a.limit], files[:a.limit]
    print(f"图数 {len(data)}  官方 WBF iou_thr={a.iou_thr}", flush=True)
    sizes = []
    for l in files:
        with Image.open(DS / l[2:]) as im:
            sizes.append(im.size)
    coco_gt = build_gt(files)

    res = {"n_img": len(data), "iou_thr": a.iou_thr, "impl": "ensemble_boxes 1.0.9 (ZFTurbo)"}

    def run(tag, fn):
        t0 = time.perf_counter()
        preds = []
        for i, it in enumerate(data):
            b, s, c = fn(it, i)
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4),
                    "sec": round(time.perf_counter() - t0, 1)}
        print(f"  {tag:<44} AP50-95={p95:.4f}  AP50={p50:.4f}  ({res[tag]['sec']}s)", flush=True)

    def ref_fn(pool_kind, conf_type):
        def f(it, i):
            W, H = sizes[i]
            fb, fs, fc = it[0], it[1], it[2]
            tb, ts, tc = it[3], it[4], it[5]
            if pool_kind == "gated" and len(tb):
                tsz = np.sqrt((tb[:, 2] - tb[:, 0]).clip(0) * (tb[:, 3] - tb[:, 1]).clip(0))
                k = tsz < 48.0
                tb, ts, tc = tb[k], ts[k], tc[k]
            boxes, scores, labels = wbf_ref(
                [norm(fb, W, H), norm(tb, W, H)],
                [fs.astype(np.float32), ts.astype(np.float32)],
                [fc.astype(np.float32), tc.astype(np.float32)],
                weights=None, iou_thr=a.iou_thr, skip_box_thr=0.0,
                conf_type=conf_type, allows_overflow=False)
            return denorm(boxes, W, H), scores, labels.astype(int)
        return f

    print("\n=== 官方 WBF，union 池（整图 + 3x1 切片两路）===", flush=True)
    for ct in ("avg", "max", "box_and_model_avg", "absent_model_aware_avg"):
        run(f"union + WBF[官方, conf_type={ct}]", ref_fn("union", ct))

    print("\n=== 官方 WBF，gated 池（>=48px 只信整图）===", flush=True)
    for ct in ("box_and_model_avg", "max"):
        run(f"gated + WBF[官方, conf_type={ct}]", ref_fn("gated", ct))

    print("\n=== 我的实现在同一子集上的对照 ===", flush=True)
    run("union + NMS（参考）", lambda it, i: m_nms(*union_pool(*it[:6]), 0.6))
    for mode, name in (("orig", "原式"), ("max", "max"), ("mean", "mean")):
        run(f"union + WBF[我的, {name}]",
            lambda it, i, mode=mode: wbf_variant(*union_pool(*it[:6]), 0.6, mode))

    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n已写入 {a.out}")


if __name__ == "__main__":
    main()
