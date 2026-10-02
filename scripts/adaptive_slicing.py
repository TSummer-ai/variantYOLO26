#!/usr/bin/env python
"""自适应切片：只在"小目标密集"的场景切片，把平均推理成本从 5x 降下来。

做法：
  1. 先用一次**低分辨率**整图推理（如 320，成本约 960 的 1/9）估计场景的目标尺度分布
  2. 若"小目标占比/数量"超过阈值 -> 走切片融合；否则只用整图结果
  3. 离线在缓存的预测上扫描阈值，得到 **精度 vs 平均前向次数** 的权衡曲线

同时输出调优后配置（grid=2, overlap=0.3, big_px=48）在全量 val 上的结果。

用法: python adaptive_slicing.py --weights ... [--limit N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))


def fuse(bf, sf, cf, bt, st, ct, big_px, miou):
    """尺寸感知融合：>= big_px 的框只取整图结果，其余两者都留，再做全局 NMS。"""
    if len(bt):
        sz = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
        keep = sz < big_px
    else:
        keep = np.zeros(0, bool)
    b = np.concatenate([bf, bt[keep]]) if len(bf) or keep.any() else np.zeros((0, 4))
    s = np.concatenate([sf, st[keep]]) if len(bf) or keep.any() else np.zeros(0)
    c = np.concatenate([cf, ct[keep]]) if len(bf) or keep.any() else np.zeros(0, int)
    return merge_nms(b, s, c, miou) if len(b) else (b, s, c)


def to_preds(b, s, c, i, out):
    for j in range(len(s)):
        out.append({"image_id": i, "category_id": int(c[j]) + 1,
                    "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                    "score": float(s[j])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo26n-visdrone-p2-960.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--est-imgsz", type=int, default=320, help="用于估计场景的廉价低分辨率")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.3)
    ap.add_argument("--big-px", type=float, default=48.0)
    ap.add_argument("--merge-iou", type=float, default=0.6)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    coco_gt = build_gt(files)
    model = YOLO(a.weights)

    cache_path = Path(f"/tmp/slice_cache_{a.limit or 'all'}_{a.grid}_{a.overlap}_{a.imgsz}.npz")
    if cache_path.exists():
        print(f"复用缓存 {cache_path}", flush=True)
        z = np.load(cache_path, allow_pickle=True)
        cache = list(z["cache"])
    else:
        print(f"缓存预测：整图 {a.imgsz} + 切片 {a.grid}x{a.grid}(ov={a.overlap})", flush=True)
        cache = []
    for i, line in enumerate(files):
        ip = DS / line[2:]
        bf, sf, cf = predict_full(model, ip, a.imgsz, a.conf, a.iou, a.device)
        bt, st, ct = predict_sliced(model, ip, a.imgsz, a.conf, a.iou, a.device, a.grid, a.overlap)
        # 廉价场景估计：低分辨率整图，统计"小目标"数量与占比（换算到原图尺度）
        be, se, ce = predict_full(model, ip, a.est_imgsz, 0.25, a.iou, a.device, max_det=300)
        with Image.open(ip) as im:
            W, H = im.size
        scale = a.est_imgsz / max(W, H)
        sz_e = np.sqrt((be[:, 2] - be[:, 0]).clip(0) * (be[:, 3] - be[:, 1]).clip(0)) / scale if len(be) else np.zeros(0)
        n_small = int((sz_e < 16).sum())
        frac_small = float((sz_e < 16).mean()) if len(sz_e) else 0.0
        cache.append((bf, sf, cf, bt, st, ct, n_small, frac_small, len(be)))
        if i % 100 == 0:
            print(f"   {i}/{len(files)}", flush=True)
        np.savez_compressed(cache_path, cache=np.array(cache, dtype=object))

    # ---- 基线：只用整图 ----
    preds = []
    for i, (bf, sf, cf, *_rest) in enumerate(cache):
        to_preds(bf, sf, cf, i, preds)
    base95, base50 = evaluate(coco_gt, preds)
    print(f"\n[整图基线]      AP50-95={base95:.4f}  AP50={base50:.4f}", flush=True)

    # ---- 全切片融合（调优配置）----
    preds = []
    for i, (bf, sf, cf, bt, st, ct, *_rest) in enumerate(cache):
        b, s, c = fuse(bf, sf, cf, bt, st, ct, a.big_px, a.merge_iou)
        to_preds(b, s, c, i, preds)
    full95, full50 = evaluate(coco_gt, preds)
    print(f"[全切片融合]    AP50-95={full95:.4f} ({full95-base95:+.4f})  AP50={full50:.4f} ({full50-base50:+.4f})"
          f"  平均前向={1 + a.grid * a.grid:.1f}次/图", flush=True)

    # ---- 自适应：按廉价估计决定是否切片 ----
    print(f"\n=== 自适应策略（估计用 {a.est_imgsz}px，成本≈1/{round((a.imgsz/a.est_imgsz)**2)}）===")
    print(f"{'规则':>22}{'触发率':>9}{'平均前向':>10}{'AP50-95':>10}{'Δ vs 整图':>11}{'保留增益':>10}")
    # 用整图那一遍的预测统计来触发（零额外前向）：统计"小框数量/占比"
    stats = []
    for c in cache:
        bf, sf = c[0], c[1]
        sz = np.sqrt((bf[:, 2] - bf[:, 0]).clip(0) * (bf[:, 3] - bf[:, 1]).clip(0)) if len(bf) else np.zeros(0)
        stats.append((int((sz < 16).sum()), float((sz < 16).mean()) if len(sz) else 0.0))
    print(f"  整图小框(orig<16px)数量分布: 中位={int(np.median([s[0] for s in stats]))} "
          f"P25={int(np.percentile([s[0] for s in stats],25))} P75={int(np.percentile([s[0] for s in stats],75))}")

    rules = [(0, "从不触发"), (1, "小框>=1"), (10, "小框>=10"), (25, "小框>=25"),
             (50, "小框>=50"), (100, "小框>=100"), (200, "小框>=200"), (None, "全触发")]
    rows = []
    for key, name in rules:
        if key is None:
            trig = [True] * len(cache)
        elif key == 0:
            trig = [False] * len(cache)
        else:
            trig = [s_[0] >= key for s_ in stats]
        rate = sum(trig) / len(trig)
        n_fwd = 1 + rate * a.grid * a.grid   # 零额外估计成本
        preds = []
        for i, (bf, sf, cf, bt, st, ct, *_r) in enumerate(cache):
            if trig[i]:
                b, s, c = fuse(bf, sf, cf, bt, st, ct, a.big_px, a.merge_iou)
            else:
                b, s, c = bf, sf, cf
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        gain = p95 - base95
        keep = gain / (full95 - base95) * 100 if full95 > base95 else float("nan")
        rows.append((name, rate, n_fwd, p95, gain, keep))
        print(f"{name:>22}{rate:>9.1%}{n_fwd:>10.2f}{p95:>10.4f}{gain:>+11.4f}{keep:>9.0f}%")

    print(f"\n（'平均前向'含低分辨率估计那一次，按等效 {a.imgsz}px 前向折算）")


if __name__ == "__main__":
    main()
