#!/usr/bin/env python
"""尺寸感知切片融合：参数寻优（缓存预测，融合规则在内存里扫）。

为了效率：对每个 (grid, overlap) 只跑一次前向，把整图与各切片的预测缓存下来，
然后对不同的 big_px / merge_iou 组合在内存里做融合与评测。

用法: python tune_slicing.py --weights <pt> --limit 120
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced, recall_by_bucket  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo26n-visdrone-p2-960.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--grids", default="2x0.2,3x0.2,2x0.3")
    ap.add_argument("--bigpx", default="16,32,48,100000")
    ap.add_argument("--mergeious", default="0.5,0.6")
    a = ap.parse_args()

    if not (DS / "val.txt").exists():
        raise SystemExit(
            f"找不到数据集标注: {DS / 'val.txt'}\n"
            "请先按 README「数据准备」运行 scripts/prepare_visdrone.sh，"
            "或在 configs/visdrone.yaml 里把 path 指向你的数据目录。"
        )
    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    coco_gt = build_gt(files)
    model = YOLO(a.weights)

    # 尺寸表（原图像素）
    sizes = {}
    for line in files:
        ip = DS / line[2:]
        with Image.open(ip) as im:
            W, H = im.size
        sizes[ip.name] = (W, H)

    results = []
    for spec in a.grids.split(","):
        grid, ov = spec.split("x")
        grid, ov = int(grid), float(ov)
        print(f"\n########## 缓存预测: grid={grid}x{grid} overlap={ov} ##########", flush=True)
        cache = []
        for i, line in enumerate(files):
            ip = DS / line[2:]
            bf, sf, cf = predict_full(model, ip, a.imgsz, a.conf, a.iou, a.device)
            bt, st, ct = predict_sliced(model, ip, a.imgsz, a.conf, a.iou, a.device, grid, ov)
            cache.append((bf, sf, cf, bt, st, ct))
            if i % 60 == 0:
                print(f"   {i}/{len(files)}", flush=True)

        for big in [float(x) for x in a.bigpx.split(",")]:
            for miou in [float(x) for x in a.mergeious.split(",")]:
                preds, per_image = [], []
                for i, (bf, sf, cf, bt, st, ct) in enumerate(cache):
                    if big >= 1e5:  # 不做尺寸门控 = 普通融合
                        b = np.concatenate([bf, bt]); s = np.concatenate([sf, st]); c = np.concatenate([cf, ct])
                    else:
                        sz_t = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0)) if len(bt) else np.zeros(0)
                        keep_t = sz_t < big
                        b = np.concatenate([bf, bt[keep_t]]); s = np.concatenate([sf, st[keep_t]]); c = np.concatenate([cf, ct[keep_t]])
                    b, s, c = merge_nms(b, s, c, miou)
                    per_image.append((b, s, c))
                    for j in range(len(s)):
                        preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                                      "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                                      "score": float(s[j])})
                p95, p50 = evaluate(coco_gt, preds)
                results.append((grid, ov, big, miou, p95, p50, len(preds)))
                print(f"   grid={grid} ov={ov} big_px={big:g} merge_iou={miou}: AP50-95={p95:.4f} AP50={p50:.4f} ({len(preds)} 框)", flush=True)

    print("\n=== 汇总（按 AP50-95 排序）===")
    print(f"{'grid':>5}{'overlap':>9}{'big_px':>9}{'merge_iou':>11}{'AP50-95':>10}{'AP50':>9}{'框数':>10}")
    for r in sorted(results, key=lambda x: -x[4]):
        print(f"{r[0]:>5}{r[1]:>9}{r[2]:>9g}{r[3]:>11}{r[4]:>10.4f}{r[5]:>9.4f}{r[6]:>10}")


if __name__ == "__main__":
    main()
