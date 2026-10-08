#!/usr/bin/env python
"""任务3c：分辨率-分尺寸漏检/召回（GPU，需训练结束后运行）。

Task 3 已证明整图 @1600 支配切片管线。本脚本补机制证据：
把 960 / 1280 / 1600 / 1920 四个分辨率下的分尺寸召回拆开，
验证"改善随目标变小单调增大"这一规律是否在更高分辨率上延续，
并定位 1920 退化的尺寸来源。

用法: python probe_resolution_by_size.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, BUCKETS, build_gt, evaluate, predict_full, recall_by_bucket  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))
WEIGHTS = ROOT / "runs/visdrone/p2_960/weights/best.pt"


def main():
    from ultralytics import YOLO
    files = [l.strip() for l in open(DS / "val.txt")]
    coco_gt = build_gt(files)
    y = YOLO(str(WEIGHTS))
    print(f"权重 {WEIGHTS}  图数 {len(files)}", flush=True)

    res = {"weights": str(WEIGHTS), "rows": {}}
    for imgsz in (960, 1280, 1600, 1920):
        print(f"\n=== imgsz={imgsz} ===", flush=True)
        t0 = time.perf_counter()
        preds, per_image = [], []
        for i, line in enumerate(files):
            b, s, c = predict_full(y, DS / line[2:], imgsz, 0.001, 0.7, "0")
            per_image.append((b, s, c))
            for j in range(len(s)):
                preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                              "bbox": [float(b[j, 0]), float(b[j, 1]),
                                       float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                              "score": float(s[j])})
            if i % 150 == 0:
                print(f"  {i}/{len(files)}", flush=True)
        p95, p50 = evaluate(coco_gt, preds)
        rec = recall_by_bucket(files, per_image)
        res["rows"][str(imgsz)] = {
            "ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds),
            "recall": {nm: (round(rec[nm][1] / rec[nm][0], 4) if rec[nm][0] else None)
                       for nm, _, _ in BUCKETS},
            "gt": {nm: rec[nm][0] for nm, _, _ in BUCKETS},
            "sec": round(time.perf_counter() - t0, 1)}
        print(f"  AP50-95={p95:.4f}  AP50={p50:.4f}  "
              f"召回={res['rows'][str(imgsz)]['recall']}", flush=True)
        torch.cuda.empty_cache()
        json.dump(res, open(ROOT / "resolution_by_size.json", "w"),
                  ensure_ascii=False, indent=1)

    names = [b[0] for b in BUCKETS]
    keys = [k for k in res["rows"]]
    print("\n" + "=" * 86)
    print(f"{'尺寸':>8}{'GT数':>8}" + "".join(f"{'@'+k:>16}" for k in keys))
    for nm in names:
        gt = res["rows"][keys[0]]["gt"][nm]
        if not gt:
            continue
        print(f"{nm:>8}{gt:>8}" +
              "".join(f"{res['rows'][k]['recall'][nm]:>16.4f}" for k in keys))
    print("-" * 86)
    print(f"{'AP50-95':>8}{'':>8}" +
          "".join(f"{res['rows'][k]['ap50_95']:>16.4f}" for k in keys))
    print("=" * 86)
    print(f"已写入 {ROOT/'resolution_by_size.json'}")


if __name__ == "__main__":
    main()
