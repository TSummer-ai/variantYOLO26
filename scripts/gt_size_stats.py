#!/usr/bin/env python
"""VisDrone 目标尺寸统计（不需要模型，纯 GT）：原图像素 vs letterbox 到 imgsz 后的等效像素。

用法: python gt_size_stats.py --imgsz 640
"""

from __future__ import annotations

import os

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
BUCKETS = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 128), (128, 1e9)]


def collect(split: str):
    rows = []
    for line in open(DS / f"{split}.txt"):
        ip = DS / line.strip()[2:]
        with Image.open(ip) as im:
            w, h = im.size
        lp = DS / "labels" / split / (ip.stem + ".txt")
        if not lp.exists():
            continue
        for ln in open(lp):
            c, _, _, bw, bh = ln.split()
            rows.append((int(c), float(bw) * w, float(bh) * h, w, h))
    return np.array([(r[0], r[1], r[2], r[3], r[4]) for r in rows], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    for split in ("train", "val"):
        d = collect(split)
        cls, bw, bh, W, H = d[:, 0].astype(int), d[:, 1], d[:, 2], d[:, 3], d[:, 4]
        size = np.sqrt(bw * bh)
        scale = np.minimum(a.imgsz / W, a.imgsz / H)          # ultralytics letterbox 缩放
        size_s = size * scale
        print(f"\n===== {split}: {len(d)} 个目标 / {len(set(zip(W, H)))} 种分辨率")
        print(f"{'尺寸(px)':>12} | {'原图 GT 占比':>12} | {f'@{a.imgsz} letterbox 占比':>18}")
        for lo, hi in BUCKETS:
            m = (size >= lo) & (size < hi)
            ms = (size_s >= lo) & (size_s < hi)
            lab = f"{lo}-{hi}" if hi < 1e9 else f">={lo}"
            print(f"{lab:>12} | {m.mean() * 100:11.1f}% | {ms.mean() * 100:17.1f}%")
        print(f"  中位数: 原图 {np.median(size):.1f}px -> @640 {np.median(size_s):.1f}px")
        print(f"  COCO small(area<32^2) 占比: 原图 {(size < 32).mean() * 100:.1f}% | @640 {(size_s < 32).mean() * 100:.1f}%")
        print("  分类别中位尺寸(原图 px):")
        for c in range(len(NAMES)):
            m = cls == c
            if m.sum():
                print(f"    {NAMES[c]:16s} n={m.sum():6d} 中位 √area={np.median(size[m]):6.1f}px  bw中位={np.median(bw[m]):6.1f} bh中位={np.median(bh[m]):6.1f}")


if __name__ == "__main__":
    main()
