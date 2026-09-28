#!/usr/bin/env python
"""检测预算分配策略研究（纯后处理，可 CPU 跑）。

问题：NMS 之后按分数取全局 top-k（默认 k=300）时，小目标容易被大目标挤出名额。
      而实测 max_det 300->1000 能白拿 +0.4 AP，说明大量真阳性被截断了。
这里比较几种"预算分配"规则：
  flat-B        : 全局 top-B（现有做法）
  strat-B-prop  : 按输入尺度分三档，名额按 GT 频率分配(小54%/中32%/大14%)
  strat-B-equal : 三档等额
  strat-B-small : 偏小目标(60%/30%/10%)
每档内部仍按分数取 top-k。

用法: python probe_budget.py --device cpu     # GPU 忙时用 cpu
"""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def collect(model, files, imgsz, device, conf=0.001, iou=0.7, raw_max=3000):
    """跑一遍预测，返回每图的 (boxes_xyxy, scores, cls, 输入尺度下的尺寸)。"""
    out = []
    for i, line in enumerate(files):
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        r = model.predict(str(ip), imgsz=imgsz, conf=conf, iou=iou, max_det=raw_max,
                          device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            out.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0)))
            continue
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        c = r.boxes.cls.cpu().numpy().astype(int)
        scale = min(imgsz / w, imgsz / h)
        size_in = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)) * scale
        out.append((b, s, c, size_in))
        if i % 100 == 0:
            print(f"  predict {i}/{len(files)}", flush=True)
    return out


def build_gt(files):
    coco = {"images": [], "annotations": [], "categories": [{"id": c + 1, "name": n} for c, n in enumerate(NAMES)]}
    aid = 0
    for idx, line in enumerate(files):
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        coco["images"].append({"id": idx, "file_name": ip.name, "width": w, "height": h})
        lp = DS / "labels/val" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        for ln in open(lp):
            c, x, y, bw, bh = ln.split()
            c, x, y, bw, bh = int(c), float(x), float(y), float(bw), float(bh)
            aid += 1
            coco["annotations"].append({
                "id": aid, "image_id": idx, "category_id": c + 1,
                "bbox": [(x - bw / 2) * w, (y - bh / 2) * h, bw * w, bh * h],
                "area": bw * w * bh * h, "iscrowd": 0,
            })
    return coco


def evaluate(coco_gt, preds):
    from faster_coco_eval import COCO, COCOeval_faster
    cg = COCO(coco_gt)
    dt = cg.loadRes(preds)
    ev = COCOeval_faster(cg, dt, iouType="bbox", print_function=lambda *a, **k: None)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def select(boxes, scores, cls, size_in, budget, rule):
    """返回保留下来的索引。"""
    n = len(scores)
    if n == 0 or budget >= n:
        return np.arange(n)
    if rule == "flat":
        return np.argsort(-scores)[:budget]
    S, M = 12.0, 32.0  # 输入尺度下的分档阈值(px)
    groups = [size_in < S, (size_in >= S) & (size_in < M), size_in >= M]
    frac = {"prop": (0.54, 0.32, 0.14), "equal": (1 / 3, 1 / 3, 1 / 3), "small": (0.60, 0.30, 0.10)}[rule.split("-")[1]]
    keep = []
    for g, f in zip(groups, frac):
        idx = np.nonzero(g)[0]
        k = int(round(budget * f))
        if len(idx):
            keep.append(idx[np.argsort(-scores[idx])[:k]])
    out = np.concatenate(keep) if keep else np.array([], dtype=int)
    if len(out) < budget:  # 名额没用完，按分数补齐
        rest = np.setdiff1d(np.argsort(-scores), out, assume_unique=False)[: budget - len(out)]
        out = np.concatenate([out, rest])
    return out[:budget]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/base_n_640/weights/best.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    files = [l.strip() for l in open(DS / "val.txt")]
    model = YOLO(a.weights)
    data = collect(model, files, a.imgsz, a.device)
    coco_gt = build_gt(files)

    rules = [("flat", 300), ("flat", 1000), ("flat", 3000),
             ("strat-prop", 300), ("strat-equal", 300), ("strat-small", 300),
             ("strat-prop", 1000), ("strat-small", 1000)]
    print(f"\n{'规则':<14}{'预算':>6}{'AP50-95':>10}{'AP50':>9}")
    res = {}
    for rule, B in rules:
        preds = []
        for idx, (b, s, c, sz) in enumerate(data):
            keep = select(b, s, c, sz, B, rule)
            for j in keep:
                preds.append({"image_id": idx, "category_id": int(c[j]) + 1,
                              "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                              "score": float(s[j])})
        ap95, ap50 = evaluate(coco_gt, preds)
        res[(rule, B)] = (ap95, ap50)
        print(f"{rule:<14}{B:>6}{ap95:>10.4f}{ap50:>9.4f}")
    print("\n对照（GPU 上 yolo val 的结果）: flat300 = 0.1820/0.3280 | flat1000 = 0.1860/0.3390")


if __name__ == "__main__":
    main()
