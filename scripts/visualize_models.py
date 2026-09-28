import os
from pathlib import Path

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))

#!/usr/bin/env python
"""跨模型定性对比：GT | baseline@640 | P2@960（同一张图，红框=漏检）。

用法: python visualize_models.py --out runs/visdrone/viz --n 2 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent))
from visualize_heads import DS, ROOT, draw, load_gt, match  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/viz"))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = [l.strip() for l in open(DS / "val.txt")]
    counts = []
    for line in files:
        lp = DS / "labels/val" / (Path(line).stem + ".txt")
        counts.append(sum(1 for _ in open(lp)) if lp.exists() else 0)
    picks = [files[i] for i in np.argsort(counts)[::-1][: a.n]]

    models = [("baseline@640", YOLO(str(ROOT / "runs/visdrone/base_n_640/weights/best.pt")), 640),
              ("P2@960", YOLO(str(ROOT / "runs/visdrone/p2_960/weights/best.pt")), 960)]

    rows = []
    for line in picks:
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        gt, gcls = load_gt(DS / "labels/val" / (ip.stem + ".txt"), w, h)
        img0 = Image.open(ip).convert("RGB")

        hit_ref, _ = None, None
        panels = []
        # 第一列：GT（用 P2@960 的命中情况着色，体现"最好模型也漏了哪些"）
        first = True
        for name, model, imgsz in models:
            r = model.predict(str(ip), imgsz=imgsz, conf=a.conf, iou=0.7, device=a.device, verbose=False)[0]
            pr = r.boxes.xyxy.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.zeros((0, 4))
            pc = r.boxes.cls.cpu().numpy().astype(int) if len(pr) else np.zeros(0, int)
            hit, used = match(gt, gcls, pr, pc)
            if first:
                panels.append(draw(img0, gt, hit, np.zeros((0, 4)), np.zeros(0, int),
                                   f"GT 共 {len(gt)} 个（红=连 P2@960 都漏）"))
                first = False
            panels.append(draw(img0, gt, hit, pr, pc,
                               f"{name}  TP={int(hit.sum())} FN={int((~hit).sum())} FP={int(len(pr)-len(used))}"))
        row = Image.new("RGB", (sum(p.width for p in panels) + 10 * len(panels), panels[0].height), (255, 255, 255))
        x = 0
        for p in panels:
            row.paste(p, (x, 0)); x += p.width + 10
        rows.append(row)

    W = max(r.width for r in rows); H = sum(r.height for r in rows) + 10 * len(rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 10
    fp = out / "models_compare.png"
    canvas.save(fp)
    print("saved", fp)


if __name__ == "__main__":
    main()
