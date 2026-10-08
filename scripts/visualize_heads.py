#!/usr/bin/env python
"""可视化：同一个模型里 one2many(默认,带NMS) 头 vs one2one(e2e, nms=False) 头的检测差异。

这是两个头差异的直观对照：同一张图、同一套权重，可以逐图比较它们的命中与漏检。
绿色 = 命中(GT)，红色 = 漏检(GT)，黄/橙 = 两个头各自的预测框。

用法:
    python visualize_heads.py --weights runs/visdrone/base_n_640/weights/best.pt \
        --out runs/visdrone/viz --n 3 --conf 0.25 --device 0
"""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"


def load_gt(label_file: Path, w: int, h: int):
    if not label_file.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    rows = np.array([ln.split() for ln in open(label_file) if ln.strip()], dtype=float)
    if rows.size == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1), cls


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(a[:, 2:] - a[:, :2], 1)[:, None]
    area_b = np.prod(b[:, 2:] - b[:, :2], 1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match(gt_xyxy, gt_cls, pr_xyxy, pr_cls, iou_thr=0.5):
    """贪心一对一匹配，返回 GT 是否被命中 与 命中的预测索引。"""
    hit = np.zeros(len(gt_xyxy), dtype=bool)
    if len(pr_xyxy) == 0 or len(gt_xyxy) == 0:
        return hit, []
    used_pr = set()
    ious = iou_matrix(gt_xyxy, pr_xyxy)
    for gi in range(len(gt_xyxy)):
        best, bj = iou_thr, -1
        for pj in range(len(pr_xyxy)):
            if pj in used_pr or pr_cls[pj] != gt_cls[gi]:
                continue
            if ious[gi, pj] >= best:
                best, bj = ious[gi, pj], pj
        if bj >= 0:
            hit[gi] = True
            used_pr.add(bj)
    return hit, sorted(used_pr)


def draw(ax_img, gt, gt_hit, pr_xyxy, pr_cls, title, color=(255, 215, 0)):
    img = ax_img.copy()
    d = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(gt):
        # 命中=绿(细)，漏检=红(粗)，直接看出漏在哪
        d.rectangle([x1, y1, x2, y2], outline=(0, 200, 0) if gt_hit[i] else (255, 0, 0),
                    width=2 if gt_hit[i] else 3)
    for (x1, y1, x2, y2) in pr_xyxy:
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
    d.rectangle([0, 0, img.width - 1, 26], fill=(0, 0, 0))
    d.text((6, 6), title, fill=(255, 255, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/viz"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--title", default="")
    ap.add_argument("--indices", default="", help="指定 val.txt 行号(逗号分隔)，默认自动挑小目标最多的图")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    files = [l.strip() for l in open(DS / "val.txt")]

    if a.indices:
        picks = [files[int(i)] for i in a.indices.split(",")]
    else:
        # 自动挑 GT 数量最多（小目标最密）的图
        counts = []
        for line in files:
            ip = DS / line[2:]
            lp = DS / "labels/val" / (ip.stem + ".txt")
            counts.append(sum(1 for _ in open(lp)) if lp.exists() else 0)
        order = np.argsort(counts)[::-1]
        picks = [files[i] for i in order[: a.n]]

    rows, summary = [], []
    for line in picks:
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        gt, gcls = load_gt(DS / "labels/val" / (ip.stem + ".txt"), w, h)
        img0 = Image.open(ip).convert("RGB")

        res = {}
        for tag, nms in (("o2m", None), ("o2o", False)):
            r = model.predict(str(ip), imgsz=a.imgsz, conf=a.conf, iou=0.7, device=a.device,
                              nms=nms, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                res[tag] = (np.zeros((0, 4)), np.zeros((0,), int))
            else:
                res[tag] = (r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int))

        panels, stats = [], {}
        for tag, label, color in (("o2m", "one2many + NMS (默认头)", (255, 215, 0)),
                                  ("o2o", "one2one e2e (nms=False)", (255, 140, 0))):
            pr, pc = res[tag]
            hit, used = match(gt, gcls, pr, pc)
            tp, fn, fp = int(hit.sum()), int((~hit).sum()), int(len(pr) - len(used))
            stats[tag] = (tp, fn, fp)
            panels.append(draw(img0, gt, hit, pr, pc,
                               f"{label}  TP={tp} FN={fn} FP={fp}  conf>={a.conf}", color))
        tp, fn, fp = stats["o2m"]
        tp2, fn2, fp2 = stats["o2o"]
        summary.append((ip.name, len(gt), tp, fn, fp, tp2, fn2, fp2, fn2 - fn))

        # 拼成 1x3（GT 面板复用 o2m 的命中判断着色）
        hit_o2m, _ = match(gt, gcls, *res["o2m"])
        gt_panel = draw(img0, gt, hit_o2m, np.zeros((0, 4)), np.zeros((0,), int), f"GT 共 {len(gt)} 个 (绿=默认头命中/红=漏检)")
        row = Image.new("RGB", (sum(p.width for p in [gt_panel] + panels) + 20, gt_panel.height), (255, 255, 255))
        x = 0
        for p in [gt_panel] + panels:
            row.paste(p, (x, 0))
            x += p.width + 10
        rows.append(row)

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows) + 10 * len(rows)
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 10
    tag = a.title or Path(a.weights).parent.parent.name
    fp_out = out / f"heads_{tag}.png"
    canvas.save(fp_out)
    print(f"saved {fp_out}")
    print(f"\n{'image':34s} {'GT':>5s} | {'o2m TP':>6s} {'FN':>5s} {'FP':>5s} | {'o2o TP':>6s} {'FN':>5s} {'FP':>5s} | {'FN差':>6s}")
    for r in summary:
        print(f"{r[0]:34s} {r[1]:5d} | {r[2]:6d} {r[3]:5d} {r[4]:5d} | {r[5]:6d} {r[6]:5d} {r[7]:5d} | {r[8]:+6d}")


if __name__ == "__main__":
    main()
