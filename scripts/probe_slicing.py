#!/usr/bin/env python
"""切片推理（slicing / SAHI 式）：把图切成带重叠的小块分别检测再合并。

机制与本项目已验证的结论一致：精度收益来自"目标在输入里更大"（P2 +1.6 AP、960 +6.5 AP）。
切片让每个目标在每次前向里进一步放大（2x2 切片 ≈ 等效 1.7 倍尺度），且**不需要训练**。

对照：整图 960 单次前向。评测用 COCO 协议（与 oracle 分析同口径）。

用法:
    python probe_slicing.py --weights ... --grid 2 --overlap 0.2 --device 0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
BUCKETS = [("0-8", 0, 8), ("8-16", 8, 16), ("16-32", 16, 32), ("32-64", 32, 64), (">=64", 64, 1e9)]


def tile_starts(size: int, n: int, overlap: float):
    """返回每个轴上的切片起点；tile = size*(1+overlap*(n-1))/n。"""
    if n <= 1:
        return [0], size
    tile = int(round(size * (1 + overlap * (n - 1)) / n))
    tile = min(tile, size)
    if n == 1:
        return [0], tile
    step = (size - tile) / (n - 1)
    return [int(round(i * step)) for i in range(n)], tile


def predict_full(model, path, imgsz, conf, iou, device, max_det=3000):
    r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)


def predict_sliced(model, path, imgsz, conf, iou, device, grid, overlap, max_det=3000):
    """切片推理：每块缩放到 imgsz 检测，坐标映射回原图。grid 可为 int（方形）或 (nx, ny)。"""
    im = Image.open(path).convert("RGB")
    W, H = im.size
    nx, ny = (grid, grid) if isinstance(grid, int) else grid
    xs, tw = tile_starts(W, nx, overlap)
    ys, th = tile_starts(H, ny, overlap)
    boxes, scores, clses = [], [], []
    for y in ys:
        for x in xs:
            crop = im.crop((x, y, min(x + tw, W), min(y + th, H)))
            r = model.predict(np.array(crop), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                              device=device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            boxes.append(b)
            scores.append(r.boxes.conf.cpu().numpy())
            clses.append(r.boxes.cls.cpu().numpy().astype(int))
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(boxes), np.concatenate(scores), np.concatenate(clses)


def merge_nms(boxes, scores, clses, iou_thr=0.6):
    """跨块合并：按类别做一次全局 NMS。"""
    if len(boxes) == 0:
        return boxes, scores, clses
    keep_all = []
    for c in np.unique(clses):
        m = clses == c
        b, s = boxes[m], scores[m]
        order = np.argsort(-s)
        b, s = b[order], s[order]
        idx = torch.from_numpy(b).float()
        sc = torch.from_numpy(s).float()
        from torchvision.ops import nms
        keep = nms(idx, sc, iou_thr).numpy()
        keep_all.append(np.nonzero(m)[0][order][keep])
    keep = np.concatenate(keep_all)
    return boxes[keep], scores[keep], clses[keep]


def recall_by_bucket(files, preds_per_image, iou_thr=0.5):
    """分尺寸召回：GT 被同类且 IoU>=thr 的预测命中即算召回（conf 已由调用方过滤）。"""
    from ultralytics.utils.metrics import box_iou
    stats = {b[0]: [0, 0] for b in BUCKETS}
    for i, line in enumerate(files):
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        lp = DS / "labels/val" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        gt, gc = [], []
        for ln in open(lp):
            c, x, y, bw, bh = ln.split()
            c, x, y, bw, bh = int(c), float(x), float(y), float(bw), float(bh)
            gt.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            gc.append(c)
        gt = np.array(gt).reshape(-1, 4); gc = np.array(gc)
        gs = np.sqrt((gt[:, 2] - gt[:, 0]).clip(0) * (gt[:, 3] - gt[:, 1]).clip(0))
        pb, ps, pc = preds_per_image[i]
        hit = np.zeros(len(gt), bool)
        if len(pb):
            ious = box_iou(torch.from_numpy(gt).float(), torch.from_numpy(pb).float()).numpy()
            for gi in range(len(gt)):
                same = pc == gc[gi]
                hit[gi] = bool(same.any() and ious[gi][same].max() >= iou_thr)
        for name, lo, hi in BUCKETS:
            m = (gs >= lo) & (gs < hi)
            stats[name][0] += int(m.sum()); stats[name][1] += int(hit[m].sum())
    return stats


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
            coco["annotations"].append({"id": aid, "image_id": idx, "category_id": c + 1,
                                        "bbox": [(x - bw / 2) * w, (y - bh / 2) * h, bw * w, bh * h],
                                        "area": bw * w * bh * h, "iscrowd": 0})
    return coco


def evaluate(coco_gt, preds):
    from faster_coco_eval import COCO, COCOeval_faster
    cg = COCO(coco_gt)
    dt = cg.loadRes(preds)
    ev = COCOeval_faster(cg, dt, iouType="bbox", print_function=lambda *a, **k: None)
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo26n-visdrone-p2-960.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--grid", default="2", help="切片数：整数=方形(2=2x2)，或 'nx,ny'（如 2,1 = 横向两列）")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--merge-iou", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipelines", default="full,tiles")
    ap.add_argument("--big-px", type=float, default=32.0, help="adaptive 模式下超过此尺寸(原图像素)只取整图结果")
    a = ap.parse_args()
    a.grid = tuple(int(v) for v in str(a.grid).split(",")) if "," in str(a.grid) else int(a.grid)

    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    model = YOLO(a.weights)
    coco_gt = build_gt(files)

    res = {}
    raw = {}
    for tag in a.pipelines.split(","):
        preds = []
        per_image = []
        for i, line in enumerate(files):
            ip = DS / line[2:]
            bf = sf = cf = None
            want_full = "full" in tag
            want_tiles = ("tiles" in tag) or ("adaptive" in tag)
            if want_full:
                bf, sf, cf = predict_full(model, ip, a.imgsz, a.conf, a.iou, a.device)
            if want_tiles:
                bt, st_, ct = predict_sliced(model, ip, a.imgsz, a.conf, a.iou, a.device, a.grid, a.overlap)
                if bf is None:
                    b, s, c = merge_nms(bt, st_, ct, a.merge_iou)
                elif "adaptive" in tag:
                    # 尺寸感知：大框只信整图（切片会把大目标切断），小框两者都留
                    with Image.open(ip) as im:
                        W, H = im.size
                    sz_t = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
                    sz_f = np.sqrt((bf[:, 2] - bf[:, 0]).clip(0) * (bf[:, 3] - bf[:, 1]).clip(0)) if len(bf) else np.zeros(0)
                    keep_t = sz_t < a.big_px
                    keep_f = np.ones(len(bf), bool) if len(bf) else np.zeros(0, bool)
                    b = np.concatenate([bf[keep_f], bt[keep_t]]) if len(bf) else bt[keep_t]
                    s = np.concatenate([sf[keep_f], st_[keep_t]]) if len(bf) else st_[keep_t]
                    c = np.concatenate([cf[keep_f], ct[keep_t]]) if len(bf) else ct[keep_t]
                    b, s, c = merge_nms(b, s, c, a.merge_iou)
                else:
                    b = np.concatenate([bf, bt]); s = np.concatenate([sf, st_]); c = np.concatenate([cf, ct])
                    b, s, c = merge_nms(b, s, c, a.merge_iou)
            else:
                b, s, c = bf, sf, cf
            per_image.append((b, s, c))
            for j in range(len(s)):
                preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                              "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                              "score": float(s[j])})
            if i % 100 == 0:
                print(f"  [{tag}] {i}/{len(files)}", flush=True)
        ap95, ap50 = evaluate(coco_gt, preds)
        res[tag] = (ap95, ap50, len(preds))
        raw[tag] = per_image
        print(f"  {tag}: AP50-95={ap95:.4f}  AP50={ap50:.4f}  ({len(preds)} 框)")

    base = res.get("full")
    print(f"\n=== 各管线（相对整图；grid={a.grid}x{a.grid}, overlap={a.overlap}, big_px={a.big_px}）===")
    for tag, (p95, p50, n) in res.items():
        if base is None or tag == "full":
            print(f"  {tag:<14} AP50-95={p95:.4f} AP50={p50:.4f} ({n} 框)")
        else:
            print(f"  {tag:<14} AP50-95={p95:.4f} ({p95-base[0]:+.4f})  AP50={p50:.4f} ({p50-base[1]:+.4f})  ({n} 框)")
    print(f"\n=== 分尺寸召回诊断（切片是否把小目标召回拉高？）===")
    rfs = {t: recall_by_bucket(files, raw[t]) for t in raw}
    tags = list(raw)
    hdr = "".join(f"{t:>11}" for t in tags)
    print(f"{'尺寸':>8}{'GT数':>8}{hdr}")
    for name, _, _ in BUCKETS:
        if rfs[tags[0]][name][0] == 0:
            continue
        vals = "".join(f"{rfs[t][name][1]/rfs[t][name][0]:>11.3f}" for t in tags)
        print(f"{name:>8}{rfs[tags[0]][name][0]:>8}{vals}")
    print(f"\n判定线（预先登记）: AP50-95 增益 >= +1.0 才算值得做成方法")


if __name__ == "__main__":
    main()
