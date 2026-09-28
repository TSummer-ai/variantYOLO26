#!/usr/bin/env python
"""Oracle 分解：把"距离完美检测器"的 AP 损失拆成 定位/分类/漏检/误检 四份。

做法：只用一份预测结果，在预测集上分别做四种"理想化"改写，然后用 COCO 协议重评 AP：
  loc-oracle : 命中预测的框换成 GT 框          -> 定位误差吃掉多少 AP
  cls-oracle : 命中预测的类别换成 GT 类别      -> 分类误差吃掉多少
  fn-oracle  : 漏掉的 GT 补成高分完美预测      -> 漏检吃掉多少
  fp-oracle  : 删掉所有未命中的预测            -> 误检/重复框吃掉多少
  all-oracle : 全部理想化                      -> 应接近 1.0（自检）

用法:
    python oracle_decompose.py --weights runs/visdrone/base_n_640/weights/best.pt --head o2m --device 0
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
IOU_THR = 0.5
CONF = 0.001


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


def load_gt(label_file: Path, w: int, h: int):
    if not label_file.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    rows = np.array([ln.split() for ln in open(label_file) if ln.strip()], dtype=float)
    if rows.size == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return xyxy, cls


def match_same_class(gt, gcls, pr, pcls, pconf):
    """按置信度贪心、同类且 IoU>=thr 的一对一匹配；返回 gt_hit(bool[]) 与 pred_matched(bool[])。"""
    gt_hit = np.zeros(len(gt), dtype=bool)
    pr_matched = np.zeros(len(pr), dtype=bool)
    if len(gt) == 0 or len(pr) == 0:
        return gt_hit, pr_matched
    ious = iou_matrix(pr, gt)
    for pi in np.argsort(-pconf):
        row = ious[pi].copy()
        row[gt_hit] = -1
        row[gcls != pcls[pi]] = -1
        gi = int(np.argmax(row))
        if row[gi] >= IOU_THR:
            gt_hit[gi] = True
            pr_matched[pi] = True
    return gt_hit, pr_matched


def match_any_class(gt, pr, pconf):
    """忽略类别的一对一匹配（用于分类 oracle）。"""
    gt_hit = np.zeros(len(gt), dtype=bool)
    pr_to_gt = -np.ones(len(pr), dtype=int)
    if len(gt) == 0 or len(pr) == 0:
        return gt_hit, pr_to_gt
    ious = iou_matrix(pr, gt)
    for pi in np.argsort(-pconf):
        row = ious[pi].copy()
        row[gt_hit] = -1
        gi = int(np.argmax(row))
        if row[gi] >= IOU_THR:
            gt_hit[gi] = True
            pr_to_gt[pi] = gi
    return gt_hit, pr_to_gt


def evaluate(gt_coco, preds, tag):
    from faster_coco_eval import COCO, COCOeval_faster

    coco_gt = COCO(gt_coco)
    dt = coco_gt.loadRes(preds)
    ev = COCOeval_faster(coco_gt, dt, iouType="bbox", print_function=lambda *a, **k: None)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return float(ev.stats[0]), float(ev.stats[1])  # AP, AP50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/base_n_640/weights/best.pt"))
    ap.add_argument("--head", choices=["o2m", "e2e"], default="o2m")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/oracle"))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.weights)
    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]

    gt_coco = {"images": [], "annotations": [], "categories": [{"id": c + 1, "name": n} for c, n in enumerate(NAMES)]}
    perf, ann_id = [], 0
    n_gt = n_pred = 0

    for idx, line in enumerate(files):
        ip = DS / line[2:]
        with Image.open(ip) as im:
            w, h = im.size
        gt_coco["images"].append({"id": idx, "file_name": ip.name, "width": w, "height": h})
        gt, gcls = load_gt(DS / "labels/val" / (ip.stem + ".txt"), w, h)
        for (x1, y1, x2, y2), c in zip(gt, gcls):
            ann_id += 1
            gt_coco["annotations"].append({
                "id": ann_id, "image_id": idx, "category_id": int(c) + 1,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0,
            })
        r = model.predict(str(ip), imgsz=a.imgsz, conf=CONF, iou=0.7, device=a.device,
                          nms=None if a.head == "o2m" else False, verbose=False)[0]
        if r.boxes is not None and len(r.boxes):
            for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.cpu().numpy(),
                                              r.boxes.cls.cpu().numpy().astype(int),
                                              r.boxes.conf.cpu().numpy()):
                perf.append({"image_id": idx, "category_id": int(c) + 1, "bbox": [float(x1), float(y1),
                                                                                  float(x2 - x1), float(y2 - y1)],
                             "score": float(s)})
        n_gt += len(gt)
        n_pred += 0 if r.boxes is None else len(r.boxes)
        if idx % 100 == 0:
            print(f"  predict {idx}/{len(files)}", flush=True)

    print(f"GT={n_gt}  pred={n_pred}（conf<={CONF}, head={a.head}）")

    # 逐图做匹配，构造四种理想化变体
    by_img = {i["id"]: [] for i in gt_coco["images"]}
    for p in perf:
        by_img[p["image_id"]].append(p)
    gt_by_img = {i["id"]: [] for i in gt_coco["images"]}
    for an in gt_coco["annotations"]:
        gt_by_img[an["image_id"]].append(an)

    def boxes(an):
        x, y, bw, bh = an["bbox"]
        return [x, y, x + bw, y + bh]

    v_loc, v_cls, v_fn, v_fp, v_all = [], [], [], [], []
    stat = dict(gt_hit=0, pred_hit=0)
    miss_size, miss_cls, all_size = {}, {}, {}
    BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]

    def bucket(s):
        for i, (lo, hi) in enumerate(BUCKETS):
            if lo <= s < hi:
                return i
        return len(BUCKETS) - 1

    for idx in by_img:
        pr = by_img[idx]
        gts = gt_by_img[idx]
        gt_xyxy = np.array([boxes(an) for an in gts], dtype=float).reshape(-1, 4)
        gcls = np.array([an["category_id"] - 1 for an in gts], dtype=int)
        pxyxy = np.array([p["bbox"] for p in pr], dtype=float)
        if len(pxyxy):
            pxyxy = np.stack([pxyxy[:, 0], pxyxy[:, 1], pxyxy[:, 0] + pxyxy[:, 2], pxyxy[:, 1] + pxyxy[:, 3]], 1)
        pcls = np.array([p["category_id"] - 1 for p in pr], dtype=int)
        pconf = np.array([p["score"] for p in pr], dtype=float)

        gt_hit, pr_hit = match_same_class(gt_xyxy, gcls, pxyxy, pcls, pconf)
        _, pr_to_gt = match_any_class(gt_xyxy, pxyxy, pconf)
        stat["gt_hit"] += int(gt_hit.sum())
        stat["pred_hit"] += int(pr_hit.sum())
        # 漏检的尺寸/类别分布（判断"漏在哪"）
        for gi in range(len(gt_xyxy)):
            bw_, bh_ = gt_xyxy[gi, 2] - gt_xyxy[gi, 0], gt_xyxy[gi, 3] - gt_xyxy[gi, 1]
            b = bucket(float(np.sqrt(max(bw_, 0) * max(bh_, 0))))
            all_size[b] = all_size.get(b, 0) + 1
            if not gt_hit[gi]:
                miss_size[b] = miss_size.get(b, 0) + 1
                miss_cls[int(gcls[gi])] = miss_cls.get(int(gcls[gi]), 0) + 1

        for pi, p in enumerate(pr):
            loc = dict(p)
            cls = dict(p)
            if pr_hit[pi]:  # 定位 oracle：同类命中 -> 换 GT 框
                gi = int(np.argmax(np.where(gcls == pcls[pi], iou_matrix(pxyxy[pi : pi + 1], gt_xyxy)[0], -1)))
                loc["bbox"] = gts[gi]["bbox"]
            if pr_to_gt[pi] >= 0:  # 分类 oracle：任意命中 -> 换 GT 类别
                cls["category_id"] = int(gcls[pr_to_gt[pi]]) + 1
            v_loc.append(loc)
            v_cls.append(cls)
            if pr_hit[pi]:
                v_fp.append(p)  # 误检 oracle：只保留命中
        for gi, an in enumerate(gts):  # 漏检 oracle：漏掉的 GT 补高分完美框
            if not gt_hit[gi]:
                v_fn.append({"image_id": idx, "category_id": int(gcls[gi]) + 1, "bbox": an["bbox"], "score": 0.999})
        v_fn.extend(pr)

    # all-oracle：只用"全部 GT 的完美预测"，不含任何真实预测（自检应接近 1.0）
    for an in gt_coco["annotations"]:
        v_all.append({"image_id": an["image_id"], "category_id": an["category_id"],
                      "bbox": an["bbox"], "score": 0.999})

    rows = [("real（真实）", perf), ("loc-oracle（定位完美）", v_loc), ("cls-oracle（分类完美）", v_cls),
            ("fn-oracle（不漏检）", v_fn), ("fp-oracle（无误检）", v_fp), ("all-oracle（全完美，自检）", v_all)]
    res = {}
    for tag, preds in rows:
        ap, ap50 = evaluate(gt_coco, preds, tag)
        res[tag] = (ap, ap50)
        print(f"{tag:26s} AP50-95={ap:.4f}  AP50={ap50:.4f}")

    base = res["real（真实）"][0]
    summary = {
        "weights": a.weights, "head": a.head, "conf": CONF, "n_gt": n_gt, "n_pred": n_pred,
        "recall_at_conf001": stat["gt_hit"] / max(n_gt, 1), "precision_like": stat["pred_hit"] / max(n_pred, 1),
        "oracle": {k: {"AP": v[0], "AP50": v[1], "gain_vs_real": v[0] - base} for k, v in res.items()},
    }
    (out / f"oracle_{a.head}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nconf=0.001 下的召回: {summary['recall_at_conf001']:.3f}")
    print("\n漏检的尺寸分布（原图像素 sqrt(area)）:")
    print(f"{'尺寸':>10} {'GT数':>8} {'漏检数':>8} {'漏检率':>8} {'占全部漏检':>10}")
    tot_miss = sum(miss_size.values())
    for i, (lo, hi) in enumerate(BUCKETS):
        t, m = all_size.get(i, 0), miss_size.get(i, 0)
        lab = f"{lo}-{hi}" if hi < 1e9 else f">={lo}"
        print(f"{lab:>10} {t:8d} {m:8d} {m / max(t, 1):8.3f} {m / max(tot_miss, 1):9.1%}")
    print("\n漏检的类别分布 TOP:")
    for c, m in sorted(miss_cls.items(), key=lambda x: -x[1])[:6]:
        print(f"  {NAMES[c]:16s} {m:6d}  ({m / max(tot_miss, 1):.1%} of 漏检)")
    print(f"saved {out / f'oracle_{a.head}.json'}")


if __name__ == "__main__":
    main()
