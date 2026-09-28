#!/usr/bin/env python
"""VisDrone 基线的错误分析：目标尺寸分布 / 分尺寸召回 / 分类混淆 / e2e(one2one) vs NMS(one2many) 差距。

用法（GPU，需放宽沙箱权限）：
    python analyze_visdrone.py --weights runs/visdrone/base_n_640/weights/best.pt --imgsz 640
输出：
    runs/visdrone/analysis/summary.md, size_recall.png, per_class.csv
"""

from __future__ import annotations

import os

import argparse
import csv
import gc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
DS = ROOT / "datasets/visdrone"
NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
BUCKETS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]


def load_gt(label_file: Path, w: int, h: int):
    """读 YOLO 标签 -> 原图像素坐标 xyxy + 类别。"""
    if not label_file.exists():
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    rows = np.array([ln.split() for ln in open(label_file) if ln.strip()], dtype=float)
    if rows.size == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int)
    cls = rows[:, 0].astype(int)
    cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return xyxy, cls


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.prod(a[:, 2:] - a[:, :2], 1)[:, None]
    area_b = np.prod(b[:, 2:] - b[:, :2], 1)[None, :]
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def greedy_match(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, iou_thr=0.5):
    """按置信度贪心一对一匹配，返回 (tp_gt_idx, fp_idx, per_gt_matched)。"""
    matched = np.zeros(len(gt_xyxy), dtype=bool)
    fp = []
    if len(pr_xyxy) == 0:
        return matched, fp
    order = np.argsort(-pr_conf)
    ious = iou_matrix(pr_xyxy, gt_xyxy)
    for pi in order:
        if len(gt_xyxy) == 0:
            fp.append(pi); continue
        row = ious[pi].copy()
        row[matched] = -1                      # 已匹配的 GT 不能再占
        row[gt_cls != pr_cls[pi]] = -1         # 类别必须一致（非 class-agnostic）
        gi = int(np.argmax(row))
        if row[gi] >= iou_thr:
            matched[gi] = True
        else:
            fp.append(pi)
    return matched, fp


def bucket_of(size):
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= size < hi:
            return i
    return len(BUCKETS) - 1


def run(weights: str, imgsz: int, conf: float, iou: float, device: str, nms: bool | None, limit: int = 0):
    """对 val 全量跑一遍预测并统计分尺寸/分类别召回与误检混淆（分块预测，避免显存随图片数线性膨胀）。"""
    tag = "e2e" if nms is False else "nms"
    model = YOLO(weights)
    im_files = [l.strip() for l in open(DS / "val.txt")]
    if limit:
        im_files = im_files[:limit]

    size_tot = Counter()          # GT 数量（原图像素）
    size_hit = Counter()          # 命中的 GT 数量
    cls_tot, cls_hit = Counter(), Counter()
    conf_mat = Counter()          # (gt_cls, pred_cls) 的 FP 混淆
    n_fp = 0
    n_gt = 0

    chunk_size = 32  # 单次 predict 传入的图片数；整表一次传入会让显存随图片数线性增长
    for start in range(0, len(im_files), chunk_size):
        chunk = im_files[start : start + chunk_size]
        results = model.predict(
            [str(DS / p[2:]) for p in chunk], imgsz=imgsz, conf=conf, iou=iou,
            device=device, nms=nms, batch=1, rect=False, stream=True, verbose=False,
        )
        for p, r in zip(chunk, results):
            ip = DS / p[2:]
            with Image.open(ip) as im:
                w, h = im.size
            gt_xyxy, gt_cls = load_gt(DS / "labels/val" / (ip.stem + ".txt"), w, h)
            if r.boxes is None or len(r.boxes) == 0:
                pr_xyxy = np.zeros((0, 4)); pr_cls = np.zeros((0,), int); pr_conf = np.zeros(0)
            else:
                pr_xyxy = r.boxes.xyxy.cpu().numpy()
                pr_cls = r.boxes.cls.cpu().numpy().astype(int)
                pr_conf = r.boxes.conf.cpu().numpy()
            matched, fp = greedy_match(gt_xyxy, gt_cls, pr_xyxy, pr_cls, pr_conf, iou_thr=0.5)
            n_gt += len(gt_xyxy); n_fp += len(fp)
            for i in range(len(gt_xyxy)):
                s = float(np.sqrt(np.prod(gt_xyxy[i, 2:] - gt_xyxy[i, :2])))
                b = bucket_of(s)
                size_tot[b] += 1; size_hit[b] += int(matched[i])
                cls_tot[int(gt_cls[i])] += 1; cls_hit[int(gt_cls[i])] += int(matched[i])
            for pi in fp:
                # 找 IoU 最大的 GT 作为"本该是谁"，统计混淆
                if len(gt_xyxy):
                    j = int(np.argmax(iou_matrix(pr_xyxy[pi : pi + 1], gt_xyxy)[0]))
                    conf_mat[(int(gt_cls[j]), int(pr_cls[pi]))] += 1
                else:
                    conf_mat[(-1, int(pr_cls[pi]))] += 1
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  [{tag}] {min(start + chunk_size, len(im_files))}/{len(im_files)}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(
        tag=tag, size_tot=size_tot, size_hit=size_hit, cls_tot=cls_tot,
        cls_hit=cls_hit, conf_mat=conf_mat, n_fp=n_fp, n_gt=n_gt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/base_n_640/weights/best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/analysis"))
    ap.add_argument("--limit", type=int, default=0, help="只分析前 N 张（0=全部），用于冒烟测试")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {
        "nms": run(a.weights, a.imgsz, a.conf, a.iou, a.device, None, a.limit),
        "e2e": run(a.weights, a.imgsz, a.conf, a.iou, a.device, False, a.limit),
    }

    lines = [
        "# VisDrone 基线错误分析",
        "",
        f"- 权重: `{a.weights}`｜imgsz={a.imgsz}｜conf={a.conf}｜IoU(匹配)={0.5}｜NMS IoU={a.iou}",
        "",
        "## 1. 分尺寸召回（原图像素，√area）",
        "",
        "| 尺寸(px) | GT 数 | NMS 召回 | e2e 召回 | 召回差 |",
        "|---|---|---|---|---|",
    ]
    for i, (lo, hi) in enumerate(BUCKETS):
        label = f"{lo}-{hi}" if hi < 1e9 else f">={lo}"
        t = res["nms"]["size_tot"].get(i, 0)
        r1 = res["nms"]["size_hit"].get(i, 0) / t if t else 0
        r2 = res["e2e"]["size_hit"].get(i, 0) / t if t else 0
        lines.append(f"| {label} | {t} | {r1:.3f} | {r2:.3f} | {r2 - r1:+.3f} |")

    lines += ["", "## 2. 分类别召回", "", "| 类别 | GT 数 | NMS 召回 | e2e 召回 |", "|---|---|---|---|"]
    with open(out / "per_class.csv", "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["cls", "name", "gt", "recall_nms", "recall_e2e"])
        for c in range(len(NAMES)):
            t = res["nms"]["cls_tot"].get(c, 0)
            if not t:
                continue
            r1 = res["nms"]["cls_hit"].get(c, 0) / t
            r2 = res["e2e"]["cls_hit"].get(c, 0) / t
            lines.append(f"| {NAMES[c]} | {t} | {r1:.3f} | {r2:.3f} |")
            wcsv.writerow([c, NAMES[c], t, f"{r1:.4f}", f"{r2:.4f}"])

    lines += [
        "",
        "## 3. 误检/混淆 TOP",
        "",
        f"- NMS 头 FP 数: {res['nms']['n_fp']}｜e2e 头 FP 数: {res['e2e']['n_fp']}",
        "",
        "| 真值类别 -> 预测类别 | 次数 |",
        "|---|---|",
    ]
    for (gc, pc), n in res["nms"]["conf_mat"].most_common(10):
        gname = NAMES[gc] if gc >= 0 else "(背景/无对应)"
        lines.append(f"| {gname} -> {NAMES[pc]} | {n} |")

    lines += [
        "",
        "## 4. 结论（自动生成，人工复核）",
        "",
        f"- 总 GT: {res['nms']['n_gt']}",
        f"- NMS 头召回: {sum(res['nms']['size_hit'].values()) / max(res['nms']['n_gt'], 1):.3f}"
        f"｜e2e 头召回: {sum(res['e2e']['size_hit'].values()) / max(res['e2e']['n_gt'], 1):.3f}",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 画图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = [f"{lo}-{hi}" if hi < 1e9 else f">={lo}" for lo, hi in BUCKETS]
        r1 = [res["nms"]["size_hit"].get(i, 0) / max(res["nms"]["size_tot"].get(i, 0), 1) for i in range(len(BUCKETS))]
        r2 = [res["e2e"]["size_hit"].get(i, 0) / max(res["e2e"]["size_tot"].get(i, 0), 1) for i in range(len(BUCKETS))]
        x = np.arange(len(labels))
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - 0.2, r1, 0.4, label="one2many + NMS")
        plt.bar(x + 0.2, r2, 0.4, label="one2one (e2e, nms=False)")
        plt.xticks(x, labels)
        plt.xlabel("GT size sqrt(area) px (original image)")
        plt.ylabel("recall@IoU0.5")
        plt.title("VisDrone val: recall by object size")
        plt.legend(); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
        plt.savefig(out / "size_recall.png", dpi=140)
        print("saved", out / "size_recall.png")
    except Exception as e:  # 画图失败不影响结论
        print("plot failed:", e)


if __name__ == "__main__":
    main()
