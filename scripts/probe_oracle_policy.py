#!/usr/bin/env python
"""per-image oracle 策略 + 精度-算力 Pareto 前沿（零训练）。

我们 14 次实验的唯一规律：唯一有效的杠杆是"每个目标拿到多少像素"
（640→960 +6.5 AP；P2 stride-4 +1.6；切片 +2.38）。但买像素很贵（切片 4× 前向），
且切片伤大目标（≥64px 召回 −4.8）。

于是真正的问题是：**在固定算力下把像素花在哪里** —— 本文测量它的上界：
  1. 每个配置的全局 mAP 与算力 → 固定配置的 Pareto 前沿
  2. per-image oracle（用标签为每张图选最优配置）→ oracle 上界与 gap
  3. 用"推理时零成本统计量"（整图 pass 的小框数）做的一参数规则 → 能拿到多少 oracle gap

文献位置（两轮独立探针）：DETR 侧有自适应 query（DQA-DETR/D3Q/RDAQ），但
**没有人把"给定算力预算下选择分辨率/切片布局以最大化 mAP"表述为约束优化，也没有 CNN 侧的精度-算力 Pareto 前沿**。

用法：
  python probe_oracle_policy.py --stage cache   --out oracle_policy_cache
  python probe_oracle_policy.py --stage analyze --out oracle_policy_cache
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced  # noqa: E402

ROOT = Path(__import__("os").environ.get("YOLO_ROOT", Path(__file__).resolve().parent))
CONF = 0.001
IOU = 0.7
OVERLAP = 0.3   # 之前调参实测：overlap 0.3 优于 0.2
BIG_PX = 48.0          # 尺寸感知融合：≥48px 的框只信整图（沿用之前的最优设置）
SMALL_STAT = 16.0      # 零成本统计量：整图 pass 中小于该尺寸的框数

# (模型, 输入分辨率, 布局) —— 算力单位 = 前向次数 × (imgsz/640)²
CONFIGS = [
    ("p2_640", 640, "full"), ("p2_640", 640, "2x1"), ("p2_640", 640, "3x1"), ("p2_640", 640, "2x2"),
    ("p2_960", 960, "full"), ("p2_960", 960, "2x1"), ("p2_960", 960, "3x1"), ("p2_960", 960, "2x2"),
]
GRID = {"full": None, "2x1": (2, 1), "3x1": (3, 1), "2x2": (2, 2)}
NFWD = {"full": 1, "2x1": 3, "3x1": 4, "2x2": 5}
WEIGHTS = {"p2_640": ROOT / "runs/visdrone/p2_n_640/weights/best.pt",
           "p2_960": ROOT / "runs/visdrone/p2_960/weights/best.pt"}


def cost(cfg) -> float:
    _, imgsz, layout = cfg
    return NFWD[layout] * (imgsz / 640.0) ** 2


def name(cfg) -> str:
    m, s, l = cfg
    return f"{m}@{s}/{l}"


def size_aware_fuse(full, tiles, merge_iou=0.6):
    """文档规定的规则：预测框 ≥ BIG_PX 时只信整图；< BIG_PX 时整图与切片**两者都保留**，最后统一去重。"""
    fb, fs, fc = full
    tb, ts, tc = tiles
    if len(fb) == 0:
        return tb, ts, tc
    if len(tb) == 0:
        return fb, fs, fc
    tsz = np.sqrt(np.clip(tb[:, 2] - tb[:, 0], 0, None) * np.clip(tb[:, 3] - tb[:, 1], 0, None))
    keep_t = tsz < BIG_PX               # 切片只贡献小框（大框可能被切断）
    b = np.concatenate([fb, tb[keep_t]])
    s = np.concatenate([fs, ts[keep_t]])
    c = np.concatenate([fc, tc[keep_t]])
    return merge_nms(b, s, c, iou_thr=merge_iou)


def load_gt_xyxy(line):
    p = DS / line[2:]
    with Image.open(p) as im:
        W, H = im.size
    lp = DS / "labels/val" / (p.stem + ".txt")
    g, gc = [], []
    if lp.exists():
        for ln in open(lp):
            c, x, y, bw, bh = ln.split()
            c, x, y, bw, bh = int(c), float(x), float(y), float(bw), float(bh)
            g.append([(x - bw / 2) * W, (y - bh / 2) * H, (x + bw / 2) * W, (y + bh / 2) * H])
            gc.append(c)
    return (np.array(g, float).reshape(-1, 4), np.array(gc, int), W, H)


def to_coco(pred_list):
    out = []
    for i, (b, s, c) in enumerate(pred_list):
        for j in range(len(s)):
            out.append({"image_id": i, "category_id": int(c[j]) + 1,
                        "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                        "score": float(s[j])})
    return out


def _np_iou(a, b):
    """a: (n,4), b: (m,4) -> (n,m) IoU（纯 numpy，避免 torch 开销）。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = ((a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0))[:, None]
    ab = ((b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0))[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-9)


def per_image_ap(pred_list, gts, iou_thr=0.5):
    """逐图 VOC 式 AP（各类平均，all-point 插值）。与全局 AP 同源，用作 oracle 的选择准则。"""
    out = np.zeros(len(pred_list))
    for i, (b, s, c) in enumerate(pred_list):
        gb, gc = gts[i][0], gts[i][1]
        if len(gb) == 0 or len(s) == 0:
            continue
        per_cls = []
        for cl in np.unique(gc):
            g = gb[gc == cl]
            m = c == cl
            if not m.any():
                per_cls.append(0.0); continue
            pb, ps = b[m], s[m]
            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            iou = _np_iou(g, pb)
            matched = np.zeros(len(g), bool); tp = np.zeros(len(pb))
            for j in range(len(pb)):
                col = iou[:, j].copy(); col[matched] = -1
                bi = int(np.argmax(col))
                if col[bi] >= iou_thr:
                    matched[bi] = True; tp[j] = 1
            if tp.sum() == 0:
                per_cls.append(0.0); continue
            cum = np.cumsum(tp)
            prec = cum / (np.arange(len(tp)) + 1)
            rec = cum / len(g)
            mr = np.concatenate([[0.0], rec, [1.0]])
            mp = np.concatenate([[0.0], prec, [0.0]])
            for k in range(len(mp) - 2, -1, -1):
                mp[k] = max(mp[k], mp[k + 1])
            idx = np.where(mr[1:] != mr[:-1])[0]
            per_cls.append(float(np.sum((mr[idx + 1] - mr[idx]) * mp[idx + 1])))
        out[i] = float(np.mean(per_cls)) if per_cls else 0.0
    return out


def per_image_scores(pred_list, gts, thr=0.25, iou_thr=0.5):
    """per-image 召回 / F1（在部署阈值下），用于 oracle 逐图选择。"""
    from ultralytics.utils.metrics import box_iou
    import torch
    rec, f1 = np.zeros(len(pred_list)), np.zeros(len(pred_list))
    small_cnt = np.zeros(len(pred_list), int)
    for i, (b, s, c) in enumerate(pred_list):
        gb, gc = gts[i][0], gts[i][1]
        m = s >= thr
        bb, cc = b[m], c[m]
        sz = np.sqrt(np.clip(bb[:, 2] - bb[:, 0], 0, None) * np.clip(bb[:, 3] - bb[:, 1], 0, None)) if len(bb) else np.zeros(0)
        small_cnt[i] = int((sz < SMALL_STAT).sum())
        if len(gb) == 0:
            rec[i], f1[i] = 0.0, 0.0
            continue
        if len(bb) == 0:
            rec[i], f1[i] = 0.0, 0.0
            continue
        iou = box_iou(torch.from_numpy(gb).float(), torch.from_numpy(bb).float()).numpy()
        tp = 0
        for gi in range(len(gb)):
            same = cc == gc[gi]
            if same.any() and iou[gi][same].max() >= iou_thr:
                tp += 1
        rec[i] = tp / len(gb)
        f1[i] = 2 * tp / (len(gb) + len(bb)) if (len(gb) + len(bb)) else 0.0
    return rec, f1, small_cnt


def do_cache(args):
    global OVERLAP
    OVERLAP = args.overlap
    files = [l.strip() for l in open(DS / "val.txt")]
    if args.limit:
        files = files[: args.limit]
    outdir = Path(args.out) / f"n{len(files)}"
    outdir.mkdir(parents=True, exist_ok=True)
    models = {}
    for cfg in CONFIGS:
        mkey, imgsz, layout = cfg
        f = outdir / f"{name(cfg).replace('@','_').replace('/','_')}.pkl"
        if f.exists():
            print(f"  [skip] {name(cfg)} 已缓存")
            continue
        if mkey not in models:
            from ultralytics import YOLO
            models[mkey] = YOLO(str(WEIGHTS[mkey]))
        model = models[mkey]
        full_needed = layout != "full"
        fullf = outdir / f"{name((mkey, imgsz, 'full')).replace('@','_').replace('/','_')}.pkl"
        full_cache = pickle.load(open(fullf, "rb")) if (full_needed and fullf.exists()) else None
        preds = []
        for i, line in enumerate(files):
            p = DS / line[2:]
            if layout == "full":
                preds.append(predict_full(model, p, imgsz, CONF, IOU, args.device))
            else:
                tiles = predict_sliced(model, p, imgsz, CONF, IOU, args.device, GRID[layout], OVERLAP)
                tiles = merge_nms(*tiles, iou_thr=0.6)
                full = full_cache[i] if full_cache is not None else predict_full(model, p, imgsz, CONF, IOU, args.device)
                preds.append(size_aware_fuse(full, tiles))
            if i % 100 == 0:
                print(f"  [{name(cfg)}] {i}/{len(files)}", flush=True)
        pickle.dump(preds, open(f, "wb"))
        print(f"  ✅ {name(cfg)} 缓存完成 (cost={cost(cfg):.2f})")
    return files, outdir


def do_analyze(args, files, outdir):
    gts = [load_gt_xyxy(l) for l in files]
    coco_gt = build_gt(files)
    P, S, C, APi, SMALL = {}, {}, {}, {}, None
    for cfg in CONFIGS:
        f = outdir / f"{name(cfg).replace('@','_').replace('/','_')}.pkl"
        if not f.exists():
            print(f"  ⚠️ 缺少 {name(cfg)}"); continue
        P[cfg] = pickle.load(open(f, "rb"))
        S[cfg] = cost(cfg)
        C[cfg], _f1, small = per_image_scores(P[cfg], gts)
        APi[cfg] = per_image_ap(P[cfg], gts)
        if SMALL is None:
            SMALL = small                       # 用 640 整图 pass 的小框数做零成本统计量
        p95, p50 = evaluate(coco_gt, to_coco(P[cfg]))
        print(f"  {name(cfg):>18}  算力 {S[cfg]:>5.2f}   mAP50-95 {p95:.4f}  mAP50 {p50:.4f}   mean召回 {C[cfg].mean():.3f}  逐图AP均值 {APi[cfg].mean():.3f}")

    print("\n=== 1. 固定配置的 Pareto 前沿 ===")
    pts = sorted(((S[c], c) for c in P))
    best = -1
    print(f"  {'配置':>18}{'算力':>8}{'mAP50-95':>12}{'是否前沿':>10}")
    frontier = []
    for sc, c in pts:
        p95, _ = evaluate(coco_gt, to_coco(P[c]))
        on = p95 > best
        if on:
            best = p95; frontier.append(c)
        print(f"  {name(c):>18}{sc:>8.2f}{p95:>12.4f}{'  ✔' if on else '':>10}")

    print("\n=== 2. per-image oracle 上界（用标签逐图选最优）===")
    for crit, scores in (("逐图AP", APi), ("召回", {c: C[c] for c in C})):
        hyb, cost_sum = [], 0.0
        for i in range(len(files)):
            cbest = max(P, key=lambda c: scores[c][i])
            hyb.append(P[cbest][i]); cost_sum += S[cbest]
        p95, p50 = evaluate(coco_gt, to_coco(hyb))
        print(f"  按{crit}选: oracle mAP50-95 {p95:.4f}  mAP50 {p50:.4f}   平均算力 {cost_sum/len(files):.2f}")

    print("\n=== 3. 一参数规则（零成本统计量 = 整图 pass 小框数）===")
    print(f"  统计量分布: 中位 {np.median(SMALL):.0f}  P25 {np.percentile(SMALL,25):.0f}  P75 {np.percentile(SMALL,75):.0f}  最大 {SMALL.max()}")
    # 用"逐图最优配置"的分布来挑两个候选（稀疏用一个、密集用一个）
    pick = [max(P, key=lambda c: APi[c][i]) for i in range(len(files))]
    from collections import Counter
    top2 = [c for c, _ in Counter(pick).most_common(2)]
    print(f"  oracle 最常选的配置: " + ", ".join(f"{name(c)}({n})" for c, n in Counter(pick).most_common(4)))
    if len(top2) == 2:
        a, b = sorted(top2, key=lambda c: S[c])
        print(f"  一参数规则: 小框数 < T 用 {name(a)} (算力{S[a]:.2f})，否则用 {name(b)} (算力{S[b]:.2f})")
        print(f"  {'阈值T':>8}{'触发率':>9}{'平均算力':>10}{'mAP50-95':>12}{'mAP50':>10}")
        for T in [1, 5, 10, 20, 40, 80, 150, 300, 600, 10**9]:
            hyb, csum = [], 0.0
            for i in range(len(files)):
                c = a if SMALL[i] < T else b
                hyb.append(P[c][i]); csum += S[c]
            p95, p50 = evaluate(coco_gt, to_coco(hyb))
            trig = float((SMALL >= T).mean()) * 100
            print(f"  {T:>8}{trig:>8.0f}%{csum/len(files):>10.2f}{p95:>12.4f}{p50:>10.4f}")
    pickle.dump({"pick": pick, "small": SMALL, "ap_img": APi, "rec": C}, open(outdir / "policy.pkl", "wb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["cache", "analyze", "all"])
    ap.add_argument("--out", default=str(ROOT / "oracle_policy_cache"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlap", type=float, default=0.3)
    a = ap.parse_args()
    files = outdir = None
    if a.stage in ("cache", "all"):
        files, outdir = do_cache(a)
    if a.stage in ("analyze", "all"):
        if files is None:
            files = [l.strip() for l in open(DS / "val.txt")]
            if a.limit:
                files = files[: a.limit]
            outdir = Path(a.out) / f"n{len(files)}"
        do_analyze(a, files, outdir)


if __name__ == "__main__":
    main()
