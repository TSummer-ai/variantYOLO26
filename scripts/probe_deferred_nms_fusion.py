#!/usr/bin/env python
"""延迟 NMS 融合（Deferred-NMS Fusion）：把视角内的 NMS 推迟到融合之后只做一次。

背景与措辞（重要）：
  本仓库的 checkpoint（以及官方 yolo26n.pt）**缺 `_end2end` 标记**，`head.end2end == False`，
  因此 `predict(nms=False)` **并不是 NMS-free(o2o) 头**，而是"o2m 头 + 关闭 NMS"
  （见 README「重大更正」与 docs/RESULTS_DUALHEAD.md）。本脚本正是**刻意使用**这个语义：
  视角内不做 NMS，把候选全部留给融合阶段，融合后用一次按类 NMS 统一收敛。

动机：
  尺寸门控融合在每个视角内部各做一次 NMS（iou=0.7），会把跨视角本可互相补全的候选提前删掉；
  把 NMS 推迟到融合之后，候选池更大，融合那一次 NMS 能在"全局最优"的意义上取舍。

四组对照（每组都是每图 4 次前向：整图 1 次 + 3×1 切片 3 次，**前向成本完全相同**）：
  A 尺寸门控融合（视角内 NMS）                      = 仓库此前的推理期方法
  B A + 跨视角一致性重打分                          = MV-Fuse（docs/RESULTS_MVFUSE.md）
  C 尺寸门控融合（延迟 NMS）
  D C + 跨视角一致性重打分

用法:
    python probe_deferred_nms_fusion.py --weights weights/yolo26n-visdrone-p2-960.pt --imgsz 960 --device 0
"""
from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced, tile_starts  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
W = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]   # 支持视角数 -> 分数权重（与 probe_mvfuse.py 一致）
BIG_PX = 48.0
OVERLAP = 0.3
MERGE_IOU = 0.6


def predict_full_nms(model, path, imgsz, conf, iou, device, max_det=3000):
    """整图视角、视角内 NMS。

    必须**显式**传 nms=True：同一进程里调用过一次 predict(nms=False) 之后，
    再调用不带 nms 参数的 predict 会继承"关闭 NMS"的状态（实测 1665 -> 3000 -> 3000），
    只有显式 nms=True 才能恢复。
    """
    r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      nms=True, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


def predict_tiles_nms(model, path, imgsz, conf, iou, device, grid, overlap, max_det=3000):
    """切片视角、视角内 NMS（同样必须显式 nms=True）。"""
    return _predict_tiles(model, path, imgsz, conf, iou, device, grid, overlap, max_det,
                          nms_arg=True)


def predict_full_noms(model, path, imgsz, conf, iou, device, max_det=3000):
    """整图视角、关闭 NMS（注意：不是 NMS-free 头，见文件头说明）。"""
    r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      nms=False, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


def predict_tiles_noms(model, path, imgsz, conf, iou, device, grid, overlap, max_det=3000):
    """切片视角、关闭 NMS（逐块），坐标映射回原图。"""
    return _predict_tiles(model, path, imgsz, conf, iou, device, grid, overlap, max_det,
                          nms_arg=False)


def _predict_tiles(model, path, imgsz, conf, iou, device, grid, overlap, max_det, nms_arg):
    im = Image.open(path).convert("RGB")
    Wd, Hd = im.size
    nx, ny = (grid, grid) if isinstance(grid, int) else grid
    xs, tw = tile_starts(Wd, nx, overlap)
    ys, th = tile_starts(Hd, ny, overlap)
    bs, ss, cs = [], [], []
    for y in ys:
        for x in xs:
            crop = np.array(im.crop((x, y, min(x + tw, Wd), min(y + th, Hd))))
            r = model.predict(crop, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                              nms=nms_arg, device=device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.cpu().numpy().copy()
            b[:, [0, 2]] += x
            b[:, [1, 3]] += y
            bs.append(b); ss.append(r.boxes.conf.cpu().numpy())
            cs.append(r.boxes.cls.cpu().numpy().astype(int))
    if not bs:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(bs), np.concatenate(ss), np.concatenate(cs)


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.prod(a[:, 2:] - a[:, :2], 1)[:, None]
    ab = np.prod(b[:, 2:] - b[:, :2], 1)[None, :]
    return inter / np.maximum(aa + ab - inter, 1e-9)


def nviews(b, c, fb, fs, fc, tb, ts, tc):
    """每个候选被几个视角支持（同类且 IoU>=0.5）。"""
    nv = np.zeros(len(b), int)
    for vb, vs, vc in ((fb, fs, fc), (tb, ts, tc)):
        m = vs >= 0.001
        vb2, vc2 = vb[m], vc[m]
        if len(vb2) == 0:
            continue
        iou = _iou(b, vb2)
        for j in range(len(b)):
            same = vc2 == c[j]
            if same.any() and iou[j][same].max() >= 0.5:
                nv[j] += 1
    return nv


def to_coco(sets):
    out = []
    for i, (b, s, c) in enumerate(sets):
        for j in range(len(s)):
            out.append({"image_id": i, "category_id": int(c[j]) + 1,
                        "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]),
                                 float(b[j, 3] - b[j, 1])], "score": float(s[j])})
    return out


def restrict(coco, ids):
    k = set(ids)
    return {**coco, "images": [x for x in coco["images"] if x["id"] in k],
            "annotations": [x for x in coco["annotations"] if x["image_id"] in k]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights/yolo26n-visdrone-p2-960.pt"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/deferred_nms.json"))
    a = ap.parse_args()

    if not (DS / "val.txt").exists():
        raise SystemExit(f"找不到数据集标注: {DS / 'val.txt'}，请先按 README「数据准备」准备数据。")
    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    coco = build_gt(files)

    from ultralytics import YOLO
    model = YOLO(a.weights)

    def gate(tb, ts, tc):
        if len(tb) == 0:
            return tb, ts, tc
        sz = np.sqrt(np.clip(tb[:, 2] - tb[:, 0], 0, None) * np.clip(tb[:, 3] - tb[:, 1], 0, None))
        k = sz < BIG_PX
        return tb[k], ts[k], tc[k]

    A, B, C, D = [], [], [], []
    for i, line in enumerate(files):
        p = DS / line[2:]
        # --- 视角内 NMS（显式 nms=True，避免被上一张图的 nms=False 污染）---
        fb, fs, fc = predict_full_nms(model, p, a.imgsz, a.conf, a.iou, a.device)
        tb, ts, tc = predict_tiles_nms(model, p, a.imgsz, a.conf, a.iou, a.device, (3, 1), OVERLAP)
        tb, ts, tc = merge_nms(tb, ts, tc, MERGE_IOU)
        tb1, ts1, tc1 = gate(tb, ts, tc)
        gb, gs, gc = merge_nms(np.concatenate([fb, tb1]), np.concatenate([fs, ts1]),
                               np.concatenate([fc, tc1]), MERGE_IOU)
        A.append((gb, gs, gc))
        nv = nviews(gb, gc, fb, fs, fc, tb, ts, tc)
        B.append((gb, gs * np.array([W[min(int(x), len(W) - 1)] for x in nv]), gc))

        # --- 延迟 NMS（视角内不 NMS）---
        nb, ns, nc = predict_full_noms(model, p, a.imgsz, a.conf, a.iou, a.device)
        mb, ms, mc = predict_tiles_noms(model, p, a.imgsz, a.conf, a.iou, a.device, (3, 1), OVERLAP)
        mb, ms, mc = merge_nms(mb, ms, mc, MERGE_IOU)
        mb1, ms1, mc1 = gate(mb, ms, mc)
        db, ds, dc = merge_nms(np.concatenate([nb, mb1]), np.concatenate([ns, ms1]),
                               np.concatenate([nc, mc1]), MERGE_IOU)
        C.append((db, ds, dc))
        nv2 = nviews(db, dc, nb, ns, nc, mb, ms, mc)
        D.append((db, ds * np.array([W[min(int(x), len(W) - 1)] for x in nv2]), dc))
        if i % 50 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    n = len(files)
    hold = sorted(set(range(n)) - set(np.random.RandomState(0).permutation(n)[: n // 2].tolist()))

    def ev(sets, ids=None):
        if ids is None:
            return evaluate(coco, to_coco(sets))
        preds = [sets[i] if i in set(ids) else (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for i in range(n)]
        return evaluate(restrict(coco, ids), to_coco(preds))

    res = {}
    print(f"\n{'方案':>34}{'全量':>17}{'留出半集':>19}")
    for nm, s in [("A 尺寸门控融合（视角内 NMS）", A), ("B A + 一致性重打分（MV-Fuse）", B),
                  ("C 尺寸门控融合（延迟 NMS）", C), ("D C + 一致性重打分", D)]:
        f95, f50 = ev(s)
        h95, h50 = ev(s, hold)
        res[nm] = {"full": [f95, f50], "hold": [h95, h50]}
        print(f"{nm:>34}{f95:>9.4f}/{f50:<7.4f}{h95:>9.4f}/{h50:<8.4f}", flush=True)

    b = res["B A + 一致性重打分（MV-Fuse）"]["full"][0]
    d = res["D C + 一致性重打分"]["full"][0]
    print(f"\nD vs B（MV-Fuse）：{d - b:+.4f} AP50-95（全量）；"
          f"{res['D C + 一致性重打分']['hold'][0] - res['B A + 一致性重打分（MV-Fuse）']['hold'][0]:+.4f}（留出）")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": a.weights, "imgsz": a.imgsz, "n_images": n, "results": res},
                              ensure_ascii=False, indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
