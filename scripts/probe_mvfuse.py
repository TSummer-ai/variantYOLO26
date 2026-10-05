#!/usr/bin/env python
"""MV-Fuse 主评测脚本：尺寸门控多视角融合 + 跨视角一致性重打分。

方法（零训练、零额外前向）：
  1. 整图 pass 与 3x1 切片 pass 各跑一次（共 5 次前向 @960）
  2. 尺寸门控融合：>=48px 的框只信整图（切片会切断大目标）；<48px 两者都保留，再统一 NMS
  3. **一致性重打分**：对每个候选统计"被几个视角支持"（两路里是否存在 IoU>=0.5 且同类的框），
     按支持数乘权重 W（利用实测规律：支持数越多 TP 率越高，且该分离度随目标变小而增强）

用法: python probe_mvfuse.py --weights <pt> --imgsz 640 --tag base640
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_oracle_policy import _np_iou, load_gt_xyxy, to_coco  # noqa: E402
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced  # noqa: E402

W = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]     # 按支持视角数(0..5)的分数权重（手工设定，未调参）
BIG_PX = 48.0
OVERLAP = 0.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/home/wang/DeepSeek/YOLO/mvfuse_results.json")
    a = ap.parse_args()

    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    coco = build_gt(files)
    from ultralytics import YOLO
    model = YOLO(a.weights)

    full, tiles, gated, gated_c, plain, plain_c = [], [], [], [], [], []
    for i, line in enumerate(files):
        p = DS / line[2:]
        fb, fs, fc = predict_full(model, p, a.imgsz, a.conf, a.iou, a.device)
        tb, ts, tc = predict_sliced(model, p, a.imgsz, a.conf, a.iou, a.device, (3, 1), OVERLAP)
        tb, ts, tc = merge_nms(tb, ts, tc, iou_thr=0.6)
        full.append((fb, fs, fc)); tiles.append((tb, ts, tc))

        def nviews(b, c):
            """每个框被几个视角支持（整图 / 切片）。"""
            nv = np.zeros(len(b), int)
            for vb, vs, vc in ((fb, fs, fc), (tb, ts, tc)):
                m = vs >= 0.001
                vb2, vc2 = vb[m], vc[m]
                if len(vb2) == 0:
                    continue
                iou = _np_iou(b, vb2)
                for j in range(len(b)):
                    same = vc2 == c[j]
                    if same.any() and iou[j][same].max() >= 0.5:
                        nv[j] += 1
            return nv

        # 尺寸门控：大框只信整图；小框两者都保留
        tsz = np.sqrt(np.clip(tb[:, 2] - tb[:, 0], 0, None) * np.clip(tb[:, 3] - tb[:, 1], 0, None)) if len(tb) else np.zeros(0)
        kt = tsz < BIG_PX
        gb, gs, gc = merge_nms(np.concatenate([fb, tb[kt]]), np.concatenate([fs, ts[kt]]),
                               np.concatenate([fc, tc[kt]]), iou_thr=0.6)
        gated.append((gb, gs, gc))
        nv = nviews(gb, gc)
        gated_c.append((gb, gs * np.array([W[min(int(x), len(W) - 1)] for x in nv]), gc))

        # 不做尺寸门控（朴素并集）作为对照
        pb, ps, pc = merge_nms(np.concatenate([fb, tb]), np.concatenate([fs, ts]), np.concatenate([fc, tc]), iou_thr=0.6)
        plain.append((pb, ps, pc))
        nv2 = nviews(pb, pc)
        plain_c.append((pb, ps * np.array([W[min(int(x), len(W) - 1)] for x in nv2]), pc))
        if i % 150 == 0:
            print(f"  [{a.tag}] {i}/{len(files)}", flush=True)

    n = len(files)
    hold = sorted(set(range(n)) - set(np.random.RandomState(0).permutation(n)[: n // 2].tolist()))

    def restrict(c, ids):
        k = set(ids)
        return {**c, "images": [x for x in c["images"] if x["id"] in k],
                "annotations": [x for x in c["annotations"] if x["image_id"] in k]}

    def ev(sets, ids=None):
        if ids is None:
            return evaluate(coco, to_coco(sets))
        preds = [sets[i] if i in set(ids) else (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)) for i in range(n)]
        return evaluate(restrict(coco, ids), to_coco(preds))

    res = {"tag": a.tag, "imgsz": a.imgsz, "weights": a.weights}
    print(f"\n=== {a.tag} @ {a.imgsz} ===")
    print(f"  {'方案':>26}{'全量 548 图':>18}{'留出半集 274 图':>20}")
    variants = [("仅整图", full), ("仅切片", tiles), ("朴素并集", plain),
                ("尺寸门控融合", gated), ("尺寸门控+一致性(本方法)", gated_c), ("朴素并集+一致性", plain_c)]
    for nm, s in variants:
        f95, f50 = ev(s)
        h95, h50 = ev(s, hold)
        res[nm] = {"full": [f95, f50], "hold": [h95, h50]}
        print(f"  {nm:>26}{f95:>9.4f}/{f50:<8.4f}{h95:>10.4f}/{h50:<9.4f}")
    # 记录相对基线
    b = res["仅整图"]["hold"][0]; g = res["尺寸门控融合"]["hold"][0]; m = res["尺寸门控+一致性(本方法)"]["hold"][0]
    print(f"\n  留出半集：本方法 vs 整图 {m-b:+.4f}   vs 尺寸门控融合 {m-g:+.4f}")
    old = json.load(open(a.out)) if Path(a.out).exists() else []
    old = [x for x in old if x.get("tag") != a.tag] + [res]
    json.dump(old, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"  已写入 {a.out}")


if __name__ == "__main__":
    main()
