#!/usr/bin/env python
"""任务4f：用官方 WBF 重算分尺寸召回，并验证"conf_type 只改分数、不改框集合"。

§2.5 的分尺寸结论此前基于自研实现。本脚本：
  1) 在若干图上直接比对官方 `max` 与 `box_and_model_avg` 的输出框集合是否逐位相同
     （若相同，则"只改排序不改召回"是结构性事实，无需实测召回）
  2) 用官方实现给出 union / gated 两种候选池下各 conf_type 的分尺寸召回，供论文引用

用法: python probe_wbf_by_size_official.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_fusion_baselines import DS, build_gt, evaluate, m_nms, nviews, to_preds, W  # noqa: E402
from probe_slicing import BUCKETS, recall_by_bucket  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")


def load_ref():
    spec = importlib.util.spec_from_file_location(
        "wbf_ref", ROOT / "third_party/ensemble_boxes/ensemble_boxes_wbf.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.weighted_boxes_fusion


REF = load_ref()


def norm(b, W_, H_):
    x = b.astype(np.float32).copy()
    if len(x):
        x[:, [0, 2]] = np.clip(x[:, [0, 2]] / W_, 0, 1)
        x[:, [1, 3]] = np.clip(x[:, [1, 3]] / H_, 0, 1)
    return x.reshape(-1, 4)


def denorm(b, W_, H_):
    x = b.astype(np.float32).copy()
    if len(x):
        x[:, [0, 2]] *= W_; x[:, [1, 3]] *= H_
    return x.reshape(-1, 4)


def ref_wbf(it, size, conf_type, gated, iou_thr=0.6):
    W_, H_ = size
    fb, fs, fc, tb, ts, tc = it[0], it[1], it[2], it[3], it[4], it[5]
    if gated and len(tb):
        k = np.sqrt((tb[:, 2] - tb[:, 0]).clip(0) * (tb[:, 3] - tb[:, 1]).clip(0)) < 48.0
        tb, ts, tc = tb[k], ts[k], tc[k]
    b, s, l = REF([norm(fb, W_, H_), norm(tb, W_, H_)],
                  [fs.astype(np.float32), ts.astype(np.float32)],
                  [fc.astype(np.float32), tc.astype(np.float32)],
                  weights=None, iou_thr=iou_thr, skip_box_thr=0.0,
                  conf_type=conf_type, allows_overflow=False)
    return denorm(b, W_, H_), s, l.astype(int)


def main():
    z = np.load(ROOT / "fusion_cache_full.npz", allow_pickle=True)
    data = list(z["cache"])
    files = [l.strip() for l in open(DS / "val.txt")]
    sizes = []
    for l in files:
        with Image.open(DS / l[2:]) as im:
            sizes.append(im.size)
    coco_gt = build_gt(files)
    print(f"图数 {len(data)}", flush=True)

    # ---- 1. 结构性验证：conf_type 只改分数？ ----
    print("\n=== 1. 验证 conf_type 是否只改分数（前 30 图逐位比对框）===", flush=True)
    same = 0
    for i in range(min(30, len(data))):
        b1, s1, l1 = ref_wbf(data[i], sizes[i], "max", gated=False)
        b2, s2, l2 = ref_wbf(data[i], sizes[i], "box_and_model_avg", gated=False)
        ok = (len(b1) == len(b2) and np.array_equal(l1, l2)
              and np.allclose(b1, b2, atol=1e-6))
        same += int(ok)
        if not ok:
            print(f"  图 {i}: 不一致  n={len(b1)} vs {len(b2)}", flush=True)
    print(f"  逐位相同的图数: {same}/30  ⇒ "
          f"{'只改分数、不改框集合（结构性成立）' if same == 30 else '存在差异，需实测'}", flush=True)

    # ---- 2. 官方实现的分尺寸召回 ----
    variants = {
        "union+NMS": lambda it, sz: m_nms(*__import__('probe_fusion_baselines').union_pool(*it[:6]), 0.6),
        "union+WBF[官方,max]": lambda it, sz: ref_wbf(it, sz, "max", False),
        "union+WBF[官方,box_and_model_avg]": lambda it, sz: ref_wbf(it, sz, "box_and_model_avg", False),
        "gated+WBF[官方,box_and_model_avg]": lambda it, sz: ref_wbf(it, sz, "box_and_model_avg", True),
        "gated+WBF[官方,max]": lambda it, sz: ref_wbf(it, sz, "max", True),
    }

    def mv(it, sz):
        from probe_fusion_baselines import gated_pool
        gb, gs, gc = gated_pool(*it[:6])
        gb, gs, gc = m_nms(gb, gs, gc, 0.6)
        nv = nviews(gb, gc, ((it[0], it[1], it[2]), (it[3], it[4], it[5])))
        return gb, gs * np.array([W[min(int(x), len(W) - 1)] for x in nv]), gc

    variants["MV-Fuse(门控+一致性)"] = mv

    res = {}
    per_image_all = {}
    print("\n=== 2. 官方实现的分尺寸召回 ===", flush=True)
    for tag, fn in variants.items():
        preds, per_image = [], []
        for i, it in enumerate(data):
            b, s, c = fn(it, sizes[i])
            per_image.append((b, s, c))
            to_preds(b, s, c, i, preds)
        p95, p50 = evaluate(coco_gt, preds)
        per_image_all[tag] = per_image
        res[tag] = {"ap50_95": round(p95, 4), "ap50": round(p50, 4)}
        print(f"  {tag:<38} AP50-95={p95:.4f}", flush=True)

    rec = {t: recall_by_bucket(files, per_image_all[t]) for t in variants}
    names = [b[0] for b in BUCKETS]
    tbl = {nm: {"gt": rec[list(variants)[0]][nm][0],
                "recall": {t: (round(rec[t][nm][1] / rec[t][nm][0], 4) if rec[t][nm][0] else None)
                           for t in variants}} for nm in names}
    res["_by_size"] = tbl
    res["_conf_type_same_boxes_30img"] = f"{same}/30"

    hdr = "".join(f"{t:>34}" for t in variants)
    print(f"\n{'尺寸':>8}{'GT数':>8}{hdr}")
    for nm in names:
        if not tbl[nm]["gt"]:
            continue
        print(f"{nm:>8}{tbl[nm]['gt']:>8}" +
              "".join(f"{tbl[nm]['recall'][t]:>34.4f}" for t in variants))

    json.dump(res, open(ROOT / "wbf_by_size_official.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n已写入 {ROOT/'wbf_by_size_official.json'}")


if __name__ == "__main__":
    main()
