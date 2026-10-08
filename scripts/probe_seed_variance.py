#!/usr/bin/env python
"""任务1 分析：baseline@640 多种子方差 / 噪声底。

对 `runs/visdrone/base_n_640`（seed=0）与 `base_n_640_seed{1,2}` 三个权重：
  1) yolo val 内部口径（imgsz=640, max_det=1000 与 300）
  2) COCO 协议（本仓库自建 harness，与 README 全栈表同口径）
然后给出均值 / 标准差 / 极差，作为后续消融"是否在噪声内"的判定基准。

用法: python probe_seed_variance.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, build_gt, evaluate, predict_full  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")
DATA = ROOT / "datasets/visdrone.yaml"
SEEDS = {0: ROOT / "runs/visdrone/base_n_640/weights/best.pt",
         1: ROOT / "runs/visdrone/base_n_640_seed1/weights/best.pt",
         2: ROOT / "runs/visdrone/base_n_640_seed2/weights/best.pt"}


def main():
    from ultralytics import YOLO

    avail = {k: v for k, v in SEEDS.items() if v.exists()}
    missing = [k for k in SEEDS if k not in avail]
    print(f"可用种子: {sorted(avail)}   缺失: {missing}", flush=True)
    if not avail:
        raise SystemExit("没有任何种子权重")

    files = [l.strip() for l in open(DS / "val.txt")]
    coco_gt = build_gt(files)

    rows = {}
    out_path = ROOT / "seed_variance_results.json"

    def _stats(key):
        vals = [rows[s][key]["map"] for s in sorted(rows)]
        if len(vals) < 2:
            return {"values": vals, "mean": vals[0] if vals else None,
                    "std": None, "range": None}
        return {"values": [round(v, 4) for v in vals],
                "mean": round(st.mean(vals), 4),
                "std": round(st.stdev(vals), 4),
                "range": round(max(vals) - min(vals), 4)}

    def dump(partial=False):
        """增量落盘：即使后续种子崩掉，已完成种子的结果也不会丢。"""
        sm = {"n_seeds": len(rows), "partial": partial,
              "coco_map": _stats("coco"),
              "val_md1000_map": _stats("val_md1000"),
              "val_md300_map": _stats("val_md300"),
              "per_seed": rows}
        json.dump(sm, open(out_path, "w"), ensure_ascii=False, indent=1)
        return sm

    for seed, w in sorted(avail.items()):
        print(f"\n=== seed={seed}  {w} ===", flush=True)
        rec = {"weights": str(w)}
        model = YOLO(w)
        for md in (1000, 300):
            r = model.val(data=str(DATA), imgsz=640, max_det=md, device="0",
                          verbose=False, plots=False)
            rec[f"val_md{md}"] = {"map50": round(float(r.box.map50), 4),
                                  "map": round(float(r.box.map), 4),
                                  "precision": round(float(r.box.mp), 4),
                                  "recall": round(float(r.box.mr), 4)}
            print(f"  yolo val md={md}:  mAP50={rec[f'val_md{md}']['map50']:.4f}  "
                  f"mAP50-95={rec[f'val_md{md}']['map']:.4f}", flush=True)
        # COCO 协议
        preds = []
        for i, line in enumerate(files):
            b, s, c = predict_full(model, DS / line[2:], 640, 0.001, 0.7, "0")
            for j in range(len(s)):
                preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                              "bbox": [float(b[j, 0]), float(b[j, 1]),
                                       float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                              "score": float(s[j])})
        p95, p50 = evaluate(coco_gt, preds)
        rec["coco"] = {"map50": round(p50, 4), "map": round(p95, 4)}
        print(f"  COCO 协议:     AP50={p50:.4f}  AP50-95={p95:.4f}", flush=True)
        rows[seed] = rec
        dump(partial=(seed != max(avail)))
        print(f"  [增量落盘] {out_path.name} 已写入（已完成 {len(rows)} 个种子）", flush=True)
        torch.cuda.empty_cache()

    summary = dump(partial=False)

    print("\n" + "=" * 74)
    print(f"{'种子':>6}{'COCO AP50-95':>16}{'val md1000 mAP50-95':>22}{'val md300':>12}")
    for s in sorted(rows):
        print(f"{s:>6}{rows[s]['coco']['map']:>16.4f}{rows[s]['val_md1000']['map']:>22.4f}"
              f"{rows[s]['val_md300']['map']:>12.4f}")
    c = summary["coco_map"]
    print("-" * 74)
    if c["std"] is not None:
        print(f"COCO AP50-95: 均值 {c['mean']:.4f}  标准差 {c['std']:.4f}  极差 {c['range']:.4f}")
        print(f"⇒ 单种子噪声底（1σ）≈ ±{c['std']:.4f} AP；2σ ≈ ±{2*c['std']:.4f} AP")
        print("  对照：P2 层 +1.45、SAR +0.3、DHCD −0.1 ⇒ 请据此判断显著性")
    else:
        print("只有 1 个种子，无法估计方差（等待另一个种子训练完成）")
    print("=" * 74)
    print(f"已写入 {ROOT/'seed_variance_results.json'}")


if __name__ == "__main__":
    main()
