#!/usr/bin/env python
"""任务3b：矩形（rect）高分辨率推理，验证 padding 浪费。

动机：VisDrone 为 1360x765 宽幅。方形推理时 imgsz=1600 的 letterbox 缩放只有
      min(1600/1360,1600/765)=1.176，且被 pad 到 1600x1600 —— 约 44% 的算力花在 pad 上。
      rect=True 时输入为 1600x900 量级，同样分辨率下算力/延迟大幅下降。

用法:
  python probe_rect_resolution.py --weights runs/visdrone/p2_960/weights/best.pt \
         --imgszs 1280 1600 1920 2240 --out rect_res.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, build_gt, evaluate  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))


def predict_full_rect(model, path, imgsz, conf, iou, device, rect, max_det=3000):
    r = model.predict(str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                      device=device, verbose=False, rect=rect)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_960/weights/best.pt"))
    ap.add_argument("--imgszs", type=int, nargs="+", default=[1280, 1600, 1920, 2240])
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rect", default="1", choices=["0", "1"])
    ap.add_argument("--out", default=str(ROOT / "rect_res.json"))
    a = ap.parse_args()
    a.rect = (a.rect == "1")

    from ultralytics import YOLO
    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    print(f"val 图数 = {len(files)}  rect={bool(a.rect)}", flush=True)
    coco_gt = build_gt(files)
    y = YOLO(a.weights)
    bench_img = str(DS / files[0][2:])

    res = {"weights": a.weights, "rect": bool(a.rect), "rows": []}
    for imgsz in a.imgszs:
        row = {"imgsz": imgsz, "compute_sq": round((imgsz / 640.0) ** 2, 2)}
        print(f"\n=== rect={bool(a.rect)} imgsz={imgsz} ===", flush=True)
        try:
            torch.cuda.empty_cache()
            preds = []
            for i, line in enumerate(files):
                ip = DS / line[2:]
                b, s, c = predict_full_rect(y, ip, imgsz, 0.001, 0.7, a.device, a.rect)
                for j in range(len(s)):
                    preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                                  "bbox": [float(b[j, 0]), float(b[j, 1]),
                                           float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                                  "score": float(s[j])})
                if i % 150 == 0:
                    print(f"  {i}/{len(files)}", flush=True)
            p95, p50 = evaluate(coco_gt, preds)
            row.update({"ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds)})
            print(f"  AP50-95={p95:.4f} AP50={p50:.4f}", flush=True)
        except RuntimeError as e:
            row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"  !! {row['error']}", flush=True)
            res["rows"].append(row); torch.cuda.empty_cache()
            json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
            continue

        # 延迟：用真实 VisDrone 尺寸图，warmup 5 / 20 次
        for _ in range(5):
            y.predict(bench_img, imgsz=imgsz, device=a.device, verbose=False, rect=a.rect,
                      conf=0.001, iou=0.7)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            y.predict(bench_img, imgsz=imgsz, device=a.device, verbose=False, rect=a.rect,
                      conf=0.001, iou=0.7)
        torch.cuda.synchronize()
        row["e2e_ms"] = round((time.perf_counter() - t0) / 20 * 1000, 3)
        row["fps"] = round(1000.0 / row["e2e_ms"], 1)
        print(f"  端到端={row['e2e_ms']}ms ({row['fps']} FPS)", flush=True)
        res["rows"].append(row)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 62)
    print(f"{'配置':>22}{'AP50-95':>10}{'AP50':>8}{'端到端ms':>11}{'FPS':>7}")
    for r in res["rows"]:
        if "ap50_95" not in r:
            print(f"{'rect@'+str(r['imgsz']):>22}{'FAIL':>10}")
            continue
        print(f"{'rect@'+str(r['imgsz']):>22}{r['ap50_95']:>10.4f}{r['ap50']:>8.4f}"
              f"{r['e2e_ms']:>11}{r['fps']:>7}")
    print("=" * 62)
    print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
