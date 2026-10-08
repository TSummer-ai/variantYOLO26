#!/usr/bin/env python
"""任务3d：切片的"实现公平性"延迟核查（GPU，训练后运行）。

针对可能的质疑："52.9 ms 是你逐块调用 predict 的实现开销，批量化后切片并不慢。"
本脚本在同一张验证图上对比三种实现，conf/iou 相同：

  A 逐块串行     3 次 predict（仓库现有 predict_sliced 的做法）
  B 单次批量     tiles 作为 list 一次 predict(..., batch=3)
  C 纯网络理论上界 4 次 960² 前向（无预/后处理）
  D 参照        整图 1600 一次 predict

若 B 仍显著慢于 D，则"切片被高分辨率整图支配"的结论与实现方式无关。

用法: python bench_batched_slicing.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, tile_starts  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))
IMGSZ, OVERLAP, CONF, IOU = 960, 0.3, 0.001, 0.7


def main():
    from ultralytics import YOLO
    files = [l.strip() for l in open(DS / "val.txt")]
    img = str(DS / files[0][2:])
    W, H = Image.open(img).size
    print(f"基准图 {Path(img).name} {W}x{H}", flush=True)

    y = YOLO(str(ROOT / "runs/visdrone/p2_960/weights/best.pt"))
    m = y.model.to("cuda:0").eval(); m.fuse()

    im = Image.open(img).convert("RGB")
    xs, tw = tile_starts(W, 3, OVERLAP)
    ys, th = tile_starts(H, 1, OVERLAP)
    crops = [np.array(im.crop((x, ys[0], min(x + tw, W), min(ys[0] + th, H))))
             for x in xs]
    print(f"tiles={len(crops)}  tile size={crops[0].shape[1]}x{crops[0].shape[0]}", flush=True)

    res = {"bench_img": Path(img).name, "imgsz": IMGSZ, "overlap": OVERLAP,
           "conf": CONF, "n_tiles": len(crops)}

    def timed(fn, n, warm):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1000

    # A 逐块串行
    try:
        a = timed(lambda: [y.predict(c, imgsz=IMGSZ, conf=CONF, iou=IOU, device="0",
                                     verbose=False) for c in crops], 8, 2)
        res["A_serial_tiles_ms"] = round(a, 2)
        print(f"  A 逐块串行      {a:.2f} ms", flush=True)
    except RuntimeError as e:
        res["A_error"] = str(e)[:150]; print(f"  A 失败 {e}"[:200], flush=True)

    # B 单次批量
    try:
        b = timed(lambda: y.predict(crops, imgsz=IMGSZ, conf=CONF, iou=IOU, device="0",
                                    verbose=False, batch=len(crops)), 8, 2)
        res["B_batched_tiles_ms"] = round(b, 2)
        print(f"  B 单次批量(batch={len(crops)}) {b:.2f} ms", flush=True)
    except Exception as e:
        res["B_error"] = f"{type(e).__name__}: {str(e)[:150]}"
        print(f"  B 失败 {res['B_error']}", flush=True)
        try:
            b = timed(lambda: y.predict(crops, imgsz=IMGSZ, conf=CONF, iou=IOU, device="0",
                                        verbose=False), 8, 2)
            res["B_batched_tiles_ms"] = round(b, 2)
            print(f"  B 单次批量(默认 batch) {b:.2f} ms", flush=True)
        except Exception as e2:
            res["B_error2"] = str(e2)[:150]

    # 整图 960 / 1600
    for tag, sz in (("full960", 960), ("full1600", 1600)):
        try:
            t = timed(lambda sz=sz: y.predict(img, imgsz=sz, conf=CONF, iou=IOU,
                                              device="0", verbose=False), 10, 3)
            res[f"{tag}_ms"] = round(t, 2)
            print(f"  {tag:<15} {t:.2f} ms", flush=True)
        except RuntimeError as e:
            res[f"{tag}_error"] = str(e)[:150]

    # C 纯网络理论上界：4 × 960² 前向
    try:
        x = torch.rand(1, 3, IMGSZ, IMGSZ, device="cuda:0")
        with torch.no_grad():
            for _ in range(5):
                m(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(15):
                for _ in range(4):
                    m(x)
            torch.cuda.synchronize()
        c = (time.perf_counter() - t0) / 15 * 1000
        x2 = torch.rand(1, 3, 1600, 1600, device="cuda:0")
        with torch.no_grad():
            for _ in range(5):
                m(x2)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(15):
                m(x2)
            torch.cuda.synchronize()
        d = (time.perf_counter() - t0) / 15 * 1000
        res["C_pure_net_4x960_ms"] = round(c, 2)
        res["D_pure_net_1600_ms"] = round(d, 2)
        res["C_over_D"] = round(c / d, 2)
        print(f"  C 纯网络 4x960²  {c:.2f} ms   D 纯网络 1600² {d:.2f} ms   C/D={c/d:.2f}",
              flush=True)
    except RuntimeError as e:
        res["C_error"] = str(e)[:150]
        print(f"  C 失败 {e}"[:200], flush=True)

    json.dump(res, open(ROOT / "bench_batched_slicing.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"\n已写入 {ROOT/'bench_batched_slicing.json'}")


if __name__ == "__main__":
    main()
