#!/usr/bin/env python
"""统一口径的推理策略延迟基准。

之前的口径问题：square 用默认 conf（0.25）、切片臂用 conf=0.001，两者不可比。
本脚本在同一张 VisDrone 验证图、同一 conf/iou 下测：

  纯网络前向  fwd_ms        —— 张量进张量出，隔离网络成本（rect 用真实矩形尺寸）
  端到端      e2e_ms        —— 含 letterbox + 解码 + NMS，与评测管线同口径

用法:
  python bench_inference_strategies.py --weights runs/visdrone/p2_960/weights/best.pt
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_slicing import DS, merge_nms, predict_full, predict_sliced  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))
BIG_PX, OVERLAP, MERGE_IOU = 48.0, 0.3, 0.6


def rect_hw(W, H, imgsz):
    s = min(imgsz / W, imgsz / H)
    return int(round(H * s)), int(round(W * s))


def bench_fwd(model, hw, n=30, warm=10):
    dev = next(model.parameters()).device
    x = torch.rand(1, 3, hw[0], hw[1], device=dev)
    with torch.no_grad():
        for _ in range(warm):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_pred(y, img, imgsz, rect, conf, iou, device, n=15, warm=4):
    for _ in range(warm):
        y.predict(img, imgsz=imgsz, rect=rect, conf=conf, iou=iou, device=device, verbose=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y.predict(img, imgsz=imgsz, rect=rect, conf=conf, iou=iou, device=device, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_slice(y, img, imgsz, conf, iou, device, n=8, warm=2):
    def one():
        bf, sf, cf = predict_full(y, img, imgsz, conf, iou, device)
        bt, st, ct = predict_sliced(y, img, imgsz, conf, iou, device, (3, 1), OVERLAP)
        kt = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0)) < BIG_PX if len(bt) else np.zeros(0, bool)
        b = np.concatenate([bf, bt[kt]]) if len(bf) or kt.any() else np.zeros((0, 4))
        s = np.concatenate([sf, st[kt]]) if len(bf) or kt.any() else np.zeros(0)
        c = np.concatenate([cf, ct[kt]]) if len(bf) or kt.any() else np.zeros(0, int)
        return merge_nms(b, s, c, MERGE_IOU) if len(b) else (b, s, c)
    for _ in range(warm):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_960/weights/best.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=str(ROOT / "bench_strategies.json"))
    a = ap.parse_args()

    from ultralytics import YOLO
    files = [l.strip() for l in open(DS / "val.txt")]
    img = str(DS / files[0][2:])
    W, H = Image.open(img).size
    print(f"基准图 {Path(img).name}  原图 {W}x{H}", flush=True)

    y = YOLO(a.weights)
    m = y.model.to(f"cuda:{a.device}").eval()
    m.fuse()

    rows = []
    res = {"weights": a.weights, "bench_img": Path(img).name, "img_wh": [W, H], "rows": rows}
    confs = [0.001, 0.25]

    for mode in ("square", "rect"):
        for imgsz in (640, 960, 1280, 1600, 1920):
            if mode == "square":
                hw = (imgsz, imgsz)
            else:
                hw = rect_hw(W, H, imgsz)
            row = {"mode": mode, "imgsz": imgsz, "input_hw": list(hw),
                   "px": hw[0] * hw[1] / 1e6}
            try:
                torch.cuda.empty_cache()
                row["fwd_ms"] = round(bench_fwd(m, hw), 3)
                for cf in confs:
                    row[f"e2e_ms_conf{cf}"] = round(
                        bench_pred(y, img, imgsz, mode == "rect", cf, 0.7, a.device), 3)
                print(f"  {mode}@{imgsz} in={hw} px={row['px']:.2f}M "
                      f"fwd={row['fwd_ms']}ms e2e.001={row['e2e_ms_conf0.001']}ms "
                      f"e2e.25={row['e2e_ms_conf0.25']}ms", flush=True)
            except RuntimeError as e:
                row["error"] = f"{type(e).__name__}: {str(e)[:140]}"
                print(f"  !! {mode}@{imgsz} {row['error']}", flush=True)
                torch.cuda.empty_cache()
            rows.append(row)
            json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)

    # 切片管线
    row = {"mode": "slice3x1_gated", "imgsz": 960, "forwards": 4}
    try:
        row["fwd_ms"] = round(bench_fwd(m, (960, 960)), 3)
        for cf in confs:
            row[f"e2e_ms_conf{cf}"] = round(bench_slice(y, img, 960, cf, 0.7, a.device), 3)
        print(f"  切片3x1@960 fwd(每块)={row['fwd_ms']}ms "
              f"e2e.001={row['e2e_ms_conf0.001']}ms e2e.25={row['e2e_ms_conf0.25']}ms", flush=True)
    except RuntimeError as e:
        row["error"] = str(e)[:140]
    rows.append(row)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)  # 修复：切片臂此前未落盘

    print("\n" + "=" * 92)
    print(f"{'策略':>20}{'输入':>12}{'Mpx':>7}{'纯前向ms':>10}{'e2e@.001':>11}{'e2e@.25':>10}{'FPS@.25':>9}")
    for r in rows:
        if "fwd_ms" not in r:
            continue
        hw = r.get("input_hw") or (960, 960)
        px = r.get("px") or (hw[0] * hw[1] / 1e6)
        e25 = r.get("e2e_ms_conf0.25")
        print(f"{r['mode']+'@'+str(r['imgsz']):>20}{str(hw):>12}{px:>7.2f}{r['fwd_ms']:>10.2f}"
              f"{r.get('e2e_ms_conf0.001','—'):>11}{e25 if e25 else '—':>10}"
              f"{round(1000/e25,1) if e25 else '—':>9}")
    print("=" * 92)
    print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
