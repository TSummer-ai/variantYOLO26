import os
from pathlib import Path

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))

#!/usr/bin/env python
"""效率对比：纯前向延迟 / 端到端 predict 延迟 / 参数量 / FLOPs。

论文的效率表就用这个跑，避免只报 FLOPs（FLOPs 会严重高估小特征图的实际开销）。

用法:
    python bench_models.py --weights a.pt b.pt --names "yolo26n" "yolo26n-p2" --imgsz 640
"""

from __future__ import annotations

import argparse
import time

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ASSET = f"{ROOT}/ultralytics/ultralytics/assets/bus.jpg"


def bench_forward(weights, imgsz, n=100, device="0"):
    model = YOLO(weights).model.to(f"cuda:{device}" if device.isdigit() else device).eval()
    x = torch.rand(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(15):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_predict(weights, imgsz, n=60, device="0"):
    model = YOLO(weights)
    for _ in range(5):
        model.predict(ASSET, imgsz=imgsz, device=device, verbose=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model.predict(ASSET, imgsz=imgsz, device=device, verbose=False)
    return (time.perf_counter() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    names = a.names or [w.split("/")[-1] for w in a.weights]

    print(f"{'模型':>22} {'params(M)':>10} {'GFLOPs':>8} {'前向(ms)':>10} {'FPS':>7} {'端到端(ms)':>11} {'FPS':>7}")
    rows = []
    for name, w in zip(names, a.weights):
        m = YOLO(w).model
        params = sum(p.numel() for p in m.parameters()) / 1e6
        fl = get_flops(m, a.imgsz)
        f = bench_forward(w, a.imgsz, device=a.device)
        p = bench_predict(w, a.imgsz, device=a.device)
        rows.append((name, params, fl, f, p))
        print(f"{name:>22} {params:10.2f} {fl:8.1f} {f:10.2f} {1000/f:7.1f} {p:11.2f} {1000/p:7.1f}")
    if len(rows) > 1:
        b = rows[0]
        print("\n相对第一行:")
        for r in rows[1:]:
            print(f"  {r[0]:>22}: params {(r[1]/b[1]-1)*100:+.1f}%  GFLOPs {(r[2]/b[2]-1)*100:+.1f}%  "
                  f"前向 {(r[3]/b[3]-1)*100:+.1f}%  端到端 {(r[4]/b[4]-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
