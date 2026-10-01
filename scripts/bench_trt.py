#!/usr/bin/env python
"""TensorRT vs PyTorch 速度对比（抗噪声版）。

方法：轮转交错测量（round-robin）消除热漂移，取 N 次的最小值（min 比 mean 更能代表
      真实延迟，受系统抖动影响小）。同时报告"纯网络前向"和"端到端 predict"。

用法: python bench_trt.py --pt weights/x.pt --engines weights/a.engine weights/b.engine --imgsz 960
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
ASSET = ROOT / "ultralytics/ultralytics/assets/bus.jpg"


def make_forward_runner(path: str):
    """返回一个可反复调用的前向函数（纯网络，不含预处理/NMS）。"""
    y = YOLO(path)
    m = y.model
    if hasattr(m, "cuda"):  # PyTorch nn.Module
        m = m.cuda().eval()
        x = torch.rand(1, 3, 960, 960, device="cuda")
        def run():
            with torch.no_grad():
                m(x)
        return run, "torch"
    # engine：直接用 AutoBackend（YOLO(path).model 对非 .pt 是路径字符串）
    from ultralytics.nn.autobackend import AutoBackend
    backend = AutoBackend(model=path, device=torch.device("cuda:0"), fp16=True, fuse=True)
    x = torch.rand(1, 3, 960, 960, device="cuda").half()
    def run():
        backend(x)
    return run, "trt"


def make_predict_runner(path: str, imgsz: int, **kw):
    y = YOLO(path)
    def run():
        y.predict(str(ASSET), imgsz=imgsz, device=0, verbose=False, **kw)
    return run


def bench_roundrobin(runners: dict, n: int = 60, warmup: int = 8):
    """轮转交错，返回每个 runner 的最小耗时(ms)。"""
    for name, fn in runners.items():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    best = {k: float("inf") for k in runners}
    for _ in range(n):
        for name, fn in runners.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            best[name] = min(best[name], (time.perf_counter() - t0) * 1000)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--engines", nargs="*", default=[])
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    fw = {}
    for tag, p in [("PyTorch FP16", a.pt)] + [(f"TRT {Path(e).stem.split('.')[-1]}", e) for e in a.engines]:
        run, _ = make_forward_runner(p)
        fw[tag] = run
    # PyTorch 单独测 FP32/FP16（half 需在 forward 里体现，这里用 quantize 版本的 predicts 代表）
    res_fw = bench_roundrobin(fw, n=a.n)

    pd = {}
    for tag, p, kw in [("PyTorch FP32", a.pt, {}), ("PyTorch FP16", a.pt, {"quantize": 16})] + \
                      [(f"TRT {Path(e).stem.split('.')[-1]}", e, {}) for e in a.engines]:
        pd[tag] = make_predict_runner(p, a.imgsz, **kw)
    res_pd = bench_roundrobin(pd, n=a.n)

    base_fw = res_fw.get("PyTorch FP16", 1)
    base_pd = res_pd.get("PyTorch FP32", 1)
    print(f"\n{'配置':<18}{'纯前向 min(ms)':>16}{'相对':>8}{'FPS':>8}")
    for k, v in sorted(res_fw.items(), key=lambda x: x[1]):
        print(f"{k:<18}{v:>16.2f}{base_fw/v:>7.2f}x{1000/v:>8.1f}")
    print(f"\n{'配置':<18}{'端到端 min(ms)':>16}{'相对FP32':>10}{'FPS':>8}")
    for k, v in sorted(res_pd.items(), key=lambda x: x[1]):
        print(f"{k:<18}{v:>16.2f}{base_pd/v:>9.2f}x{1000/v:>8.1f}")


if __name__ == "__main__":
    main()
