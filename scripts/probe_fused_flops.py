#!/usr/bin/env python
"""在 CPU 上重算 **fuse 后** 的 GFLOPs（不需要 GPU）。

背景：仓库早期用 `get_flops(YOLO(w).model)` 在**未 fuse** 的模型上算 FLOPs，
会同时计入 one2many + one2one 两个头（`ultralytics/nn/modules/head.py:183-190`），
因此偏高（baseline@640 报 5.9，实际 5.32）。本脚本先 `.fuse()` 再 profile。

用途：为 fig3_pareto / fig_tradeoff 提供权威的 GFLOPs，
包括此前没有 fuse 口径数据的两个点（baseline@960 推理、P2-pruned）。

用法: python probe_fused_flops.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent))

# (名称, 权重, imgsz)  —— imgsz 为该模型的实际推理分辨率
JOBS = [
    ("baseline@640",            "runs/visdrone/base_n_640/weights/best.pt", 640),
    ("P2@640",                  "runs/visdrone/p2_n_640/weights/best.pt", 640),
    ("baseline@960 (640训/960推)", "runs/visdrone/base_n_640/weights/best.pt", 960),
    ("P2-pruned@640",           "runs/visdrone/p2_pruned_det/weights/best.pt", 640),
    ("P2@960",                  "runs/visdrone/p2_960/weights/best.pt", 960),
]

# 文档中已有的 fuse 口径值，用于自检
EXPECT = {"baseline@640": 5.32, "P2@640": 6.57, "P2@960": 15.13}


def fused_flops(weights: Path, imgsz: int):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    model = YOLO(str(weights)).model
    model.eval()
    # 关键：先 fuse 再 profile（删掉推理时不用的 one2one 分支）
    model.fuse()
    fl = get_flops(model, imgsz)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    nhead = sum(1 for m in model.modules() if type(m).__name__ == "Detect")
    return float(fl), float(params)


def main():
    out = {}
    print(f"{'配置':<26} {'GFLOPs':>8} {'params(M)':>10}  {'文档值':>8}  自检")
    for name, rel, imgsz in JOBS:
        w = ROOT / rel
        if not w.exists():
            print(f"{name:<26} {'--':>8}  （缺权重 {rel}）")
            continue
        fl, pr = fused_flops(w, imgsz)
        exp = EXPECT.get(name)
        if exp is None:
            mark = "(新增)"
            exps = "—"
        else:
            exps = f"{exp:.2f}"
            mark = "OK" if abs(fl - exp) < 0.15 else f"⚠ 偏差 {fl - exp:+.2f}"
        print(f"{name:<26} {fl:8.2f} {pr:10.2f}  {exps:>8}  {mark}")
        out[name] = {"weights": rel, "imgsz": imgsz, "gflops_fused": round(fl, 2),
                     "params_M": round(pr, 3), "doc_value": exp}

    (ROOT / "fused_flops.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\nsaved fused_flops.json")


if __name__ == "__main__":
    main()
