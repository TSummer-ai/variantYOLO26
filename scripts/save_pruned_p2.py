#!/usr/bin/env python
"""把"P2 模型 - P2 检测层"保存成可加载的 checkpoint，用于分尺寸分析。

注意：这个配置 **不是零成本**（6.6 vs 基线 5.9 GFLOPs），P2 的 neck 路径仍在运行。
保存它只是为了做尺寸分解，看"留在权重里的那部分增益"是否集中在小目标上。

用法: python save_pruned_p2.py
"""

from __future__ import annotations

import os

import copy
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
P2 = ROOT / "runs/visdrone/p2_n_640/weights/best.pt"
OUT = ROOT / "runs/visdrone/p2_pruned_det/weights/best.pt"


def main():
    yolo = YOLO(str(P2))
    ckpt = yolo.ckpt if isinstance(yolo.ckpt, dict) else {}
    model = copy.deepcopy(yolo.model).eval()
    head = model.model[-1]

    head.cv2 = torch.nn.ModuleList(list(head.cv2)[1:])
    head.cv3 = torch.nn.ModuleList(list(head.cv3)[1:])
    head.one2one_cv2 = torch.nn.ModuleList(list(head.one2one_cv2)[1:])
    head.one2one_cv3 = torch.nn.ModuleList(list(head.one2one_cv3)[1:])
    head.f = list(head.f)[1:]
    head.nl = len(head.cv2)
    head.stride = head.stride[1:]
    print(f"剪枝后: Detect 输入层 {head.f} | 分支数 {head.nl} | stride {head.stride.tolist()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in ckpt.items() if k not in ("model", "ema")}
    save["model"] = model
    torch.save(save, OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
