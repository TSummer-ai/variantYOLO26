#!/usr/bin/env python
"""量化 P2 的收益"长在哪"：只用 P2 模型训练出来的 P3/P4/P5 三层头做推理，看还剩多少增益。

意义：
  - 若三层头仍保留大部分增益 -> 说明 P2 训练**改善了共享特征**，存在"训练时用 P2 监督、推理不要 P2"的零成本空间
  - 若几乎不剩 -> 增益全在 stride-4 那一层本身，任何不打细粒度特征的方案都没戏

用法: python eval_p345_only.py --weights runs/visdrone/p2_n_640/weights/best.pt --device 0
"""

from __future__ import annotations

import os

import argparse
import copy
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_n_640/weights/best.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tag", default="p2_model_P345_only")
    a = ap.parse_args()

    yolo = YOLO(a.weights)
    model = yolo.model.eval()
    head = model.model[-1]
    print(f"原头: {type(head).__name__} | 输入层 f={head.f} | 分支数 cv2={len(head.cv2)} cv3={len(head.cv3)}")

    # 丢掉第一个分支（P2），只保留 P3/P4/P5 —— 权重按原索引 1: 切片
    new_head = copy.deepcopy(head)
    new_head.cv2 = torch.nn.ModuleList(list(head.cv2)[1:])
    new_head.cv3 = torch.nn.ModuleList(list(head.cv3)[1:])
    if getattr(head, "one2one_cv2", None) is not None:
        new_head.one2one_cv2 = torch.nn.ModuleList(list(head.one2one_cv2)[1:])
        new_head.one2one_cv3 = torch.nn.ModuleList(list(head.one2one_cv3)[1:])
    new_head.f = list(head.f)[1:]
    new_head.nl = len(new_head.cv2)          # forward_head 按 nl 索引分支，必须同步
    new_head.stride = head.stride[1:]        # make_anchors 用 stride 还原 anchor 坐标
    model.model[-1] = new_head
    print(f"新头: 输入层 f={new_head.f} | 分支数 cv2={len(new_head.cv2)} cv3={len(new_head.cv3)}")

    yolo.model = model
    m = yolo.val(
        data=str(ROOT / "datasets/visdrone.yaml"),
        imgsz=a.imgsz,
        batch=8,
        device=a.device,
        split="val",
        plots=False,
        project=str(ROOT / "runs/visdrone"),
        name=f"val_{a.tag}",
        exist_ok=True,
    )
    b = m.box
    print(f"\n=== {a.tag} ===")
    print(f"mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}  P={b.mp:.4f}  R={b.mr:.4f}")
    print("\n对照: baseline(P3-P5) 0.3280 / 0.1820  |  P2 完整 4 层 0.3490 / 0.1980")


if __name__ == "__main__":
    main()
