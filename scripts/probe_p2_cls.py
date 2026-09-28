#!/usr/bin/env python
"""预检：P2 这一层的分类头到底有没有用？

做法（纯推理，不训练）：把 P2 检测层的分类分数替换成 P3 层分类分数的 2× 最近邻上采样，
其余全部不变，然后跑 val。若精度几乎不掉，说明"P2 只做定位、分类从 P3 借"这个
轻量化设计成立（可省掉 P2 两个分类头的 1.68 GFLOPs ≈ P2 额外开销的 47%）。

用法: python probe_p2_cls.py --weights runs/visdrone/p2_n_640/weights/best.pt --device 0
"""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))
_ORIG = Detect.forward_head


def patched_forward_head(self, x, box_head=None, cls_head=None):
    """把 level0（P2）的 cls 分数替换为 level1（P3）分数的 2× 最近邻上采样。"""
    out = _ORIG(self, x, box_head, cls_head)
    if not out or box_head is None or len(x) < 2:
        return out
    counts = [xi.shape[-2] * xi.shape[-1] for xi in x]
    scores = out["scores"]
    b, c = scores.shape[0], scores.shape[1]
    parts, o = [], 0
    for n in counts:
        parts.append(scores[:, :, o : o + n])
        o += n
    h0, w0 = x[0].shape[-2:]
    h1, w1 = x[1].shape[-2:]
    up = F.interpolate(parts[1].view(b, c, h1, w1), size=(h0, w0), mode="nearest").view(b, c, -1)
    out["scores"] = torch.cat([up] + parts[1:], dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_n_640/weights/best.pt"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--mode", choices=["cls_from_p3", "box_from_p3"], default="cls_from_p3")
    a = ap.parse_args()

    if a.mode == "cls_from_p3":
        Detect.forward_head = patched_forward_head
        print("模式：P2 的分类分数 <- P3 上采样（省 1.676 GFLOPs）")
    else:
        raise NotImplementedError

    yolo = YOLO(a.weights)
    m = yolo.val(
        data=str(ROOT / "datasets/visdrone.yaml"),
        imgsz=a.imgsz, batch=8, device=a.device, split="val", plots=False,
        project=str(ROOT / "runs/visdrone"), name="val_p2_clsfromp3", exist_ok=True,
    )
    b = m.box
    print(f"\n=== P2 with cls from P3 upsample ===")
    print(f"mAP50={b.map50:.4f}  mAP50-95={b.map:.4f}  P={b.mp:.4f}  R={b.mr:.4f}")
    print("对照: P2 原版 0.3490 / 0.1980 | 基线 0.3280 / 0.1820")


if __name__ == "__main__":
    main()
