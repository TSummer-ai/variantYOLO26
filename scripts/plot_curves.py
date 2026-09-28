#!/usr/bin/env python
"""训练曲线对比图：把若干次实验的 results.csv 画在同一张图上（mAP50-95 / mAP50 / dhcd_loss）。

用法:
    python plot_curves.py --runs base_n_640 a_dhcd_w10 a_dhcd_w20 --out runs/visdrone/curves.png
"""

from __future__ import annotations

import os

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("YOLO_ROOT", Path(__file__).resolve().parent.parent))


def load(name: str):
    p = ROOT / "runs/visdrone" / name / "results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    ep = [int(r["epoch"]) for r in rows]
    get = lambda k: [float(r[k]) if r.get(k) not in (None, "") else float("nan") for r in rows]
    return {
        "epoch": ep,
        "mAP50": get("metrics/mAP50(B)"),
        "mAP50-95": get("metrics/mAP50-95(B)"),
        "precision": get("metrics/precision(B)"),
        "recall": get("metrics/recall(B)"),
        "dhcd_loss": get("train/dhcd_loss") if "train/dhcd_loss" in rows[0] else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["base_n_640", "a_dhcd_w10", "a_dhcd_w20"])
    ap.add_argument("--out", default=str(ROOT / "runs/visdrone/curves.png"))
    a = ap.parse_args()

    data = {n: d for n in a.runs if (d := load(n))}
    if not data:
        raise SystemExit("没有可用的 results.csv")

    panels = [("mAP50-95", "metrics/mAP50-95(B)"), ("mAP50", "metrics/mAP50(B)"),
              ("recall", "metrics/recall(B)"), ("precision", "metrics/precision(B)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, _) in zip(axes.ravel(), panels):
        for name, d in data.items():
            ax.plot(d["epoch"], d[title], label=name, linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("VisDrone 训练曲线（训练期 val 走默认 one2many 头）", fontsize=13)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)

    # 蒸馏项单独一张（如果有）
    if any(d["dhcd_loss"] for d in data.values()):
        fig2, ax = plt.subplots(figsize=(7, 4))
        for name, d in data.items():
            if d["dhcd_loss"]:
                ax.plot(d["epoch"], d["dhcd_loss"], label=f"{name} (train/dhcd_loss)", linewidth=2)
        ax.set_title("DHCD 蒸馏项随训练变化（0 表示教师 mask 为空）")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(str(a.out).replace(".png", "_dhcd.png"), dpi=130)
        print("saved", str(a.out).replace(".png", "_dhcd.png"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
