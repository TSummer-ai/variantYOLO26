#!/usr/bin/env python
"""重画仓库 results/figures/ 的图，数据全部更新到**最新权威口径**（2026-10-08）。

与旧版（2026-09-28）的差异
------------------------------------------------
1. **效率口径更正**（2026-10-02）：旧图用的是**未 fuse** 的模型，
   同时计入 one2many + one2one 两个头（`head.py:183-190`），GFLOPs 与延迟都偏高。
   本版全部改为 **fuse 后**的部署口径：
       baseline@640  5.9  -> 5.32 GFLOPs ;  4.57 -> 2.86 ms
       P2@640        7.7  -> 6.57 GFLOPs ;  5.88 -> 3.47 ms
       P2@960       17.6  -> 15.13 GFLOPs;  8.96 -> 6.44 ms
2. **补齐两个此前没有 fuse 口径 GFLOPs 的点**（本机 CPU 实测，见 `probe_fused_flops.py`）：
       baseline@960 推理（640 训 / 960 推）-> 12.31 GFLOPs
       P2-pruned（P2 neck + 三层头）      -> 6.06 GFLOPs
   其**延迟**在 GPU 不可用时无法重测，按同架构在 640 上实测的 fuse 修正系数折算，
   图内以 `*` 标注为"待重测"。
3. fig1 / fig2 / fig4 的数据源（oracle 日志、GT 尺寸统计）自 2026-09-27 起**未变**，
   本版重画只为口径统一，数值与旧图一致。

用法:
    python make_figures_v2.py --out figures_v2
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ---------------------------------------------------------------- 字体
_CJK = next((f for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                         "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
             if Path(f).exists()), None)
if _CJK:
    font_manager.fontManager.addfont(_CJK)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CJK).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 11

HERE = Path(__file__).resolve().parent
BUCKETS = ["0-8", "8-16", "16-32", "32-64", ">=64"]


def _candidates() -> list[Path]:
    """脚本可能位于工作目录，也可能位于仓库的 scripts/ 下。

    典型布局：<workdir>/visdrone-yolo26/scripts/ —— 此时数据集在 <workdir>/datasets/，
    oracle 日志在 <workdir>/visdrone-yolo26/results/。
    """
    c = [HERE, HERE.parent, HERE.parent.parent,
         Path(os.environ.get("YOLO_ROOT", HERE))]
    c += [HERE / "visdrone-yolo26", HERE.parent / "visdrone-yolo26"]
    seen, out = set(), []
    for p in c:
        p = p.resolve()
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_oracle_root() -> Path:
    """oracle 日志在仓库的 results/oracle_logs/ 下。"""
    for cand in _candidates():
        if (cand / "results/oracle_logs/oracle.log").exists():
            return cand
    return HERE


def dataset_root() -> Path:
    """数据集（只用于 fig4 的 GT 尺寸统计）在工作目录的 datasets/ 下。"""
    for cand in _candidates():
        for rel in ("datasets/visdrone/val.txt", "visdrone-yolo26/datasets/visdrone/val.txt"):
            if (cand / rel).exists():
                return cand / Path(rel).parent
    return _candidates()[0] / "datasets/visdrone"


ORACLE_ROOT = find_oracle_root()
DS_ROOT = dataset_root()

# ---------------------------------------------------------------- 权威数据
# 全部来自 README.md / docs/RESULTS_FINAL.md（fuse 后部署口径，RTX 4060 Laptop, FP32, batch 1）
#   name: (GFLOPs, 前向 ms, mAP50-95 @max_det=300, @max_det=1000, 颜色, 是否需重测)
POINTS = {
    "baseline@640":  dict(gf=5.32,  lat=2.86, ap=0.1820, ap_md=0.1860, color="#7f7f7f", recheck=False),
    "P2@640":        dict(gf=6.57,  lat=3.47, ap=0.1980, ap_md=0.2030, color="#1f77b4", recheck=False),
    "baseline@960\n(640 训/960 推)":
                     dict(gf=12.31, lat=3.73, ap=0.2240, ap_md=None, color="#ff7f0e", recheck=True),
    "P2-pruned\n(P2 neck + 三层头)":
                     dict(gf=6.06,  lat=3.19, ap=0.1888, ap_md=None, color="#2ca02c", recheck=True),
    "P2@960":        dict(gf=15.13, lat=6.44, ap=0.2630, ap_md=0.2690, color="#d62728", recheck=False),
}
RECHECK_NOTE = ("* 该点延迟为推算值（按同架构 640 上实测的 fuse 修正系数折算），GFLOPs 为 fuse 后 CPU 实测；"
                "GPU 可用后应重测延迟。")

ORACLE_LOGS = {"baseline@640": "results/oracle_logs/oracle.log",
               "P2@640": "results/oracle_logs/oracle_p2.log",
               "P2@960": "results/oracle_logs/oracle_p2_960.log"}


# ---------------------------------------------------------------- 解析
def parse_oracle(path: Path):
    """返回 (分尺寸漏检率 dict, oracle AP dict)，取 head=o2m 段。"""
    if not path.exists():
        return {}, {}
    lines = path.read_text(errors="ignore").splitlines()
    starts = [i for i, l in enumerate(lines) if "head=o2m" in l]
    ends = [i for i, l in enumerate(lines) if "head=e2e" in l]
    if not starts:
        return {}, {}
    end = next((i for i in ends if i > starts[0]), len(lines))
    sec = "\n".join(lines[starts[0]:end])
    miss = {}
    for line in sec.splitlines():
        m = re.match(r"\s*(0-8|8-16|16-32|32-64|>=64)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)%", line)
        if m:
            miss[m.group(1)] = float(m.group(4))
    aps = {}
    for line in sec.splitlines():
        m = re.match(r"(real|loc-oracle|cls-oracle|fn-oracle|fp-oracle)（[^）]*）\s+AP50-95=([\d.]+)", line)
        if m:
            aps[m.group(1)] = float(m.group(2))
    return miss, aps


# ---------------------------------------------------------------- 各面板
def fig_size_missrate(ax):
    vals = {}
    for name, log in ORACLE_LOGS.items():
        miss, _ = parse_oracle(ORACLE_ROOT / log)
        vals[name] = [miss.get(b, np.nan) for b in BUCKETS]
        if not miss:
            print(f"  ⚠ 读不到 {ORACLE_ROOT / log} 的分尺寸漏检率")
    x = np.arange(len(BUCKETS))
    w = 0.26
    for i, (name, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1) * w, v, w, label=name, color=POINTS[name]["color"])
        for xi, vi in zip(x + (i - 1) * w, v):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.015, f"{vi:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} px" for b in BUCKETS])
    ax.set_ylabel("miss rate (IoU 0.5, conf 0.001)")
    ax.set_xlabel("GT size sqrt(area), original image")
    ax.set_title("(a) 分尺寸漏检率 — 改善随目标变小单调增大")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.0)


def fig_oracle(ax):
    keys = ["loc-oracle", "cls-oracle", "fn-oracle", "fp-oracle"]
    labels = ["+定位完美", "+分类完美", "+不漏检", "+无误检"]
    x = np.arange(len(keys))
    w = 0.26
    for i, (name, log) in enumerate(ORACLE_LOGS.items()):
        _, aps = parse_oracle(ORACLE_ROOT / log)
        base = aps.get("real", 0)
        gains = [aps.get(k, 0) - base for k in keys]
        ax.bar(x + (i - 1) * w, gains, w, label=f"{name} (real={base:.3f})", color=POINTS[name]["color"])
        for xi, gi in zip(x + (i - 1) * w, gains):
            ax.text(xi, gi + 0.008, f"{gi:+.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP50-95 gain (COCO protocol)")
    ax.set_title("(b) Oracle 分解：漏检始终是最大项")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)


def fig_pareto(ax, compact: bool = False):
    """精度-延迟帕累托（fuse 后部署口径）。

    compact=True 用于 误差分解四联图 的小面板：改用时序图例，避免标注互相压字。
    """
    # 每点的标注方位（错开，避免互相压字）；ha/va 一并给出
    off = {
        "baseline@640":  dict(xy=(-8, -10), ha="right", va="top"),
        "P2-pruned":     dict(xy=(8, -10), ha="left", va="top"),
        "P2@640":        dict(xy=(8, 8), ha="left", va="bottom"),
        "baseline@960":  dict(xy=(8, 8), ha="left", va="bottom"),
        "P2@960":        dict(xy=(-8, -6), ha="right", va="top"),
    }
    for name, p in POINTS.items():
        tag = "*" if p["recheck"] else ""
        first = name.splitlines()[0]
        ax.scatter(p["lat"], p["ap"], s=(70 if compact else 110), color=p["color"], zorder=3,
                   edgecolor="white", linewidth=1.0,
                   label=f"{first}{tag}: {p['ap']:.3f} AP / {p['lat']:.2f} ms" if compact else None)
        if compact:
            continue
        o = off.get(first, dict(xy=(9, 8), ha="left", va="bottom"))
        ax.annotate(f"{first}{tag}\n{p['ap']:.3f} AP / {p['lat']:.2f} ms",
                    (p["lat"], p["ap"]), textcoords="offset points",
                    xytext=o["xy"], ha=o["ha"], va=o["va"], fontsize=8)
    xs = [p["lat"] for p in POINTS.values()]
    ys = [p["ap"] for p in POINTS.values()]
    order = np.argsort(xs)
    ax.plot(np.array(xs)[order], np.array(ys)[order], "--", color="gray", alpha=0.6, zorder=1)
    ax.set_xlabel("单图前向延迟 (ms, RTX 4060 Laptop, FP32, fuse 后)")
    ax.set_ylabel("mAP50-95 (VisDrone val)")
    ax.set_title("(c) 精度-延迟帕累托（部署口径）")
    ax.grid(alpha=0.3)
    ax.set_xlim(2.15, 7.7)
    ax.set_ylim(0.170, 0.280)
    if compact:
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.text(0.98, 0.02, "* 延迟为推算值，待重测", transform=ax.transAxes,
            fontsize=7, color="#666666", va="bottom", ha="right")


def fig_gt_size(ax):
    ds = DS_ROOT
    if not (ds / "val.txt").exists():
        ax.text(0.5, 0.5, "需要 VisDrone 数据集\n（准备方式见 README）", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        return
    from collections import Counter
    from PIL import Image
    sizes = {}
    for line in open(ds / "val.txt"):
        ip = ds / line.strip()[2:]
        with Image.open(ip) as im:
            w, h = im.size
        lp = ds / "labels/val" / (ip.stem + ".txt")
        if not lp.exists():
            continue
        for ln in open(lp):
            _, _, _, bw, bh = ln.split()
            sizes.setdefault("orig", []).append(np.sqrt(float(bw) * w * float(bh) * h))
            for tag, imgsz in (("640", 640), ("960", 960)):
                s = min(imgsz / w, imgsz / h)
                sizes.setdefault(tag, []).append(np.sqrt(float(bw) * w * float(bh) * h) * s)
    bins = [0, 8, 12, 16, 24, 32, 64, 1e9]
    labels = ["0-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
    x = np.arange(len(labels))
    w = 0.38
    for i, (tag, lab) in enumerate((("640", "letterbox @640"), ("960", "letterbox @960"))):
        arr = np.array(sizes[tag])
        pct = [np.mean((arr >= lo) & (arr < hi)) * 100 for lo, hi in zip(bins[:-1], bins[1:])]
        ax.bar(x + (i - 0.5) * w, pct, w, label=lab, color=["#1f77b4", "#d62728"][i])
        for xi, p in zip(x + (i - 0.5) * w, pct):
            if p > 1:
                ax.text(xi, p + 0.6, f"{p:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("% of GT")
    ax.set_xlabel("object size at network input (px)")
    ax.set_title("(d) 分辨率把目标'变大'：<8px 占比 31.1% → 13.4%")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------- 权衡图
def panel_tradeoff(ax, xkey: str, xlabel: str):
    key = "lat" if xkey == "lat" else "gf"
    pts = sorted(POINTS.items(), key=lambda kv: kv[1][key])
    ax.plot([p[1][key] for p in pts], [p[1]["ap"] for p in pts], "-",
            color="#444444", alpha=0.55, linewidth=1.6, zorder=1)
    # 手工指定方位：旧图里 baseline@640 与 P2-pruned 的标注会互相压字
    off = {
        "baseline@640":  dict(xy=(-10, -6), ha="right", va="top"),
        "P2-pruned":     dict(xy=(10, -16), ha="left", va="top"),
        "P2@640":        dict(xy=(10, 10), ha="left", va="bottom"),
        "baseline@960":  dict(xy=(10, 10), ha="left", va="bottom"),
        "P2@960":        dict(xy=(-10, 6), ha="right", va="bottom"),
    }
    for name, p in pts:
        x = p[key]
        ax.scatter([x], [p["ap"]], s=150, color=p["color"], zorder=3,
                   edgecolor="white", linewidth=1.2)
        if p["ap_md"]:
            ax.scatter([x], [p["ap_md"]], s=70, facecolor="none", edgecolor=p["color"],
                       linewidth=1.6, zorder=3)
            ax.plot([x, x], [p["ap"], p["ap_md"]], "-", color=p["color"], alpha=0.5, linewidth=1)
        tag = "*" if p["recheck"] else ""
        first = name.splitlines()[0]
        o = off.get(first, dict(xy=(10, 8), ha="left", va="bottom"))
        label = f"{name}{tag}\n{p['ap']:.3f}" + (f" → {p['ap_md']:.3f}" if p["ap_md"] else "")
        ax.annotate(label, (x, p["ap"]), textcoords="offset points",
                    xytext=o["xy"], ha=o["ha"], va=o["va"], fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mAP50-95 (VisDrone val)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.165, 0.290)
    if key == "gf":
        ax.set_xlim(4.0, 17.5)
    else:
        ax.set_xlim(2.3, 7.6)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures_v2")
    ap.add_argument("--only", choices=["all", "figs", "tradeoff"], default="all",
                    help="figs=四联图+fig1..4；tradeoff=精度-速度权衡系列")
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    print(f"oracle 日志根目录: {ORACLE_ROOT}")
    print(f"数据集根目录:      {DS_ROOT}")
    print(f"输出目录:          {out}\n")

    written: list[Path] = []

    def _save(fg, path: Path, dpi: int):
        fg.savefig(path, dpi=dpi)
        written.append(path)

    if a.only in ("all", "figs"):
        # 四联图 + 单图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig_size_missrate(axes[0, 0])
        fig_oracle(axes[0, 1])
        fig_pareto(axes[1, 0], compact=True)
        fig_gt_size(axes[1, 1])
        fig.suptitle("VisDrone 小目标检测：误差分解 → 表示侧优化（YOLO26n, 100 epoch，效率为 fuse 后口径）",
                     fontsize=14)
        fig.tight_layout()
        _save(fig, out / "误差分解四联图.png", 140)
        plt.close(fig)

        for name, fn in (("分尺寸漏检率", fig_size_missrate), ("误差分解四象限", fig_oracle),
                         ("精度延迟帕累托", fig_pareto), ("误差分解_真值尺寸分布", fig_gt_size)):
            f, ax = plt.subplots(figsize=(7, 5))
            fn(ax)
            if name == "精度延迟帕累托":
                f.tight_layout(rect=(0, 0.03, 1, 1))
            else:
                f.tight_layout()
            _save(f, out / f"{name}.png", 150)
            plt.close(f)

    if a.only in ("all", "tradeoff"):
        # 精度-速度权衡（双面板 + 单面板）
        f, axes = plt.subplots(1, 2, figsize=(16, 6.2))
        panel_tradeoff(axes[0], "lat",
                       "单图前向延迟 (ms, RTX 4060 Laptop, FP32, batch 1, fuse 后) — 越左越好")
        axes[0].set_title("(a) 精度 - 延迟")
        panel_tradeoff(axes[1], "gf", "计算量 GFLOPs (fuse 后) — 越左越好")
        axes[1].set_title("(b) 精度 - 计算量")
        f.suptitle("VisDrone 小目标检测：精度-速度权衡（实心=同分辨率训练推理，空心=max_det=1000；* 为待重测点）",
                   fontsize=13)
        f.tight_layout(rect=(0, 0.04, 1, 1))
        f.text(0.5, 0.012, RECHECK_NOTE, ha="center", fontsize=8.5, color="#555555")
        _save(f, out / "精度与计算量权衡.png", 140)
        plt.close(f)

        f, ax = plt.subplots(figsize=(9.5, 6))
        panel_tradeoff(ax, "lat", "单图前向延迟 (ms, RTX 4060 Laptop, FP32, fuse 后) — 越左越好")
        ax.set_title("精度 - 速度折线图（VisDrone val, YOLO26n, 100 epoch, fuse 后口径）")
        f.tight_layout(rect=(0, 0.04, 1, 1))
        f.text(0.5, 0.012, RECHECK_NOTE, ha="center", fontsize=8, color="#555555")
        _save(f, out / "精度与延迟权衡.png", 150)
        plt.close(f)

    for p in written:
        print(f"saved {p}  ({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
