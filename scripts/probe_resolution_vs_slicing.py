#!/usr/bin/env python
"""任务3：等算力下"更高分辨率整图" vs "960 + 3x1 切片融合" 的对照。

动机：原 Pareto 表只有 640/整图(1.00) 与 960/整图(2.25)，缺 1280~1920 整图点。
      若在 9.00 算力（= 960 上切 3 列、共 4 次前向）附近，高分辨率整图更好，
      则"买分辨率比买切片划算"需要改写。

做法：
  A. 精度：对 p2_960 权重，在 imgsz in {640,960,1280,1600,1920} 上做整图全量 val（COCO 协议）。
     注意全部为"零训练"的推理分辨率变化，与 640训/960推 同一性质。
  B. 效率：融合(fuse)后的纯前向延迟、端到端 predict 延迟、GFLOPs。
  C. 切片管线（960 + 3x1 + 尺寸门控）的 AP 与端到端延迟，作为对照臂。

用法:
  python probe_resolution_vs_slicing.py --weights runs/visdrone/p2_960/weights/best.pt \
         --imgszs 640 960 1280 1600 1920 --limit 0 --out res_vs_slice.json
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
from probe_slicing import DS, build_gt, evaluate, merge_nms, predict_full, predict_sliced  # noqa: E402

ROOT = Path(os.environ.get("YOLO_ROOT", "/home/wang/DeepSeek/YOLO"))
ASSET = ROOT / "ultralytics/ultralytics/assets/bus.jpg"
BIG_PX = 48.0
OVERLAP = 0.3
MERGE_IOU = 0.6


def fused_model(weights, device="0"):
    from ultralytics import YOLO
    y = YOLO(weights)
    m = y.model.to(f"cuda:{device}").eval()
    m.fuse()                      # 移除推理时无用的 one2one 分支
    return y, m


def bench_forward(m, imgsz, n=30, warm=10):
    dev = next(m.parameters()).device
    x = torch.rand(1, 3, imgsz, imgsz, device=dev)
    with torch.no_grad():
        for _ in range(warm):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            m(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_predict(y, imgsz, n=20, warm=5, device="0"):
    for _ in range(warm):
        y.predict(str(ASSET), imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        y.predict(str(ASSET), imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def bench_slice_pipeline(y, path, imgsz, device, n=10, warm=2):
    """整图 960 + 3x1 切片 + 尺寸门控 融合 的端到端单图延迟。"""
    def one():
        bf, sf, cf = predict_full(y, path, imgsz, 0.001, 0.7, device)
        bt, st, ct = predict_sliced(y, path, imgsz, 0.001, 0.7, device, (3, 1), OVERLAP)
        if len(bt):
            tsz = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
            kt = tsz < BIG_PX
        else:
            kt = np.zeros(0, bool)
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


def full_val_ap(y, files, imgsz, device, tag=""):
    preds = []
    for i, line in enumerate(files):
        ip = DS / line[2:]
        b, s, c = predict_full(y, ip, imgsz, 0.001, 0.7, device)
        for j in range(len(s)):
            preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                          "bbox": [float(b[j, 0]), float(b[j, 1]),
                                   float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                          "score": float(s[j])})
        if i % 100 == 0:
            print(f"  [{tag}] {i}/{len(files)}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/visdrone/p2_960/weights/best.pt"))
    ap.add_argument("--imgszs", type=int, nargs="+", default=[640, 960, 1280, 1600, 1920])
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "res_vs_slice.json"))
    a = ap.parse_args()

    files = [l.strip() for l in open(DS / "val.txt")]
    if a.limit:
        files = files[: a.limit]
    print(f"val 图数 = {len(files)}", flush=True)
    coco_gt = build_gt(files)

    y, m = fused_model(a.weights, a.device)
    try:
        from ultralytics.utils.torch_utils import get_flops
    except Exception:
        get_flops = None

    res = {"weights": a.weights, "n_img": len(files), "rows": []}

    # ---------- A. 整图精度 + 效率 ----------
    for imgsz in a.imgszs:
        row = {"imgsz": imgsz, "compute": round((imgsz / 640.0) ** 2, 2)}
        print(f"\n=== 整图 imgsz={imgsz} (算力 {row['compute']}) ===", flush=True)
        try:
            torch.cuda.empty_cache()
            t0 = time.perf_counter()
            preds = full_val_ap(y, files, imgsz, a.device, tag=f"full{imgsz}")
            p95, p50 = evaluate(coco_gt, preds)
            row.update({"ap50_95": round(p95, 4), "ap50": round(p50, 4),
                        "n_box": len(preds), "eval_sec": round(time.perf_counter() - t0, 1)})
            print(f"  AP50-95={p95:.4f} AP50={p50:.4f}  boxes={len(preds)}", flush=True)
        except RuntimeError as e:
            row.update({"error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"  !! 失败：{row['error']}", flush=True)
            torch.cuda.empty_cache()
            res["rows"].append(row)
            continue
        try:
            row["gflops"] = round(float(get_flops(m, imgsz)), 2) if get_flops else None
        except Exception:
            row["gflops"] = None
        try:
            row["fwd_ms"] = round(bench_forward(m, imgsz), 3)
            row["e2e_ms"] = round(bench_predict(y, imgsz, device=a.device), 3)
            row["fps_e2e"] = round(1000.0 / row["e2e_ms"], 1)
            print(f"  GFLOPs={row['gflops']} 前向={row['fwd_ms']}ms 端到端={row['e2e_ms']}ms "
                  f"({row['fps_e2e']} FPS)", flush=True)
        except RuntimeError as e:
            row["bench_error"] = str(e)[:200]
            print(f"  !! 测速失败：{row['bench_error']}", flush=True)
        res["rows"].append(row)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)

    # ---------- C. 切片管线（960 + 3x1 + 尺寸门控）精度与延迟 ----------
    print("\n=== 切片管线 960 + 3x1 + 尺寸门控（4 次前向）===", flush=True)
    sl = {"tag": "slice_3x1_gated@960", "forwards": 4, "compute": 9.00}
    preds = []
    for i, line in enumerate(files):
        ip = DS / line[2:]
        bf, sf, cf = predict_full(y, ip, 960, 0.001, 0.7, a.device)
        bt, st, ct = predict_sliced(y, ip, 960, 0.001, 0.7, a.device, (3, 1), OVERLAP)
        if len(bt):
            tsz = np.sqrt((bt[:, 2] - bt[:, 0]).clip(0) * (bt[:, 3] - bt[:, 1]).clip(0))
            kt = tsz < BIG_PX
        else:
            kt = np.zeros(0, bool)
        b = np.concatenate([bf, bt[kt]]) if len(bf) or kt.any() else np.zeros((0, 4))
        s = np.concatenate([sf, st[kt]]) if len(bf) or kt.any() else np.zeros(0)
        c = np.concatenate([cf, ct[kt]]) if len(bf) or kt.any() else np.zeros(0, int)
        if len(b):
            b, s, c = merge_nms(b, s, c, MERGE_IOU)
        for j in range(len(s)):
            preds.append({"image_id": i, "category_id": int(c[j]) + 1,
                          "bbox": [float(b[j, 0]), float(b[j, 1]),
                                   float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                          "score": float(s[j])})
        if i % 100 == 0:
            print(f"  [slice] {i}/{len(files)}", flush=True)
    p95, p50 = evaluate(coco_gt, preds)
    sl.update({"ap50_95": round(p95, 4), "ap50": round(p50, 4), "n_box": len(preds)})
    print(f"  AP50-95={p95:.4f} AP50={p50:.4f}", flush=True)
    try:
        sl["e2e_ms"] = round(bench_slice_pipeline(y, DS / files[0][2:], 960, a.device), 3)
        sl["fps_e2e"] = round(1000.0 / sl["e2e_ms"], 1)
        print(f"  端到端={sl['e2e_ms']}ms ({sl['fps_e2e']} FPS)", flush=True)
    except RuntimeError as e:
        sl["bench_error"] = str(e)[:200]
    res["slice"] = sl
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print(f"{'配置':>26}{'算力':>8}{'AP50-95':>10}{'AP50':>8}{'GFLOPs':>9}{'端到端ms':>10}{'FPS':>7}")
    for r in res["rows"]:
        if "ap50_95" not in r:
            print(f"{'整图@'+str(r['imgsz']):>26}{r['compute']:>8.2f}{'FAIL':>10}")
            continue
        print(f"{'整图@'+str(r['imgsz']):>26}{r['compute']:>8.2f}{r['ap50_95']:>10.4f}"
              f"{r['ap50']:>8.4f}{str(r.get('gflops')):>9}{str(r.get('e2e_ms')):>10}{str(r.get('fps_e2e')):>7}")
    print(f"{'切片 3x1+门控@960':>26}{sl['compute']:>8.2f}{sl['ap50_95']:>10.4f}{sl['ap50']:>8.4f}"
          f"{'—':>9}{str(sl.get('e2e_ms')):>10}{str(sl.get('fps_e2e')):>7}")
    print("=" * 78)
    print(f"已写入 {a.out}")


if __name__ == "__main__":
    main()
