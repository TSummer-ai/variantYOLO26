#!/usr/bin/env python
"""多视角一致性 → 重打分：能否把候选池里的 TP 与 FP 分开？

背景（实测）：960 下切片把 0–8px 的"可检出率"从 15.3% 翻倍到 30.5%（机制），
但候选池比最佳单视角还多 +5.4 个点，AP 却没变 —— 说明这些被找到的目标**排不上分**。
本脚本验证：用"被几个视角支持"（一致性）重打分，能否把它们推上去。

两趟结构：pass1 缓存一致性（慢），pass2 扫权重方案（快）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probe_oracle_policy import _np_iou, load_gt_xyxy, name, to_coco  # noqa: E402
from probe_slicing import DS, build_gt, evaluate, merge_nms  # noqa: E402

VIEWS = [("p2_960", 960, "full"), ("p2_960", 960, "2x1"), ("p2_960", 960, "3x1"), ("p2_960", 960, "2x2")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/wang/DeepSeek/YOLO/oracle_policy_o3/n548")
    ap.add_argument("--stage", default="all", choices=["pass1", "pass2", "all"])
    a = ap.parse_args()
    CACHE = Path(a.cache)
    files = [l.strip() for l in open(DS / "val.txt")]
    out1 = CACHE / "consistency.pkl"

    if a.stage in ("pass1", "all"):
        P = {v: pickle.load(open(CACHE / f"{name(v).replace('@','_').replace('/','_')}.pkl", "rb")) for v in VIEWS}
        gts = [load_gt_xyxy(l) for l in files]
        rows = []            # (n_views_supporting, score, is_tp, size)
        percand = []         # 每图 (boxes, scores, classes, counts) 用于 pass2
        for i in range(len(files)):
            allb = np.concatenate([P[v][i][0] for v in VIEWS])
            alls = np.concatenate([P[v][i][1] for v in VIEWS])
            allc = np.concatenate([P[v][i][2] for v in VIEWS])
            if len(alls) == 0:
                percand.append((np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int)))
                continue
            b, s, c = merge_nms(allb, alls, allc, iou_thr=0.6)
            cnt = np.zeros(len(s), int)
            for v in VIEWS:
                vb, vs, vc = P[v][i]
                m = vs >= 0.001
                vb2, vc2 = vb[m], vc[m]
                if len(vb2) == 0:
                    continue
                iou = _np_iou(b, vb2)
                for j in range(len(b)):
                    same = vc2 == c[j]
                    if same.any() and iou[j][same].max() >= 0.5:
                        cnt[j] += 1
            gb, gc, _, _ = gts[i]
            sz = np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)) if len(b) else np.zeros(0)
            for j in range(len(s)):
                tp = False
                if len(gb):
                    iou = _np_iou(gb, b[j:j + 1]).ravel()
                    same = gc == c[j]
                    tp = bool(same.any() and iou[same].max() >= 0.5)
                rows.append((int(cnt[j]), float(s[j]), tp, float(sz[j])))
            percand.append((b, s, c, cnt))
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", flush=True)
        pickle.dump({"rows": rows, "percand": percand, "n": len(files)}, open(out1, "wb"))
        print(f"  ✅ pass1 完成: {len(rows)} 个候选")
        if a.stage == "pass1":
            return

    if a.stage in ("pass2", "all"):
        d = pickle.load(open(out1, "rb"))
        rows = np.array([(r[0], r[1], float(r[2]), r[3]) for r in d["rows"]])
        percand = d["percand"]
        print(f"\n=== A. TP/FP 与多视角一致性（{len(rows)} 个候选）===")
        print(f"{'支持视角数':>10}{'候选数':>9}{'占比':>8}{'其中TP':>9}{'TP率':>9}{'平均分数':>11}")
        for k in range(0, 5):
            m = rows[:, 0] == k
            if m.sum() == 0:
                continue
            print(f"{k:>10}{int(m.sum()):>9}{m.mean()*100:>7.1f}%{int(rows[m, 2].sum()):>9}"
                  f"{rows[m, 2].mean()*100:>8.1f}%{rows[m, 1].mean():>11.3f}")
        print("\n=== B. 分尺寸：TP 与 FP 的'被≥2视角支持'比例 ===")
        for lo, hi, lab in [(0, 8, "0-8"), (8, 16, "8-16"), (16, 32, "16-32"), (32, 1e9, ">=32")]:
            m = (rows[:, 3] >= lo) & (rows[:, 3] < hi)
            if m.sum() < 50:
                continue
            r = rows[m]
            t = r[r[:, 2] == 1][:, 0]; f = r[r[:, 2] == 0][:, 0]
            print(f"  [{lab:>5}] 候选 {int(m.sum()):>6}  TP被≥2支持 {100*np.mean(t>=2):5.1f}%  |  FP被≥2支持 {100*np.mean(f>=2):5.1f}%"
                  f"   (分离度 {100*(np.mean(t>=2)-np.mean(f>=2)):+.1f} 点)")

        coco = build_gt(files)
        schemes = [("原分数", [1, 1, 1, 1, 1]),
                   ("线性 w=cnt+1", None),
                   ("弱共识 w=[0.5,0.8,1,1.2,1.4]", [0.5, 0.8, 1, 1.2, 1.4]),
                   ("强共识 w=[0.3,0.6,1,1.5,2]", [0.3, 0.6, 1, 1.5, 2]),
                   ("只要≥2 w=[0,0,1,1,1]", [0, 0, 1, 1, 1]),
                   ("只要≥3 w=[0,0,0,1,1]", [0, 0, 0, 1, 1]),
                   ("只要4视角 w=[0,0,0,0,1]", [0, 0, 0, 0, 1])]
        print("\n=== C. 一致性重打分的 AP ===")
        print(f"  {'权重方案':>30}{'mAP50-95':>12}{'mAP50':>10}")
        for wn, w in schemes:
            hyb = []
            for (b, s, c, cnt) in percand:
                if len(s) == 0:
                    hyb.append((b, s, c)); continue
                ww = w if w is not None else (cnt + 1).astype(float)
                s2 = np.array([s[j] * float(ww[min(int(cnt[j]), len(ww) - 1)]) for j in range(len(s))])
                hyb.append((b, s2, c))
            p95, p50 = evaluate(coco, to_coco(hyb))
            print(f"  {wn:>30}{p95:>12.4f}{p50:>10.4f}")
        print("\n  对照：手工尺寸门控规则 = 0.2750   |   per-image oracle 上界 = 0.2768")


if __name__ == "__main__":
    main()
