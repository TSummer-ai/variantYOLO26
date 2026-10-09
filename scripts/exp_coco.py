#!/usr/bin/env python
"""跨数据集泛化：COCO val2017 上的两个实验。

变体 A：**VisDrone 模型 → COCO val2017**（6 类映射，标准跨数据集口径）
    VisDrone idx → COCO cat_id: 0 pedestrian→1 person, 2 bicycle→2, 3 car→3,
                                9 motor→4 motorcycle, 8 bus→6, 5 truck→8
    其余 VisDrone 类（people/van/tricycle/awning-tricycle）在 COCO 无对应，丢弃。

变体 B：**COCO 模型 → COCO val2017**，对比 整图 vs MV-Fuse（SSG + CVA）
    回答问题：这套推理期方法是否只在 VisDrone 有效。

用法: python exp_coco.py --what A,B --limit 0
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from probe_oracle_policy import _np_iou  # noqa: E402
from probe_slicing import evaluate, merge_nms, tile_starts  # noqa: E402

ROOT = Path("/home/wang/DeepSeek/YOLO")
COCO = ROOT / "datasets/coco"
ANN = COCO / "annotations/instances_val2017.json"
OUT = ROOT / "exp_coco.json"
W_VIS = ROOT / "runs/visdrone/p2_960/weights/best.pt"
W_COCO = ROOT / "weights/yolo26n.pt"
MAP_VIS2COCO = {0: 1, 2: 2, 3: 3, 9: 4, 8: 6, 5: 8}      # VisDrone idx -> COCO cat_id
KEEP_CATS = set(MAP_VIS2COCO.values())
W_SCHEME = [0.3, 0.6, 1.0, 1.5, 2.0, 2.5]
BIG_PX, OVER = 48.0, 0.2


def idx2cocoid():
    """模型输出索引 -> COCO 真实 category_id（COCO 的 id 不连续，必须用类名搭桥）。"""
    import yaml
    import ultralytics
    y = Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml"
    names = yaml.safe_load(open(y))["names"]                 # {idx: name}
    cats = json.load(open(ANN))["categories"]                # [{id, name}]
    n2i = {c["name"]: c["id"] for c in cats}
    return {i: n2i[n] for i, n in names.items() if n in n2i}


def load_coco_gt(cats=None, image_ids=None):
    """GT 必须与"实际跑过的图"严格一致，否则未跑图的 GT 会稀释 AP。"""
    d = json.load(open(ANN))
    keep_img = set(image_ids) if image_ids is not None else None
    imgs = [{"id": im["id"], "file_name": im["file_name"], "width": im["width"], "height": im["height"]}
            for im in d["images"] if (keep_img is None or im["id"] in keep_img)]
    if cats is None:
        anns = [{"id": a["id"], "image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
                for a in d["annotations"] if (keep_img is None or a["image_id"] in keep_img)]
        categories = [{"id": c["id"], "name": c["name"]} for c in d["categories"]]
    else:
        anns = [{"id": a["id"], "image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"]}
                for a in d["annotations"] if a["category_id"] in cats
                and (keep_img is None or a["image_id"] in keep_img)]
        categories = [{"id": c["id"], "name": c["name"]} for c in d["categories"] if c["id"] in cats]
    return {"images": imgs, "annotations": anns, "categories": categories}


def paths_and_ids(limit=0, only_with_ann=None):
    d = json.load(open(ANN))
    imgs = d["images"]
    if only_with_ann is not None:
        ok = {a["image_id"] for a in d["annotations"] if a["category_id"] in only_with_ann}
        imgs = [im for im in imgs if im["id"] in ok]
    if limit:
        imgs = imgs[:limit]
    return [str(COCO / "images/val2017" / im["file_name"]) for im in imgs], [im["id"] for im in imgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="A,B")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--imgsz-a", type=int, default=960)
    ap.add_argument("--imgsz-b", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=200, help="每次交给 predict 的图数（防 OOM）")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    from ultralytics import YOLO
    res = {}

    # ---------------- 变体 A：VisDrone 模型 → COCO（6 类） ----------------
    if "A" in a.what:
        print("=== A. VisDrone 模型 → COCO val2017（6 类映射）===", flush=True)
        paths, ids = paths_and_ids(a.limit)          # 全部图（不再只挑含 6 类的图）
        gt = load_coco_gt(KEEP_CATS, image_ids=ids)  # GT 与实际跑过的图严格一致
        print(f"  评测图数 {len(paths)}，GT 标注 {len(gt['annotations'])}", flush=True)
        m = YOLO(str(W_VIS))
        dt = []
        t0 = time.time()
        # 逐图 predict：把图片列表交给 predict 会导致显存异常累积（实测 OOM），本项目其它脚本一律用逐图
        for n, pp in enumerate(paths):
            r = m.predict(pp, imgsz=a.imgsz_a, conf=0.001, iou=0.7, max_det=3000,
                          device=a.device, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                sc = r.boxes.conf.cpu().numpy()
                c = r.boxes.cls.cpu().numpy().astype(int)
                for j in range(len(sc)):
                    cc = MAP_VIS2COCO.get(int(c[j]))
                    if cc is None:
                        continue
                    dt.append({"image_id": ids[n], "category_id": cc,
                               "bbox": [float(b[j, 0]), float(b[j, 1]),
                                        float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                               "score": float(sc[j])})
            if n % 500 == 0:
                print(f"   {n}/{len(paths)}  ({time.time()-t0:.0f}s, dt={len(dt)})", flush=True)
                torch.cuda.empty_cache()
        p95, p50 = evaluate(gt, dt)
        res["A_VISDRONE_to_COCO_6cls"] = {"mAP50-95": p95, "mAP50": p50,
                                          "n_img": len(paths), "n_dt": len(dt)}
        print(f"  → mAP50-95 {p95:.4f}  mAP50 {p50:.4f}  (预测 {len(dt)} 个)", flush=True)

    # ---------------- 变体 B：COCO 模型 → COCO，整图 vs MV-Fuse ----------------
    if "B" in a.what:
        print("=== B. COCO 模型 → COCO val2017：整图 vs MV-Fuse ===", flush=True)
        paths, ids = paths_and_ids(a.limit)
        gt = load_coco_gt(None, image_ids=ids)
        print(f"  评测图数 {len(paths)}，GT 标注 {len(gt['annotations'])}", flush=True)
        m = YOLO(str(W_COCO))
        I2C = idx2cocoid()
        print(f"  类别映射: 模型索引->COCO id 共 {len(I2C)} 项（前 3: {list(I2C.items())[:3]}）", flush=True)
        # (i) 整图
        full = {}
        t0 = time.time()
        for n, pp in enumerate(paths):
            r = m.predict(pp, imgsz=a.imgsz_b, conf=0.001, iou=0.7, max_det=3000,
                          device=a.device, verbose=False)[0]
            full[ids[n]] = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int)) \
                if (r.boxes is not None and len(r.boxes)) else (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
            if n % 500 == 0:
                print(f"   [full] {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)
                torch.cuda.empty_cache()
        dt = [{"image_id": i, "category_id": I2C[int(c[j])],
               "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
               "score": float(s[j])} for i, (b, s, c) in full.items() for j in range(len(s))]
        p95, p50 = evaluate(gt, dt)
        res["B_COCO_full"] = {"mAP50-95": p95, "mAP50": p50}
        print(f"  → 整图: mAP50-95 {p95:.4f}  mAP50 {p50:.4f}", flush=True)

        # (ii) 切片 + 尺寸门控 + CVA
        tiles = {}
        t0 = time.time()
        for n, p in enumerate(paths):
            with Image.open(p) as im:
                im = im.convert("RGB"); Wd, Hd = im.size
            xs, tw = tile_starts(Wd, 3, OVER)
            bs, ss, cs = [], [], []
            for x in xs:
                crop = im.crop((x, 0, min(x + tw, Wd), Hd))
                r = m.predict(np.array(crop), imgsz=a.imgsz_b, conf=0.001, iou=0.7, max_det=3000,
                              device=a.device, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                bb = r.boxes.xyxy.cpu().numpy().copy(); bb[:, [0, 2]] += x
                bs.append(bb); ss.append(r.boxes.conf.cpu().numpy())
                cs.append(r.boxes.cls.cpu().numpy().astype(int))
            if bs:
                b = np.concatenate(bs); s = np.concatenate(ss); c = np.concatenate(cs)
                tiles[ids[n]] = merge_nms(b, s, c, iou_thr=0.6)
            else:
                tiles[ids[n]] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
            if n % 200 == 0:
                print(f"   [tiles] {n}/{len(paths)} ({time.time()-t0:.0f}s)", flush=True)

        def build(gate=True, cva=True):
            out = []
            for i in ids:
                fb, fs, fc = full[i]; tb, ts, tc = tiles[i]
                if gate and len(tb):
                    tsz = np.sqrt(np.clip(tb[:, 2] - tb[:, 0], 0, None) * np.clip(tb[:, 3] - tb[:, 1], 0, None))
                    k = tsz < BIG_PX
                    b, s, c = merge_nms(np.concatenate([fb, tb[k]]), np.concatenate([fs, ts[k]]),
                                        np.concatenate([fc, tc[k]]), iou_thr=0.6)
                else:
                    b, s, c = merge_nms(np.concatenate([fb, tb]), np.concatenate([fs, ts]),
                                        np.concatenate([fc, tc]), iou_thr=0.6) if len(tb) else (fb, fs, fc)
                if cva and len(s):
                    nv = np.zeros(len(s), int)
                    for vb, vs, vc in ((fb, fs, fc), (tb, ts, tc)):
                        msk = vs >= 0.001
                        vb2, vc2 = vb[msk], vc[msk]
                        if len(vb2) == 0:
                            continue
                        iou = _np_iou(b, vb2)
                        for j in range(len(b)):
                            same = vc2 == c[j]
                            if same.any() and iou[j][same].max() >= 0.5:
                                nv[j] += 1
                    s = s * np.array([W_SCHEME[min(int(x), len(W_SCHEME) - 1)] for x in nv])
                out += [{"image_id": i, "category_id": I2C[int(c[j])],
                         "bbox": [float(b[j, 0]), float(b[j, 1]), float(b[j, 2] - b[j, 0]), float(b[j, 3] - b[j, 1])],
                         "score": float(s[j])} for j in range(len(s))]
            return out

        for tag, gate, cva in (("B_COCO_gate+CVA", True, True), ("B_COCO_plain_union", False, False)):
            p95, p50 = evaluate(gt, build(gate, cva))
            res[tag] = {"mAP50-95": p95, "mAP50": p50}
            print(f"  → {tag}: mAP50-95 {p95:.4f}  mAP50 {p50:.4f}", flush=True)

    old = json.load(open(OUT)) if OUT.exists() else {}
    old.update(res)
    json.dump(old, open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"\n已写入 {OUT}")
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
