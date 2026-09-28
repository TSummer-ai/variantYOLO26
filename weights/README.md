# 预训练权重

三个主模型（VisDrone2019-DET，100 epoch，seed=0）。均为 `best.pt`，已剥离 optimizer/EMA，
每个约 5 MB。权重基于 ultralytics 预训练 `yolo26n.pt` 微调，遵循 **AGPL-3.0**（见根目录 `NOTICE.md`）。

| 文件 | 结构 | 训练分辨率 | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|---|---|
| `yolo26n-visdrone-base-640.pt` | P3–P5（官方 `yolo26.yaml`） | 640 | 0.328 | 0.182 | 基线 |
| `yolo26n-visdrone-p2-640.pt` | P2–P5（官方 `yolo26-p2.yaml`） | 640 | 0.349 | 0.198 | +1.6 AP |
| **`yolo26n-visdrone-p2-960.pt`** | P2–P5 | 960 | **0.440** | **0.263** | **最终模型** |

> 指标为 ultralytics `yolo val` 内部口径、`max_det=300`。开 `max_det=1000` 后最终模型为
> **mAP50 0.455 / mAP50-95 0.269**。COCO 协议（faster-coco-eval）下同一模型为 0.413 / 0.247。
> 两套口径在同一数据集上相差约 0.9 AP，引用时请注明。

## 用法

```bash
source activate_yolo.sh

# 推理
yolo predict model=weights/yolo26n-visdrone-p2-960.pt source=your_image.jpg imgsz=960

# 复现表格里的指标（需要先按 README 准备数据集）
yolo val model=weights/yolo26n-visdrone-p2-960.pt data=configs/visdrone.yaml imgsz=960 max_det=1000

# 精度/延迟实测
python scripts/bench_models.py --weights weights/yolo26n-visdrone-p2-960.pt --imgsz 960
```

⚠️ **推理分辨率必须与训练一致**（960 模型用 `imgsz=960`）。用 640 推理 960 训练的模型会掉点。

## 未包含的权重

SAR / DHCD 两个方向的 8 个变体权重（共约 42 MB）未入库 —— 它们是负结果，
用 `experiments/negative-results/patches/` 打补丁重训即可复现，对应的
`results/results_csv/` 里有完整训练曲线。如确需权重，向维护者索取。
