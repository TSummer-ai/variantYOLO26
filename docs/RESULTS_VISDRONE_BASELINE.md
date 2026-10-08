# VisDrone2019-DET × YOLO26n 基线 + 错误分析报告

日期：2026-09-25 ｜ 代码：`./ultralytics` @ v8.4.160（`9d25f80da`）｜ 权重：COCO 预训练 `yolo26n.pt` 微调
数据：VisDrone2019-DET（6,471 train / 548 val，10 类，34.3 万 / 3.88 万标注）
训练：100 epoch，imgsz=640，batch=16，MuSGD(lr=0.01)，seed=0，`close_mosaic=10`，约 63 s/epoch（RTX 4060 Laptop 8G）

---

## 1. 基线指标（best.pt，548 图 / 38,759 个 GT）

| 头 | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **one2many + NMS（默认）** | 0.464 | 0.348 | 0.328 | **0.182** |
| one2one e2e（`nms=False`） | 0.452 | 0.347 | 0.320 | **0.177** |
| Δ（e2e − NMS） | −0.012 | −0.001 | −0.008 | **−0.005** |

训练过程中最优：epoch 79，mAP50 = 0.338，mAP50-95 = 0.1855（与独立 val 一致，±0.004）。

### 分类别 mAP50-95

| 类别 | GT 数 | NMS | e2e | | 类别 | GT 数 | NMS | e2e |
|---|---|---|---|---|---|---|---|---|
| pedestrian | 8844 | 0.151 | 0.148 | | tricycle | 1045 | 0.115 | 0.106 |
| people | 5125 | 0.098 | 0.0985 | | awning-tricycle | 532 | 0.061 | 0.060 |
| bicycle | 1287 | 0.0339 | 0.0304 | | bus | 251 | 0.281 | 0.273 |
| car | 14064 | 0.492 | 0.485 | | motor | 4886 | 0.159 | 0.155 |
| van | 1975 | 0.241 | 0.235 | | truck | 750 | 0.192 | 0.180 |

> 注：VisDrone 不是 COCO 数据集，Ultralytics 内部指标不输出 APs/APm/APl 分档，因此用下面的**分尺寸召回**替代（更直接回答"漏检发生在哪"）。

---

## 2. 错误分析（conf=0.25，匹配 IoU=0.5，全量 548 图）

### 2.1 分尺寸召回（原图像素，√area）——最关键的一张表

| 尺寸(px) | GT 数 | 占比 | NMS 召回 | e2e 召回 | e2e−NMS |
|---|---|---|---|---|---|
| 0–8 | 2,424 | 6.3% | **0.061** | 0.050 | −0.011 |
| 8–16 | 9,545 | 24.6% | **0.222** | 0.194 | **−0.028** |
| 16–32 | 14,608 | 37.7% | **0.439** | 0.388 | **−0.050** |
| 32–64 | 9,004 | 23.2% | 0.681 | 0.638 | −0.043 |
| ≥64 | 3,178 | 8.2% | 0.849 | 0.817 | −0.032 |

- **<32 px 的目标占 68.6%（26,577/38,759），但召回只有 0.061 / 0.222 / 0.439**
- 几何视角：letterbox 到 640 后，GT 中位尺寸只有 **11.3 px**；**92.4% 属于 COCO-small，31% 连 8 px 都不到**（P3/8 的一个格子就是 8 px）
- 结论：**基线的瓶颈是"小目标定位 + 召回"，不是分类**（大目标召回已 0.85）

### 2.2 分类别召回（NMS 头）

| 类别 | GT | 召回 | | 类别 | GT | 召回 |
|---|---|---|---|---|---|---|
| car | 14064 | 0.726 | | motor | 4886 | 0.361 |
| pedestrian | 8844 | 0.325 | | people | 5125 | 0.246 |
| tricycle | 1045 | 0.199 | | awning-tricycle | 532 | 0.107 |
| truck | 750 | 0.280 | | **bicycle** | 1287 | **0.099** |

行人/人/摩托合计 18,855 个 GT（48.6%），召回却只有 0.25–0.36；自行车、带篷三轮车几乎检测不到。

### 2.3 误检与混淆（NMS 头，FP 共 9,827；e2e 头 6,828）

| 真值 → 预测 | 次数 | 说明 |
|---|---|---|
| car → car | 2,258 | 定位不达标（IoU<0.5）的重复/漂移框 |
| **van → car** | 1,099 | van/car 混淆，最大单项错误 |
| car → van | 500 | 同上，反向 |
| people → pedestrian | 433 | 语义近邻混淆 |
| pedestrian → people | 345 | 同上 |
| car → pedestrian | 250 | 车辆误判为行人（尺度相近） |
| people → motor | 235 | 人/摩托混淆 |

---

## 3. 补充发现（可写进 motivation）

- **van↔car 混淆（1,599 次 FP）** 与 bicycle/awning-tricycle 的 AP≈0.03，是**分类侧**瓶颈，与尺度无关；
- 因此故事线建议表述为："本文针对小目标**定位与召回**（占错误主体），而非类别混淆"。

---

## 4. 基线是否够用？

- 当前是 **100 epoch、单尺度 640、无 Objects365 预训练** 的轻量基线，属于"能开消融"的水平，不是"能发表"的水平；
- 论文定稿前建议：**① 300 epoch 或 ② 直接换 yolo26s** 作为最终 baseline，并把两者都记录；
- 对比公平性提醒：公开的 VisDrone 小目标改进（PSSL-YOLO / DSG-YOLO26 / Aero-YOLO 等）多在不同 epoch/分辨率下报告，**跨设置比数字无效**，要么同设置复现其思路（P2 + 注意力），要么只引数字并标注差异。

---

## 5. 工程坑（本次踩到，后续复用）

1. **`model.predict(一个长图片列表)` 的显存会随图片数线性膨胀**：548 张一次传入时，第一张图的推理就要占 2.1 GB（n=256）并最终 OOM；原因是单次调用会按 source 长度保留预分配（与 `rect` 无关）。**解决：分块预测（每 32 张一次）+ 块间 `empty_cache()`**，现在全量分析稳定跑通。
2. **沙箱禁止 `/dev/shm` 的 POSIX 信号量**，会让 Ultralytics 的 `ThreadPool`（标签扫描 `cache_labels`）直接 `EACCES` 崩溃；CUDA 初始化也需写 `/proc/self/task/<tid>/comm`。→ 所有 yolo 跑批都需要放宽沙箱（当前 session 已全开放，不再需要逐次审批）。
3. **AMP 自检会去 GitHub 下载一份 yolo26n.pt 到 `weights_dir`**（15 KB/s，6 分钟）：提前把权重放到 `ultralytics/weights/` 即可跳过。

---

## 6. 产物清单

| 文件 | 内容 |
|---|---|
| `runs/visdrone/base_n_640/` | 基线训练产物（`results.csv`、曲线、`weights/best.pt`） |
| `val_visdrone_nms.log` / `val_visdrone_e2e.log` | 两套头的独立 val 指标 |
| `runs/visdrone/analysis/summary.md` | 错误分析结论表 |
| `runs/visdrone/analysis/size_recall.png` | 分尺寸召回对比图 |
| `runs/visdrone/analysis/per_class.csv` | 分类别召回（NMS / e2e） |
| `runs/visdrone/analysis/gt_size_stats.txt` | GT 尺寸分布统计（原图 vs @640） |
| `analyze_visdrone.py` / `gt_size_stats.py` / `run_train_exp.sh` | 分析脚本与通用消融训练入口 |
