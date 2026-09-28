# VisDrone 小目标检测：误差分解驱动的表示侧优化

基于 **YOLO26n**（ultralytics 8.4.160）在 **VisDrone2019-DET** 上的小目标检测研究。
核心思路不是"再加一个模块"，而是**先用误差分解定位瓶颈，再对症下药**，并量化否证一批常见改进方向。

## 主要结果

VisDrone2019-DET val（548 图 / 38,759 GT），全部 100 epoch 同设定训练，`yolo val` 内部指标：

| 模型 | imgsz | max_det | P | R | mAP50 | **mAP50-95** | GFLOPs | 前向延迟 |
|---|---|---|---|---|---|---|---|---|
| baseline（P3–P5） | 640 | 300 | 0.464 | 0.348 | 0.328 | **0.182** | 5.9 | 4.57 ms |
| P2（P2–P5） | 640 | 300 | 0.492 | 0.356 | 0.349 | **0.198** | 7.7 | 5.88 ms |
| **P2（最终）** | **960** | **1000** | **0.549** | **0.446** | **0.455** | **0.269** | 17.6 | 8.96 ms |
| P2（NMS-free 分支） | 960 | 300 | 0.527 | 0.428 | 0.418 | 0.251 | 17.6 | 8.96 ms |

**相对基线：mAP50 +11.6，mAP50-95 +8.3**（同口径 `max_det=1000`）。

延迟在 RTX 4060 Laptop / FP32 / batch 1 下实测（100 次平均）：**8.96 ms ≈ 112 FPS**。

## 关键发现：增益来自特征分辨率，且随目标变小单调增强

漏检率（conf=0.001）：

| 尺寸（原图 √area） | GT 数 | baseline@640 | P2@640 | **P2@960** |
|---|---|---|---|---|
| 0–8 px | 2,424 | 0.835 | 0.718 | **0.587** |
| 8–16 px | 9,545 | 0.593 | 0.495 | **0.380** |
| 16–32 px | 14,608 | 0.364 | 0.321 | **0.237** |
| 32–64 px | 9,004 | 0.193 | 0.178 | **0.131** |
| ≥64 px | 3,178 | 0.097 | 0.095 | **0.086** |
| **总体召回** | 38,759 | 0.612 | 0.663 | **0.743** |

改善幅度**随目标变小单调增大、大目标几乎不动**（0–8px 档 −24.8 点，≥64px 档仅 −1.1 点）。
若增益来自正则化/参数增加等无关因素，大目标档也应改善 —— 三个配置构成一致的因果链。

### 分辨率收益可以拆开

| 来源 | mAP50-95 | 说明 |
|---|---|---|
| baseline@640 | 0.182 | |
| + **仅提高推理分辨率**（640 训练 / 960 推理，不重训） | 0.224 | **+4.2**，零训练成本 |
| + 960 训练 + P2 结构 | 0.269 | +4.5（这两项目前未分离，见「待补实验」） |

## 误差分解（oracle 四象限）

用 `scripts/oracle_decompose.py` 把 AP 损失拆成定位/分类/漏检/误检（COCO 协议）：

| 理想化 | baseline@640 | P2@640 | P2@960 |
|---|---|---|---|
| 真实 | 0.1731 | 0.1876 | 0.2474 |
| 定位完美 | 0.3047 (+0.132) | 0.3250 (+0.137) | 0.4125 (+0.165) |
| 分类完美 | 0.2741 (+0.101) | 0.2925 (+0.105) | 0.3604 (+0.113) |
| **不漏检** | **0.7547 (+0.582)** | 0.7387 (+0.551) | 0.7035 (+0.456) |
| 无误检 | 0.2310 (+0.058) | 0.2481 (+0.061) | 0.3157 (+0.068) |

**三个配置下漏检始终占可回收 AP 的 55–71%**，分类与误检合计不足 25%。
⇒ 在 VisDrone 上，任何以"类别混淆"为靶子的改进（层次软标签、难例挖掘等）天花板都在 +0.1 AP 以内。

## 附带发现

**`max_det` 协议问题**：截断发生在 NMS **之后**（`ultralytics/utils/nms.py`），所以调大几乎不增加计算：

| max_det | baseline | P2@640 | P2@960 |
|---|---|---|---|
| 300（多数文献默认） | 0.328 / 0.182 | 0.349 / 0.198 | 0.440 / 0.263 |
| **1000** | **0.339 / 0.186** | **0.361 / 0.203** | **0.455 / 0.269** |

**+0.4~0.6 AP / +1.1~1.5 mAP50，零训练成本**，且 P/R 几乎不变 → 增益纯粹来自"低分真阳性不再被截断"。

## 已量化否证的方向（负结果）

| 方向 | 结论 | 证据 |
|---|---|---|
| 尺度自适应回归重加权（L1） | +0.3 AP，单种子噪声内 | `docs/RESULTS_SAR_B.md` |
| 同上 + IoU 项重加权 | 无效（IoU 本身已尺度不敏感） | 同上 |
| 双头一致性蒸馏（DHCD） | 前提错误：e2e 头"召回缺口"是 conf 阈值假象 | `docs/RESULTS_DHCD_A.md` |
| 子网格目标标签分配修正 | 多余：分配器已把小于 `stride_val` 的框边垫到 16px，**2px 目标也有 432 个候选 anchor** | `docs/RESULTS_ORACLE.md` |
| 训练时挂 P2、推理摘掉（零成本辅助头） | 不是零成本（+12% FLOPs），且小目标召回反而更低 | `docs/RESULTS_FINAL.md` |
| 轻量 P2 / P2 只做定位 | 砍宽度省不下（9.6→8.9 GFLOPs）；分类借 P3 会崩 13 个点 | 同上 |
| 层次化软标签 | 天花板仅 +0.02~0.07 AP（由 oracle 推出，未训练即否决） | 同上 |
| 检测预算按尺寸分档 | 无收益，全局 top-k 已最优 | `scripts/probe_budget.py` |

**共性结论**：损失/训练/分配/后处理侧的余量已被穷尽，**唯一有效的是表示侧**（细粒度特征层、输入分辨率）。

前两个方向的代码以 patch 形式保留在 `experiments/negative-results/patches/`，供复现。

## 环境

- Python 3.12.3、CUDA 12.6、单卡 **RTX 4060 Laptop (8GB)**
- ultralytics 8.4.160、torch 2.14.0+cu126、tensorboard 2.21.0、faster-coco-eval 1.8.0
- 完整依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
source activate_yolo.sh          # 设置缓存目录与 PYTHONPATH
```

## 数据准备

```bash
# 1. 下载 VisDrone2019-DET train/val（官方站点或 Kaggle）
# 2. 转换标注（⚠️ 注意 category_id 的 0/1-based 偏移，脚本里已处理）
bash scripts/prepare_visdrone.sh
# 3. 检查 configs/visdrone.yaml 里的 path 指向你的数据目录
```

## 复现主要结果

```bash
# 三个主模型（每个约 2~9 小时，取决于 imgsz）
bash scripts/run_train_baseline.sh                    # baseline@640
bash scripts/run_p2.sh                                # P2@640
bash scripts/run_p2_960.sh                            # P2@960（显存紧张，batch=4）

# 评估（两个头：NMS / NMS-free）
bash scripts/run_post_baseline.sh

# 误差分解 + 分尺寸召回（核心分析）
bash scripts/run_oracle.sh

# 出图（不需要数据集也能出前三个面板）
python scripts/make_figures.py
python scripts/make_tradeoff_figure.py
python scripts/visualize_models.py
```

## 仓库结构

```
├── activate_yolo.sh              # 环境自举（缓存目录、PYTHONPATH）
├── requirements.txt
├── configs/
│   ├── visdrone.yaml             # 数据集配置模板
│   └── models/                   # 自建 yaml（轻量 P2 变体，未训练）
├── scripts/                      # 全部训练/评估/分析/出图脚本
│   ├── prepare_visdrone.sh       # 数据转换
│   ├── oracle_decompose.py       # ★ 误差四象限分解（核心方法论）
│   ├── analyze_visdrone.py       # ★ 分尺寸/分类别召回
│   ├── bench_models.py           # ★ 延迟与 FLOPs 实测
│   ├── probe_budget.py           # 检测预算分配研究
│   └── make_figures.py 等        # 出图
├── results/
│   ├── results_csv/              # ★ 全部实验的 results.csv（训练原始记录）
│   ├── figures/                  # ★ 论文用图
│   ├── oracle*/                  # 误差分解结果
│   └── *_COMPARISON.md           # 各阶段对比表
├── docs/                         # 分阶段实验报告
└── experiments/negative-results/ # 负结果的 patch 与说明
```

**权重不在仓库里**：复现训练即可得到（见上方命令）；或从 Releases 下载最终模型。

## 指标口径（重要）

本仓库同时用到两套指标，**绝对数值不可直接混用**：

| 口径 | 工具 | baseline@640 | P2@960 |
|---|---|---|---|
| ultralytics 内部指标 | `yolo val`（表格里的主结果） | 0.182 | 0.263 |
| COCO 协议 | `faster-coco-eval`（oracle 分析用） | 0.173 | 0.247 |

**同数据集上两者相差约 0.9 AP。** 引用本仓库数字时请注明口径。

## 待补实验

- [ ] `baseline@960`（base 结构 + 960 训练）：补齐 2×2 消融，把"960 训练"与"P2 结构"的贡献分离
- [ ] 各配置多种子（当前主结果均为单种子）
- [ ] 与公开 VisDrone 方法在同设置下的对比

## 许可

本仓库包含对 [ultralytics](https://github.com/ultralytics/ultralytics)（**AGPL-3.0**）的修改补丁，故整体以 **AGPL-3.0** 发布，详见 `NOTICE.md`。

## 引用

```bibtex
@misc{visdrone-yolo26,
  title  = {VisDrone Small Object Detection via Error-Decomposition-Guided Representation Optimization},
  note   = {Built on ultralytics 8.4.160 / YOLO26n},
  year   = {2026}
}
```
