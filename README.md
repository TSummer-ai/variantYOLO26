# VisDrone 小目标检测：误差分解驱动的表示侧优化

基于 **YOLO26n**（ultralytics 8.4.160）在 **VisDrone2019-DET** 上的小目标检测研究。
核心思路不是"再加一个模块"，而是**先用误差分解定位瓶颈，再对症下药**，并量化否证一批常见改进方向。


> ### ⚠️ 效率数字更正（2026-10-02）：此前报的是**未 fuse** 的训练图口径
>
> `get_flops()` 与在 `YOLO(w).model` 上的延迟测量都作用于**未 fuse** 的模型；而未 fuse 的 eval 模型会
> **同时计算 one2many 与 one2one 两个头再丢掉一个**（`head.py:183-190`）。`Detect.fuse()`（`head.py:277`，
> *"Remove the unused detection branch for inference"*）才会删掉无用分支。
>
> 正确（部署）口径：
>
> | 模型 | GFLOPs（fuse 后） | 前向延迟（fuse 后） | 此前误报 |
> |---|---|---|---|
> | baseline@640 | **5.32** | **2.86 ms** | 5.9 / 4.6 ms |
> | P2@640 | **6.57** | **3.47 ms** | 7.7 / 5.9 ms |
> | P2@960 | **15.13** | **6.44 ms** | 17.6 / 9.0 ms |
>
> **P2 的实际代价是 +23.5% GFLOPs / +21.4% 延迟**（原报 +63% / +29%）；960 是 +184% / +125%。
> 另外要记住一个**架构事实**：`head.py:185` 对 o2o 分支做了 `.detach()`，**它的梯度不进 backbone/neck** ——
> 所以两个头在训练图里的开销才是 2×，推理时只有 1×。


> ### ⚠️ 重大更正（2026-10-02）：此前的 "e2e / NMS-free" 数字测错了对象
>
> 本仓库早期版本中标注为 `e2e`（`nms=False`）的数字，**实际测的是 one2many 头 + 关闭 NMS**，而非
> one2one(NMS-free) 头。原因：`Detect.end2end` 属性（`head.py:155`）要求 `_end2end` 标记，而
> **本地 fine-tune 出的 checkpoint（乃至官方 `yolo26n.pt`）都不带该标记** → `end2end=False` →
> 推理走 o2m 分支。这正是上游 **PR #26478**（未合并）描述的隐患。
>
> 显式设置 `head._end2end = True` 后的正确结果：
>
> | 模型 | o2m（NMS 头） | **o2o（NMS-free，正确评测）** | 旧误报 e2e |
> |---|---|---|---|
> | base@640 | 0.3280 / 0.1820 | **0.3285 / 0.1824** | 0.3200 / 0.1770 |
> | P2@640 | 0.3490 / 0.1980 | **0.3487 / 0.1981** | 0.3320 / 0.1900 |
> | P2@960 | 0.4400 / 0.2630 | **0.4402 / 0.2628** | 0.4180 / 0.2510 |
>
> **结论：两个头性能等价（差异 ≤ 0.0004）。** 因此：
> 1. 不存在"o2o 头更弱"这一现象（此前基于错误评测的 DHCD 动机与"双头差距"分析全部作废）
> 2. 引用本仓库时请以上表为准

## 贡献总结（诚实版）

> 本节明确区分**标准做法**与**本工作的原创部分** —— 避免把工程增益包装成创新主张。

### 全栈结果（COCO 协议，同口径可比）

| 阶段 | mAP50-95 | mAP50 | 增量 |
|---|---|---|---|
| baseline @640 | 0.1731 | 0.3053 | — |
| + P2 层（stride 4） | 0.1876 | 0.3259 | +1.45 |
| + 960 分辨率 | 0.2474 | 0.4135 | +5.98 |
| + 尺寸感知切片融合 | 0.2750 | 0.4683 | +2.76 |
| **+ MV-Fuse 一致性重打分** | **0.2788** | **0.4754** | +0.38 |
| **合计** | | | **+10.57 AP50-95（+61%）｜+17.01 AP50** |

### 逐项定性（收益 / 代价 / 是否原创）

| 项 | 类别 | 收益 | 代价 | 定性 |
|---|---|---|---|---|
| P2 层 | 训练期·结构 | +1.45 AP | +23.5% GFLOPs / +21.4% 延迟 | **标准做法** |
| 960 分辨率 | 训练期·输入 | +5.98 AP | +184% GFLOPs / +125% 延迟 | **标准做法** |
| `max_det=1000` | 评测协议 | +0.4~0.6 AP50-95 / +1.1~1.5 AP50 | ≈0 | 协议修正（上游 v8.4.135 已自动处理） |
| 尺寸感知切片融合 | 推理期·零训练 | +2.76 AP | 4× 前向 | 框架被 SAHI/ASAHI 占据；差异在**按预测框尺寸门控信息源** |
| **MV-Fuse：跨视角一致性重打分** | 推理期·零训练 | **+0.38 AP** | **0 额外前向** | **未发现先行工作** |

**一句话**：**大收益来自标准/工程手段（P2、960、max_det），本工作的原创部分是推理期零训练流水线** ——
其核心机制是"多视角一致性"这一**免费**信号（支持视角数越多，TP 率越高；且该分离度**随目标变小而增强**，
0–8px 达 +23.7 点）。详见 [`docs/RESULTS_MVFUSE.md`](docs/RESULTS_MVFUSE.md)。

### 另一部分产出：分析骨架（非正向方法，但构成论文主体）

- **已量化否证的方向**：SAR 尺度重加权 +0.3（噪声内）、DHCD 双头蒸馏 −0.1、小目标定向标签分配
  **−0.4~−1.1**（缺陷修好仍掉点）、XRD 跨分辨率蒸馏 −0.1、**双头权重共享 −5.1**、去掉 o2o detach 中性。
- **分析工具与结论**：oracle 四象限误差分解、分尺寸漏检率、**精度-算力 Pareto 前沿（8 配置）**、
  **per-image oracle 上界 0.2768（被 MV-Fuse 超过）**、标注噪声仅 1.8%、局部对比度效应（尺寸无关）、
  GT 互重叠仅 0.2%、误检结构、退化鲁棒性。
- **部署**：TensorRT FP16 **476 FPS**（fused FP16 236 FPS，TRT 加速 2.02×）。
- **上游 bug 的量化复现**：[PR #26478](https://github.com/ultralytics/ultralytics/pull/26478)（`_end2end`
  标记缺失 → NMS-free 推理静默降级为 o2m+NMS）。

---

## 主要结果

VisDrone2019-DET val（548 图 / 38,759 GT），全部 100 epoch 同设定训练，`yolo val` 内部指标：

| 模型 | imgsz | max_det | P | R | mAP50 | **mAP50-95** | GFLOPs | 前向延迟 |
|---|---|---|---|---|---|---|---|---|
| baseline（P3–P5） | 640 | 300 | 0.464 | 0.348 | 0.328 | **0.182** | 5.32 | 2.86 ms |
| P2（P2–P5） | 640 | 300 | 0.492 | 0.356 | 0.349 | **0.198** | 6.57 | 3.47 ms |
| **P2（最终）** | **960** | **1000** | **0.549** | **0.446** | **0.455** | **0.269** | 15.13 | 6.44 ms |
| P2（NMS-free 分支） | 960 | 300 | 0.549 | 0.446 | 0.440 | **0.263** | 15.13 | 6.44 ms |

**相对基线：mAP50 +11.6，mAP50-95 +8.3**（同口径 `max_det=1000`）。

延迟在 RTX 4060 Laptop / **已 fuse** / FP32 / batch 1 下实测（100 次平均）：**6.44 ms ≈ 155 FPS**。

### 推理期方法：MV-Fuse = 尺寸感知切片融合 + 跨视角一致性重打分（**+2.76 AP50-95，零训练、零额外前向**）

在**已训练**的 960 模型上追加（COCO 协议，val 全量 548 图）：

| 管线 | 前向次数 | AP50-95 | AP50 | 延迟 / 吞吐 |
|---|---|---|---|---|
| 整图 960（基线） | 1× | 0.2512 | 0.4221 | 7.2 ms / 139 FPS |
| + 尺寸感知切片融合（3×1） | 4× | 0.2750（+2.38） | 0.4684（+4.63） | ~28 ms / ~36 FPS |
| **+ MV-Fuse 跨视角一致性重打分** | **4×（0 额外）** | **0.2788（+2.76）** | **0.4754（+5.33）** | 同上 |

三条量化结论（都有消融支撑）：

1. **切片对大小目标作用相反**：小目标召回大涨（8–16px 档 **+8.4 点**），但会切断大目标（≥64px 档 **−4.8 点**）
2. **必须按尺寸选择信息源**：仅切片 −1.9 → 普通 NMS 合并 +0.3 → **尺寸感知融合 +2.0**（大框只信整图）
3. **该沿宽度切列，不该切方形网格**：letterbox 缩放 `min(960/W,960/H)` 受**宽度**限制，
   3 列（放大 1.78×）比 2×2（1.67×）**既更准又更省**；4 列后高度成为限制、不再提升

**两个组件都必要**：朴素并集+一致性 0.2781 < 尺寸门控+一致性 0.2788 < 只有门控 0.2750。

详见 [`results/SLICING.md`](results/SLICING.md) 与 [`docs/RESULTS_MVFUSE.md`](docs/RESULTS_MVFUSE.md)。**训练成本为 0、额外前向为 0**，可套用在任何已训练检测器上。

> ⚠️ **新颖性声明**：切片推理 + 整图融合的**框架并非本工作首创** —— 先行工作包括
> [SAHI](https://obss.github.io/sahi/guides/sliced-inference/)（2022，固定尺寸切片 + 融合）与
> [ASAHI](https://ar5iv.labs.arxiv.org/html/2604.19233)（2026，**双流 + 自适应切片 + Cluster-DIoU-NMS 合并**，
> VisDrone-val mAP50 56.8%，比 SAHI +1.7）。
> 本工作只主张窄范围差异：**按预测框尺寸门控信息源** + **列切分放大倍数的量化分析**。
> 详见 `results/SLICING.md` 的「相关工作与新颖性声明」一节。

对应的权重文件已随仓库提供（`weights/`），可直接推理或复现评测：
`yolo26n-visdrone-base-640.pt` / `yolo26n-visdrone-p2-640.pt` / **`yolo26n-visdrone-p2-960.pt`**。

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

## 快速开始（不训练，直接用现成权重）

```bash
git clone <仓库地址> && cd visdrone-yolo26
pip install -r requirements.txt
source activate_yolo.sh

# 推理（注意分辨率必须与训练一致：960 模型用 imgsz=960）
yolo predict model=weights/yolo26n-visdrone-p2-960.pt source=your.jpg imgsz=960

# 精度/延迟实测（不需要数据集）
python scripts/bench_models.py --weights weights/yolo26n-visdrone-p2-960.pt --imgsz 960

# 零训练的精度提升：尺寸感知切片融合（需要数据集做评测）
python scripts/probe_slicing.py --weights weights/yolo26n-visdrone-p2-960.pt \
       --imgsz 960 --grid 3,1 --overlap 0.3 --pipelines "full,full+adaptive" --big-px 48 --merge-iou 0.6
```

复现表格指标需要先准备数据集（见下节），然后：

```bash
yolo val model=weights/yolo26n-visdrone-p2-960.pt data=configs/visdrone.yaml imgsz=960 max_det=1000
# 期望：mAP50 0.455 / mAP50-95 0.269
```

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
├── weights/                      # ★ 三个主模型权重（约 16 MB，含说明见其 README）
│   ├── yolo26n-visdrone-base-640.pt
│   ├── yolo26n-visdrone-p2-640.pt
│   └── yolo26n-visdrone-p2-960.pt   # 最终模型
├── configs/
│   ├── visdrone.yaml             # 数据集配置模板
│   └── models/                   # 自建 yaml（轻量 P2 变体，未训练）
├── scripts/                      # 全部训练/评估/分析/出图脚本
│   ├── prepare_visdrone.sh       # 数据转换
│   ├── oracle_decompose.py       # ★ 误差四象限分解（核心方法论）
│   ├── analyze_visdrone.py       # ★ 分尺寸/分类别召回
│   ├── bench_models.py           # ★ 延迟与 FLOPs 实测
│   ├── probe_budget.py           # 检测预算分配研究
│   ├── probe_slicing.py          # ★ 尺寸感知切片融合（评测）
│   ├── tune_slicing.py           #   切片布局/融合参数寻优
│   ├── adaptive_slicing.py       #   自适应触发（精度-成本曲线）
│   └── make_figures.py 等        # 出图
├── results/
│   ├── results_csv/              # ★ 全部实验的 results.csv（训练原始记录）
│   ├── figures/                  # ★ 论文用图
│   ├── oracle*/                  # 误差分解结果
│   ├── SLICING.md                # ★ 切片融合的方法与全部消融
│   └── *_COMPARISON.md           # 各阶段对比表
├── docs/                         # 分阶段实验报告
└── experiments/negative-results/ # 负结果的 patch 与说明
```

**权重已随仓库提供**（`weights/`，共约 16 MB）—— 克隆下来就能推理，不必先训练。
数据集与训练产物（`runs/`）不入库，按上方「数据准备」自行生成。

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
