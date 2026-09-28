# YOLO26 创新改进方案

**目标场景**：无人机密集小目标检测（VisDrone2019-DET）
**算力现实**：单卡 RTX 4060 Laptop 8 GB + 32 核 CPU，只能用 COCO 预训练权重做微调，**不可能从零训 COCO**
**基线代码**：`./ultralytics` @ v8.4.160（`git log`: `9d25f80da`），editable 已装好，val 链路已跑通（COCO yolo26n = 40.8 vs 官方 40.9）

---

## 0. 先决条件

| 项 | 状态 |
|---|---|
| 环境 / GPU / 权重 | ✅ 已就绪（`source activate_yolo.sh`，GPU 需放宽沙箱） |
| COCO val 复现链路 | ✅ 已跑通，可作为**泛化性验证集** |
| VisDrone2019-DET | ⚠️ 数据在 `/home/wang/DeepSeek/datasets/VisDrone2019/`（6471 train / 548 val），已带 COCO 格式 json（`annotations/instances_{train,val}.json`），但**还没有 YOLO 格式标签 + data yaml** |
| VisDrone 基线 | ❌ 未跑，**这是所有创新点的前提** |

> 第 0 步不是可选项：没有基线，"涨点"无从谈起；而且必须先做**错误分析**，确认要打的痛点是"小目标漏检"而不是"分类混淆"。

---

## 1. 问题定位：YOLO26 的三个可攻击点

YOLO26 相对 YOLO11 的四个设计区（见 `docs/en/models/yolo26.md`）：

1. **NMS-free 端到端**：one-to-one 头 + one-to-many 头双头训练（`nn/modules/head.py: Detect`）
2. **去 DFL**：`reg_max=1`，回归损失退化为 **纯 L1**（`utils/loss.py: v8DetectionLoss.use_dfl=False`）
3. **训练配方**：MuSGD + **Progressive Loss**（`E2ELoss.decay`: o2m 权重 0.8→0.1）+ **STAL**（小目标正样本覆盖）
4. 各任务专用头

对应地，在**密集小目标**场景有三个具体薄弱点（都是 YOLO26 自身设计带来的，而不是通用缺陷 → 创新故事干净）：

| # | 薄弱点 | 证据/机制 |
|---|---|---|
| **P1** | **e2e 头小目标召回不足** | `E2ELoss` 里 one2one 的 `tal_topk=1`，每个 GT 只有 1 个正样本；Progressive Loss 后期把 o2m 权重压到 0.1，等于**后期几乎只训 one2one**。COCO 上 e2e 比 O2M 低 0.8 AP（官方 40.1 vs 40.9），小目标密集时差距通常更大 |
| **P2** | **去 DFL 后回归损失与尺度无关** | `BboxLoss` 在 `reg_max=1` 时用纯 L1；对 8px 的小目标，1px 误差 = 12.5% 尺度误差，IoU 掉得极快，梯度却和大框完全一样 → 小目标定位学不好 |
| **P3** | **分配对极小目标偏严** | `TaskAlignedAssigner` 的 align_metric = p^α · IoU^β（`α=0.5, β=6.0`），β=6 让 IoU 主导；小目标框稍微偏一点 IoU 就趋零 → 正样本被"判死"。YOLO26 的 STAL 只是保证覆盖，没有改度量的尺度敏感性 |

> 三个点互相独立、可分别消融，也能组合成一个完整故事：**"让 NMS-free 的 YOLO26 在密集小目标上真正可用"**。

---

## 2. 主推方案

### 创新点 A：双头一致性蒸馏（DHCD, Dual-Head Consistency Distillation）

**一句话**：训练时让 one-to-many 头当"小目标老师"，把它的软分类分布、回归共识和分配结构蒸馏给 one-to-one 头，专门补 e2e 支路在小目标上的监督稀疏。

**为什么合理**：O2M 头有 topk=10 个正样本，对小目标的定位/分类估计方差更小；O2O 头只有 1 个正样本，且 Progressive Loss 后期几乎没有 O2M 监督。把 O2M 的"多数投票"知识迁移给 O2O，等价于给小目标提供了**多正样本的软监督**，但不改变推理结构（仍是 NMS-free，零推理开销）。

**技术方案**（三个可拆分的蒸馏项，逐个消融）：

1. **软分类蒸馏**：对同一 anchor，O2M 的分类 logits 经温度 τ 平滑后作为 O2O 的软目标，只对 O2M 判定为"候选正样本"的 anchor 生效（用 `fg_mask` 或 top-k 掩码），权重按 O2M 的 `align_metric` 加权，避免污染背景。
2. **回归共识蒸馏**：以 O2M 的 **align_metric 加权平均框**作为软回归目标（而非直接挑一个正样本），约束 O2O 回归。小目标尤其受益——多正样本平均能显著降低小框的定位方差。
3. **分配结构迁移**：warm-up 阶段把 O2M 的正样本掩码直接用于 O2O 的回归监督（后期解耦）。与现有 Progressive Loss 组合时要注意：**建议把 o2m 权重下限从 0.1 提到 0.2~0.3，或让蒸馏项替代一部分衰减**，否则后期无老师可蒸。

**代码落点**（都很集中，不需要动 backbone）：

```
ultralytics/utils/loss.py
  ├─ class E2ELoss.__call__()          # 双头 loss 组合处，插入蒸馏项
  ├─ v8DetectionLoss.get_assigned_targets_and_loss()
  │     # 已有 fg_mask / target_gt_idx / target_bboxes / align_metric 可复用；需 return 出去
  └─ 新增 class DHCDLoss(nn.Module)
ultralytics/nn/modules/head.py: Detect.forward()
  # 训练时 one2one 输入是 x_detach（已 detach），正好适合当 student；O2M 输出保持 teacher 不回传
ultralytics/cfg/default.yaml           # 新增开关: dhcd_w, dhcd_tau, dhcd_box_w, dhcd_mask_w
ultralytics/nn/tasks.py:607            # E2ELoss 的构造处，按需注入
```

**预期收益**：VisDrone 上 e2e（`nms=False`）AP50-95 **+1.0 ~ +2.0**，同时把 O2M 与 O2O 的差距从 ~1 AP 压到 0.3 AP 以内；COCO 上也能缩小 e2e 差距（可在我们已跑通的 COCO val 上验证，不改一行评测脚本）。

**风险**：蒸馏权重调不好会拖累 O2M 分支。→ 蒸馏只加到 O2O loss 上，O2M 分支完全不动，天然可控。

---

### 创新点 B：尺度自适应回归损失（SAR, Scale-Aware Regression）

**一句话**：在保持 YOLO26 "DFL-free" 部署优势的前提下，把纯 L1 换成尺度自适应的混合回归损失，救回小目标定位精度。

**技术方案**：

```
L_reg = λ(s)·(1 − CIoU) + μ(s)·L1_log
  s = 目标等效边长（像素），s0 = 16 为分界
  λ(s) = σ((s0 − s)/s0)        小目标 → IoU 项权重大（IoU 对小框更敏感，直接优化它）
  μ(s) = 1 − λ(s)              大目标 → 保持 L1 的稳定收敛
  L1_log = smoothL1(log(w/w_ref), log(h/h_ref))   # 压缩尺度动态范围，避免小框梯度被淹没
```

- **和 DFL 的区别**：DFL 是让网络学分布再积分（头变重、导出复杂，YOLO26 特意删掉）；SAR 只改**损失函数**，头部结构、导出格式、推理速度完全不变——这是它比"把 DFL 加回来"更优的地方。
- **可选升级**：用 **NWD（Normalized Wasserstein Distance）** 替换 CIoU 项。小目标 IoU 对位置极敏感、且几乎不提供梯度，NWD 把它变成高斯分布距离，是当前小目标检测的常用有效手段。

**代码落点**：
```
ultralytics/utils/loss.py: class BboxLoss.forward()   # reg_max==1 的 L1 分支（唯一改动点）
ultralytics/cfg/default.yaml                          # 新增 sar_w / sar_s0 / sar_mode
```
改动量极小（单函数内 ~20 行），**消融最干净、见效最快**，适合作为整篇工作的"保底创新点"。

**预期收益**：APs **+1 ~ +2**，整体 AP50-95 +0.5 ~ +1.0；参数量/FLOPs **零变化**（论文里是很漂亮的"零成本涨点"）。

---

### 创新点 C（加分项，可选）：尺度感知标签分配

**方案**：在 `TaskAlignedAssigner` 上做两点——

1. **动态 topk**：按 GT 尺度设 topk（小目标 topk 更大，如 `k = clip(k0 · (s0/s), k_min, k_max)`），给小目标更多梯度；
2. **尺度自适应度量**：`align_metric = p^α · (IoU + γ·center_score)^β`，或直接把 IoU 换成 NWD；β 随尺度降低（小目标不再被 β=6 的 IoU 项一票否决）。

**落点**：`ultralytics/utils/tal.py: TaskAlignedAssigner.forward / get_box_metrics / select_topk_candidates`。
**注意**：与创新点 B 有耦合（都用了 NWD/尺度信息），**消融时必须分开跑**，否则审稿人会质疑增益归属。

---

## 3. 备选方案（如果主推路线不顺）

| 方案 | 内容 | 优点 | 缺点 |
|---|---|---|---|
| **B1 轻量化 + 蒸馏** | 用 yolo26l 蒸馏 yolo26n/s（特征 + logits + 分配蒸馏），做无人机边缘部署的精度-速度帕累托 | 工程价值明确，涨点稳 | 新颖度中等，偏"组合式创新" |
| **B2 P2 + 高分辨率 + SAHI** | P2 头（官方已给 `yolo26-p2.yaml`）+ 960/1280 训练 + 切片推理 | 立竿见影（VisDrone 上很有效） | **已高度饱和**，2026 年已有大量同类论文（PSSL-YOLO、DSG-YOLO26、Aero-YOLO 等），只能当 baseline 增强，不能当创新点 |
| **B3 频域/小波增强** | 小波变换分支增强小目标高频细节，融入 C3k2 或 P3/P2 融合 | 视觉故事好讲 | 增参增量，容易掉进"堆模块"陷阱 |

**建议**：主线走 **A + B**（一个偏训练范式、一个偏损失设计，互补且都不增加推理开销），B2 只作为 baseline 增强项老实标注。

---

## 4. 实验设计

### 4.1 数据与训练设置

```yaml
# 数据：VisDrone2019-DET，COCO json 已有 → 用已验证过的 convert_coco 转 YOLO 格式
# 类别 10 类，忽略区域（raw 标注里的 class 0/11）必须在转换时丢掉，否则 AP 会被污染
```

| 项 | 设置 |
|---|---|
| 预训练 | `yolo26n.pt` / `yolo26s.pt`（COCO 官方权重，微调） |
| 分辨率 | 640 为主；小目标实验补 960（VisDrone 原图 1360×765，宽高比特殊 → 用 `rect=True` 或 960 方形） |
| epochs | 100（`patience=30`），`close_mosaic=10`；先用 n 跑通，s 只跑最终方案 |
| batch | 8–16 @640（8 GB 显存）；`cache=ram`（VisDrone 训练集约 1.4 GB，32 GB 内存可全缓存，能省大量 IO） |
| 优化器 | `optimizer=MuSGD` 或 `auto`（官方配方 @640/batch128 的 lr0=0.0054，微调时按 `lr0 ∝ batch` 缩放，建议 lr0≈1e-3） |
| 增强 | 保留官方配方（mosaic≈0.9、scale 0.5~0.95、close_mosaic=10）；**小目标场景慎用大 scale-down 和 mixup**，可单独消融 |
| 评测 | 每轮 val：AP50-95 / AP50 / **APs** / APm / APl（VisDrone 几乎全在 APs）；同时记录 NMS 版与 `nms=False` 两套 |
| 成本指标 | params / GFLOPs / 4060 上的 ms-per-image（batch=16）——创新点 A/B/C **都不应增加推理开销**，这是卖点 |

### 4.2 消融矩阵（n 尺度，640，100 ep，单种子）

| # | 配置 | 目的 |
|---|---|---|
| 1 | baseline yolo26n | 参照 |
| 2 | +A1 软分类蒸馏 | 拆解 A |
| 3 | +A2 回归共识蒸馏 | 拆解 A |
| 4 | +A1+A2（完整 DHCD） | A 的完整增益 |
| 5 | +B (SAR-L1+CIoU) | B 的增益 |
| 6 | +B (SAR+NWD) | B 的变体对比 |
| 7 | +C 动态 topk | C 的独立增益 |
| 8 | **A+B** | 主线组合 |
| 9 | A+B+C | 上限探测 |

- 关键实验（1/4/5/8）跑 **3 个种子**报均值±方差；其余单种子。
- **必须同时报 NMS 与 e2e 两套数字**（A 的价值主要体现在 e2e 上）。
- 泛化性：把最终模型在 **COCO val2017** 上 val 一次（环境已就绪，脚本现成），证明改进不是过拟合无人机域。

### 4.3 对比对象

- 同设定下自复现的 YOLO11n / yolo26n / yolo26n-p2；
- 公开的 YOLO26 小目标改进（PSSL-YOLO、DSG-YOLO26、Aero-YOLO 等）——**要么在相同数据/epoch 下复现其思路（P2+注意力），要么只引数字并注明设置差异**，不要直接跨设置比。

---

## 5. 算力预算与排期（单卡 4060 Laptop 8 GB）

先实测一次单 epoch 时间再排（下面按 yolo26n@640 batch16 ≈ 60–120 s/epoch 估）：

| 阶段 | 内容 | 估计 |
|---|---|---|
| D0 | VisDrone → YOLO 格式 + data yaml + 基线跑通（20 ep 试跑） | 0.5 天 |
| D1 | 基线 yolo26n 100 ep（640）+ 错误分析 | 2–4 h |
| D2 | 创新点 B（改动最小、见效最快）— 实现 + 3 组消融 | 1 天（含 3×2–3 h 训练） |
| D3 | 创新点 A（DHCD）— 实现 + 4 组消融 | 1.5 天 |
| D4 | A+B 组合 + 多种子 + 960 分辨率复验 | 1.5 天 |
| D5 | COCO 泛化性验证 + 速度/参数量表 + 出图 | 0.5 天 |
| D6 | 消融补齐 + 写作 | 1–2 天 |

**合计约 6–8 天**（不含写作返工）。显存紧张时优先降 batch、开 AMP、`cache=ram`，不要降分辨率到 512（小目标会直接消失）。

---

## 6. 代码落点总览

| 文件 | 改动 |
|---|---|
| `ultralytics/utils/loss.py` | **主战场**：`E2ELoss`（蒸馏项挂载）、`v8DetectionLoss.get_assigned_targets_and_loss`（导出 fg_mask/align_metric）、`BboxLoss.forward`（SAR）、新增 `DHCDLoss` |
| `ultralytics/utils/tal.py` | `TaskAlignedAssigner`（动态 topk / 尺度自适应度量） |
| `ultralytics/nn/modules/head.py` | `Detect.forward`（确认 teacher/student 数据流；一般只需读不改） |
| `ultralytics/nn/tasks.py:607` | `E2ELoss` 构造处（注入自定义 criterion） |
| `ultralytics/cfg/default.yaml` | 新增超参开关（`dhcd_*`, `sar_*`），保证可消融 |
| `ultralytics/cfg/models/26/yolo26-p2.yaml` | P2 baseline（官方已有，直接用） |
| `datasets/visdrone.yaml` | 新建（10 类，train/val 指向转换结果） |

**开发规范**（仓库 `AGENTS.md` 要求）：不要直接改主 checkout 的 `main`，用 worktree + 分支：

```bash
cd $YOLO_ROOT/ultralytics
git worktree add ../wt-dhcd -b feat/dhcd-sar
```

**每次实验都要留档**：`runs/visdrone/<exp_name>/args.yaml + results.csv + 日志`，命名规范 `a1_o2m_distill` / `b_sar_nwd` / `ab_full`，否则一周后自己都对不上号。

---

## 7. 与已有工作的差异化

2026 年已经有一批"YOLO26 小目标改进"：[PSSL-YOLO](https://ieeexplore.ieee.org/document/11677952)（P2-SGF 头 + SOLA-Lite 分配）、[DSG-YOLO26](https://www.sciencedirect.com/science/article/abs/pii/S0923596526002006)（细节保留 + 尺度门控）、[Aero-YOLO](https://dl.acm.org/doi/abs/10.1007/978-981-92-3507-0_17)、[SDD-YOLO](https://ar5iv.labs.arxiv.org/html/2603.25218)，以及 [基于图像增强的改进 YOLO26](https://ieeexplore.ieee.org/document/11637904)。

它们的共同套路是：**换/加模块（P2、注意力、门控）+ 改分配**。本方案的差异点：

1. **打的是 YOLO26 独有的双头结构**（O2M→O2O 一致性蒸馏），别人没做；
2. **打的是 YOLO26 删掉 DFL 这个具体决策**（尺度自适应回归），是"在作者的设计取舍上做增量"，故事比"再堆一个注意力"好讲；
3. **零推理开销**（不增参数、不增 FLOPs、不破坏 NMS-free 导出），在"实时无人机"场景是硬卖点；
4. 有 **COCO 泛化性验证**，不只在一个数据集上刷点。

---

## 8. 风险与降级路径

| 风险 | 概率 | 应对 |
|---|---|---|
| 基线都跑不出合理数字（数据转换错/忽略区域没丢） | 中 | D0 阶段先目视校验标签可视化（`val_batch0_labels.jpg` 现成），确认框对齐再开训 |
| 创新点 A 增益 < 0.3 AP | 中 | 降级：A 只保留软分类蒸馏；把主线重心移到 B（更稳） |
| 8 GB 显存不够 960 分辨率 | 高 | batch 降到 4–8 + AMP；或只对最终方案跑 960 |
| 增益被"多加参数"解释 | 低 | 我们的改动**不加参数**，表格里直接列 params/FLOPs 对比 |
| 审稿人质疑与已有工作雷同 | 中 | 老老实实把 P2+注意力作为 baseline 复现进表格，用同一训练配置对比 |

---

## 9. 立刻可执行的第一步（今天就能做）

```bash
# 1) VisDrone COCO json -> YOLO 标签（复用已验证的 convert_coco）
source activate_yolo.sh
#    —— 注意：instances_train.json 里若含 ignore/others 类，转换时要过滤

# 2) 写 datasets/visdrone.yaml（10 类），目视确认标签对齐
# 3) 跑 20 epoch 短基线，确认 loss 正常下降、mAP 合理
# 4) 跑满 100 epoch 基线，做错误分析（哪些类漏检、目标尺寸分布、APs 占比）
#    —— 错误分析结果决定 A/B/C 三个创新点的优先级
```

> 第 4 步的错误分析**不要跳过**：如果漏检主要发生在 <16px 的目标上，创新点 B（回归损失）优先级最高；如果主要发生在 O2O 分支（e2e vs NMS 差距大），则 A 优先。
