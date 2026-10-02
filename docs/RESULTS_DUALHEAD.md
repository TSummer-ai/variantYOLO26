# 创新点：NMS-free 双头设计的再审视（效率型）

日期：2026-10-02 ｜ 分支 `feat/o2o-grad`（`wt-nodt`）
对照：`base_n_640`（yolo26n，640，batch16，seed0，100ep）＝ o2m **0.3280 / 0.1820**，o2o **0.3285 / 0.1824**

## 一、动机（已由文献探针确认是空白）

检测器的**双头设计**（one2many 教师 + one2one NMS-free）在 YOLOv10 / YOLO26 中都是
`copy.deepcopy` 一份头。文献核查（6 路独立探针，逐个给出 URL）：

| 相关做法 | 状态 |
|---|---|
| 推理时删掉 o2m 分支 | ❌ **YOLOv10 原文即如此**；ultralytics `Detect.fuse()`（`head.py:277`）已实现 |
| o2m/o2o 监督差的理论推导 | ❌ YOLOv10 Eq.2 |
| "o2o 是校准问题" | ❌ OneNet（ICML 2021）的 score gap |
| 训练期辅助 o2m 分支 | ❌ H-DETR / Co-DETR / Group DETR |
| 密集场景自适应 query 预算 | ❌ DQA-DETR / D3Q / RDAQ |
| **两个头共享权重** | ✅ **未发现任何工作**（YOLOv10 与 YOLO26 都 deepcopy） |
| **o2o 分支的梯度隔离（detach）** | ⚠️ 设计由 PR #25548 文档化，但**无人实测其影响** |

**为什么这对小目标检测有意义**：小目标必须加细粒度层（P2），而双头使**每个新增层级在训练图里成本 ×2**
（参数、GFLOPs、显存），推理时只需 ×1。若权重可共享，新增层级的训练成本从 2× 回到 1×。

## 二、两条消融轴

| 轴 | 超参 | 含义 |
|---|---|---|
| **A. 权重** | `head_share ∈ {none, cls, box, all}` | o2o 分支复用 o2m 权重的范围；配套 `o2o_affine`（`o2o = s·o2m + b`，s=1,b=0 起步 = 恒等） |
| **B. 梯度** | `o2o_detach ∈ {True, False}` | True = 原生（o2o 梯度不进 backbone，`head.py:185`）；False = 让 NMS-free 分支也能塑造特征 |

## 三、已完成的验证

| 验证 | 结果 |
|---|---|
| 关闭全部开关 = 原生 | 逐位一致（探针输出与 main 完全相同）✅ |
| `o2o_detach=False` 真的打开梯度 | 只留 o2o 项时 backbone 前 3 层梯度范数 **0.000e+00 → 5.750e+00** ✅ |
| `head_share` 真的别名 | `cls`/`all` 时 `one2one_cv3 is cv3` 为真，省掉复制的分类头参数 **83,008** ✅ |
| 可行性（8GB 卡） | 去掉 detach 后显存 **不变**（2.50 GiB），仅 +11% 步耗时 ✅ |
| 仿射参数进优化器 | 参数在 `Detect.__init__` 创建（早于优化器构建）✅ |

## 四、预先登记的判定线

| 臂 | 配置 | 判定 |
|---|---|---|
| A1 | `head_share=all` | **e2e AP 与基线差 ≤ 0.3** → 主张"双分配监督不需要复制头"成立 |
| A2 | `head_share=all` + `o2o_affine=True` | 若 A1 掉 >0.5，则 A2 恢复 → **验证 YOLOv10 监督差理论并给出廉价修法**（更好的故事） |
| A3 | `head_share=cls` | 若 A1 失败而 A3 成立 → "只有分类头需要独立" |
| B1 | `o2o_detach=False` | 与基线差 ≤0.3 为中性；>+0.3 为增益（NMS-free 分支能塑造特征更好） |

## 五、附带产出：上游 bug（可提 issue）

我们的 checkpoint（以及官方 `yolo26n.pt`）**都缺 `_end2end` 标记** → `Detect.end2end` 为 False →
`predict(nms=False)` **静默走 o2m 分支 + 关 NMS**，而不是 NMS-free 头。这正是上游
**PR #26478**（未合并）描述的隐患，我们在三个模型上复现并量化（见 `RESULTS_FINAL.md` 的「重大更正」）。
建议向上游提交带复现步骤与量化证据的 issue。
