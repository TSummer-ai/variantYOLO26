# 负结果：SAR 与 DHCD

这两个方向都完整实现并训练过，**结果为负**。保留在此是为了可复现性与消融章节的完整性 ——
它们证明了"损失侧/训练侧的余量已被穷尽"，而不是没试过。

## 1. SAR：尺度自适应回归重加权（`patches/0001`, `0002`）

**动机**：VisDrone 目标中位仅 11.3px，怀疑小目标的回归项被大目标淹没。

**做法**：在 `BboxLoss` 里按目标尺度重加权 L1（以及 CIoU）项，权重
`sar_w`、`sar_s0`（尺度基准）、`sar_cap`（权重上限）、`sar_iou_w`。

**结果**：

| 配置 | NMS mAP50-95 | 相对基线 |
|---|---|---|
| baseline | 0.1820 | — |
| SAR L1 w=0.5 | 0.1830 | +0.1 |
| SAR L1 w=1.0 | 0.1850 | **+0.3（单种子噪声内）** |
| SAR L1 w=2.0 | 0.1830 | +0.1 |
| SAR L1+CIoU w=1.0 | 0.1820 | 0.0 |

**原因**：0–8px 档的漏检率**完全没动**；尺度重加权只改变了损失权重，不改变
"小目标在 stride-8 网格上难以回归"这个表示层限制。

## 2. DHCD：双头一致性蒸馏（`patches/0003`）

**动机**：YOLO26 有 one2many / one2one 双头，NMS-free 的 e2e 头精度低于 NMS 头，怀疑其召回不足。

**做法**：在 `E2ELoss` 里对 teacher（one2many）高置信 anchor 加 BCE，把置信度蒸到 student（one2one）。

**结果**：e2e mAP50-95 **−0.001 ~ −0.002**，两头差距反而略微扩大。

**原因（诊断错误）**：所谓"e2e 召回缺口"是 **conf 阈值假象** ——
在 conf=0.25 下 e2e 召回低，但在 conf=0.001 下 **e2e 召回反而更高（+4.7）**，只是误检更多。
靶子不存在，方法自然无效。**这个教训写进了 `docs/RESULTS_DHCD_A.md`：做误差分析时必须与指标同口径。**

## 如何打补丁

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout 9d25f80da          # 必须，否则上下文可能不匹配
git am /path/to/patches/*.patch # 0002 依赖 0001，按序号顺序
```

补丁修改了 `ultralytics/utils/loss.py`、`cfg/default.yaml`、`cfg/__init__.py`，
属 AGPL-3.0 衍生作品，见仓库根目录 `NOTICE.md`。
