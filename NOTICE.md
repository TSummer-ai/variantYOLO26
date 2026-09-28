# NOTICE：第三方组件与许可

## ultralytics（AGPL-3.0）

本项目的训练、推理、评测均基于 [ultralytics](https://github.com/ultralytics/ultralytics)，
基线版本固定为 **commit `9d25f80da`（v8.4.160）**。

`experiments/negative-results/patches/` 中的补丁**修改了 ultralytics 源码**
（`ultralytics/utils/loss.py`、`cfg/default.yaml`、`cfg/__init__.py`），
这些修改属于 AGPL-3.0 定义下的衍生作品，因此：

> **本仓库整体以 AGPL-3.0 发布。**

### 使用本仓库时的注意事项

1. **如果只是把 ultralytics 当依赖使用**（`pip install ultralytics==8.4.160` + 运行 `scripts/` 里的脚本），
   `scripts/`、`results/`、`docs/` 中的原创内容在多数解读下不构成衍生作品。若团队希望这部分采用
   更宽松的许可（MIT / Apache-2.0），需要**先移除 `experiments/negative-results/patches/`**，
   并在文档中说明补丁另行发布。**请在团队内达成一致后再改许可，避免后续返工。**

2. **如果打上补丁再分发**（`git am` 后发布修改后的 ultralytics），则必须遵守 AGPL-3.0：
   提供完整对应源码，且通过网络提供服务时同样要向用户提供源码。

3. 本仓库**不包含** VisDrone / COCO 数据集，也不包含 ultralytics 的模型权重文件。
   数据集与预训练权重遵循各自原始许可：
   - VisDrone2019-DET：学术研究用途
   - ultralytics 预训练权重：AGPL-3.0

## 其他依赖

| 组件 | 许可 |
|---|---|
| PyTorch / torchvision | BSD-3-Clause |
| faster-coco-eval | Apache-2.0 |
| pycocotools | BSD-2-Clause |
| matplotlib / numpy / opencv / pillow | 各自开源许可（BSD/MIT/Apache 系） |
| tensorboard | Apache-2.0 |

## 数据集的 0-based 标注坑

VisDrone 官方 COCO 格式标注的 `category_id` 从 **0** 开始，而 ultralytics 的 `convert_coco`
会执行 `id-1`。直接转换会导致**全部类别错位**。`scripts/prepare_visdrone.sh` 已处理
（categories 与 annotations 同时 +1），复现时请勿自行简化该步骤。
