# TensorRT 加速（实测 3.87× 网络加速，精度零损失）

## 实测结果（RTX 4060 Laptop 8GB，imgsz=960，batch=1）

| 配置 | 纯网络前向 | 端到端 predict | 端到端 FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| PyTorch FP32 | ~9.6 ms | 8.67 ms | 115 | 0.440 | 0.263 |
| PyTorch FP16 | 8.33 ms | 8.03 ms | 125 | 0.438 | 0.261 |
| TRT FP32 | 4.11 ms (2.03×) | 7.38 ms | 135 | 0.441 | 0.263 |
| **TRT FP16** | **2.15 ms (3.87×)** | **5.39 ms (1.61×)** | **186** | **0.440** | **0.262** |

测量方法：轮转交错（round-robin）消除热漂移 + 取 60 次最小值（min 比 mean 更抗系统抖动）。
早期用 mean 测得的数字波动 ±9%，不可靠。

**结论**：
1. TensorRT FP16 把**网络本体**加速 3.87×，精度与 PyTorch 完全一致（0.262 vs 0.263）
2. 端到端只有 1.61× —— 因为网络降到 2.15ms 后，**预处理/后处理（约 3.2ms）成了新瓶颈**
3. 下一步提速方向已从"网络"转向"前处理（letterbox）与解码/NMS"

## 两个必须知道的坑

### 坑 1：`pip install tensorrt` 装的是 CUDA 13 版本

```bash
pip install tensorrt          # ❌ 装到 tensorrt_cu13，与 torch cu126 不匹配
pip install tensorrt-cu12     # ✅ 必须带 -cu12
```

### 坑 2：TensorRT 11 取消了 FP16 编译开关，ultralytics 会强拉 nvidia-modelopt

TensorRT 11 改为 **strongly-typed**：FP16/INT8 的 builder flag 被移除，精度必须表达在 ONNX 图里。
ultralytics 因此硬依赖 `nvidia-modelopt[onnx]>=0.44`（会连带装 `cupy-cuda12x`、`onnxruntime-gpu`、
`polygraphy`、`onnx_graphsurgeon`，并把 `onnx` 降级到 1.21、升级 `scipy`），**有污染现有环境的风险**，
且从默认 PyPI 下载极易卡死。

**本项目采用轻量替代路线**（不装 modelopt）：

```bash
# 1) 只装这些（onnxconverter-common 是纯 Python 小包）
pip install tensorrt-cu12 onnx onnxruntime onnxsim onnxconverter-common

# 2) 先导出 ONNX（这一步 ultralytics 不会触发 modelopt）
yolo export model=weights/yolo26n-visdrone-p2-960.pt format=onnx imgsz=960 half=True

# 3) 手工把 ONNX 转成 FP16（keep_io_types=True：输入输出保持 FP32，避免预处理不匹配）
python -c "
import onnx
from onnxconverter_common import float16
m = onnx.load('weights/yolo26n-visdrone-p2-960.onnx')
onnx.save(float16.convert_float_to_float16(m, keep_io_types=True),
          'weights/yolo26n-visdrone-p2-960.fp16.onnx')"

# 4) 构建 engine（复刻 ultralytics 的 metadata 格式：4 字节小端长度 + JSON + 序列化 engine）
python scripts/build_trt_engine.py --onnx weights/yolo26n-visdrone-p2-960.fp16.onnx \
       --engine weights/yolo26n-visdrone-p2-960.fp16.engine

# 5) 验证精度（应复现 mAP50 0.440 / mAP50-95 0.262）
yolo val model=weights/yolo26n-visdrone-p2-960.fp16.engine data=configs/visdrone.yaml imgsz=960

# 6) 实测速度
python scripts/bench_trt.py --pt weights/yolo26n-visdrone-p2-960.pt \
       --engines weights/yolo26n-visdrone-p2-960.fp16.engine --imgsz 960
```

## 注意事项

1. **engine 文件是"设备+版本"绑定的**：在 A 机器/驱动/TRT 版本上构建的 engine 不能拿到 B 机器用，
   必须各自重建。因此**不要提交 `.engine`**（`.gitignore` 已排除）。
2. 构建耗时：FP32 约 73 秒，FP16 约 110 秒（960 输入）。
3. engine 体积偏大（FP32 275 MB / FP16 171 MB）—— TensorRT 会为 76500 个 anchor 展开头部算子，
   属正常现象，不影响推理。
4. `batch=1` 是静态 engine；如需批量推理请用 `--batch N` 重新导出与构建。
