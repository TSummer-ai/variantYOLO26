#!/usr/bin/env python
"""手工构建 TensorRT engine（绕过 ultralytics 在 TRT 11 下对 nvidia-modelopt 的强依赖）。

背景：TensorRT 11 取消了 FP16/INT8 的 builder flag（strongly-typed），ultralytics 因此要求
      nvidia-modelopt 在 ONNX 图上做精度转换；但那会拉进 cupy-cuda12x / onnxruntime-gpu /
      polygraphy 等一大堆依赖，且有降级 onnx 的风险。
      本脚本直接从已有 ONNX 构图，用 FP32 精度（精度来自图本身），只做 kernel 融合，
      用来验证"融合本身能带来多少加速"。

metadata：复刻 ultralytics 的格式（4 字节小端长度 + JSON），从 ONNX 的 metadata_props 里读回。

用法:
    python build_trt_engine.py --onnx weights/xxx.onnx --engine weights/xxx.engine
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorrt as trt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    onnx_path = Path(a.onnx)
    engine_path = Path(a.engine) if a.engine else onnx_path.with_suffix(".fp32.engine")

    logger = trt.Logger(trt.Logger.VERBOSE if a.verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    # TRT 11: EXPLICIT_BATCH 已移除（默认即显式 batch）；STRONGLY_TYPED 表示精度由 ONNX 图决定
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    print(f"network flags = {flags} (STRONGLY_TYPED={'on' if flags else 'off'})")
    parser = trt.OnnxParser(network, logger)

    print(f"解析 ONNX: {onnx_path} ({onnx_path.stat().st_size/1048576:.1f} MB)")
    if not parser.parse(onnx_path.read_bytes()):
        for i in range(parser.num_errors):
            print("  ONNX 解析错误:", parser.get_error(i))
        raise SystemExit("ONNX 解析失败")

    print(f"网络输入 {network.num_inputs} 个 / 输出 {network.num_outputs} 个:")
    dyn = False
    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = tuple(inp.shape)
        print(f"  in  {inp.name}: {shape}")
        if any(d < 0 for d in shape):
            dyn = True
            # 动态维统一固定为 1x3x960x960（本模型只做 batch=1 静态推理）
            fixed = tuple(1 if d < 0 and j == 0 else (960 if d < 0 else d) for j, d in enumerate(shape))
            profile.set_shape(inp.name, fixed, fixed, fixed)
            print(f"      -> 动态维度固定为 {fixed}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  out {out.name}: {tuple(out.shape)}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    if dyn:
        config.add_optimization_profile(profile)
    # TRT 11 是 strongly-typed：不再设置 FP16 flag，精度由 ONNX 图决定（此处为 FP32）
    print(f"workspace={a.workspace_gb} GB | 精度由 ONNX 图决定（FP32）| 动态形状={dyn}")

    print("开始构建 engine（可能几分钟）...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine 构建失败，请加 --verbose 看详细日志")

    # 从 ONNX 取回 metadata，按 ultralytics 的格式写入（4 字节长度 + JSON）
    meta_bytes = b""
    try:
        import onnx
        m = onnx.load(str(onnx_path), load_external_data=False)
        props = {p.key: p.value for p in m.metadata_props}
        if props:
            meta = json.dumps(props)
            meta_bytes = len(meta).to_bytes(4, byteorder="little", signed=True) + meta.encode()
            print(f"写入 metadata {len(props)} 项（names/task/stride 等，供 ultralytics AutoBackend 读取）")
    except Exception as e:
        print(f"metadata 读取失败（不影响推理，但 ultralytics 可能缺少类别名）: {e}")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(meta_bytes)
        f.write(serialized)
    print(f"完成: {engine_path} ({engine_path.stat().st_size/1048576:.1f} MB)")


if __name__ == "__main__":
    main()
