# 实验：MNIST INT8 Benchmark

## 目标
比较 FP32 与 INT8 在延迟、模型大小、精度上的差异。

## 实验步骤
1. 训练或加载基线模型
2. 导出 ONNX
3. 进行 PTQ 量化
4. 运行推理 benchmark
5. 输出结果与分析

## 指标
- Top-1 Accuracy
- P50 / P95 Latency
- Model Size
