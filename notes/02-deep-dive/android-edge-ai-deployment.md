# Android 端侧 AI 部署路线

> 目标：把“模型能跑”拆成可验证的工程链路，而不是只看 TOPS 或 demo 截图。

## 1. 第一阶段：CPU baseline

先用 TensorFlow Lite Java Interpreter 跑 CPU baseline。

原因：

1. 环境最简单，问题最少。
2. 可以先验证模型输入输出、预处理、标签和结果是否正确。
3. 后续接 NNAPI / GPU / NPU delegate 时，CPU 结果可作为参考。

最小指标：

- 单次 latency
- P50 latency
- P95 latency
- Top-1 结果是否合理

## 2. 第二阶段：量化模型

端侧优先尝试 INT8 / UINT8 模型。

重点检查：

1. 输入 tensor 的 dtype 和 shape。
2. input scale / zero-point 是否和预处理一致。
3. 输出 tensor 是否需要反量化。
4. accuracy 是否出现明显下降。

常见错误：

- 模型要求 `[-1, 1]` 归一化，但 APP 按 `[0, 1]` 输入。
- RGB / BGR 顺序弄反。
- labels 和模型类别顺序不一致。
- 只测平均时延，没有看 p95。

## 3. 第三阶段：Delegate 加速

CPU baseline 稳定后再接 delegate。

优先顺序建议：

1. NNAPI delegate：观察 Android 系统能否把算子下发到 NPU / DSP。
2. GPU delegate：适合部分 FP16 视觉模型。
3. 厂商 SDK：如 Qualcomm QNN / SNPE，适合深入 Android NPU。

不要只看是否“启用了 delegate”，要确认：

- unsupported op 数量
- 是否 fallback 到 CPU
- 端到端 latency 是否真的下降
- p95 是否变差

## 4. 记录模板

```text
模型：
手机：
SoC：
Android：
Runtime：
Delegate：
输入尺寸：
精度：
模型大小：
单次 latency：
P50：
P95：
结果是否正确：
主要问题：
下一步：
```

## 5. 当前工程

当前仓库工程：

```text
apps/android-edge-inference
```

第一版只实现 TFLite CPU 推理，后续再扩展 NNAPI / GPU / QNN。
