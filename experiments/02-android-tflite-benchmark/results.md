# Results

## 设备信息

| 项目 | 值 |
|---|---|
| 手机型号 | 待填写 |
| SoC | 待填写 |
| Android 版本 | 待填写 |
| APP 版本 | 0.1.0 |
| Runtime | TensorFlow Lite Java Interpreter |
| Delegate | CPU / NNAPI |
| 网络 | Wi-Fi / 5G / 离线，待填写 |

## 模型信息

| 项目 | 值 |
|---|---|
| 模型 | 待填写 |
| 模型 URL | 待填写 |
| Labels URL | 待填写 |
| 输入 shape | 待填写 |
| 精度 | INT8 / UINT8 / FP32 |
| 模型大小 | 待填写 |
| labels | 待填写 |
| 预设模型 | Small INT8 / Full INT8 / Float model / 自定义 |
| 预处理 | RGB [0,1] / RGB [-1,1] / RGB ImageNet mean/std |
| Runtime | CPU / NNAPI |

## Benchmark

| 配置 | Top-1 检查 | First run(ms) | Warm min(ms) | Warm P50(ms) | Warm P95(ms) | Warm max(ms) | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| CPU | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| NNAPI | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待扩展 |
| GPU | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待扩展 |

## 结论

待实验完成后填写：

1. 这个模型在手机 CPU 上是否可用？
2. INT8 的速度收益是否明显？
3. 是否存在首帧慢、发热、p95 波动等问题？
4. 下一轮是否值得接入 NNAPI / GPU / QNN？
