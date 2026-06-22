# 主流 AI 芯片对比（持续更新）

> 最近更新：2026-05-24  
> 目标：不是背参数，而是建立“端侧 AI 芯片怎么比较、怎么选型、怎么验证”的学习框架。

## 先看结论

端侧 AI 芯片不能只看 TOPS。更可靠的判断顺序是：

1. 目标模型能不能被工具链完整编译 / 部署
2. 关键算子是否落在 NPU / GPU / DSP 上，而不是悄悄 fallback 到 CPU
3. INT8 / FP16 / BF16 / INT4 等精度是否被模型和 runtime 同时支持
4. 内存带宽、片上 SRAM、显存 / 统一内存是否够用
5. 实测端到端延迟、功耗、温度和稳定性
6. 生态是否适合你的产品形态：手机、PC、机器人、摄像头、工业网关或开发板

简单说：**TOPS 是入口指标，不是选型答案。**

## 对比维度

| 维度 | 重点问题 | 为什么重要 |
|---|---|---|
| 算力类型 | INT8 / FP16 / BF16 / INT4 支持什么 | 量化策略会直接决定模型能否高效运行 |
| 内存系统 | 片上 SRAM、外部内存带宽、统一内存大小 | LLM、ViT、检测模型常常被内存而不是算力卡住 |
| 算子覆盖 | Conv、GEMM、Attention、LayerNorm、Softmax 是否支持 | 不支持的算子会 fallback，端到端延迟会突然变差 |
| 工具链 | Core ML、SNPE / QNN、TensorRT、OpenVINO、RKNN、Hailo Dataflow Compiler | 工具链决定模型转换、调试和部署成本 |
| 生态成熟度 | 文档、示例、社区、量产案例 | 学习和工程落地时能少踩很多坑 |
| 功耗和散热 | 持续性能是否会降频 | 边缘设备经常不是跑一次 benchmark，而是长期运行 |
| 部署形态 | 手机 SoC、AI PC、边缘盒子、摄像头、PCIe/M.2 加速卡 | 芯片能力必须和产品硬件形态匹配 |

## 平台总览

| 平台 | 典型硬件形态 | 主要工具链 | 适合场景 | 学习重点 |
|---|---|---|---|---|
| Apple Neural Engine | iPhone / iPad / Mac | Core ML、Metal、Accelerate | iOS/macOS 端侧 AI、影像、语音、小模型推理 | Core ML 转换、ANE/GPU/CPU 调度、统一内存 |
| Qualcomm Hexagon NPU | Android 手机、XR、AI PC、车载 | Qualcomm AI Stack、QNN、SNPE、ONNX Runtime EP | 安卓端侧、移动多模态、低功耗常驻 AI | HTP/NPU delegate、量化、算子落点分析 |
| MediaTek NPU/APU | Android 手机、平板、IoT SoC | NeuroPilot、厂商 SDK | 手机端 AI、影像增强、生成式 AI 功能 | Android 生态、模型转换、厂商适配 |
| NVIDIA Jetson | Jetson Nano / Orin / AGX 模块 | CUDA、TensorRT、JetPack、DeepStream | 机器人、无人机、边缘盒子、多路视频分析 | TensorRT engine、CUDA 生态、端到端 pipeline |
| Intel Core Ultra NPU | AI PC、轻量边缘设备 | OpenVINO、ONNX Runtime、DirectML | Windows AI PC、本地视觉/语音/小模型 | CPU/GPU/NPU 异构调度、OpenVINO 优化 |
| AMD Ryzen AI NPU | AI PC、嵌入式 x86 平台 | Ryzen AI Software、ONNX Runtime、Vitis AI / XDNA 相关工具 | AI PC、本地助手、轻量推理 | XDNA NPU、模型量化、Windows 生态 |
| Rockchip RK3588 NPU | SBC、低成本边缘盒子、工业板 | RKNN Toolkit、rknpu2 | 低成本视觉检测、摄像头盒子、教学实验 | RKNN 转换、INT8 量化、Linux 部署 |
| Hailo-8 / Hailo-8L | M.2 / mini PCIe 加速卡、边缘设备 | Hailo Dataflow Compiler、HailoRT | 多路摄像头、视频分析、低功耗视觉 | 数据流架构、模型编译、视觉 pipeline |
| Google Edge TPU | Coral USB / M.2 / Dev Board | TensorFlow Lite、Edge TPU Compiler | 小型视觉分类/检测、低功耗原型 | TFLite INT8、算子限制、模型压缩 |

## 重点平台学习笔记

### Apple Neural Engine

Apple 的优势不是开放裸硬件细节，而是系统级集成：Core ML 会在 CPU、GPU、Neural Engine 之间做调度，开发者通常通过模型转换和性能分析来影响落点。

适合学习：

- Core ML 模型转换：PyTorch / TensorFlow / ONNX -> Core ML
- `mlprogram`、量化、palettization、weight compression
- Xcode Instruments 里观察模型是否跑在 ANE
- 移动端影像、语音、文本小模型的功耗和延迟

注意：

- 不是所有算子都会落到 ANE。
- 动态 shape、复杂 control flow、部分 transformer 算子可能影响部署。
- Apple 生态适合产品落地，但不适合学习底层 NPU 指令细节。

### Qualcomm Hexagon NPU

Qualcomm 的 Hexagon NPU 是 Android / XR / 移动 SoC 里非常重要的端侧 AI 加速器。实际部署通常走 QNN、SNPE、ONNX Runtime Execution Provider 或厂商封装。

适合学习：

- Android 端侧推理链路
- INT8 / INT16 量化和校准
- CPU / GPU / NPU / DSP 异构调度
- 常驻低功耗 AI：语音唤醒、传感器感知、相机增强

注意：

- 同样叫 Snapdragon，不同代际和不同 SKU 的 NPU 能力差异很大。
- Android 手机上经常受系统权限、厂商 ROM、runtime 版本影响。
- 需要关注 unsupported op 和 fallback，否则 nominal TOPS 没意义。

### NVIDIA Jetson

Jetson 更像“边缘 GPU 计算平台”，优势是 CUDA / TensorRT / DeepStream 生态完整，适合把研究模型快速变成边缘 demo 或产品原型。

适合学习：

- TensorRT engine 构建和 profiling
- FP16 / INT8 calibration
- 多路视频解码 + 推理 + 后处理 pipeline
- ROS、机器人、工业视觉、边缘盒子部署

注意：

- 功耗和散热设计会强烈影响持续性能。
- GPU 很灵活，但也需要处理显存、batch、stream、数据拷贝。
- Jetson 适合学习工程化部署，不一定是最低功耗方案。

### Intel Core Ultra NPU

Intel AI PC 的核心学习点是 CPU / GPU / NPU 协同。OpenVINO 是主要入口，适合研究 Windows / x86 生态里的端侧推理。

适合学习：

- OpenVINO 模型优化和部署
- ONNX 模型在 CPU / GPU / NPU 的分配
- AI PC 场景：背景虚化、语音增强、OCR、本地小模型
- 混合精度和图优化

注意：

- NPU 适合低功耗、固定图推理；复杂模型可能仍需要 GPU 或 CPU。
- 不同 Core Ultra 代际的平台 TOPS 和 NPU 能力差异明显。

### AMD Ryzen AI NPU

AMD Ryzen AI 的重点是 XDNA / XDNA2 NPU 和 Windows AI PC 生态。它适合观察 PC 侧 NPU 如何从“有硬件”走向“有实际软件负载”。

适合学习：

- Ryzen AI Software
- ONNX Runtime 路径
- 小模型本地推理和多媒体增强
- NPU 与 iGPU / CPU 的分工

注意：

- 生态还在快速发展，工具链版本影响很大。
- 许多生成式 AI 工作负载仍可能优先使用 GPU 或 CPU。

### Rockchip RK3588 NPU

RK3588 是低成本边缘板卡里很常见的选择，常见标称 NPU 算力为 6 TOPS。它适合做“便宜、可摸到、能跑视觉模型”的教学和原型。

适合学习：

- RKNN Toolkit 模型转换
- YOLO / MobileNet / OCR 等视觉模型 INT8 部署
- Linux SBC 上摄像头输入、推理、后处理
- 低成本设备的工程约束

注意：

- 算子支持和转换限制要提前验证。
- 文档、驱动、板卡厂商镜像质量会影响开发体验。
- 适合视觉类边缘推理，不适合作为通用大模型平台。

### Hailo-8

Hailo-8 是面向边缘视觉的专用 AI 加速器，官方标称 Hailo-8 可达 26 TOPS。它的特点是数据流架构，常见形态是 M.2 / mini PCIe 加速卡。

适合学习：

- 多路摄像头视频分析
- 视觉模型编译和部署
- 边缘低功耗 AI 加速卡选型
- host CPU + accelerator 的数据流设计

注意：

- 模型要经过 Hailo 编译器适配。
- 非视觉类、动态结构或 unsupported op 的模型不一定合适。
- 端到端性能要算上视频解码、预处理、后处理和数据传输。

## TOPS 怎么看

TOPS = Tera Operations Per Second，表示每秒万亿次操作。它有用，但容易误导。

读 TOPS 时要问：

1. 是 INT8、INT4、FP16 还是稀疏 INT8？
2. 是 NPU TOPS、GPU TOPS，还是整个平台总 TOPS？
3. 是峰值理论值，还是真实模型端到端吞吐？
4. 是否依赖 sparsity、特定 batch、特定算子？
5. 内存带宽能不能喂饱计算单元？

一个常见现象：

```text
芯片 A：TOPS 高，但模型大量 fallback 到 CPU
芯片 B：TOPS 低一些，但算子覆盖完整、工具链稳定
```

真实项目里，芯片 B 可能更快、更省电、更容易量产。

## 选型路径

### 1. 先确定模型类型

| 模型类型 | 优先关注 |
|---|---|
| CNN 分类 / 检测 | INT8 Conv、NMS、图像预处理、视频输入 |
| ViT / Transformer | MatMul、Attention、LayerNorm、Softmax、内存带宽 |
| LLM / SLM | 权重量化、KV cache、内存容量、token latency |
| 语音模型 | 流式推理、低延迟、低功耗常驻 |
| 多模态模型 | 内存、算子覆盖、CPU/GPU/NPU 协同 |

### 2. 再确认部署形态

| 形态 | 候选平台 |
|---|---|
| iOS / macOS App | Apple Neural Engine + Core ML |
| Android App | Qualcomm / MediaTek NPU + NNAPI / QNN / 厂商 SDK |
| Windows AI PC | Intel Core Ultra NPU、AMD Ryzen AI NPU、GPU |
| 机器人 / 边缘盒子 | NVIDIA Jetson |
| 低成本 Linux 摄像头盒子 | RK3588、Hailo、Edge TPU |
| 多路视频分析 | Hailo、Jetson、部分工业 AI 加速卡 |

### 3. 最后做实测

每个平台都建议跑同一套最小 benchmark：

```text
model convert success?
unsupported ops count?
CPU fallback count?
single inference latency p50/p95?
end-to-end latency with preprocessing/postprocessing?
peak memory?
steady-state power?
temperature after 30 minutes?
accuracy drop after quantization?
```

## 推荐实验路线

### 入门：理解量化和算子

1. 跑本仓库的 INT8 MatMul 实验
2. 跑 Softmax from Scratch 实验
3. 理解 Conv / MatMul / Softmax / LayerNorm 为什么是推理核心算子

### 进阶：跑一个真实视觉模型

建议模型：

- MobileNetV2
- YOLOv5n / YOLOv8n
- PP-OCR 轻量模型

对比：

- FP32 CPU
- FP16 GPU
- INT8 NPU / accelerator
- 端到端 pipeline，包括 resize、normalize、NMS、绘图

### 高阶：跑一个小语言模型

建议先从小模型开始：

- TinyLlama / Qwen 小尺寸模型
- 量化到 INT4 / INT8
- 测 token latency、prefill latency、内存占用

重点不是“能不能启动”，而是：

- 首 token 延迟
- 每秒 token 数
- KV cache 内存增长
- 长上下文下是否稳定
- CPU/GPU/NPU 各自承担了什么

## 学习时容易踩的坑

1. **只看 TOPS**：忽略工具链、算子覆盖和内存。
2. **只测模型推理**：没算预处理、后处理、数据搬运。
3. **只看平均延迟**：没有看 p95 / p99 和热稳定性。
4. **忽略量化精度**：INT8 快了，但 mAP / accuracy 掉太多。
5. **把开发板当量产设备**：开发板能跑不代表产品能长期稳定跑。
6. **忽略生态锁定**：模型转换链路越封闭，后续迁移成本越高。

## 持续更新清单

每次更新芯片信息时，建议补这几项：

```text
平台：
具体芯片 / 模块：
官方工具链版本：
支持精度：
标称 TOPS：
实测模型：
输入尺寸：
量化方式：
平均延迟：
p95 延迟：
功耗：
温度：
准确率变化：
主要 unsupported ops：
是否出现 CPU fallback：
参考链接：
结论：
```

## 参考链接

- Apple Core ML：https://developer.apple.com/machine-learning/core-ml/
- Apple Neural Engine Transformer 研究：https://machinelearning.apple.com/research/neural-engine-transformers
- Qualcomm Hexagon NPU：https://www.qualcomm.com/processors/hexagon
- Qualcomm AI Engine：https://www.qualcomm.com/processors/ai-engine
- NVIDIA Jetson Orin：https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- Intel OpenVINO / Core Ultra AI PC：https://newsroom.intel.com/client-computing/more-than-500-ai-models-run-optimized-on-intel-core-ultra-processors
- AMD Ryzen AI Software：https://www.amd.com/en/products/software/ryzen-ai-software.html
- MediaTek Dimensity AI：https://www.mediatek.com/dimensity-9400
- Rockchip RK3588：https://www.rockchips.net/product/rk3588/
- Hailo-8：https://hailo.ai/product-hailo/hailo-8/
