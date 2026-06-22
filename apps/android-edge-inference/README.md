# Android Edge Inference

这是一个用于学习端侧 AI 推理的 Android 工程骨架。第一阶段目标不是做漂亮 APP，而是把下面这条链路跑通：

```text
联网下载 TFLite 模型 -> Android APP 私有目录 -> 手机 CPU 推理 -> 单张图片分类 -> latency benchmark
```

## 技术路线

- 平台：Android 手机
- Runtime：TensorFlow Lite Java Interpreter
- 首选模型：INT8 / UINT8 TFLite 图像分类模型
- 输入假设：`[1, 224, 224, 3]`，NHWC
- 当前加速：CPU 多线程
- 模型来源：APP 内联网下载，assets 内置模型兜底
- 后续扩展：NNAPI delegate / GPU delegate / Qualcomm QNN

## 目录

```text
apps/android-edge-inference/
  settings.gradle.kts
  build.gradle.kts
  app/
    build.gradle.kts
    src/main/
      AndroidManifest.xml
      java/com/edgeai/inference/
        MainActivity.java
        ModelRepository.java
        ModelDownloadManager.java
        TfliteImageClassifier.java
        AppPreferences.java
      assets/
```

## 工程结构

- `MainActivity`：负责 UI、用户操作和页面状态。
- `ModelRepository`：负责模型加载、assets 兜底、下载模型安装和 labels 管理。
- `ModelDownloadManager`：负责网络下载、临时文件和下载进度。
- `TfliteImageClassifier`：负责图像预处理、TFLite 推理和 benchmark。
- `AppPreferences`：保存上次输入的模型 URL / labels URL。

## 准备模型

APP 支持两种模型来源。

### 方式 1：APP 内联网下载

在 APP 首页输入：

```text
Model URL:  一个可直接下载的 .tflite 链接
Labels URL: 一个可直接下载的 labels.txt 链接，可选
```

然后点击 `Download model`。下载完成后，模型会保存到 APP 私有目录，并立即重新加载。

APP 默认已经填入一个轻量图片分类模型：

```text
Model:  MobileNet V1 0.25 224 quantized
URL:    https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_0.25_224_quantized/1/default/1?lite-format=tflite
Labels: https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt
```

如果 URL 被改乱，可以点击 `Default model` 恢复默认值。

也可以直接从页面里的预设模型库选择：

| 预设 | 用途 | 模型 |
|---|---|---|
| Small INT8 | 快速验证下载和推理链路，体积小 | MobileNet V1 0.25 224 quantized |
| Full INT8 | 更完整的 INT8 分类基线 | MobileNet V1 1.0 224 quantized |
| Float model | 对比浮点模型行为和体积 | MobileNet V1 1.0 224 float |

注意：

- URL 必须是 HTTPS 文件直链，不是网页预览地址。
- 当前 starter app 只支持图像分类模型，输入需为 `[1,H,W,3]`。
- 下载模型会先写入临时文件并创建 TFLite Interpreter 校验，校验通过后才会替换当前模型。
- 如果下载失败或模型无效，APP 会保留原来的可用模型。
- 如果 `Labels URL` 留空，APP 会清除旧的下载标签，避免旧标签和新模型错配。
- 上次输入过的 URL 会保存在本机，下次打开 APP 自动恢复。
- labels 文件可选；不提供时会显示 `class_0`、`class_1` 这类数字类别。

### 方式 2：assets 内置兜底模型

把模型和标签放到：

```text
app/src/main/assets/model_int8.tflite
app/src/main/assets/labels.txt
```

`labels.txt` 可选，每行一个类别名。如果没有标签，APP 会显示 `class_0`、`class_1` 这类数字类别。

推荐第一批模型：

- MobileNetV2 INT8
- EfficientNet-Lite INT8
- 任何输入为 `[1, 224, 224, 3]` 的 TFLite 分类模型

## 运行方式

1. 用 Android Studio 打开 `apps/android-edge-inference`。
2. 等 Gradle Sync 完成。
3. 连接 Android 手机并开启 USB 调试。
4. 运行 `app`。
5. 在 APP 里选择一张图片，点击 `Run inference` 或 `Benchmark x30`。

## 命令行构建

本工程已带 Gradle Wrapper。Windows 下可在当前目录执行：

```powershell
$env:JAVA_HOME="C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
$env:PATH="$env:JAVA_HOME\bin;$env:PATH"
.\gradlew.bat assembleDebug
```

Debug APK 输出位置：

```text
app/build/outputs/apk/debug/app-debug.apk
```

如果 Android Studio 使用自己的 JDK，也可以在 Android Studio 中直接构建，不需要手动设置 `JAVA_HOME`。

## 当前会记录什么

APP 页面会显示：

- 当前加载的是下载模型还是 assets 兜底模型
- 模型大小
- 输入 tensor 类型和 shape
- 输出 tensor 类型和 shape
- 当前预处理模式
- Top-1 类别
- 单次推理 latency
- 30 次 benchmark 的 min / P50 / P95 / max latency
- 可复制、分享、保存当前实验结果
- 保存结果会写入 APP 私有外部目录下的 `edge_ai_results.txt`
- About / Privacy / Diagnostics 支撑入口

复制、分享、保存的实验报告会包含：

- 模型来源、大小、input/output shape
- model URL / labels URL
- runtime、线程数、benchmark 次数
- 图片尺寸
- 推理或 benchmark 结果

Diagnostics 报告会包含：

- App 版本
- Android 设备信息
- runtime、线程数、benchmark 次数、预处理方式
- 当前模型信息和下载状态
- model URL / labels URL
- 最近一次结果
- 错误日志路径和最近一次错误摘要

Privacy 页面说明：

- 推理在本机执行。
- 网络只用于通过 HTTPS 下载用户选择或输入的模型和标签文件。
- 模型、设置、实验记录、错误日志保存在 APP 私有存储中，除非用户主动分享。

## 发布级配置

当前工程已经包含：

- 自定义 adaptive launcher icon。
- 禁用 Android 系统备份，避免模型和实验数据被系统备份。
- 禁用明文流量，只允许 HTTPS 下载。
- release buildType 和 ProGuard 规则占位。
- 本地错误日志：`edge_ai_errors.txt`。

## Runtime 与 Benchmark

当前支持两种 runtime：

| Runtime | 说明 |
|---|---|
| CPU | 默认路径，适合做稳定 baseline |
| NNAPI | 尝试通过 Android NNAPI 使用系统可用加速路径，实际是否落到 NPU 取决于手机 SoC、系统和算子支持 |

Benchmark 会输出：

- First run latency：本轮 benchmark 的第一次推理，常用于观察首轮开销。
- Warm min / P50 / P95 / max latency：第一次推理之后的多轮统计，更接近稳定运行状态。

## 当前预处理假设

当前版本面向图像分类模型，预处理链路为：

```text
Bitmap -> resize 到模型输入 H/W -> RGB -> preprocess mode -> 按 tensor dtype 写入
```

支持三种预处理模式：

| 模式 | 说明 |
|---|---|
| `RGB [0,1]` | 每个通道除以 255，适合多数 TFLite 示例模型 |
| `RGB [-1,1]` | 每个通道映射到 `[-1,1]`，适合部分 MobileNet / TF 模型 |
| `RGB ImageNet mean/std` | 使用 ImageNet mean/std 标准化，适合部分 PyTorch 转换模型 |

支持输入 tensor：

- `FLOAT32`
- `UINT8`
- `INT8`

当前只支持 RGB。若模型要求 BGR、中心裁剪、letterbox 或检测模型预处理，需要继续扩展。

## 下一步建议

第一阶段先跑 CPU baseline。等 CPU baseline 稳定后，再做三组对比：

```text
FP32 CPU
INT8 CPU
INT8 NNAPI delegate
```

每次都记录：

- 模型名
- 手机型号
- Android 版本
- runtime / delegate
- accuracy 或肉眼检查结果
- P50 latency
- P95 latency
- 是否发热或降频
