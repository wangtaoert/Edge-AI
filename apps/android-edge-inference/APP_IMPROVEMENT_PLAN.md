# APP 完善计划

> 目标：把 Android Edge Inference 从最小 demo 推进为接近成熟应用的端侧 AI 学习工程。

## 成熟应用标准

- 信息架构清楚：模型管理、图片输入、推理结果和实验记录各有明确位置。
- 状态可恢复：用户上次输入的模型 URL / labels URL 能自动恢复。
- 错误可解释：下载失败、模型无效、shape 不支持、推理失败都有明确反馈。
- 数据不乱：坏模型不能覆盖好模型，labels 不能和模型错配。
- 代码可扩展：Activity 不直接承载模型仓库、下载器、预处理和推理细节。
- 性能可测：单次推理和 benchmark 都能稳定输出可记录指标。
- 后续可发布：逐步补齐图标、签名、隐私说明、release 构建和崩溃日志。
- 支撑能力完整：提供关于、隐私说明和诊断信息导出。

## 阶段 1：可用性和稳定性

- [x] Android 工程可通过 Gradle 构建。
- [x] APP 支持从网络下载 `.tflite` 模型。
- [x] APP 支持 assets 内置模型兜底。
- [x] 下载前后校验模型，避免坏模型覆盖可用模型。
- [x] 模型、图片、推理结果分区展示。
- [x] 明确显示当前模型来源、shape、dtype、labels 数量和模型大小。
- [x] 推理按钮在缺少模型或图片时禁用。

## 阶段 2：推理链路完善

- [x] 把模型加载、下载、推理逻辑从 Activity 中拆分。
- [x] 支持清除已下载模型，回到 assets 兜底模型。
- [x] 记录 last model URL / labels URL。
- [x] 对输入 shape 做更清晰的错误提示。
- [x] 记录 preprocessing 设置：RGB、`[0,1]`、NHWC。

## 阶段 3：性能实验

- [x] benchmark 输出 min / p50 / p95 / max。
- [x] 记录首轮 latency 和热身后 latency。
- [ ] 支持不同线程数对比。
- [x] 接入 NNAPI delegate 开关。
- [ ] 接入 GPU delegate 开关。

## 阶段 3.5：成熟应用体验

- [x] 下载过程显示进度状态。
- [x] 内置默认图片分类模型下载 URL。
- [x] 增加模型库页面或预设模型列表。
- [x] 增加设置区：线程数、benchmark 次数。
- [x] 增加设置区：预处理方式。
- [x] 支持复制 benchmark 结果。
- [x] 支持分享 benchmark 结果。
- [x] 支持保存实验结果到本地文本。
- [x] 增加空状态、成功状态和错误状态的统一视觉风格。

## 阶段 3.8：发布级支撑能力

- [x] 增加 About 页面。
- [x] 增加 Privacy 页面。
- [x] 增加 Diagnostics 诊断报告。
- [x] 支持复制、分享、保存诊断报告。
- [x] 增加正式 App 图标。
- [x] 禁用系统备份并限制 HTTPS 下载。
- [x] 增加 release buildType 和 ProGuard 规则占位。
- [x] 增加崩溃日志或错误日志文件。
- [ ] 增加 release 签名配置。

## 阶段 4：模型类型扩展

- [ ] 图像分类：当前主线。
- [ ] 目标检测：增加 bounding box 后处理和绘制。
- [ ] 文本模型：增加 tokenizer / text input。
- [ ] 小语言模型：记录 prefill、decode、token latency。

## 当前优先级

继续补阶段 3.5 的成熟应用体验，然后进入 NNAPI / GPU delegate 和预设模型库。
