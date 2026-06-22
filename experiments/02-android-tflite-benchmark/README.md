# 实验：Android TFLite 端侧推理 Benchmark

## 目标

用真实 Android 手机跑一个 TFLite 图像分类模型，记录端侧推理的准确性、时延和部署问题。

## 实验链路

```text
TFLite model
-> Android APP assets
-> TensorFlow Lite Interpreter
-> image preprocess
-> inference
-> top-1 result + latency benchmark
```

## 最小任务

- [ ] 准备 `model_int8.tflite`
- [ ] 准备 `labels.txt`
- [ ] 在 APP 中通过 URL 下载模型
- [ ] Android Studio 成功构建 APP
- [ ] 真机安装并选择图片推理
- [ ] 记录单次 latency
- [ ] 记录 30 次 benchmark P50 / P95
- [ ] 写出本次实验结论

## 推荐模型

第一轮建议使用输入为 `[1, 224, 224, 3]` 的分类模型：

- APP 默认模型：MobileNet V1 0.25 224 quantized
- MobileNetV2 INT8
- EfficientNet-Lite INT8

不要一开始就上 YOLO 或 LLM。先把最小图像分类链路跑通，再扩展到检测和生成式模型。
