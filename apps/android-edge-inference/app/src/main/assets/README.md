# Assets

Place model files here before building the app:

- `model_int8.tflite`: TensorFlow Lite model, preferably INT8 or UINT8 quantized.
- `labels.txt`: optional label file, one label per line.

Recommended first model:

- MobileNetV2 / EfficientNet-Lite INT8 image classification model.
- Input shape `[1, 224, 224, 3]`.
