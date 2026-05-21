# 01 - Quantized MatMul

## 目标

从零实现一个简化版 INT8 量化矩阵乘，理解三件事：

1. `scale / zero-point` 如何把 FP32 映射到 INT8
2. INT8 乘法为什么通常用 INT32 累加
3. 反量化、bias、requantize 这些步骤如何影响最终误差

这不是追求最快的实现，而是把端侧推理里最核心的量化路径拆开看清楚。

## 最小任务

- [x] 实现 INT8 quantize / dequantize
- [x] 实现 INT8 matmul + bias
- [x] 与 FP32 结果对比误差
- [x] 增加 per-channel weight quantization
- [x] 把 bias 量化成 INT32
- [x] 增加输出 requantize
- [x] 增加纯 Python MAC 循环
- [x] 对比 NumPy 向量化版本与手写循环版本的速度差异

## 快速运行

在仓库根目录执行：

```powershell
python -m pip install -r code\01-quantized-matmul\requirements.txt
```

然后运行基础实验：

```powershell
python code\01-quantized-matmul\quantized_matmul.py
```

你会看到类似输出：

```text
shape: A=(4, 8), B=(8, 5), bias=(5,)
mode: tensor, accumulator: numpy, bias: int32
A qparams: scale=..., zero_point=...
B qparams: scale=..., zero_point=...
max_abs_error: ...
mean_abs_error: ...
mean_relative_error: ...
```

## 可视化量化过程

生成一张图，直观看 FP32 如何被映射到 INT8，以及反量化误差长什么样：

```powershell
python code\01-quantized-matmul\visualize_quantization.py
```

默认输出：

```text
code\01-quantized-matmul\quantization_visualization.png
```

图里包含四块内容：

1. FP32 连续值与反量化后阶梯值的对比
2. FP32 数值到 INT8 code 的映射关系
3. 原始矩阵 `A` 的数值热力图
4. `A` 的量化误差热力图

## 扩展 1：per-channel weight quantization

默认的 per-tensor 量化是整个权重矩阵 `B` 共用一个 scale：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --mode tensor --symmetric
```

per-channel 量化是让 `B` 的每个输出通道各自拥有一个 scale。对矩阵乘 `A @ B` 来说，`B` 的每一列对应一个输出通道：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --mode per-channel --symmetric
```

观察点：

- per-tensor 简单，元数据少，但容易被某个离群通道拉大 scale。
- per-channel 多保存一组 scale，但通常能降低权重量化误差。
- 端侧推理里，weight per-channel 是非常常见的精度折中方案。

## 扩展 2：bias 量化成 INT32

真实 INT8 推理里，bias 通常不是直接以 FP32 加回去，而是先按输入和权重的 scale 量化成 INT32：

```text
bias_int32[j] = round(bias_fp32[j] / (scale_a * scale_b[j]))
acc_int32[:, j] += bias_int32[j]
```

当前脚本默认使用 INT32 bias：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --mode per-channel --symmetric
```

如果想对比教学版的 FP32 bias：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --mode per-channel --symmetric --fp32-bias
```

观察点：

- INT32 bias 更贴近真实部署链路。
- FP32 bias 误差通常略小，但硬件路径不够真实。
- bias 的量化 scale 必须和 accumulator 的反量化 scale 对齐。

## 扩展 3：输出 requantize

矩阵乘输出常常不会停在 FP32，而是继续量化成下一个算子的 INT8 输入：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --mode per-channel --symmetric --requantize-output
```

链路变成：

```text
FP32 A/B -> INT8 A/B -> INT32 accumulator -> INT8 output -> dequantized output
```

观察点：

- requantize 会再引入一轮离散化误差。
- 如果输出范围估计不准，误差会明显变大。
- 多算子串联时，输出量化策略会影响后续层。

## 扩展 4：纯 Python MAC 循环

用手写三重循环替代 NumPy 的矩阵乘：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --m 4 --k 8 --n 5 --loop --symmetric
```

核心逻辑等价于硬件 MAC 阵列在做的事：

```text
for row in M:
  for col in N:
    acc = 0
    for inner in K:
      acc += int8_a[row, inner] * int8_b[inner, col]
```

观察点：

- 手写循环更慢，但最直观。
- NumPy 版本更像调用底层优化 kernel。
- NPU / DSP 的价值，就是把大量 MAC 并行化、流水化、低功耗化。

## 扩展 5：速度对比

运行 benchmark：

```powershell
python code\01-quantized-matmul\benchmark_quantized_matmul.py
```

调整尺寸：

```powershell
python code\01-quantized-matmul\benchmark_quantized_matmul.py --m 32 --k 64 --n 32
```

你会看到几类实现的平均耗时：

- FP32 NumPy matmul
- INT8 per-tensor NumPy accumulator
- INT8 per-channel NumPy accumulator
- INT8 per-tensor Python-loop accumulator

注意：这个 benchmark 是教学用，不代表真实硬件 INT8 性能。NumPy 在普通 CPU 上未必会走 INT8 专用加速路径；真实收益要看硬件 ISA、kernel 实现和内存布局。

## 核心公式

### 量化

把浮点数 `x` 映射到整数 `q`：

```text
q = round(x / scale + zero_point)
q = clamp(q, -128, 127)
```

### 反量化

把整数 `q` 近似恢复为浮点数：

```text
x_hat = scale * (q - zero_point)
```

### INT8 MatMul

矩阵乘的整数累加形式：

```text
acc_int32 = sum((qa - za) * (qb - zb))
y_hat = acc_int32 * scale_a * scale_b + bias
```

per-channel 时，每个输出通道使用自己的 `scale_b[j]`：

```text
y_hat[:, j] = acc_int32[:, j] * scale_a * scale_b[j]
```

## 文件说明

- [`quantized_matmul.py`](./quantized_matmul.py)：量化矩阵乘主实验
- [`visualize_quantization.py`](./visualize_quantization.py)：生成量化过程可视化图片
- [`benchmark_quantized_matmul.py`](./benchmark_quantized_matmul.py)：对比不同实现的耗时
- [`requirements.txt`](./requirements.txt)：Python 依赖

主要函数：

- `calculate_qparams`：per-tensor `scale / zero_point`
- `calculate_per_channel_qparams`：per-channel `scale / zero_point`
- `quantize` / `dequantize`：per-tensor 量化和反量化
- `quantize_per_channel` / `dequantize_per_channel`：per-channel 量化和反量化
- `quantize_bias_to_int32`：把 FP32 bias 转成 INT32 accumulator bias
- `int32_matmul_loop`：纯 Python 三重循环 MAC
- `quantized_matmul`：per-tensor 量化矩阵乘
- `quantized_matmul_per_channel`：per-channel 权重量化矩阵乘
- `requantize_output`：把输出再量化成 INT8

## 观察建议

1. `K` 越大，乘加链越长，量化误差越容易累积。
2. 输入范围里有离群值时，`scale` 会变大，普通值的分辨率会变差。
3. 对称量化实现更简单，非对称量化对偏移分布可能更友好。
4. per-channel 权重量化通常比 per-tensor 更稳，但需要额外 scale 元数据。
5. INT32 accumulator 和 INT32 bias 是端侧 INT8 推理链路里非常关键的稳定性设计。
