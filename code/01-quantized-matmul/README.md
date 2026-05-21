# 01 - Quantized MatMul

## 目标

从零实现一个简化版 INT8 量化矩阵乘，理解三件事：

1. `scale / zero-point` 如何把 FP32 映射到 INT8
2. INT8 乘法为什么通常用 INT32 累加
3. 反量化后结果与 FP32 baseline 的误差来自哪里

这不是追求最快的实现，而是把端侧推理里最核心的量化路径拆开看清楚。

## 最小任务

- [x] 实现 INT8 quantize / dequantize
- [x] 实现 INT8 matmul + bias
- [x] 与 FP32 结果对比误差

## 快速运行

在仓库根目录执行：

```powershell
python -m pip install -r code\01-quantized-matmul\requirements.txt
```

然后运行实验：

```powershell
python code\01-quantized-matmul\quantized_matmul.py
```

你会看到类似输出：

```text
shape: A=(4, 8), B=(8, 5), bias=(5,)
A qparams: scale=..., zero_point=...
B qparams: scale=..., zero_point=...
max_abs_error: ...
mean_abs_error: ...
mean_relative_error: ...
```

可以调整矩阵尺寸和随机种子：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --m 16 --k 32 --n 8 --seed 7
```

切换为对称量化：

```powershell
python code\01-quantized-matmul\quantized_matmul.py --symmetric
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

也可以换矩阵尺寸、随机种子或输出路径：

```powershell
python code\01-quantized-matmul\visualize_quantization.py --m 16 --k 32 --n 8 --seed 7 --output code\01-quantized-matmul\demo.png
```

## 核心公式

### 量化

把浮点数 `x` 映射到整数 `q`：

```text
q = round(x / scale + zero_point)
q = clamp(q, -128, 127)
```

其中：

- `scale` 控制一个整数步长对应多少浮点范围
- `zero_point` 表示浮点 0 在整数域里的位置

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

注意累加器使用 INT32。即使输入是 INT8，`K` 维度很大时，连续乘加也很容易超过 INT8/INT16 的范围。

## 文件说明

- [`quantized_matmul.py`](./quantized_matmul.py)：可直接运行的最小实现

主要函数：

- `calculate_qparams`：根据张量范围计算 `scale / zero_point`
- `quantize`：FP32 -> INT8
- `dequantize`：INT8 -> FP32
- `quantized_matmul`：INT8 operands + INT32 accumulator + FP32 output
- `compare_outputs`：输出误差指标

## 观察建议

运行时重点看这几个现象：

1. `K` 越大，乘加链越长，量化误差越容易累积。
2. 输入范围里有离群值时，`scale` 会变大，普通值的分辨率会变差。
3. 对称量化实现更简单，但非对称量化对偏移分布可能更友好。
4. bias 仍然以 FP32 加回去，这和很多教学版实现一致；真实部署里 bias 常会按 `scale_a * scale_b` 量化到 INT32。

## 下一步扩展

- 增加 per-channel weight quantization
- 把 bias 也量化成 INT32
- 增加输出 requantize：FP32 / INT32 accumulator -> INT8 output
- 用纯 Python 循环写一版，直观看出硬件 MAC 的工作方式
- 对比 NumPy 向量化版本与手写循环版本的速度差异
