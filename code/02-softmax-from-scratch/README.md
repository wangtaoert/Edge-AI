# 02 - Softmax from Scratch

## 目标

从零实现 softmax，理解它在神经网络推理里的三个关键问题：

1. 为什么 naive softmax 容易发生数值溢出
2. 为什么实际实现通常要先减去每一行的最大值
3. 低精度推理里，fp16 / int8 / lookup table 等近似策略会怎样影响概率分布

这一节不追求最快的 kernel，而是把 softmax 的数学形式、数值稳定性和端侧部署里的精度折中拆开看清楚。

## 最小任务

- [x] 实现 naive softmax
- [x] 实现 stable softmax（减 max）
- [x] 支持 batch / row-wise softmax
- [x] 对比 fp32 / fp16 的误差
- [x] 尝试 int8 输入反量化后再做 softmax
- [x] 可视化 logits、exp 值和输出概率

## 快速运行

在仓库根目录执行：

```powershell
python -m pip install -r code\02-softmax-from-scratch\requirements.txt
```

然后运行基础实验：

```powershell
python code\02-softmax-from-scratch\softmax_from_scratch.py
```

你会看到类似输出：

```text
shape: logits=(4, 5), case=manual
logits range: [...]
naive contains_nan=True, contains_inf=False
stable row sum range: [...]
int8 qparams: scale=..., zero_point=...
fp16_vs_fp32
  max_abs_error: ...
int8_dequant_vs_fp32
  max_abs_error: ...
```

`manual` case 里包含 `[1000, 1001, 1002, ...]` 这一行，所以 naive softmax 会暴露 overflow / NaN 问题；stable softmax 仍然可以正常输出概率。

也可以切换不同输入分布：

```powershell
python code\02-softmax-from-scratch\softmax_from_scratch.py --case small
python code\02-softmax-from-scratch\softmax_from_scratch.py --case large
python code\02-softmax-from-scratch\softmax_from_scratch.py --case close
python code\02-softmax-from-scratch\softmax_from_scratch.py --case wide
```

## 快速理解

softmax 把一组任意实数 `logits` 转成概率分布：

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

它有两个重要性质：

1. 每个输出都在 `[0, 1]` 之间
2. 同一行输出的总和等于 `1`

例如：

```text
logits = [1.0, 2.0, 3.0]
softmax(logits) ≈ [0.0900, 0.2447, 0.6652]
```

最大的 logit 会得到最大的概率，但 softmax 保留了相对差距，而不是简单地只选最大值。

## 为什么 naive softmax 不稳定

直接计算 `exp(x)` 时，如果 `x` 很大，浮点数会溢出：

```text
exp(1000) -> inf
```

这样后面的除法会变成：

```text
inf / inf -> NaN
```

所以 naive softmax 在小输入上看起来没问题，但遇到大 logits 时会崩。

## 稳定版本

softmax 有一个平移不变性：

```text
softmax(x) = softmax(x - c)
```

通常选择：

```text
c = max(x)
```

稳定实现变成：

```text
m = max(x)
z_i = x_i - m
y_i = exp(z_i) / sum(exp(z_j))
```

减去最大值后，最大的 `z_i` 会变成 `0`，因此最大的 `exp(z_i)` 是 `exp(0) = 1`，不会再向上溢出。

## 推荐实现路径

### 1. naive softmax

先写最直观的版本：

```text
exp_x = exp(x)
y = exp_x / sum(exp_x)
```

观察点：

- 输入较小时结果正常
- 输入包含 `1000`、`2000` 这类大数时容易出现 `inf` 或 `NaN`
- 它适合理解公式，但不适合真实推理

### 2. stable softmax

再实现减 max 的版本：

```text
shifted = x - max(x)
exp_x = exp(shifted)
y = exp_x / sum(exp_x)
```

观察点：

- 大 logits 不再溢出
- 输出和 naive softmax 在正常范围内应几乎一致
- 每一行概率和应接近 `1.0`

### 3. row-wise softmax

真实模型里 softmax 通常不是只处理一维向量，而是处理二维或更高维张量。

常见场景：

```text
shape = [batch, classes]
shape = [batch, sequence_length]
shape = [batch, heads, query_length, key_length]
```

本节建议先实现二维 row-wise softmax：

```text
for each row:
  row = row - max(row)
  output = exp(row) / sum(exp(row))
```

观察点：

- `max` 要按行计算
- `sum` 也要按行计算
- 不同 batch 之间不能混在一起归一化

## 精度实验

### fp32

`fp32` 是本节的参考结果。通常用它作为 baseline：

```text
y_ref = softmax_fp32(x)
```

### fp16

`fp16` 的指数、求和和除法精度更低，容易出现两类问题：

1. 很小的概率被下溢成 `0`
2. 多个接近的 logits 输出差异被压缩或放大

可以对比：

```text
max_abs_error = max(abs(y_fp16 - y_fp32))
mean_abs_error = mean(abs(y_fp16 - y_fp32))
```

### int8 近似

softmax 本身通常不直接在 int8 上完整计算。更常见的教学实验是：

```text
FP32 logits -> INT8 quantize -> dequantize -> stable softmax
```

链路变成：

```text
x_fp32 -> x_int8 -> x_dequant -> softmax(x_dequant)
```

观察点：

- logits 量化会改变类别之间的相对差距
- 如果 scale 太大，接近的 logits 会被量化到同一个整数
- softmax 会放大 logits 排序和间距上的细小变化

## 可视化 softmax

生成一张图，对比 logits、指数值、输出概率和误差：

```powershell
python code\02-softmax-from-scratch\visualize_softmax.py
```

默认输出：

```text
code\02-softmax-from-scratch\softmax_visualization.png
```

图里包含四块内容：

1. 原始 logits
2. `exp(logits)` 和 `exp(logits - max(logits))` 对比
3. naive / stable / fp16 / int8-dequant softmax 输出对比
4. naive / fp16 / int8-dequant 相对 stable fp32 的最大绝对误差

推荐观察几组输入：

```text
[1, 2, 3]
[10, 20, 30]
[1000, 1001, 1002]
[0.01, 0.02, 0.03]
[-10, 0, 10]
```

这些输入能分别暴露：

- 正常概率分布
- 指数放大
- naive overflow
- 小间距 logits 的精度敏感性
- 极端类别偏置

## 速度对比

运行 benchmark：

```powershell
python code\02-softmax-from-scratch\benchmark_softmax.py
```

调整规模：

```powershell
python code\02-softmax-from-scratch\benchmark_softmax.py --batch 128 --classes 512
```

你会看到几类实现的平均耗时：

- stable fp32 NumPy
- stable fp16 NumPy
- int8 dequant + stable fp32
- stable fp32 Python row loop

注意：这个 benchmark 是教学用，不代表真实硬件 softmax kernel 性能。真实端侧性能取决于 CPU / NPU 指令、查表策略、内存布局和 runtime kernel 实现。

## 核心公式

### Naive Softmax

```text
y_i = exp(x_i) / sum(exp(x_j))
```

### Stable Softmax

```text
m = max(x)
y_i = exp(x_i - m) / sum(exp(x_j - m))
```

### 误差指标

```text
abs_error = abs(y_test - y_ref)
max_abs_error = max(abs_error)
mean_abs_error = mean(abs_error)
```

也可以检查概率和：

```text
sum_error = abs(sum(y) - 1.0)
```

## 建议文件结构

- [`softmax_from_scratch.py`](./softmax_from_scratch.py)：naive / stable / row-wise softmax 主实验
- [`visualize_softmax.py`](./visualize_softmax.py)：softmax 数值稳定性和精度误差可视化
- [`benchmark_softmax.py`](./benchmark_softmax.py)：对比 Python loop、NumPy vectorized 和不同 dtype 的耗时
- [`requirements.txt`](./requirements.txt)：Python 依赖

建议主函数包括：

- `naive_softmax`：直接按公式计算
- `stable_softmax`：减 max 的稳定版本
- `softmax_rowwise`：二维按行 softmax
- `quantize_int8` / `dequantize_int8`：用于 int8 近似实验
- `compare_outputs`：计算 max / mean error 和 sum error

## 观察建议

1. softmax 的输出取决于 logits 的相对差距，而不是绝对大小。
2. 减去 `max` 不改变数学结果，但能显著提升数值稳定性。
3. logits 差距越大，softmax 越接近 one-hot。
4. logits 很接近时，低精度量化更容易改变输出排序或概率比例。
5. 在端侧推理里，softmax 往往位于分类头、attention 或采样前，误差会直接影响最终决策。

## 和端侧 AI 的关系

softmax 看起来只是一个简单归一化，但在部署时很关键：

- 分类模型里，它决定最终类别概率。
- Transformer attention 里，它决定每个 token 关注谁。
- 低精度推理里，它常常需要特殊 kernel 或近似查表。
- 在算力受限设备上，`exp` 和除法都比加法、乘法更昂贵。

所以这一节的重点不是“会调用库函数”，而是理解真实推理系统为什么要特别处理 softmax。
