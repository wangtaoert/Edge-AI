# Quantization Deep Dive

## 目标
在尽量不损失精度的前提下，降低模型计算与存储成本。

## 路线
- PTQ：后训练量化，快速落地
- QAT：量化感知训练，精度更稳定

## 重点关注
- 对称 / 非对称量化
- per-tensor / per-channel
- 激活校准策略（minmax / percentile / KL）

## 实验建议
- 从 CNN（如 MNIST）开始，先比较 FP32 vs INT8。
