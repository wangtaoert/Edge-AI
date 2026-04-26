# Edge AI Overview

## 1. 什么是 Edge AI
Edge AI 指在靠近数据源的终端设备上完成模型推理（甚至训练）的 AI 形态。

## 2. 为什么需要 Edge AI
- 低时延：避免云端往返
- 隐私保护：数据尽量不离端
- 带宽节省：减少原始数据上传
- 离线可用：网络不稳定场景更可靠

## 3. 系统栈速览
1. 模型层：分类、检测、分割、LLM 等
2. 编译层：图优化、算子融合、量化
3. 运行时层：调度、内存管理、线程并行
4. 硬件层：CPU/GPU/NPU/DSP

## 4. 关键指标
- Latency（端到端延迟）
- Throughput（吞吐）
- Accuracy（精度）
- Power（功耗）
- Memory Footprint（内存占用）
