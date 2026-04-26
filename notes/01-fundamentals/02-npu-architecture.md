# NPU Architecture

## 1. NPU 的核心目标
在可控功耗下提供更高的矩阵与张量运算效率。

## 2. 常见组成
- 计算阵列（MAC 阵列 / Tensor Core 类似模块）
- 片上 SRAM（降低访存开销）
- DMA / NoC（数据搬运）
- 指令与调度单元（控制执行流）

## 3. 关键瓶颈
- 内存带宽墙
- 数据重排和 layout 转换成本
- 算子覆盖度不足导致回退 CPU

## 4. 学习建议
- 先理解 dataflow（weight stationary / output stationary）
- 再看具体厂商 SDK 的编译与 profiling 工具
