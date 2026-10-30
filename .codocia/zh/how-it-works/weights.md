---
title: 零拷贝加载权重
source: #c787796eed8b
---

`ModelFiles::weight_source` 通过 Candle 的 `VarBuilder` 对 Safetensors 做内存映射。GGUF 走项目自己的解析器和基于 mmap 的 `GgufWeightSource`，块量化矩阵在模型构建期间始终保持压缩，并以 `QMatMul` 运行。启用 `metal` feature 时，`file_backed_quant` 把每个 GGUF 分片的只读映射包装成一个共享的 Metal buffer，并按文件偏移绑定每个张量。而 Candle 自带的加载器会把每个权重复制到匿名 buffer 中，使整个模型都变成 dirty 的统一内存。

正因如此，模型加载期间不能修改模型文件；加载器因此是 `unsafe` 的。加载完成后，会请求 macOS 归还构建 tokenizer 和元数据时释放的内存页（`malloc_zone_pressure_relief`）。
