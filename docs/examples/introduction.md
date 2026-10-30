<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Examples

:::{note}

This section provides practical, runnable examples of using CUDA Graphs in real-world scenarios.
:::

## Overview

The following examples demonstrate CUDA Graph usage across different application domains. Each example is designed to be self-contained and adaptable to your specific use case.

## Available Examples

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🎤 RNN-T (RNN Transducer)
:link: rnnt
:link-type: doc

Speech recognition with dynamic shapes, greedy decoding, and bucketing strategies
:::

:::{grid-item-card} 🎨 Stable Diffusion v2
:link: stable-diffusion-v2
:link-type: doc

Diffusion models with UNet, full-iteration capture, and PyTorch Lightning.
:::

:::{grid-item-card} 🧠 GPT-3 175B
:link: gpt-3-175b
:link-type: doc

Per-layer graphing with FP8, pipeline parallelism, and complex interleaved microbatch execution.
:::

:::{grid-item-card} 🦙 Llama 2 70B LoRA
:link: llama2-70b-lora
:link-type: doc

Fine-tuning with LoRA adapters, FP8+FP4, and per-layer graphing.
:::

:::{grid-item-card} 🦙 Llama 3.1 405B
:link: llama-31-405b
:link-type: doc

Full-iteration graphing at massive scale with FP4 and Megatron-LM for training and inference.
:::

:::{grid-item-card} 🐳 DeepSeek-V3 671B: Paged Stashing
:link: deepseek-v3-paged-stash
:link-type: doc

Full forward/backward graphs for token-dropless MoEs with device-initiated grouped GEMM, compact activation storage, and overflow recovery.
:::

::::

## Comparison Table

The following table compares CUDA Graph implementations across different models and frameworks:

| Aspect | RNN-T | Stable Diffusion v2 | GPT-3 175B | Llama 2 70B LoRA | Llama 3.1 405B | DeepSeek-V3 671B |
|--------|-------|---------------------|-----------|------------------|----------------| ------------------ |
| **MLPerf Version** | v3.1 (2023/11) | v5.0 (2025/06) | v4.1 (2024/11) | v5.0 (2025/06) | v5.1 (2025/11) | v6.0 |
| **Software Stack** | Pure PyTorch + APEX + DALI | NeMo + PyTorch Lightning + APEX | NeMo + Megatron-LM + TE | Megatron-LM + TE | NeMo + Megatron-LM + PyTorch Lightning + TE | Megatron-Bridge + Megatron-Core + TE |
| **Model Size** | ~49M parameters | ~865M parameters (UNet) | ~175B parameters | ~70B base + LoRA adapters | ~405B parameters | ~671B parameters |
| **Capture Scope** | Partial (encoder + prediction networks) | Full iteration (forward + backward + optimizer) | Partial (per-layer) | Partial (per-layer) | Full iteration (forward + backward) | Full iteration (forward + backward); optimizer graphed separately |
| **Graph Count** | 8 graphs with bucketing (4 per network × 2 networks) | 1 graph (training) | per layer × per microbatch x 2 | per layer × per microbatch x 2 | 2 graphs (training + validation) | Training + validation; separate optimizer graph |
| **Capture Mechanism** | Custom `graph()` function | NeMo CUDAGraphCallback | TE `make_graphed_callables` | TE `CudaGraphManager` | Megatron-LM `FullCudaGraphWrapper` | `FullCudaGraphWrapper` + `PagedStashRunner` |
| **Static Shapes** | ❌ Bucketing (audio: 400-1600, text: 150-600) | ✅ Fixed (latents: 4×64×64, text: 77×1024) | ✅ Fixed (seq_len: 2048) | ✅ Fixed shapes | ✅ Fixed (seq_len: 8192) | Fixed-capacity buffers; dynamic valid-token counts |
| **Static Control Flow** | ❌ Greedy decoding with masks | ❌ Conditional optimizer step | ❌ Conditional FP8 weight transpose | ✅ No dynamic control flow | ✅ No dynamic control flow | Static captured schedule; eager overflow recovery outside the graph |
| **Distributed Training** | ✅ DDP (via DistributedFusedLAMB) | ✅ DDP | ✅ 3D parallelism (DP + TP + PP) | ✅ DP + TP + CP | ✅ DP + TP + PP + CP | ✅ DP + PP + EP (GB300 recipe) |
| **Mixed Precision** | FP16 | FP16 | BF16 + FP8 | BF16 + FP8 + FP4 | BF16 + FP4 | MXFP8 |
| **Key Challenges** | • Variable-length sequences<br>• Iterative greedy decoding<br>• Conditional branches | • PyTorch Lightning integration<br>• Conditional optimizer step<br>• Full-iteration graphing | • FP8 compatibility<br>• Pipeline schedule ordering<br>• Memory pool management | • FP8 + FP4 compatibility<br>• Per-layer graphing | • FP4 compatibility<br>• Full-iteration graphing | • Dynamic expert token counts<br>• Activation memory<br>• Overflow recovery |
| **Notable Techniques** | • Bucketing<br>• Loop unrolling (4× iterations)<br>• Async CPU-GPU pipeline | • Capturable optimizer (noop flag)<br>• Full iteration graphing with PyTorch Lightning | • Per-layer per-microbatch graphing<br>• GPU-controlled no-ops for FP8 caching<br>• Manual graphing for PP schedule | • Per-layer per-microbatch graphing<br>• Auto graphing for PP schedule | • Full iteration graphing with Megatron-LM | • Device-initiated grouped GEMM<br>• Sync-free HybridEP dispatch<br>• Paged stashing |
| **Performance Gain** | Significant | Up to 1.58× (58% faster) cumulative | 2.2% (256 GPUs) → 3.0% (11,616 GPUs) | Variable (depends on LoRA config) | ~15-25% at large scale (preliminary) | 1.68× training speedup; memory 221 → 209 GB (256 GB300 GPUs) |
| **Best For** | Models with:<br>• Variable sequence lengths<br>• Dynamic decoding | Models with:<br>• PyTorch Lightning <br>• Maximum performance needs | Models with:<br>• FP8 training<br>• Pipeline parallelism<br>• Per-layer graphing | Models with:<br>• Megatron-LM `TransformerLayer` <br>• per-layer graphing | Models with:<br>• Maximum performance needs<br>• Megatron-LM workloads | Token-dropless MoEs with dynamic routing and full forward/backward capture |

**Legend**:

- ✅ = Supported/Handled
- ❌ = Not present/Not supported

## General Patterns

The examples demonstrate key CUDA graph concepts:

1. **Capture Approaches**: Per-layer graphing (GPT-3, Llama 2 LoRA) and full-iteration graphing (Stable Diffusion, Llama 3.1, DeepSeek-V3)
2. **Framework Integration**: Pure PyTorch (RNN-T), PyTorch Lightning (Stable Diffusion), Transformer Engine and Megatron-LM (GPT-3, Llama), and Megatron-Bridge with Megatron-Core (DeepSeek-V3)
3. **Distributed Training**: DDP (RNN-T, Stable Diffusion), combinations of data, tensor, pipeline, and context parallelism (GPT-3, Llama), and expert parallelism with pipeline parallelism (DeepSeek-V3)
4. **Mixed Precision**: FP16 (RNN-T, Stable Diffusion), FP8 (GPT-3, Llama 2 LoRA), FP4 (Llama 2 LoRA, Llama 3.1), and MXFP8 (DeepSeek-V3)
5. **Dynamic Challenges**: Variable shapes and bucketing (RNN-T), conditional optimizer steps (Stable Diffusion), pipeline schedule ordering (GPT-3, Llama 3.1), and dropless expert routing with GPU-resident shape metadata (DeepSeek-V3)

## What You'll Learn

- **Per-layer vs. full-iteration graphing**: When to use each approach and their tradeoffs
- **Low-precision training with CUDA graphs**: Handling FP8/FP4 state, weight caching, and MXFP8 expert computation
- **Pipeline and expert parallelism**: Managing interleaved schedules, memory pools, and expert dispatch
- **Framework-specific patterns**: Integration with PyTorch Lightning, Transformer Engine, Megatron-LM, and Megatron-Bridge
- **Handling dynamic patterns**: Bucketing, conditional execution, and device-initiated grouped GEMM for dropless MoE
- **Activation memory management**: Using paged stashing to separate temporary compute buffers from saved activations
- **Performance optimization**: Measuring iteration time, memory use, and recovery overhead across models and scales

## What's Next?

Select an example that matches your use case from the cards above, or explore all examples:

| Example | Best For |
|---------|----------|
| 🎤 **[RNN-T](rnnt.md)** | Variable sequence lengths, dynamic control flow |
| 🎨 **[Stable Diffusion v2](stable-diffusion-v2.md)** | PyTorch Lightning, maximum performance with full-iteration graphing |
| 🧠 **[GPT-3 175B](gpt-3-175b.md)** | Per-layer graphing with FP8, pipeline parallelism,  |
| 🦙 **[Llama 2 70B LoRA](llama2-70b-lora.md)** | Fine-tuning, LoRA adapters |
| 🦙 **[Llama 3.1 405B](llama-31-405b.md)** | Maximum scale, Megatron-LM |
| 🐳 **[DeepSeek-V3 671B: Paged Stashing](deepseek-v3-paged-stash.md)** | Dropless MoE routing, full forward/backward graphs, activation memory |

After Reviewing Examples

- 📖 **[Best Practices](../torch-cuda-graph/best-practices.md)**: General guidance for all use cases
- 🛠️ **[Troubleshooting](../troubleshooting/introduction.md)**: Debug common issues and failures
- 🐍 **[PyTorch Integration](../torch-cuda-graph/torch-integration.md)**: Deep dive into PyTorch's CUDA Graph APIs
