<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# PyTorch CUDA Graphs

:::{note}

This section provides comprehensive guidance for using CUDA Graphs in PyTorch, covering integration, best practices, and common challenges.
:::

This chapter focuses on **using CUDA Graphs in PyTorch**. While previous chapters covered CUDA graph fundamentals, this section addresses the practical challenges of integrating graphs into real-world PyTorch code: eliminating synchronization, handling dynamic patterns, and navigating PyTorch-specific constraints.

## What This Chapter Covers

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🐍 PyTorch Integration
:link: torch-integration
:link-type: doc

PyTorch's CUDA graph APIs (`torch.cuda.CUDAGraph`, `torch.cuda.graph()`, `torch.cuda.make_graphed_callables()`) and basic usage patterns.
:::

:::{grid-item-card} ⚡ Transformer Engine & Megatron-LM
:link: te-megatron-cuda-graphs
:link-type: doc

Megatron-LM's built-in CUDA graph support (`CudaGraphManager` and `FullCudaGraphWrapper`) for distributed training.
:::

:::{grid-item-card} ✅ Best Practices
:link: best-practices
:link-type: doc

Systematic approach to adopting CUDA graphs (quantifying benefits, choosing scope, verifying correctness, and writing compatible code).
:::

:::{grid-item-card} 🔇 Writing Sync-Free Code
:link: sync-free-code
:link-type: doc

Eliminating CPU-GPU synchronization—a prerequisite for CUDA graph capture.
:::

:::{grid-item-card} 🔄 Handling Dynamic Patterns
:link: handling-dynamic-patterns
:link-type: doc

Adapting dynamic control flow, tensors, scalars, and shapes to static graph execution.
:::

:::{grid-item-card} ✔️ Quick Checklist
:link: quick-checklist
:link-type: doc

Pre-capture checklist to verify your code is CUDA Graph compatible.
:::

::::

## Quick Navigation

| 🎯 Your Situation | 📍 Start Here |
|------------------|---------------|
| New to CUDA graphs in PyTorch | [PyTorch Integration](torch-integration.md) |
| Ready to capture | [Quick Checklist](quick-checklist.md) |
| Capture fails with sync errors | [Writing Sync-Free Code](sync-free-code.md) |
| Wrong results or model divergence | [Handling Dynamic Patterns](handling-dynamic-patterns.md) |
| Need adoption strategy guidance | [Best Practices](best-practices.md) |
| Debugging capture/runtime failures | [Troubleshooting](../troubleshooting/introduction.md) |

## What's Next?

Begin with **[PyTorch Integration](torch-integration.md)** to learn PyTorch's CUDA graph APIs and integration patterns.
