<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Troubleshooting CUDA Graphs

:::{note}

This section provides comprehensive guidance for debugging and resolving issues when working with CUDA Graphs in PyTorch.
:::

## Overview

PyTorch with CUDA Graphs introduces a fundamentally different execution model—operations are captured once and replayed as a fixed unit. Issues that would be obvious in eager mode can manifest as silent failures, memory errors, or unexpected behavior in graphed code. This chapter provides systematic approaches to diagnose and fix these issues efficiently.

## What This Chapter Covers

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} 🛠️ Debugging Strategies
:link: debugging-strategies
:link-type: doc

Systematic approaches to debug CUDA graph issues (detection techniques, isolation strategies, debugging workflows).
:::

:::{grid-item-card} 🚫 Capture Failures
:link: capture-failures
:link-type: doc

Diagnosing and fixing errors during graph capture (constraint violations, common error messages, workarounds).
:::

:::{grid-item-card} 🔢 Numerical Errors
:link: numerical-errors
:link-type: doc

Identifying wrong results, NaN/Inf issues, and numerical precision problems (tensor reassignment, uninitialized memory, gradient issues, RNG state).
:::

:::{grid-item-card} 💾 Memory Issues
:link: memory-issues
:link-type: doc

Debugging memory-related problems (OOM errors during capture/replay, memory leaks, pool management).
:::

:::{grid-item-card} ⏳ Process Hang
:link: process-hang
:link-type: doc

Resolving hanging or stuck processes (deadlocks, NCCL conflicts, infinite loops, distributed training hangs).
:::

:::{grid-item-card} 📉 Performance Issues
:link: performance-issues
:link-type: doc

Diagnosing cases where graphs don't deliver expected speedup (bottleneck identification, graph overhead, optimization opportunities).
:::

::::

## Failure Modes

| Failure Mode | Symptoms | When It Occurs | Debugging Difficulty |
|--------------|----------|----------------|---------------------|
| 🛠️ **[Debugging Strategies](debugging-strategies.md)** | Any issue | All phases | N/A (methodology) |
| 🚫 **[Capture Failures](capture-failures.md)** | RuntimeError during capture | Capture phase | ⭐ Easy (immediate error) |
| 🔢 **[Numerical Errors](numerical-errors.md)** | Wrong results, NaN/Inf | Replay phase | ⭐⭐⭐ Hard (silent failures) |
| 💾 **[Memory Issues](memory-issues.md)** | OOM, allocation errors | Capture or replay | ⭐⭐ Medium (clear errors) |
| ⏳ **[Process Hang](process-hang.md)** | Freezes, no progress | Any phase | ⭐⭐⭐⭐ Very Hard (no error) |
| 📉 **[Performance Issues](performance-issues.md)** | Slower than expected | Replay phase | ⭐⭐ Medium (requires profiling) |

## What's Next?

Start with **[Debugging Strategies](debugging-strategies.md)** for systematic approaches, or jump directly to a specific failure mode from the table above.

:::{tip} After Troubleshooting

- ✅ **[Quick Checklist](../torch-cuda-graph/quick-checklist.md)**: Verify your code meets all requirements before capture
- 📖 **[Best Practices](../torch-cuda-graph/best-practices.md)**: Proactive strategies to avoid issues
- 💡 **[Examples](../examples/introduction.md)**: Real-world CUDA Graph implementations
:::
