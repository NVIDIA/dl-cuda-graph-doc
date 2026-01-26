<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Introduction

:::{note}

This section provides a quick, high-level introduction to the background of CUDA Graphs.
:::

## Background

Over the past decade, deep learning has revolutionized artificial intelligence, enabling breakthroughs in computer vision, natural language processing, and scientific computing. At the heart of this revolution lies GPU computing—GPUs have become the workhorses of modern AI, accelerating matrix operations and neural network training by orders of magnitude compared to CPUs. As models grow larger and more complex, with architectures like transformers containing billions of parameters, efficiently utilizing GPU resources has become paramount. Every microsecond of overhead matters when training models for weeks or serving millions of inference requests per day.

Beneath this GPU computing revolution lies CUDA—NVIDIA's parallel computing platform and programming model that serves as the foundation for modern GPU acceleration. CUDA is the soil from which the deep learning ecosystem has grown. While most practitioners write their models in Python using frameworks like PyTorch or TensorFlow, these high-level APIs are ultimately translated into CUDA operations running on the GPU. Every time you call `torch.matmul()` or `tf.keras.layers.Dense()`, CUDA kernels are being launched behind the scenes. Understanding CUDA and its performance characteristics is crucial for squeezing every bit of performance from your GPU, whether you're training the next breakthrough model or deploying inference services at scale.

## The Challenge: Launch Overhead

Over the past decade, GPU performance has grown exponentially—modern GPUs can deliver petaFLOPs of compute power, with each generation bringing significant improvements in throughput and efficiency. In contrast, CPU performance improvements have slowed considerably, constrained by power limits and the end of Dennard scaling.

In typical deep learning applications, the CPU and GPU work in a producer-consumer relationship: the CPU acts as the orchestrator, continuously launching CUDA operations, while the GPU driver queues these operations and the GPU hardware executes them. This asynchronous model allows the CPU to stay ahead, queuing work faster than the GPU consumes it, keeping the GPU fully utilized. However, this widening performance gap creates an interesting challenge: **the CPU, which orchestrates GPU work, can become the bottleneck**. When GPUs execute operations in microseconds, the time spent on the CPU preparing and launching those operations becomes increasingly significant. If kernel launch overhead approaches or exceeds kernel execution time, the GPU sits idle waiting for the CPU to queue the next operation—wasting valuable GPU resources that could cost thousands of dollars per device.

## Sources of Launch Overhead

To understand the overhead, let's look at what happens when you launch a CUDA kernel in a Deep Learning application. Every time you call a CUDA operation (e.g., `torch.matmul()`), multiple steps occur, each contributing to the cumulative overhead. This entire process happens for **every single kernel launch**, even when launching the same kernel repeatedly with the same configuration.

The kernel launch overhead comes from multiple sources:

**1. Language Transitions** (10-100 μs)

- Python ↔ C++ boundary crossing in frameworks like PyTorch/TensorFlow
- Object marshalling and reference counting
- GIL (Global Interpreter Lock) acquisition/release

**2. Runtime Processing** (5-20 μs)

- Operation dispatch logic, e.g. dispatch the GEMM operator to correct logic depending on operands' shape, type, etc.
- Memory management, e.g. workspace memory allocation/preparation
- Library transitions, e.g. calling cuDNN/cuBLAS from PyTorch, crossing library boundaries
- Parameter validation and type checking
- Choosing the best kernel from multiple implementations (e.g., different cuDNN algorithms)

**3. Driver Operations** (5-15 μs)

- Validates parameters and configurations to ensure correctness
- CUDA driver command buffer management
- Packages work and submits to the GPU via the driver's command queue

**4. Hardware Submission** (1-5 μs)

- PCIe communication overhead
- GPU command processor ingestion
- Hardware scheduling decisions

:::{note} Timing Note

The overhead time ranges mentioned above are measured on typical hardware and software configurations. Actual timings may vary depending on your specific hardware, driver version, framework, and workload characteristics.
:::

As you can see, launching a CUDA kernel is a complex multi-stage process involving coordination across multiple software layers—from Python to C++ to CUDA runtime to the driver—each adding its own overhead. This complexity becomes problematic in scenarios where the GPU execution is disproportionately fast compared to the CPU's ability to prepare and submit work. Common scenarios include:

- GPU compute is far faster than CPU preparation, especially when pairing older CPUs with modern GPUs (e.g., a 5-year-old CPU with an B200)
- Complex software stack with significant dispatch overhead, e.g., deep framework logic to determine the next kernel, algorithm selection, or memory management
- Executing many small kernels where individual kernel execution times are only a few microseconds

In these cases, the GPU can complete its current kernel and sit idle, waiting for the CPU to finish preparing the next kernel launch. This GPU idle time translates directly to underutilization—you're paying for expensive GPU compute resources that are spending a significant portion of their time doing nothing, simply because the CPU cannot keep up with feeding the GPU work fast enough.

So how do we solve this launch overhead problem? The answer lies in a clever technique: instead of launching each kernel individually every time, what if we could construct a graph of GPU operations—capturing their dependencies and execution order—and then replay the entire graph as a single unit? This is exactly what CUDA Graph enables—a way to define and execute a graph of GPU operations together, dramatically reducing the cumulative overhead and keeping your expensive GPU hardware fully utilized.

## What's Next?

Now that you understand the launch overhead challenge, continue learning about CUDA Graphs:

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 📖 CUDA Graph
:link: cuda-graph
:link-type: doc

Learn what CUDA Graphs are, how they work, and when to use them.
:::

:::{grid-item-card} 🚫 Constraints
:link: constraints
:link-type: doc

Understand the fundamental limitations imposed by CUDA runtime.
:::

:::{grid-item-card} 📊 Quantitative Benefits
:link: quantitative-benefits
:link-type: doc

How to measure and estimate performance gains.
:::

::::
