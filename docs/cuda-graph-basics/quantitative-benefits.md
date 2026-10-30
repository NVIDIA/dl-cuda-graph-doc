<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Quantitative Benefits

:::{note}

This section provides practical methods to quantify CUDA Graph performance benefits in your application, including measuring GPU utilization, kernel launch overhead reduction, and timing consistency improvements.
:::

## Common CUDA Graph Benefits

CUDA Graphs provide several quantifiable performance improvements:

### 1. Higher GPU Utilization

By eliminating CPU-side launch overhead, CUDA Graphs allow the GPU to spend more time executing kernels and less time idle waiting for the CPU to queue work. This is especially noticeable when the CPU becomes a bottleneck in feeding work to the GPU, as discussed in the [introduction](introduction.md). With graphs, the CPU submits an entire sequence of operations at once, keeping the GPU pipeline full.

### 2. Reduced Kernel Launch Overhead

CUDA Graphs significantly reduce the driver-side and hardware-side overhead associated with kernel launches. Instead of processing each kernel launch individually through the CUDA driver and GPU command processor, the entire graph is submitted as a single unit. This eliminates the per-kernel overhead of command buffer management, parameter setup, and hardware scheduling decisions.

### 3. Less Jitter (Improved Timing Consistency)

CUDA Graphs provide more deterministic execution with less run-to-run variation. This manifests in two ways:

- **Run-to-run jitter**: Without graphs, variations in CPU scheduling, driver state, and system load can cause different execution times for the same workload. Graphs eliminate most of this variability by bypassing the dynamic dispatch path.
- **Rank-to-rank jitter**: In multi-GPU distributed training, different GPUs may experience different CPU overhead, causing stragglers that slow down collective operations. Graphs help synchronize execution across ranks by making each GPU's timing more predictable.

## How to Quantify the Benefits

### Measuring GPU Utilization

Use `nvidia-smi dmon` to monitor GPU utilization in real-time:

```bash
nvidia-smi dmon -s u
```

This displays the **SM (streaming multiprocessor) utilization percentage**, which indicates how busy the GPU's compute units are. Higher and more stable SM utilization means better GPU efficiency. Run your workload and observe:

- **Without CUDA Graphs**: You may see utilization dropping periodically or remaining consistently low (e.g., less than 80%, depending on workload and hardware) as the GPU frequently waits for the CPU to launch the next kernel. In severe cases with very small kernels, utilization can stay below 50% because the GPU spends more time idle than computing.
- **With CUDA Graphs**: Utilization should be more consistent and higher (e.g., 90-100%) since the GPU has a continuous stream of pre-queued work without waiting for individual kernel launches.

**Example output**:
```
# Without graphs
gpu   sm   mem   enc   dec
  0   78    45     0     0
  0   82    48     0     0
  0   75    43     0     0  ← Drops due to CPU bottleneck
  0   80    46     0     0

# With graphs
gpu   sm   mem   enc   dec
  0   95    52     0     0
  0   96    53     0     0
  0   95    52     0     0  ← More consistent, higher
  0   96    53     0     0
```

The improvement in SM utilization directly translates to better hardware efficiency and faster execution.

:::{tip} Other Monitoring Tools

Beyond `nvidia-smi dmon`, you can also measure GPU utilization using:

- **DCGM** (Data Center GPU Manager): `dcgmi dmon` for enterprise-grade monitoring with more metrics
- **NVML** (NVIDIA Management Library): Programmatic access via `pynvml` Python package for custom monitoring
- **Nsight Systems**: Timeline view shows GPU active periods visually. Use `nsys profile --gpu-metrics-devices=all` to include GPU utilization metrics in the timeline (e.g., `SMs Active` shows GPU saturation over time). Note this adds profiling overhead.

All these tools report similar utilization metrics—choose based on your environment and needs.
:::

### Estimating Kernel Launch Overhead Reduction

Use NVIDIA Nsight Systems to profile your application and count kernel launches:

```bash
nsys profile -o baseline python your_script.py
```

Then analyze the profile to see CUDA API call statistics:

```bash
nsys stats --report cuda_api_sum baseline.nsys-rep
```

This report shows a summary of CUDA API calls, including the count of `cudaLaunchKernel` invocations.

**Calculation**:

1. **Count total kernels** in one iteration from the Nsight Systems report. Look for the number of `cudaLaunchKernel` API calls (or `cudaLaunchKernelExC` for newer CUDA versions) in the CUDA API summary. This tells you how many individual kernel launches occur, each of which incurs overhead
2. **Estimate overhead savings**: Each kernel launch saved contributes to overall speedup. The actual savings depend on your system, but you can assume ~1-5 μs per kernel for driver and hardware overhead (command buffer processing, GPU submission).
3. **Calculate total savings**: Number of kernels × overhead per kernel

:::{note} Overhead Estimates

The actual overhead saved per kernel varies based on kernel complexity, framework overhead, and hardware. Profile both graphed and non-graphed executions to measure actual savings in your specific case. The 1-5 μs range is a reasonable starting estimate based on typical CUDA applications.
:::

### Measuring Jitter Reduction

Jitter refers to variation in execution time, which CUDA Graphs reduce by providing more deterministic execution.

**For run-to-run jitter (single GPU)**:

Run your workload multiple times (e.g., 100 iterations) and measure the execution time for each run. Calculate the mean and standard deviation of these times. The **coefficient of variation** (CV, = standard deviation / mean) is a good metric—lower values indicate more consistent performance. With CUDA Graphs, you should see:

- **Without graphs**: CV typically 3-10% due to CPU scheduling variations and driver state differences
- **With graphs**: CV typically <1-2% due to deterministic graph execution

**For rank-to-rank jitter (multi-GPU)**:

Measuring rank-to-rank jitter accurately is more involved than simply comparing iteration times. Factors like data parallel collectives, model parallel communication, and load imbalance all contribute to timing variations, and the impact varies by application and configuration.

A practical approach: measure iteration time on each GPU/rank and identify the **fastest rank without CUDA Graphs**—this represents relatively better performance with less CPU bottleneck compared to other ranks. Assuming CUDA Graphs can bring all ranks to this fastest rank's speed is a reasonable estimate. In practice, the actual performance gain may be higher since CUDA Graphs also reduce the fastest rank's CPU overhead.

CUDA Graphs reduce rank-to-rank variation by making each rank's execution time more deterministic, since all ranks execute their graphs with minimal CPU involvement.

**Expected improvements**:

- **Without graphs**: 5-20% variation between ranks
- **With graphs**: <2% variation between ranks

## Quick Decision Rules

After profiling and measuring the benefits in your application, you can use these rules of thumb to quickly assess whether CUDA Graphs are worthwhile. These guidelines are based on common patterns across various workloads and can help you make an informed decision without extensive analysis.

**Strong candidates for CUDA Graphs**:

- ✅ Training loops (thousands of iterations)
- ✅ Inference serving (millions of requests)
- ✅ Many small kernels (>50 per iteration)
- ✅ CPU-bound workloads (low GPU utilization)
- ✅ Multi-GPU with rank stragglers

**Poor candidates**:

- ❌ One-shot operations (<20 iterations)
- ❌ Compute-bound (>95% GPU utilization already)
- ❌ Dynamic shapes or control flow (require dedicated skill to make them work)

## Summary: Expected Improvements

The actual benefits vary significantly based on your workload characteristics. Here are typical speedup ranges observed in practice:

| Workload Type | Typical Speedup | Notes |
|--------------|-----------------|-------|
| **Many small kernels** | 1.5-3× | Launch overhead dominates |
| **Medium workload** | 1.3-2× | Moderate overhead contribution |
| **Large kernels** | 1.1-1.4× | Compute dominates, overhead small |
| **Already optimized** (>90% GPU util) | 1.05-1.15× | Limited room for improvement |

**Key factors affecting improvement**:

- **Kernel count**: More kernels = more launch overhead to eliminate = bigger gains
- **Kernel size**: Smaller, faster kernels benefit more (launch overhead is larger fraction of total time)
- **CPU performance**: Slower/older CPUs see bigger improvements as they struggle more to feed the GPU
- **Framework overhead**: Python-based frameworks (PyTorch/TensorFlow) have higher per-launch overhead than C++

The improvements are most dramatic when your workload is **CPU-limited** (low GPU utilization) with **many kernel launches**. If your GPU utilization is already >95%, CUDA Graphs will provide only modest speedups, though jitter reduction can still be valuable for consistency.

## What's Next?

Now that you know how to quantify the benefits:

- **[PyTorch CUDA Graph Integration](../torch-cuda-graph/torch-integration.md)**: Learn how PyTorch integrates with CUDA Graphs and how to use them in your applications
- **[Examples](../examples/introduction.md)**: Explore graphing approaches and performance tradeoffs in RNN-T, Stable Diffusion, Llama 3.1 405B, and DeepSeek-V3 671B
