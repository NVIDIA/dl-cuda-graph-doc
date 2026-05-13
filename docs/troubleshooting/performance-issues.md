<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Performance Issues

:::{note}

This section covers scenarios where CUDA graphs don't provide the expected speedup or cause performance regression.
:::

## Introduction

CUDA graphs reduce CPU overhead by capturing and replaying GPU operations as a single unit. However, if not used correctly, you may see no performance benefit or even a slowdown. This section helps you identify and resolve common performance issues when using CUDA graphs.

The key insight is that CUDA graphs primarily benefit **CPU-bound** workloads. If your workload is already GPU-bound, or if the graphed portion doesn't cover the actual bottleneck, you may see minimal improvement or even regression.

## Debugging Strategy: Compare Nsight Systems Profiles

The most effective way to diagnose CUDA graph performance issues is to compare Nsight Systems profiles with and without CUDA graphs. Run `nsys profile -o baseline python train.py` without graphs, then run with graphs enabled and compare the timelines. Look for where the slowdown occurs: extra memory copy operations before graph replay, long synchronization gaps, multiple streams with visible serialization, or GPU idle time despite graph replay.

For profiling CUDA graphs specifically, use the `--cuda-graph-trace` option. The default `graph` granularity traces graphs as a whole with minimal overhead, while `node` granularity collects individual node activities but may cause significant runtime overhead.

## Performance Issues

Performance issues with CUDA graphs typically stem from these main causes:

1. **[Not CPU-bound or graphing the wrong range](#not-cpu-bound-or-graphing-the-wrong-range)**: CUDA graphs only help CPU-bound workloads; if you're already GPU-bound or graph the wrong portion, you won't see improvement
2. **[Too many small CUDA graphs](#too-many-small-cuda-graphs)**: Graph breaks fragment your workload into many small graphs, each with launch overhead
3. **[Input tensor copy overhead](#input-tensor-copy-overhead)**: Copying data into static input tensors before each replay adds overhead that can offset graph benefits
4. **[Deferred gradient hooks with `make_graphed_callables`](#deferred-gradient-hooks-with-make_graphed_callables)**: Gradient hooks execute after graph completion, eliminating computation-communication overlap (e.g., DDP allreduce)
5. **[Too many CUDA streams (channel serialization)](#too-many-cuda-streams-channel-serialization)**: Excessive streams exceed hardware channel limits, causing kernel serialization

The following subsections cover these performance issues and their solutions.

### Not CPU-Bound or Graphing the Wrong Range

**Symptom**: Minimal or no speedup despite successful graph capture.

**Why it happens**: CUDA graphs eliminate CPU overhead by replaying captured operations without re-issuing kernel launches. This only helps when CPU kernel launch overhead is the bottleneck.

- **Not CPU-bound**: If your workload is already GPU-bound, the GPU is fully utilized and CPU runs well ahead of GPU execution. In Nsight Systems, you'll see high GPU utilization with minimal GPU idle time—there's simply no CPU overhead to eliminate.
- **Graphing the wrong range**: If you graph a portion that isn't the bottleneck, GPU utilization won't improve and the timeline will still show idle regions where the actual bottleneck (outside the graph) occurs.

**How to diagnose**:

Compare Nsight Systems profiles with and without CUDA graphs:

- **Not CPU-bound**: High GPU utilization (>95%) in both cases, CPU timeline shows kernel launches completing far ahead of GPU execution, minimal GPU idle time
- **Wrong capture range**: GPU utilization doesn't improve with graphs, timeline still shows significant GPU idle regions outside the graphed portion

**How to fix**:

1. **Check if CPU-bound**: Use `nvidia-smi dmon -s u` or `nsys profile --gpu-metrics-devices=all` to check GPU utilization—if already >95%, CUDA graphs won't help. In Nsight Systems, also check if CUDA API calls on the CPU timeline are far ahead of kernel execution on the GPU timeline; if so, the workload is GPU-bound and graphs won't improve performance
2. **Identify the actual bottleneck**: Use Nsight Systems to find where kernel launch gaps or GPU idle regions occur
3. **Graph the CPU-bound portion**: Focus on regions with many small kernels and visible launch overhead

### Too Many Small CUDA Graphs

**Symptom**: Multiple small graphs instead of one large graph, with visible gaps between graph launches.

**Why it happens**: Capturing too many small graphs provides limited runtime overhead elimination, resulting in limited performance improvement. Each graph launch still has overhead, and the gaps between graphs can negate the benefits. This commonly occurs when using `torch.compile` with `mode="reduce-overhead"`, where graph breaks fragment your workload into many small graphs.

:::{note} Graph Break

"Graph break" here refers to TorchDynamo's graph break—when `torch.compile` encounters code it cannot trace (e.g., data-dependent control flow, `.item()` calls), it splits the computation into separate compiled regions. Additionally, each TorchDynamo graph can be further partitioned by Inductor into multiple CUDA graphs when it contains operations not compatible with CUDA graphs (e.g., CPU ops).
:::

**How to diagnose**:

```bash
# Check for graph breaks with torch.compile
TORCH_LOGS=graph_breaks python train.py
```

In Nsight Systems, look for multiple `cudaGraphLaunch` calls per iteration instead of one.

**Example**: The following code has two graph breaks—`.item()` forces GPU synchronization and returns to Python, while the `if x.sum() > 0` condition requires evaluating tensor data at runtime. This fragments what could be a single graph into multiple smaller graphs:

```python
import torch
import torch.nn as nn

layer1 = nn.Linear(64, 64).cuda()
layer2 = nn.Linear(64, 64).cuda()
layer3 = nn.Linear(64, 64).cuda()

# BAD: Many small graphs due to graph breaks
@torch.compile(mode="reduce-overhead")
def forward(x):
    x = layer1(x)
    print(x.sum().item())  # Graph break! .item() syncs GPU and returns to Python
    x = layer2(x)
    if x.sum() > 0:  # Graph break! Data-dependent control flow
        x = layer3(x)
    return x

x = torch.randn(1, 64, device="cuda")
for _ in range(4):
    forward(x)
```

**How to fix**:

1. **Eliminate graph breaks** (if using `torch.compile`): Try to eliminate graph breaks and unsupported operations—for example, remove `.item()` calls, data-dependent control flow, and CPU operations from graphed regions. Use `TORCH_LOGS=graph_breaks python train.py` to identify graph break locations. However, not all graph breaks can be easily eliminated; some may require significant code restructuring or may be unavoidable.

2. **Use manual graphing and extend graph range**: With the low-level `torch.cuda.graph()` API, you have full control over what gets captured. Try to include as much of the computation as possible in a single graph.

### Input Tensor Copy Overhead

**Symptom**: Graph replay is fast, but copying input tensors before each graph launch takes relatively longer time than graph kernel execution, reducing overall performance gains.

**Why it happens**: CUDA graphs require static memory addresses, so you must copy new data into static input tensors before each replay. These `copy_()` or `fill_()` operations add overhead that can offset the graph's benefits—especially for large inputs or when copying random generator states.

**How to diagnose**: In Nsight Systems, look for long `copy_` or `fill_` kernels immediately before `cudaGraphLaunch`.

**Example**: When graphing a simple operation like `sum()`, the input copy overhead can exceed the graphed computation itself, negating any benefit:

```python
import torch

x = torch.randn(1024, 1024, device='cuda')
static_input = torch.empty_like(x)

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = static_input.sum()

for _ in range(100):
    static_input.copy_(x)  # Copy may take longer than sum()!
    g.replay()
```

**How to fix**:

1. **Write directly to static tensors** when possible—use `out=` parameter to avoid separate copy:

    ```python
    torch.add(a, b, out=static_input)
    torch.randn(size, out=static_input)  # For random tensors
    ```

2. **Increase graphed operations**: If input copy overhead dominates, the graphed workload is too small. Include more computation in the graph to amortize the copy cost.

### Deferred Gradient Hooks with `make_graphed_callables`

**Symptom**: Training with `torch.cuda.make_graphed_callables()` is slower than eager mode, especially with DDP.

**Why it happens**: `make_graphed_callables()` uses [`torch.autograd.grad()`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/graphs.py#L483-L490) internally to capture the backward graph:

```python
# Inside make_graphed_callables (torch/cuda/graphs.py)
with torch.cuda.graph(bwd_graph, pool=mempool):
    grad_inputs = torch.autograd.grad(
        outputs=outputs_grad,
        inputs=...,
        grad_outputs=...,
        only_inputs=True,  # Key difference from loss.backward()!
        allow_unused=allow_unused_input,
    )
```

Unlike `loss.backward()` in eager mode which [accumulates gradients into `.grad` attributes](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/autograd/__init__.py#L262-L263), `torch.autograd.grad()` [returns gradients directly without accumulation](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/autograd/__init__.py#L397-L398).

Here's what happens when you call `loss.backward()` on a graphed model:

1. **Autograd engine calls `Graphed.backward()`**—this is the custom [`torch.autograd.Function`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/graphs.py#L524-L551) created by `make_graphed_callables`
2. **Inside `backward()`, the CUDA graph replays**—this executes the captured `torch.autograd.grad()` call, computing all gradients without accumulation
3. **`backward()` returns gradients to the autograd engine**—the computed gradients are returned as a tuple
4. **Engine propagates gradients to `AccumulateGrad` nodes**—the engine's [`evaluate_function`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/engine.cpp#L1164-L1224) sends the returned gradients to the next edges, which are `AccumulateGrad` nodes for leaf parameters
5. **`AccumulateGrad` nodes execute and accumulate gradients**—these run *after* the graph replay completes, writing gradients to `.grad` attributes and triggering any registered hooks

The result: all gradient accumulation `add` kernels are serialized after the backward graph, eliminating any overlap between gradient computation and accumulation. With DDP, the impact is more severe because NCCL allreduce hooks are also deferred, completely eliminating computation-communication overlap.

**Example**: This script demonstrates the issue with DDP on a single GPU:

```python
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def run_iterations(model, x, mode, num_iters=4, warmup_iters=11):
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(warmup_iters):
            model.zero_grad(set_to_none=False)
            loss = model(x).sum()
            loss.backward()

    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    with torch.cuda.stream(torch.cuda.Stream()):
        for i in range(num_iters):
            with torch.cuda.nvtx.range(f"{mode}_iter_{i}"):
                model.zero_grad(set_to_none=False)
                loss = model(x).sum()
                loss.backward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

def main():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

    x = torch.randn(1024, 16384, device='cuda')

    model = torch.nn.Sequential(
        *[torch.nn.Linear(16384, 16384, bias=False) for _ in range(4)]
    ).cuda()
    with torch.cuda.stream(torch.cuda.Stream()):
        model_eager = DDP(model)
    run_iterations(model_eager, x, mode="eager")

    model_graphed = torch.cuda.make_graphed_callables(model_eager, (x,))
    run_iterations(model_graphed, x, mode="graphed")

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
```

Run with: `nsys profile -c cudaProfilerApi --capture-range-end=repeat:2 --cuda-graph-trace=node -o defer-hooks python 045.defer-hooks.py`

**How to diagnose**: In Nsight Systems, compare backward pass timelines:

- **Eager mode**: Backward compute kernels interleave with gradient accumulation `add` kernels throughout the backward pass
- **With `make_graphed_callables`**: All backward compute kernels run first (inside the graph), then all `add` kernels run sequentially after the graph completes

:::{warning} DDP Impact

With DDP, the impact is more severe because NCCL allreduce hooks are also deferred, completely eliminating computation-communication overlap.
:::

**How to fix**:

Use the low-level `torch.cuda.graph()` API to capture the entire backward pass including accumulation and hooks:

```python
# This captures EVERYTHING: compute + accumulation + DDP hooks
with torch.cuda.graph(g):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()  # Full backward with accumulation!
```

Note that `torch.compile` with `mode="reduce-overhead"` uses [compiled autograd](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/_dynamo/compiled_autograd.py#L790-L802), which traces `AccumulateGrad` and hooks into the FX graph. This is different from `make_graphed_callables`—compiled autograd can capture gradient accumulation and hooks within the graph, potentially avoiding this issue.

### Too Many CUDA Streams (Channel Serialization)

**Symptom**: Parallel kernels on multiple streams become serialized when using CUDA graphs, even though they run concurrently in eager mode.

**Why it happens**: CUDA GPUs have a limited number of hardware channels (called "device connections") for concurrent stream execution. When two streams are mapped to the same device channel, kernels on those streams serialize instead of running in parallel. The default `CUDA_DEVICE_MAX_CONNECTIONS` value is `8`.

This issue is more pronounced with CUDA graphs because:

1. **PyTorch stream pool**: PyTorch maintains a [pool of 32 streams per priority level](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDAStream.cpp#L20-L21). When you call `torch.cuda.Stream()`, you get a stream from this pool in round-robin fashion. If you request more than 32 streams, you start reusing the same underlying CUDA streams.

2. **CUDA graph stream expansion**: When capturing a CUDA graph with multiple streams, the CUDA driver may create additional internal streams to fully exploit concurrency during replay. This can exceed the channel limit even if your code uses fewer streams.

**How to diagnose**: Profile with `nsys profile --cuda-graph-trace=node` to see individual kernels within CUDA graphs. Compare eager mode vs graph mode—look for kernels that run concurrently in eager mode but become serialized in the CUDA graph replay.

**How to fix**:

1. **Limit stream count in user code**: Reduce the number of streams to stay well within the channel limit.

2. **Increase max connections**: Set `CUDA_DEVICE_MAX_CONNECTIONS` to a larger value, such as `16` or `32`.

---

## Quick Reference

**Not CPU-Bound or Graphing the Wrong Range**:

- [ ] Check GPU utilization—if >95% without graphs, you're already GPU-bound
- [ ] Graph the region with kernel launch gaps, not GPU-bound regions

**Too Many Small CUDA Graphs**:

- [ ] Eliminate graph breaks (`.item()`, data-dependent control flow)
- [ ] Use low-level `torch.cuda.graph()` API for full control

**Input Tensor Copy Overhead**:

- [ ] Write directly to static tensors using `out=` parameter
- [ ] Increase graphed operations to amortize copy cost

**Deferred Gradient Hooks with `make_graphed_callables`**:

- [ ] Use low-level `torch.cuda.graph()` API to capture full backward pass
- [ ] Consider `torch.compile` with compiled autograd

**Too Many CUDA Streams (Channel Serialization)**:

- [ ] Limit stream count to stay within 32-channel hardware limit
- [ ] On Blackwell+, set `CUDA_DEVICE_MAX_CONNECTIONS=128`

**Debugging**:

- [ ] Profile with `nsys profile --cuda-graph-trace=node` to see individual kernels
- [ ] Compare Nsight Systems profiles with and without CUDA graphs

---

## What's Next?

- **[Memory Issues](memory-issues.md)**: OOM troubleshooting
- **[Numerical Errors](numerical-errors.md)**: Wrong results or precision issues
- **[Process Hang](process-hang.md)**: Debugging hanging processes
- **[Debugging Strategies](debugging-strategies.md)**: Systematic debugging
