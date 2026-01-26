<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Numerical Errors

:::{note}

This section covers **numerical errors**—issues where CUDA graph capture succeeds but produces incorrect results (a.k.a. silent failures), NaN/Inf values, or numerical inconsistencies.
:::

## Understanding Numerical Errors

Numerical errors are the **most challenging** aspect of applying CUDA graphs with PyTorch. Unlike capture failures ([Capture Failures](capture-failures.md)) which immediately tell you something is wrong, numerical errors allow your code to run seemingly fine while producing subtly or drastically incorrect results. Your training loss might diverge, gradients might contain NaN, or model outputs might differ from eager execution—all without any error messages.

### Why Numerical Errors Are Harder

Numerical errors are difficult to debug because:

1. **No error messages**: The code runs to completion. You only notice something is wrong when you check results.
2. **Delayed symptoms**: The failure might not manifest immediately—loss might diverge after many iterations, or numerical drift might accumulate slowly.
3. **Multiple potential causes**: Wrong results can stem from many sources: dynamic control flow, stale scalars, memory pool issues, missing warmup, RNG state, etc.
4. **Reproduction challenges**: Some issues are timing-dependent or only appear with specific data patterns.
5. **Limited introspection**: During graph replay, CPU code within the graph region is skipped, making it hard to inspect intermediate states with print statements or debuggers.

The fundamental issue is that **CUDA graphs are static and GPU-only, but your eager code might be dynamic and involve essential CPU logic**. Graphs capture one execution path and replay only GPU operations. Any dynamic behavior—control flow, changing tensors, varying shapes—won't update during replay, and any CPU code—print statements, Python conditionals, host-side state mutations—is skipped entirely.

### Categories of Numerical Errors

Numerical errors fall into several categories:

1. **[Dynamic Behavior Failures](#dynamic-behavior-failures)**: Control flow, capture-aware code, dynamic tensors, dynamic scalars, or dynamic shapes that change across iterations but get frozen in the graph
2. **[Race Conditions](#race-conditions)**: CPU-GPU synchronization issues, such as modifying pinned memory while GPU is still reading
3. **[Deconstructed Input Tensors](#deconstructed-input-tensors)**: CUDA or CPU tensors freed or reassigned before graph replay
4. **[Output Tensor Reuse](#output-tensor-reuse)**: Graph outputs overwritten on each replay, corrupting accumulated results
5. **[CPU Code Eliminated During Graph Replay](#cpu-code-eliminated-during-graph-replay)**: Host-side state mutations that no longer execute during replay
6. **[Missing Operations in Captured Graph](#missing-operations-in-captured-graph)**: Library stream unawareness or module hooks not firing
7. **[Memory Pool Sharing Issues](#memory-pool-sharing-issues)**: Shared pool corruption, replay order mismatch, or parallel replay conflicts

Let's examine each failure mode in detail.

---

## Dynamic Behavior Failures

Graphs capture a single execution path. Any runtime variability causes incorrect results. For detailed patterns and solutions, see [Handling Dynamic Patterns](../torch-cuda-graph/handling-dynamic-patterns.md).

### Dynamic Control Flow: Frozen Execution Path

**What it means**: Your code contains `if` statements or other control flow that depends on runtime values (data, iteration counter, etc.). The graph captured one branch and replays that same branch every time, even when conditions change.

**Why it happens**: CUDA graphs are **purely GPU-side**. During capture, the CPU executes all Python code normally, making decisions and choosing branches. The graph records which GPU operations were launched. During replay, **only GPU kernels execute**—no CPU code runs. The graph replays the exact operations from capture, regardless of current conditions.

**Example**:

```python
import torch

def test():
    def func(idx, x, y):
        if idx % 2 == 0:  # ❌ Dynamic control flow!
            z = x + y
        else:
            z = x - y
        return z * x

    x = torch.ones(1, device="cuda", dtype=torch.int32)
    y = torch.ones(1, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        if i < 3:  # warmup
            z = func(i, x, y)
        elif i == 3:  # capture - idx=3 is odd, captures subtraction
            with torch.cuda.graph(graph):
                z = func(i, x, y)  # Captures: z = x - y; return z * x
            graph.replay()
        else:  # replay
            graph.replay()  # Always executes subtraction, even when idx is even!
        print(f"{i=}, {z.item()=}")
```

**Output**:

```text
i=0, z.item()=2   ← Eager: x + y = 2, result = 2 * 1 = 2
i=1, z.item()=0   ← Eager: x - y = 0, result = 0 * 1 = 0
i=2, z.item()=2   ← Eager: x + y = 2, result = 2 * 1 = 2
i=3, z.item()=0   ← Graph captured: x - y = 0, result = 0 * 1 = 0
i=4, z.item()=0   ← Graph replay: still subtraction! (idx % 2 == 0 is ignored)
i=5, z.item()=0   ← Graph replay: still subtraction! (idx % 2 == 1 would match, but idx isn't used)
```

**Why this is wrong**: On iteration 4, `idx % 2 == 0` is True, so eager execution would do addition (result = 2). But the graph captured subtraction (from idx=3), so it produces 0.

**How to fix**:

**Solution 1: Eliminate Control Flow by Executing Both Branches**

Execute both branches unconditionally, then use GPU-side operations like `torch.where()` to select the result. The condition must be a GPU tensor:

```python
import torch

def func(idx_tensor, x, y):
    # idx_tensor must be a GPU tensor, not Python int
    condition = (idx_tensor % 2 == 0)
    # Both branches execute, result selected on GPU
    add_result = x + y
    sub_result = x - y
    z = torch.where(condition, add_result, sub_result)
    return z * x

x = torch.ones(1, device="cuda", dtype=torch.int32)
y = torch.ones(1, device="cuda", dtype=torch.int32)
idx_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
graph = torch.cuda.CUDAGraph()

for i in range(6):
    idx_tensor.fill_(i)  # Update GPU tensor before each iteration
    if i < 3:  # warmup
        z = func(idx_tensor, x, y)
    elif i == 3:  # capture
        with torch.cuda.graph(graph):
            z = func(idx_tensor, x, y)
        graph.replay()
    else:  # replay
        graph.replay()
    print(f"{i=}, {z.item()=}")
```

**Output**:

```text
# Output:
# i=0, z.item()=2  ← idx=0 (even), addition: (1+1)*1 = 2
# i=1, z.item()=0  ← idx=1 (odd), subtraction: (1-1)*1 = 0
# i=2, z.item()=2  ← idx=2 (even), addition
# i=3, z.item()=0  ← idx=3 (odd), subtraction (captured)
# i=4, z.item()=2  ← idx=4 (even), addition ✅ Correct!
# i=5, z.item()=0  ← idx=5 (odd), subtraction ✅ Correct!
```

**Limitation**: Both branches execute always, wasting computation.

**Solution 2: Lower Condition to CUDA Kernel**

For performance-critical code where you want to avoid computing both branches, write a custom CUDA kernel that handles the conditional logic entirely on the GPU. The kernel should use PyTorch's current stream (or use proper stream synchronization if using a different stream) to be captured correctly:

```python
import torch
from torch.utils.cpp_extension import load_inline

# Custom CUDA kernel with conditional logic
cuda_source = """
#include <c10/cuda/CUDAStream.h>

__global__ void conditional_op_kernel(
    int* output, const int* x, const int* y, const int* idx, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (idx[0] % 2 == 0) {
            output[i] = (x[i] + y[i]) * x[i];  // Addition path
        } else {
            output[i] = (x[i] - y[i]) * x[i];  // Subtraction path
        }
    }
}

torch::Tensor conditional_op(torch::Tensor x, torch::Tensor y, torch::Tensor idx) {
    auto output = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Use PyTorch's current stream for CUDA graph capture compatibility
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    conditional_op_kernel<<<blocks, threads, 0, stream>>>(
        output.data_ptr<int>(), x.data_ptr<int>(), y.data_ptr<int>(),
        idx.data_ptr<int>(), n
    );
    return output;
}
"""

cpp_source = "torch::Tensor conditional_op(torch::Tensor x, torch::Tensor y, torch::Tensor idx);"

# Load the custom kernel
conditional_module = load_inline(
    name="conditional_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["conditional_op"],
    verbose=False,
)

x = torch.ones(1, device="cuda", dtype=torch.int32)
y = torch.ones(1, device="cuda", dtype=torch.int32)
idx_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)
graph = torch.cuda.CUDAGraph()

for i in range(6):
    idx_tensor.fill_(i)
    if i < 3:  # warmup
        z = conditional_module.conditional_op(x, y, idx_tensor)
    elif i == 3:  # capture
        with torch.cuda.graph(graph):
            z = conditional_module.conditional_op(x, y, idx_tensor)
        graph.replay()
    else:  # replay
        graph.replay()
    print(f"{i=}, {z.item()=}")

# Output:
# i=0, z.item()=2  ← idx=0 (even), addition
# i=1, z.item()=0  ← idx=1 (odd), subtraction
# i=2, z.item()=2  ← idx=2 (even), addition
# i=3, z.item()=0  ← idx=3 (odd), subtraction (captured)
# i=4, z.item()=2  ← idx=4 (even), addition ✅ Correct!
# i=5, z.item()=0  ← idx=5 (odd), subtraction ✅ Correct!
```

**Advantage**: Only the selected branch executes, no wasted computation. Best for simple conditionals that can be easily expressed in a CUDA kernel.

**Solution 3: Multiple Graphs**

Capture separate graphs for each branch and select at runtime:

```python
def func_add(x, y):
    return (x + y) * x

def func_sub(x, y):
    return (x - y) * x

# Capture both paths with separate output tensors
g_add = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_add):
    z_add = func_add(x, y)

g_sub = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_sub):
    z_sub = func_sub(x, y)

# Select graph at runtime
if idx % 2 == 0:
    g_add.replay()
    z = z_add
else:
    g_sub.replay()
    z = z_sub
```

**Solution 4: Partial Capture**

When control flow is complex or involves non-graphable operations, keep the control flow in eager mode and only graph the static computation portions. This is useful when the dynamic part is a small fraction of the overall computation—you still get speedup from graphing the expensive static parts while maintaining flexibility for the dynamic logic.

:::{note} CUDA 12.4+ Conditional Nodes

CUDA 12.4 introduced [Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/) for dynamic control flow within graphs. However, as of PyTorch 2.9, conditional nodes are not yet integrated into PyTorch's CUDA graph APIs.

For complex control flow, the pragmatic solution is to keep those parts in eager mode and graph only the static portions.
:::

For real-world examples including gradient clipping and early exit inference, see [Handling Dynamic Patterns - Dynamic Control Flow](../torch-cuda-graph/handling-dynamic-patterns.md#dynamic-control-flow).

### Capture-Aware Code: Behavioral Differences

**What it means**: Code checks `torch.cuda.is_current_stream_capturing()` and behaves differently during capture vs replay. This is a special case of dynamic control flow where the condition depends on whether the stream is currently capturing. The graph captures one branch (the capture-time branch), but eager execution takes a different branch.

**Why it happens**: Sometimes code intentionally checks capture status to handle graph constraints. But if this changes the computational logic (not just how it's executed), the graph captures different operations than would run in eager mode.

**Example**:

```python
import torch

def test():
    def func(x: torch.Tensor, y: torch.Tensor):
        z = x + y
        if torch.cuda.is_current_stream_capturing():
            z *= -1  # ❌ Different logic during capture!
        return z

    x = torch.zeros(1, device='cuda', dtype=torch.int32)
    y = torch.zeros(1, device='cuda', dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        y.fill_(i)
        if i < 3:  # warmup - not capturing
            z = func(x, y)  # Computes: x + y
        elif i == 3:  # capture - is capturing!
            with torch.cuda.graph(graph):
                z = func(x, y)  # Computes: -(x + y)
            graph.replay()
        else:  # replay - not capturing, but uses captured graph!
            graph.replay()  # Replays: -(x + y)
        print(f"{i=}, {z.item()=}")
```

**Output**:
```text
i=0, z.item()=0   ← Eager: 0 + 0 = 0
i=1, z.item()=1   ← Eager: 0 + 1 = 1
i=2, z.item()=2   ← Eager: 0 + 2 = 2
i=3, z.item()=-3  ← Captured: -(0 + 3) = -3
i=4, z.item()=-4  ← Replays captured logic: -(0 + 4) = -4
i=5, z.item()=-5  ← Replays captured logic: -(0 + 5) = -5
```

**Why this is wrong**: During replay, `is_current_stream_capturing()` returns False (not actively capturing), but the graph still executes the negation that was captured. Eager execution (iterations 0-2) didn't negate, but the graph always does.

**How to fix**:

**Use is_current_stream_capturing() Only for Implementation, Not Logic**

```python
def func(x: torch.Tensor, y: torch.Tensor):
    z = x + y

    # ✅ Correct use: change implementation, not computation
    if torch.cuda.is_current_stream_capturing():
        # Use graph-safe alternative
        z = z.contiguous()  # Ensure contiguous for graph
    else:
        # Normal path might use different layout
        pass

    return z  # Same result regardless of capture status
```

:::{warning} Avoid Computational Changes Based on Capture Status

Bad patterns:

```python
# ❌ Wrong: different math during capture
if torch.cuda.is_current_stream_capturing():
    output = model_variant_1(input)
else:
    output = model_variant_2(input)

# ❌ Wrong: skipping operations
if not torch.cuda.is_current_stream_capturing():
    output = postprocess(output)

# ❌ Wrong: different precision
if torch.cuda.is_current_stream_capturing():
    output = output.half()
```

Good patterns:

```python
# ✅ Correct: same result, different implementation
if torch.cuda.is_current_stream_capturing():
    # Use graph-safe sleep alternative
    torch.cuda._sleep(1000)
else:
    time.sleep(0.001)
```
:::

### Dynamic Input Tensors: Stale Memory Addresses

**What it means**: You're creating new tensors between replays instead of updating existing tensors. The graph captured pointers to specific memory addresses and continues reading from those addresses during replay, even though your code has moved on to different memory locations.

**Why it happens**: CUDA graphs record memory addresses of graph's input tensors. If you reassign a variable to a new tensor (`input = new_tensor`), Python updates the variable but the graph still points to the old memory address. The graph has no knowledge of Python variables—it only knows GPU memory addresses.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    base_lr = torch.zeros(1, device="cuda", dtype=torch.int32)

    # warmup
    for i in range(2):
        lr = base_lr + i  # ❌ Creates NEW tensor each iteration
        out = 1 * lr      # This is our "captured operation"
        print(f"{i=}, {out.item()=}")

    # capture - captured when lr points to (base_lr + 1)
    with torch.cuda.graph(graph):
        out = 1 * lr  # Graph captures address of lr (which currently = base_lr + 1)

    # replay
    for i in range(2, 6):
        lr = base_lr + i  # Creates NEW tensor, graph doesn't see this!
        graph.replay()    # Graph reads from OLD lr address (still contains 1)
        print(f"{i=}, {out.item()=}")
```

**Output**:
```text
i=0, out.item()=0  ← Eager: lr = 0 + 0 = 0
i=1, out.item()=1  ← Eager: lr = 0 + 1 = 1
i=2, out.item()=1  ← Graph: reads from captured lr address (contains 1)
i=3, out.item()=1  ← Graph: same address, same value
i=4, out.item()=1  ← Graph: same address, same value
i=5, out.item()=1  ← Graph: same address, same value
```

**Why this is wrong**: New `lr` tensors are created on iterations 2-5, but the graph still reads from the memory address captured during capture (which contained 1). The new tensors are never seen by the graph.

:::{warning} Potential Illegal Memory Access

This issue can also cause illegal memory access errors. If the original tensor is deallocated and its memory is recycled by PyTorch's caching allocator for other tensors, the graph replay will attempt to access freed or repurposed memory, leading to crashes or corrupted data.
:::

**How to fix**:

**Solution: Use Static Input Tensors with In-Place Updates**

Identify all input tensors of the CUDA graph, allocate them once before capture, and keep them persistent for the entire graph lifetime. Use in-place operations like `.copy_()` or `.fill_()` to update their values between replays instead of reassigning to new tensors.

```python
# Allocate static tensor ONCE
lr = torch.zeros(1, device="cuda", dtype=torch.int32)

# warmup
for i in range(2):
    lr.copy_(base_lr + i)  # ✅ Update values, same address
    out = 1 * lr

# capture
with torch.cuda.graph(graph):
    out = 1 * lr  # Captures this address

# replay
for i in range(2, 6):
    lr.copy_(base_lr + i)  # ✅ Updates values at captured address
    graph.replay()         # Sees new values!
    print(f"{i=}, {out.item()=}")
```

:::{warning} Common Hidden Dynamic Tensors

Dynamic tensors aren't always obvious, especially when they are deeply nested inside the capture region or when users forget that certain tensors change between iterations. Watch for:

- **Global tensors used within the graph**: Any global tensor referenced inside the graphed region is a graph input, even if it seems "internal" to the model
- **Preprocessing outputs**: Results from data augmentation, tokenization, or other preprocessing steps that create new tensors each iteration
- **Cached computations**: Tensors stored in module attributes or global caches that get updated

Always audit your graphed region to identify all tensors that cross the graph boundary from outside.
:::

**Autocast cache**: A subtle variant occurs with `torch.amp.autocast`. When autocast casts a tensor (e.g., FP32 weight to FP16), the result is cached. If you capture a graph that uses this cached tensor, the graph references the cache's memory address. When the autocast context exits, the cache is cleared and the memory may be recycled, leaving the graph with a stale reference.

```python
import torch

w = torch.ones(1, 1, device='cuda', dtype=torch.float32, requires_grad=True)
x = torch.ones(1, 1, device='cuda', dtype=torch.float32)
g = torch.cuda.CUDAGraph()

with torch.amp.autocast("cuda"):
    _ = x @ w  # w cast to FP16 and cached

    with torch.cuda.graph(g):
        y = x @ w  # Graph captures reference to cached FP16 weight
# Exiting autocast clears the cache, cached FP16 weight may be recycled

g.replay()
print(f"{w.item()=}, {y.item()=}")  # w.item()=1.0, y.item()=1.0

with torch.no_grad():
    w.fill_(2)
g.replay()
print(f"{w.item()=}, {y.item()=}")  # w.item()=2.0, y.item()=1.0 (stale!)
```

**How to fix**: Either disable autocast caching (`cache_enabled=False`) or capture the entire autocast context inside the graph so the cast operation itself is replayed:

```python
# ✅ Option 1: Disable cache
with torch.amp.autocast("cuda", cache_enabled=False):
    with torch.cuda.graph(g):
        y = x @ w  # Cast happens fresh each replay

# ✅ Option 2: Capture autocast inside graph
with torch.cuda.graph(g):
    with torch.amp.autocast("cuda"):
        y = x @ w  # Cast is captured in graph
```

For real-world examples including global statistics tensors and grouped GEMM in MoE models, see [Handling Dynamic Patterns - Dynamic Tensors](../torch-cuda-graph/handling-dynamic-patterns.md#dynamic-tensors).

### Dynamic Scalars: Captured Constants

**Symptom**: Operations always use the same scalar value (like exponent, threshold, etc.) regardless of what you pass during replay

**What it means**: A Python scalar (int, float) is used in a CUDA kernel. The scalar value is baked into the graph during capture and never updates during replay.

**Why it happens**: Python scalars passed to CUDA kernels become kernel parameters—immediate values embedded in the kernel launch. The graph captures the kernel launch with that specific scalar value. During replay, the exact same kernel launch is executed with the exact same scalar value, regardless of what your Python code does.

**Example**:

```python
import torch

def test():
    def func(tensor: torch.Tensor, exponent: int):  # exponent is Python int
        return torch.float_power(tensor, exponent)

    x = torch.tensor(2, device="cuda", dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        if i < 3:  # warmup
            y = func(x, i)
        elif i == 3:  # capture with exponent=3
            with torch.cuda.graph(graph):
                y = func(x, i)  # Captures: power(x, 3)
            graph.replay()
        else:  # replay
            graph.replay()  # Always power(x, 3), even though i changes!
        print(f"{i=}, {y.item()=}")
```

**Output**:
```text
i=0, y.item()=1.0  ← 2^0 = 1
i=1, y.item()=2.0  ← 2^1 = 2
i=2, y.item()=4.0  ← 2^2 = 4
i=3, y.item()=8.0  ← Graph captured: 2^3 = 8
i=4, y.item()=8.0  ← Graph replay: still 2^3 = 8 (i=4 ignored)
i=5, y.item()=8.0  ← Graph replay: still 2^3 = 8 (i=5 ignored)
```

**Why this is wrong**: On iterations 4 and 5, we'd expect 2^4=16 and 2^5=32, but the graph replays the captured operation (2^3=8).

**How to fix**:

**Solution: Convert Scalars to Tensors**

Dynamic or changing CPU side scalars are CUDA graph inputs. Convert them to static, persistent GPU tensors allocated before capture, and use in-place operations like `.fill_()` to update their values before each replay. If the scalar is passed to a custom CUDA kernel, you must also refactor the kernel to accept a device pointer and load the actual value from that pointer instead of taking the scalar as a kernel parameter.

```python
def func(tensor: torch.Tensor, exponent: torch.Tensor):  # ✅ Tensor, not int
    return torch.float_power(tensor, exponent)

x = torch.tensor(2, device="cuda", dtype=torch.int32)
exponent_tensor = torch.zeros(1, device="cuda", dtype=torch.int32)

# warmup
for i in range(3):
    exponent_tensor.fill_(i)
    y = func(x, exponent_tensor)

# capture
with torch.cuda.graph(graph):
    y = func(x, exponent_tensor)  # Captures tensor address, not value

# replay
for i in range(3, 6):
    exponent_tensor.fill_(i)  # Update tensor value
    graph.replay()            # Uses new value!
```

:::{note} When Scalars Are Safe

Python scalars are safe if they **truly never change** across graph replays:

```python
# ✅ Safe: constants
DROPOUT_RATE = 0.1  # Never changes
HIDDEN_DIM = 512    # Never changes

# ❌ Unsafe: variables
current_lr = scheduler.get_lr()    # Changes per step
temperature = compute_temperature()  # Computed value
```
:::

:::{warning} Common Hidden Dynamic Scalars

Dynamic scalars aren't always obvious, especially when the CUDA graph capture range is large. If your graph includes any of these, ensure they are converted to static tensors updated with `.fill_()`:

- **Learning rate**: Changes every step

    ```python
    # ❌ Wrong
    for step in range(100):
        lr = scheduler.get_lr()  # Different each step!
        graph.replay()

    # ✅ Correct
    lr_tensor = torch.zeros(1, device='cuda')
    for step in range(100):
        lr_tensor.fill_(scheduler.get_lr())
        graph.replay()
    ```

- **GradScaler (AMP)**: The `GradScaler` maintains internal state (loss scale) that changes. Keep `scaler.step()` and `scaler.update()` outside the graph:
    ```python
    # ❌ Wrong: scaler operations inside graph
    with torch.cuda.graph(g):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # ✅ Correct: only scale(loss).backward() inside, step/update outside
    with torch.cuda.graph(g):
        scaler.scale(loss).backward()
    g.replay()
    scaler.step(optimizer)
    scaler.update()
    ```
    To fully include GradScaler in the CUDA graph, you must convert its internal state to GPU tensors and modify any operations that read the scaling factor to load from device memory instead.

- **Epoch/step counters**: If passed to model

    ```python
    # ❌ Wrong: Python int
    output = model(input, step=global_step)

    # ✅ Correct: CUDA tensor
    step_tensor = torch.zeros(1, device='cuda', dtype=torch.int64)
    step_tensor.fill_(global_step)
    output = model(input, step=step_tensor)
    ```

- **RNG state**: Random operations need different values each replay. Register generators with the graph to convert RNG state to GPU tensors:

    ```python
    g = torch.cuda.CUDAGraph()
    # Register default generator (often automatic, but explicit is safer)
    g.register_generator_state(torch.cuda.default_generators[0])

    # For custom generators
    custom_gen = torch.Generator(device='cuda')
    g.register_generator_state(custom_gen)

    with torch.cuda.graph(g):
        output = F.dropout(input, p=0.5, training=True)
    ```
:::

For real-world examples including RNG state management and learning rate in APEX FusedAdam, see [Handling Dynamic Patterns - Dynamic Scalars](../torch-cuda-graph/handling-dynamic-patterns.md#dynamic-scalars).

### Dynamic Shapes: Fixed Tensor Dimensions

**Symptom**: Model produces truncated or incorrect outputs when input dimensions change

**What it means**: CUDA graphs require static tensor shapes. The graph captures kernel configurations (grid/block dimensions) optimized for specific tensor sizes. Changing shapes during replay leads to silent incorrect results—the graph processes only the originally captured dimensions.

**Why it happens**: When you capture a graph with tensor shape `(4,)`, the CUDA kernels are configured to process exactly 4 elements. During replay, even if you want to process 8 elements, the graph still launches kernels configured for 4 elements. The extra elements are simply ignored.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()

    # capture with shape (4,)
    x = torch.ones(4, device="cuda", dtype=torch.int32)
    with torch.cuda.graph(graph):
        y = x * 2

    # replay with same shape - works correctly
    x.fill_(3)
    graph.replay()
    print(f"same shape: {y=}")  # tensor([6, 6, 6, 6]) ✅

    # attempt replay with different shape - silent error!
    # graph only processes 4 elements, ignores the rest
    x_large = torch.ones(8, device="cuda", dtype=torch.int32) * 5
    x.copy_(x_large[:4])  # can only update the captured tensor's 4 elements
    graph.replay()
    print(f"larger input: {y=}")  # still only 4 elements! ❌
```

**Why this is wrong**: The user wants to process 8 elements, but the graph was captured with 4 elements. The output `y` remains shape `(4,)` regardless of input size.

**How to fix**:

**Solution 1: Pad to Fixed Maximum Size**

Pad all inputs to a fixed maximum size and capture a single graph:

```python
max_size = 1024
static_input = torch.zeros(max_size, device='cuda')

with torch.cuda.graph(g):
    static_output = model(static_input)

# Pad variable-length inputs at runtime
for data in dataloader:
    actual_size = data.shape[0]
    static_input[:actual_size].copy_(data)
    static_input[actual_size:].zero_()  # Zero padding
    g.replay()
    result = static_output[:actual_size].clone()  # Trim to actual size
```

**Solution 2: Bucketing with Multiple Graphs**

Capture multiple graphs for common shape configurations:

```python
buckets = [128, 256, 512, 1024]
graphs = {}

for size in buckets:
    static_input = torch.zeros(size, device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_output = model(static_input)
    graphs[size] = {'graph': g, 'input': static_input, 'output': static_output}

# Select appropriate bucket at runtime
def forward(data):
    size = data.shape[0]
    bucket = min([s for s in buckets if s >= size], default=max(buckets))
    graphs[bucket]['input'][:size].copy_(data)
    graphs[bucket]['graph'].replay()
    return graphs[bucket]['output'][:size].clone()
```

For real-world examples including variable sequence lengths in NLP and MoE models with token dropless, see [Handling Dynamic Patterns - Dynamic Shapes](../torch-cuda-graph/handling-dynamic-patterns.md#dynamic-shapes).

---

## Race Conditions

### Pinned Memory Race Condition

**Symptom**: Incorrect numerical results when using pinned memory as graph input; all iterations may produce the same value or values from a different iteration.

**What it means**: The CPU is writing to pinned memory while the GPU is still reading from it. Since `graph.replay()` is asynchronous—it returns immediately while the GPU is still executing—the CPU can overwrite the pinned memory before the GPU has finished reading the previous value.

**Why it happens**: Pinned (page-locked) memory enables fast asynchronous transfers between CPU and GPU. When you call `tensor.to(device="cuda", non_blocking=True)` on a pinned tensor, the copy is enqueued on the GPU but returns immediately to the CPU. If the CPU modifies the pinned memory before the GPU completes the copy, the GPU reads corrupted or future values.

This issue is **not specific to CUDA graphs**—it exists with any asynchronous GPU operations. However, CUDA graphs make it more likely because `graph.replay()` is always asynchronous and returns immediately.

This issue was originally identified in [pytorch/pytorch#166765](https://github.com/pytorch/pytorch/pull/166765#issuecomment-3487509660), which also introduces detection for CPU-write-after-GPU-read hazards during stream capture.

**Example**:

```python
import torch
from torch.testing._internal.common_utils import get_cycles_per_ms

cycles_per_ms = get_cycles_per_ms()

def func(x):
    torch.cuda._sleep(int(50 * cycles_per_ms))  # Simulate GPU work
    x_cuda = x.to(device="cuda", non_blocking=True)
    y = x_cuda ** 2
    return y

def run_eager(x):
    stream = torch.cuda.Stream()
    outputs = []
    with torch.cuda.stream(stream):
        for i in range(4):
            x.fill_(i)  # ❌ CPU writes while GPU may still be reading
            y = func(x)
            outputs.append(y.clone())
    torch.cuda.synchronize()
    return outputs

def run_graph(x):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = func(x)

    outputs = []
    for i in range(4):
        x.fill_(i)      # ❌ CPU writes while GPU may still be reading
        graph.replay()  # Asynchronous - returns immediately!
        outputs.append(y.clone())
    torch.cuda.synchronize()
    return outputs

x = torch.empty(1, device="cpu", pin_memory=True)
eager_outputs = run_eager(x)
graph_outputs = run_graph(x)

for i in range(4):
    print(f"{i=}, expect {i**2}, eager {eager_outputs[i].item()}, graph {graph_outputs[i].item()}")
```

**Output**:

```
i=0, expect 0, eager 0.0, graph 9.0
i=1, expect 1, eager 9.0, graph 9.0
i=2, expect 4, eager 9.0, graph 9.0
i=3, expect 9, eager 9.0, graph 9.0
```

Both eager and graph modes produce wrong results. The GPU reads the final value (`3`) for most iterations because the CPU has already looped ahead and overwritten the pinned memory.

**How to fix**:

1. **Synchronize before CPU writes**: Ensure the GPU has finished reading before modifying pinned memory. Use `torch.cuda.synchronize()` for full device sync, or use events for more fine-grained control:

    ```python
    # Option A: Full device synchronization
    for i in range(4):
        torch.cuda.synchronize()  # ✅ Wait for GPU to finish reading
        x.fill_(i)
        graph.replay()

    # Option B: Event-based synchronization (more efficient)
    event = torch.cuda.Event()
    for i in range(4):
        event.synchronize()  # ✅ Wait for previous replay to finish
        x.fill_(i)
        graph.replay()
        event.record()
    ```

2. **Use CUDA tensors instead**: Copy data to a CUDA tensor and use that as graph input. CUDA-to-CUDA copies within the same stream are properly ordered:

    ```python
    x_cpu = torch.empty(1, device="cpu", pin_memory=True)
    x_cuda = torch.empty(1, device="cuda")  # Static graph input

    with torch.cuda.graph(graph):
        y = x_cuda ** 2

    for i in range(4):
        x_cpu.fill_(i)
        x_cuda.copy_(x_cpu)  # ✅ Copy is enqueued on stream, properly ordered
        graph.replay()
    ```

---

## Deconstructed Input Tensors

CUDA graphs capture memory addresses of input tensors. If these tensors are deallocated (deconstructed) before or during graph replay, the graph accesses freed memory, leading to corrupted results, segmentation faults, or illegal memory access errors. This applies to both CUDA tensors and CPU tensors used in host-to-device copies.

### Deconstructed CUDA Tensors

**Symptom**: Illegal memory access errors, corrupted outputs, or silent incorrect results

**What it means**: A CUDA tensor that serves as a graph input was deallocated after capture but before replay. The graph still holds the original memory address, which may now point to freed memory or memory reused for other tensors.

**Why it happens**: This can occur in several ways: reassigning a tensor variable (`x = new_tensor`) instead of updating in-place (`x.copy_(new_data)`), explicitly deleting a tensor (`del x`), or losing the last reference to a tensor deep in a software stack (e.g., a library function that doesn't preserve references). Once Python garbage-collects the tensor, PyTorch's caching allocator may reuse that memory for other tensors. The graph continues to use the old address, which now contains unrelated data or has been freed entirely.

**Example**:

```python
import torch

def capture_graph():
    # ❌ Local tensor - will be garbage collected after function returns!
    static_input = torch.ones(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = static_input * 2
    return g, output  # static_input goes out of scope here!

g, output = capture_graph()
# static_input is now garbage collected, its memory may be reused

# Allocate another tensor that reuses the freed memory
other_tensor = torch.randn(4, device="cuda")

g.replay()  # ❌ Reads from reused memory (now contains random values)!
print(output)  # Expected: [2., 2., 2., 2.], Actual: random values like [-0.12, -1.11, ...]
```

**How to fix**: Keep graph input tensors persistent for the entire graph lifetime. Return them from the capture function or store them in a class/module:

```python
def capture_graph():
    static_input = torch.ones(4, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = static_input * 2
    return g, static_input, output  # ✅ Return static_input to keep it alive

g, static_input, output = capture_graph()
static_input.fill_(3)  # Update via in-place operation
g.replay()
print(output)  # tensor([6., 6., 6., 6.]) ✅
```

See [Dynamic Input Tensors: Stale Memory Addresses](#dynamic-input-tensors-stale-memory-addresses) for similar issues and solutions.

### Deconstructed CPU Tensors

**Symptom**: Corrupted results, wrong values, or segmentation faults

**What it means**: A CPU tensor was created during capture but deconstructed (garbage collected) before graph replay. The graph holds a pointer to freed host memory.

**Why it happens**: During capture, the graph records an H2D copy operation with a specific source host address. While PyTorch's host caching allocator ensures the CPU tensor remains valid during the initial capture, the tensor may be freed afterward if no Python references remain. When the graph replays, it attempts to copy from the original host address, but that memory has been freed or reused for other data.

**Example**:

```python
import torch

def test():
    def func(i: int):
        # Note: 'i' is a CPU scalar which can also cause issues (see Dynamic Scalars section)
        # Here we use it only to illustrate the CPU tensor deallocation problem
        t_cpu = torch.tensor(i)  # CPU tensor created
        t_gpu = t_cpu.to(device='cuda', non_blocking=True)  # H2D copy captured
        return t_gpu  # ❌ t_cpu loses reference after function returns

    graph = torch.cuda.CUDAGraph()
    for i in range(6):
        if i < 3:  # warmup
            y = func(i)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = func(i)  # Graph captures H2D copy from t_cpu's host address
            graph.replay()  # t_cpu freed after capture, replay reads freed memory!
        else:  # replay
            graph.replay()  # Reads from freed/reused host memory!
        print(f"{i=}, {y.item()=}")
```

**Output**:
```text
i=0, y.item()=0
i=1, y.item()=1
i=2, y.item()=2
i=3, y.item()=94457301632419  ← Corrupted! Replay reads freed host memory
i=4, y.item()=94457301632419
i=5, y.item()=94457301632419
```

**Why this happens**: The graph captured an H2D copy from `t_cpu`'s host address. After capture completes and the function returns, `t_cpu` loses its last reference and is freed. During replay, the H2D copy reads from that freed host address, getting garbage data.

**How to fix**:

**Solution 1: Keep CPU Tensors Alive**

```python
def func(i: int, cpu_tensors: list):
    t_cpu = torch.tensor(i)
    cpu_tensors.append(t_cpu)  # ✅ Keep reference alive
    t_gpu = t_cpu.to(device='cuda', non_blocking=True)
    return t_gpu

cpu_tensors = []
with torch.cuda.graph(graph):
    y = func(i, cpu_tensors)
# cpu_tensors kept alive throughout graph lifetime
```

**Solution 2: Avoid CPU Tensors in Graphs**

```python
def func(i: int):
    # ✅ Create directly on GPU
    t_gpu = torch.tensor(i, device='cuda')
    return t_gpu
```

:::{warning} Device Pointer Arrays

This issue is particularly common with operations that use device pointer arrays (arrays of pointers to GPU memory):

- [Strided Batched GEMM](https://docs.nvidia.com/cuda/cublas/#cublasgemmstridedbatchedex) (cuBLAS `cublasGemmStridedBatchedEx`)
- [Grouped GEMM](https://docs.nvidia.com/cuda/cublas/#cublasgemmgroupedbatchedex) (cuBLAS `cublasGemmGroupedBatchedEx`)
- Custom kernels with pointer arrays

Ensure all tensors in the pointer array remain allocated for the graph's lifetime.
:::

---

## Output Tensor Reuse

Graph outputs are static tensors—the same memory is overwritten on every replay. If you collect outputs across multiple replays without cloning, all references point to the same tensor containing only the final replay's result.

### Accumulating Graph Outputs

**Symptom**: All collected outputs have the same value (the last replay's result), or aggregated results are incorrect.

**What it means**: You're appending or storing graph output tensors across iterations, but they all reference the same underlying memory. Each replay overwrites that memory, so previous "outputs" are silently corrupted.

**Why it happens**: Unlike eager execution where each forward pass creates new output tensors, CUDA graph replay reuses the same output tensor allocated during capture. When you append `y` to a list, you're appending a reference to that tensor, not a copy. After multiple replays, every element in your list points to the same tensor—which contains only the most recent result.

**Example**:

```python
import torch

x = torch.empty(1, device='cuda')
func = lambda x: x ** 2 + 1

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = func(x)

ys = []
for i in range(3):
    x.fill_(i)
    graph.replay()
    ys.append(y)  # ❌ Appending reference to same tensor!

out = torch.sum(torch.cat(ys))
print(f"{ys[0].item()=}, {ys[-1].item()=}, {out.item()=}")
```

**Output**:
```text
ys[0].item()=5.0, ys[-1].item()=5.0, out.item()=15.0
```

**Why this is wrong**: The expected outputs are `0² + 1 = 1`, `1² + 1 = 2`, `2² + 1 = 5`, with sum = 8. But all three elements in `ys` show 5.0 (the final result), and the sum is 15.0 (= 5 × 3). Each `graph.replay()` overwrites `y`, and since `ys` contains three references to the same `y`, they all reflect the final value.

**How to fix**:

**Solution: Clone Outputs Before Storing**

```python
ys = []
for i in range(3):
    x.fill_(i)
    graph.replay()
    ys.append(y.clone())  # ✅ Clone to preserve each result

out = torch.sum(torch.cat(ys))
print(f"{ys[0].item()=}, {ys[-1].item()=}, {out.item()=}")
# ys[0].item()=1.0, ys[-1].item()=5.0, out.item()=8.0 ✓
```

:::{warning} Common Scenarios

This issue commonly appears in:

- **Accumulating predictions**: Collecting model outputs across batches for evaluation
- **Gradient accumulation**: Storing intermediate results before optimizer step
- **Pipeline parallelism**: Passing outputs between stages across micro-batches
- **Metric computation**: Gathering outputs for loss or accuracy calculation

Always clone graph outputs if you need to preserve values across replays.
:::

---

## CPU Code Eliminated During Graph Replay

CUDA graph replay executes only the captured GPU kernels—all CPU code within the graphed region is skipped. This becomes a problem when eager code *after* the graph depends on CPU state that was supposed to be updated *inside* the graphed region. During capture, the CPU code runs and updates state normally. During replay, the CPU code is eliminated, so the state remains frozen at its capture-time value. Any subsequent code that reads this state will see outdated values and may produce incorrect results.

### Host State Mutation: Frozen CPU State

**Symptom**: Logic after graph replay depends on state that should change but doesn't

**What it means**: Code inside the capture region modifies CPU variables (Python lists, counters, etc.). These mutations happen during capture but are skipped during replay. Eager code after the graph sees stale state.

**Example**:

```python
import torch

def test():
    def graphed_func(state: list, x: torch.Tensor):
        y = x * 1
        state[0] = state[0] + 1  # ❌ CPU mutation - skipped during replay!
        return y

    def eager_func(state: list, x: torch.Tensor):
        if state[0] % 2 == 0:  # Depends on state updated in graphed_func
            x *= -1
        return x

    state = [0]
    x = torch.ones(1, device='cuda', dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()

    for i in range(6):
        if i < 3:  # warmup
            y = graphed_func(state, x)
        elif i == 3:  # capture
            with torch.cuda.graph(graph):
                y = graphed_func(state, x)  # state[0] becomes 4 during capture
            graph.replay()
        else:  # replay
            graph.replay()  # CPU code skipped, state[0] stays 4!

        y = eager_func(state, y)  # Eager code reads stale state
        print(f"{i=}, {state=}, {y.item()=}")
```

**Output**:
```text
i=0, state=[1], y.item()=1    ← state[0]=1 (odd), no negation
i=1, state=[2], y.item()=-1   ← state[0]=2 (even), negated
i=2, state=[3], y.item()=1    ← state[0]=3 (odd), no negation
i=3, state=[4], y.item()=-1   ← Capture: state[0]=4 (even), negated
i=4, state=[4], y.item()=-1   ← Replay: state[0] frozen at 4, eager_func sees wrong value
i=5, state=[4], y.item()=-1   ← Replay: state[0] still 4, eager_func still wrong
```

**Why this is wrong**: Iterations 4 and 5 should increment state to 5 and 6, changing the parity and affecting `eager_func`'s behavior. But since graph replay skips the CPU mutation, state freezes at 4, and `eager_func` produces incorrect results.

**How to fix**:

**Solution 1: Move State Updates Outside Graph**

```python
for i in range(6):
    if i < 3:
        y = graphed_func(state, x)
    else:
        graph.replay()
        state[0] += 1  # ✅ Update in eager code, not inside graph

    y = eager_func(state, y)
```

**Solution 2: Use GPU Tensors for State**

If the state needs to be updated inside the graph and read by eager code afterward, convert it to a GPU tensor:

```python
def graphed_func(state_tensor: torch.Tensor, x: torch.Tensor):
    y = x * 1
    state_tensor += 1  # ✅ GPU operation, captured and replayed
    return y

state_tensor = torch.tensor([0], device='cuda')
with torch.cuda.graph(graph):
    y = graphed_func(state_tensor, x)

# Eager code can read state_tensor.item() after replay
```

:::{warning} Common Host Mutations to Watch For

- Counter increments: `step += 1`
- List operations: `results.append(x)`
- Dictionary updates: `cache[key] = value`
- Object attribute changes: `self.iteration = i`

These all execute during capture only. If eager code after the graph depends on them, move the mutations outside the graph or convert to GPU tensors.
:::

---

## Missing Operations in Captured Graph

Due to various implementation choices, some operations that run correctly in eager mode may be lost in the captured graph. This typically happens when external code or libraries launch CUDA kernels on a different stream than the capture stream—only operations on the capture stream are recorded. These missing operations execute during capture (so eager results look correct), but they are not part of the captured graph and won't execute during replay.

### Library Stream Awareness: Operations on Wrong Stream

**Symptom**: Some library or extension operations don't occur during graph replay, producing wrong results even though eager execution works correctly

**What it means**: You're calling a library or extension that launches CUDA kernels, but it doesn't use PyTorch's current stream (the capture stream). Operations launch on the default stream or another stream without proper stream synchronization, so they are not captured into the graph.

**Why it happens**: PyTorch uses `torch.cuda.current_stream()` for all operations. When you call external code (C++ extension, non-PyTorch library), it might use CUDA's default stream instead. During eager execution, this often works because of implicit synchronization. During graph capture, only operations on the capture stream (or streams properly forked from it) are recorded—operations on other streams are simply not captured.

**Example**:

```python
import torch

def library_call(x):
    # Simulates library that assumes code runs on default stream
    stream = torch.cuda.default_stream()
    with torch.cuda.stream(stream):
        return x * 2

# Static input tensor
x = torch.ones(1, device='cuda', dtype=torch.int32)
graph = torch.cuda.CUDAGraph()

# Warmup
for i in range(3):
    x.fill_(i)
    y = x + 1
    y = library_call(y)
    print(f"warmup {i=}, {y.item()=}")

# Capture
x.fill_(3)
with torch.cuda.graph(graph):
    y = x + 1              # Captured on capture stream
    y = library_call(y)    # ❌ Runs on default stream

# Replay
for i in range(4, 7):
    x.fill_(i)
    graph.replay()
    print(f"replay {i=}, {y.item()=}")
```

**Output**:
```text
warmup i=0, y.item()=2   ← Eager: (0+1)*2 = 2
warmup i=1, y.item()=4   ← Eager: (1+1)*2 = 4
warmup i=2, y.item()=6   ← Eager: (2+1)*2 = 6
replay i=4, y.item()=0   ← Graph: multiplication missing, y contains stale data
replay i=5, y.item()=0
replay i=6, y.item()=0
```

**Why this is wrong**: With `torch.cuda.graph()`, the current stream is a side stream (capture stream), not the default stream. The library assumes code runs on the default stream, so `x * 2` launches on the default stream. Without proper stream synchronization between the capture stream and the default stream, the multiplication is **not recorded into the graph**. During replay, only the captured `x + 1` executes—the multiplication is missing entirely, and `y` contains stale data.

**How to fix**:

**Solution: Make Library Stream-Aware**

For PyTorch extensions, use PyTorch's API to get the current stream:

```python
def library_call(x):
    # ✅ Use PyTorch's current stream
    current = torch.cuda.current_stream()
    with torch.cuda.stream(current):
        x = x * 2
    return x
```

For external libraries, pass PyTorch's current stream to the library. The library should either use the provided stream directly or add proper stream synchronization:

```python
# Option A: Use the provided stream directly
def library_call_a(x, stream):
    with torch.cuda.stream(stream):
        return x * 2

# Option B: Wait on provided stream before using internal stream
def library_call_b(x, stream):
    internal_stream = torch.cuda.Stream()
    internal_stream.wait_stream(stream)
    with torch.cuda.stream(internal_stream):
        result = x * 2
    stream.wait_stream(internal_stream)
    return result

with torch.cuda.graph(graph):
    y = x + 1
    stream = torch.cuda.current_stream()
    y = library_call_a(y, stream)  # or library_call_b(y, stream)
```

:::{note} Checking Library Compatibility

To check if a library is graph-compatible:

1. Profile with Nsight Systems—look for operations on wrong streams
2. Check library documentation for CUDA stream parameters
3. Test: compare eager vs graph results with `torch.allclose()`

Common issues:
- cuDNN: Ensure passing correct stream handle
- Custom CUDA extensions: Must accept stream parameter
- Third-party libraries: May need patches for stream awareness
:::

### Module Hooks Not Firing

**Symptom**: Registered hooks don't fire during graph replay, or only some hooks fire.

**What it means**: `torch.cuda.make_graphed_callables()` replaces the top module's `forward` method with a customized function that replays the captured graph. During replay, submodule `forward` methods are never called—the graph executes as a monolithic sequence of pre-recorded CUDA kernels. As a result, only the top-level module's hooks are invoked; all submodule hooks are skipped.

**Why it happens**: Hooks are triggered by `forward` method calls. Since graph replay bypasses the normal module hierarchy traversal, submodule hooks never fire. If your application relies on these hooks for specific functionality (e.g., logging, gradient modification, activation caching), those operations will be silently skipped after graph capture, leading to incorrect behavior or missing results.

:::{note} Hook Registration Timing

Registering hooks *before* calling `make_graphed_callables()` is not allowed. Hooks must be registered *after* graphing.
:::

**Example**:

```python
import torch

model = torch.nn.Sequential(
    torch.nn.Embedding(4096, 4096, device='cuda'),
    torch.nn.Linear(4096, 4096, device='cuda'),
    torch.nn.Linear(4096, 4096, device='cuda'),
).cuda()

# Capture graph
inp = torch.randint(high=128, size=(256,), device='cuda')
model = torch.cuda.make_graphed_callables(model, (inp,))

# Register hooks AFTER graphing
num_registered, num_called = 0, 0
def hook(module, input, output):
    global num_called
    num_called += 1

for module in model.modules():  # Sequential + 3 submodules = 4 modules
    num_registered += 1
    module.register_forward_hook(hook)

model(inp)
print(f"Registered: {num_registered}, Called: {num_called}")
```

**Output**:

```text
Registered: 4, Called: 1
```

Only the top-level `Sequential` module's hook fires; the three submodule hooks are never called.

**How to work around**:

:::{note} Accessing Parameters from Top-Level Hook

Although submodule hooks don't fire, you can still access all parameters from the top-level hook using `recurse=True`:

```python
def hook(module, input, output):
    # recurse=False: only direct parameters (0 for Sequential)
    # recurse=True: all parameters in the module tree
    for param in module.parameters(recurse=True):
        # Process param
        pass

model.register_forward_hook(hook)
```
:::

:::{warning} Fundamental Limitation

This is a fundamental limitation of `make_graphed_callables`. If you need per-layer hooks to fire, consider:

- Using `torch.cuda.CUDAGraph` directly with finer-grained capture boundaries
- Implementing hook logic outside the graphed region
:::

---

## Memory Pool Sharing Issues

Sharing memory pools between graphs can corrupt data if not done correctly.

### Shared Memory Pool Corruption

**Symptom**: Output tensors from one graph get corrupted after replaying another graph that shares the same memory pool.

**What it means**: When multiple graphs share a memory pool, tensors from different graphs may be allocated at the same memory address. If you read a graph's output after replaying another graph, the output may have been overwritten.

**Why it happens**: The CUDA graph memory pool reuses memory across graphs for efficiency. An intermediate tensor in graph A and an output tensor in graph B might share the same address. When graph A replays, its intermediate tensor overwrites graph B's output—even though you haven't explicitly touched graph B's output.

**Example**:

```python
import torch

def func1(x1):
    t1 = x1 * 3      # Intermediate tensor
    y1 = t1 + 5
    if torch.cuda.is_current_stream_capturing():
        print(f"{x1.data_ptr()=:#x}, {t1.data_ptr()=:#x}, {y1.data_ptr()=:#x}")
    return y1

def func2(x2):
    y2 = x2 ** 2     # Output tensor
    if torch.cuda.is_current_stream_capturing():
        print(f"{x2.data_ptr()=:#x}, {y2.data_ptr()=:#x}")
    return y2

x1 = torch.tensor([1.0], device='cuda')
x2 = torch.tensor([2.0], device='cuda')
graph1 = torch.cuda.CUDAGraph()
graph2 = torch.cuda.CUDAGraph()

# Warmup
y1 = func1(x1)
y2 = func2(x2)
print(f"Correct: {y1.item()=}, {y2.item()=}")

# Capture with shared pool
with torch.cuda.graph(graph1):
    y1 = func1(x1)
with torch.cuda.graph(graph2, pool=graph1.pool()):
    y2 = func2(x2)

# Replay in DIFFERENT order than capture
graph2.replay()  # y2 = 2^2 = 4 ✓
graph1.replay()  # t1 overwrites y2's memory!
print(f"Actual: {y1.item()=}, {y2.item()=}")
```

**Output**:
```text
Correct: y1.item()=8.0, y2.item()=4.0
x1.data_ptr()=0x7f2c15000000, t1.data_ptr()=0x7f2bfa600000, y1.data_ptr()=0x7f2bfa600200
x2.data_ptr()=0x7f2c15000200, y2.data_ptr()=0x7f2bfa600000
Actual: y1.item()=8.0, y2.item()=3.0  # y2 corrupted!
```

**Why y2 becomes 3.0**: The memory addresses reveal the issue—`t1` and `y2` share the same address (`0x7f2bfa600000`). When graph1 replays, it computes `t1 = x1 * 3 = 1.0 * 3 = 3.0` and writes to that address, overwriting `y2`'s correct value of 4.0.

:::{note} When Shared Pool Causes Corruption

Pool sharing causes corruption when **all three conditions** are met:

1. **A graph holds memory allocated from the shared pool** — e.g., graph2 holds `y2` at address `0x7f2bfa600000`
2. **That memory address is reused by another graph** — the allocator assigns the same address to a different tensor, e.g., graph1 uses `0x7f2bfa600000` for intermediate tensor `t1`
3. **The memory is still in use when another graph replays** — e.g., graph2's `y2` at `0x7f2bfa600000` is still needed after graph1 replays, so `y2` gets corrupted by `t1`

If you read a graph's output immediately after its replay (before any other graph replays), the result will be correct—even with a shared pool.

A simple but stringent rule of thumb: **when sharing memory pools, keep replay order the same as capture order**.
:::

**How to fix**:

**Solution 1: Use output immediately after replay**

```python
graph2.replay()
result2 = y2.clone()  # Save before other graphs corrupt it
graph1.replay()
result1 = y1.clone()
```

**Solution 2: Don't share pools between unrelated graphs**

```python
# Each graph gets its own pool
with torch.cuda.graph(graph1):
    y1 = func1(x1)
with torch.cuda.graph(graph2):  # No pool= argument
    y2 = func2(x2)
```

**Solution 3: Maintain consistent capture and replay order**

```python
# Capture order: graph1, graph2
# Replay order: graph1, graph2 (same order)
graph1.replay()
graph2.replay()
```

### Replay Order Mismatch

This issue commonly occurs when using `torch.cuda.make_graphed_callables` to capture multiple graphs with a shared memory pool.

**Symptom**: Some graphs produce wrong results when multiple graphs share a memory pool.

**Why it happens**: In training, forward graphs hold intermediate tensors (activations) for backward passes, and backward graphs hold computed gradients. When graphs share a memory pool, these tensors can be allocated at the same addresses across different graphs. If replay order differs from capture order, a later graph may overwrite memory that an earlier graph still needs.

**Example**:

```python
import torch

x0 = torch.tensor([1.0], device='cuda', requires_grad=True)
x1 = torch.tensor([2.0], device='cuda', requires_grad=True)
x2 = torch.tensor([3.0], device='cuda', requires_grad=True)
func = lambda x: x ** 2 + 2 * x + 1
funcs = tuple(func for _ in range(3))
inputs = tuple(((x0,), (x1,), (x2,)))

graphed_funcs = torch.cuda.make_graphed_callables(funcs, inputs)

# Recording order: f0_fprop, f0_bprop, f1_fprop, f1_bprop, f2_fprop, f2_bprop
# Replay order:    f0_fprop, f1_fprop, f0_bprop, f2_fprop, f1_bprop, f2_bprop

y0 = graphed_funcs[0](x0)
y1 = graphed_funcs[1](x1)
y0.backward()
y2 = graphed_funcs[2](x2)
y1.backward()
y2.backward()

gs = [2 * t[0] + 2 for t in inputs]
print(f"Actual grad: {x0.grad.item()=}, {x1.grad.item()=}, {x2.grad.item()=}")
print(f"Correct grad: {gs[0].item()=}, {gs[1].item()=}, {gs[2].item()=}")
```

**Output**:
```text
Actual grad: x0.grad.item()=6.0, x1.grad.item()=3.0, x2.grad.item()=8.0
Correct grad: gs[0].item()=4.0, gs[1].item()=6.0, gs[2].item()=8.0
```

The example captures three functions with `make_graphed_callables`, which internally captures both forward and backward graphs in order: `f0_fprop`, `f0_bprop`, `f1_fprop`, `f1_bprop`, `f2_fprop`, `f2_bprop`. However, the replay order differs—`f1_fprop` runs before `f0_bprop`. Since `f0_bprop` needs intermediate tensors saved during `f0_fprop`, but `f1_fprop` has already overwritten those tensors in the shared memory pool, the gradients are corrupted.

This issue is most common in **Pipeline Parallelism**, where multiple micro-batches are processed with overlapping forward and backward passes. Each micro-batch may have its own graphed callable, but they share a memory pool. The interleaved execution pattern (forward micro-batch 0, forward micro-batch 1, backward micro-batch 0, ...) naturally creates replay order mismatches with the sequential capture order.

**How to fix**:

**Solution 1: Match Replay Order to Capture Order**

```python
graphed_funcs = torch.cuda.make_graphed_callables(funcs, inputs)

# ✅ Replay in same order as capture
y0 = graphed_funcs[0](x0)  # f0_fprop
y0.backward()              # f0_bprop
y1 = graphed_funcs[1](x1)  # f1_fprop
y1.backward()              # f1_bprop
y2 = graphed_funcs[2](x2)  # f2_fprop
y2.backward()              # f2_bprop
```

**Solution 2: Use Separate Pools**

Each graph gets its own pool, so order doesn't matter. Trade-off: higher memory usage.

```python
# Capture each function separately with its own pool
g0 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0):
    y0 = func(x0)

g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):  # No pool= argument, gets its own pool
    y1 = func(x1)

# Now replay order doesn't matter
g1.replay()
g0.replay()
```

**Solution 3: Capture in Replay Order**

Capture graphs in the order you'll replay them. This requires manual graph construction instead of `make_graphed_callables`.

```python
# Replay order: f0_fprop, f1_fprop, f0_bprop, f2_fprop, f1_bprop, f2_bprop
# Capture in that order with shared pool

g0_fwd = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0_fwd):
    y0 = func(x0)

g1_fwd = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1_fwd, pool=g0_fwd.pool()):
    y1 = func(x1)

g0_bwd = torch.cuda.CUDAGraph()
with torch.cuda.graph(g0_bwd, pool=g0_fwd.pool()):
    y0.backward(retain_graph=True)

# Continue capturing in replay order...
```

### Parallel Replay: Concurrent Pool Access

**Symptom**: Corrupted results when graphs share a pool and are replayed on different streams simultaneously

**What it means**: Multiple graphs sharing a memory pool are replayed concurrently on different streams. They race to read/write shared memory addresses, causing corruption.

**Why it happens**: Memory pool sharing assumes sequential execution. When you replay graphs on separate streams without synchronization, they execute in parallel. Graph A might write to address X while Graph B is reading from X, or vice versa.

**Example**:

```python
import torch

# Make sure the kernel is long enough to observe race conditions
x1 = torch.ones(1024**3, device='cuda') * 1
x2 = torch.ones(1024**3, device='cuda') * 2
x3 = torch.ones(1024**3, device='cuda') * 3
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()
g3 = torch.cuda.CUDAGraph()
f1 = lambda x: x ** 2 + 2 * x + 1
f2 = lambda x: x ** 2 + 2 * x + 2
f3 = lambda x: x ** 2 + 2 * x + 3

# Warmup
for _ in range(3):
    y1 = f1(x1)
    y2 = f2(x2)
    y3 = f3(x3)
print(f"Correct: {y1[0].item()=}, {y2[0].item()=}, {y3[0].item()=}")

# Capture with shared pool
with torch.cuda.graph(g1):
    y1 = f1(x1)
with torch.cuda.graph(g2, pool=g1.pool()):
    y2 = f2(x2)
with torch.cuda.graph(g3, pool=g1.pool()):
    y3 = f3(x3)

# Replay in parallel on different streams
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()
s3 = torch.cuda.Stream()
current_stream = torch.cuda.current_stream()
s1.wait_stream(current_stream)
s2.wait_stream(current_stream)
s3.wait_stream(current_stream)

for _ in range(100):
    with torch.cuda.stream(s1):
        g1.replay()  # ❌ All three graphs replay simultaneously
    with torch.cuda.stream(s2):
        g2.replay()  # ❌ Racing for shared memory!
    with torch.cuda.stream(s3):
        g3.replay()  # ❌ Data corruption

current_stream.wait_stream(s1)
current_stream.wait_stream(s2)
current_stream.wait_stream(s3)
print(f"Actual: {y1[0].item()=}, {y2[0].item()=}, {y3[0].item()=}")
```

**Output**:
```text
Correct: y1[0].item()=4.0, y2[0].item()=10.0, y3[0].item()=18.0
Actual: y1[0].item()=13.0, y2[0].item()=12.0, y3[0].item()=19.0
```

**Why this is wrong**: Results are corrupted due to race conditions on shared memory buffers.

**How to fix**:

**Solution 1: Don't Share Pools for Concurrent Graphs**

```python
# Each graph gets its own pool
with torch.cuda.graph(g1):
    y1 = x1 ** 2 + 2 * x1 + 1
with torch.cuda.graph(g2):  # No pool parameter
    y2 = x2 ** 2 + 2 * x2 + 2
with torch.cuda.graph(g3):  # No pool parameter
    y3 = x3 ** 2 + 2 * x3 + 3

# ✅ Safe to replay in parallel
with torch.cuda.stream(s1):
    g1.replay()
with torch.cuda.stream(s2):
    g2.replay()
with torch.cuda.stream(s3):
    g3.replay()
```

**Solution 2: Synchronize Between Replays**

```python
# Sequential replay with shared pool
s1.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s1):
    g1.replay()
s2.wait_stream(s1)  # Wait for g1 to finish
with torch.cuda.stream(s2):
    g2.replay()
s3.wait_stream(s2)  # Wait for g2 to finish
with torch.cuda.stream(s3):
    g3.replay()
```

:::{warning} Pool Sharing Rule

**When sharing memory pools, keep replay order the same as capture order, and never replay concurrently.**

Safe patterns:

- ✅ Graphs replayed in the same order they were captured
- ✅ Sequential replay on the same stream
- ✅ Read graph output immediately after its replay, before other graphs replay

Unsafe patterns:

- ❌ Replay order differs from capture order (e.g., interleaved forward/backward in pipeline parallelism)
- ❌ Concurrent replay on different streams
- ❌ Reading a graph's output after another graph has replayed in different order (may be overwritten)
:::

---

## Debugging Numerical Errors

For systematic debugging of numerical errors, see [Debugging Strategies](debugging-strategies.md).

To isolate where numerical issues start, compare intermediate tensors between eager mode and graph replay layer by layer. Here is a simple example:

```python
import torch

# Enable determinism and fixed seed
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Run eager mode first
eager_intermediates = []
x = input.clone()
for layer in model.layers:
    x = layer(x)
    eager_intermediates.append(x.clone())

# Reset seed and capture graph
torch.manual_seed(42)
torch.cuda.manual_seed(42)
graph_intermediates = []
g = torch.cuda.CUDAGraph()
static_input = input.clone()

with torch.cuda.graph(g):
    x = static_input
    for layer in model.layers:
        x = layer(x)
        graph_intermediates.append(x.clone())
    graph_output = x

# Reset seed and replay graph
torch.manual_seed(42)
torch.cuda.manual_seed(42)
static_input.copy_(input)
g.replay()
torch.cuda.synchronize()

# Compare layer by layer
threshold = 1e-5
for i, (eager, graph) in enumerate(zip(eager_intermediates, graph_intermediates)):
    diff = (eager - graph).abs().max()
    print(f"Layer {i}: max diff = {diff:.2e}")
    if diff > threshold:
        print(f"Issue first appears at layer {i}")
        break
```

## Quick Reference

**Dynamic Behavior**:

- [ ] No Python `if/else` based on runtime values inside graph
- [ ] No capture-aware code (`is_current_stream_capturing()`) that changes behavior
- [ ] Input tensors are static and persistent—use `.copy_()` to update
- [ ] Python scalars converted to GPU tensors if they change
- [ ] Tensor shapes are fixed—no dynamic dimensions

**Race Conditions**:

- [ ] Synchronize before CPU writes to pinned memory (full sync or event-based)
- [ ] Use CUDA tensors instead of pinned memory for graph inputs when possible

**Input Tensor Management**:

- [ ] CUDA input tensors not freed or reassigned before replay
- [ ] CPU input tensors (for H2D copy) kept alive during graph lifetime
- [ ] Use `.copy_()` for input updates, not reassignment

**Output Tensor Management**:

- [ ] Clone graph outputs before storing if accumulating across replays

**CPU Code and State**:

- [ ] No host-side state mutations that affect code after graph

**Library and Hook Integration**:

- [ ] External libraries use PyTorch's current stream or sync properly
- [ ] Module hooks registered—but note they only fire during capture, not replay

**Memory Pool Sharing**:

- [ ] Replay order matches capture order when sharing pools
- [ ] No concurrent replay on different streams with shared pools

---

## What's Next?

- **[Capture Failures](capture-failures.md)**: When capture itself fails
- **[Memory Issues](memory-issues.md)**: OOM and memory-related problems
- **[Process Hang](process-hang.md)**: Debugging hanging or stuck processes
- **[Debugging Strategies](debugging-strategies.md)**: Systematic debugging approach
