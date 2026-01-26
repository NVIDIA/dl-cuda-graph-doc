<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Capture Failures

:::{note}

This section covers **explicit failures** that occur during CUDA graph capture—errors that prevent the graph from being created in the first place.
:::

## Understanding Capture Failures

When you attempt to capture a CUDA graph, it records the sequence of CUDA operations launched during the capture scope. This recording process imposes strict requirements: operations must be deterministic, asynchronous, and execute on properly synchronized streams. Any violation of these requirements causes **capture to fail immediately** with an explicit error message.

These failures are actually the **easier category** to debug compared to silent failures (see [Numerical Errors](numerical-errors.md)). The CUDA runtime provides clear error codes and PyTorch often gives helpful context about what went wrong. The challenge is understanding **why** these constraints exist and **how** to restructure your code to satisfy them.

### Why Capture Fails

PyTorch CUDA graphs work by recording a fixed sequence of GPU operations that can be replayed as a single unit. This fundamental design creates several requirements:

1. **No CPU-GPU synchronization**: Graphs capture GPU work only. Any operation that blocks the CPU waiting for GPU results (like `.item()` or `.cpu()`) cannot be captured because it would require CPU involvement during replay.

2. **Proper stream dependencies**: All streams used during capture must form a tree branching from the capture stream. This ensures the graph is self-contained and doesn't depend on external work.

3. **No forbidden operations**: Certain CUDA operations are incompatible with capture mode, such as querying event status from other threads.

4. **Avoid default stream**: The default stream (stream 0) has implicit synchronization behavior that makes it incompatible with CUDA graphs. All capture and graphed operations must run on explicit non-default streams.

### Common Error Codes

Understanding CUDA error codes helps diagnose issues quickly:

| Error Code | Name | Meaning |
|------------|------|---------|
| 900 | `cudaErrorStreamCaptureUnsupported` | Forbidden operation during capture |
| 901 | `cudaErrorStreamCaptureInvalidated` | Capture was interrupted by external work or thread |
| 904 | `cudaErrorStreamCaptureUnjoined` | Stream didn't rejoin before capture ended |
| 905 | `cudaErrorStreamCaptureIsolation` | Captured stream depends on uncaptured work |
| 906 | `cudaErrorStreamCaptureImplicit` | Operation would create forbidden dependency on default stream |

The sections below explain each failure mode in detail with examples and solutions.

---

## Synchronization Failures

CPU-GPU synchronization is one of the most frequent capture failures. Any operation that blocks the CPU waiting for GPU results will fail.

### Host-Device Sync: Forbidden Operations

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: Your code called an operation that synchronizes CPU with GPU, such as `.item()`, `.cpu()`, `torch.cuda.synchronize()`, or printing GPU tensor values. These operations block the CPU until GPU work completes, which cannot be captured.

**Why it happens**: CUDA graphs record GPU work only. During capture, the CPU is recording what operations to perform, not executing them. Synchronization operations need actual GPU results, which don't exist yet during capture.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    data = torch.randn(1, device="cuda")

    with torch.cuda.graph(graph, capture_error_mode="global"):
        result = data * 2
        value = result.item()  # ❌ Fails! Needs GPU result
        print(f"Result: {value}")
```

**Error message**:

> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing
> cudaErrorStreamCaptureInvalidated(901): operation failed due to a previous error during capture

**Common culprits**:

| Operation | Why it syncs | Alternative |
|-----------|-------------|-------------|
| `.item()` | Copies single value to CPU | Keep as tensor or move outside graph |
| `.cpu()` | Copies entire tensor to CPU | Use `.to('cpu', non_blocking=True)` outside graph |
| `print(tensor)` | Needs values for display | Log after graph replay |
| `.numpy()` | Converts to NumPy (CPU) | Convert outside graph |
| `torch.cuda.synchronize()` | Explicit sync | Remove or move outside |
| `if tensor > 0:` | Evaluates tensor as bool | Use `torch.where()` for GPU-side conditional |

**How to fix**:

```python
# ✅ Solution 1: Move sync outside graph
with torch.cuda.graph(graph):
    result = data * 2

# Sync happens here, outside graph
value = result.item()
print(f"Result: {value}")

# ✅ Solution 2: Keep everything on GPU
with torch.cuda.graph(graph):
    result = data * 2
    # Use GPU-side operations only
    final = result * 3  # No sync needed
```

**Why this works**: By separating GPU work (inside graph) from CPU operations (outside graph), you eliminate the synchronization points that prevent capture.

:::{note} Sync Detection

Use `torch.cuda.set_sync_debug_mode(1)` to detect synchronizations:

```python
torch.cuda.set_sync_debug_mode(1)  # Warn on syncs
# Or
torch.cuda.set_sync_debug_mode(2)  # Error on syncs
```

This helps identify hidden synchronizations in your code. See [Writing Sync-Free Code](../torch-cuda-graph/sync-free-code.md) for a comprehensive guide.
:::

---

## Memory Allocation and Deallocation

Memory operations like `cudaMalloc`, `cudaFree`, `cudaHostAlloc`, and `cudaFreeHost` have varying levels of support during CUDA graph capture. As covered in [CUDA Graph Constraints](../cuda-graph-basics/constraints.md#memory-allocation-apis), these APIs are generally forbidden during capture in Global and ThreadLocal modes. However, PyTorch's caching allocators implement special handling to work around these restrictions.

### Device Memory: Allocation Allowed, Free Skipped

**Behavior**: `cudaMalloc` is allowed during capture (PyTorch uses relaxed mode internally), but `cudaFree` is intentionally skipped.

**Why no error**: PyTorch's caching allocator is CUDA graph-aware. For `cudaMalloc`, PyTorch [wraps the call in relaxed capture mode](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L1003-L1014) to allow allocation. When you call `torch.cuda.empty_cache()` during capture, the allocator [checks if any capture is underway](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L3258-L3259) and skips the `cudaFree` calls entirely. This makes `empty_cache()` a safe no-op during capture rather than an error.

**Example**:

```python
import torch

def test():
    # Monkey patch torch.cuda.empty_cache as noop
    noop = lambda *args, **kwargs: None
    empty_cache = torch.cuda.empty_cache
    torch.cuda.empty_cache = noop

    graph = torch.cuda.CUDAGraph()
    x = torch.randn(8, device="cuda")
    del x

    # torch.cuda.graph invokes `torch.cuda.empty_cache` internally when entering the context,
    # but we've set it to noop to avoid recycle `x`
    with torch.cuda.graph(graph, capture_error_mode="global"):
        # Recover `torch.cuda.empty_cache`
        torch.cuda.empty_cache = empty_cache

        # Invokes cudaMalloc (allowed with relaxed mode internally)
        y = torch.randn(32, device="cuda")

        # This doesn't invoke cudaFree - skipped due to captures_underway check
        torch.cuda.empty_cache()

    # After capture: cudaFree is now allowed
    torch.cuda.empty_cache()
```

**Why this works**: The caching allocator tracks active captures via a [`captures_underway`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L1160-L1165) vector. In [`release_cached_blocks()`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L3258-L3259), the allocator checks if `captures_underway` is empty before calling `cudaFree`. If a capture is in progress, the free is skipped entirely.

:::{note} Key Insight

Device memory allocation during capture is safe because PyTorch's caching allocator ensures addresses allocated within the graph's private memory pool remain stable across replays. Deallocation is skipped during capture to prevent freeing memory that the graph references during replay.
:::

### Pinned Memory: Forbidden in Global Mode

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: Code attempted to allocate or free pinned (page-locked) host memory during capture with global mode. Operations like `cudaHostAlloc` and `cudaFreeHost` are not allowed.

**Why it happens**: Unlike device memory, pinned memory operations are not wrapped in relaxed capture mode by PyTorch. The [host caching allocator](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CachingHostAllocator.cpp) calls `cudaHostAlloc` directly without capture mode guards. Additionally, `cudaFreeHost` causes implicit device synchronization, which violates capture requirements.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    x = torch.randn(8*1024**2, device="cpu")

    # Global mode fails, relaxed mode succeeds
    with torch.cuda.graph(graph, capture_error_mode="global"):
        # Invokes cudaHostAlloc for page-locked memory
        y = x.pin_memory()  # ❌ Fails with global mode!
        z = torch.randn(8*1024**2, device="cpu", pin_memory=True)  # ❌ Fails!
        del y
        # Invokes cudaFreeHost to recycle page-locked memory
        torch._C._host_emptyCache()  # ❌ Fails!
```

**Error message**:

> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

**How to fix**:

```python
# ✅ Solution 1: Allocate pinned memory before capture
x = torch.randn(8*1024**2, device="cpu")
pinned_x = x.pin_memory()  # Allocate outside capture
pinned_buffer = torch.empty(8*1024**2, device="cpu", pin_memory=True)

with torch.cuda.graph(graph):
    # Use pre-allocated pinned memory
    gpu_data = pinned_x.to("cuda", non_blocking=True)

# ✅ Solution 2: Use relaxed capture mode (if pinned allocation is unavoidable)
with torch.cuda.graph(graph, capture_error_mode="relaxed"):
    y = x.pin_memory()  # Works with relaxed mode

# ✅ Solution 3: Avoid pinned memory in graphed code entirely
with torch.cuda.graph(graph):
    # Allocate directly on GPU instead
    data = torch.randn(8*1024**2, device="cuda")
```

**Why this works**: Pre-allocating pinned memory ensures `cudaHostAlloc` is called before capture begins. Relaxed mode allows the operation but may have implications for graph portability.

:::{warning} Host Allocator vs Device Allocator

Note that `torch.cuda.empty_cache()` only frees **device memory**, not pinned host memory. To free pinned memory, use `torch._C._host_emptyCache()`. These are separate caching allocators with different behaviors:

| Function | Frees | CUDA API |
|----------|-------|----------|
| `torch.cuda.empty_cache()` | GPU device memory | `cudaFree` |
| `torch._C._host_emptyCache()` | Pinned host memory | `cudaFreeHost` |
:::

:::{note} Event Queries in Pinned Memory Allocator

The pinned memory caching allocator uses [`cudaEventQuery()`](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CachingHostAllocator.cpp#L153-L162) to check when memory can be reused. This query happens during allocation (`process_events()` is [called in `allocate()`](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/core/CachingHostAllocator.h#L197-L199)), which can also cause capture failures since event queries are forbidden during capture.
:::

---

## Stream-Related Failures

Stream management is the most common source of capture failures. CUDA graphs require all streams to follow strict dependency rules.

### Inlet Stream: Missing Entry Synchronization

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: A side stream is used during capture, but it didn't wait for the capture stream before starting work. This violates the "tree structure" requirement—all streams must branch from the capture stream.

**Why it happens**: When you use a side stream inside the capture region, CUDA needs to know the dependency relationship. Without `stream.wait_stream(capture_stream)`, the side stream's work is independent, which is not allowed during capture.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    def func():
        # ❌ Missing: stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            y = x * 2
        torch.cuda.current_stream().wait_stream(stream)
        return x + y

    # warmup
    for _ in range(2):
        func()

    with torch.cuda.graph(graph, capture_error_mode="global"):
        func()  # Fails!
```

**Error message**:

> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

**How to fix**:

```python
def func():
    # ✅ Correct: side stream waits for capture stream
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = x * 2
    torch.cuda.current_stream().wait_stream(stream)
    return x + y
```

**Why this works**: `stream.wait_stream(capture_stream)` creates an explicit dependency, forming a tree structure where the side stream branches from the capture stream. This satisfies CUDA graph's requirement that all streams have well-defined entry points.

:::{tip} Stream Synchronization Pattern

The correct pattern for using side streams during capture is:

1. **Entry**: Side stream waits for capture stream
2. **Work**: Execute operations on side stream
3. **Exit**: Capture stream waits for side stream

This creates a "fork-join" structure that CUDA graphs can record.
:::

### Outlet Stream: Missing Exit Synchronization

**Error**: `cudaErrorStreamCaptureUnjoined (904)`

**What it means**: A side stream performed work during capture but didn't rejoin the capture stream before capture ended. The graph would have "dangling" work with no way to know when it completes.

**Why it happens**: CUDA graphs need to know the full dependency structure. If a stream doesn't rejoin, the graph can't guarantee completion of all work when it finishes replaying.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    with torch.cuda.graph(graph, capture_error_mode="global"):
        x.norm()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x.mean()
        # ❌ Missing: torch.cuda.current_stream().wait_stream(stream)
```

**Error message**:

> cudaErrorStreamCaptureUnjoined(904): capturing stream has unjoined work

**How to fix**:

```python
with torch.cuda.graph(graph, capture_error_mode="global"):
    x.norm()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.mean()
    # ✅ Correct: capture stream waits for side stream
    torch.cuda.current_stream().wait_stream(stream)
```

**Why this works**: The final `wait_stream()` ensures all side stream work is complete before the graph ends. This guarantees that when the graph finishes replaying, all operations—including those on side streams—have completed.

### Isolation: Dependency on External Work

**Error**: `cudaErrorStreamCaptureIsolation (905)`

**What it means**: The capture region tried to wait for work that was launched outside the capture. This violates the "self-contained" requirement—graphs cannot depend on external state.

**Why it happens**: CUDA graphs must be completely self-contained during replay. If the graph depends on external work, that work won't exist during replay, causing the graph to wait forever or execute incorrectly.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    x = torch.randn(128, device="cuda")

    # Work launched BEFORE capture
    with torch.cuda.stream(stream):
        x.mean()

    # Capture tries to wait for that work
    with torch.cuda.graph(graph, capture_error_mode="global"):
        x.norm()
        torch.cuda.current_stream().wait_stream(stream)  # ❌ Fails!
        x.norm()
```

**Error message**:

> cudaErrorStreamCaptureIsolation(905): dependency created on uncaptured work in another stream

**How to fix**:

```python
# ✅ Solution 1: Move external work outside the wait
with torch.cuda.stream(stream):
    x.mean()
torch.cuda.current_stream().wait_stream(stream)  # Wait BEFORE capture

with torch.cuda.graph(graph, capture_error_mode="global"):
    x.norm()

# ✅ Solution 2: Move all work inside capture
with torch.cuda.graph(graph, capture_error_mode="global"):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x.mean()
    torch.cuda.current_stream().wait_stream(stream)
    x.norm()

# ✅ Solution 3: Use external events for cross-graph synchronization
# External events create event nodes (not edges) in the graph,
# allowing graphs to wait on events recorded by other graphs.
sync_event = torch.cuda.Event(external=True)

g1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g1):
    # ... do work ...
    sync_event.record()  # Record node in g1

g2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(g2):
    sync_event.wait()    # Wait node in g2
    # ... work that depends on g1 ...
```

**Why this works**: Solutions 1 and 2 ensure the graph no longer depends on external state—either the dependency is resolved before capture, or all dependent work is inside the graph. Solution 3 uses external events to create explicit synchronization points between graphs: the wait node in g2 will block until the record node in g1 has executed during replay.

:::{warning} Isolation Principle

CUDA graphs must be **completely self-contained** unless using external events. They cannot:

- Wait for work launched before capture (unless via external events)
- Launch work that continues after capture ends
- Depend on state from other graphs or eager execution (unless via external events)

Think of graphs as hermetically sealed units that can be replayed anywhere, anytime—with external events providing the only bridge between them.
:::

---

## Initialization Failures

Certain PyTorch components must be initialized on a side stream to work with CUDA graphs.

### Gradient Tape: Default Stream Conflict

**Error**: `cudaErrorStreamCaptureImplicit (906)`

**What it means**: You're trying to capture a backward pass, but a leaf tensor's (e.g. parameter) gradient accumulation buffer (`AccumulateGrad` node) was created on the default stream. When you attempt to accumulate gradients on the default stream during capture, CUDA detects a forbidden dependency on the default stream.

**Why it happens**: PyTorch's autograd creates the `AccumulateGrad` node for a leaf tensor on the stream where the **first operation** on that tensor executes. In the example below, `y = y + 1` executes on the default stream, which creates `y`'s `AccumulateGrad` node on the default stream. Later, during graph capture on a side stream, `z.sum().backward()` tries to accumulate gradients w.r.t. all leaf nodes including `y`, attempting to write to `y.grad` which is associated with the default stream. CUDA sees this as the blocking default stream depending on the non-blocking capture stream—which is forbidden.

**Technical detail**: The `AccumulateGrad` node is created lazily during the first operation involving a `requires_grad=True` tensor. The node's stream affinity is set to the current stream at creation time. If created on the default stream, any subsequent gradient accumulation also happen on the default stream, preventing CUDA graph capture.

**Example**:

```python
import torch

def test():
    def func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = x + y
        return z * x

    x = torch.randn([2, 2], device="cuda", requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)

    # ❌ Default stream: First operation on y creates AccumulateGrad on default stream
    y = y + 1  # y's AccumulateGrad node created here on default stream

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        z = func(x, y)
        z.sum().backward()  # ❌ Fails! Tries to accumulate y.grad on side stream,
                            # but y's AccumulateGrad was created on default stream
```

**Error message**:

> cudaErrorStreamCaptureImplicit(906): operation would make the legacy stream depend on a capturing blocking stream

**How to fix**:

```python
# ✅ Correct: Initialize on side stream
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    y = y + 1  # grad buffer created on side stream
torch.cuda.current_stream().wait_stream(s)

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, capture_error_mode="global"):
    z = func(x, y)
    z.sum().backward()  # ✅ Works! Both on side stream
```

**Why this works**: By running `y = y + 1` on a side stream, `y`'s `AccumulateGrad` node is created on that side stream. When you later capture on a side stream, `backward()` accumulates gradients on the same side stream where the `AccumulateGrad` node was created, avoiding the forbidden cross-stream dependency with the default stream.

:::{tip} Warmup on Side Stream

The standard practice is to run warmup iterations on the same side stream where you'll capture:

```python
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    # Warmup - initializes all AccumulateGrad nodes
    for _ in range(3):
        z = func(x, y)
        z.sum().backward()

# Capture on same stream
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=s):
    z = func(x, y)
    z.sum().backward()
```

This ensures all `AccumulateGrad` nodes are created on the same stream used for capture.
:::

### DistributedDataParallel: Default Stream Conflict

**Error**: `cudaErrorStreamCaptureImplicit (906)`

**What it means**: DDP was initialized on the default stream, but you're trying to capture on a side stream. This causes the same cross-stream dependency issue as with gradient tape.

**Why it happens**: DDP registers gradient hooks on model parameters to perform all-reduce when gradients are ready. During DDP initialization, the [C++ Reducer constructor](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/reducer.cpp#L184) calls [`torch::autograd::impl::grad_accumulator()`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/variable.cpp#L285-L311) for each parameter to get or create `AccumulateGrad` nodes (in mixed precision mode, this is done via [`expand_as` in Python](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/nn/parallel/distributed.py#L1145-L1146) instead). These nodes are [associated with the current CUDA stream at creation time](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/input_metadata.cpp#L39). When DDP is initialized on the default stream, the `AccumulateGrad` nodes are created on the default stream. Later, when you try to capture backward passes on a side stream, the autograd engine attempts to execute these `AccumulateGrad` nodes on the default stream to accumulate gradients, causing a forbidden cross-stream dependency during capture.

**Example**:

```python
# torchrun --nproc_per_node=1 010.ddp.py

import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def test():
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.barrier()

    model = torch.nn.Linear(128, 128).cuda()
    x = torch.randn((32, 128), device="cuda")

    # ❌ DDP initialized on default stream
    model = DDP(model)

    # warmup
    for _ in range(15):
        model.zero_grad()
        y = model(x)
        y.sum().backward()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="global"):
        model.zero_grad()
        y = model(x)
        y.sum().backward()  # ❌ Fails! Cross-stream dependency
```

**Error message**:
> cudaErrorStreamCaptureImplicit(906): operation would make the legacy stream depend on a capturing blocking stream

**How to fix**:

```python
# ✅ Correct: Initialize DDP on side stream
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    model = DDP(model)

# Warmup on same side stream
with torch.cuda.stream(stream):
    for _ in range(15):
        model.zero_grad()
        y = model(x)
        y.sum().backward()

# Capture on same stream
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    model.zero_grad()
    y = model(x)
    y.sum().backward()  # ✅ Works!
```

**Why this works**: By initializing DDP on the side stream, operations like `expand_as` (which register DDP hooks) execute on that stream, creating `AccumulateGrad` nodes on the side stream. When you capture on the same stream, gradient accumulation happens on the same stream where the `AccumulateGrad` nodes were created, avoiding cross-stream dependencies with the default stream.

:::{warning} 11 Warmup Iterations for DDP

DDP requires **11 warmup iterations** (not 3) because it performs internal setup and logging around iteration 10. See [PyTorch Integration - DDP Setup](../torch-cuda-graph/torch-integration.md#ddp-distributeddataparallel-setup) for details.
:::

---

## Event and Stream Query Restrictions

CUDA forbids querying event or stream status during graph capture. These queries would require checking completion status, which conflicts with the capture mechanism that records operations without executing them.

### Direct Query: Forbidden Operations

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: Code attempted to query the status of a CUDA stream or event during capture. Operations like `stream.query()` and `event.query()` are forbidden because they check execution state, which is incompatible with graph recording.

**Why it happens**: During capture, CUDA records operations into a graph without actually executing them. Querying stream or event status requires checking whether work has completed, which doesn't make sense in the capture context. CUDA rejects these operations to prevent undefined behavior.

**Example**:

```python
import torch

def test():
    event = torch.cuda.Event()
    event.record()
    stream = torch.cuda.Stream()

    graph = torch.cuda.CUDAGraph()
    x = torch.randn(8, device="cuda")

    with torch.cuda.graph(graph):
        z = x + 1
        stream.query()  # ❌ Fails! Cannot query stream during capture
        event.query()   # ❌ Fails! Cannot query event during capture
        z = z * 2

    graph.replay()
```

**Error message**:

> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

**How to fix**:

```python
# ✅ Solution 1: Query before capture
event = torch.cuda.Event()
event.record()
stream = torch.cuda.Stream()

# Query status before entering capture
is_complete = stream.query()
event_done = event.query()

# Now safe to capture
with torch.cuda.graph(graph):
    z = x + 1

# ✅ Solution 2: Query after capture
with torch.cuda.graph(graph):
    z = x + 1
    # Don't query here

# Query after capture completes
is_complete = stream.query()
event_done = event.query()

# ✅ Solution 3: Avoid queries entirely in captured code
with torch.cuda.graph(graph):
    # Just perform computations, no status checks
    z = x + 1
```

**Why this works**: Moving stream/event queries outside the capture region allows CUDA to record the graph without attempting to check execution state.

### Background Thread Queries

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: Another thread queried a CUDA event or stream while capture was in progress on the main thread.

**Why it happens**: CUDA's capture mode is global by default (`capture_error_mode="global"`). When you capture on one thread, **all threads** in your process enter capture mode for that device. If any other thread queries an event or stream (even unrelated to the graph), it fails.

**Example**:

```python
import time
import threading
import queue
import torch

def worker(q):
    while True:
        event = q.get()
        event.query()  # ❌ Fails during capture!
        q.task_done()

def test():
    q = queue.Queue()
    event = torch.cuda.Event()
    event.record()
    threading.Thread(target=worker, args=(q,), daemon=True).start()

    graph = torch.cuda.CUDAGraph()
    x = torch.randn(8, device="cuda")
    y = torch.randn(8, device="cuda")

    with torch.cuda.graph(graph, capture_error_mode="global"):
        z = x + y
        q.put(event)  # Worker thread will query this event
        time.sleep(1)  # Give worker time to query
        z = z / y
    q.join()
```

**Error message**:

> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

**How to fix**:

```python
# ✅ Solution 1: Use thread-local capture mode
with torch.cuda.graph(graph, capture_error_mode="thread_local"):
    z = x + y
    q.put(event)
    time.sleep(1)
    z = z / y

# ✅ Solution 2: Stop background threads during capture
# Stop worker before capture
q.join()
# ... capture graph ...
# Restart worker after capture

# ✅ Solution 3: Coordinate to avoid queries during capture
# Ensure worker thread doesn't query events during capture window
```

**Why this works**: Thread-local mode restricts capture to the current thread only, allowing other threads to continue normal CUDA operations. Alternatively, coordinating threads to avoid event queries during capture eliminates the conflict.

:::{warning} PyTorch Limitations with thread_local

`capture_error_mode="thread_local"` may not work for backward passes, which PyTorch executes in a separate thread. For training graphs, you typically need `capture_error_mode="global"` and must ensure no other threads perform forbidden operations during capture.
:::

### Pinned Memory Allocation: Hidden Event Queries

**Error**: `cudaErrorStreamCaptureUnsupported (900)`

**What it means**: A background thread attempted to allocate pinned memory while capture was in progress on another thread, triggering hidden event queries.

**Why it happens**: PyTorch's pinned memory caching allocator uses `cudaEventQuery()` internally to track when pinned memory can be reused. When a background thread calls `.pin_memory()` during capture, this hidden event query fails because event queries are forbidden during capture.

**Example**:

```python
import time
import threading
import queue
import torch

def worker(q):
    while True:
        data = q.get()
        data.pin_memory()  # ❌ May query events!
        q.task_done()

def test():
    q = queue.Queue()
    threading.Thread(target=worker, args=(q,), daemon=True).start()

    graph = torch.cuda.CUDAGraph()
    x = torch.randn(8, device="cuda")

    # Setup that causes events to be tracked
    t1 = torch.randn(8).pin_memory()
    t2 = t1.to(device="cuda", non_blocking=True)  # Records event on stream
    del t1  # Will be freed when event completes

    data = torch.randn(8)

    with torch.cuda.graph(graph, capture_error_mode="global"):
        z = x + 1
        q.put(data)  # Worker will pin this, triggering event query
        time.sleep(1)
        z = z / 2
    q.join()
```

**Error message**:
> cudaErrorStreamCaptureUnsupported(900): operation not permitted when stream is capturing

**How to fix**:

```python
# ✅ Solution 1: Pin memory before capture
pinned_data = data.pin_memory()
with torch.cuda.graph(graph):
    gpu_data = pinned_data.to('cuda', non_blocking=True)

# ✅ Solution 2: Stop background threads during capture
q.join()  # Wait for worker to finish
with torch.cuda.graph(graph):
    # ... capture graph ...

# ✅ Solution 3: Use thread_local capture mode
with torch.cuda.graph(graph, capture_error_mode="thread_local"):
    # Other threads can continue pinned memory operations
    z = x + 1
```

**Why this works**: Pre-pinning avoids the event query during capture. Stopping background threads prevents concurrent pinning operations. Thread-local mode allows other threads to perform event queries.

### DataLoader Pin Memory Thread

**Error**: `cudaErrorStreamCaptureInvalidated (901)`

**What it means**: PyTorch's DataLoader with `pin_memory=True` spawns a background thread ([`_pin_memory_loop`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/utils/data/_utils/pin_memory.py#L18-L52)) that continuously pins batches. When this thread pins memory, the caching host allocator may call `cudaEventQuery()` to check if memory can be reused, which can invalidate an ongoing CUDA graph capture.

**Why it happens**: When iterating through a DataLoader with `pin_memory=True`, PyTorch spawns a dedicated thread to pin memory asynchronously. This thread runs concurrently with your main training loop. When the thread calls `.pin_memory()`, PyTorch's caching host allocator may call `cudaEventQuery()` to check if previously pinned memory can be reused. If this query happens while CUDA graph capture is in progress on another thread, it invalidates the capture.

**Example**:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

def test():
    # Create a simple dataset with large tensors to trigger pinning
    data = torch.randn(100, 1024*1024)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(data, labels)

    # DataLoader with pin_memory=True spawns a pin_memory thread
    loader = DataLoader(
        dataset,
        batch_size=2,
        pin_memory=True,  # ❌ Spawns background pin_memory thread
        num_workers=1,
    )

    # Capture graph while iterating - pin_memory thread runs concurrently
    for batch in loader:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="global"):
            torch.randn(64, device="cuda")  # ❌ May fail intermittently!

if __name__ == "__main__":
    test()
```

**Error message**:

> cudaErrorStreamCaptureInvalidated(901): operation failed due to a previous error during capture

:::{warning} Intermittent Failures

This failure is **intermittent** because it depends on timing. The pin_memory thread may or may not call `cudaEventQuery()` during the capture window. Whether the failure occurs depends on the timing relationship between the pin_memory thread and the capture. Large batch sizes and multi-worker DataLoaders increase the likelihood of collision.
:::

**How to fix**:

```python
# ✅ Solution 1: Use thread_local capture mode
for batch in loader:
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        torch.randn(64, device="cuda")

# ✅ Solution 2: Delay graph capture to let pin_memory thread finish its work
import time
for batch in loader:
    time.sleep(2)  # Wait for pin_memory thread to finish event queries
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.randn(64, device="cuda")

# ✅ Solution 3: Disable pin_memory
loader = DataLoader(dataset, batch_size=2, pin_memory=False)
```

**Why this works**: Thread-local mode allows the pin_memory thread to continue event queries without affecting capture. Adding a short delay before capture gives the pin_memory thread time to finish its current event queries. Disabling pin_memory eliminates the background thread entirely.

### NCCL Watchdog Thread

**What it is**: PyTorch's NCCL process group spawns background threads that monitor collective operations for timeouts and errors. These threads call `cudaEventQuery()` via [`finishedGPUExecutionInternal()`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L647-L658) to check operation completion status.

**Example** (would fail on older PyTorch without proper configuration):

```python
# Run with: torchrun --nproc_per_node=2 nccl_cuda_graph.py
import torch
import torch.distributed as dist

def test():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    x = torch.randn(1024, device="cuda")

    # On older PyTorch (< 2.2), the watchdog thread may query
    # CUDA events during capture, causing cudaErrorStreamCaptureUnsupported
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dist.all_reduce(x)

    graph.replay()
    dist.destroy_process_group()

if __name__ == "__main__":
    test()
```

**Historical evolution**: PyTorch's handling of NCCL threads and CUDA graph capture has evolved over time:

- **PyTorch 1.9-2.1 (2021-2023)**: The [official documentation](https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel) (introduced in [PR #63269](https://github.com/pytorch/pytorch/pull/63269)) recommended setting `NCCL_ASYNC_ERROR_HANDLING=0` to disable the work cleanup thread that would query events during capture. This worked but disabled async error detection. Note: This environment variable was renamed to `TORCH_NCCL_ASYNC_ERROR_HANDLING` in PyTorch 2.2 ([PR #114077](https://github.com/pytorch/pytorch/pull/114077)), though the old name is still supported for backward compatibility.

- **PyTorch 2.2-2.5 (2023-2025)**: Introduced a `pending_event_queries` counter mechanism via [PR #110665](https://github.com/pytorch/pytorch/pull/110665). Before starting capture, PyTorch would wait for all pending event queries to complete. Additionally, NCCL operations issued during capture were not enqueued to the watchdog, so no event queries would occur for them until after capture ended. This allowed async error handling to remain enabled while preventing capture conflicts.

- **PyTorch 2.6+ (2025)**: Simplified via [PR #148594](https://github.com/pytorch/pytorch/pull/148594). The watchdog now uses [`CUDAStreamCaptureModeGuard`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp#L2291) with `cudaStreamCaptureModeThreadLocal` before event queries, allowing the watchdog to query events even during capture without interference.

:::{tip} Handled Automatically in Recent PyTorch

For PyTorch 2.2+, no special configuration is needed—the coordination is handled automatically. For older versions, follow the [official PyTorch CUDA Graphs documentation](https://pytorch.org/docs/stable/notes/cuda.html#usage-with-distributeddataparallel) which recommends setting `NCCL_ASYNC_ERROR_HANDLING=0` when using CUDA graphs with DDP or NCCL operations.
:::

---

## Random Number Generator Failures

Random number generation requires special handling for CUDA graphs because the RNG offset is a CPU-side scalar that advances with each random operation. Without proper registration, the offset captured during graph capture becomes a frozen constant, causing every replay to produce identical random sequences.

### Custom Generator: Registration Required

**Error**: `RuntimeError: Attempt to increase offset for a CUDA generator not in capture mode`

**What it means**: Your code uses a custom `torch.Generator` for random operations, but you didn't register it with the graph before capture.

**Why it happens**: PyTorch uses the Philox algorithm for CUDA RNG, which requires a seed and an offset to determine the random sequence. The offset advances after each random operation. During graph capture, if the generator isn't registered, the offset is captured as a constant scalar in kernel parameters. Every replay uses this same frozen offset, producing identical random values.

**How registration fixes it**: When you call `register_generator_state()`, PyTorch stores the seed and offset in GPU tensors instead of passing them as scalar constants. Before each replay, PyTorch updates these GPU tensors with the current offset value, ensuring each replay gets fresh random numbers.

**Why PyTorch throws an error**: Note that this error is thrown by PyTorch, not by CUDA. The underlying issue—a dynamic scalar (the RNG offset) being captured as a constant—would otherwise cause silent numerical errors where every replay produces identical "random" values. PyTorch explicitly detects unregistered generators during capture and raises this error to prevent such hard-to-debug correctness issues.

For a detailed explanation of how CUDA RNG works with graphs and the internal implementation, see [Handling Dynamic Patterns - Random Number Generator State](../torch-cuda-graph/handling-dynamic-patterns.md#example-1-random-number-generator-state).

**Example**:

```python
import torch

def test():
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    graph = torch.cuda.CUDAGraph()

    # warmup
    for i in range(2):
        x = torch.randint(1024, (1,), generator=gen, device="cuda")
        print(f"{x.item()=}")

    # ❌ Forgot to register generator!
    # graph.register_generator_state(gen)

    # capture
    with torch.cuda.graph(graph):
        x = torch.randint(1024, (1,), generator=gen, device="cuda")  # ❌ Fails!
```

**Error message**:
> RuntimeError: Attempt to increase offset for a CUDA generator not in capture mode.

**How to fix**:

```python
# ✅ Register generator before capture
graph.register_generator_state(gen)

with torch.cuda.graph(graph):
    x = torch.randint(1024, (1,), generator=gen, device="cuda")
```

**Output**:
```text
Warmup 0: 840
Warmup 1: 565
Replay 0: 216
Replay 1: 574
Replay 2: 727
Replay 3: 457
```

**Why this works**: The generator's offset is now stored in GPU memory and updated during graph replay, producing different random values each time.

:::{note} Default Generator

PyTorch's default generator (`torch.cuda.default_generators[0]`) is automatically registered. You only need explicit registration for custom generators.

```python
# For default generator (automatic)
with torch.cuda.graph(g):
    x = torch.randn(10, device='cuda')  # Uses default generator

# For custom generator (requires registration)
gen = torch.Generator(device='cuda')
g.register_generator_state(gen)
with torch.cuda.graph(g):
    x = torch.randn(10, generator=gen, device='cuda')
```
:::

### RNG State Operations: Use Graph-Safe APIs

**Error**:

- `RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture` (from `get_rng_state()`)
- `RuntimeError: Please ensure to utilize the CUDAGeneratorImpl::set_state_index method during capturing` (from `set_rng_state()`)

**What it means**: Your code calls `torch.cuda.get_rng_state()` or `torch.cuda.set_rng_state()` during graph capture, which is not allowed.

**Why it happens**: The standard RNG state APIs (`get_rng_state()`, `set_rng_state()`) read and write CPU-side scalar values (seed and offset). If allowed during capture, these values would be frozen as constants in the graph, causing every replay to use the same RNG state and produce identical "random" sequences. PyTorch [explicitly blocks these operations](https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L302-L304) during capture to prevent such silent numerical errors.

**Example**:

```python
import torch

def test():
    graph = torch.cuda.CUDAGraph()

    # warmup
    for i in range(2):
        rng_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state)
        x = torch.randint(1024, (1,), device="cuda")

    # capture
    with torch.cuda.graph(graph):
        rng_state = torch.cuda.get_rng_state()  # ❌ Fails!
        torch.cuda.set_rng_state(rng_state)     # ❌ Fails too!
        x = torch.randint(1024, (1,), device="cuda")
```

**How to fix**:

Use PyTorch's graph-safe RNG state APIs instead:

- `generator.graphsafe_get_state()` instead of `torch.cuda.get_rng_state()`
- `generator.graphsafe_set_state()` instead of `torch.cuda.set_rng_state()`

```python
# ✅ Use graph-safe APIs
gen = torch.Generator(device="cuda")
graph.register_generator_state(gen)

with torch.cuda.graph(graph):
    state = gen.graphsafe_get_state()
    gen.graphsafe_set_state(state)
    x = torch.randint(1024, (1,), generator=gen, device="cuda")
```

These graph-safe APIs store the RNG state in GPU tensors that can be properly captured and replayed.

### Activation Checkpointing with RNG State Preservation

**Error**: `RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture`

**What it means**: Your model uses `torch.utils.checkpoint.checkpoint()` with `preserve_rng_state=True`, which internally calls `get_rng_state()` and `set_rng_state()` to save and restore RNG state around checkpointed regions.

**Why it happens**: Activation checkpointing with `preserve_rng_state=True` saves the RNG state before each checkpointed segment and restores it during recomputation in the backward pass. This ensures deterministic behavior for operations like dropout—the same random values are generated during both forward and recomputation. However, the underlying `get_rng_state()` call is blocked during CUDA graph capture (as explained above).

**Example**:

```python
import torch
import torch.utils.checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self, dim=512, num_blocks=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                block, x,
                use_reentrant=True,
                preserve_rng_state=True,  # ❌ Causes failure during capture
            )
        return x

def test():
    graph = torch.cuda.CUDAGraph()
    model = CheckpointedModel(dim=128).cuda()
    x = torch.randn((32, 128), device="cuda", requires_grad=True)

    # warmup
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        y = model(x)
        y.sum().backward()

    # capture
    with torch.cuda.graph(graph):
        y = model(x)
        y.sum().backward()  # ❌ Fails!
```

**How to fix**:

Disable RNG state preservation in checkpointing:

```python
x = torch.utils.checkpoint.checkpoint(
    block, x,
    use_reentrant=True,
    preserve_rng_state=False,  # ✅ Disable RNG state preservation
)
```

:::{warning} Trade-off

Setting `preserve_rng_state=False` means dropout and other random operations may produce different values during forward and recomputation, potentially affecting numerical results. If your checkpointed blocks don't contain random operations (like dropout), this is safe. If they do, consider removing dropout from checkpointed regions or accepting the non-determinism.
:::

For more details on activation checkpointing and RNG state preservation, see [PyTorch Checkpoint Documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html#torch-utils-checkpoint).

### Partial Graphing with Reentrant Activation Checkpointing

**Error**: `RuntimeError: When use_reentrant=True, torch.utils.checkpoint is incompatible with .grad() or passing an 'inputs' parameter to .backward()`

**What it means**: You're using `torch.cuda.make_graphed_callables()` on a model that contains reentrant activation checkpointing (`use_reentrant=True`).

**Why it happens**: `make_graphed_callables()` internally uses `torch.autograd.grad()` to compute gradients for the captured backward graph. However, reentrant checkpointing (`use_reentrant=True`) is incompatible with `autograd.grad()` due to how it manages the autograd tape during recomputation.

**Example**:

```python
import torch
import torch.utils.checkpoint

class CheckpointedModel(torch.nn.Module):
    def __init__(self, dim=512, num_blocks=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
            ) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(
                block, x,
                use_reentrant=True,  # ❌ Incompatible with make_graphed_callables
                preserve_rng_state=False,
            )
        return x

def test():
    model = CheckpointedModel(dim=128).cuda()
    x = torch.randn((32, 128), device="cuda", requires_grad=True)

    model = torch.cuda.make_graphed_callables(model, (x,))  # ❌ Fails!

    y = model(x)
    y.sum().backward()
```

**How to fix**:

Use non-reentrant checkpointing:

```python
x = torch.utils.checkpoint.checkpoint(
    block, x,
    use_reentrant=False,  # ✅ Compatible with make_graphed_callables
    preserve_rng_state=False,
)
```

:::{note} Not a CUDA Graph Error

This error is not caused by CUDA graph capture itself, but by a PyTorch-level API incompatibility between reentrant checkpointing and `autograd.grad()`. The same error would occur even without CUDA graphs if you tried to use `torch.autograd.grad()` with reentrant checkpointing.
:::

:::{note} Reentrant vs Non-Reentrant Checkpointing

Non-reentrant checkpointing (`use_reentrant=False`) is the recommended approach in modern PyTorch. It supports `autograd.grad()`, works correctly with `torch.compile`, and handles keyword arguments. The reentrant variant is considered legacy and may be deprecated in future versions. See [PyTorch Checkpoint Documentation](https://docs.pytorch.org/docs/2.9/checkpoint.html#torch.utils.checkpoint.checkpoint) for detailed differences between the two variants.
:::

### torch.compile During Graph Capture

**Error**: `torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Cannot call CUDAGeneratorImpl::current_seed during CUDA graph capture`

**What it means**: You're calling a `torch.compile`-decorated function inside a CUDA graph capture context without warming it up first.

**Why it happens**: When `torch.compile` compiles a function for the first time, TorchDynamo's [`preserve_global_state`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/_dynamo/convert_frame.py#L265-L333) decorator saves and restores the RNG state to ensure compilation doesn't affect the random sequence. This involves calling `torch.cuda.get_rng_state()`, which is forbidden during CUDA graph capture.

**Example**:

```python
import torch

@torch.compile
def func(x):
    return x**2 + 3*x + 5

def test():
    x = torch.randn(8, device="cuda")
    graph = torch.cuda.CUDAGraph()

    # ❌ Without warmup, torch.compile triggers compilation during capture
    with torch.cuda.graph(graph):
        func(x)  # ❌ Fails!
```

**How to fix**:

Warm up the compiled function before capture to trigger compilation:

```python
x = torch.randn(8, device="cuda")
graph = torch.cuda.CUDAGraph()

# ✅ Warmup triggers compilation outside capture
func(x)

with torch.cuda.graph(graph):
    func(x)  # ✅ Already compiled, no RNG state access
```

:::{tip} General Rule

Always warm up `torch.compile`-decorated functions before CUDA graph capture. The warmup triggers JIT compilation, which may access RNG state and perform other operations incompatible with capture. Once compiled, subsequent calls use the cached compiled code and work correctly within graph capture.
:::

---

## Quick Reference

**Synchronization**:

- [ ] No `.item()`, `.cpu()`, `.numpy()` calls
- [ ] No `print(tensor)` or logging that accesses tensor values
- [ ] No `torch.cuda.synchronize()` calls

**Memory Allocation**:

- [ ] No pinned memory allocation (`cudaHostAlloc`, `.pin_memory()`)
- [ ] Device memory allocation allowed (relaxed mode)

**Stream Management**:

- [ ] All side streams have entry sync: `side_stream.wait_stream(capture_stream)`
- [ ] All side streams have exit sync: `capture_stream.wait_stream(side_stream)`
- [ ] No waits for work launched before capture (unless using external events)
- [ ] For cross-graph synchronization: use `torch.cuda.Event(external=True)`

**Initialization**:

- [ ] DDP constructed on side stream
- [ ] Gradient-requiring tensors initialized on side stream
- [ ] Warmup iterations before capture (3+, or 11 for DDP)

**Event and Stream Queries**:

- [ ] No `stream.query()` or `event.query()` during capture
- [ ] No background threads querying events (or use thread-local mode)
- [ ] DataLoader: use `pin_memory=False` or delay capture
- [ ] NCCL watchdog: handled automatically in PyTorch 2.2+

**Random Number Generation**:

- [ ] Custom generators registered: `graph.register_generator_state(gen)`
- [ ] Use graph-safe APIs: `graphsafe_get_state()`, `graphsafe_set_state()`
- [ ] Activation checkpointing: `preserve_rng_state=False`
- [ ] Partial graphing: `use_reentrant=False`
- [ ] `torch.compile` functions warmed up before capture

---

## What's Next?

- **[Numerical Errors](numerical-errors.md)**: When capture succeeds but results are wrong
- **[Memory Issues](memory-issues.md)**: Detailed memory troubleshooting
- **[Process Hang](process-hang.md)**: Debugging hanging or stuck processes
- **[Debugging Strategies](debugging-strategies.md)**: Systematic debugging approach
