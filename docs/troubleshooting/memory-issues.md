<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Memory Issues

:::{note}

This section covers memory-related issues when using CUDA graphs, especially illegal memory access, segmentation faults, and out-of-memory errors.
:::

---

## Illegal Memory Access and Segmentation Faults

**Symptom**: Illegal memory access errors, segmentation faults, or corrupted outputs during graph replay.

**What it means**: The graph is accessing memory addresses that are no longer valid—either freed, reallocated for other purposes, or never properly allocated.

**Why it happens**: CUDA graphs capture memory addresses at capture time and replay operations using those exact addresses. If the underlying tensors are deallocated before replay, the graph accesses freed memory. This commonly occurs when:

- **Input tensors go out of scope**: A tensor created inside a function is garbage-collected after the function returns, but the graph still references its memory address.
- **Tensor reassignment instead of in-place update**: Using `x = new_tensor` creates a new tensor at a different address, leaving the graph pointing to the old (now potentially freed) address.
- **CPU tensors for H2D copies are freed**: The graph captures a host-to-device copy with a specific source address, but the CPU tensor is garbage-collected before replay.

The result depends on timing and memory reuse patterns:

- **Illegal memory access**: The memory was freed and returned to the system
- **Segmentation fault**: Severe memory corruption or accessing unmapped pages
- **Corrupted outputs**: The memory was reused by another tensor, so the graph reads unrelated data (this is the most insidious case—no error, just wrong results)

**How to fix**: See [Deconstructed Input Tensors](numerical-errors.md#deconstructed-input-tensors) for detailed examples and solutions. The key principles are:

1. **Keep input tensors persistent**: Ensure all tensors used as graph inputs remain alive for the entire graph lifetime
2. **Use in-place updates**: Update tensor contents with `.copy_()`, `.fill_()`, or other in-place operations instead of reassigning variables
3. **Return or store input tensors**: If capturing inside a function, return the input tensors along with the graph, or store them in a class/module

---

## Understanding Caching Allocator Statistics

PyTorch uses a [caching allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) to manage GPU memory. Instead of calling `cudaMalloc` and `cudaFree` for every tensor allocation and deallocation, it maintains pools of cached memory blocks:

- **Allocation**: The allocator first tries to find a suitable cached block. If the cached block is larger than requested, it splits the block and uses part of it. Only if no suitable block exists does it request new memory from CUDA via `cudaMalloc`.
- **Deallocation**: When you delete a tensor, its memory returns to the cache rather than being immediately freed to CUDA. The allocator tries to merge adjacent free blocks to reduce fragmentation.

This caching and splitting strategy dramatically reduces the overhead of frequent allocations.

**Pool and Stream Isolation**: The caching allocator organizes memory by **pools** and **streams**. Each memory pool (global pool, or CUDA graph's private pool) maintains its own set of cached blocks, and within a pool, blocks are further organized by the stream they were allocated on. When searching for a free block, the allocator only considers blocks from the same pool that match the allocation stream (see [`get_free_block()`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L2957)). This means **cached memory cannot be directly reused across different pools or streams**—a block freed on stream A cannot satisfy an allocation request on stream B, even if the memory is available. This isolation ensures correctness (avoiding race conditions) but can lead to higher memory usage when workloads span multiple streams or pools.

The allocator tracks several key statistics via [`torch.cuda.memory_stats()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html):

- **Reserved**: Total memory obtained from CUDA via `cudaMalloc`. This is the allocator's "working set"—memory it has reserved for potential use.
- **Allocated**: Memory currently assigned to tensors. When you create a tensor, allocated increases; when you delete it, allocated decreases.
- **Active**: Memory for tensors that are still in use. This differs from allocated when a tensor is deleted but its memory cannot be immediately recycled (e.g., due to `record_stream()` or pending events on other streams).
- **Inactive Split**: Cached memory that cannot be returned to CUDA because it's part of a larger block that was split—this represents fragmented memory within reserved blocks. For example, if a 4GB block is split to satisfy a 1GB request, the remaining 3GB becomes inactive split—it's available for future allocations but cannot be freed via `cudaFree` until the 1GB piece is also freed.

**Example**: The following code demonstrates how these statistics change during allocation, deallocation, and cross-stream usage:

```python
import torch

stream = torch.cuda.Stream()

x1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB on default stream
del x1
x2 = torch.empty(1*1024**3, dtype=torch.uint8, device="cuda")  # 1GB, splits x1's block
del x2

x3 = torch.empty(1*1024**3, dtype=torch.uint8, device="cuda")  # 1GB on default stream
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    x3.record_stream(stream)  # Mark x3 as used on `stream`
    del x3                    # Memory deferred until event completes
    t1 = torch.empty(1, dtype=torch.uint8, device="cuda")  # Triggers event processing
    del t1
    x4 = torch.empty(1*1024**3, dtype=torch.uint8, device="cuda")  # 1GB on `stream`
    torch.cuda.empty_cache()  # Release unused cached memory
    x5 = torch.empty(1*1024**3, dtype=torch.uint8, device="cuda")  # 1GB on `stream`
torch.cuda.current_stream().wait_stream(stream)
```

**Output**:

```text
         Tag (GB)      Allocated         Active  InactiveSplit       Reserved
-----------------------------------------------------------------------------
   After alloc x1          4.000          4.000          0.000          4.000
     After del x1          0.000          0.000          0.000          4.000
   After alloc x2          1.000          1.000          3.000          4.000
     After del x2          0.000          0.000          0.000          4.000
   After alloc x3          1.000          1.000          3.000          4.000
     After del x3          0.000          1.000          3.000          4.000
   After alloc t1          0.000          0.000          0.000          4.000
   After alloc x4          1.000          1.000          0.000          5.000
After empty cache          1.000          1.000          0.000          1.000
   After alloc x5          2.000          2.000          0.000          2.000
```

**Explanation**:

1. **After alloc x1**: Allocator requests 4GB from CUDA. All 4GB is allocated, active, and reserved.

2. **After del x1**: `x1` is deleted. Allocated and active drop to 0, but reserved stays at 4GB—the memory is cached for reuse.

3. **After alloc x2**: A 1GB allocation is requested. The allocator splits the cached 4GB block into 1GB (for `x2`) and 3GB (remaining). The 3GB becomes **inactive split**—it's cached but cannot be returned to CUDA until the 1GB piece is also freed.

4. **After del x2**: `x2` is deleted. The allocator merges the 1GB and 3GB fragments back into a single 4GB block. Inactive split returns to 0.

5. **After alloc x3**: Same as step 3—the 4GB block is split again.

6. **After del x3**: Here's the key difference. `x3` called `record_stream(stream)` before deletion, indicating it's used on another stream. When `del x3` is called, the allocator records a CUDA event on each stream that was registered via `record_stream()`. The memory can only be recycled once all these events complete (i.e., when those streams finish their work on `x3`). So while allocated drops to 0, **active stays at 1GB**. The memory is in a "limbo" state: not allocated to any tensor, but not yet available for reuse.

7. **After alloc t1**: `t1` is a tiny allocation (1 byte). At the beginning of each allocation, the allocator calls [`process_events()`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L3567) which queries outstanding CUDA events via `cudaEventQuery()`. If an event has completed, the associated block's event count is decremented; when the count reaches zero, the block is finally freed. In this case, the allocator detects that `stream` has completed its work on `x3`. Now the 1GB can be merged with the 3GB fragment, restoring the full 4GB block. Active and inactive split both return to 0.

8. **After alloc x4**: A 1GB allocation on `stream`. Although x3's 4GB block is now free, it was allocated on the **default stream**, not `stream`. Due to stream isolation, the allocator cannot reuse this block for `x4`. It must request new memory from CUDA—Reserved jumps from 4GB to 5GB.

9. **After empty cache**: `torch.cuda.empty_cache()` releases all unused cached memory back to CUDA. The 4GB block from the default stream (not used by any tensor) is freed. Reserved drops from 5GB to 1GB (only x4's block remains).

10. **After alloc x5**: Another 1GB allocation on `stream`. The allocator requests new memory from CUDA—Reserved grows from 1GB to 2GB.

This example illustrates why `Active > 0` when `Allocated = 0` can occur with multi-stream usage, how fragmentation (inactive split) accumulates when blocks are split, how stream isolation can cause additional memory allocations even when free memory exists, and how `empty_cache()` can reclaim unused memory across all streams.

**Further Reading**:

- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html#torch-cuda-memory): PyTorch provides tools for understanding and debugging GPU memory usage, including `torch.cuda.memory._record_memory_history()` for capturing memory snapshots and visualizing allocation patterns over time.
- [Optimizing Memory Usage with `PYTORCH_CUDA_ALLOC_CONF`](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf): The caching allocator behavior can be tuned via the `PYTORCH_CUDA_ALLOC_CONF` environment variable. Options include `max_split_size_mb` (to reduce fragmentation by preventing large blocks from being split), `expandable_segments` (to allow more flexible memory reclamation), and `garbage_collection_threshold` (to trigger proactive memory cleanup).

---

## Out of Memory (OOM)

OOM errors with CUDA graphs typically stem from seven main causes related to how the caching allocator manages memory (see [Understanding Caching Allocator Statistics](#understanding-caching-allocator-statistics) above):

1. **Static input tensors can't be freed**: Graph inputs must be static tensors that persist throughout the graph's lifetime, which usually means allocating additional input tensors that can't be freed
2. **Intermediate tensors can't be reused across memory pools**: Different graphs use their private memory pools to ensure address stability, and intermediate tensors reserved in one pool cannot be reused by another pool
3. **Intermediate tensors after capture can't reuse graph pool memory**: Temporary tensors allocated after capture go to the global pool and cannot reuse memory cached in the graph's private pool
4. **Memory fragmentation across pools**: Fragmented allocations across pools (global pool and graph's private pool) cannot be consolidated and reused
5. **Deferred memory recycling**: Memory used by multiple streams during capture cannot be freed until after capture completes
6. **Gradient accumulator cross-stream memory growth**: When gradient accumulators are created on a different stream than capture, gradients cannot be recycled during capture
7. **`cudaFree` is suppressed during capture**: Cached memory cannot be released back to CUDA during capture, preventing OOM recovery that would normally work outside capture

The following subsections cover common OOM scenarios and solutions.

### Static Input Tensors Can't Be Freed

**Symptom**: OOM when using multiple CUDA graphs, even though each graph individually fits in memory.

**Why it happens**: When using multiple CUDA graphs (e.g., partial capturing where different model sections are graphed separately), each graph typically requires its own static input tensors. These tensors must persist throughout the graph's lifetime and receive data via `.copy_()` from the original input (e.g., dataloader output). With many graphs, this memory overhead accumulates.

**Example**: The following code creates 10 CUDA graphs, each with its own static input tensor. Even though each graph individually fits in memory, the cumulative memory for all static inputs can cause OOM:

```python
import torch

def test():
    # Simulate partial capturing with multiple graphs
    # Each graph has its own static input tensor
    graphs = []
    static_inputs = []
    static_outputs = []

    for i in range(10):
        # Each graph needs its own static input - memory adds up!
        static_input = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB each
        static_inputs.append(static_input)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = static_input * 2
        graphs.append(graph)
        static_outputs.append(output)

    # With 10 graphs × 4GB static inputs = 40GB just for inputs!
```

**How to fix**:

**Solution 1: Reuse Static Input Tensors Across Graphs**

If multiple graphs process the same input data, share a single static input tensor instead of allocating one per graph:

```python
# ✅ Share static input when graphs use the same input
static_input = torch.empty_like(x)

# Capture both graphs with static_input
# Replay
static_input.copy_(x)
graph1.replay()
graph2.replay()  # reuses same static_input
```

**Solution 2: Share Static Input Buffer Across Sequential Graphs**

If multiple graphs don't run concurrently and the static input data is not needed after replay, those graphs can share a common static input buffer. If the shapes differ, allocate the maximum size needed:

```python
# ✅ Share a single buffer for graphs that run sequentially
max_size = max(input1.numel(), input2.numel())
shared_buffer = torch.empty(max_size, device="cuda")

# Capture each graph using views into the shared buffer
static_input1 = shared_buffer[:input1.numel()].view_as(input1)
static_input2 = shared_buffer[:input2.numel()].view_as(input2)

# Replay sequentially - buffer is reused
static_input1.copy_(input1)
graph1.replay()
# static_input1 no longer needed, safe to overwrite buffer
static_input2.copy_(input2)
graph2.replay()
```

**Solution 3: Chain Graph Outputs as Inputs**

When graphs represent adjacent layers, use the output of one graph directly as the input to the next, avoiding extra static input allocations:

```python
# ✅ Use graph1's output as graph2's input during capture
static_input = torch.empty_like(x)

# Capture graph1
graph1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph1):
    out1 = layer1(static_input)

# Capture graph2 using out1 (graph1's output) as its input
graph2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph2):
    out2 = layer2(out1)  # out1 is already a static tensor from graph1

# Replay - no intermediate copy needed between graphs
static_input.copy_(x)
graph1.replay()
graph2.replay()  # uses out1 directly, no copy required
```

**Solution 4: Reduce Number of Graphs**

Instead of capturing many small graphs, capture fewer larger graphs. This reduces the number of static input tensors needed. For example, use full-iteration graphing (like [`FullCudaGraphWrapper`](../torch-cuda-graph/te-megatron-cuda-graphs.md#fullcudagraphwrapper-full-iteration-graphing)) instead of per-layer graphing when memory is constrained.

### Intermediate Tensors Can't Be Reused Across Memory Pools

**Symptom**: OOM when using multiple CUDA graphs, even though each graph's intermediate memory could theoretically be reused.

**Why it happens**: By default, PyTorch allocates a separate private memory pool for each CUDA graph to ensure memory addresses remain constant across replays. **This means intermediate tensors allocated within one graph's pool cannot be reused by another graph**—each graph maintains its own isolated memory space. When you have multiple graphs, this isolation prevents the caching allocator from consolidating and reusing memory across graphs, leading to higher overall memory consumption.

**Example**: The following code creates two graphs, each with its own memory pool. Although we explicitly `del intermediate` during capture, its memory is still required during replay—the graph records the memory addresses used during capture and replays operations at those same addresses. So each memory pool must retain:

- **Intermediate tensor**: The Python reference is deleted after capture, but the memory remains **reserved** in the pool because the graph needs it during replay
- **Output tensor**: Still alive and used both during and after replay

Because each graph has its own pool, graph2 cannot reuse graph1's intermediate memory:

```python
import torch

def test():
    x1 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB
    x2 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB

    # Graph 1 with its own memory pool
    graph1 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph1):
        intermediate1 = x1 * 2   # ~4GB in graph1's pool
        out1 = intermediate1 + 1  # ~4GB in graph1's pool
        del intermediate1  # Allow memory reuse within this pool

    # Graph 2 with its own memory pool
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        # Cannot reuse graph1's pool memory!
        intermediate2 = x2 * 3   # ~4GB in graph2's pool
        out2 = intermediate2 + 2  # ~4GB in graph2's pool
        del intermediate2  # Allow memory reuse within this pool
```

**Output** (tracking large pool stats at key points):

```text
               Tag (GB)      Allocated         Active  InactiveSplit       Reserved
-----------------------------------------------------------------------------------
     After alloc x1, x2          8.000          8.000          0.000          8.000
After del intermediate1         12.000         12.000          0.000         16.000
After del intermediate2         16.000         16.000          0.000         24.000
```

The output shows memory accumulating across separate pools:

1. **After alloc x1, x2**: 8GB allocated in the global pool for the two input tensors.
2. **After del intermediate1**: Graph1's private pool reserves 8GB (4GB for `intermediate1` + 4GB for `out1`). Even though we deleted `intermediate1`, its memory stays reserved for replay. Total Reserved jumps to 16GB (8GB global + 8GB graph1's pool).
3. **After del intermediate2**: Graph2's private pool reserves another 8GB. It cannot reuse graph1's freed `intermediate1` memory because each pool is isolated. Total Reserved reaches 24GB (8GB global + 8GB graph1 + 8GB graph2).

**How to fix**:

**Solution 1: Share Memory Pools Across Graphs**

Multiple graphs can share the same memory pool if they execute sequentially (not concurrently). Use the `pool` parameter to specify a shared pool:

```python
x1 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB
x2 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB

# ✅ Graph 1 creates its memory pool
graph1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph1):
    intermediate1 = x1 * 2
    out1 = intermediate1 + 1
    del intermediate1  # Allow memory reuse

# ✅ Graph 2 shares graph1's memory pool
graph2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph2, pool=graph1.pool()):
    intermediate2 = x2 * 3  # Can reuse intermediate1's memory
    out2 = intermediate2 + 2
    del intermediate2  # Allow memory reuse

# Replay sequentially - shared pool is safe
graph1.replay()
graph2.replay()

# Total: ~20GB (2 inputs + 2 outputs + 1 shared intermediate)
```

:::{warning} Shared Pool Constraints

When graphs share a memory pool, they must be replayed in the same order as captured. Out-of-order replay can cause memory corruption. See [Shared Memory Pool Corruption](numerical-errors.md#shared-memory-pool-corruption) for details.
:::

**Solution 2: Reduce Number of Graphs**

Fewer graphs mean fewer separate memory pools. Capture larger portions of your model in a single graph:

```python
x1 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB
x2 = torch.randn(1024, 1024, 1024, device="cuda")  # ~4GB

# ✅ Single graph with one pool
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    intermediate1 = x1 * 2
    out1 = intermediate1 + 1
    del intermediate1  # Memory can be reused for intermediate2
    intermediate2 = x2 * 3
    out2 = intermediate2 + 2
    del intermediate2

graph.replay()

# Total: ~20GB (2 inputs + 2 outputs + 1 shared intermediate)
```

### Intermediate Tensors After Capture Can't Reuse Graph Pool Memory

**Symptom**: Higher peak memory when allocating temporary tensors after CUDA graph capture, compared to allocating them before capture.

**Why it happens**: This is a special case of the cross-pool isolation described above. When you allocate and free temporary tensors *before* capture, `torch.cuda.graph()` internally calls `empty_cache()` when entering the capture context, which releases that cached memory back to CUDA. The graph's private pool can then request fresh memory from CUDA, effectively "reusing" that space. However, if you allocate temporary tensors *after* capture completes, they go into the global pool and cannot reuse memory cached in the graph's private pool—that memory must remain reserved for graph replay.

**Example**: The following code allocates a temporary tensor `t1` before capture and another `t2` inside capture. After capture, allocating `t3` in the global pool cannot reuse `t2`'s cached memory from the graph's private pool:

```python
import os
import torch
from contextlib import nullcontext

USE_GRAPH = bool(int(os.environ.get("USE_GRAPH", "0")))
graph = torch.cuda.CUDAGraph()
ctx = torch.cuda.graph(graph) if USE_GRAPH else nullcontext()

x1 = torch.randn(1024, 1024, 1024, device="cuda")  # 4GB
t1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB temporary
del t1  # t1's memory is now cached in global pool

with ctx:
    # torch.cuda.graph() calls empty_cache() here, releasing t1's memory
    t2 = x1 * 2  # 4GB intermediate in graph's pool
    out1 = t2 + 1  # 4GB output in graph's pool
    del t2  # t2's memory stays cached in graph's pool

t3 = torch.randn(1024, 1024, 1024, device="cuda")  # 4GB - can it reuse t2's memory?
del t3
```

**Output without graph capture** (`USE_GRAPH=0`):

```text
                Tag (GB)      Allocated         Active  InactiveSplit       Reserved
------------------------------------------------------------------------------------
  After alloc x1, del t1          4.000          4.000          0.000          8.000
     After enter context          4.000          4.000          0.000          8.000
After alloc out1, del t2          8.000          8.000          0.000         12.000
  After alloc t3, del t3          8.000          8.000          0.000         12.000
```

**Output with graph capture** (`USE_GRAPH=1`):

```text
                Tag (GB)      Allocated         Active  InactiveSplit       Reserved
------------------------------------------------------------------------------------
  After alloc x1, del t1          4.000          4.000          0.000          8.000
     After enter context          4.000          4.000          0.000          4.000
After alloc out1, del t2          8.000          8.000          0.000         12.000
  After alloc t3, del t3          8.000          8.000          0.000         16.000
```

The comparison reveals the key difference:

1. **After alloc x1, del t1**: Both cases show 4GB allocated (for `x1`) and 8GB reserved (4GB `x1` + 4GB cached from `t1`).

2. **After enter context**: This is where the paths diverge:
    - **Without capture**: Reserved stays at 8GB—nothing special happens.
    - **With capture**: `torch.cuda.graph()` calls `empty_cache()`, releasing `t1`'s cached 4GB back to CUDA. Reserved drops to 4GB.

3. **After alloc out1, del t2**:
    - **Without capture**: 12GB reserved, all in the global pool (4GB `x1` + 4GB `t2` cached + 4GB `out1`).
    - **With capture**: 12GB reserved across two pools (4GB `x1` in global pool + 8GB for `t2` and `out1` in graph's private pool).

4. **After alloc t3, del t3**: Needs 4GB for `t3`:
    - **Without capture**: `t3` reuses `t2`'s cached memory in the global pool. Reserved stays at 12GB.
    - **With capture**: `t3` must go to the global pool, but `t2`'s memory is cached in the graph's private pool and cannot be reused. The allocator requests fresh memory from CUDA. Reserved grows to 16GB.

**Consequence**: With capture, peak Reserved is 16GB vs 12GB without capture. The 4GB difference comes from the cross-pool isolation: `t2`'s cached memory in the graph's pool cannot satisfy `t3`'s allocation in the global pool.

This is a common issue in partial graphing scenarios. For example, if you capture forward and backward passes in a CUDA graph but leave the optimizer step in eager mode, the optimizer may allocate temporary tensors (e.g., for gradient scaling or momentum updates) that go into the global pool. These allocations cannot reuse the intermediate memory cached in the graph's private pool, resulting in higher peak memory than running the entire training step in eager mode.

**How to fix**:

**Solution 1: Expand Capture Range**

Include the operations that allocate temporary tensors within the CUDA graph capture. For example, if the optimizer step runs in eager mode and allocates temporary tensors, capture the optimizer along with forward and backward passes.

Note: This is a simplified example. In practice, optimizers may contain operations that are not directly compatible with CUDA graphs, such as conditional logic (e.g., skipping steps based on gradient values). You may need to refactor such logic to make them graph-compatible:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(static_input)
    loss = criterion(output, static_target)
    loss.backward()
    optimizer.step()  # ✅ Now optimizer's temporaries are in graph's pool
    optimizer.zero_grad()
```

This way, all intermediate tensors share the same memory pool and can reuse each other's memory.

**Solution 2: Reduce Temporary Tensor Allocations**

Use kernel fusion to eliminate intermediate tensor allocations. `torch.compile` can automatically fuse operations:

```python
@torch.compile
def fused_optimizer_step(params, grads):
    # Fused operations avoid separate temporary allocations
    for p, g in zip(params, grads):
        p.data.add_(g, alpha=-lr)
```

Alternatively, use fused optimizer implementations (e.g., `torch.optim.AdamW(fused=True)`) or custom CUDA kernels that perform multiple operations without materializing intermediate tensors

### Memory Fragmentation Across Pools

**Symptom**: OOM or higher-than-expected memory usage, even though `torch.cuda.memory_summary()` shows significant free memory in the cache.

**Why it happens**: Even with a single CUDA graph, you have two memory pools: the global (default) memory pool used by eager execution and the graph's private memory pool. PyTorch's caching allocator caches previously freed memory blocks and splits them to satisfy smaller allocation requests. However, the remaining fragments in one pool cannot be reused by another pool. Over time, this leads to fragmentation where free memory is scattered across pools in unusable small pieces.

For example, if the global pool has an 8GB cached block and you allocate 1GB from it, the remaining 7GB stays in the global pool but is now split around the 1GB allocation. If the graph's private pool then needs memory, it cannot use the global pool's free fragments—it must allocate fresh memory, potentially causing OOM.

With multiple CUDA graphs (each with its own private pool), the fragmentation issue becomes more severe. Each additional pool adds another isolated memory region that cannot share fragments with other pools, compounding the memory overhead.

**Example**: The following shows how fragmentation between the global pool and a graph's private pool leads to higher memory usage:

```python
import torch

def test():
    # Allocate large block in global pool, then free - caching allocator keeps it
    temp = torch.randn(1024, 1024, 2048, device="cuda")  # 8GB in global pool
    del temp  # Freed but cached in global pool

    # Allocate smaller tensors - splits the cached 8GB block
    small = torch.randn(256, 1024, 1024, device="cuda")  # 1GB in global pool
    x = torch.randn(256, 1024, 1024, device="cuda")      # 1GB in global pool

    # Now capture a graph - uses its own private pool
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):  # calls empty_cache() internally
        # Graph's private pool cannot reuse global pool's 6GB free fragments!
        intermediate = x * 2  # 1GB allocated in private pool
        out = intermediate + 1  # 1GB allocated in private pool
        del intermediate  # Allow memory reuse within private pool

    # Global pool: 2GB used (small + x), 6GB free but fragmented
    # Graph's private pool: 1GB used (out), 1GB reserved for intermediate during replay
```

**Output** (tracking large pool stats at key points):

```text
              Tag (GB)      Allocated         Active  InactiveSplit       Reserved
----------------------------------------------------------------------------------
        After del temp          0.000          0.000          0.000          8.000
         After alloc x          2.000          2.000          6.000          8.000
     After empty cache          2.000          2.000          6.000          8.000
After del intermediate          3.000          3.000          6.000         10.000
```

The output shows how fragmentation develops:

1. **After del temp**: The 8GB block is freed but cached (Reserved = 8GB). No fragmentation yet (InactiveSplit = 0).
2. **After alloc x**: Allocating `small` and `x` (2GB total) splits the cached 8GB block. The remaining 6GB becomes inactive split—cached but fragmented around the 2GB allocation.
3. **After empty cache**: `torch.cuda.graph()` internally calls `empty_cache()` before capture, but this doesn't help—`empty_cache()` only releases inactive memory blocks that haven't been split. Once a cached block is fragmented (split to satisfy smaller allocations), the remaining fragments cannot be released back to CUDA until all fragments from that block are freed. The 6GB inactive split remains in the global pool, usable within that pool but inaccessible to the graph's private pool.
4. **After del intermediate**: The graph's private pool allocates 2GB for `intermediate` and `out`. Since it cannot reuse the global pool's 6GB fragments, it must allocate fresh memory from CUDA. Reserved jumps to 10GB (8GB global + 2GB private).

Without CUDA graph capture, the same operations would use only the global pool. Both `intermediate` and `out` tensors could reuse the 6GB free fragments from `temp`. Reserved memory would stay at ~8GB instead of 10GB.

**How to fix**:

**Solution 1: Enable Expandable Segments**

PyTorch's [expandable segments](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) feature uses CUDA's low-level virtual memory APIs (`cuMemCreate`, `cuMemAddressReserve`, `cuMemMap`) to create memory segments differently from traditional `cudaMalloc`:

- **Virtual address reservation**: On first allocation, PyTorch reserves a large virtual address space (up to 1⅛× total GPU memory) without allocating physical memory.
- **On-demand physical mapping**: Physical memory is allocated and mapped in pages (2MB for small pool, 20MB for large pool) only when needed.
- **Page-level unmapping**: When memory is freed and `empty_cache()` is called (or during OOM recovery), unused physical pages can be individually unmapped via `cuMemUnmap` and released via `cuMemRelease`, returning them to CUDA for reuse by any pool.

Enable it via environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

With expandable segments enabled, the same example produces:

```text
              Tag (GB)      Allocated         Active  InactiveSplit       Reserved
----------------------------------------------------------------------------------
        After del temp          0.000          0.000          0.000          8.008
         After alloc x          2.000          2.000          0.000          8.008
     After empty cache          2.000          2.000          0.000          2.012
After del intermediate          3.000          3.000          0.000          4.023
```

Key differences from the default behavior:

1. **After alloc x**: InactiveSplit stays at 0 because expandable segments don't create traditional split blocks—they manage memory at page granularity.
2. **After empty cache**: Reserved drops from ~8GB to ~2GB. When `torch.cuda.graph()` calls `empty_cache()`, unused physical pages can be individually unmapped via `cuMemUnmap` and returned to CUDA, even though `small` and `x` are still allocated.
3. **After del intermediate**: Reserved is only ~4GB (2GB global + 2GB private) instead of 10GB. The graph's private pool obtains fresh physical pages from CUDA rather than being blocked by the global pool's reservations.

This eliminates cross-pool fragmentation at the cost of slightly slower initial allocation (due to virtual memory mapping overhead) and incompatibility with IPC-based multi-process data loading (use `expandable_segments:False` for DataLoader workers if needed).

However, expandable segments cannot reclaim memory *during* CUDA graph capture. CUDA graph capture records the exact memory addresses used, so if memory were freed/unmapped during capture, the recorded addresses would become invalid, causing crashes or corruption during replay. PyTorch's allocator enforces this by skipping memory release operations when capture is in progress, and a graph's private pool is only eligible for `empty_cache()` after the graph is destroyed.

The following example demonstrates this behavior:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    temp = torch.randn(1024, 1024, 2048, device="cuda")  # 8GB
    del temp

    x = torch.randn(512, 1024, 1024, device="cuda")   # 2GB
    y = torch.randn(1024, 1024, 2048, device="cuda")  # 8GB
    del x, y

    z = torch.randn(1024, 1024, 4096, device="cuda")  # 16GB
    del z

    torch.cuda.empty_cache()  # Does nothing during capture!

del graph
torch.cuda.empty_cache()  # NOW memory can be released
```

Output (with `expandable_segments:True`):

```text
         Tag (GB)      Allocated         Active  InactiveSplit       Reserved
-----------------------------------------------------------------------------
   After del temp          0.000          0.000          0.000          8.008
    After alloc x          2.000          2.000          0.000          8.008
    After alloc y         10.000         10.000          0.000         10.000
   After del x, y          0.000          0.000          0.000         10.000
    After alloc z         16.000         16.000          0.000         16.016
      After del z          0.000          0.000          0.000         16.016
After empty cache          0.000          0.000          0.000         16.016
  After del graph          0.000          0.000          0.000         16.016
After empty cache          0.000          0.000          0.000          0.000
```

Key observations:

- **Reserved memory grows but never shrinks during capture**: Even with expandable segments, Reserved increases from 8GB → 10GB → 16GB as larger tensors are allocated, and stays at 16GB despite `del` and `empty_cache()` calls.
- **InactiveSplit stays at 0**: Expandable segments manage memory at page granularity, avoiding traditional block splitting.
- **Memory is only released after graph destruction**: The final `empty_cache()` after `del graph` returns Reserved to 0GB.

This means the peak memory during capture determines the private pool's reservation. To minimize memory usage, avoid allocating large temporary tensors during capture, or restructure your code to reduce peak allocation.

**Solution 2: Limit Block Splitting with `max_split_size_mb`**

The [max_split_size_mb](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) option prevents the allocator from splitting blocks larger than the specified size. When a large block (≥ `max_split_size_mb`) is freed, it remains as a single unsplit block in the cache. This prevents fragmentation from accumulating in large allocations:

```bash
# Prevent splitting blocks >= 128MB
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

With `max_split_size_mb:128`, the same example produces:

```text
              Tag (GB)      Allocated         Active  InactiveSplit       Reserved
----------------------------------------------------------------------------------
        After del temp          0.000          0.000          0.000          8.000
         After alloc x          2.000          2.000          0.000         10.000
     After empty cache          2.000          2.000          0.000          2.000
After del intermediate          3.000          3.000          0.000          4.000
```

Key differences from the default behavior:

1. **After alloc x**: InactiveSplit stays at 0 because the 8GB block is never split. Instead, `small` and `x` trigger new 1GB allocations (Reserved = 10GB total).
2. **After empty cache**: Reserved drops from 10GB to 2GB. When `torch.cuda.graph()` calls `empty_cache()`, the unsplit 8GB block can be returned to CUDA via `cudaFree`.
3. **After del intermediate**: Reserved is only 4GB (2GB global + 2GB private) instead of 10GB.

How it helps:

- **Without `max_split_size_mb`**: A freed 8GB block can be split into smaller pieces (e.g., 1GB + 7GB). The 7GB remainder becomes "inactive split" memory that cannot be returned to CUDA until the 1GB piece is also freed.
- **With `max_split_size_mb:128`**: Blocks ≥ 128MB are never split. When `temp` (8GB) is deleted, it stays as a single unsplit block. Before graph capture, `torch.cuda.graph()` calls `empty_cache()`, which returns this 8GB block to CUDA. The graph's private pool can then allocate from CUDA directly.

The tradeoff is that unsplit blocks can only satisfy requests within a small rounding tolerance of their size, potentially leading to more `cudaMalloc` calls for varied allocation sizes. Finding the optimal value requires experimentation—try different values or use [torch.cuda.memory._record_memory_history()](https://pytorch.org/docs/stable/torch_cuda_memory.html) to profile allocation patterns and identify appropriate thresholds.

**Solution 3: Share Memory Pools Across Graphs**

If using multiple sequential graphs, share pools to consolidate memory and reduce cross-pool fragmentation:

```python
graph1 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph1):
    out1 = layer1(x)

# Share pool with graph1
graph2 = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph2, pool=graph1.pool()):
    out2 = layer2(x)
```

:::{warning} Shared Memory Pool Constraints

When graphs share a memory pool, they must be replayed in the same order as captured. Out-of-order replay can cause memory corruption. See [Shared Memory Pool Corruption](numerical-errors.md#shared-memory-pool-corruption) and [Replay Order Mismatch](numerical-errors.md#replay-order-mismatch) for details.
:::

### Deferred Memory Recycling

**Symptom**: OOM or higher-than-expected memory usage during CUDA graph capture, even though tensors have been deleted.

**Why it happens**: When a tensor is used on multiple streams via [`record_stream()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html), deleting it doesn't immediately free its memory. Instead, the caching allocator records events on all streams where the tensor was used and only recycles the memory once all events complete. Normally, this event processing happens at the beginning of the next allocation.

However, **during CUDA graph capture, event queries (`cudaEventQuery`) are illegal**, so the allocator cannot check whether events have completed. As a result, memory recycling is **deferred until capture ends**. This means tensors marked with `record_stream()` accumulate in memory throughout the capture, even after being deleted.

**Example**: The following code demonstrates how `record_stream()` defers memory recycling during capture. We compare the same workload with and without graph capture (`USE_GRAPH=0` vs `USE_GRAPH=1`):

```python
import os
import torch

USE_GRAPH = bool(int(os.environ.get("USE_GRAPH", "0")))

graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
ctx = torch.cuda.graph(graph) if USE_GRAPH else torch.cuda.stream(stream)

with ctx:
    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    x1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB

    # Use x1 on a side stream
    side_stream.wait_stream(current_stream)
    with torch.cuda.stream(side_stream):
        x1.record_stream(side_stream)  # Mark x1 as used on side_stream
        x1.fill_(1)
    current_stream.wait_stream(side_stream)

    del x1  # Memory recycling depends on whether we're capturing

    t1 = torch.empty(1, dtype=torch.uint8, device="cuda")  # Triggers event processing (if not capturing)
    del t1

    x2 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB - can we reuse x1's memory?

t2 = torch.empty(1, dtype=torch.uint8, device="cuda")  # After capture ends
```

**Output without graph capture** (`USE_GRAPH=0`):

```text
      Tag (GB)      Allocated         Active  InactiveSplit       Reserved
--------------------------------------------------------------------------
After alloc x1          4.000          4.000          0.000          4.000
  After del x1          0.000          4.000          0.000          4.000
After alloc t1          0.000          0.000          0.000          4.000
After alloc x2          4.000          4.000          0.000          4.000
After alloc t2          4.000          4.000          0.000          4.000
```

**Output with graph capture** (`USE_GRAPH=1`):

```text
      Tag (GB)      Allocated         Active  InactiveSplit       Reserved
--------------------------------------------------------------------------
After alloc x1          4.000          4.000          0.000          4.000
  After del x1          0.000          4.000          0.000          4.000
After alloc t1          0.000          4.000          0.000          4.000
After alloc x2          4.000          8.000          0.000          8.000
After alloc t2          4.000          4.000          0.000          8.000
```

The comparison reveals the deferred recycling behavior:

1. **After del x1**: In both cases, Allocated drops to 0 but Active stays at 4GB because x1 was marked with `record_stream()`. The memory is pending recycling, waiting for the event to complete.

2. **After alloc t1**: This is the key difference:

    - **Without capture**: The allocation triggers [`process_events()`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L3567), which queries the event and finds it complete. x1's memory is recycled—Active drops from 4GB to 0.
    - **With capture**: Event queries are illegal during capture, so `process_events()` is skipped. x1's memory remains Active at 4GB.

3. **After alloc x2**: Needs 4GB:

    - **Without capture**: Reuses x1's recycled memory. Reserved stays at 4GB.
    - **With capture**: x1's memory is still Active (deferred), so the allocator must request new memory from CUDA. Reserved doubles to 8GB.

4. **After alloc t2**: Capture has ended, so deferred frees are processed. Active drops from 8GB to 4GB as x1's memory is finally recycled. However, Reserved remains at 8GB—the peak memory usage during capture.

**Consequence**: Without capture, Reserved stays at 4GB throughout. With capture, deferred recycling causes Reserved to grow to 8GB because x1's memory couldn't be reused. In workloads with more cross-stream tensors, this memory overhead can accumulate significantly.

**How to fix**:

**Solution 1: Avoid `record_stream()` with Proper Synchronization**

The [`record_stream()` documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html) states: *"you must manually ensure that any non-creation stream uses of a tensor are synced back to the creation stream before you deallocate the tensor."* If you follow this pattern—synchronizing back to the creation stream before deletion—you don't need `record_stream()` at all, because the synchronization guarantees the tensor's usage is complete.

```python
with torch.cuda.graph(graph):
    current_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()

    x1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")

    # Use x1 on side_stream
    side_stream.wait_stream(current_stream)
    with torch.cuda.stream(side_stream):
        # ❌ x1.record_stream(side_stream)  # Not needed!
        x1.fill_(1)
    current_stream.wait_stream(side_stream)  # ✅ Sync back before deletion

    del x1  # Safe to delete - synchronization ensures x1 is no longer in use
    x2 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # Can reuse x1's memory
```

By synchronizing the side stream back to the capture stream (`current_stream.wait_stream(side_stream)`), you guarantee that all work on `x1` is complete before any subsequent operations. This makes `record_stream()` unnecessary, avoiding deferred recycling entirely.

:::{warning} Limitation

This approach only works for tensors **allocated within the graph's memory pool** during capture. Tensors allocated outside the capture context (e.g., static inputs) are in the global pool, and the graph's private pool cannot reclaim their memory regardless of synchronization.
:::

**Solution 2: Enable `graph_capture_record_stream_reuse`**

This approach is essentially the same as Solution 1—both rely on the graph DAG topology to determine when memory can be safely reused. The difference is that Solution 1 requires you to manually remove `record_stream()` calls and ensure proper synchronization, while this option **automatically detects** when streams have rejoined and memory can be recycled, without requiring code changes. Instead of relying on CUDA events (which are illegal during capture), it uses the captured graph's DAG topology to determine when memory can be safely reused.

PyTorch 2.9.0+ provides this experimental allocator option:

```bash
PYTORCH_CUDA_ALLOC_CONF=graph_capture_record_stream_reuse:True python your_script.py
```

With this option enabled on the same example:

```text
      Tag (GB)      Allocated         Active  InactiveSplit       Reserved
--------------------------------------------------------------------------
After alloc x1          4.000          4.000          0.000          4.000
  After del x1          0.000          4.000          0.000          4.000
After alloc t1          0.000          0.000          0.000          4.000
After alloc x2          4.000          4.000          0.000          4.000
After alloc t2          4.000          4.000          0.000          4.000
```

Now x1's memory is recycled during capture, and Reserved stays at 4GB—matching the non-capture behavior.

**How it works**: When a tensor with `record_stream()` is freed during capture, the allocator inserts "empty nodes" into the graph DAG as markers indicating where the tensor's usage ends on each stream. Before each allocation, the allocator performs a reverse DFS traversal of the graph to find empty nodes that are predecessors of **all current terminal nodes**. If a freed block's markers satisfy this condition, the block can be safely reused—any new work will be guaranteed to occur after the block's previous usage. See [`free_safe_blocks_in_capture()`](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L1827) for implementation details.

**Trade-offs**:

- **Pros**: Reduces peak memory during long captures with multi-stream usage, especially when streams frequently synchronize (join) during capture.
- **Cons**: Significantly increases capture time due to the graph traversal overhead (O(nodes) per allocation). Only helps when streams converge; if streams remain diverged throughout capture, blocks cannot be reused.

For more details, see [Optimizing Memory Usage with `PYTORCH_CUDA_ALLOC_CONF`](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).

:::{warning} Limitation

Like Solution 1, this only works for tensors **allocated within the graph's memory pool**. Tensors allocated before capture (in the global pool) cannot be reclaimed by the graph's private pool, regardless of this setting.
:::

### Gradient Accumulator Cross-Stream Memory Growth

**Symptom**: Memory grows during CUDA graph capture when the `AccumulateGrad` node was created on a different stream than the capture stream, even though tensors are being properly deleted.

**Why it happens**: This is a special case of [Deferred Memory Recycling](#deferred-memory-recycling). The key is that [`AccumulateGrad`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/functions/accumulate_grad.cpp#L90-L93) nodes capture the **current stream during initialization** as their canonical stream. Common cases where this happens include:

- Registering backward hooks via `param.expand_as(param)` (e.g., for DDP/FSDP gradient synchronization) on a different stream
- Running warmup iterations on a different stream than capture

During CUDA graph capture, if gradients are **produced** on the capture stream but **consumed** (accumulated) by an `AccumulateGrad` node whose canonical stream is the initialization stream, the autograd engine detects this cross-stream flow. To ensure correctness—preventing the gradient from being freed before actual accumulation happens—the engine calls `record_stream()` on the gradient tensor. However, during capture, event queries are illegal, so these gradients cannot be recycled until capture completes. With **gradient accumulation**, this issue is more severe because each micro-batch backward pass allocates new gradients that cannot be freed, causing memory to grow linearly with the number of accumulation steps.

The root cause is in how `AccumulateGrad` nodes capture their stream. When you call `param.expand_as(param)` to access the gradient accumulator, the [`InputMetadata` constructor](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/input_metadata.cpp#L38-L39) captures the current stream during initialization:

```cpp
// input_metadata.cpp:38-39
stream_ = c10::impl::getDeviceGuardImpl(device_.type())->getStream(device_);
```

During backward, the autograd engine uses this stored stream as the "consumer stream" for gradient accumulation. When the producer stream (where gradients are computed during capture) differs from the consumer stream (where the `AccumulateGrad` was initialized), the [`InputBuffer::add()`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/input_buffer.cpp#L242-L246) method calls `record_stream()`:

```cpp
// input_buffer.cpp:242-246
if (*opt_consumer_stream != *opt_producer_stream) {
    record_stream_any_impl(var, *opt_consumer_stream);
}
```

This marks the gradient tensor for multi-stream use to prevent premature freeing. When the gradient is freed, the allocator [defers recycling](https://github.com/pytorch/pytorch/blob/v2.9.0/c10/cuda/CUDACachingAllocator.cpp#L1909-1921) because event queries are illegal during capture.

**Example**: The following code demonstrates this issue. The `AccumulateGrad` node is initialized on `side_stream`, but capture runs on a different stream (`torch.cuda.graph()` uses its own internal stream for capture):

```python
import torch

def step(x, accum=5):
    x.grad = None
    for i in range(accum):
        y = (x * 2).sum()
        y.backward()
    x.grad = None

def backward_hook(*args):
    """Empty hook - just having it registered causes the issue"""
    pass

x = torch.randn(1024, 1024, 1024, device='cuda', requires_grad=True)  # 4GB
side_stream = torch.cuda.Stream()

# Warmup on side stream - creates AccumulateGrad with side_stream as canonical stream
with torch.cuda.stream(side_stream):
    x_expanded = x.expand_as(x)  # Creates AccumulateGrad node on side_stream
    grad_acc = x_expanded.grad_fn.next_functions[0][0]
    grad_acc.register_hook(backward_hook)
    step(x)

torch.cuda.current_stream().wait_stream(side_stream)

# Capture - different stream from AccumulateGrad's canonical stream
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    step(x)
```

**Output** (tracking large pool stats):

```text
      Tag (GB)      Allocated         Active  InactiveSplit       Reserved
--------------------------------------------------------------------------
 Before warmup          4.000          4.000          0.000          4.000
        Step 0          8.000          8.000          0.000          8.000
        Step 1          8.000          8.000          0.000         12.000
        Step 2          8.000          8.000          0.000         12.000
        Step 3          8.000          8.000          0.000         12.000
        Step 4          8.000          8.000          0.000         12.000
  After warmup          4.000          4.000          0.000         12.000
Before capture          4.000          4.000          0.000          4.000
        Step 0          8.000          8.000          0.000          8.000
        Step 1          8.000         12.000          0.000         12.000
        Step 2          8.000         16.000          0.000         16.000
        Step 3          8.000         20.000          0.000         20.000
        Step 4          8.000         24.000          0.000         24.000
 After capture          4.000         24.000          0.000         24.000
```

**Explanation**:

1. **Warmup (Steps 0-4)**: During warmup on `side_stream`, the `AccumulateGrad` node is created with `side_stream` as its canonical stream. Gradients are allocated and recycled normally—Reserved stabilizes at 12GB (4GB for `x` + 4GB for `x.grad` + 4GB for intermediate tensor like `x * 2`). Active equals Allocated throughout warmup, indicating normal memory recycling.

2. **Before capture**: `torch.cuda.graph()` calls `empty_cache()` when entering the capture context. Since the cached 8GB (gradient + intermediate from warmup) is in unsplit blocks, it can be released back to CUDA. Reserved drops from 12GB to 4GB (only `x` remains).

3. **Capture (Step 0)**: The first backward pass allocates 4GB for the intermediate tensor (`x * 2`), which is then reused for the gradient. Reserved grows to 8GB. Since this is the first gradient and `x.grad` is not yet defined, [`AccumulateGrad::accumulateGrad()`](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/autograd/functions/accumulate_grad.h#L179) directly assigns the gradient without accumulation—no cross-stream `record_stream()` is triggered yet.

4. **Capture (Steps 1-4)**: Starting from Step 1, **Active grows while Allocated stays constant**. This is the key indicator of deferred recycling:

    - Gradients are **produced** on the capture stream (`torch.cuda.graph()` uses its own internal stream)
    - Gradients are **consumed** by `AccumulateGrad` whose canonical stream is `side_stream` (from initialization)
    - The stream mismatch triggers `record_stream()` on each gradient tensor to prevent premature freeing
    - During capture, `process_events()` is skipped, so these gradients cannot be recycled
    - Each backward frees the old gradient (Allocated stays at 8GB) but the memory remains Active (grows by 4GB per step)
    - The allocator must request fresh memory from CUDA for each new gradient

5. **After capture**: Allocated drops to 4GB (only `x`), but Active remains at 24GB—all the deferred gradient memory is still waiting for event processing. Reserved stays at 24GB, six times what it would be if the `AccumulateGrad` was initialized on the same stream as capture. Note that this memory is allocated in the graph's private pool, so it cannot be freed via `empty_cache()` after capture—the pool must remain reserved for graph replay.

Note that this example uses 5 gradient accumulation steps (`accum=5`). In real training scenarios with more accumulation steps (e.g., 8 or 16), the memory growth would be proportionally worse—each micro-batch backward adds another unreclaimable gradient tensor.

**How to fix**:

**Solution 1: Initialize `AccumulateGrad` on the Same Stream as Capture**

The simplest fix is to ensure the `AccumulateGrad` nodes are initialized on the same stream where capture will occur. When the `AccumulateGrad`'s canonical stream matches the capture stream, the autograd engine sees no stream mismatch during backward—gradients are produced and consumed on the same stream, so `record_stream()` is never called. Without `record_stream()`, gradient memory can be recycled immediately within the graph's memory pool, just like normal tensor reuse.

In practice, this means initializing DDP/FSDP and running warmup on the same stream that will be used for capture. Since DDP/FSDP register backward hooks during initialization, the `AccumulateGrad` nodes inherit the current stream at that time.

```python
# ✅ Initialize gradient accumulator and capture on the same stream
capture_stream = torch.cuda.Stream()

with torch.cuda.stream(capture_stream):
    # Initialize gradient accumulator on capture_stream
    x_expanded = x.expand_as(x)
    grad_acc = x_expanded.grad_fn.next_functions[0][0]
    grad_acc.register_hook(backward_hook)

    # Warmup on the same stream
    for _ in range(warmup_iters):
        step(x)

# Capture on the same stream - AccumulateGrad's stream matches
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=capture_stream):
    step(x)
```

:::{warning} Trade-off: Gradient Accumulation Cannot Overlap with Backward

When all operations run on a single stream, gradient accumulation (the `AccumulateGrad` nodes) executes sequentially with backward computation. In contrast, when `AccumulateGrad` runs on a different stream, it can potentially overlap with ongoing backward computation on the main stream. The performance impact of losing this overlap depends on your workload—profile your application to determine whether memory savings or compute overlap is more beneficial.
:::

**Solution 2: Enable `graph_capture_record_stream_reuse`**

If you cannot easily change the warmup stream (e.g., in complex distributed training setups), enable the allocator option that allows memory reuse during capture:

```bash
PYTORCH_CUDA_ALLOC_CONF=graph_capture_record_stream_reuse:True python your_script.py
```

This allows the allocator to recycle memory marked with `record_stream()` during capture by using graph DAG topology instead of event queries. See [Deferred Memory Recycling](#deferred-memory-recycling) for details.

### `cudaFree` Is Suppressed During Capture

**Symptom**: OOM during CUDA graph capture, even though `torch.cuda.empty_cache()` is called and there appears to be enough cached memory available.

**Why it happens**: During CUDA graph capture, `cudaFree` and `cuMemUnmap` (for expandable segments) are **suppressed**—calling them would invalidate the graph being captured. This means `torch.cuda.empty_cache()` cannot actually release cached memory back to CUDA during capture.

Outside of capture, if memory is tight, PyTorch's allocator can call `empty_cache()` to release unused cached blocks back to CUDA, then request fresh memory via `cudaMalloc`. This pattern is common during OOM recovery. However, during capture, this fallback is blocked: cached memory stays reserved even if it's not being used, and the allocator cannot reclaim it.

**Example**: Consider a scenario where a tensor `x1` was allocated before capture but is no longer needed. If the user forgets to free it before entering the capture context and instead deletes it during capture, the memory cannot be returned to CUDA:

```python
import os
import torch

USE_GRAPH = bool(int(os.environ.get("USE_GRAPH", "0")))

graph = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
ctx = torch.cuda.graph(graph) if USE_GRAPH else torch.cuda.stream(stream)

# x1 is NOT a graph input—it's just a tensor that wasn't freed in time
x1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB in global pool
with ctx:
    x2 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB
    del x1  # x1's memory returns to cache, but can't be released back to CUDA
    torch.cuda.empty_cache()  # Suppressed during capture—no effect
    x3 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")  # 4GB - needs new memory
```

Note that `x1` is **not** a graph input tensor (deleting a graph input would cause numerical errors or illegal memory access during replay). Here, `x1` is simply a tensor the user didn't free before capture. The issue is that freeing it *inside* the capture context doesn't release its memory back to CUDA.

**Output without graph capture** (`USE_GRAPH=0`):

```text
         Tag (GB)      Allocated         Active  InactiveSplit       Reserved
-----------------------------------------------------------------------------
   After alloc x1          4.000          4.000          0.000          4.000
   After alloc x2          8.000          8.000          0.000          8.000
     After del x1          4.000          4.000          0.000          8.000
After empty cache          4.000          4.000          0.000          4.000
   After alloc x3          8.000          8.000          0.000          8.000
```

**Output with graph capture** (`USE_GRAPH=1`):

```text
         Tag (GB)      Allocated         Active  InactiveSplit       Reserved
-----------------------------------------------------------------------------
   After alloc x1          4.000          4.000          0.000          4.000
   After alloc x2          8.000          8.000          0.000          8.000
     After del x1          4.000          4.000          0.000          8.000
After empty cache          4.000          4.000          0.000          8.000
   After alloc x3          8.000          8.000          0.000         12.000
```

The comparison reveals the key difference:

1. **After del x1**: In both cases, `x1`'s 4GB is freed (Allocated drops from 8GB to 4GB), but the memory remains cached in the global pool (Reserved stays at 8GB).

2. **After empty cache**: This is the critical difference:
    - **Without capture**: `empty_cache()` calls `cudaFree` to release `x1`'s cached 4GB back to CUDA. Reserved drops from 8GB to 4GB.
    - **With capture**: `cudaFree` is suppressed, so `empty_cache()` has no effect. Reserved stays at 8GB.

3. **After alloc x3**: Needs 4GB:
    - **Without capture**: The allocator requests fresh memory from CUDA. Reserved grows back to 8GB.
    - **With capture**: `x1`'s cached memory cannot be released, so the allocator must request additional memory from CUDA. Reserved grows to 12GB (4GB `x1` cached + 4GB `x2` + 4GB `x3`).

**Consequence**: Without capture, the workload peaks at 8GB Reserved. With capture, it peaks at 12GB because the cached memory from `x1` cannot be released. On memory-constrained GPUs, this can cause OOM during capture even when the same workload runs fine without capture.

**Real-world example**: A common case is setting gradients to `None` at the beginning of each training iteration. Consider a typical training loop:

```python
for _ in range(warmup_iters):
    optimizer.zero_grad(set_to_none=True)  # or model.zero_grad(set_to_none=True)
    loss = model(input).sum()
    loss.backward()
    optimizer.step()

# Capture one iteration
with torch.cuda.graph(g):
    optimizer.zero_grad(set_to_none=True)  # ← Frees gradient tensors inside capture!
    loss = model(input).sum()
    loss.backward()
    optimizer.step()
```

After warmup, gradient tensors are allocated and cached in the global memory pool. When `zero_grad(set_to_none=True)` is called inside the capture context, it frees these gradient tensors, but `cudaFree` is suppressed—so the memory remains reserved but unusable. The subsequent `backward()` must allocate fresh memory for gradients, causing memory usage to spike. This can push a workload that runs fine in eager mode into OOM during capture.

**How to fix**: Free unused tensors—especially temporary gradients—before entering the capture context. Since `cudaFree` is suppressed during capture, all memory cleanup must happen beforehand:

```python
x1 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")
# ... use x1 ...

# Clean up before capture
del x1
# gc.collect()  # Optional: torch.cuda.graph() calls this if force_cudagraph_gc is enabled
# torch.cuda.empty_cache()  # torch.cuda.graph() calls this internally

with torch.cuda.graph(graph):
    # Now capture with maximum available memory
    x2 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")
    x3 = torch.empty(4*1024**3, dtype=torch.uint8, device="cuda")
```

---

## Debugging OOM

When encountering OOM errors with CUDA graphs, first check if your code matches any of the patterns described above. If the issue isn't obvious, try reducing tensor sizes (e.g., batch size) or model size (e.g., number of layers) to make debugging easier. Once you can run without OOM, use the following tools to diagnose the root cause.

PyTorch provides powerful memory profiling tools to help identify the source of excessive memory usage.

### Using PyTorch Memory Profiler

PyTorch's memory snapshot tool captures detailed allocation history, which can be visualized using the interactive [Memory Visualizer](https://pytorch.org/memory_viz):

```python
import torch

# Start recording memory history
torch.cuda.memory._record_memory_history(max_entries=100000)

# Run your code
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(static_input)

# Save snapshot for visualization
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
```

Upload the generated `memory_snapshot.pickle` file to [pytorch.org/memory_viz](https://pytorch.org/memory_viz) to visualize the allocation timeline.

:::{warning} Memory Visualizer Shows Allocated, Not Reserved

The memory visualization tool displays **Allocated** memory (memory actively used by tensors), not **Reserved** memory (total memory obtained from CUDA). After CUDA graph capture, you may observe that Allocated memory appears reasonable while Reserved memory is much higher. This discrepancy is expected—graph pools reserve memory for replay even after intermediate tensors are deleted.
:::

### Comparing Memory With and Without CUDA Graphs

The most effective debugging strategy is to compare memory behavior with and without CUDA graph capture. You can use the memory profiler above, or simply print memory statistics using `torch.cuda.memory_stats()` or `torch.cuda.memory_summary()`:

```python
import os
import torch
from contextlib import nullcontext

USE_GRAPH = bool(int(os.environ.get("USE_GRAPH", "0")))

def print_memory_stats(label):
    stats = torch.cuda.memory_stats()
    allocated = stats["allocated_bytes.all.current"] / 1e9
    reserved = stats["reserved_bytes.all.current"] / 1e9
    print(f"{label}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

graph = torch.cuda.CUDAGraph()
ctx = torch.cuda.graph(graph) if USE_GRAPH else nullcontext()

print_memory_stats("Before capture")
with ctx:
    output = model(static_input)
print_memory_stats("After capture")
```

Run with `USE_GRAPH=0` and `USE_GRAPH=1` to compare:

```bash
USE_GRAPH=0 python your_script.py  # Baseline without graph
USE_GRAPH=1 python your_script.py  # With graph capture
```

If Reserved memory is significantly higher with CUDA graphs, the difference indicates memory overhead from graph capture.

### Identifying the Problematic Allocation

To pinpoint the source of increased memory usage, insert logging checkpoints at key positions—before capture, at various stages during capture, and after capture. Run the same code with `USE_GRAPH=0` (eager mode) and `USE_GRAPH=1` (graph mode), then compare the memory statistics at each checkpoint to identify where the divergence occurs.

For quick debugging, add `print(torch.cuda.memory_summary())` at suspected locations. For deeper analysis, dump memory snapshots and compare them in the visualizer to trace the exact allocation responsible for the overhead.

For comprehensive guides on PyTorch memory profiling:

- [Understanding GPU Memory 1: Visualizing All Allocations over Time](https://pytorch.org/blog/understanding-gpu-memory-1/) — Introduction to memory visualization
- [Understanding GPU Memory 2: Finding and Removing Reference Cycles](https://pytorch.org/blog/understanding-gpu-memory-2/) — Advanced debugging techniques
- [torch.cuda.memory API Reference](https://docs.pytorch.org/docs/main/torch_cuda_memory.html) — Complete API documentation for memory profiling

---

## Quick Reference

**Illegal Memory Access and Segmentation Faults**:

- [ ] Don't let local tensors used as graph inputs go out of scope
- [ ] Don't reassign graph input tensors
- [ ] Don't free CPU tensors used for host-to-device copy

**OOM - Static Input Tensors Can't Be Freed**:

- [ ] Reuse static input tensors across graphs
- [ ] Share static input buffer across sequential graphs
- [ ] Chain graph outputs as inputs
- [ ] Reduce number of graphs

**OOM - Intermediate Tensors Can't Be Reused Across Memory Pools**:

- [ ] Share memory pools with `pool=graph1.pool()`
- [ ] Reduce number of graphs

**OOM - Intermediate Tensors After Capture Can't Reuse Graph Pool Memory**:

- [ ] Expand capture range to include post-capture operations
- [ ] Reduce temporary tensor allocations with kernel fusion

**OOM - Memory Fragmentation Across Pools**:

- [ ] Enable `expandable_segments:True`
- [ ] Limit block splitting with `max_split_size_mb:N`
- [ ] Share memory pools across graphs

**OOM - Deferred Memory Recycling**:

- [ ] Avoid `record_stream()` with proper synchronization
- [ ] Enable `graph_capture_record_stream_reuse:True`

**OOM - Gradient Accumulator Cross-Stream Memory Growth**:

- [ ] Initialize `AccumulateGrad` on the same stream as capture (e.g., init DDP/FSDP on capture stream)
- [ ] Enable `graph_capture_record_stream_reuse:True`

**OOM - `cudaFree` Is Suppressed During Capture**:

- [ ] Free unused tensors (especially temporary gradients) before entering capture context

**Debugging**:

- [ ] Check Reserved (not just Allocated) memory
- [ ] Compare memory stats with and without CUDA graph
- [ ] Use `torch.cuda.memory._dump_snapshot()` for detailed analysis

---

## What's Next?

- **[Capture Failures](capture-failures.md)**: Common capture errors
- **[Numerical Errors](numerical-errors.md)**: Wrong results, NaN/Inf issues
- **[Process Hang](process-hang.md)**: Debugging hanging or stuck processes
- **[Performance Issues](performance-issues.md)**: Poor speedup
