<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Writing Sync-Free Code

:::{note}

This guide helps you identify and eliminate host-device synchronizations in PyTorch code, which is critical for both performance and CUDA graph compatibility.
:::

## What is Sync-Free Code?

**Sync-free code** means your CPU can continuously queue work to the GPU without waiting for GPU operations to complete. PyTorch applications can introduce synchronizations unconsciously—PyTorch's high-level API is highly flexible and hides many implementation details. What looks like a simple operation in Python (like `if tensor.item() > 0:`) may trigger a host-device synchronization under the hood. Understanding where these hidden syncs occur is crucial for both performance and CUDA graph compatibility.

**Why sync-free matters**:

1. **Performance improvement**: Making your code sync-free can significantly improve performance on its own by eliminating idle GPU time. When CPU doesn't wait for GPU, the GPU can work continuously without stalls.
2. **CUDA Graphs prerequisite**: Sync-free code is a **prerequisite** for CUDA graphs. Graph capture cannot include synchronization points—any sync operation will cause capture to fail. You must eliminate syncs before you can apply CUDA graphs.

When CPU synchronizes with GPU, it ultimately calls one of these CUDA **driver APIs**, blocking the CPU thread until the GPU operation completes:

- `cuEventSynchronize` - CPU waits (blocks) until the specified event is signaled on the GPU
- `cuStreamSynchronize` - CPU waits (blocks) until all GPU work submitted to the specified stream finishes
- `cuCtxSynchronize` - CPU waits (blocks) until all GPU work on the entire device finishes across all streams

Making your code sync-free essentially means eliminating any PyTorch operations that invoke these APIs under the hood.

## Why Synchronizations Occur

Synchronizations fall into two categories: false dependencies (avoidable) and true dependencies (require restructuring).

### False Dependencies (Avoidable)

False dependencies occur when CPU doesn't actually need GPU results but gets blocked anyway. These syncs are **unnecessary** and can be eliminated without changing program logic.

**Redundancy and unconscious behavior**: Many syncs are accidental—debug prints left in code, unnecessary `.item()` calls for logging, or operations whose sync behavior you didn't realize. These are introduced unconsciously and are the easiest to remove once identified.

**Poor coding practices**: Two common patterns cause avoidable syncs:

- **Not tracking tensor devices**: Without knowing where tensors live, code moves data back and forth between CPU and GPU unnecessarily. Treat `device` as seriously as data type—always know your tensors' locations.
- **Using blocking APIs**: Using `.cuda()` instead of `.to('cuda', non_blocking=True)`, or allocating CPU tensors then doing blocking transfers instead of allocating directly on the target device.

### True Dependencies (Requires Restructuring)

True dependencies represent **genuine** cases where CPU operations depend on GPU results. The CPU operation needs the GPU value to proceed, typically involving dynamic behavior.

**Control flow dependency**: CPU branches based on GPU values—checking if loss exceeds a threshold, implementing early stopping, or choosing execution paths based on computed conditions.

**Data dependency**: CPU needs GPU results for subsequent operations:

- **Dynamic memory allocation**: Allocating tensors whose size depends on GPU results (e.g., output from `torch.nonzero()` or `torch.masked_select()` has unknown size until computed)
- **CPU computation using GPU values**: Computing statistics for logging, updating learning rates based on metrics, or saving checkpoints with computed values

True dependencies require code restructuring: move CPU logic outside the graphed region, use GPU-side alternatives like `torch.where()`, or accept that those parts cannot be included in CUDA graphs.

## Detecting Synchronizations

Identifying where syncs occur is the first step to eliminating them. PyTorch and NVIDIA tools provide several methods to find synchronization points in your code.

### Using PyTorch's Sync Debug Mode

For quick detection of stream synchronizations:

```python
import torch

# Enable sync detection (prints stack trace on sync)
torch.cuda.set_sync_debug_mode('warn')  # or 'error' to raise exception

# Your code here
# Any sync will print a warning with stack trace
```

:::{note} Scope of `set_sync_debug_mode`

This mode only detects synchronizations that go through PyTorch's wrapped `cuStreamSynchronize` API, which covers most PyTorch operations. Syncs from extensions or third-party libraries that directly call CUDA sync APIs will not be detected.
:::

### Using Nsight Systems

Profile with call stack capture to see exactly where syncs occur:

```bash
# Capture with Python and C++ call stacks
nsys profile --capture-range=cudaProfilerApi \
             --python-sampling=true \
             --backtrace=dwarf \
             python your_script.py
```

In the Nsight Systems GUI, check the **CUDA API** timeline row. Search for `cudaStreamSynchronize`, `cudaEventSynchronize`, or `cudaDeviceSynchronize` calls. When you find one, the call stack panel will show you which Python line triggered the sync.

## Strategies for Removing Synchronizations

Once you've identified where syncs occur, you can systematically eliminate them. Start with false dependencies (easy wins), then tackle true dependencies (requires more effort).

**1. Remove redundancy**: The easiest wins—simply delete operations that don't need to happen:

- Delete/disable debug prints and logging inside hot loops
- Remove unnecessary API calls like `.item()`
- Eliminate duplicate synchronizations

**2. Use non-blocking APIs**: Make data transfers asynchronous so CPU doesn't wait, examples:

- Replace `.cuda()` or `.type()` with `.to('cuda', non_blocking=True)`
- Replace `.cpu()` with `.to('cpu', non_blocking=True)` on condition

:::{warning} Non-Blocking GPU-to-CPU Transfers Require Care

Only use `non_blocking=True` when CPU doesn't immediately use the transferred data. If CPU operations depend on the GPU result, non-blocking transfer can cause CPU to operate on incomplete data before the transfer finishes, leading to incorrect results. Use non-blocking only for fire-and-forget transfers.
:::

**3. Switch to sync-free alternatives**: Use different APIs that achieve the same result without syncing, examples:

- Instead of: `x_gpu.type(torch.LongTensor)` → use: `x_gpu.type(torch.long)`
- Instead of: `torch.zeros(..., device='cpu').to('cuda')` → allocate directly: `torch.zeros(..., device='cuda')`
- Instead of: `torch.tensor(0, device='cuda')` → use: `torch.zeros(1, device='cuda', dtype=torch.int64).squeeze()`
- Instead of: `x_gpu[idx_gpu] = 0` → use: `x_gpu[idx_gpu] = zero_tensor`

**4. Delay synchronization**: If you must sync, move it to less critical moments:

- Move logging/validation to end of iteration rather than middle of fprop/bprop, allowing you to capture fprop and bprop in a single graph
- Example: Log loss after optimizer step, not immediately after loss computation

**5. Coalesce multiple syncs**: If you need multiple values from GPU, get them all at once:

- Gather values into a list, sync once to get all instead of syncing for each value
- Reduces total sync overhead from `N × sync_cost` to `1 × sync_cost`
- Example: Collect loss, accuracy, and gradient norm, then call `.cpu()` once on a stacked tensor

**6. Offload to GPU** (advanced): Move CPU operations to GPU so no sync is needed:

- Use GPU-native operations instead of CPU logic (e.g., use `torch.max()` instead of Python `max()`)
- Develop custom CUDA kernels for CPU operations (e.g., custom gradient clipping kernel to lower *clipping or not* to GPU)

**7. Exclude from capture range** (last resort): Accept partial graphing:

- If a sync is unavoidable and critical, keep that section outside your CUDA graph
- Graph the sync-free regions only
- Partial graphing is better than no graphing

The goal is to eliminate as many syncs as possible, starting with the easiest (redundant ones) and working toward the harder ones (necessary dependencies). Even if you can't remove all syncs, reducing them improves performance and expands what you can capture in CUDA graphs.

## Common PyTorch Synchronizations

In the sections below, operations marked with `# cuStreamSynchronize`, `# cuEventSynchronize`, or `# cuCtxSynchronize` indicate they introduce host-device synchronization. These operations block the CPU until the GPU completes the work, breaking async execution and making them incompatible with CUDA graph capture.

**Variable naming conventions** in this section:

- `x_scalar` - CPU scalar value
- `x_list` - Python list
- `x_cpu` - CPU tensor (on host)
- `x_gpu` - GPU tensor (on device)

### Explicit Synchronizations

These are obvious synchronization calls that directly block the CPU waiting for GPU:

```python
stream = torch.cuda.Stream()
event = torch.cuda.Event()
cur_stream = torch.cuda.current_stream()

event.record()
event.wait()              # cudaStreamWaitEvent, non-blocking for host
event.synchronize()       # cuEventSynchronize - BLOCKS CPU
stream.synchronize()      # cuStreamSynchronize - BLOCKS CPU
torch.cuda.synchronize()  # cuCtxSynchronize - BLOCKS CPU, waits for all streams
```

**Pattern**: Blocking synchronizations (`event.synchronize()`, `stream.synchronize()`, `torch.cuda.synchronize()`) must be removed or moved outside CUDA graph capture regions.

### Tensor Movement Between Devices

Moving tensors between CPU and GPU often triggers synchronization:

```python
x_gpu = torch.randint(0, 10, (3, 4), device='cuda')
x_cpu = x_gpu.cpu()           # cuStreamSynchronize
x_cpu = x_gpu.to(device='cpu', non_blocking=True)       # Async, no sync
x_gpu.type(torch.long)        # cuda tensor, no sync
x_gpu.type(torch.LongTensor)  # cuStreamSynchronize, creates CPU tensor

idx_list = list(reversed(range(x_gpu.size(0))))
idx_cpu = torch.tensor(idx_list, dtype=torch.int64, device='cpu')
idx_gpu = idx_cpu.cuda()      # cuStreamSynchronize
idx_gpu = idx_cpu.cuda(non_blocking=True)               # Async, no sync
idx_gpu = idx_cpu.to(device='cuda', non_blocking=True)  # Async, no sync
```

**Pattern**: GPU-to-CPU transfers (`.cpu()`, `.to('cpu')`) and CPU-to-GPU transfers (`.cuda()`, `.to('cuda')`) synchronize by default. Use `non_blocking=True` to make them async. Type conversions: `.type(torch.long)` (dtype conversion) stays on GPU with no sync, but `.type(torch.LongTensor)` (creates CPU tensor) synchronizes.

### Tensor Creation and Allocation

Creating tensors in an inappropriate way on CUDA device can cause syncs:

```python
torch.tensor(0).to(device='cuda')  # cuStreamSynchronize
torch.tensor(0).to(device='cuda', non_blocking=True)               # Async, no sync
torch.tensor(0, device='cuda')     # cuStreamSynchronize
zero = torch.zeros(1, device='cuda', dtype=torch.int64).squeeze()  # No sync

arr = np.array([1, 2, 3])
arr_cpu = torch.as_tensor(arr, device='cpu')
arr_gpu = torch.as_tensor(arr, device='cuda')  # cuStreamSynchronize
arr_gpu = arr_cpu.to(device='cuda', non_blocking=True)             # Async, no sync

bool_list = [False]
torch.cuda.BoolTensor(bool_list)   # cuStreamSynchronize
bool_cpu = torch.tensor(bool_list, dtype=torch.bool, device='cpu')
bool_gpu = bool_cpu.to(device='cuda', non_blocking=True)           # Async, no sync
```

**Pattern**: Creating tensors from Python objects (scalars, lists) or NumPy arrays directly on CUDA (`device='cuda'`) synchronizes. Two-step approach avoids sync: create on CPU first, then transfer with `non_blocking=True`. Direct CUDA allocation with `torch.zeros()`, `torch.ones()`, etc. doesn't sync.

### Data-Dependent Control Flow

Using GPU tensor values in Python control flow statements requires syncing the value to CPU:

```python
loss = torch.randn(1, device="cuda")
loss.item()  # cuStreamSynchronize - explicit sync to get CPU scalar
torch.is_nonzero(loss)  # cuStreamSynchronize, implicit `.item()`
if loss:     # cuStreamSynchronize, implicit `torch.is_nonzero()`
    pass

flag = x_gpu.sum() > 16  # Creates bool GPU tensor
if flag:     # cuStreamSynchronize, implicit `torch.is_nonzero()`
    pass

flag = x_gpu.isnan().any()
if flag:  # cuStreamSynchronize
    pass

max(x_gpu[0, 0], x_gpu[0, 1])        # cuStreamSynchronize, Python max needs CPU values
torch.max(x_gpu[0, 0], x_gpu[0, 1])  # No sync, stays on GPU
```

**Common example**: AMP gradient scaler's [_maybe_opt_step](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/amp/grad_scaler.py#L343) checks for inf:

```python
if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()): # cuStreamSynchronize from .item()
    retval = optimizer.step(*args, **kwargs)
```

**Pattern**: Using GPU tensor values in Python control flow (either through `.item()` or implicit `.is_nonzero()`) forces GPU→CPU sync to materialize the value. Common culprits: checking loss values, NaN detection, boolean conditions from comparisons. Solutions: (1) use GPU-native operations like `torch.where()` instead of `if/else`, (2) use `torch.max()` instead of Python `max()`, (3) implement as custom CUDA kernel, or (4) move data-dependent control flow outside graph capture.

### Indexing Tensors

Indexing with tensors from different objects or devices can causes implicit transfers and syncs:

```python
# index_select index tensor with integers
x_gpu[0]         # No sync
x_gpu[idx_list]  # cuStreamSynchronize, implicit blocking `.to()`
x_gpu[idx_cpu]   # cuStreamSynchronize, implicit blocking `.to()`
# x_cpu[idx_gpu]  # indices should be on cpu or the same device as the indexed tensor
x_gpu[idx_gpu]   # No sync
x_gpu[idx_gpu] = 0  # cuStreamSynchronize, implicit blocking `.to()`
x_gpu[idx_gpu] = zero_gpu  # No sync
torch.index_select(x_gpu, 0, idx_gpu)  # No sync
```

**Pattern**: Indexing GPU tensors with CPU tensors or Python lists triggers implicit device transfer and sync. Integer scalar indexing (`x_gpu[0]`) is fine. Assigning scalar values (`x_gpu[idx_gpu] = 0`) also syncs because the scalar `0` must be transferred to GPU. Solutions: (1) keep indices on same device as indexed tensor, (2) for assignments, use GPU tensors as values (`zero_gpu`) instead of Python scalars, (3) use explicit operations like `torch.index_select()` which stay on GPU.

### Slicing with Tensor Indices

Python slice syntax (`x[start:stop]`) requires integer values. When you use CUDA tensors as slice bounds, PyTorch must call `.item()` to convert them to Python integers, triggering a sync:

```python
x = torch.randn((32, 32), device='cuda')
i = torch.tensor([8], dtype=torch.long, device="cuda")
j = torch.tensor([16], dtype=torch.long, device="cuda")
s = torch.tensor(list(range(8, 16)), dtype=torch.long, device="cuda")

# ✅ No sync - Python integers
x[8:16]
x[:, 8:16]
x[8:16, 8:16]

# ❌ Syncs - CUDA tensors converted via .item()
x[i:j]       # cuStreamSynchronize x2 (i and j each converted)
x[:, i:j]    # cuStreamSynchronize x2
x[i:j, i:j]  # cuStreamSynchronize x4 (each bound converted separately)

# ✅ No sync - index select with CUDA tensor (not slice syntax)
x[:, s]      # No sync
x[s, :]      # No sync
```

**Pattern**: Slice bounds (`:` syntax) must be Python integers. Using CUDA tensors as bounds triggers `.item()` → sync. Index select with a CUDA index tensor (`x[:, s]`) does *not* sync—only slice syntax with tensor bounds does.

**Solutions**: Use Python integers for slice bounds when they are known at Python execution time. For GPU-computed bounds, convert slice operations to index select: build a GPU index tensor (`s = torch.arange(start, end, device='cuda')`) and use `x[:, s]` or `torch.index_select()` instead of slice syntax.

### Operations with Dynamic Output Shapes

Operations that produce dynamically-sized outputs often sync to determine the size on CPU:

**Masked selection** is the most common case. Masked selection extracts elements from a tensor using a boolean mask (e.g., `x[x > 5]`). The output shape is unknown until we count how many `True` values exist in the mask, which is a GPU computation. PyTorch must sync to get this count from GPU to CPU to allocate the correctly-sized output tensor:

```python
# masked_select: index tensor with boolean mask
# stream synchronize after D2H of integer of non-zero count
# used to determine output size and whether need to copy: nonzero_cuda_out_impl
mask_gpu = x_gpu > 5
torch.nonzero(mask_gpu)  # cuStreamSynchronize
mask_cpu = mask_gpu.to(device='cpu', non_blocking=True)
x_gpu[mask_gpu]          # cuStreamSynchronize, implicit `torch.nonzero()`
x_gpu[mask_cpu]          # No sync
torch.where(mask_gpu)    # cuStreamSynchronize, implicit `torch.nonzero(mask_gpu, as_tuple=True)`
x_gpu[torch.where(mask_gpu)]          # cuStreamSynchronize
torch.masked_select(x_gpu, mask_gpu)  # cuStreamSynchronize

# Sync-free alternatives with static shapes
x_gpu & mask_gpu  # shape is the same, torch.bitwise_and, 8 & 1 = 0
torch.where(mask_gpu, x_gpu, 0)  # shape is the same, loss.mean() and var() differ
torch.where(mask_gpu, x_gpu, 0).float().sum() / torch.sum(mask_gpu)

# Assignments
x_gpu[torch.where(mask_gpu)] = -1  # cuStreamSynchronize x 2
x_gpu[mask_gpu] = -1               # No sync
x_gpu.masked_fill_(mask_gpu, -1)   # No sync
x_gpu[mask_gpu] = zero_gpu         # cuStreamSynchronize
x_gpu.masked_fill_(mask_gpu, zero_gpu)          # cuStreamSynchronize
x_gpu = torch.where(mask_gpu, zero_gpu, x_gpu)  # No sync

# Backward with masked indexing
x_fp32 = x_gpu.float().requires_grad_()
grad_gpu = torch.randn_like(x_fp32)
x_fp32.backward(grad_gpu)         # No sync
x_fp32[0].backward(grad_gpu[0])  # No sync
x_fp32[idx_gpu].backward(grad_gpu[idx_gpu])    # No sync
x_fp32[mask_cpu].backward(grad_gpu[mask_cpu])  # No sync
x_fp32[mask_gpu].backward(grad_gpu[mask_gpu])  # cuStreamSynchronize x 2
torch.where(mask_gpu, zero_gpu.float(), x_fp32).backward(grad_gpu)  # cuStreamSynchronize
```

**Other dynamic shape operations**:

```python
y = torch.tensor([[1, 2], [3, 4]], device='cuda')  # cuStreamSynchronize
repeats = torch.tensor([1, 2], device='cuda')      # cuStreamSynchronize
torch.repeat_interleave(y, repeats, dim=0)         # cuStreamSynchronize x 2 - output size unknown
torch.repeat_interleave(y, repeats, dim=0, output_size=3)  # No sync if size specified

torch.unique(idx_gpu)  # cuStreamSynchronize - unknown output size
torch.unique_consecutive(idx_gpu)  # cuStreamSynchronize
```

**Pattern**: Operations with dynamic output sizes sync to determine allocation size.

**Masked indexing** is the most common case:

- `torch.nonzero()`, `x_gpu[mask_gpu]`, `torch.masked_select()` all sync because output size depends on GPU computation
- `x_gpu[mask_cpu]` doesn't sync (size known on CPU)

**Assignments** behave differently:

- `x_gpu[mask_gpu] = -1` doesn't sync (no new allocation needed)
- `x_gpu[mask_gpu] = zero_gpu` syncs (needs tensor metadata)

**Backward passes** with masked indexing also sync.

**Solutions**:

1. Use fixed-size operations like `torch.where(mask, x, 0)` which preserves input shape
2. For `repeat_interleave()`, specify `output_size` parameter if known
3. Avoid `unique()`, `nonzero()` in graph-captured regions
4. Restructure to compute dynamic operations outside capture

### Embedding Layers

Embedding backward can introduce syncs depending on PyTorch version and number of indices:

```python
embedding = torch.nn.Embedding(50304, 768).cuda()
idx = torch.randint(0, 50304, (4, 1024), dtype=torch.int64)
idx = idx.to(device="cuda", non_blocking=True)

y = embedding(idx)
y.sum().backward()  # cuStreamSynchronize x 2 (in older PyTorch or large indices)
```

**Pattern**: If index count >3072 and PyTorch uses CUB <1.16, embedding backward falls back to `thrust::unique_by_key_copy` which syncs. With CUB >=1.16, it uses `cuda::cub::unique_by_key` which is sync-free. Upgrade to recent PyTorch to avoid this.

### Distributed Operations

Distributed collectives and barriers involve synchronization:

```python
# Collective operations
torch.distributed.all_reduce(x_gpu)  # Syncs via events internally
work = torch.distributed.all_reduce(x_gpu, async_op=True)  # Returns immediately
work.wait()  # Waits for completion, may sync

torch.distributed.barrier()  # cuCtxSynchronize - blocks all ranks
```

**DistributedDataParallel (DDP) logging**: DDP invokes `cuEventSynchronize` x5 for performance logging in older versions. See [torch/nn/parallel/distributed.py#L855](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/parallel/distributed.py#L855) and [torch/csrc/distributed/c10d/logger.cpp#L334](https://github.com/pytorch/pytorch/blob/v1.10.0/torch/csrc/distributed/c10d/logger.cpp#L334).

**Pattern**: Use `async_op=True` where possible, and ensure NCCL version >=2.9.6 which supports graph capture of collectives.

## Summary and Best Practices

### Quick Reference: Sync-Free Alternatives

| Sync-Inducing Pattern | Sync-Free Alternative |
|----------------------|----------------------|
| **Device Transfers** | |
| `.cpu()` or `.to('cpu')` | `.to('cpu', non_blocking=True)` |
| `.cuda()` or `.to('cuda')` | `.to('cuda', non_blocking=True)` |
| `.type(torch.LongTensor)` | `.type(torch.long)` (dtype only) |
| **Tensor Creation** | |
| `torch.tensor(obj, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| `torch.as_tensor(arr, device='cuda')` | Create on CPU, then `.to('cuda', non_blocking=True)` |
| Direct creation: `torch.zeros(..., device='cuda')` | ✅ No sync, use freely |
| **Control Flow** | |
| `.item()` in conditionals | Use `torch.where()` or move outside graph |
| `if gpu_tensor:` or `if flag:` | Keep logic on GPU with `torch.where()` |
| Python `max(a, b)` on GPU tensors | `torch.max(a, b)` |
| **Indexing** | |
| `x_gpu[idx_cpu]` or `x_gpu[idx_list]` | `x_gpu[idx_gpu]` (same device) |
| `x_gpu[idx] = 0` (scalar) | `x_gpu[idx] = zero_gpu` (GPU tensor) |
| **Dynamic Shapes** | |
| `x_gpu[mask_gpu]` (selection) | `torch.where(mask_gpu, x_gpu, 0)` (fixed size) |
| `x_gpu[mask_gpu] = val_gpu` | `x_gpu.masked_fill_(mask_gpu, scalar)` |
| `torch.nonzero(mask)` | Move outside graph or use `torch.where()` |
| `torch.repeat_interleave(x, repeats)` | Specify `output_size` if known |

### Core Principles

Writing sync-free code requires avoiding CPU-GPU synchronization. Four main categories cause syncs:

1. **Blocking device transfers**: GPU↔CPU transfers sync unless `non_blocking=True`
2. **Host control flow on GPU data**: Using GPU values in Python `if`, `.item()`, or `max()` forces sync
3. **Cross-device operations**: Indexing with CPU tensors/lists or assigning Python scalars triggers implicit transfers
4. **Dynamic output sizes**: Operations like `nonzero()`, masked selection, or `unique()` must sync to determine allocation size

### Strategies for Elimination

**Start with easy wins**:

- **Remove redundancy**: Delete debug prints, unnecessary `.item()` calls, duplicate syncs
- **Use `non_blocking=True`**: Make transfers async where CPU doesn't immediately need the data
- **Switch to sync-free alternatives**: Direct GPU allocation (`torch.zeros(device='cuda')`), dtype conversion (`.type(torch.long)` not `.type(torch.LongTensor)`), GPU tensors for scalar assignments

**For harder cases**:

- **Delay synchronization**: Move logging/validation to end of iteration
- **Coalesce multiple syncs**: Batch multiple GPU→CPU transfers into one
- **Offload to GPU**: Use `torch.where()` instead of `if/else`, `torch.max()` instead of Python `max()`, or custom CUDA kernels
- **Exclude from capture**: Keep unavoidable syncs outside CUDA graph, graph the rest

The goal is to eliminate as many syncs as possible. Even partial sync removal improves performance and expands what you can capture in CUDA graphs.

---

For CUDA graph capture, every synchronization is a potential failure point. Systematically eliminate syncs using this guide, verify with profiling, then attempt capture.

## What's Next?

- **[Handling Dynamic Patterns](handling-dynamic-patterns.md)**: Workarounds for dynamic control flow, shapes, and other common patterns
- **[Quick Checklist](quick-checklist.md)**: Complete checklist to verify your code is ready for capture
- **[Examples](../examples/introduction.md)**: See sync-free patterns in real-world implementations
- **[Constraints](../cuda-graph-basics/constraints.md)**: Full list of CUDA graph requirements
- **[Troubleshooting](../troubleshooting/introduction.md)**: Debug capture failures and other issues
