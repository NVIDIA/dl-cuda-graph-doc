<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# PyTorch CUDA Graph Integration

:::{note}

This section explains how PyTorch integrates CUDA Graphs, what features are supported, and the PyTorch-specific considerations when using graphs.
:::

## How PyTorch Integrates CUDA Graphs

### Design Philosophy

PyTorch and CUDA Graphs have fundamentally different design goals, which shapes how they integrate:

**PyTorch** is designed for **flexibility and ease of use**—dynamic computation graphs, eager execution, intuitive Python APIs, and seamless debugging. This flexibility is what makes PyTorch popular for research and rapid prototyping.

**CUDA Graphs** are designed for **maximum performance and full asynchrony**—static execution plans, minimal CPU overhead, and deterministic replay. They sacrifice flexibility for speed.

The integration bridges these philosophies by providing opt-in graph support that unlocks CUDA Graph performance when you need it. While CUDA Graphs' requirements (e.g. static shapes, no CPU sync, fixed memory addresses) may require some code adjustments in PyTorch's dynamic environment, the performance gains—often 1.2-3x speedups—make it worthwhile for production workloads. This guide helps you navigate these requirements and successfully adopt CUDA Graphs in your PyTorch applications.

### Integration Approach

PyTorch adopts a focused, pragmatic approach to CUDA Graph integration built on stream capture:

- **Stream capture only**: PyTorch uses stream capture exclusively (as introduced in [CUDA Graph Basics](../cuda-graph-basics/cuda-graph.md#method-2-stream-capture)). No explicit graph construction or manual node/edge building.
- **No graph updates**: Graph update APIs are not exposed. If you need different parameters, recapture the graph.
- **Capture once, replay repeatedly**: The core pattern—capture your workload once, then replay it many times with minimal overhead.

PyTorch provides two levels of APIs for CUDA Graph capture:

**Stream Capture APIs**:

- **[`torch.cuda.CUDAGraph`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)** - Low-level class wrapping `cudaGraph_t` and `cudaGraphExec_t`, exposing capture/replay methods directly
- **[`torch.cuda.graph()`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)** - Context manager that wraps CUDA's stream capture APIs (`cudaStreamBeginCapture`/`cudaStreamEndCapture`) with automatic side stream management and Python-friendly interface
- Use when you need explicit control over capture timing and replay logic

**Model/Function Graphing API**:

- **[`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)** - Built on top of `torch.cuda.graph()`, this high-level API automatically handles model/function graphing with additional considerations: captures both forward and backward passes separately for autograd compatibility, manages memory pool sharing across multiple callables, and provides autograd-compatible wrappers that can replace original modules/functions
- Use when graphing modules or functions that require autograd support or when you want automatic warmup and memory pool management

The key design principle: **PyTorch provides simple Python APIs** that wrap CUDA's stream capture mechanism, handling essential integration details like memory pool isolation for the caching allocator and automatic side stream setup. However, users are still responsible for ensuring their code satisfies CUDA Graph constraints (static shapes, no CPU sync, etc.).

## Responsibility Sharing

When using CUDA Graphs in PyTorch, responsibilities are split between PyTorch's integration layer and user. Understanding this division helps clarify what PyTorch automates and what you must manage.

### What PyTorch Handles

PyTorch's integration automatically manages:

- **Side stream setup**: Automatically creates and uses a non-default stream (CUDA requirement)
- **Stream capture orchestration**: Wraps `cudaStreamBeginCapture`/`cudaStreamEndCapture` in Python APIs
- **Memory pool isolation**: Uses a private memory pool for CUDA Graph allocations, separate from the eager execution pool. This is critical for PyTorch's caching allocator integration—during capture, memory is allocated and freed from the dedicated graph pool (just as in normal execution); during replay, only kernels execute with no allocator calls. This ensures that memory addresses allocated during capture remain alive and stable throughout the graph's lifetime, preventing the caching allocator from reusing those addresses. The trade-off is increased memory usage when both eager and graph allocations are substantial, but this separation is necessary for graph correctness.

### What User Must Ensure

User is responsible for satisfying other CUDA Graph constraints. The major constraints from the [Constraints](../cuda-graph-basics/constraints.md) documentation are:

1. **Asynchronous restrictions**: No CPU-GPU synchronization during capture
2. **Static graph topology**: Deterministic execution path, no dynamic control flow
3. **Static graph parameters**: Fixed memory addresses, kernel configs, and shapes
4. **Self-contained stream capture**: No external stream/event dependencies
5. **Memory lifetime**: All memory allocated outside the graph and referenced by the graph must remain valid for graph lifetime

See the [Constraints](../cuda-graph-basics/constraints.md) page for comprehensive details on each category.

### Responsibility Table

The table below clarifies what PyTorch automates versus what you must manage for the major constraints:

| Constraint Category | PyTorch Handles | User Responsible For |
|---------------------|-----------------|---------------------|
| **Asynchronous execution** | ✅ Side stream setup | ✅ No CPU sync in captured code |
| **Static topology** | - | ✅ Deterministic execution path, no dynamic control flow |
| **Static parameters** | - | ✅ Fixed shapes and arguments, no dynamic kernels |
| **Memory addresses** | ✅ Internal allocations via separate pool | ✅ Fixed graph inputs, use `.copy_()` |
| **Stream capture** | ✅ API wrapping and orchestration | ✅ Self-contained dependencies |
| **Error detection** | ⚠️ Limited runtime checking | ✅ Validate constraints before capture |

**Key takeaway**: PyTorch handles the PyTorch-CUDA integration mechanics, but **you must ensure your code satisfies all CUDA Graph constraints** documented in the [Constraints](../cuda-graph-basics/constraints.md) section. Most issues arise from constraint violations, not from PyTorch's integration itself.

## PyTorch CUDA Graph APIs

PyTorch provides three approaches for using CUDA Graphs:

### Stream Capture API: `torch.cuda.graph()`

A context manager for manual graph capture. It provides a thin wrapper around CUDA's stream capture APIs (`cudaStreamBeginCapture`/`cudaStreamEndCapture`) with automatic side stream setup and Python-friendly interface:

```python
g = torch.cuda.CUDAGraph()
static_input = torch.zeros(batch_size, *input_shape, device='cuda')

# Warmup on side stream: run a few iterations to stabilize memory allocations
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        _ = model(static_input)
torch.cuda.current_stream().wait_stream(s)

# Capture: record operations into the graph
with torch.cuda.graph(g):
    static_output = model(static_input)

# Replay: execute the graph multiple times with different data
for new_data in data_loader:
    static_input.copy_(new_data)  # Update values, not addresses
    g.replay()
    # Process static_output
```

**Important points**:

- **Allocate static tensors**: Create static input tensors before capture and reuse them
- **Warmup is required**: Run 3+ iterations before capture to trigger all memory allocations
- **Use `.copy_()` for input updates**: Never reassign tensors (`static_input = new_data` breaks the graph)
- **Side stream handled automatically**: The context manager creates a non-default stream for capture
- **Single graph object**: Create the `CUDAGraph` once, replay many times

See [torch.cuda.graph() documentation](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html) for full parameter details.

### Model Graphing API: `torch.cuda.make_graphed_callables()`

High-level API that graphs models/functions with autograd support:

```python
model = MyModel().cuda()
sample_input = torch.zeros(batch_size, *input_shape, device='cuda')

# Graph the model (warmup happens automatically)
graphed_model = torch.cuda.make_graphed_callables(
    model,
    (sample_input,),
    num_warmup_iters=3  # Use 11 for DDP
)

# Use in training loop - drop-in replacement
for data, target in dataloader:
    optimizer.zero_grad()
    output = graphed_model(data)  # Forward is graphed
    loss = criterion(output, target)
    loss.backward()  # Backward is also graphed
    optimizer.step()
```

**Important points**:

- **Automatic warmup**: Handles warmup iterations internally (unlike `torch.cuda.graph()`)
- **Forward and backward graphed separately**: Creates two graphs for full autograd support
- **Drop-in replacement**: Returns a callable that can replace the original model
- **Memory pool sharing**: Multiple callables can share the same memory pool for efficiency
- **Limitations**: No double backward, no module hooks during capture, cannot modify module structure after graphing

See [torch.cuda.make_graphed_callables() documentation](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html) for full parameter details and constraints.

### Automatic CUDA Graphs with `torch.compile()`

For users who want CUDA Graph benefits without manual capture/replay code:

```python
@torch.compile(mode="reduce-overhead")
def model_forward(x):
    return model(x)
```

**How it works**: The `"reduce-overhead"` mode has two CUDA Graph implementations:

1. **CUDAGraph Trees** (default): All graphs share a single memory pool, organized in a tree structure to handle different execution paths
2. **Legacy mode**: Each graph uses a separate memory pool, simpler but less memory-efficient

**CUDAGraph Trees features**:

- **Automatic detection**: Identifies graph-compatible code regions without user intervention
- **Tree structure**: Records multiple graphs and organizes them in a tree to handle different execution paths (e.g., conditional branches)
- **Smart partitioning**: Splits out incompatible operations (CPU ops, control flow, cudagraph_unsafe ops) into non-graphed segments
- **Shared memory pool**: All graphs in the tree share a single memory pool for maximum memory efficiency
- **Dynamic handling**: Dynamic shapes trigger new graph captures (with limited graph count); dynamic control flow regions are excluded from graphs and run eagerly

**Key advantage**: Zero manual effort—no warmup, capture, or replay code required. CUDAGraph Trees handle everything automatically.

**Key drawbacks**:

- **Graph fragmentation**: Automatic partitioning often creates many small CUDA graphs instead of a few large ones, limiting performance gains
- **Limited control**: Reducing graph count requires minimizing TorchDynamo graph breaks, which requires advanced knowledge of compiler internals and graph break causes

**Trade-off**: Automatic graphs prioritize ease of use over optimal performance. For maximum performance gains, manual capture with `torch.cuda.graph()` is often better, though it requires more effort.

For details, see the [CUDAGraph Trees documentation](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html).

---

For comprehensive information on using CUDA Graphs in PyTorch, including advanced patterns and constraints, see the [PyTorch CUDA Graphs Guide](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

## PyTorch-Specific Constraints

Beyond the CUDA-level [constraints](../cuda-graph-basics/constraints.md), PyTorch adds framework-specific requirements when applying CUDA Graph:

### Mandatory Warmup Iterations

PyTorch **requires** warmup before capture (not a CUDA requirement, but enforced by PyTorch):

- **Minimum 3 warmup iterations** for standard use
- **Minimum 11 warmup iterations** when using DistributedDataParallel
- **Warmup must occur on a side stream** (not the default stream)

**Why warmup is required**:

- **Memory allocator stabilization**: Triggers all memory allocations from PyTorch's caching allocator before capture begins
- **Initialization completion**: Allows JIT compilation, lazy initialization, and library setup to finish
- **DDP-specific**: DDP performs internal logging and setup around the 10th iteration, which must complete before capture

### RNG State Management

CUDA RNG works in graphs, but PyTorch requires specific handling:

- Multiple `torch.Generator` instances must be registered with `CUDAGraph.register_generator_state()` before capture
- Use `Generator.graphsafe_get_state()` and `graphsafe_set_state()` instead of regular state APIs during capture

**Why this is required**: Regular `get_state()`/`set_state()` return/accept CPU scalar values and can mutate across random generation, which cannot be captured in a graph. The `graphsafe_*` APIs move the RNG state to device tensors, allowing the state to be part of the captured graph and properly managed across replays.

For detailed explanation of how CUDA RNG works with graphs, see [Random Number Generator State](handling-dynamic-patterns.md#example-1-random-number-generator-state).

### Autograd Constraints (with `make_graphed_callables()`)

These apply only when using `make_graphed_callables()` for training:

- **No higher-order gradients**: Double backward is not supported
- **No module hooks during capture**: Hooks can be added after graphing, but not before
- **Frozen module structure**: Cannot add/remove module parameters or buffers after graphing
- **Buffer constraints**: Module buffers must have `requires_grad=False`
- **Fixed argument signature**: Arguments must be passed in the same order/format as provided in `sample_args`

### AMP (Automatic Mixed Precision) Integration

**How autocast cache works**: PyTorch's AMP autocast maintains a global cache (`Dict[TensorImpl*, Tuple[WeakRef, CastedTensor]]`) that maps FP32 tensors to their FP16/BF16 casted versions. Eligible tensors for caching are FP32 leaf tensors with `requires_grad=True` (typically model parameters, but not exclusively). On first encounter, tensors are cast and cached; subsequent uses return the cached version. The cache is cleared when exiting the autocast context ([`autocast_mode.py:403-404`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/amp/autocast_mode.py#L403-L404)).

**Requirement**: If **cached casted tensors** become CUDA graph inputs, you must disable caching:

```python
with torch.amp.autocast("cuda", cache_enabled=False):  # Required!
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = model(input)
```

**Why**: Cached tensors created before graph capture become graph inputs. When autocast exits, the cache is cleared and these cached tensors may be freed, leaving the graph with stale memory addresses.

**Scenario 1: First cast outside graph capture** (problematic)

```python
with torch.amp.autocast("cuda", cache_enabled=True):
    # First cast happens here - result is cached
    _ = model(warmup_input)  # weight.to(fp16) → cached at address 0x2000

    # Graph capture uses the cached tensor
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        output = model(input)  # Uses cached tensor at 0x2000 as INPUT

# ⚠️ Autocast exits → cache cleared → tensor at 0x2000 may be freed
# Graph replay references stale address 0x2000 → undefined behavior!
```

**Scenario 2: First cast inside graph capture** (safe)

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.cuda.amp.autocast(cache_enabled=True):  # OK!
        output = model(input)  # First cast is INSIDE graph
```

The cast operation is part of the captured graph, not a graph input. The cached tensor is created and used entirely within the graph's execution, so it's not invalidated.

**Key distinction**:

- **Graph input** (external dependency): Cached tensor created outside → invalidated on autocast exit
- **Graph operation** (internal): Cast operation captured in graph → no external dependency

**With `cache_enabled=False`**: No caching occurs, so no cached tensors can become graph inputs. Each cast produces a fresh tensor, and the cast operation is explicitly captured in the graph.

:::{note} Recommendation: Always Disable Cache with CUDA Graphs

For simplicity, you can always disable autocast cache when using CUDA graphs, regardless of capture pattern. While this causes redundant cast operations each iteration, the overhead is usually negligible compared to CUDA graph benefits. PyTorch enforces this in [`make_graphed_callables`](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/cuda/graphs.py#L296-L299), raising an error if `cache_enabled=True` when capturing inside autocast.
:::

### DDP (DistributedDataParallel) Setup

Special requirements for distributed training:

**NCCL >= 2.9.6** (for full graph capture):

1.  **Disable async error handling**: Set `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` before `init_process_group()`

    **Why**: PyTorch's NCCL backend uses a watchdog thread that calls `cudaEventQuery()` to monitor collective operations. These cross-thread CUDA API calls are incompatible with CUDA graph capture. Disabling async error handling stops the watchdog thread, eliminating the conflict.

    **Trade-off**: You lose automatic timeout detection and error propagation for NCCL operations. **Important**: Ensure your distributed training code works correctly without CUDA graphs first. With async error handling disabled, NCCL errors may cause hangs instead of clear error messages, making debugging much harder.

2.  **Construct DDP in side-stream context**:

    ```python
    with torch.cuda.stream(s):
        model = DistributedDataParallel(model)
    ```

    **Why**: DDP initializes gradient accumulation buffers when operators first use them. The buffers are created on the stream that's current during DDP construction. If DDP is constructed on the default stream, gradient accumulation will execute on the default stream, which cannot be captured in a graph. Constructing DDP in a side stream ensures all DDP-related operations use capturable streams. See [DistributedDataParallel: Default Stream Conflict](../troubleshooting/capture-failures.md#distributeddataparallel-default-stream-conflict) for details.

3.  **Perform 11 warmup iterations** (not 3)

    **Why**: DDP performs internal logging and state setup around the 10th iteration. These operations must complete before capture to avoid capturing transient initialization work that shouldn't be part of the graph.

**NCCL < 2.9.6**: Collectives cannot be captured—must use partial-network capture

:::{note} NCCL Buffer Registration with Expandable Segments

PyTorch's expandable segments allocator mode reduces memory fragmentation by extending memory segments on demand, providing a contiguous virtual address space while using multiple physical segments. However, older NCCL versions only support registering a single physical segment for buffer registration. When buffers span multiple physical segments, this can cause issues like illegal memory access during CUDA graph capture or replay. If you encounter problems, set `NCCL_GRAPH_REGISTER=0` to disable buffer registration.
:::

For detailed examples and advanced usage patterns, see the [PyTorch CUDA Graphs Guide](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).

## What's Next?

- **[Quick Checklist](quick-checklist.md)**: Verify your code is ready for CUDA Graph capture before attempting to graph
- **[Transformer Engine and Megatron-LM CUDA Graphs](te-megatron-cuda-graphs.md)**: Learn about Megatron-LM's CudaGraphManager and FullCudaGraphWrapper for distributed training
- **[Best Practices for PyTorch](best-practices.md)**: Practical recommendations, common pitfalls, and decision guides for successfully using CUDA Graphs in PyTorch
- **[Examples](../examples/introduction.md)**: Real-world implementations showing these concepts in action
