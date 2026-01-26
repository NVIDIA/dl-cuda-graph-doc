<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Constraints

:::{note}

This section covers the fundamental constraints imposed by the CUDA runtime on CUDA Graphs.
:::

While CUDA Graphs are powerful, they come with fundamental constraints imposed by the CUDA runtime. Understanding these limitations is essential for determining whether your workload is suitable for graph capture and how to work within the restrictions.

## 1. Asynchronous Restrictions

CUDA Graphs rely on the asynchronous execution model, where the CPU submits work to the GPU and continues without waiting for completion. This imposes several restrictions related to synchronization and stream behavior:

### No Host-Device Synchronization

Operations that synchronize the CPU with the GPU are prohibited during stream capture. This includes:

- `cudaDeviceSynchronize()` - Blocks CPU until all device work completes
- `cudaStreamSynchronize()` - Blocks CPU until specific stream completes (forbidden on capturing stream)
- `cudaEventSynchronize()` - Blocks CPU until event is signaled
- Synchronous memory operations (like `cudaMemcpy` without the `Async` suffix)

Synchronization would break the capture process because the driver cannot record operations that require immediate CPU-GPU coordination. All synchronization must occur outside the capture boundaries.

### No Default Stream

Graph capture must occur on a non-default stream. The default stream (stream 0) has special semantics with implicit synchronization behavior that conflicts with graph capture requirements. When capturing, you must explicitly create and use a dedicated CUDA stream.

### No Stream/Event Query Operations

During capture, you cannot query stream or event status using operations like:

- `cudaStreamQuery()` - Check if stream has completed all work
- `cudaEventQuery()` - Check if event has been signaled

These query operations require inspecting runtime state, which is incompatible with CUDA Graphs. Any timing or status checks must be performed outside the graph.

## 2. Static Graph Topology

The graph topology must be static—defined during capture/creation and fixed at instantiation. The graph records a fixed set of operations and their dependencies, and the runtime cannot alter this structure based on intermediate results.

:::{note} Conditional Nodes (CUDA 12.3+)

CUDA 12.3+ introduced [Conditional Nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#conditional-graph-nodes) that enable limited GPU-side branching within graphs. Importantly, the graph topology remains static even with conditional nodes—the conditional node itself and all its possible execution paths (subgraphs) must be defined at capture/creation time. What's dynamic is only which branch the GPU selects at runtime based on a GPU-evaluated condition. Constraints include: the branching condition must be evaluated on the GPU (not CPU), and you must use special CUDA APIs (`cudaGraphConditionalHandle`) to express the control flow. This is an advanced feature primarily for CUDA C++ developers working on complex GPU-driven workflows.
:::

Without conditional nodes, data-dependent branching—where control flow decisions are based on values computed during execution—is not supported. Operations like:

- Conditional kernels based on runtime data, example:

  ```cpp
  if (value > threshold):
      launch_kernel_a()
  else:
      launch_kernel_b()
  ```

- Variable loop counts determined by computation results
- Dynamic graph expansion based on runtime conditions

are all prohibited. The graph structure is frozen at instantiation.

## 3. Static Graph Parameters

All parameters and memory configurations in a graph are captured at definition time and used as-is during graph launch by default. [Graph Update APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#updating-instantiated-graphs) (CUDA 11.0+) allow limited in-place modifications to certain node parameters using `cudaGraphExecNodeSetParams()`, but these updates have restrictions: you can only modify parameters of existing nodes and cannot change the graph topology.

### Memory Addresses

By default, the graph records the exact pointers to host/device memory used by the graph, and every graph launch reads from and writes to the same virtual memory addresses used during capture. Without Graph Update APIs, changing these addresses would require rebuilding the entire graph structure. This means you cannot reallocate host/device memory or change buffer locations between launches—you can only update the values at those fixed addresses. Graph Update APIs can modify memory addresses for certain node types, allowing address changes without full recapture.

Importantly, **all memory addresses referenced by the graph must remain valid for the entire lifetime of the graph executable**. See [Memory Lifetime Requirements](#memory-lifetime-requirements) for details.

### Kernel Parameters and Configurations

All kernel arguments, including scalar parameters, grid dimensions, block dimensions, and shared memory sizes, are captured at definition time. The graph stores the complete kernel launch descriptor, and by default the runtime uses these exact parameters on every graph launch. **Without Graph Update APIs, these parameters must be static and cannot be changed between launches.** Graph Update APIs can modify kernel arguments and launch configurations.

:::{warning} Silent Failures with Stale Parameters

If you modify the underlying data that was captured (for example, changing a variable that was used as a kernel argument) without using Graph Update APIs, the graph will **not** automatically pick up the new value—it continues using the captured value. This typically does **not raise an error**, leading to silent incorrect results. For values that need to change between launches, either use Graph Update APIs to modify node parameters, or pass data through **device memory addresses** (where you update the values at those addresses) rather than scalar arguments (which are frozen at capture).
:::

### Static Shape Requirement

Data shapes must remain constant across all graph launches. This is one of the most impactful constraints because changing shapes typically affects both kernel parameters (grid dimensions, memory access patterns) and control flow (loop bounds, conditional operations).

When data shapes change:

- **Kernel configurations change**: Different shapes require different grid/block dimensions and shared memory sizes, violating the static kernel parameter constraint
- **Memory access patterns change**: Strides and indexing calculations differ, which are fixed in the captured kernels
- **Control flow may change**: Dynamic shapes often lead to data-dependent loops or conditionals (e.g., "process N elements where N varies"), violating the static topology constraint
- **New memory allocations may be needed**: A new shape may require allocating buffers of different sizes. While relaxed capture mode allows allocations, memory addresses can change across launches, breaking any kernels that depend on fixed addresses

**Workarounds**: Common solutions include padding to fixed sizes, bucketing (multiple graphs for different shape ranges), or falling back to eager execution for rare shapes. See [Dynamic Shapes](../torch-cuda-graph/handling-dynamic-patterns.md#dynamic-shapes) for detailed strategies and examples.

## 4. Self-Contained Stream Capture

When using stream capture, the captured graph must form a self-contained execution unit with well-defined boundaries. The graph must originate from a single capturing stream, and all work must eventually synchronize back to that stream. By default, you cannot create dependencies on external streams or events that are not part of the capture context—though external event nodes provide an exception for cross-graph synchronization.

### Stream Fork-Join Model

During capture, work can be distributed across multiple streams (fork), but all streams must eventually synchronize back (join) to the original capturing stream before capture ends. The captured graph forms a self-contained execution unit where all dependencies are internal to the graph. This means:

- **Fork**: You can issue work to child streams from the capturing stream during capture
- **Join**: All forked streams must synchronize back to the capturing stream before `cudaStreamEndCapture()`
- **No external waits by default**: You cannot wait on events created outside the capture context, unless using external event flags (see below)
- **Disconnected streams are not captured**: Operations issued within the capture region but on streams that don't fork from and join back to the capturing stream are not included in the graph—the captured graph simply won't contain those operations

### External Events and Streams

By default, a graph cannot have dependencies on events or streams that exist outside its capture context:

```cpp
// This is NOT allowed by default
cudaEvent_t event;
cudaEventCreate(&event);
// ... event recorded on some other stream ...

cudaStreamBeginCapture(stream, ...);
cudaStreamWaitEvent(stream, event, 0);  // ❌ Error: external dependency
// ... capture more work ...
cudaStreamEndCapture(stream, &graph);
```

The event was not captured as part of the graph, so the graph cannot depend on it.

**External Event Nodes**: Normally, events captured during stream capture create **edges** (dependencies between nodes) in the graph. However, CUDA provides special flags to create **event nodes** instead of edges, enabling cross-graph synchronization:

- `cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal)` — inserts an event record **node** into the graph
- `cudaStreamWaitEvent(stream, event, cudaEventWaitExternal)` — inserts an event wait **node** into the graph

These flags are only valid during stream capture. The event handle must remain valid for the lifetime of the graph.

```cpp
// External events allow cross-graph synchronization
cudaEvent_t sync_event;
cudaEventCreate(&sync_event);

// Graph 1: records the event
cudaStreamBeginCapture(stream1, ...);
// ... do work ...
cudaEventRecordWithFlags(sync_event, stream1, cudaEventRecordExternal);  // ✓ Event record node
cudaStreamEndCapture(stream1, &graph1);

// Graph 2: waits on the event
cudaStreamBeginCapture(stream2, ...);
cudaStreamWaitEvent(stream2, sync_event, cudaEventWaitExternal);  // ✓ Event wait node
// ... do work after sync ...
cudaStreamEndCapture(stream2, &graph2);
```

This enables scenarios like cross-graph synchronization, pipelining between graphs, or signaling completion from one graph to another.

### Self-Contained Execution

This constraint ensures that a graph is a self-contained unit of work with no hidden external dependencies. When you launch a graph, it should not depend on any state or events from outside the graph itself—unless external events are used between graphs. This independence is what allows graphs to be launched efficiently and reliably—the driver knows exactly what needs to execute and in what order, without having to coordinate with external asynchronous work.

## 5. CPU Code Is Not Captured

CUDA Graphs eliminate runtime overhead by defining GPU operations once and launching them without CPU involvement. Only GPU operations (kernel launches, memory copies, etc.) are included in the graph—CPU code that runs during graph definition (whether via stream capture or explicit construction) is **not** part of the graph and will **not** execute during graph launch, unless explicitly added as a host function node via `cudaLaunchHostFunc()`.

A common pitfall is host state mutation inside the graph definition region: if CPU code updates a variable that is read after graph launch, that variable will only be updated once (during definition) and remain stale on all subsequent launches.

For CPU code that must execute on every launch, you have several options: move the code outside the graph definition region, move the logic to the GPU, or wrap it as a host function node using `cudaLaunchHostFunc()`. Host function nodes execute on each graph launch, but the callback cannot call CUDA APIs. See the `cudaLaunchHostFunc()` [documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g05841eaa5f90f27124241baafb3e856f) for additional restrictions.

## 6. Multi-Threading and Capture Modes

During stream capture, certain operations like `cudaMalloc()` are considered "potentially unsafe" because they execute immediately rather than being enqueued to a stream. If captured work depends on these operations, the graph would be invalid when launched. The `cudaStreamCaptureMode` controls how CUDA restricts these unsafe operations across threads.

### Capture Modes

**`cudaStreamCaptureModeGlobal`** (default):

- If the local thread has an ongoing capture (not initiated with Relaxed mode), OR if any other thread has a concurrent capture initiated with Global mode, then this thread is prohibited from potentially unsafe API calls.
- Provides process-wide protection—most restrictive and safest.

**`cudaStreamCaptureModeThreadLocal`**:

- If the local thread has an ongoing capture (not initiated with Relaxed mode), only that thread is prohibited from potentially unsafe API calls.
- Concurrent captures in other threads are ignored, allowing independent per-thread captures.

**`cudaStreamCaptureModeRelaxed`**:

- The thread is not prohibited from potentially unsafe API calls.
- However, operations that necessarily conflict with capture (e.g., `cudaEventQuery()` on events recorded inside capture) are still forbidden.
- Use only when you need unsafe operations during capture and understand the risks.

**Recommendation**: Use `cudaStreamCaptureModeGlobal` unless you have specific multi-threading requirements. It prevents accidental interference that could invalidate the graph.

For detailed information on capture mode behavior, see [cudaThreadExchangeStreamCaptureMode](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85) in the CUDA Runtime API.

## 7. Memory Constraints

CUDA Graphs have specific requirements and restrictions regarding memory operations, both during capture and execution.

### Memory Allocation APIs

The following table summarizes which memory APIs can be used during graph capture:

| API | During Capture | Notes |
|-----|----------------|-------|
| `cudaMalloc()` | ❌ Prohibited | Returns `cudaErrorStreamCaptureUnsupported` in Global/ThreadLocal modes |
| `cudaFree()` | ❌ Prohibited | Returns `cudaErrorStreamCaptureUnsupported` in Global/ThreadLocal modes |
| `cudaMallocHost()` | ❌ Prohibited | Forbidden in **all** capture modes |
| `cudaFreeHost()` | ❌ Prohibited | Forbidden in **all** capture modes |
| `cudaMallocAsync()` | ✅ Captured | Becomes an allocation node in the graph |
| `cudaFreeAsync()` | ✅ Captured | Only for memory allocated via `cudaMallocAsync()` within the **same** capture |

:::{note}
In **Relaxed mode**, synchronous allocations (`cudaMalloc()`, `cudaFree()`) are allowed as side effects but are not captured into the graph. However, pinned memory APIs remain forbidden in all modes.
:::

### Memory Lifetime Requirements

Memory referenced by a graph has different lifetime requirements depending on how it was allocated.

**External memory** must remain valid for the graph's entire lifetime. This includes:

- Memory allocated **before** capture (via `cudaMalloc()`, `cudaMallocHost()`, etc.)
- Memory allocated **during** capture in Relaxed mode via `cudaMalloc()` (executes as side effect, not captured)

| Memory Type | Requirements |
|-------------|--------------|
| Device memory | Must remain allocated while graph executable exists. Freeing and then replaying causes undefined behavior |
| Host memory (pageable) | Buffer address is captured, not contents. Changes between replays are reflected in subsequent copies. For `cudaMemcpyAsync()`, CUDA stages through internal pinned buffer |
| Host memory (pinned) | Same as above, but transfers are truly async. Must synchronize before CPU writes to avoid race conditions. Must not be unpinned or freed while graph exists |

:::{warning}
Always destroy graph executables before freeing their referenced external memory.
:::

**Internal memory** (allocated with `cudaMallocAsync()` during capture) becomes part of the graph:

- Allocation executes **each time** the graph is replayed (fresh memory per replay)
- Memory is managed by the CUDA stream-ordered allocator
- If paired with `cudaFreeAsync()` in the same capture, memory is freed each replay
- Does not need to remain valid externally—the graph owns it

:::{note}
The key difference: `cudaMalloc()` in Relaxed mode allocates **once** during capture (side effect), while `cudaMallocAsync()` allocates on **every graph launch** (captured as graph node).
:::

**cudaFreeAsync restrictions during capture:**

- ✅ `cudaMallocAsync()` → use → `cudaFreeAsync()` (all within same capture)
- ❌ Memory allocated before capture → `cudaFreeAsync()` during capture → `cudaErrorInvalidValue`

### Stream-Ordered Allocation in Graphs

When `cudaMallocAsync()` and `cudaFreeAsync()` are used during stream capture, they become **memory allocation nodes** and **memory free nodes** in the graph. The graph then manages these allocations internally, with optimizations for memory reuse and pooling.

For the full details on graph memory node behavior, optimizations, and advanced usage, see the [CUDA Programming Guide: Graph Memory Nodes](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#graph-memory-nodes).

## 8. Multi-Device Considerations

CUDA Graphs can include operations that span multiple devices.

### Multi-GPU Operations

CUDA Graphs support multi-GPU operations within a single graph. For example:

- **Peer-to-peer (P2P) memory copies**: `cudaMemcpyPeer` operations can be captured to copy data between GPUs
- **NCCL collectives**: Operations like AllReduce can be captured, allowing you to graph an entire multi-GPU training iteration including gradient synchronization
- **Cross-device memory operations**: Copies involving memory on different devices can be included

For distributed workloads, the typical pattern is:

- Each GPU captures its own graph containing both local operations (forward/backward) and multi-GPU operations (NCCL collectives)
- All participating GPUs launch their respective graphs, optionally synchronized via CPU-side orchestration
- NCCL operations within the graphs communicate across devices as part of the graph execution

:::{warning} NCCL Requirements

NCCL 2.9.6+ is required for capturing NCCL collectives in CUDA Graphs. Earlier versions do not support graph capture for collectives.
:::

:::{note} NCCL Buffer Registration with Multi-Segment Buffers

NCCL can register user buffers to improve communication performance. However, if a buffer spans multiple non-contiguous physical memory segments (common with memory allocators that extend segments on demand), older NCCL versions only support registering a single physical segment. Multi-segment buffers may cause issues like illegal memory access during CUDA graph capture or launch. If you encounter problems, set `NCCL_GRAPH_REGISTER=0` to disable buffer registration.
:::

For detailed information on using NCCL with CUDA Graphs, including buffer registration for better performance and other special considerations, see the [NCCL User Guide on Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html).

### Conditional Node Restrictions

While general CUDA graphs can span multiple devices, **body graphs of conditional nodes** have stricter requirements—all nodes within a conditional node's body graph must reside on a single device.

For more information on multi-device considerations, see the [CUDA Programming Guide on CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## 9. Other Restrictions

Beyond the constraints covered above, the CUDA runtime prohibits additional operations during graph capture. If you attempt to use a forbidden operation, CUDA will return an error such as `cudaErrorStreamCaptureUnsupported` or `cudaErrorStreamCaptureInvalidated`.

For the complete list of restrictions, refer to the [CUDA Programming Guide on Stream Capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture).

## What's Next?

For comprehensive details on these limitations, see:

- [CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)
- [CUDA Programming Guide - Stream Capture](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#stream-capture)

Now that you understand the constraints, explore:

- **[Quantitative Benefits](quantitative-benefits.md)**: Learn how to quantify and estimate CUDA Graph performance benefits for your specific application with concrete metrics and break-even analysis
