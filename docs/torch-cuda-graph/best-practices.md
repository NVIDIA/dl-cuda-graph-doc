<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Best Practices for PyTorch CUDA Graphs

:::{note}

This section provides practical best practices and recommendations for successfully using CUDA Graphs in PyTorch applications.
:::

## General Best Practices

This guide provides a systematic approach to successfully adopting CUDA Graphs in PyTorch applications.

### 1. Quantify the Benefits First

Before investing effort in CUDA graphs, measure whether they'll help your workload. Check if your GPU utilization is low (<80%) indicating CPU bottleneck, and count how many kernels are launched per iteration. Workloads with low GPU utilization and many small kernel launches are strong candidates for CUDA graphs.

For detailed measurement methods including using `nvidia-smi`, `nsys`, and estimating expected speedups, see [Quantitative Benefits](../cuda-graph-basics/quantitative-benefits.md).

### 2. Identify Idle GPU Regions

Profile your application with Nsight Systems and examine the GPU timeline to identify where the GPU sits idle waiting for CPU. In the timeline, look for:

- **Long gaps between kernels**: GPU idle while CPU prepares the next launch
- **Many small kernels**: Each kernel <50 μs, densely packed but with visible gaps
- **Low kernel row occupancy**: Timeline shows sparse execution

These are prime candidates for CUDA graphs. Conversely, skip regions with large, long-running kernels (>1ms) and sustained high utilization—they're already efficient.

:::{tip} Optional: Use NVTX Annotations

Annotate your code with `torch.cuda.nvtx.range()` to add meaningful labels in the timeline:

```python
with torch.cuda.nvtx.range("transformer_layer_5"):
    output = layer(input)
```

These ranges appear in Nsight Systems, making it easy to correlate idle GPU time with specific code regions.
:::

### 3. Choose the Right Capture Scope

Start small and expand progressively to minimize risk and debugging complexity. **Incremental approach**:

1. **Start with a single module/layer**: Begin with something small and self-contained, like one transformer layer, attention block, or MLP module. Verify it works correctly before moving forward. This initial step is easy to debug, quick to validate, and carries minimal risk—if something goes wrong, you know exactly where. For example, try graphing just the first layer.

2. **Expand to multiple layers**: Once you have one layer working successfully, extend to 2-3 layers, then gradually include more. Watch for operations that happen between layers, such as layer normalization or residual connections—these need to either be included in the graph or handled carefully. You might graph layers 0-5 while leaving the rest in eager mode, which gives you partial speedup while maintaining safety.

3. **Graph the full forward pass**: When all layers are compatible, capture the entire forward computation as one graph. Exclude data loading, preprocessing, and post-processing, which often involve dynamic behavior or lots of CPU operators. The main challenge at this stage is ensuring your forward path has no synchronization or dynamic patterns.

4. **Add backward pass**: For training workloads, extend graphing to include gradient computation, which typically consists of just the `loss.backward()` call. You have two options: capture backward as a **separate graph** from forward (giving you independent control over each), or capture **forward and backward together** in a single graph for maximum efficiency. If using `torch.cuda.make_graphed_callables()`, it captures forward and backward in a single call—you pass a module/function, and it returns a graphed version that handles both passes automatically with separate internal graphs. For Megatron-LM users, FullCudaGraphWrapper similarly captures forward and backward together across all microbatches. For manual control, use `torch.cuda.graph()` to explicitly decide what gets captured in each graph.

5. **Full iteration with optimizer** (very advanced): The most aggressive approach captures nearly everything in a training iteration—forward, backward, gradient scaling, gradient clipping, and optimizer step—as a single graph. When successful, this can achieve GPU utilization close to 100% by eliminating virtually all CPU launch overhead, delivering maximum performance boost. However, this requires making all these components CUDA graph compatible, which demands significant effort and careful code restructuring. This level of graphing is rare and highly advanced, requiring deep understanding of both CUDA graphs and your application internals.

**Where to focus your graphing efforts**:

Prioritize regions that will benefit most from reduced launch overhead. Note that carefully excluding low-value regions is **optional**—sometimes including everything is simpler than selective exclusion, even if some parts don't benefit much.

**High-value targets** (focus here first):

- Regions with GPU utilization <80% (CPU overhead dominates)
- Code with many small kernels (<50 μs each)—each launch saved reduces overhead
- Transformer layers, attention, convolutions—but only if they show low GPU utilization

**Low-value regions** (can skip or include based on convenience):

- Data loading/preprocessing—often has dynamic shapes or CPU operations, may be easier to exclude
- Loss computation/logging—involves CPU sync, usually small overhead if included
- Already GPU-bound regions (>95% utilization)—graphing won't help too much

:::{tip} Rule of Thumb: The 80/20 Principle

For most applications, graph the code that takes 80% of GPU time and is graph-compatible—don't force the remaining 20% if it has dynamic behavior or CPU sync, as the effort often isn't worth the marginal gains.

However, for **performance-critical applications** (production inference serving, large-scale training running for weeks), investing the effort to graph everything—even low-value regions—can be worthwhile, as the cumulative benefits compound over millions of iterations.
:::

### 4. Choose the Right API

PyTorch and frameworks provide multiple CUDA graph APIs at different levels of abstraction. Choose based on your framework and performance needs.

**For Transformer Engine / Megatron-LM users** (try these first):

1. **Transformer Engine `make_graphed_callables`**: Manual per-callable graphing with FP8 and pipeline parallelism support. Works with any PyTorch model, making it ideal for custom models or when you need fine-grained control over what gets graphed. Requires manual PP scheduling via the `_order` parameter. See [Transformer Engine `make_graphed_callables`](te-megatron-cuda-graphs.md#transformer-engine-make_graphed_callables) for usage and comparison.

2. **Megatron CudaGraphManager**: Automatic per-layer graphing designed for Megatron-LM models. Works seamlessly with all parallelism strategies including pipeline parallelism, and handles FP8 automatically. Enable via `TransformerConfig(enable_cuda_graph=True)`. This is the recommended default for Megatron training. See [CudaGraphManager](te-megatron-cuda-graphs.md#cudagraphmanager-per-layer-graphing) for details.

3. **Megatron FullCudaGraphWrapper**: Captures forward and backward passes across all microbatches as a single graph for maximum performance. Supports all parallelism configurations with automatic FP8 handling. Enable via `TransformerConfig(enable_cuda_graph=True, cuda_graph_scope="full_iteration")` or command-line `--cuda-graph-scope full_iteration`. See [FullCudaGraphWrapper](te-megatron-cuda-graphs.md#fullcudagraphwrapper-full-iteration-graphing) for comparison.

If you're using Transformer Engine or Megatron-LM with FP8/PP, start with these implementations before considering general PyTorch APIs. They're designed specifically for distributed training workloads with FP8 and pipeline parallelism.

**For general PyTorch users**:

4. **`torch.compile(mode="reduce-overhead")`**: Easiest option—automatic detection, zero manual effort, works out of the box. Trade-off is potential graph fragmentation (many small graphs). Good starting point if you're already using `torch.compile()`.

5. **`torch.cuda.make_graphed_callables()`**: For training with custom loops (non-FP8). Pass your model/function, get back a graphed version that handles forward and backward automatically. More manual than compile but gives cleaner graphs.

6. **`torch.cuda.graph()`**: For maximum control. Manually manage capture, warmup, and replay. Most flexible but requires the most code. Use when you need fine-grained control over what gets graphed.

For detailed information on PyTorch's CUDA graph APIs, including usage examples and PyTorch-specific constraints, see [PyTorch Integration](torch-integration.md).

:::{tip} API Selection Strategy

Start with the highest-level API available for your framework (Megatron implementations for Megatron-LM, `torch.compile` for general PyTorch). Move to lower-level APIs if you need more control, hit limitations with automatic approaches, or don't achieve the expected performance improvement (e.g., due to excessive graph fragmentation).
:::

### 5. Ensure CUDA Graph Compatibility

Once you've selected your capture scope, the next step is making the code within that range CUDA graph compatible. This involves reviewing your code against CUDA graph constraints and fixing any violations.

**Step 1: Check CUDA-level constraints**

Review the [CUDA Graph Constraints](../cuda-graph-basics/constraints.md) documentation and verify your code doesn't violate fundamental rules. Key examples include:

- ✅ No forbidden CUDA API calls during capture
- ✅ No CPU-GPU synchronization operations
- ✅ Static control flow, static memory addresses, static kernel parameters, static shapes

See the full constraints documentation for the complete list. Identify and fix any violations in your selected scope before proceeding to PyTorch-specific considerations.

**Step 2: Check PyTorch-specific constraints**

Review the [PyTorch-Specific Constraints](torch-integration.md#pytorch-specific-constraints) and ensure you meet the requirements. Top considerations:

- ✅ Adequate warmup iterations (3+ standard, 11 for DDP)
- ✅ AMP cache disabled if using autocast
- ✅ Custom RNG generators registered with graph-safe APIs
- ✅ Proper DDP setup for distributed training

See the full PyTorch constraints documentation for additional requirements. Fix any PyTorch-specific issues in your code.

For detailed guidance on writing CUDA graph-compatible PyTorch code with common patterns and fixes, see [Writing CUDA Graph-Compatible Code](#writing-cuda-graph-compatible-code) below.

### 6. Handle Common Issues

When working with CUDA Graphs, you'll probably encounter various types of issues. Understanding the categories helps you debug efficiently.

**Capture failures**: Graph capture fails with an explicit error message:

- e.g. "CUDA error: operation failed due to a previous error during capture" → Check for forbidden operations (e.g. CPU sync)
- Read PyTorch's error message carefully—it often tells you exactly what operation is forbidden
- See [Capture Failures](../troubleshooting/capture-failures.md) for specific errors and solutions

**Silent errors** (wrong results):

- Capture succeeds but outputs differ from eager execution—usually means some form of dynamic behavior is occurring
- Check for any source of dynamism: tensor reassignment, shape changes, data-dependent control flow, unregistered RNG
- See [Handling Dynamic Patterns](handling-dynamic-patterns.md) for solutions to common dynamic issues
- See [Numerical Errors](../troubleshooting/numerical-errors.md) for debugging wrong results

**Memory issues** (OOM or illegal memory access):

- Illegal memory access during replay → Input tensors freed or reassigned; use `.copy_()` and keep inputs alive
- OOM with multiple graphs → Enable memory pool sharing with `pool=graph1.pool()`
- OOM during capture → Free unused tensors before capture; enable `expandable_segments:True`
- See [Memory Issues](../troubleshooting/memory-issues.md) for comprehensive debugging

**Performance issues**:

- Graphs work correctly but speedup is minimal or negative
- Too many small graphs → Gradually expand your capture range to create fewer, larger graphs
- Low speedup → Profile to confirm you're CPU-bound
- See [Performance Issues](../troubleshooting/performance-issues.md) for optimization

Beyond debugging actual errors, you'll encounter common code patterns that aren't immediately compatible with CUDA graphs but have well-known workarounds:

**Common incompatible patterns**:

- Gradient clipping, mixed precision training, variable batch sizes, logging—these often appear as the root cause in the failure categories above
- Most have established solutions and patterns that let you work with or around the limitations
- See [Handling Dynamic Patterns](handling-dynamic-patterns.md) for detailed guidance on handling these obstacles

For systematic debugging approach covering all issue types, start with [Debugging Strategies](../troubleshooting/debugging-strategies.md).

### 7. Verify Correctness

Always validate that your CUDA graph implementation produces correct results by comparing against eager mode execution.

**Compare critical tensors**: Run both eager and graphed versions on the same input and compare key tensors—loss values, gradients, and model parameters. Use `torch.allclose()` with appropriate tolerances (e.g., rtol=1e-5 for FP32). For consistent comparisons, you may need to enable deterministic mode. See [PyTorch Reproducibility documentation](https://pytorch.org/docs/stable/notes/randomness.html) for setting `torch.use_deterministic_algorithms(True)` and related configurations.

**Additional validation**:

- **Multiple test inputs**: Verify on diverse data samples, not just one
- **Edge cases**: Test minimum/maximum batch sizes if using bucketing
- **Gradient verification**: For training graphs, ensure gradients match between eager and graphed execution
- **Loss curve monitoring**: During training, watch for divergence which indicates a correctness issue
- **Validation/evaluation metrics**: Run validation with and without graphs—metrics should be very close

Once verification passes across all these checks, congratulations! Your CUDA graph implementation is working correctly. You can now enjoy the performance benefits with confidence.

## Writing CUDA Graph-Compatible Code

To successfully write CUDA graph-compatible PyTorch code, three principles are most fundamental among all CUDA graph constraints:

:::{tip} Principle 1: GPU-ONLY - Keep Everything on GPU

Only GPU operations are captured (except CUDA host functions via `cudaLaunchHostFunc`). CPU-side code is executed during capture but eliminated during replay.
:::

:::{tip} Principle 2: ASYNC - Keep Everything Asynchronous

No CPU-GPU synchronization. The CPU queues work continuously without waiting for GPU results.
:::

:::{tip} Principle 3: STATIC - Keep Everything Static

No runtime variability. Operations, control flow, memory addresses, scalars, and shapes must be fixed across all replays.
:::

Here's a practical guide for PyTorch users without deep CUDA experience:

### Keep Everything on GPU (GPU-Only)

During graph capture, CUDA records only GPU operations. All CPU-side code (Python logic, CPU computations, I/O) is executed during capture but **completely eliminated during graph replay**. The graph replays only the GPU kernels that were launched during capture.

**What this means in practice**:

- ❌ File I/O: `data = torch.load("file.pt")` won't reload on each replay
- ❌ CPU preprocessing: `tokens = tokenizer.encode(text)` won't re-tokenize
- ❌ Logging: `print(f"Step {i}")` won't print during replay
- ❌ CPU computations: `threshold = compute_threshold()` won't recompute
- ❌ Random number generation on CPU: `random.randint(0, 10)` won't regenerate
- ❌ **CPU bookkeeping inside graph**: `buffer.append(tensor)` won't populate buffers during replay
- ✅ Move all CPU-side operations (I/O, preprocessing, logging) outside the graphed region
- ✅ Perform data loading and preprocessing before graph replay
- ✅ Use GPU-side operations for anything that must vary across replays

**Why it matters**: Graph replay is purely GPU-side kernel execution. Any CPU work you need on each iteration must be moved outside the graph, either before or after replay. **Critically, code outside the graph cannot depend on CPU code inside the graphed region** - for example, if you accumulate tensors into Python lists during the graph and use those lists outside the graph, those lists may not be correct during replay.

### Keep Everything Asynchronous (Sync-Free)

Your code should never wait for GPU operations to complete. The CPU continuously queues work without checking results.

**What this means in practice**:

- ❌ Don't use `.item()` to get scalar values for logic: `if loss.item() > threshold:`
- ❌ Don't use `.cpu()` to move tensors for inspection: `x_cpu = x.cpu()`
- ❌ Don't print GPU tensor values: `print(f"Loss: {loss}")`
- ✅ Keep all operations on GPU until absolutely necessary
- ✅ Move any value inspection outside the graphed region

**Why it matters**: Every `.item()` or `.cpu()` call blocks the CPU waiting for GPU, which (a) hurts performance and (b) cannot be captured in a graph. See [Writing Sync-Free Code](sync-free-code.md) for comprehensive guidance.

### Keep Everything Static

All aspects of your computation must be determined before capture and remain fixed across replays.

**What must be static**:

- **Operations**: Same sequence of operations every time
- **Control flow**: Execution path must be identical (no `if` or `for` loop based on tensor values)
- **Memory addresses**: Reuse the same tensor objects via `.copy_()`, never create new ones
- **Shapes**: Tensor dimensions cannot change (e.g. `batch_size=32` always, not variable)

**What this means in practice**:

- ❌ Data-dependent branching: `if x.sum() > 0: path_a() else: path_b()`
- ❌ Dynamic loops: `for i in range(x.size(0))` where size varies
- ❌ Tensor reassignment: `input = new_tensor` (changes address of input tensor)
- ❌ Variable batch sizes or sequence length: different sized inputs across iterations
- ❌ Global buffer deletion/reassignment: `del buffer[key]` or `buffer[key] = new_tensor` (global tensors used within graphs are graph inputs)
- ✅ Always-executed paths or GPU-side conditionals (`torch.where()`)
- ✅ Fixed-count loops with static ranges
- ✅ Update via `.copy_()`: `input.copy_(new_data)` (same address)
- ✅ Fixed batch sizes / sequence length or bucketing (graph per size)
- ✅ Global buffer in-place updates: `buffer[key].copy_(data)` (same memory addresses)

## What's Next?

**Within PyTorch CUDA Graphs**:

- **[Writing Sync-Free Code](sync-free-code.md)**: Identify and eliminate host-device synchronizations in PyTorch
- **[Handling Dynamic Patterns](handling-dynamic-patterns.md)**: Workarounds for gradient clipping, mixed precision, variable shapes, and other common patterns
- **[Quick Checklist](quick-checklist.md)**: Verify your code is ready for CUDA Graph capture
- **[PyTorch Integration](torch-integration.md)**: Deep dive into PyTorch's CUDA Graph APIs and constraints
- **[Transformer Engine and Megatron-LM CUDA Graphs](te-megatron-cuda-graphs.md)**: CudaGraphManager and FullCudaGraphWrapper for distributed training

**Other sections**:

- **[CUDA Graph Basics](../cuda-graph-basics/introduction.md)**: Fundamentals of how CUDA Graphs work
- **[Examples](../examples/introduction.md)**: Real-world implementations in RNN-T, Stable Diffusion, and Llama 3.1 405B
- **[Troubleshooting](../troubleshooting/debugging-strategies.md)**: Debug capture failures, silent errors, and performance issues
- **[Reference](../reference.md)**: Additional resources and documentation links
