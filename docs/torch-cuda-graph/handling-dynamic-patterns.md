<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Handling Dynamic Patterns

:::{note}

This section provides practical patterns for handling common dynamic behaviors when adopting CUDA Graphs in PyTorch.
:::

CUDA Graphs traditionally require static execution patterns—the same sequence of operations with the same memory addresses every replay. Recent CUDA versions (12.3+) introduced conditional nodes that support limited dynamic control flow at the graph level, but as of PyTorch 2.9, these features are not yet exposed through PyTorch's Python APIs. This document focuses on practical patterns for adapting real-world code with dynamic behavior to work with current PyTorch CUDA graph implementations.

## Types of Dynamic Patterns

Dynamic patterns fall into four categories:

1. **Dynamic control flow**: Conditional execution based on runtime GPU values that requires checking values on the CPU.

2. **Dynamic tensors**: Tensors whose memory addresses change across iterations, requiring new allocations or reallocations.

3. **Dynamic scalars**: Scalar values that change each iteration and must be correctly passed to GPU operations.

4. **Dynamic shapes**: Input dimensions that vary across iterations, involving both different kernel launches and dynamic memory allocation.

For each pattern, we'll explore solutions ranging from simple workarounds to advanced techniques. When these patterns cannot be easily resolved, the last resort is to **move them outside the CUDA graph capture range** and execute them in eager mode.

## Dynamic Control Flow

Dynamic control flow refers to conditional execution based on runtime GPU values that requires checking values on the CPU. These patterns are problematic because checking GPU values in Python requires CPU-GPU synchronization (usually through `.item()` or implicit `.is_nonzero()`), which breaks CUDA graph capture.

### Common Patterns and Why They Break

Dynamic control flow appears in many real-world scenarios. The following code snippets illustrate common patterns that break graph capture:

- **Training stability checks**: Detecting NaN/Inf loss values to skip corrupted iterations
- **Gradient management**: Clipping gradients when norms exceed thresholds to prevent exploding gradients
- **Mixed precision training**: Conditionally skipping optimizer steps when gradients contain invalid values
- **Adaptive inference**: Early exit mechanisms that route simple inputs through cheaper paths while complex inputs use the full network

All these patterns share a common problem: Python needs to inspect GPU-computed values to make branching decisions, forcing CPU-GPU synchronization.

**Example 1: NaN/Inf Loss Detection**

Training stability often requires checking for invalid loss values. However, checking whether a GPU tensor contains NaN or Inf requires moving that information to the CPU, causing synchronization.

```python
# ❌ Breaks graph capture - synchronization on loss.isnan()
    loss = criterion(output, target)
if loss.isnan() or loss.isinf():
    print("Invalid loss, skipping iteration")
    continue
loss.backward()
```

**Example 2: Gradient Clipping**

Gradient clipping prevents exploding gradients by computing the global norm and scaling gradients when the norm exceeds a threshold. The `clip_grad_norm_()` function returns the computed norm to the CPU, triggering synchronization.

```python
# ❌ Breaks graph capture - synchronization in clip_grad_norm_
    loss.backward()

# What clip_grad_norm_() does internally (simplified):
# 1. Compute total gradient norm across all parameters
total_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters()))
# 2. Check if clipping is needed
# This comparison implicitly calls .is_nonzero() - synchronization point!
if total_norm > max_norm:
    clip_coef = max_norm / total_norm
    for p in model.parameters():
        p.grad.mul_(clip_coef)

optimizer.step()
```

**Example 3: Conditional Optimizer Step (AMP)**

Automatic Mixed Precision (AMP) training uses gradient scaling to prevent underflow in fp16. The gradient scaler checks for NaN/Inf gradients and conditionally skips the optimizer step, which involves checking GPU values on the CPU.

```python
# ❌ Breaks graph capture - scaler.step() checks for NaN/Inf internally
scaler.scale(loss).backward()

# What scaler.step() does internally (simplified):
# See: https://github.com/pytorch/pytorch/blob/v2.4.0/torch/amp/grad_scaler.py#L343
# 1. Unscale gradients and check for infs/nans
optimizer_state = scaler._per_optimizer_states[id(optimizer)]
inf_grad_tensor = optimizer_state["found_inf_per_device"]
# 2. Check if any inf/nan was found - this calls .item()!
if inf_grad_tensor.item() == 0:  # Synchronization point
    # No inf/nan found, safe to step
    optimizer.step()
# Otherwise skip the optimizer step

scaler.update()
```

**Example 4: Early Exit and Adaptive Inference**

Some models use early exit mechanisms where simpler inputs can exit through earlier, cheaper layers while complex inputs go through the full network. The decision to exit early requires checking confidence scores on the CPU, causing synchronization.

```python
# ❌ Breaks graph capture - early exit based on confidence threshold
# Example: Early exit neural network (like BERTExit, DeeBERT)
hidden = early_layers(input)
exit_logits = early_exit_classifier(hidden)
confidence = exit_logits.softmax(dim=-1).max().item()  # Synchronization point

if confidence > early_exit_threshold:
    # High confidence - exit early (cheaper)
    output = exit_logits
else:
    # Low confidence - continue through full model (expensive)
    hidden = remaining_layers(hidden)
    output = final_classifier(hidden)

loss = criterion(output, target)
```

### General Solutions

The following subsections demonstrate specific solutions for each dynamic control flow pattern above. However, most solutions follow three general strategies that can be applied across different scenarios:

1. **Use GPU-native alternatives**: Replace Python conditionals with GPU operations like `torch.where()`, `torch.clamp()`, or `torch.masked_fill_()` that keep computation on GPU without synchronization.

2. **Implement custom CUDA kernels**: For non-trivial logic, write CUDA kernels that perform conditional operations entirely on GPU. This is worthwhile when the control flow is moderately complex but can be expressed in CUDA.

3. **Move outside graph with partial capture**: For complex dynamic control flow (e.g., dispatching to different models, multi-path execution), keep the decision-making outside the graph. Capture individual paths as separate graphs and select which to replay. This approach benefits from graph trees or checkpointing techniques to manage multiple graph instances efficiently.

:::{note} Future: CUDA Conditional Nodes

CUDA 12.3+ introduced conditional nodes (IF, WHILE, and SWITCH in CUDA 12.8) that allow limited branching within graphs without CPU synchronization. These enable GPU-side conditionals based on device memory values. However, PyTorch does not yet expose these features through its Python APIs. For now, use the workarounds described below. Future PyTorch versions may support conditional nodes, enabling more flexible dynamic control flow within graphs.

For more information about CUDA conditional nodes, see [Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/) on the NVIDIA Developer Blog.
:::

### Example 1: Gradient Clipping

PyTorch's `clip_grad_norm_` implementation evolution provides an excellent example of how to remove synchronization and dynamic control flow to make code CUDA graph compatible.

**Problem: PyTorch 1.0 implementation with synchronization**

The [PyTorch 1.0 `clip_grad_norm_` implementation](https://github.com/pytorch/pytorch/blob/v1.0.0/torch/nn/utils/clip_grad.py#L34) contained explicit synchronization:

```python
# Simplified from PyTorch 1.0 clip_grad_norm_ implementation
total_norm = 0
for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type  # Synchronization here!
total_norm = total_norm ** (1. / norm_type)

clip_coef = max_norm / (total_norm + 1e-6)
if clip_coef < 1:  # Dynamic control flow - different code paths
    for p in parameters:
        p.grad.data.mul_(clip_coef)
```

This version had a critical synchronization point: calling `.item()` during norm computation, which moves the GPU value to CPU. Once `total_norm` is a CPU scalar, the subsequent `if clip_coef < 1:` conditional executes on CPU without additional sync, but it creates dynamic control flow—different code paths execute depending on runtime values.

**Solution: PyTorch 1.13+ sync-free implementation**

The [PyTorch 1.13 `clip_grad_norm_` implementation](https://github.com/pytorch/pytorch/blob/v1.13.1/torch/nn/utils/clip_grad.py#L56) eliminates synchronization and converts dynamic control flow into pure GPU operations:

```python
# Simplified from PyTorch 1.13+ clip_grad_norm_ implementation
total_norm = torch.norm(
    torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
    norm_type
)
clip_coef = max_norm / (total_norm + 1e-6)
# Clamp avoids the conditional - always multiply, no sync needed
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
for g in grads:
    g.detach().mul_(clip_coef_clamped)
```

**Key techniques PyTorch used**:

1. **Keep computation on GPU**: Compute `total_norm` entirely on GPU as a tensor, avoiding `.item()` that would sync.

2. **Replace conditionals with GPU operations**: Use `torch.clamp(clip_coef, max=1.0)` instead of `if clip_coef < 1:`. This eliminates the conditional branch—the code always executes the same path (multiplying gradients), making the control flow static.

3. **Accept redundant work**: Always multiply gradients by the clamped coefficient, even when it equals 1.0. This redundant multiplication is cheap and maintains static execution.

**Applying this pattern to your code**:

When you encounter similar synchronization or dynamic control flow issues, apply the same principles: keep values on GPU, replace Python conditionals with GPU operations like `torch.where()` or `torch.clamp()`, and accept redundant computation to maintain static control flow. For complex logic that can't be expressed with PyTorch operations, consider implementing a custom CUDA kernel that performs the entire operation on GPU. If the logic is too complex even for a custom kernel, use partial capture—keep the dynamic control flow outside the graph and capture only the static computation regions.

### Example 2: Early Exit and Adaptive Inference

Recall **Example 4** above where early exit models check confidence and conditionally route through different paths. This creates dynamic control flow that breaks graph capture. Here are two approaches to make this pattern CUDA graph compatible.

:::{note} Simplified Code Examples

The code examples below are simplified to illustrate the core logic. Production implementations require additional warmup iterations, proper error handling, and memory management.
:::

**Solution 1 - Capture shared layers once, replay remaining layers conditionally**:

This approach captures the early layers and remaining layers as separate graphs, then replays them based on confidence.

```python
# Capture early layers + early exit classifier (always executed)
graph_early = torch.cuda.CUDAGraph()
static_input = torch.zeros_like(input)
static_hidden = None
static_early_logits = None

with torch.cuda.graph(graph_early):
    static_hidden = early_layers(static_input)
    static_early_logits = early_exit_classifier(static_hidden)

# Capture remaining layers (conditional)
graph_remaining = torch.cuda.CUDAGraph()
static_hidden_remaining = torch.zeros_like(static_hidden)
static_full_output = None

with torch.cuda.graph(graph_remaining):
    static_full_output = remaining_layers(static_hidden_remaining)

# Inference loop
for input in dataloader:
    # Always execute early layers + early exit classifier
    static_input.copy_(input)
    graph_early.replay()

    # Check confidence outside graph - .item() call here
    confidence = static_early_logits.max().item()

    if confidence > threshold:
        # High confidence - use early exit
        output = static_early_logits
    else:
        # Low confidence - continue with remaining layers
        static_hidden_remaining.copy_(static_hidden)
        graph_remaining.replay()
        output = static_full_output
```

- **Pros**: Memory efficient (shared early layers), flexible routing decision
- **Cons**: One sync per iteration for confidence check

**Solution 2 - Capture complete paths separately**:

This approach captures the full early exit path and full model path as separate end-to-end graphs.

```python
# Capture early exit path
graph_early_exit = torch.cuda.CUDAGraph()
static_input_early = torch.zeros_like(input)
static_early_output = None

with torch.cuda.graph(graph_early_exit):
    hidden = early_layers(static_input_early)
    static_early_output = early_exit_classifier(hidden)

# Capture full model path
graph_full = torch.cuda.CUDAGraph()
static_input_full = torch.zeros_like(input)
static_full_output = None

with torch.cuda.graph(graph_full):
    hidden = early_layers(static_input_full)
    hidden = remaining_layers(hidden)
    static_full_output = final_classifier(hidden)

# Inference loop with heuristic-based routing (no sync)
for input in dataloader:
    # Decide which path based on cheap heuristic
    use_early_exit = (input.norm() < threshold)

    if use_early_exit:
        static_input_early.copy_(input)
        graph_early_exit.replay()
        output = static_early_output
    else:
        static_input_full.copy_(input)
        graph_full.replay()
        output = static_full_output
```

- **Pros**: Fully captured computation, routing can be sync-free with cheap heuristics
- **Cons**: Higher memory (duplicate early layers), less flexible (can't check actual confidence without sync)

**Which solution to choose**:

- **Solution 1** if you need confidence-based routing and early layers are expensive
- **Solution 2** if you can use heuristic routing or want maximum speedup from full graph capture

Both patterns apply to hierarchical classification, cascaded models, and adaptive computation architectures.

## Dynamic Tensors

Dynamic tensors refer to **graph input tensors** whose **memory addresses** change across different executions of the graphed region. This primarily concerns graph input tensors—tensors created and used entirely within the graph maintain stable addresses thanks to PyTorch's caching allocator. Unlike tensors with changing values (which can be updated in-place with `.copy_()`), changing input tensor addresses break CUDA graphs because graphs record specific memory locations during capture.

In this context, **"static"** (or **"persistent"**) means a tensor remains allocated with the same memory address for the entire CUDA graph lifecycle (from capture through all replays). Dynamic tensors, by contrast, are recreated or reallocated, causing their addresses to change between iterations.

**Understanding which tensors can be dynamic**:

Not all tensors in your PyTorch program are graph inputs—only tensors that cross the graph boundary (coming from outside the graphed region) matter. The table below shows how different tensor types typically behave without and with CUDA graphs:

:::{note} Typical Cases

The behaviors shown are typical patterns. Actual behavior may vary depending on your specific implementation, model architecture, and training setup.
:::

| Tensor Type | Without CUDA Graph | With CUDA Graph | Notes |
|-------------|-------------------|-----------------|-------|
| **Model parameters** | Static (same address) | Static (same address) | Allocated once at initialization, reused across iterations |
| **Optimizer states** | Static (same address) | Static (same address) | Momentum buffers, Adam moments—created once, updated in-place |
| **Dataloader outputs** | Dynamic (new tensors) | **Graph input—must be static** | Most common issue: new objects each iteration. Solution: preallocate static buffers, use `.copy_()` |
| **Type-casted parameters** | May change | Usually inside graph | E.g., FP32→FP16 in AMP—typically handled within graph automatically. Only manual casting outside graph requires stable addresses |
| **Internal activations** | Dynamic (address may change) | Static (same address) | Without graphs: caching allocator may return different addresses. With graphs: capture fixes addresses, caching allocator ensures reuse |
| **External tensors** | May change | **Graph input—must be static** | Preprocessing results, global state—preallocate and use `.copy_()` |

**Key insight**: Internal tensors (created and consumed within the graphed region) are automatically handled by PyTorch's caching allocator during graph capture—their addresses become fixed. The challenge is **graph input tensors** (coming from outside): dataloader outputs, external preprocessing, or any tensor passing into the graph. These must be explicitly managed to maintain stable addresses across replays.

**Why this problem occurs**:

Dynamic input tensors are problematic because CUDA graphs record specific memory addresses during capture and expect those addresses to remain unchanged during replay. If tensor addresses change between replays, the graph continues to access the old (stale) memory addresses, leading to incorrect results or memory errors.

Graph input tensors can become dynamic in several scenarios:

1. **Unnoticed input tensors**: Users don't realize an input-changing tensor is being used inside the graphed region.

2. **Assignment instead of `copy_()`**: Using `tensor = new_tensor` instead of `tensor.copy_(new_tensor)` creates a new tensor object with a different address.

3. **Tensor address collections**: Tensors that store pointers to other tensors (e.g., grouped GEMM operations with lists of tensor addresses) can become invalid when the referenced tensors change addresses.

**Consequences**:

- **Silent correctness issues**: Model diverges or produces wrong results without obvious errors—the graph reads stale data or data from unrelated tensors at the old addresses
- **Illegal memory access**: If the old tensor is deallocated and its memory is recycled by PyTorch's caching allocator, the graph accesses freed memory, causing CUDA errors or crashes

### General Solutions

The key principle is to maintain stable memory addresses for all graph input tensors across replays:

1. **Identify all graph input tensors**: Carefully audit the graphed region to find every tensor that comes from outside. This includes obvious inputs (data, targets) and subtle ones (learning rates, counters, hyperparameters stored as tensors).

2. **Preallocate input tensors before capture**: Create all graph input tensors before capture and reuse them across all replays. Never create new input tensors during replay.

3. **Use `.copy_()` for updates**: Always update graph input tensors with `static_tensor.copy_(new_data)` instead of reassignment (`static_tensor = new_data`). The `.copy_()` method preserves the tensor's memory address while updating its contents.

4. **Wrap dataloader with static buffer (optional)**: For cleaner code, wrap your dataloader to automatically copy data into static buffers. This encapsulates the `.copy_()` logic and makes graph replay code simpler. See NeMo's [StaticBufferLoader implementation](https://github.com/NVIDIA-NeMo/NeMo/blob/v2.5.3/nemo/utils/callbacks/cuda_graph.py#L88) for a reference implementation.

5. **Be cautious with tensor address collections**: Some operations (e.g., grouped GEMM) use tensors that store pointers to other tensors. For these cases, ensure both the pointer-holding tensor and all referenced tensors maintain stable addresses across replays. The pointer-holding tensor itself must also be preallocated and reused.

The following subsections demonstrate these solutions for specific common patterns.

### Example 1: Global Statistics Tensor for Normalization

Some models maintain running statistics (e.g., mean/std for normalization, EMA of activations) as global tensors that are updated each iteration. These are easy to overlook as graph inputs.

**Problem - Unnoticed global tensor as graph input**:

```python
# Global running statistics for custom normalization layer
# Common in research code, video models, or online learning scenarios
running_mean = torch.zeros(num_features, device='cuda')
running_std = torch.ones(num_features, device='cuda')

class CustomNormLayer(nn.Module):
    def forward(self, x):
        # Uses global running statistics
        normalized = (x - running_mean) / (running_std + 1e-5)
        return normalized

model = nn.Sequential(CustomNormLayer(), ...)

# Capture the graph
g = torch.cuda.CUDAGraph()
static_input = torch.zeros(batch_size, num_features, device='cuda')

with torch.cuda.graph(g):
    output = model(static_input)

# During training - this breaks the graph!
for batch in dataloader:
    # Update statistics (common pattern: compute from current batch)
    running_mean = batch.mean(dim=0)  # ❌ New tensor with different address!
    running_std = batch.std(dim=0)    # ❌ New tensor with different address!

    static_input.copy_(batch)
    g.replay()  # Graph still uses OLD running_mean/running_std - incorrect normalization!
```

**Why it fails**: The graph captured the original `running_mean` and `running_std` addresses. When you reassign these variables to new tensors (from `.mean()` and `.std()`), the graph continues reading from the old addresses, which may contain stale data or be deallocated.

**Solution - Use `.copy_()` to update in-place**:

```python
# Preallocate statistics tensors before capture
running_mean = torch.zeros(num_features, device='cuda')
running_std = torch.ones(num_features, device='cuda')

# Capture graph (same as before)
with torch.cuda.graph(g):
    output = model(static_input)

# During training - update statistics in-place
for batch in dataloader:
    # Compute new statistics
    new_mean = batch.mean(dim=0)
    new_std = batch.std(dim=0)

    # Update at the SAME addresses
    running_mean.copy_(new_mean)  # ✅ Same address, updated value
    running_std.copy_(new_std)    # ✅ Same address, updated value

    static_input.copy_(batch)
    g.replay()  # Graph reads updated values at same addresses
```

**Key lesson**: Global tensors referenced inside the graphed region are graph inputs, even if they seem "internal" to the model. Always identify them before capture and update using `.copy_()` to preserve memory addresses.

### Example 2: Grouped GEMM in MOE Models

Mixture of Experts (MOE) models use a gating network to route each input token to a subset of "expert" sub-networks. Different tokens route to different experts, resulting in variable-sized inputs per expert. Grouped GEMM efficiently processes these variable-sized batches in one kernel call, performing multiple independent matrix multiplications where each pair can have different dimensions.

```python
# Pseudo code for grouped GEMM
def grouped_gemm(A_list, B_list):
    """
    Computes: [A_0 @ B_0, A_1 @ B_1, ..., A_n @ B_n]
    where A_i has shape [M_i, K_i] and B_i has shape [K_i, N_i]

    Implementation (simplified):
    1. Extract pointers to each A_i and B_i tensor
    2. Store pointers in host arrays (CPU memory)
    3. Copy pointer arrays from host to device
    4. Call cuBLAS grouped GEMM with device pointer arrays
    """
    n_groups = len(A_list)

    # Step 1: Extract tensor addresses into Python lists (host memory)
    A_ptrs = [A.data_ptr() for A in A_list]  # Python list of integers (CPU)
    B_ptrs = [B.data_ptr() for B in B_list]  # Python list of integers (CPU)

    # Step 2: Create host tensors with correct data type (int64 for pointers)
    A_ptrs_host = torch.tensor(A_ptrs, dtype=torch.int64, device='cpu')
    B_ptrs_host = torch.tensor(B_ptrs, dtype=torch.int64, device='cpu')

    # Step 3: Copy pointer arrays to device (non-blocking H2D copy - this gets captured!)
    A_ptrs_device = A_ptrs_host.to('cuda', non_blocking=True)  # Non-blocking H2D
    B_ptrs_device = B_ptrs_host.to('cuda', non_blocking=True)  # Non-blocking H2D

    # Step 4: Call CUDA kernel with device pointer arrays
    outputs = _cuda_grouped_gemm_kernel(A_ptrs_device, B_ptrs_device, ...)
    return outputs

    # Problem: A_ptrs_host and B_ptrs_host are DYNAMIC tensors - they are recreated
    # and recycled on every call. But the H2D copy makes them CUDA graph inputs!
    # Graph inputs must be STATIC (persistent) tensors with stable addresses.
```

If user tries to capture this Grouped GEMM function into a CUDA graph like this:

```python
# MOE training setup
num_experts = 8
hidden_dim = 1024
expert_sizes = [256, 512, 128, 1024, 256, 512, 256, 128]  # Variable sizes per expert

# Create expert input tensors
expert_inputs = [torch.randn(size, hidden_dim, device='cuda') for size in expert_sizes]
expert_weights = [torch.randn(hidden_dim, hidden_dim, device='cuda') for _ in range(num_experts)]

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    # Grouped GEMM captures addresses of expert_inputs
    outputs = grouped_gemm(expert_inputs, expert_weights)

# During training - update expert inputs and replay
for batch in dataloader:
    # Route batch to experts (common in MOE models)
    routed_data = [route_to_expert(batch, i) for i in range(num_experts)]

    # Update expert inputs in-place
    for i in range(num_experts):
        expert_inputs[i][:routed_data[i].shape[0]].copy_(routed_data[i])  # ✅ Update device tensors

    g.replay()  # This still fails! Why?
```

**Why it fails**:

In normal (eager) execution, `A_ptrs_host` and `B_ptrs_host` are **dynamic tensors**—created fresh on every call and immediately recycled. When CUDA graph captures the non-blocking H2D copy, it records the **source host memory addresses**, making these host tensors **CUDA graph inputs**. All graph inputs must be static (persistent) with stable addresses, but these are dynamic (recreated each call).

1. **During capture**: `grouped_gemm` creates temporary host tensors `A_ptrs_host` at host address `0x5000` containing device pointers `[0x7f3a2000, 0x7f3a4000, ...]`. The graph captures: "copy from host address `0x5000` to device".

2. **After capture**: When the function returns, `A_ptrs_host` is deallocated and host address `0x5000` is recycled by Python's memory allocator.

3. **During replay**: The captured graph executes: "copy from host address `0x5000` to device". But `0x5000` now contains unrelated data, garbage, or is unmapped → segmentation fault or incorrect pointer values.

**Solution - Keep host pointer arrays alive**:

```python
# Preallocate expert input tensors before capture
max_expert_size = max(expert_sizes)
expert_inputs = [
    torch.zeros(max_expert_size, hidden_dim, device='cuda')
    for _ in range(num_experts)
]
expert_weights = [torch.randn(hidden_dim, hidden_dim, device='cuda') for _ in range(num_experts)]

# Preallocate host pointer arrays (keep alive for graph lifecycle!)
# These must persist across all graph replays
A_ptrs_host_list = []  # Store host tensors for each grouped_gemm call
B_ptrs_host_list = []

def grouped_gemm_static(A_list, B_list, call_index):
    """Modified grouped_gemm that reuses preallocated host pointer arrays"""
    # Extract current device pointers
    A_ptrs = [A.data_ptr() for A in A_list]
    B_ptrs = [B.data_ptr() for B in B_list]

    # Update preallocated host pointer arrays (same host addresses!)
    A_ptrs_host_list[call_index].copy_(torch.tensor(A_ptrs, dtype=torch.int64))
    B_ptrs_host_list[call_index].copy_(torch.tensor(B_ptrs, dtype=torch.int64))

    # Copy to device from persistent host addresses (non-blocking)
    A_ptrs_device = A_ptrs_host_list[call_index].to('cuda', non_blocking=True)
    B_ptrs_device = B_ptrs_host_list[call_index].to('cuda', non_blocking=True)

    outputs = _cuda_grouped_gemm_kernel(A_ptrs_device, B_ptrs_device, ...)
    return outputs

# Preallocate host pointer arrays before capture (assume 2 grouped_gemm calls)
num_gemm_calls = 2
for _ in range(num_gemm_calls):
    A_ptrs_host_list.append(torch.zeros(num_experts, dtype=torch.int64, device='cpu'))
    B_ptrs_host_list.append(torch.zeros(num_experts, dtype=torch.int64, device='cpu'))

# Capture graph
g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
    outputs1 = grouped_gemm_static(expert_inputs, expert_weights, call_index=0)
    outputs2 = grouped_gemm_static(expert_inputs, expert_weights, call_index=1)

# During training - update both expert inputs and host pointer arrays
for batch in dataloader:
    routed_data = [route_to_expert(batch, i) for i in range(num_experts)]

    # Copy routed data into preallocated tensors (maintains device pointer stability)
    for i in range(num_experts):
        expert_inputs[i][:routed_data[i].shape[0]].copy_(routed_data[i])  # ✅ Same addresses

    # Host pointer arrays are updated via .copy_() inside grouped_gemm_static
    g.replay()  # Graph reads from persistent host addresses
```

**Key lesson**: When operations store tensor addresses in pointer arrays and copy them via H2D transfers, you must ensure:

1. **Device tensors** (the actual data) maintain stable addresses across replays (use `.copy_()` for updates)
2. **Host pointer arrays** remain allocated for the entire graph lifecycle (not just during capture)

For multiple grouped GEMM calls, preallocate one host pointer array per call. The non-blocking nature of H2D copies means the host memory must stay valid throughout all graph replays.

:::{tip} Performance Optimization

The H2D copy in the captured graph is redundant during replay—it copies the same pointer values from the same host address every time. You can optimize by moving the H2D copy **outside the capture region**: perform it once before graph capturing instead of replaying it every iteration. This way, you only need to ensure `A_ptrs_device` (the device pointer array) is static/persistent for the graph lifecycle, not the host arrays. This eliminates the redundant H2D copy on every replay, improving performance.
:::

## Dynamic Scalars

Dynamic scalars refer to CPU-side scalar values that change across iterations and are used in GPU operations. Unlike dynamic tensors (which concern memory addresses), dynamic scalars are about changing values that originate on the CPU. Common examples include random number generator seeds, learning rates, global step counters, and gradient scaling factors in mixed precision training.

```python
# ❌ Breaks graph - exponent changes but graph uses captured constant
exponent = 2.0

# Capture graph with exponent=2.0
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    result = torch.pow(x, exponent)  # Captures exponent=2.0 as constant

# Try to use different exponents during replay
for iteration in range(num_iters):
    exponent = compute_exponent(iteration)  # Returns 2.5, then 3.0, then 3.5...
    g.replay()  # Always computes x^2.0, ignoring current exponent value!
```

**Why this is problematic**:

When you pass a Python scalar (like `exponent=2.0`) to a GPU operation, the CUDA kernel is launched with that scalar value as a kernel parameter. During graph capture, CUDA records this kernel launch with the specific scalar value—essentially turning the scalar into a constant in the captured graph. On replay, the graph launches the same kernel with the same constant value (2.0), completely ignoring any changes you made to the Python variable `exponent` (now 2.5, 3.0, etc.). **The scalar is "baked into" the captured graph as a constant.**

### General Solutions

**Identify all changing scalars**: Before attempting to graph your code, audit all scalar values passed to GPU operations within the graphed region. Look for learning rates, step counters, scale factors, exponents, and any other CPU scalars that change across iterations. These must be converted to GPU tensors if they are captured into CUDA graph.

To make dynamic scalars work with CUDA graphs, follow these steps:

**Step 1: Store scalars as GPU tensors and update in-place**:

Convert all identified CPU scalars to GPU tensors before capture, then update them in-place between replays:

```python
# Preallocate GPU tensor for scalar
exponent_tensor = torch.tensor(2.0, device='cuda')

# Capture graph using GPU tensor
g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
    result = torch.pow(x, exponent_tensor)  # ✅ Uses GPU tensor instead of Python scalar

# Update scalar tensor before each replay
for iteration in range(num_iters):
    new_exponent = compute_exponent(iteration)  # Returns 2.5, then 3.0, then 3.5...
    exponent_tensor.fill_(new_exponent)
    g.replay()  # Uses updated exponent value!
```

**Step 2: Modify custom CUDA kernels to accept device pointers** (only for custom kernels):

For **custom CUDA kernels** (not PyTorch native operators), you must also modify the kernel implementation to accept device pointers instead of scalar values as kernel parameters. For example, if you wrote a custom `pow` kernel:

```cuda
// ❌ Original custom pow kernel - exponent captured as constant
__global__ void custom_pow_kernel(float* output, float* input, float exponent, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = powf(input[idx], exponent);  // exponent captured as constant
    }
}

// Launch: custom_pow_kernel<<<...>>>(output, input, exponent, n);  // exponent captured as constant


// ✅ Modified custom pow kernel - exponent read from device memory
__global__ void custom_pow_kernel(float* output, float* input, float* exponent_ptr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float exponent = *exponent_ptr;  // Dereference pointer to read current value
        output[idx] = powf(input[idx], exponent);
    }
}

// Launch: custom_pow_kernel<<<...>>>(output, input, exponent_tensor.data_ptr(), n);
```

**When this is needed**: Only for custom CUDA kernels (via PyTorch C++ extensions or third-party libraries). PyTorch native operators like `torch.pow()` don't require kernel modification—they already handle GPU tensors correctly.

The following examples demonstrate these steps in real-world scenarios.

### Example 1: Random Number Generator State

Random operations like dropout require different random values each iteration, but naive graph capture would replay the same random sequence.

**Background - How CUDA RNG works**:

CUDA random number generation uses the **Philox4_32_10** algorithm, which requires three parameters to initialize the random state in each CUDA thread:

1. **Seed** (uint64): Determines which random sequence to use—same seed produces the same sequence
2. **Subsequence ID** (typically thread index): Each CUDA thread uses its own subsequence to generate independent random numbers in parallel
3. **Offset** (uint64): Position within the random sequence—the offset advances by the number of random values consumed (e.g., dropout over 1024 elements advances offset by 1024)

Each random operation in PyTorch follows this pattern:

1. Host prepares `PhiloxCudaState` containing current seed and offset from the global RNG state
2. CUDA kernel receives this state as a parameter
3. Each thread initializes its own `curandStatePhilox4_32_10_t` with: `curand_init(seed, thread_idx, offset, &state)`
4. Threads generate random numbers from their independent subsequences
5. Host increments the global offset for the next operation

**The CUDA graph problem**: During graph capture, if the seed and offset values are captured as scalar constants in the kernel launch parameters, every replay would use the exact same random sequence—your dropout mask would be identical across all iterations of graph replays!

**PyTorch's solution**: Register the RNG generator state with the graph using `register_generator_state()`. This changes how `PhiloxCudaState` is constructed during capture:

- **Without registration**: `PhiloxCudaState` captures the seed and offset as constant scalar values—replays always use the same values
- **With registration**: `PhiloxCudaState` captures **pointers** to GPU tensors holding the seed and offset—replays read the current values from GPU memory

```python
# Example with custom generator (default generator is auto-registered)
custom_gen = torch.Generator(device='cuda')
g = torch.cuda.CUDAGraph()

# Register custom generator if using one (default generator auto-registers)
g.register_generator_state(custom_gen)

# Capture graph with dropout using custom generator
with torch.cuda.graph(g):
    output = F.dropout(input, p=0.5, training=True, generator=custom_gen)

# Each replay uses different random values
for _ in range(10):
    g.replay()  # PyTorch automatically advances RNG offset after each replay
```

**How it works internally** (see [PyTorch v2.4.0 source code](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L194-L225)):

1. **During capture** ([`capture_prologue`](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L194-L199)): Instead of passing `PhiloxCudaState(seed_value, offset_value)` to kernels, PyTorch passes `PhiloxCudaState(seed_ptr, offset_ptr, intra_graph_offset)` where `seed_ptr` and `offset_ptr` are device pointers to 1-element GPU tensors. At capture end ([`capture_epilogue`](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L205-L208)), PyTorch records the total offset consumed by the graph (`wholegraph_increment`)
2. **Before each replay** ([`replay_prologue`](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L214-L225)): PyTorch increments the CPU-side offset counter by `wholegraph_increment`, then copies the updated seed and offset values to GPU tensors using `.fill_()`: `seed_extragraph_.fill_(seed)` and `offset_extragraph_.fill_(offset)`
3. **During replay** ([kernel execution](https://github.com/pytorch/pytorch/blob/v2.4.0/aten/src/ATen/cuda/detail/UnpackRaw.cuh#L16-L26)): Kernels dereference the device pointers to read the current seed and offset from GPU memory: `seed = *seed_ptr; offset = *offset_ptr + intra_graph_offset`

This follows the same pattern as our "General Solutions"—store changing scalars in GPU tensors, update them on the CPU side, copy to GPU before each replay, and pass device pointers to kernels.

**Key lesson**: For RNG-based operations, use PyTorch's built-in `register_generator_state()` API. This is handled automatically by `torch.cuda.graph()` and `make_graphed_callables()`—you rarely need to call it manually.

### Example 2: Learning Rate and Global Step in APEX FusedAdam

Optimizers like Adam use learning rate and global step (for bias correction) that change each iteration. APEX's FusedAdam provides a CUDA graph-compatible implementation by moving these scalars to GPU.

**Problem with standard Adam**:

```python
# Standard Adam - not CUDA graph compatible
# Capture graph at iteration 0 with lr=0.001
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    # Adam internally uses:
    # - lr (learning rate) as Python float (0.001)
    # - step (global step counter) as Python int (0)
    optimizer.step()  # ❌ Captures lr=0.001 and step=0 as constants!

# Try to use different learning rates during replay
for iteration in range(num_iters):
    lr = scheduler.get_lr()  # Returns 0.0005, 0.0001, etc.
    g.replay()  # Always uses lr=0.001 and step=0 from capture!
```

**APEX FusedAdam's solution** (see [APEX FusedAdam implementation](https://github.com/NVIDIA/apex/blob/2386a912164b0c5cfcd8be7a2b890fbac5607c82/apex/optimizers/fused_adam.py)):

1.  Store `lr` and `step` as GPU tensors instead of CPU scalars:

    - `lr` initialized as CPU tensor, then moved to device if `capturable=True` ([lines 78, 105](https://github.com/NVIDIA/apex/blob/2386a912164b0c5cfcd8be7a2b890fbac5607c82/apex/optimizers/fused_adam.py#L78))
    - `step` as GPU tensor when `capturable=True` ([line 151-154](https://github.com/NVIDIA/apex/blob/2386a912164b0c5cfcd8be7a2b890fbac5607c82/apex/optimizers/fused_adam.py#L151))

2.  Use capturable CUDA kernel variant that reads from device pointers ([lines 219-232](https://github.com/NVIDIA/apex/blob/2386a912164b0c5cfcd8be7a2b890fbac5607c82/apex/optimizers/fused_adam.py#L219-L232), calls `multi_tensor_adam_capturable` kernel)

```python
from apex.optimizers import FusedAdam

# APEX FusedAdam stores lr and step on GPU
optimizer = FusedAdam(model.parameters(), lr=0.001, set_grad_none=True)

# Capture graph - FusedAdam kernel reads from device pointers
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    optimizer.step()

# Access GPU tensors for lr and step (after capture)
lr_tensor = optimizer.param_groups[0]['lr']  # GPU tensor, not Python float
step_tensor = optimizer.state[p]['step']  # GPU tensor, not Python int

# Update lr before each replay
for iteration in range(num_iters):
    lr_tensor.fill_(scheduler.get_lr()[0])  # Update GPU tensor in-place
    # step is automatically incremented by FusedAdam's CUDA kernel

    g.replay()  # Uses current lr value from GPU memory
```

**How it works**: APEX modified the Adam CUDA kernel signature to accept GPU tensors instead of scalars (see [multi_tensor_adam.cu](https://github.com/NVIDIA/apex/blob/2386a912164b0c5cfcd8be7a2b890fbac5607c82/csrc/multi_tensor_adam.cu#L251-L254)):

```cuda
// Standard Adam - takes scalars as kernel parameters (captured as constants)
adam_kernel<<<...>>>(param_ptr, grad_ptr, lr_value, step_value, ...)

// FusedAdam capturable - takes GPU tensors (reads dynamic values from device memory)
fused_adam_capturable<<<...>>>(param_ptr, grad_ptr, lr_tensor, step_tensor, ...)
```

The kernel dereferences `lr_ptr` and `step_ptr` to read the current values from GPU memory on each replay, allowing them to change between iterations.

**Key lesson**: To make optimizers CUDA graph-compatible with dynamic scalars:

1. Store scalars as GPU tensors (not CPU Python values)
2. Modify CUDA kernels to accept device pointers and dereference them, rather than embedding scalar values as kernel parameters
3. Update GPU tensors in-place before each replay

This pattern applies to any custom CUDA kernel that needs dynamic scalar parameters.

## Dynamic Shapes

Dynamic shapes occur when tensor dimensions vary across iterations. Common examples include variable batch sizes, variable sequence lengths in NLP, and different image resolutions. This is the most challenging dynamic pattern because changing shapes means different tensor sizes, different memory allocations, and often different kernels, all of which break CUDA graph capture.

**Why this is problematic**: CUDA graphs capture not just the sequence of kernel launches, but also the specific kernel configurations (grid/block dimensions) and memory allocation patterns. When shapes change:

1. **Different kernel configurations**: A matmul with shape `[64, 512] @ [512, 1024]` launches different grid/block sizes than `[128, 512] @ [512, 1024]`
2. **Different memory allocations**: Changing shapes trigger dynamic memory allocation during replay, which is forbidden in CUDA graphs
3. **Different execution paths**: Some kernels (e.g., cuDNN convolutions) select different algorithms based on input shapes

During graph replay, the captured kernel configurations and memory addresses are reused regardless of actual input shapes, leading to incorrect results or crashes.

### General Solutions

There are two main approaches to handle dynamic shapes:

1. **Padding to fixed size**: Pad all inputs to a fixed size, capture a single graph. Can pad during data loading or at runtime
2. **Bucketing**: Capture multiple graphs for common shape configurations (e.g., sequence lengths 128, 256, 512), select the appropriate graph at runtime. Usually works together with padding—pad inputs to the nearest bucket size

Start with padding (simplest), then move to bucketing only if profiling shows significant wasted computation.

**When padding and bucketing are inefficient**: If shapes vary unpredictably (e.g., token routing with conventional MoE kernels, variable image sizes, irregular scientific data), padding may waste substantial work and bucketing may require too many graphs. For implementations that require host-side sizing:

- Graph only the static parts (see Example 2 below), or
- Keep the model in eager mode entirely

MoE kernels that consume device-side shape metadata can also support full-iteration graphing; see the alternative in [Example 2](#example-2-moe-models-with-token-dropless).

### Example 1: Variable Sequence Lengths (NLP)

**Problem**: NLP models often have variable sequence lengths (and sometimes variable batch sizes):

```python
# Sequence lengths vary: 128, 256, 512, 1024, ...
# Batch sizes may also vary: 16, 32, 64, ...
for input_ids in dataloader:  # Shape: [batch, seq_len]
    output = model(input_ids)
```

**Solution 1 - Padding to fixed size**:

Pad all inputs to a fixed size and capture a single graph. This trades computational efficiency for implementation simplicity.

```python
max_seq_len = 2048
batch_size = 32
static_input = torch.zeros(batch_size, max_seq_len, dtype=torch.long, device='cuda')
static_output = None

# Capture with max size
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# Pad variable-length sequences at runtime
def forward(input_ids):
    actual_seq_len = input_ids.shape[1]

    # Pad to max length
    if actual_seq_len < max_seq_len:
        input_ids = F.pad(input_ids, (0, max_seq_len - actual_seq_len), value=pad_token_id)

    # Copy to static buffer and replay
    static_input.copy_(input_ids)
    g.replay()

    # Extract actual results (trim padding)
    return static_output[:, :actual_seq_len].clone()
```

**How it works**: The model always processes `max_seq_len` tokens, even if the actual input is shorter. For transformers with attention masking, padding tokens are masked out so they don't affect results—but the model still computes attention scores for them. A 128-token sequence padded to 2048 wastes ~93% of computation in attention layers.

**Alternative**: You can also pad during data loading (e.g., in DataLoader's `collate_fn`) to eliminate runtime padding overhead, at the cost of less flexibility in handling variable lengths.

**When this is acceptable**: If sequence lengths cluster near the maximum (e.g., most sequences are 1800-2048 tokens out of 2048 max), wasted computation is minimal. This approach makes sense when simplicity and memory constraints (only one graph) outweigh efficiency concerns.

- **Pros**: Single graph, simple implementation, handles any size ≤ max, minimal memory overhead
- **Cons**: Wastes computation on padding (especially for short sequences), can significantly reduce throughput

**Solution 2 - Bucketing**:

Capture multiple graphs, one for each common shape configuration. At runtime, select the graph that best matches the input shape (usually rounding up to the nearest bucket). This provides the best efficiency for diverse shape distributions at the cost of increased complexity and memory usage.

```python
class MultiGraphModel:
    def __init__(self, model, seq_lengths=[128, 256, 512, 1024, 2048], batch_size=32):
        self.model = model
        self.graphs = {}

        # Capture one graph per sequence length bucket
        for seq_len in seq_lengths:
            static_input = torch.zeros(batch_size, seq_len, dtype=torch.long, device='cuda')
            static_output = None

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                _ = model(static_input)

            # Capture graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_output = model(static_input)

            self.graphs[seq_len] = {
                'graph': g,
                'input': static_input,
                'output': static_output
            }

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]

        # Round up to nearest bucket
        bucket = min([s for s in self.graphs.keys() if s >= seq_len],
                     default=max(self.graphs.keys()))

        # Pad to bucket size if needed
        if seq_len < bucket:
            input_ids = F.pad(input_ids, (0, bucket - seq_len), value=pad_token_id)

        # Copy to static buffer and replay
        self.graphs[bucket]['input'].copy_(input_ids)
        self.graphs[bucket]['graph'].replay()
        return self.graphs[bucket]['output'][:, :seq_len].clone()  # Trim padding
```

**How it works**: Each bucket graph is optimized for its specific shape—a 512-token sequence runs different kernel configurations than a 1024-token sequence. By capturing multiple buckets, you avoid the wasted computation of always padding to maximum length. The runtime overhead is minimal: selecting a bucket (dictionary lookup), padding if needed (cheap for small gaps like 480→512), and trimming output (simple slicing).

**Memory considerations**: Each graph stores the entire computation graph including all intermediate activations at that specific shape. For a transformer model with 5 buckets, expect 5× the memory overhead compared to a single graph. This is acceptable if you have enough GPU memory to accommodate multiple graph instances—the performance gain justifies the cost when memory permits.

- **Pros**: Optimal performance for each bucket, handles diverse lengths efficiently, minimal wasted computation
- **Cons**: Memory overhead (one graph per bucket), complex management, need to choose buckets wisely

**When to use each**:

Start with padding for simplicity. Move to bucketing only if profiling shows significant wasted computation (e.g., many short sequences padded to a very long max) and you have sufficient GPU memory for multiple graphs.

**Key decision factors**:

- **Shape distribution**: If 90%+ of sequences cluster in a narrow range (e.g., 1800-2048 tokens), padding to max is simplest and wastes minimal computation. If shapes spread widely (e.g., distributed across 128, 256, 512, 1024), bucketing avoids wasted work
- **Memory constraints**: Bucketing requires more memory for multiple graph instances. If memory-limited, padding is the only option
- **Hybrid approach**: Graph only the most frequent shapes (e.g., top 3 buckets covering 95% of data) and fall back to eager mode for rare shapes. This balances performance, memory, and complexity

### Example 2: MoE Models with Token Dropless

**Problem**: Mixture-of-Experts (MoE) models with token dropless mechanisms route different numbers of tokens to each expert based on the gating network's output. This creates highly dynamic shapes that change every forward pass:

```python
class MoELayer(nn.Module):
    def __init__(self, num_experts=8, expert_capacity=None):
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.expert_capacity = expert_capacity  # tokens per expert

    def forward(self, tokens):  # Shape: [batch * seq_len, hidden_dim]
        # Gate decides which tokens go to which experts
        routing_weights = self.gate(tokens)  # [num_tokens, num_experts]
        expert_assignments = routing_weights.argmax(dim=-1)  # Dynamic routing

        # Group tokens by expert - HIGHLY DYNAMIC SHAPES!
        expert_inputs = []
        for expert_id in range(len(self.experts)):
            # Number of tokens varies per expert, per iteration
            mask = (expert_assignments == expert_id)
            expert_tokens = tokens[mask]  # Shape: [dynamic_count, hidden_dim]
            expert_inputs.append(expert_tokens)

        # Process each expert - each has different input shape
        expert_outputs = []
        for expert, inputs in zip(self.experts, expert_inputs):
            if len(inputs) > 0:
                outputs = expert(inputs)  # Shape varies!
                expert_outputs.append(outputs)

        # Combine outputs (also dynamic)
        return combine_expert_outputs(expert_outputs, expert_assignments)
```

**Why this is extremely dynamic**:

- Token distribution changes every forward pass based on input data
- Each expert processes a different number of tokens (could be 0 to all tokens)
- No predictable pattern—can't use bucketing effectively
- Even with capacity limits, shapes vary significantly

**Partial graphing approach - Graph only static parts**:

The implementation above creates variable-sized expert inputs and uses Python control flow to select expert computation. Keep that expert region in eager mode and graph the surrounding static layers:

```python
class MoEModel(nn.Module):
    def __init__(self):
        self.pre_layers = nn.Sequential(...)   # Static computation
        self.moe_layer = MoELayer(...)          # Dynamic computation
        self.post_layers = nn.Sequential(...)  # Static computation

    def forward(self, x):
        x = self.pre_layers(x)   # Can be graphed
        x = self.moe_layer(x)    # This implementation runs eagerly
        x = self.post_layers(x)  # Can be graphed
        return x

# Graph only the static parts
pre_layers_graphed = torch.cuda.make_graphed_callables(
    model.pre_layers,
    sample_args=(torch.zeros(batch_size, seq_len, hidden_dim, device='cuda'),)
)
post_layers_graphed = torch.cuda.make_graphed_callables(
    model.post_layers,
    sample_args=(torch.zeros(batch_size, seq_len, hidden_dim, device='cuda'),)
)

# Training loop with partial graphing
for batch in dataloader:
    # Pre-processing: graphed (fixed shape)
    pre_output = pre_layers_graphed(batch)

    # MoE routing: eager mode (dynamic shapes)
    moe_output = model.moe_layer(pre_output)

    # Post-processing: graphed (fixed shape)
    final_output = post_layers_graphed(moe_output)

    loss = criterion(final_output, targets)
    loss.backward()
    optimizer.step()
```

**Performance considerations**:

- Even with partial graphing, you can still achieve significant speedup if the non-MoE layers dominate compute time
- For MoE-heavy models (many MoE layers), partial graphing may provide limited benefit because expert computation remains eager

**Alternative - Full-iteration graphing for dropless MoE**:

With sync-free dispatch and device-initiated grouped GEMM, dynamic token counts remain on the GPU while computation uses preallocated buffers. Paged stashing reduces the activation memory retained for backward, and overflow recovery preserves dropless training when routing exceeds the configured capacity. See [DeepSeek-V3 671B: Paged Stashing for Dropless MoE](../examples/deepseek-v3-paged-stash.md) for the full-iteration approach.

**Other alternatives that change routing behavior**:

- **Token dropping**: Set a fixed expert capacity (e.g., max 128 tokens per expert). If routing assigns more tokens, drop the excess. This makes shapes static and fully graphable, but may degrade model quality if too many tokens are dropped
- **Static expert assignment**: Assign tokens to experts in a round-robin or predetermined pattern instead of learned routing. This makes shapes predictable and graphable, but sacrifices the adaptive routing that makes MoE effective

**Key lesson**: Use partial graphing when the expert implementation requires variable-sized allocations or host-side decisions, as in the example above. Full-iteration graphing supports dynamic routing when compatible dispatch and expert kernels consume device-side shape metadata with preallocated buffers.

## What's Next?

- **[Quick Checklist](quick-checklist.md)**: Complete checklist to verify your code is ready for capture
- **[Examples](../examples/introduction.md)**: Real-world implementations in RNN-T, Stable Diffusion, Llama 3.1 405B, and DeepSeek-V3 671B
- **[Troubleshooting](../troubleshooting/introduction.md)**: Comprehensive guide for debugging capture failures, silent failures, memory errors, and performance issues
- **[Constraints](../cuda-graph-basics/constraints.md)**: Full list of CUDA Graph restrictions
- **[PyTorch-Specific Constraints](torch-integration.md#pytorch-specific-constraints)**: PyTorch requirements
