<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Transformer Engine and Megatron-LM CUDA Graph Support

:::{note}

This section covers CUDA Graph implementations from Transformer Engine and Megatron-LM, including manual graphing with Transformer Engine's `make_graphed_callables`, automatic per-layer graphing (`CudaGraphManager`), and full-iteration graphing (`FullCudaGraphWrapper`).
:::

## Overview

[Transformer Engine](https://github.com/NVIDIA/TransformerEngine) and [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) provide complementary CUDA Graph implementations optimized for distributed training workloads:

**Transformer Engine** provides:

1. **[`make_graphed_callables`](https://github.com/NVIDIA/TransformerEngine/blob/v2.2/transformer_engine/pytorch/graph.py#L105)**: Manual per-callable CUDA graph wrapper with FP8 support and advanced pipeline parallelism schedule support (builds on PyTorch's API, works with any PyTorch model)

**Megatron-LM** provides two automatic CUDA Graph implementations:

2. **[CudaGraphManager](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/cuda_graphs.py#L953)**: Automatic per-layer/per-module CUDA graph management
3. **[FullCudaGraphWrapper](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/full_cuda_graph.py#L91)**: Full training iteration CUDA graph capture

Both Megatron implementations are designed to work with Megatron's unique requirements: pipeline parallelism, tensor parallelism, distributed data parallelism, and gradient accumulation.

:::{note} Version Reference

All code links in this document refer to **Transformer Engine v2.2** and **Megatron-LM core_v0.14.0**. If you're using different versions, the APIs and implementation details may vary. Please refer to the [latest Transformer Engine documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html) and [Megatron-LM documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) and adapt accordingly.
:::

While PyTorch provides native CUDA graph APIs (`torch.cuda.CUDAGraph`, `torch.cuda.make_graphed_callables`), applying them to large-scale distributed training introduces challenges that cannot be easily resolved with these basic primitives alone. The CUDA graph implementations from Transformer Engine and Megatron-LM arose specifically to address two key challenges: **FP8 training integration** (managing dynamic scaling state across graph captures) and **pipeline parallelism support** (handling complex interleaved microbatch execution patterns). Let's examine these challenges and how each implementation addresses them.

## Challenges with FP8 and Pipeline Parallelism

As mentioned earlier, the specialized CUDA graph implementations from Transformer Engine and Megatron-LM address two key challenges that arise when applying **partial graphing** to large-scale distributed training. Partial graphing—capturing only specific parts of the computation (e.g., individual layers or callables) while leaving others in eager mode—provides flexibility but introduces complexity when combined with FP8 training and pipeline parallelism. Understanding these challenges helps motivate the design choices in each implementation.

### The FP8 Challenge

FP8 (8-bit floating point) training accelerates computation and reduces memory usage, but introduces complexity for CUDA graphs due to its **global buffer management**, **dynamic scaling mechanism**, and **weight quantization caching**. FP8 maintains global scaling state (amax history, scale factors) that must be updated across all layers and synchronized across GPUs after each training iteration.

**Challenge 1: Global FP8 buffers**

Transformer Engine maintains global buffers to store FP8 metadata (amax history, scale, scale_inv) and transposed FP8 weights across all layers. These buffers are **CUDA graph inputs**—they are accessed by operations inside the captured graph. The challenge is that these buffers were originally designed to be dynamically constructed and deleted during training (built via `append()` during forward passes, deleted at autocast exit), causing memory addresses to change across iterations. Since CUDA graph inputs must have static, persistent memory addresses, this dynamic allocation pattern is incompatible with CUDA graphs.

**Challenge 2: Dynamic scaling state**

In standard eager execution, Transformer Engine calls `reduce_and_update_fp8_tensors()` at the end of each `fp8_autocast` context to:

1. Perform an all-reduce of amax (maximum absolute value) across all GPUs
2. Update scaling factors for the next iteration based on the collected amax values

When using **partial graphing** (capturing individual layers or callables), if this all-reduce operation is captured inside each graph, it would execute during every graph replay, causing:

- **Incorrect scaling**: Each layer's graph would use outdated scaling factors instead of globally synchronized ones
- **Numerical errors**: FP8 quantization with wrong scaling factors degrades model accuracy
- **Redundant communication**: The all-reduce would execute per-layer instead of once per iteration

**Challenge 3: Weight quantization and transpose caching**

FP8 training requires quantizing BF16/FP32 weights to FP8 format and computing weight transposes for backward passes. These operations are expensive and ideally should be cached across microbatches. However, with partial graphing:

- **Weight quantization is captured per-graph**: Each graph replay re-quantizes weights, causing redundant computation
- **Weight transposes are computed per-graph**: Transpose operations are unnecessarily repeated across microbatches
- **Cache invalidation needed**: When weights are updated (e.g., during optimizer step), cached FP8 weights and transposes must be invalidated

The challenge is to cache FP8 weight quantizations and transposes across microbatches within a graph while ensuring they are properly updated when weights change.

### The Pipeline Parallelism Challenge

Pipeline parallelism (PP) introduces a different challenge for CUDA graphs: **interleaved microbatch execution**. With PP enabled, multiple microbatches execute in an interleaved schedule to keep all pipeline stages busy, maximizing hardware utilization.

The challenge arises when graphs share memory pools—a common practice for memory efficiency. Graphs that share a memory pool must be replayed in the exact same order they were captured; otherwise, one graph may overwrite memory that another graph still needs, causing **wrong gradients** or **corrupted results**.

This becomes particularly problematic with pipeline parallelism because:

1. **Execution order is complex**: Layer N processes microbatches in a non-sequential pattern: MB1_fwd → MB2_fwd → MB3_fwd → ... → MB3_bwd → MB2_bwd → MB1_bwd
2. **Full-iteration graphing requires more effort**: Capturing all microbatches across all pipeline stages demands careful handling of static buffers and memory management
3. **Memory pool sharing demands order preservation**: Any deviation from capture order risks memory corruption

For a detailed explanation of the memory pool corruption issue, see [Troubleshooting: Replay Order Mismatch](../troubleshooting/numerical-errors.md#replay-order-mismatch).

The challenge is to ensure graphs are captured and replayed in the correct order, either through manual scheduling or automatic order tracking.

---

Now that we understand these challenges, let's examine how each implementation addresses them, starting with the most flexible option.

## Transformer Engine `make_graphed_callables`

### What It Is

Transformer Engine's [`make_graphed_callables`](https://github.com/NVIDIA/TransformerEngine/blob/v2.2/transformer_engine/pytorch/graph.py#L105) is a wrapper around PyTorch's `torch.cuda.make_graphed_callables` that adds FP8-specific handling. It allows you to manually capture specific callables (functions, modules, or methods) as CUDA graphs.

**Scope**: This is a **manual** API—you choose what to graph and when to capture. Unlike Megatron's automatic approaches that are tightly integrated with Megatron layers, `make_graphed_callables` works with any PyTorch module or callable, making it the most flexible option for custom models.

### How to Use It

The following is a rough example showing how to use `make_graphed_callables` with virtual pipeline parallelism (VPP).

```python
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.graph import make_graphed_callables
from transformer_engine.pytorch.fp8 import fp8_autocast

# Model configuration
num_layers_per_chunk = 4  # Layers per model chunk
num_model_chunks = 2      # Virtual pipeline parallel size (VPP)
num_microbatches = 3

# Define model layers grouped into chunks (for VPP)
# Total layers = num_layers_per_chunk * num_model_chunks
layers = [MyTransformerLayer().cuda()
          for _ in range(num_layers_per_chunk * num_model_chunks)]

# Create sample inputs: one per layer per microbatch per chunk
# Total = num_layers_per_chunk * num_model_chunks * num_microbatches
sample_args = tuple(
    (torch.randn(batch_size, seq_len, hidden_size, device='cuda'),)
    for _ in range(num_layers_per_chunk * num_model_chunks * num_microbatches)
)

# Define pipeline schedule for interleaved pipeline parallelism
# Format: 1-indexed chunk IDs, positive=forward, negative=backward
# _order length = num_model_chunks * num_microbatches * 2
# Example: 2 chunks, 3 microbatches → interleaved schedule
layer_order = [1, 2, 1, 2, 1, 2, -2, -1, -2, -1, -2, -1]
# Meaning: fwd chunk1 MB0, fwd chunk2 MB0, fwd chunk1 MB1, fwd chunk2 MB1,
#          fwd chunk1 MB2, fwd chunk2 MB2, bwd chunk2 MB0, bwd chunk1 MB0, ...

# Wrap layers in CUDA graphs
graphed_layers = make_graphed_callables(
    tuple(layers),
    sample_args=sample_args,
    fp8_enabled=True,  # Enable FP8-aware graphing
    fp8_recipe=fp8_recipe,  # FP8 configuration
    fp8_weight_caching=True,  # Cache FP8 weight quantization across microbatches
    _order=layer_order,  # Pipeline schedule (None for no PP)
)

# Training loop - replay order must match capture order (_order)
# Note: This shows the full PP communication pattern. In practice, Megatron-LM
# handles this via pipeline schedules in megatron/core/pipeline_parallel/schedules.py
for batch_idx, (inputs, targets) in enumerate(dataloader):
    # inputs/targets: list of microbatch tensors (only used on first/last PP stage)
    optimizer.zero_grad()

    # fp8_autocast is required during replay for FP8 state management.
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        chunk_outputs = {}  # (chunk_idx, mb_idx) -> output tensor

        for step_idx, chunk_id in enumerate(layer_order):
            if chunk_id > 0:  # Forward pass
                mb_idx = ...  # Derive based on schedule
                chunk_idx = chunk_id - 1

                # Get input: from dataloader (first stage) or recv from prev stage
                if first_pp_stage and chunk_idx == 0:
                    x = inputs[mb_idx]
                else:
                    torch.distributed.recv(inputs[mb_idx], src=prev_rank)
                    x = inputs[mb_idx]

                # Run layers in this chunk
                start_layer = chunk_idx * num_layers_per_chunk
                end_layer = start_layer + num_layers_per_chunk
                for layer in graphed_layers[start_layer:end_layer]:
                x = layer(x, is_first_microbatch=(mb_idx == 0))

                chunk_outputs[(chunk_idx, mb_idx)] = x

                # Compute loss (last stage) or send to next stage
                if last_pp_stage and chunk_idx == num_model_chunks - 1:
                    loss = criterion(x, targets[mb_idx])
                else:
                    torch.distributed.send(x, dst=next_rank)

            else:  # Backward pass
                mb_idx = ...  # Derive based on schedule
                chunk_idx = -chunk_id - 1

                # Get grad: from loss.backward (last stage) or recv from next stage
                if last_pp_stage and chunk_idx == num_model_chunks - 1:
                    loss.backward()
                else:
                    torch.distributed.recv(d_out, src=next_rank)
                    torch.autograd.backward(
                        chunk_outputs[(chunk_idx, mb_idx)],
                        grad_tensors=d_out
                    )

                # Send input grad to previous stage (if not first stage/chunk)
                if not (first_pp_stage and chunk_idx == 0):
                    d_in = inputs[mb_idx].grad
                    torch.distributed.send(d_in, dst=prev_rank)

    # FP8 forward scaling is auto-updated on fp8_autocast exit
    optimizer.step()
```

### How It Works

`make_graphed_callables` extends PyTorch's CUDA graph capture with FP8-specific handling to address the [FP8 challenges](#the-fp8-challenge):

**Capture timing (AOT - Ahead-of-Time)**:

- Graphs are captured **before the training loop starts** when you call `make_graphed_callables()`
- Performs `num_warmup_iters` (default 3) warmup passes during capture to stabilize memory allocations
- Returns wrapped callables that replay the captured graphs during training
- When `_order` is provided, graphs are captured in the exact order specified by the pipeline schedule

**Replay requirements**:

- **`fp8_autocast` context is required during replay**: The graphed callables check `FP8GlobalStateManager.is_fp8_enabled()` at replay time to manage FP8 metadata (fp8_group, recipe). Without `fp8_autocast`, FP8 state won't be properly configured.
- **Replay order must match capture order**: When using `_order`, the training loop must execute graphs in the same interleaved order as specified during capture. This is critical because all graphs share a memory pool, and out-of-order replay can corrupt intermediate tensors.

**Handling dynamic scaling state**:

- Uses `fp8_autocast(..., _graph=True)` internally during capture to skip per-callable amax reduction (the `_graph=True` flag defers FP8 scaling updates)
- Saves and restores FP8 scaling tensors (amax, scale, scale_inv) around graph capture to prevent corruption
- During replay, wrap your training iteration with `fp8_autocast(enabled=True)` (default `_graph=False`). On context exit, FP8 scaling factors are automatically reduced and updated via `reduce_and_update_fp8_tensors(forward=True)`

**Handling weight quantization caching** (optional):

- Set `fp8_weight_caching=True` to cache FP8 weight quantization and transposes inside graphs
- Provide `is_first_microbatch` kwarg to control when weights are requantized (only on first microbatch)
- Reduces redundant quantization overhead across microbatches, but requires recapturing if weights change significantly (e.g., during fine-tuning)

### Use Cases

`make_graphed_callables` is best for scenarios where you need **manual control** or work with **non-Megatron models**:

- **Custom models**: You're not using Megatron-LM infrastructure and want to add CUDA graphs to your own PyTorch model
- **Selective graphing**: You want fine-grained control over exactly what gets graphed (e.g., only attention layers, or only specific operations)
- **Research/experimentation**: Prototyping new models or training approaches where Megatron's automatic solutions don't fit
- **Pipeline parallelism with manual scheduling**: Your training uses PP and you can provide the correct pipeline schedule to handle execution ordering

### Pros and Cons

**Advantages**:

- ✅ **Works with any PyTorch model**: Not limited to Megatron layers
- ✅ **Full control**: You decide what to graph and when
- ✅ **Simple conceptually**: Direct wrapper around PyTorch's API
- ✅ **FP8 integration**: Built-in support for FP8 training with TE
- ✅ **Pipeline parallelism support**: Works with PP when user provides correct scheduling

**Disadvantages**:

- ❌ **Manual PP scheduling**: For PP, you must manually ensure correct execution ordering via the `_order` parameter
- ❌ **Requires careful setup**: Must provide correct sample inputs and manage RNG state registration

While `make_graphed_callables` offers maximum flexibility, it may require significant manual effort. For Megatron-LM users, automatic solutions are available that handle much of this complexity.

## CudaGraphManager (Per-Layer Graphing)

### What It Is

For Megatron-LM users, [`CudaGraphManager`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/cuda_graphs.py#L953) provides an automatic solution that captures individual layers as separate CUDA graphs. This fine-grained approach allows graphing specific parts of the model while leaving others in eager mode, and crucially, it automatically handles the pipeline parallelism challenge described earlier.

**Scope**: CudaGraphManager only works with layers that inherit from Megatron's base layer classes:

- [`TransformerLayer`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/transformer_layer.py#L256)
- [`MambaLayer`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/ssm/mamba_layer.py#L50)

It is automatically created and managed by these layer classes when CUDA graphs are enabled via config. Custom layers or non-Megatron modules cannot use CudaGraphManager directly.

### How to Use It

**For Megatron-LM pretrain scripts** (e.g., [`pretrain_gpt.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/pretrain_gpt.py)):

Simply add command-line flags—no code changes needed:

```bash
python pretrain_gpt.py \
    --enable-cuda-graph \
    # cuda_graph_scope defaults to "full" (captures whole Transformer layer) \
    --cuda-graph-num-warmup-steps 3
```

**For custom models using Megatron Core**:

If building custom models or tests with Megatron Core, set `enable_cuda_graph` in TransformerConfig:

```python
from megatron.core.transformer import TransformerConfig, TransformerBlock
from megatron.core.tensor_parallel.random import initialize_rng_tracker

# Initialize RNG tracker first
initialize_rng_tracker(use_te_rng_tracker=True)

# Configure with CUDA graphs enabled
config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    enable_cuda_graph=True,
    cuda_graph_num_warmup_steps=3,
)

# Build your model
model = TransformerBlock(config, layer_spec)  # Or GPTModel, etc.
```

Once configured, Megatron Core automatically creates `CudaGraphManager` for each `TransformerLayer`/`MambaLayer` during model initialization. See [test_cuda_graphs.py](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/tests/unit_tests/transformer/test_cuda_graphs.py) for examples.

### How It Works

[`CudaGraphManager`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/cuda_graphs.py#L953) uses a two-phase JIT (Just-in-Time) approach to automatically handle both the [FP8 challenges](#the-fp8-challenge) and [pipeline parallelism challenge](#the-pipeline-parallelism-challenge).

**Phase 1: Recording execution order** (warmup iterations)

- Graphs are captured **during the first training iterations** (default: 3 warmup steps)
- Each layer records its execution sequence: "Layer N fwd (MB1) → Layer N fwd (MB2) → ... → Layer N bwd (MB2) → Layer N bwd (MB1)"
- A complete execution trace is built via [`_CudagraphGlobalRecord`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/cuda_graphs.py#L164)
- Recording the order is critical because when multiple graphs share a memory pool, they must be created in execution order to ensure proper memory allocation sequencing

**Phase 2: Graph capture** (after warmup, triggered by `create_cudagraphs()`)

- At the end of the first forward-backward pass, `create_cudagraphs()` is called from the pipeline schedule
- Creates **two separate graphs per layer**: one for forward pass, one for backward pass
    - Each [`_CudaGraphRunner`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/transformer/cuda_graphs.py#L484) holds a pair: `fwd_graph` and `bwd_graph`
- **Memory pool strategy** (controlled by `cuda_graph_use_single_mempool`, default `False`):
    - **Default (`False`)**: Uses separate memory pools per microbatch. This enables **graph reuse across microbatches**—the same graph can be replayed for different microbatches since they use independent memory pools. Example: 80 layers × 2 (fwd+bwd) = **160 graphs total per GPU** (reused across all microbatches)
    - **Single mempool (`True`)**: All graphs share one memory pool. Graphs cannot be reused across microbatches due to memory dependencies, requiring one graph per layer per microbatch. Example: 80 layers × 8 microbatches × 2 (fwd+bwd) = **1,280 graphs total per GPU**. This may reduce memory fragmentation overhead but increases graph count.
    - **Without pipeline parallelism (PP=1)**: Automatically uses single mempool with graph reuse, since microbatches execute sequentially in the same order
- Captures graphs in exact recorded order, ensuring proper memory pool sequencing
- Replay automatically follows capture order, preventing corruption
- RNG states are registered for each graph, and optional buffer sharing between layers reduces memory overhead

**FP8 handling** (automatic):

- **Dynamic scaling**: Uses `fp8_autocast(..., _graph=True)` to skip per-layer amax reduction during capture/replay; reduction happens once after all backward graphs
- **Weight quantization caching**: Automatically caches FP8 weights and transposes, updating only on the first microbatch via `skip_fp8_weight_update` flag
- **State preservation**: Saves and restores FP8 tensors around graph capture to prevent corruption

**Result**: This JIT approach works seamlessly with any pipeline parallelism schedule, remains memory efficient by capturing one layer at a time, automatically handles FP8 training, and avoids the replay order mismatch problem.

### Use Cases

CudaGraphManager is the recommended approach when you need **fine-grained control** or work with **pipeline parallelism**:

- **Pipeline parallelism enabled** (PP > 1): This is the primary use case. CudaGraphManager handles the complex interleaved execution of microbatches across pipeline stages, creating separate graphs for each layer-microbatch combination
- **Selective optimization**: You want to graph only performance-critical layers (e.g., transformer blocks) while leaving other parts (embeddings, loss computation) in eager mode for easier debugging

### Pros and Cons

**Advantages**:

- ✅ **Granular control**: Graph only specific layers
- ✅ **Works with pipeline parallelism**: Automatically handles microbatch interleaving
- ✅ **Automatic FP8 handling**: Built-in support for FP8 training with proper state management
- ✅ **Memory efficient**: Optional buffer sharing between layers

**Disadvantages**:

- ❌ **Multiple small graphs**: Each layer = separate graph, limiting overhead reduction
- ❌ **Megatron-only**: Only works with Megatron's TransformerLayer/MambaLayer classes

### Advanced Options

**Buffer Sharing**:

Each CUDA graph requires static input and output buffers allocated upfront. With per-layer graphing, this means every layer needs its own set of buffers, which can quickly consume significant memory. Buffer sharing addresses this by reusing the previous layer's output buffer as the next layer's input buffer, dramatically reducing memory consumption.

Enable buffer sharing:
```python
config = TransformerConfig(
    cuda_graph_share_io_buffers=True  # Reduces memory significantly
)
```

**How it works**: Instead of allocating separate input buffers for each layer, layer N's output buffer becomes layer N+1's input buffer. This chains the layers together in a memory-efficient way.

**Limitation**: Buffer sharing requires that there are **no operations between transformer layers**. If your model has operations (normalization, residual connections, etc.) that happen between layers rather than inside them, buffer sharing may not work correctly.

**External Graph Mode**:

For advanced users who want manual control over graph creation:

```python
config = TransformerConfig(
    external_cuda_graph=True  # Don't create CudaGraphManager automatically
)
# Manually manage graphs via model.cuda_graphs list
```

**Memory Pool Strategy**:

The memory pool strategy depends on pipeline parallelism and user configuration:

- **Without pipeline parallelism (PP=1)**: Automatically uses single mempool with graph reuse—microbatches execute sequentially in the same order, so graphs can be safely reused
- **With pipeline parallelism (PP>1)**: Controlled by `TransformerConfig.cuda_graph_use_single_mempool` (default `False`):

    - `False` (default): Uses separate memory pools per microbatch, enabling graph reuse across microbatches—lower graph count
    - `True`: All graphs share one memory pool, no graph reuse—higher graph count but may reduce memory fragmentation

For even greater overhead reduction, Megatron-LM provides a full-iteration graphing approach that captures entire training iterations.

## FullCudaGraphWrapper (Full-Iteration Graphing)

### What It Is

While `CudaGraphManager` graphs individual layers, [`FullCudaGraphWrapper`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/full_cuda_graph.py#L91) takes a more aggressive approach: it captures the forward and backward passes across all microbatches as a single CUDA graph. This maximizes overhead reduction but requires the entire forward-backward computation to be capturable. Note that optimizer steps, gradient clipping, and learning rate scheduling remain in eager mode outside the graph.

**Scope**: FullCudaGraphWrapper is designed primarily for Megatron-LM's training loop, where it wraps Megatron's `forward_backward_func` (from [`pipeline_parallel.schedules`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/pipeline_parallel/schedules.py)). However, it can also be used with custom models and training loops by providing a compatible `forward_backward_func` that follows the expected interface (see [Examples](#examples) below).

### How to Use It

The commands, requirements, and implementation walkthrough below describe Megatron-LM `core_v0.14.0`. For the newer CLI, see the version note after the requirements.

**For Megatron-LM pretrain scripts** (e.g., `pretrain_gpt.py`):

Enable via command-line flags:

```bash
python pretrain_gpt.py \
    --enable-cuda-graph \
    --cuda-graph-scope full_iteration \
    --cuda-graph-warmup-steps 1 \
    --te-rng-tracker \
    --no-check-for-nan-in-loss-and-grad  # Required in core_v0.14.0
```

Megatron's [training script](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/training/training.py#L1990) automatically wraps `forward_backward_func` with `FullCudaGraphWrapper` when `--cuda-graph-scope full_iteration` is set.

**For custom training scripts**:

If writing a custom training script based on Megatron-LM (similar to [`pretrain_gpt.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/pretrain_gpt.py), [`pretrain_bert.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/pretrain_bert.py), etc.), use the same command-line flags. The automatic wrapping in [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/training/training.py#L1990) handles it automatically—no manual wrapping code needed.

**Requirements for `core_v0.14.0`**:

- **Megatron-LM infrastructure**: Must use `megatron.training` framework with Megatron Core model components (GPTModel, TransformerBlock, etc.)—this is where automatic wrapping happens
- **TE RNG tracker** (`--te-rng-tracker`): Required because standard RNG uses CPU scalars that can't be captured; TE RNG uses device tensors compatible with graphs
- **Disable NaN checking** (`--no-check-for-nan-in-loss-and-grad`): **Mandatory in `core_v0.14.0`** for full_iteration scope. NaN checking requires CPU-GPU synchronization (`.item()` calls) which is forbidden during graph capture

:::{note} Newer Megatron-LM CLI

At revision [`6e7e6ed`](https://github.com/NVIDIA/Megatron-LM/tree/6e7e6ed435e41af04fc41e7007ef862049153fdc), enable full-iteration graphing with:

```bash
--cuda-graph-impl full_iteration \
--cuda-graph-warmup-steps 1
```

This replaces the older `--enable-cuda-graph --cuda-graph-scope full_iteration` pair. Omit `--cuda-graph-modules` for full-iteration capture.

With `--transformer-impl transformer_engine`, the [CLI automatically enables the TE RNG tracker](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/training/arguments.py#L1971-L1977) for CUDA graphs. An explicit `--te-rng-tracker` flag is optional in that configuration.

Loss and gradient checks can remain enabled. The [graph wrapper](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/megatron/core/full_cuda_graph.py#L232-L256) captures validation values and runs deferred host-side checks after replay, before the training loop decides whether to rerun or update the optimizer. The older requirement to pass `--no-check-for-nan-in-loss-and-grad` does not apply to this revision. See the [versioned CUDA graph guide](https://github.com/NVIDIA/Megatron-LM/blob/6e7e6ed435e41af04fc41e7007ef862049153fdc/docs/user-guide/features/cuda_graph.md) for the newer interface.
:::

### How It Works

FullCudaGraphWrapper uses a JIT (Just-in-Time) capture approach and operates in three phases:

**Warmup phase** (default: 1 iteration, JIT): Runs in eager mode while the [`StaticBufferLoader`](https://github.com/NVIDIA/Megatron-LM/blob/core_v0.14.0/megatron/core/full_cuda_graph.py#L57) pre-allocates static buffers for all microbatch inputs. The graph is **not yet captured** during this phase.

**Capture iteration** (iteration `warmup_steps + 1`, JIT): Graphs are captured **during training** at this specific iteration. Reads all microbatch data from the dataloader and copies it to static buffers. Then registers all RNG states and captures the entire `forward_backward_func` call as a single CUDA graph using `thread_local` capture mode. This graph contains forward and backward passes for all microbatches. Separate graphs are created for training and validation modes.

**Replay phase** (all subsequent iterations): Copies new data to the static buffers and replays the graph, executing all forward and backward passes. Operations outside `forward_backward_func`—optimizer step, gradient clipping, LR scheduler updates, and logging—remain in eager mode and execute after each graph replay.

### Use Cases

FullCudaGraphWrapper is best suited for scenarios where you want maximum overhead reduction and performance boost:

- **Maximum performance priority**: When you need the absolute best forward/backward performance
- **Static workloads**: Training with fixed batch sizes, sequence lengths, and microbatch counts—no dynamic control flow and shapes
- **Training loop flexibility**: Since optimizer, gradient clipping, and LR scheduling remain in eager mode, you can experiment with different optimizers or hyperparameters without recapturing graphs

### Pros and Cons

**Advantages**:

- ✅ **Simple conceptually**: One capture, one replay per iteration
- ✅ **Minimal graph count**: Only 2 graphs (training + validation)
- ✅ **Comprehensive forward/backward**: All microbatches' fprop and bprop in one graph
- ✅ **Maximum overhead reduction**: Runtime/Host overhead reduced most

**Disadvantages**:

- ❌ **Less flexible**: All-or-nothing approach
- ❌ **Debugging harder**: Entire iteration is opaque

### Examples

For users who need to manually integrate `FullCudaGraphWrapper` in custom training loops outside the standard Megatron pretraining scripts, here's a minimal example with the optimizer in eager mode:

```python
import torch
import torch.distributed as dist
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.tensor_parallel.random import initialize_rng_tracker

# Initialize distributed and RNG
dist.init_process_group(backend='nccl', rank=0, world_size=1)
initialize_rng_tracker()

# Model and training setup
model = torch.nn.Sequential(
    torch.nn.Linear(4096, 2048),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(2048, 1024)
).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.MSELoss()

# Define forward-backward function matching FullCudaGraphWrapper's signature
def forward_backward_func(data_iterator, model, num_microbatches, seq_length, forward_only):
    data = next(data_iterator[0])
    y_pred = model(data['input'])
    loss = loss_fn(y_pred, data['target'])
    if not forward_only:
        loss.backward()
    return loss

# Wrap with FullCudaGraphWrapper
forward_backward_func = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)

# Training loop
for data, target in zip(inputs, targets):
    data_iterator = iter([{'input': data, 'target': target}])

    # Call wrapped function (graph captured on iteration warmup_steps+1, then replayed)
    loss = forward_backward_func(
        data_iterator=data_iterator,
        model=model,
        num_microbatches=1,
        seq_length=4096,
        forward_only=False
    )

    # Optimizer step remains in eager mode (outside the graph)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

**Key points**:

- `forward_backward_func` must accept the specific arguments expected by `FullCudaGraphWrapper` (`data_iterator`, `model`, `num_microbatches`, `seq_length`, `forward_only`)
- The wrapper handles graph capture automatically after warmup iterations
- Optimizer step and zero_grad remain outside the graph in eager mode

For maximum performance, you can include the optimizer inside the graph. The key changes are:

```python
# Modify forward_backward_func to include optimizer operations
def forward_backward_func(data_iterator, model_and_optimizer, num_microbatches, seq_length, forward_only):
    model, optimizer = model_and_optimizer  # Unpack tuple
    data = next(data_iterator[0])

    optimizer.zero_grad(set_to_none=True)  # Inside graph
    y_pred = model(data['input'])
    loss = loss_fn(y_pred, data['target'])

    if not forward_only:
        loss.backward()
        optimizer.step()  # Inside graph

    return loss

# Pass model and optimizer as tuple when calling
loss = forward_backward_func(
    data_iterator=data_iterator,
    model=(model, optimizer),  # Pass as tuple
    num_microbatches=1,
    seq_length=4096,
    forward_only=False
)

# No optimizer.step() or optimizer.zero_grad() outside the graph
```

**Benefits**: All optimizer operations are captured and replayed during graph execution, eliminating the overhead of launching optimizer kernels in eager mode.

Now that we've examined all three approaches, let's compare them side-by-side.

## Comparison

| Aspect | TE `make_graphed_callables` | `CudaGraphManager` | `FullCudaGraphWrapper` |
|--------|----------------------------|------------------|---------------------|
| **Scope** | Per-callable (manual) | Per-layer/module | Full iteration |
| **Capture timing** | AOT (before training) | JIT (after `cuda_graph_warmup_steps` iterations) | JIT (after `cuda_graph_warmup_steps` iterations) |
| **Graph count** | User-defined | Many (one per layer per microbatch) | Few (training + validation) |
| **Overhead reduction** | Depends on what you graph | Moderate | Maximum |
| **Memory usage** | Depends on implementation | Lower with buffer sharing | Lowest |
| **FP8 handling** | Manual (with helpers) | Automatic | Automatic |
| **Pipeline parallelism** | Supported (manual scheduling) | Supported (automatic) | Supported |
| **Flexibility** | Maximum (full manual control) | High (selective layering) | Low (all-or-nothing) |
| **Setup complexity** | Manual (requires code changes) | Automatic (via config) | Automatic (via config) |
| **Effort** | High (manual capture, replay & PP scheduling) | Medium | High (ensure fprop & bprop capturable) |
| **Model compatibility** | Any PyTorch model | Megatron layers only | Megatron-LM training loop |

**When to use each**:

- **TE `make_graphed_callables`**: Custom PyTorch models, research prototypes, or when you need fine-grained control over what gets graphed
- **CudaGraphManager**: Default choice for Megatron-LM training, especially with pipeline parallelism (PP > 1)
- **FullCudaGraphWrapper**: Maximum performance when you can make forward+backward fully capturable

## What's Next?

- **[Best Practices for PyTorch](best-practices.md)**: General PyTorch CUDA Graph best practices
- **[PyTorch Integration](torch-integration.md)**: Understanding the underlying PyTorch CUDA Graph APIs
- **[Examples](../examples/introduction.md)**: See CUDA Graphs in action with RNN-T, Stable Diffusion, Llama 3.1 405B, and DeepSeek-V3 671B
- **[Megatron Core Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)**: Official Megatron Core developer guide
