<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Llama 3.1 405B

:::{note}

CUDA Graph optimization for Llama 3.1 405B training, based on NVIDIA's MLPerf Training v5.1 (2025/11) implementation. This example demonstrates full-iteration graphing with `FullCudaGraphWrapper` for large-scale transformer training with pipeline parallelism.
:::

## Overview

Llama 3.1 405B is Meta's flagship large language model released in July 2024, representing the first frontier-level open source AI model. As part of the Llama 3.1 family, it is a dense transformer-based autoregressive decoder-only model with 405 billion parameters, featuring a 128K token context window, multilingual support across eight languages, and state-of-the-art capabilities that rival the best closed-source models ([Meta AI blog](https://ai.meta.com/blog/meta-llama-3-1/)).

The model incorporates several innovative architectural techniques that enhance performance and scalability:

1. **Root Mean Square Layer Normalization (RMSNorm)**: A more efficient normalization technique that reduces computational cost while maintaining training stability
2. **Rotary Position Embeddings (RoPE)**: Advanced position encoding that enables the model to handle extended context lengths effectively
3. **Grouped-Query Attention (GQA)**: An attention mechanism that balances computational efficiency with model quality by grouping key-value heads
4. **SwiGLU Activation**: A gated linear unit activation function that improves model expressiveness

The model architecture follows the standard decoder-only transformer design with **126 transformer blocks**, each containing self-attention and feed-forward (MLP) layers, bookended by token embeddings and an output projection head.

![Llama Architecture](https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/452342830_524225500031704_780745667054798266_n.png?_nc_cat=110&ccb=1-7&_nc_sid=e280be&_nc_ohc=K-oHFwMB2T0Q7kNvwHRAVC8&_nc_oc=Adks6ePCq4TE-YYDTu6LDdVxnggzw65_wlsfsQaLrvS4yGDOYw2TPuW3QaNAoqKcDxk&_nc_zt=14&_nc_ht=scontent-hkg4-1.xx&_nc_gid=Kv-f9DKZrWpyq6X3pqUVcA&oh=00_AfiircJ5VrO7-AzxF-ZRNV_A1fYC2ZvbAlaEx5viWy_SuA&oe=693E0E0D)

*Figure: Llama 3.1 architecture showing the decoder-only transformer design with 126 layers (Image source: [Meta AI Blog](https://ai.meta.com/blog/meta-llama-3-1/))*

Training such a massive model requires advanced distributed training techniques including tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), and data parallelism (DP) to partition the model across hundreds or thousands of GPUs.

## MLPerf Training Llama 3.1 405B Benchmark

In May 2025, MLCommons introduced Llama 3.1 405B into the MLPerf Training v5.0 benchmark suite, replacing the previous GPT-3 benchmark. This change reflects the rapid evolution of AI models and the need for benchmarks that accurately represent current state-of-the-art architectures. Llama 3.1 405B offers several advantages as a benchmark:

- **Modern architecture**: Incorporates recent innovations like RMSNorm, RoPE, and GQA that are now standard in frontier models
- **Extended context**: Uses 8,192-token sequences—four times larger than GPT-3's 2,048 tokens—enabling evaluation of models' ability to process and understand longer contexts
- **Open source**: As Meta's first frontier-level open model, it provides transparency and reproducibility for the research community

**Dataset and Task**:

The benchmark uses the **AllenAI C4-en (Colossal Clean Crawled Corpus)** dataset, comprising 365 million English-language paragraphs extracted from web crawl data. The reference implementation starts from a fully trained and stable checkpoint from Hugging Face, with modifications to adapt to a new token distribution. This approach maintains consistency and reduces complexity in benchmark execution.

**Reference Implementation**:

The MLPerf reference implementation is built on the **NVIDIA NeMo Framework**, an open-source and scalable AI framework tailored for large language models. The implementation uses **NVIDIA NeMo + Megatron-Core + PyTorch Lightning + Transformer Engine** as the software stack and has been tested on NVIDIA H100, B200, and GB200 GPUs.

Training employs 3D parallelism combining tensor parallelism (TP), pipeline parallelism (PP), context parallelism (CP), and data parallelism (DP). Example configurations include:

- **512 H100 GPUs**: TP=8, PP=9, CP=2, DP=4, global batch size 2,304
- **448 B200 GPUs**: TP=4, PP=8, CP=2, DP=7, global batch size 2,016

Training uses BF16 mixed precision with FP8 (Transformer Engine) for compute-intensive operations, gradient accumulation across multiple microbatches, and fixed-size sequences (8,192 tokens, no sequence packing).

**Quality Metric**:

The benchmark measures **log perplexity on the C4 validation set**, with a target of ≤ 5.6. This metric evaluates how well the model predicts the next token in sequences, with lower perplexity indicating better language understanding. Evaluation runs every 46,080 sequences, using 5,760 sequences per evaluation.

For complete model architecture details, dataset preparation, and optimizer configuration, see the [MLPerf Training reference](https://github.com/mlcommons/training/blob/95f90e857ade5392f83f54393cdccb55d12ac67f/large_language_model_pretraining/nemo/README.md).

**CUDA Graph Challenge**:

To maximize performance benefits, CUDA graphs should capture as many operations as possible—ideally the entire forward-backward computation. However, achieving full-iteration graphing is extremely challenging in practice:

1. **Complex software stack**: The training infrastructure (NeMo + Megatron-Core + PyTorch Lightning + Transformer Engine) involves multiple layers of abstraction, each potentially introducing CUDA graph incompatibilities
2. **Graph-incompatible operations**: Components throughout the stack may perform CPU-GPU synchronization, use Python control flow, or call non-capturable CUDA APIs
3. **Distributed training complexity**: Pipeline parallelism with interleaved microbatch execution, combined with tensor/data/context parallelism, creates intricate execution patterns that are difficult to capture
4. **High-level framework abstractions**: Deploying full-iteration CUDA graphs with NeMo/Megatron-LM/PyTorch Lightning requires navigating framework callbacks, hooks, and automatic optimizations that weren't designed with graphing in mind

Despite these challenges, NVIDIA's MLPerf Training v5.1 (2025/11) implementation successfully uses Megatron-LM's `FullCudaGraphWrapper` to capture the entire forward-backward computation across all microbatches as a single CUDA graph, achieving significant performance improvements at scale.

## Integration Approach: Full-Iteration Graphing

This section describes how the MLPerf v5.1 implementation overcomes the challenges above to deploy full-iteration CUDA graphs with `FullCudaGraphWrapper`.

- **Implementation**: [training_results_v5.1/NVIDIA/.../tyche_ngpu512_ngc25.09_nemo](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo)
- **Framework versions**: PyTorch NGC 25.09, NeMo 25.09-alpha.rc1, Megatron-Core 25.09-alpha.rc1, Transformer Engine v2.8

### Capture Scope

The v5.1 implementation uses **full-iteration graphing** with [`FullCudaGraphWrapper`](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py), capturing the entire `forward_backward_func` as a single CUDA graph.

**Why full-iteration over partial graphing?**

Full-iteration CUDA graphs provide significantly better benefits compared to per-layer (partial) graphing:

- **Higher memory savings with Pipeline Parallelism**: Eliminates the need for separate input/output buffers while allowing activation reuse between layers and microbatches
- **Better performance**: Reduced jitter and CPU overheads by including pipeline parallel and data parallel communication kernels in the graph
- **Reduced adaptation overhead**: Little or no special handling required to adapt to new software optimizations like FSDP, offloading, combined 1F1B layers, etc.
- **Necessary on modern hardware**: On latest GPUs with FP8/FP4 precision, even partial CUDA graphs are insufficient to hide CPU launch overheads and communication jitter

**Captured in the graph**:

- **Forward and backward passes**: Complete forward and backward computation for all microbatches
- **All transformer layers**: 126 decoder layers with self-attention and MLP components
- **Model components**: Embedding layer, output projection (loss computation), LayerNorm, dropout, activation functions (SwiGLU)
- **Distributed training operations**:

    - Data Parallelism (DP): Gradient all-reduce across data-parallel ranks
    - Pipeline Parallelism (PP): Send/recv communication between pipeline stages
    - Tensor Parallelism (TP): All-reduce for tensor-parallel operations
    - Context Parallelism (CP): Sequence-parallel communication

- **Gradient accumulation**: Accumulation across all microbatches within the training iteration

**Remaining in eager mode**:

- Data loading (dataloader)
- Optimizer step (Megatron distributed optimizer)
- Learning rate scheduling
- Loss logging and validation metrics

**Scale**: CUDA graphs deployed across multiple configurations:

- **H100 (DGX H100)**: 24-1,024 nodes (192-8,192 GPUs), e.g., 512 nodes × 8 GPUs = 4,096 GPUs
- **B200 (DGX B200)**: 64 nodes (512 GPUs), e.g., 64 nodes × 8 GPUs = 512 GPUs
- **GB200**: 64-1,280 nodes (256-5,120 GPUs), e.g., 128 nodes × 4 GPUs = 512 GPUs
- **GB300**: Similar configurations to GB200

**Configuration examples**:

- **512 H100 nodes (4,096 GPUs)**: TP=8, PP=8, CP=2, interleaved (virtual) PP=8, gradient accumulation steps=40
- **64 B200 nodes (512 GPUs)**: TP=4, PP=8, CP=2, interleaved (virtual) PP=8, gradient accumulation steps=144
- **128 GB200 nodes (512 GPUs)**: TP=4, PP=8, CP=2, interleaved (virtual) PP=8, gradient accumulation steps=72

Single CUDA graph captures all 126 transformer layers across all pipeline stages for all gradient accumulation steps (microbatches).

### FullCudaGraphWrapper Integration

`FullCudaGraphWrapper` is enabled via configuration flags in NeMo (see [Transformer Engine and Megatron-LM CUDA Graphs](../torch-cuda-graph/te-megatron-cuda-graphs.md#fullcudagraphwrapper-full-iteration-graphing) for full implementation details).

**Configuration** ([custom.yaml](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/conf/custom.yaml)):

```yaml
model:
  enable_cuda_graph: ${oc.decode:${oc.env:MCORE_CUDA_GRAPH,False}}
  cuda_graph_scope: ${oc.decode:${oc.env:CUDA_GRAPH_SCOPE,${if:${eq:'1',${oc.env:FULL_CUDA_GRAPH,0}},'full_iteration','full'}}}
```

Set via environment variables ([config_common_cg.sh](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/config_common_cg.sh)):

```bash
export FULL_CUDA_GRAPH=1
export MCORE_CUDA_GRAPH=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False
```

**How it works**:

1. **`MCORE_CUDA_GRAPH=1`**: Enables Megatron-Core CUDA graph support (`enable_cuda_graph=True`)
2. **`FULL_CUDA_GRAPH=1`**: Sets `cuda_graph_scope='full_iteration'`, which triggers `FullCudaGraphWrapper` instead of per-layer graphing
3. **`TORCH_NCCL_AVOID_RECORD_STREAMS=1`**: Required for CUDA graph compatibility - avoids `record_stream()` calls that cannot be captured
4. **`OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False`**: Disables parameter gather overlap to avoid stream synchronization issues during capture

When these are set, NeMo's training loop automatically wraps the `forward_backward_func` with `FullCudaGraphWrapper`. During the first training iteration (after warmup), it captures the entire forward-backward computation across all microbatches as a single CUDA graph.

**Alternative: Megatron-LM command-line arguments**

For Megatron-LM models without NeMo, full-iteration CUDA graphs can be enabled using runtime arguments:

```bash
python pretrain_gpt.py \
  --cuda-graph-impl local \
  --cuda-graph-scope full_iteration \
  ... # other training arguments
```

The `FullCudaGraphWrapper` class handles all necessary operations automatically:

- Copy CUDA graph inputs to static memory locations
- Store separate CUDA graphs for training and validation
- Track current training/validation steps for warmup management
- Register random generator instances with CUDA graph before capture
- Replay respective CUDA graphs depending on training/validation cycles

**Key feature**: The MLPerf implementation includes custom warmup infrastructure ([custom_callbacks.py](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/custom_callbacks.py)) to handle synthetic data generation and state management required for MLPerf timing compliance.

### Custom Warmup Infrastructure

CUDA graphs require warmup iterations before capture to allow NCCL and the CUDA memory allocator to stabilize their internal states. The number of iterations needed depends on the DDP implementation:

- **PyTorch DDP**: Requires ~11 iterations for gradient bucketing setup and initialization
- **Megatron-LM DDP**: Requires fewer iterations (typically 2) as it uses custom gradient bucketing without PyTorch DDP's initialization overhead

**MLPerf Llama 3.1 405B** uses Megatron-LM's `DistributedDataParallel`, allowing for shorter warmup.

:::{tip} MLPerf-Specific: Synthetic Data and State Reset

The following MLPerf-specific implementation (synthetic data generation, optimizer state reset, FP8 state reset) addresses MLPerf timing compliance requirements. For general CUDA graph adoption with Megatron-LM, you can use real data for warmup iterations and skip the state reset logic. The base `FullCudaGraphWrapper` handles CUDA graph capture automatically.
:::

**MLPerf's additional challenge**: MLPerf timing starts when the model first touches real data, but running warmup with real data would count toward timing. The solution is to run warmup with synthetic (fake) data before timing starts, then reset optimizer and FP8 states to prevent pollution from fake gradients.

**Configuration** ([custom.yaml](https://github.com/mlcommons/training_results_v5.1/blob/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/conf/custom.yaml#L196)):

```yaml
warmup_train_steps: ${oc.decode:${oc.env:WARMUP_TRAIN_STEPS,2}}
```

Default is 2 iterations, which is sufficient for Megatron-LM DDP to stabilize NCCL and allocator state before capture.

**Implementation** (simplified version, see [custom_callbacks.py](https://github.com/mlcommons/training_results_v5.1/blob/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/custom_callbacks.py) for complete code):

```python
def run_training_warmup(trainer, warmup_train_steps, full_cuda_graph):
    # Create synthetic data loader with static buffers
    mock_dataloader = iter(StaticBufferLoader(
        trainer.mock_dataset.train_dataloader(),
        state=CUDAGraphState(stage="training")
    ))

    # Save and disable optimizer momentum during warmup
    optimizer = trainer.strategy.optimizers[0]
    for group in optimizer.param_groups:
        group["betas_"] = group["betas"]  # Save original
        group["betas"] = [1.0, 1.0]  # Disable momentum

    # Run warmup iterations in eager mode (allow Megatron DDP/NCCL/allocator to stabilize)
    for i in range(warmup_train_steps - 1):
        trainer.model.training_step(data=mock_dataloader)
        optimizer.step()
        optimizer.zero_grad()

    # Final iteration: capture CUDA graph
    trainer.model.training_step(data=mock_dataloader, capture_cuda_graph=full_cuda_graph)

    # Restore optimizer momentum (prevent fake gradient pollution)
    for group in optimizer.param_groups:
        group["betas"] = group["betas_"]  # Restore original

    # Reset FP8 stats (prevent fake data pollution, see below)
```

**Key features**:

1. **Synthetic data generation**: Uses `MockGPTDataset` to create random token sequences matching the model's input shape
2. **Static buffer loader**: `StaticBufferLoader` wraps the dataloader to copy batches to fixed GPU addresses, ensuring graph inputs remain valid
3. **Optimizer state reset**: Disables momentum (`betas=[1.0, 1.0]`) during warmup to prevent polluting optimizer state with synthetic gradients, then restores original settings
4. **FP8 state reset**: Resets Transformer Engine FP8 scaling factors (`amax`, `scale`, `scale_inv`) after warmup to ensure they calibrate to real data distributions rather than synthetic data

**Why FP8 reset matters**: Transformer Engine's FP8 training maintains scaling factors that are updated based on observed tensor magnitudes. During warmup with synthetic random data, these statistics accumulate based on random values rather than real data distributions. Resetting the `fp8_initialized` flag ([implementation](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/custom_callbacks.py)) forces re-calibration on the first real training iteration, preventing potential numerical instability or convergence issues.

**Validation warmup**: Similar process for validation graphs, though validation graphs provide smaller benefits since backward passes are skipped.

## Essential Modifications for CUDA Graph Compatibility

This section covers the essential changes required to make Megatron-LM's `FullCudaGraphWrapper` work correctly with Llama 3.1 405B training.

### 1. Eliminating CPU-GPU Synchronizations

**Problem**: Operations that cause host-device synchronizations cannot be captured in CUDA graphs. The most common issue is creating tensors directly on GPU with `torch.tensor(device='cuda')`, which triggers blocking `cuStreamSynchronize` calls.

**Solution**: Replace sync-inducing operations with direct GPU allocation (`torch.zeros`, `torch.empty`) and in-place fills. For comprehensive background on synchronizations and detection methods, see [Writing Sync-Free Code](../torch-cuda-graph/sync-free-code.md).

**Example 1: Replace scalar tensor creation**

```python
# Before (causes sync)
total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

# After (sync-free)
total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
```

**Why**: `torch.zeros()` allocates directly on GPU, while `torch.tensor(0).cuda()` creates a CPU tensor then performs a blocking transfer.

**Example 2: Replace index tensor with transfer**

```python
# Before (causes sync)
index = torch.tensor(
    [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
).cuda(non_blocking=True)

# After (sync-free)
index = torch.zeros(2, dtype=torch.int64, device=val.device)
index[0].fill_(cp_rank)
index[1].fill_(2 * cp_size - cp_rank - 1)
```

**Why**: Direct GPU allocation with `.fill_()` avoids the synchronization from `torch.tensor()` creating a CPU tensor and transferring it (even with `non_blocking=True`).

**Key pattern**: Replace `torch.tensor(value, device='cuda')` with `torch.zeros(..., device='cuda')` + in-place operations (`.fill_()`, indexing). Use `torch.cuda.set_sync_debug_mode('warn')` to detect synchronizations during development.

### 2. Static Input Buffers

**Problem**: CUDA graphs record GPU memory addresses during capture. If input data tensors have different memory addresses on each iteration, the graph will access stale or incorrect memory locations during replay, causing wrong results or crashes.

**Why this matters**: PyTorch dataloaders typically allocate new tensors for each batch, which get different GPU addresses each time. CUDA graphs require all input tensors to reside at the same GPU addresses across all replays.

**Solution**: `FullCudaGraphWrapper` includes an internal `StaticBufferLoader` that automatically manages static buffers for all microbatches ([full_cuda_graph.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py#L57-L91)):

```python
class StaticBufferLoader:
    """Load data to static buffers."""

    static_buffers: dict = {'training': [], 'validation': []}

    def __init__(self):
        self.stream = torch.cuda.Stream()

    def __call__(self, inputs, stage, microbatch):
        # stage: 'training' or 'validation'
        # microbatch: index of current microbatch

        if microbatch == len(StaticBufferLoader.static_buffers[stage]):
            # First time: allocate static buffer for this microbatch
            with torch.cuda.stream(self.stream):
                StaticBufferLoader.static_buffers[stage].append(
                    copy_tensors_in_struct(inputs)
                )
        else:
            # Subsequent iterations: copy data to existing buffer
            with torch.cuda.stream(self.stream):
                clone_tensors_in_struct(
                    StaticBufferLoader.static_buffers[stage][microbatch],
                    inputs
                )

        return StaticBufferLoader.static_buffers[stage][microbatch]
```

**How it works**:

1. **Separate buffers per microbatch**: Maintains a list of static buffers, one for each microbatch in the iteration
2. **Stage separation**: Separate buffer pools for training vs. validation to avoid conflicts
3. **First iteration**: Allocates GPU tensors for each microbatch as they arrive
4. **Subsequent iterations**: Copies new data into existing buffers, maintaining the same GPU addresses
5. **Asynchronous copying**: Uses a dedicated CUDA stream to overlap data copying with computation

**Integration**: `FullCudaGraphWrapper` automatically uses `StaticBufferLoader` in its `data_read()` method ([full_cuda_graph.py line 106-137](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py#L106-L137)):

```python
class FullCudaGraphWrapper:
    def __init__(self, forward_backward_func, cuda_graph_warmup_steps=1):
        self.forward_backward_func = forward_backward_func
        self.static_loader = StaticBufferLoader()  # Internal static buffer loader
        self.cuda_graph_warmup_steps = cuda_graph_warmup_steps

    def data_read(self, data_iterator, model, training, num_microbatches):
        """Read all microbatch inputs and copy to static buffers."""
        data_list = []
        for b in range(num_microbatches):
            data_list.append(
                self.static_loader(
                    next(data_iterator),
                    'training' if training else 'validation',
                    b  # microbatch index
                )
            )
        return [iter(data_list)]
```

**User code**: The MLPerf implementation doesn't need to explicitly manage `StaticBufferLoader`. It simply uses regular dataloaders, and `FullCudaGraphWrapper` handles the static buffer management automatically when wrapping the forward-backward function.

### 3. Pipeline Parallelism Considerations

With pipeline parallelism, microbatches execute in an interleaved pattern across pipeline stages, with send/recv communication between stages. **Unlike per-layer graphing which requires dedicated handling for pipeline communication, full-iteration graphing naturally captures the entire pipeline schedule** - all send/recv operations, microbatch interleaving, and the complete execution pattern are included in the graph automatically.

**Key requirements**:

- Warmup must use the same pipeline schedule and number of microbatches as actual training
- Each microbatch must use static buffers for input data (handled by `StaticBufferLoader` discussed in the previous subsection)

This ensures the captured graph structure and memory addresses match what will be replayed during training.

### 4. RNG State Management for CUDA Graphs

**Problem**: PyTorch's default RNG uses CPU scalars for seed and offset, which cannot be captured in CUDA graphs. When operations like dropout use the global RNG during graph capture, the seed becomes a constant, causing identical dropout masks on every replay.

**Solution**: `FullCudaGraphWrapper` automatically registers all RNG generator states with the CUDA graph during capture. This includes both Transformer Engine's RNG tracker (when enabled) and Megatron-Core's built-in RNG tracker. The RNG states are stored as GPU tensors that can be updated between graph replays using PyTorch's `register_generator_state` API.

**Configuration**: Enable Transformer Engine's RNG tracker for tensor parallelism. In NeMo Lightning ([MLPerf config](https://github.com/mlcommons/training_results_v5.1/blob/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/conf/custom.yaml#L129)):

```yaml
model:
  use_te_rng_tracker: True
```

Or in Megatron-LM command-line ([argument definition](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/training/arguments.py#L1337-L1339)):

```bash
--te-rng-tracker
```

**How it works**: During CUDA graph capture, `FullCudaGraphWrapper` calls [`get_all_rng_states()`](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/tensor_parallel/random.py#L304-L323) to collect all RNG generator states from the active tracker (either Transformer Engine's [`TECudaRNGStatesTracker`](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/extensions/transformer_engine.py#L1745-L1777) or Megatron's [`CudaRNGStatesTracker`](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/tensor_parallel/random.py#L126-L293)), and [registers each state](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py#L168-L169) with the CUDA graph. Both trackers maintain separate RNG states for different parallelism groups (TP, DP, CP, EP), with Transformer Engine's tracker providing additional optimizations for tensor-parallel operations and FP8 training.

### 5. Stream Management

**Why side streams are required**: CUDA graphs cannot capture on the default stream because it is a blocking (synchronizing) stream that implicitly synchronizes with other streams. Graph capture requires a non-blocking stream to properly record the execution sequence without implicit synchronization barriers.

**DDP initialization on side stream**: Megatron-LM wraps DDP initialization in a side stream context ([implementation](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/training/training.py#L987-L999)):

```python
with torch.cuda.stream(torch.cuda.Stream()):
    model = [
        DP(
            config=config,
            ddp_config=ddp_config,
            module=model_chunk,
            disable_bucketing=(model_chunk_idx > 0)
            or args.overlap_param_gather_with_optimizer_step,
        )
        for (model_chunk_idx, model_chunk) in enumerate(model)
    ]
```

This ensures DDP's internal state initialization doesn't interfere with subsequent CUDA graph capture on the default stream.

**Graph capture on side stream**: `FullCudaGraphWrapper` captures the training/validation loop on a separate stream ([implementation](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py#L170-L180)):

```python
capture_stream = torch.cuda.Stream()
with torch.cuda.graph(
    FullCudaGraphWrapper.cuda_graph[training_str],
    stream=capture_stream,
    capture_error_mode="thread_local",
):
    FullCudaGraphWrapper.result[training_str] = self.forward_backward_func(*args, **kwargs)
```

**Key aspects**:

- **Thread-local error mode**: Uses `capture_error_mode="thread_local"` to allow CUDA operations from the current thread while rejecting operations from other threads (required for NCCL collectives on background streams)
- **Automatic stream handling**: PyTorch's `torch.cuda.graph()` context manager handles stream synchronization before and after capture

Users don't need to configure stream management manually - Megatron-LM handles all stream orchestration internally.

### 6. Disabling PyTorch NCCL Record Streams

**Problem**: PyTorch's `record_stream()` marks tensors as used on multiple streams. When allocating memory during graph capture, the caching allocator must check if those streams have completed using `cudaEventQuery()`, which is forbidden during capture. This prevents memory recycling and causes memory accumulation.

**Solution**: Disable `record_stream()` for NCCL operations ([config_common_cg.sh](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/config_common_cg.sh)):

```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

PyTorch's NCCL process group uses an alternative mechanism (stashing tensor references) that doesn't require `cudaEventQuery()`, enabling memory recycling during graph capture.

For detailed explanation, see the [Llama 2 70B LoRA example](llama2-70b-lora.md#2-disabling-pytorch-nccl-record-streams).

### 7. NCCL Buffer Registration with Expandable Segments

The MLPerf implementation enables PyTorch's expandable segments allocator mode to reduce memory fragmentation:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Expandable segments allow PyTorch's caching allocator to extend memory segments on demand, providing a contiguous virtual address space while using multiple physical segments internally. This significantly reduces fragmentation for large-scale training workloads.

**Problem**: NCCL can register user buffers to improve communication performance. However, current NCCL versions only support registering a single physical segment. When buffers span multiple physical segments (as with expandable segments), this can cause issues during CUDA graph capture or replay.

**Solution**: Disable NCCL buffer registration:

```bash
export NCCL_GRAPH_REGISTER=0
```

Future NCCL versions will add multi-segment registration support, eliminating the need for this workaround.

### 8. Disabling Parameter Gather Overlap with Optimizer Step

**Problem**: The distributed optimizer normally overlaps parameter all-gather communication with computation in two complementary ways:

1. **Standard overlap**: When a layer's forward pre-hook fires ([_make_forward_pre_hook](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/distributed/distributed_data_parallel.py#L409-L443)), it waits for that layer's parameter all-gather to complete, then dispatches the *next* layer's all-gather ([finish_param_sync → next bucket dispatch](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/distributed/param_and_grad_buffer.py#L288-L301)). This allows the next layer's all-gather to overlap with the current layer's forward computation.
2. **Optimizer-step overlap** (when `OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=True`): All parameters' all-gather is dispatched immediately after the optimizer step completes ([ChainedOptimizer.step_with_ready_grads](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/optimizer/optimizer.py#L1243-L1246)), allowing it to overlap with subsequent optimizer computation (if any) *and* the entire forward pass of the next iteration.

However, with full-iteration CUDA graphs, this second optimization creates an external event dependency.

**How the optimization works** (when enabled):

1. **Optimizer step launches async parameter all-gather**: After updating parameters in the optimizer step (eager mode, outside graph), an asynchronous all-gather is dispatched on a side stream ([ChainedOptimizer.step_with_ready_grads](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/optimizer/optimizer.py#L1238-L1248))

    ```python
    # In ChainedOptimizer.step_with_ready_grads()
    if self.config.overlap_param_gather_with_optimizer_step and optimizer_idx == 0:
        # Launch async all-gather immediately after first optimizer's step
        optimizer.model_chunks[0].start_param_sync(force_dispatch=True)
    ```

2. **Forward pass waits for all-gather completion**: At the start of the next forward pass, the forward pre-hook ([_make_forward_pre_hook](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/distributed/distributed_data_parallel.py#L409-L443)) waits for the all-gather to complete before each module accesses its parameters

    ```python
    # In _make_forward_pre_hook()
    skip_next_bucket_dispatch = (
        self.ddp_config.align_param_gather
        or self.overlap_param_gather_with_optimizer_step  # All-gather already dispatched
    )
    self.param_to_bucket_group[param].finish_param_sync(...)  # Waits for handle
    ```

**Why this breaks CUDA graphs**: With full-iteration CUDA graphs:

- **Forward/backward passes are inside the graph** (captured region)
- **Optimizer step remains outside the graph** (eager mode)
- When the graphed forward pass tries to wait on an all-gather event recorded by the optimizer step (outside the graph), it creates a dependency on an **external event**
- This violates CUDA graph's [self-contained execution constraint](../cuda-graph-basics/constraints.md#4-self-contained-stream-capture): graphs cannot wait on events or streams created outside the capture context

**Solution**: Disable the optimizer-step overlap by setting the environment variable ([config_common_cg.sh](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/config_common_cg.sh)):

```bash
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False
```

This environment variable is parsed by the MLPerf configuration ([custom.yaml](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/conf/custom.yaml#L167)) and passed to Megatron-LM's distributed optimizer configuration, which controls the `overlap_param_gather_with_optimizer_step` flag.

**Impact**: With this disabled, parameter all-gather still overlaps with forward computation (standard overlap #1 above) through the layer-by-layer pipeline in forward pre-hooks, keeping the graph self-contained. The trade-off is losing the early all-dispatch from the optimizer step that would have allowed all parameters to begin gathering immediately. However, CUDA graph benefits far outweigh this minor regression.

## Performance Impact

CUDA graph optimization for Llama 3.1 405B with `FullCudaGraphWrapper` provides significant performance improvements, with benefits varying by configuration and scale.

**Full-Iteration Graphing Benefits**:

- **Maximum kernel launch overhead reduction**: Entire forward-backward computation across all microbatches captured as a single graph, eliminating CPU launch overhead for thousands of kernels per iteration
- **Reduced communication jitter**: Pipeline parallel and data parallel communication kernels are included in the graph, providing deterministic execution timing
- **Higher memory savings with PP**: Eliminates separate input/output buffers between pipeline stages while enabling activation reuse between layers and microbatches
- **Better GPU utilization**: GPUs remain fully saturated throughout the training iteration with deterministic kernel scheduling

**Measured performance improvements** (MLPerf Training v5.1 results on 512 GPUs):

| Configuration | Speedup with Full-Iteration CUDA Graph |
|---------------|---------------------------------------|
| **512×B200** (FP4) | **1.15×** |
| **512×GB200** (FP4) | **1.54×** |
| **512×GB300** (FP4) | **1.79×** |

**Why FP4 benefits but FP8 doesn't**: The speedups above are for FP4 precision. For this Llama 3.1 405B benchmark, FP8 shows no speedup on B200 or GB200 because FP8 GEMMs are large enough that their execution time hides CPU kernel launch overhead - the GPU stays busy while the CPU prepares the next kernel. With FP4, the GEMMs are smaller and faster, completing before the CPU can launch the next kernel, making CPU overhead the bottleneck. CUDA graphs eliminate this CPU overhead, providing significant speedups for FP4 (1.15-1.79×) while FP8 sees no benefit because it wasn't CPU-bound to begin with.

**Memory savings** (Full-iteration vs Per-layer graphing):

Across various LLM models tested with pipeline parallelism, full-iteration CUDA graphs achieve **up to 19% memory savings** compared to per-layer graphing. This reduction primarily comes from eliminating the separate input/output buffers that per-layer graphs require at each graph boundary to interface with eager mode code between layers. Full-iteration graphing captures the entire forward-backward pass end-to-end, removing the need for these intermediate staging buffers.

**Key factors affecting benefits**:

- **Precision level**: Lower precision (FP4 > FP8 > BF16) benefits more because faster, smaller GEMMs expose CPU launch overhead as the bottleneck - CUDA graphs eliminate this overhead
- **Hardware generation**: Newer GPUs (GB300 > GB200 > B200) have faster compute engines, making CPU overhead more dominant and CUDA graph benefits more pronounced
- **Pipeline parallelism configuration**: Higher PP depth with per-layer graphing requires more intermediate staging buffers between pipeline stages - full-iteration graphing eliminates these buffers entirely, providing up to 19% memory savings
- **Microbatch count**: More microbatches per iteration means more kernel launches per iteration, amplifying the overhead reduction from single graph capture

## Key Lessons

The Llama 3.1 405B CUDA graph implementation demonstrates practical deployment of full-iteration graph optimization for large-scale transformer training:

1. **Full-iteration graphing outperforms per-layer approaches**: Provides 1.15-1.79× speedup (higher on newer hardware and lower precision) plus up to 19% memory savings by eliminating intermediate staging buffers between layers

2. **Lower precision training benefits more**: For this benchmark, FP4 achieves 1.15-1.79× speedups while FP8/BF16 shows limited improvement, because faster FP4 GEMMs can't hide CPU launch overhead, which CUDA graphs eliminate

3. **FullCudaGraphWrapper handles complexity automatically**: Megatron-LM's `FullCudaGraphWrapper` captures the entire forward-backward pass including all distributed communication (DP, PP, TP, CP) without manual intervention

4. **Static buffers are automatic**: `StaticBufferLoader` is built into `FullCudaGraphWrapper` and handles input data copying to fixed GPU addresses transparently

5. **Minimal warmup required with Megatron DDP**: Only 2 warmup iterations needed (vs 11+ for PyTorch DDP) before graph capture

6. **GPU-based RNG required**: Use Transformer Engine's RNG tracker (`--te-rng-tracker`) to avoid CPU scalar dependencies in dropout and stochastic operations

7. **Stream management handled automatically**: DDP initialization and graph capture use side streams internally - no manual stream configuration needed

8. **Key environment variables required**: Set `TORCH_NCCL_AVOID_RECORD_STREAMS=1`, `OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False`, and `NCCL_GRAPH_REGISTER=0` (when using expandable segments) for CUDA graph compatibility

9. **Optimizer stays in eager mode**: Keeping optimizer, gradient clipping, and LR scheduling outside the graph provides flexibility while still capturing forward-backward computation

10. **Avoid CPU code inside graphed regions**: Python operations don't execute during replay, so code outside the graph cannot depend on CPU-side state changes that happen inside the graphed region

## References

**MLPerf Training Benchmark**:

- [MLCommons Blog: Training Llama 3.1 405B](https://mlcommons.org/2025/05/training-llama31405b/) - Official MLPerf Training v5.0 benchmark introduction
- [MLPerf Training Reference Implementation](https://github.com/mlcommons/training/blob/95f90e857ade5392f83f54393cdccb55d12ac67f/large_language_model_pretraining/nemo/README.md) - Reference implementation details

**MLPerf Training v5.1 Implementation**: [NVIDIA Llama 3.1 405B PyTorch](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo)

- [custom_callbacks.py](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/custom_callbacks.py) - Warmup infrastructure and MLPerf-specific callbacks
- [conf/custom.yaml](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/conf/custom.yaml) - Model configuration including CUDA graph flags
- [config_common_cg.sh](https://github.com/mlcommons/training_results_v5.1/tree/main/NVIDIA/benchmarks/llama31_405b/implementations/tyche_ngpu512_ngc25.09_nemo/config_common_cg.sh) - Environment variables for CUDA graph enablement

**Framework Source Code**:

- **Megatron-LM** (v25.09-alpha.rc1) - [GitHub Repository](https://github.com/NVIDIA/Megatron-LM/tree/25.09-alpha.rc1)
    - [megatron/core/full_cuda_graph.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/full_cuda_graph.py) - `FullCudaGraphWrapper` implementation with `StaticBufferLoader`
    - [megatron/core/tensor_parallel/random.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/tensor_parallel/random.py) - RNG state management (`get_all_rng_states`, `CudaRNGStatesTracker`)
    - [megatron/core/extensions/transformer_engine.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/extensions/transformer_engine.py) - Transformer Engine RNG tracker (`TECudaRNGStatesTracker`)
    - [megatron/training/training.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/training/training.py) - DDP initialization on side stream
    - [megatron/core/distributed/distributed_data_parallel.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/distributed/distributed_data_parallel.py) - Megatron DDP with parameter gather overlap
    - [megatron/core/optimizer/optimizer.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/optimizer/optimizer.py) - `ChainedOptimizer` with async parameter gather
    - [megatron/core/tensor_parallel/layers.py](https://github.com/NVIDIA/Megatron-LM/blob/25.09-alpha.rc1/megatron/core/tensor_parallel/layers.py) - `ColumnParallelLinear` implementation
- [NeMo Framework](https://github.com/NVIDIA/NeMo) (v25.09-alpha.rc1) - NVIDIA NeMo toolkit
- [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (v2.8) - FP8 training and TE RNG tracker

**Model and Dataset**:

- [Meta AI: Llama 3.1 Release](https://ai.meta.com/blog/meta-llama-3-1/) - Official Llama 3.1 announcement and model details
- [AllenAI C4 Dataset](https://huggingface.co/datasets/allenai/c4) - C4-en dataset information

**Related Documentation**:

- [Transformer Engine and Megatron-LM CUDA Graphs](../torch-cuda-graph/te-megatron-cuda-graphs.md) - Deep dive into `FullCudaGraphWrapper` implementation
- [Writing Sync-Free Code](../torch-cuda-graph/sync-free-code.md) - Eliminating CPU-GPU synchronizations
- [CUDA Graph Constraints](../cuda-graph-basics/constraints.md) - Self-contained execution and other fundamental constraints
- [PyTorch Integration](../torch-cuda-graph/torch-integration.md) - DDP setup, NCCL configuration, and RNG state management
- [Best Practices](../torch-cuda-graph/best-practices.md) - General CUDA graph adoption guidance
  - [Handling Dynamic Patterns](../torch-cuda-graph/handling-dynamic-patterns.md) - Solutions for dynamic shapes and RNG state

## What's Next?

**Explore more examples**:

- [DeepSeek-V3 671B](deepseek-v3-paged-stash.md) - Full-iteration graphing for dropless MoE with device-initiated grouped GEMM and paged stashing
- [Stable Diffusion v2](stable-diffusion-v2.md) - Another full-iteration graphing example with PyTorch Lightning
- [Llama 2 70B LoRA](llama2-70b-lora.md) - LoRA fine-tuning with per-layer CUDA graphs
- [GPT-3 175B](gpt-3-175b.md) - Large-scale training with Megatron-LM
- [RNN Transducer](rnnt.md) - Speech recognition with dynamic shapes and control flow

**Deep dive into implementation**:

- [Transformer Engine and Megatron-LM CUDA Graphs](../torch-cuda-graph/te-megatron-cuda-graphs.md) - `FullCudaGraphWrapper` implementation details and usage examples
- [Writing Sync-Free Code](../torch-cuda-graph/sync-free-code.md) - Eliminating CPU-GPU synchronizations
- [PyTorch Integration](../torch-cuda-graph/torch-integration.md) - DDP, NCCL, and RNG configuration

**Debug and troubleshoot**:

- [Troubleshooting](../troubleshooting/introduction.md) - Common issues and solutions
