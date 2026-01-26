<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# RNN-T (RNN Transducer)

:::{note}

CUDA Graph optimization for RNN-T speech recognition models, based on NVIDIA's MLPerf Training implementation for RNN-T (v1.0, 2021/06 - v3.1, 2023/11). This example demonstrates handling dynamic shapes (variable sequence lengths) via bucketing and dynamic control flow (greedy decoding during evaluation) via mask-based transformations.
:::

## Overview

RNN-T (Recurrent Neural Network Transducer) is a sequence-to-sequence neural network architecture designed primarily for streaming, end-to-end automatic speech recognition (ASR). Proposed by Alex Graves in 2012 ([Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)), RNN-T can transform input sequences of any length into output sequences of any finite length without requiring pre-aligned input-output pairs.

RNN-T consists of three major building blocks:

1. **Encoder network** (also called transcription network): Encodes acoustic input with RNNs, processing the raw audio features
2. **Prediction network**: Predicts text based on previously emitted text, similar to a language model
3. **Joint network**: Combines the encoded acoustic information and the generated text to produce output probabilities

![RNN-T Architecture](https://3.bp.blogspot.com/-Z_W8MILyUsk/XIauDigQ93I/AAAAAAAAD5E/0LsZWC3mWaIDrzSN0I4QCGGBWBzg5XNYgCEwYBhgL/s400/image2.png)

*Figure: RNN-T architecture showing the encoder, prediction, and joint networks (Image source: [Google AI Blog](https://ai.googleblog.com/2019/03/an-all-neural-on-device-speech.html))*

Unlike traditional CTC (Connectionist Temporal Classification), RNN-T models dependencies between output labels, making it more powerful for speech recognition tasks.

## MLPerf Training RNN-T

The NVIDIA MLPerf Training RNN-T implementation has the following characteristics:

**Model Architecture**:

- **Model size**: ~49M parameters (49,071,616 total)
- **Encoder**: 5 LSTM layers (2 pre-RNN + 3 post-RNN) with 1024 hidden units each
- **Prediction network**: 2 LSTM layers with 512 hidden units
- **Joint network**: 512 hidden units with ReLU, dropout, and final classification to 1024 classes

**Training Setup**:

- **Software stack**: Pure PyTorch implementation (no NeMo), using NVIDIA DALI for data loading
- **Fixed batch size**: Training uses a constant batch size (e.g., 192 samples per GPU on H100)
- **Variable sequence lengths**: Audio sequences and text sequences vary in length across batches
- **Distributed training**: Multi-GPU/multi-node training using `torch.distributed` for communication but **WITHOUT PyTorch DDP**. Instead of wrapping the model with `DistributedDataParallel`, gradient synchronization is handled entirely by the APEX `DistributedFusedLAMB` optimizer, which performs all-reduce operations internally. The model remains unwrapped throughout training.
- **Mixed precision**: FP16 training with gradient scaling

**Dataset**:

- **Training data**: LibriSpeech corpus (960 hours of English speech, 281,241 samples per epoch)
- **Audio features**: 80-dimensional mel-filterbank with adaptive SpecAugment augmentation
- **Text encoding**: SentencePiece tokenization (1023 subword vocabulary + 1 blank token)

The main challenge for CUDA graph adoption is handling variable-length audio and text sequences while maintaining the performance benefits of graph capture.

## Integration Approach for CUDA Graph

The RNN-T implementation uses CUDA graphs to accelerate the forward and backward passes of the encoder and prediction networks. This section describes the **integration infrastructure** - the capture mechanisms, code organization, and tooling created to enable CUDA graphs, though not strictly required for correctness.

### Capture Scope (Partial Graphing)

RNN-T applies partial graphing separately for training and evaluation, graphing only the expensive recurrent components while keeping dynamic operations in eager mode. For training, the implementation captures separate CUDA graphs for forward and backward passes.

**Training** (Partial):

- **Graphed components**: Encoder network (forward + backward) and prediction network (forward + backward), captured as separate forward and backward graphs
- **Non-graphed components**: Joint network, loss computation, gradient scaling (`GradScaler`), optimizer step, and EMA updates
- **Rationale**: The encoder and prediction networks are expensive recurrent LSTM operations, while the joint network and loss require dynamic metadata that varies per batch (e.g., `batch_offset`, `g_len`, `packed_batch` for packing/unpacking sequences based on actual lengths)

**Evaluation** (Partial, enabled by default with `BATCH_EVAL_MODE=cg_unroll_pipeline`):

- **Graphed components**: The greedy decoding main loop (prediction network + joint network inference for iterative token generation)
- **Non-graphed components**: Encoder network (runs once per batch before graphed decoding loop)
- **Three modes available**: `no_cg` (eager), `cg` (graphed), and `cg_unroll_pipeline` (graphed with loop unrolling for better pipelining)

**Scale**: CUDA graphs were successfully deployed across a wide range of training scales:

- **Single-node**: 1 node × 8 GPUs = 8 GPUs
- **Multi-node DGX A100**: 16–192 nodes (128–1,536 GPUs), with the largest configuration at 192 nodes × 8 GPUs = 1,536 GPUs
- **Multi-node DGX H100**: 16–96 nodes (128–768 GPUs), with the largest configuration at 96 nodes × 8 GPUs = 768 GPUs

### Custom Graph Capture Function

RNN-T uses a custom graph capture function to enable fine-grained control over capturing encoder and prediction networks separately with different bucketing strategies, which wasn't possible with PyTorch's built-in APIs at the time.

**Purpose**: Provide a low-level graph capture utility that handles both forward and backward passes, compatible with autograd.

**Implementation** ([rnnt/function.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/function.py)): The `graph()` function wraps a module or function with a custom `torch.autograd.Function` that replays captured graphs:

**Key features**:

- Captures forward pass and backward gradient computation as separate graphs
- Handles module parameters and input tensors uniformly
- Supports training mode (forward + backward) and eval mode (forward only)
- Uses graph memory pools to share memory between forward/backward graphs

```python
def graph(func_or_module, sample_args, graph_stream=None,
          warmup_iters=2, warmup_only=False):
    # Warmup to prime memory allocator
    for _ in range(warmup_iters):
        outputs = func_or_module(*sample_args)
        # ... backward pass ...

    # Capture forward graph
    fwd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(fwd_graph):
        outputs = func_or_module(*sample_args)

    # Capture backward graph (shares memory pool)
    bwd_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(bwd_graph, pool=fwd_graph.pool()):
        grad_inputs = torch.autograd.grad(...)

    # Return wrapped module that replays graphs
    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):
            # Copy inputs to captured buffers
            for i, arg in zip(buffer_inputs, inputs):
                if i.data_ptr() != arg.data_ptr():
                    i.copy_(arg)
            fwd_graph.replay()
            return buffer_outputs

        @staticmethod
        def backward(ctx, *grads):
            # Copy incoming gradients
            for g, grad in zip(buffer_incoming_grads, grads):
                if g is not None:
                    g.copy_(grad)
            bwd_graph.replay()
            return buffer_grad_inputs
```

This overall approach (custom capture function with separate forward/backward graphs and autograd integration) is an early prototype of what later became PyTorch's `make_graphed_callables()` API.

:::{tip} Use PyTorch's Native API

This RNN-T implementation predates PyTorch's built-in CUDA graph support. For new projects, use PyTorch's native [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html) API instead of implementing custom capture functions. The native API is better maintained, more feature-complete, and handles edge cases automatically.
:::

### Wrapper Classes for Independent Graphing

To facilitate separate graphing of encoder and prediction networks with different bucketing strategies, the implementation introduces wrapper classes that isolate each component. While not strictly required for correctness, this design simplifies the capture logic and enables cleaner separation of concerns.

**Motivation**: The main `RNNT` model's `forward()` method calls both `encode()` and `predict()` together, then `joint()`. To graph encoder and prediction separately with independent bucketing, wrapper classes provide clean isolation of each network.

**Implementation** ([rnnt/model.py#L416-L465](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/model.py#L416-L465)): The wrapper classes encapsulate encoder+joint_enc and prediction+joint_pred, used **only** during CUDA graph capture:

```python
class RNNTEncode(nn.Module):
    def __init__(self, encoder, joint_enc, min_lstm_bs):
        self.encoder = encoder
        self.joint_enc = joint_enc

    def forward(self, x, x_lens):
        # Encode audio features
        x, _ = self.encoder["pre_rnn"](x, None)
        x, x_lens = self.encoder["stack_time"](x, x_lens)
        x, _ = self.encoder["post_rnn"](x, None)
        f = self.joint_enc(x.transpose(0, 1))
        return f, x_lens

class RNNTPredict(nn.Module):
    def __init__(self, prediction, joint_pred, min_lstm_bs):
        self.prediction = prediction
        self.joint_pred = joint_pred

    def forward(self, y):
        # Embed and predict text
        y = self.prediction["embed"](y)
        y = torch.nn.functional.pad(y, (0, 0, 1, 0))  # Prepend blank
        g, _ = self.prediction["dec_rnn"](y.transpose(0, 1), None)
        g = self.joint_pred(g.transpose(0, 1))
        return g
```

**Benefits**: This separation enables each network to use graphs optimized for their respective sequence length distributions - audio sequences are typically much longer than text sequences (e.g., 1600 frames vs 600 tokens), so they benefit from different bucketing granularities.

## Essential Modifications for CUDA Graph Compatibility

This section details the **essential modifications** required to make CUDA graphs work correctly with RNN-T. Without these changes, graph capture would fail or produce incorrect results due to dynamic behavior, control flow, or CPU-GPU synchronization issues.

### 1. Sync-Free Execution

A critical requirement for CUDA graphs is avoiding CPU-GPU synchronization within the captured region, as synchronization operations cannot be recorded in graphs. The RNN-T implementation carefully ensures that the graphed encoder and prediction networks ([train.py#L290](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/train.py#L290)) execute without any CPU-GPU synchronization points. Operations that require synchronization - such as loss computation, NaN checking, and metric logging - are kept outside the graphed region. The joint network, which remains in eager mode, also avoids introducing synchronization by using purely GPU-side operations.

### 2. Warmup Iterations on Side Stream

**Problem**: If memory allocations occur during graph capture, the `cudaMalloc` operations get recorded in the graph, causing failures or incorrect behavior during replay. This was particularly critical before CUDA introduced graph capture "relaxed mode" (CUDA 11.6+), which can handle some dynamic allocations.

Additionally, if warmup/initialization is performed on the default stream instead of the capture stream, operations may create dependencies on the default stream. When graph capture later occurs on a side stream, these cross-stream dependencies violate CUDA Graph's self-contained stream capture requirement, causing capture to fail.

**Solution** ([function.py#L68-L95](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/function.py#L68-L95)): Run warmup iterations (default 2) on a dedicated side stream before capture to populate the memory pool that will be used during capture:

```python
# Create dedicated stream for capture
stream = torch.cuda.Stream() if graph_stream is None else graph_stream
ambient_stream = torch.cuda.current_stream()
stream.wait_stream(ambient_stream)

with torch.cuda.stream(stream):
    # Warmup iters before capture - on the SAME side stream
    for _ in range(warmup_iters):
        # Warmup iters should warm up the same memory pool capture will use.
        # If they don't, and we use the capture pool for the first time during
        # capture, we'll almost certainly capture some cudaMallocs.
        outputs = func_or_module(*sample_args)
        # ... run backward pass ...
```

**Why this matters**: Warmup must happen on the **same side stream** that will be used for capture. This ensures memory allocations populate the correct stream-ordered memory pool, preventing `cudaMalloc` operations from being captured in the graph.

### 3. Bucketing for Dynamic Shape

For handling variable-length sequences, the implementation uses **bucketing with padding**: capturing multiple CUDA graphs for different sequence length buckets and selecting the appropriate graph at runtime. This avoids the prohibitive compute waste of padding all sequences to the absolute maximum length.

**How it works**: The `RNNTGraph` class ([rnnt/rnnt_graph.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/rnnt_graph.py)) manages multiple captured graphs, each optimized for a specific maximum sequence length. For example, with `num_cg=4` and max lengths of 1600 audio frames and 600 text tokens:

- **Encoder graphs**: Captures 4 graphs for 1600, 1200, 800, and 400 frames
- **Prediction graphs**: Captures 4 graphs for 600, 450, 300, and 150 tokens
- **Runtime selection**: Uses hash tables for O(1) lookup to select the smallest graph that fits the current sequence, then pads to that size

**Why this matters**: A 950-frame sequence uses the 1200-frame graph (padding only 21%), instead of the 1600-frame graph (padding 41%). This dramatically reduces wasted computation.

**Implementation** ([rnnt_graph.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/rnnt_graph.py)): The `RNNTGraph` class implements bucketing in two phases:

**1. Graph capture** ([lines 56-84](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/rnnt_graph.py#L56-L84)) - Captures `num_cg` graphs with decreasing max lengths and builds hash tables for fast lookup:

```python
def capture_graph(self):
    for i in range(self.num_cg):
        feat_len = self.max_feat_len - (i * self.max_feat_len // self.num_cg)
        encode_graph = self._gen_encode_graph(feat_len)

        txt_len = self.max_txt_len - (i * self.max_txt_len // self.num_cg)
        predict_graph = self._gen_predict_graph(txt_len)

    # Build hash tables for O(1) lookup: length -> (max_len, graph)
    self.dict_encode_graph = {...}
    self.dict_predict_graph = {...}
```

**2. Runtime replay** ([lines 94-101](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/rnnt_graph.py#L94-L101)) - Selects smallest suitable graphs via hash table lookup, pads inputs, and replays:

```python
def step(self, feats, feat_lens, txt, txt_lens, dict_meta_data):
    # O(1) lookup to find smallest graph that fits
    max_feat_len, encode_block = self.dict_encode_graph[feats.size(0)]
    max_txt_len, predict_block = self.dict_predict_graph[txt.size(1)]

    # Pad to graph size and replay
    feats = F.pad(feats, (0, 0, 0, 0, 0, max_feat_len - feats.size(0)))
    txt = F.pad(txt, (0, max_txt_len - txt.size(1)))

    return self._model_segment(encode_block, predict_block,
                                feats, feat_lens, txt, txt_lens)
```

For more details on bucketing techniques, see [Handling Dynamic Patterns - Dynamic Shapes](../torch-cuda-graph/handling-dynamic-patterns.md#example-1-variable-sequence-lengths-nlp).

### 4. CUDA Graph for Evaluation

Beyond training, RNN-T provides optional CUDA graph support for evaluation to accelerate greedy decoding. Evaluation graphs are significantly more complex than training graphs due to fundamental algorithmic differences.

**Why evaluation is different from training:**

**Training uses teacher forcing**—a simple, static forward pass:

- **Input**: Audio features + ground truth text
- **Execution**: Single pass through encoder → prediction → joint → loss → backward
- **Characteristic**: Fixed computation graph, each network runs exactly once per batch

**Evaluation uses greedy decoding**—an iterative algorithm with dynamic control flow:

-   **Input**: Audio features only (no ground truth)
-   **Execution**: Encoder runs once, then a decoding loop repeatedly calls prediction + joint networks until all sequences complete
-   **Decoding logic**: At each step, predict next token (argmax), then based on the result:

    -   If **blank**: advance to the next audio frame
    -   If **non-blank**: emit token and update hidden state

-   **Characteristic**: Dynamic loop executing 10-100+ iterations per utterance with iteration counts unknown until runtime

**The challenge: Dynamic control flow**

The eager implementation ([decoder.py#L292-L332](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L292-L332)) has two types of dynamic behavior incompatible with CUDA graphs:

1. **Variable iteration count**: The outer loop continues `while not_blank` and checks completion conditions at runtime—the number of iterations depends on decoding results
2. **Conditional branches**: Inner loop contains `if k == blank_idx` conditionals that determine behavior based on predicted tokens

**The solution: Graph one step, loop on CPU**

Instead of graphing the entire greedy decoding, the implementation graphs only **one decoding step** while keeping dynamic loop control on the CPU. Specifically:

**What gets graphed** ([decoder.py#L215-L267](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L215-L267)): One iteration of the decoding loop body, including:

- Prediction network forward pass (LSTM)
- Joint network combining encoder + prediction outputs
- Argmax selection and blank/non-blank logic
- State updates (hidden states, time indices, label buffers, completion masks)
- Completion check (`batch_complete` flag)

**What stays in eager mode**: The encoder runs once before the loop—it's not the bottleneck since the prediction loop executes 10-100+ times per utterance.

**Eliminating conditional branches**: Within the graphed step, `if/else` statements are transformed to mask-based operations:

- `non_blank_mask = (k != self.blank_idx)` instead of `if` branch
- State updates use masked tensors: `current_label = current_label * ~non_blank_mask + k * non_blank_mask`
- Time advancement via masks: `advance_mask = (~non_blank_mask | exceed_mask) & ~complete_mask`

**CPU-side loop control** ([decoder.py#L519-L533](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L519-L533)): A `while True` loop on CPU repeatedly replays the captured graph until `batch_complete` signals all sequences finished, checking completion asynchronously to avoid blocking GPU execution.

**Evaluation modes**

The implementation provides three modes via `--batch_eval_mode` ([train.py#L85-L86](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/train.py#L85-L86)):

- **`no_cg`**: Eager execution without CUDA graphs
- **`cg`**: Graphs one decoding step with GPU-side completion check
- **`cg_unroll_pipeline`** (default): Graphs 4 unrolled steps + async CPU-GPU pipeline for better performance

**Implementation techniques**

The implementation uses four key techniques to maximize performance:

**1. Graph capture** ([decoder.py#L428-L453](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L428-L453)): Preallocates all state tensors and captures the loop body using `graph_simple()` (inference-only variant without backward):

```python
def _capture_cg(self, x, out_len):
    # Preallocate state tensors with fixed batch size and max output length
    hidden = [torch.zeros((pred_rnn_layers, batch, pred_n_hid), dtype=x.dtype, device='cuda')]*2
    label_tensor = torch.zeros(batch, max_symbol_per_sample, dtype=torch.int, device='cuda')
    time_idx = torch.zeros(batch, dtype=torch.int64, device='cuda')
    complete_mask = torch.zeros(batch, dtype=torch.bool, device='cuda')

    # Capture the decoding loop iteration (or unrolled iterations)
    self.main_loop_cg = self._capture_cg_for_main_loop([label_tensor, hidden[0], ...])
```

**2. Loop unrolling** ([decoder.py#L269-L274](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L269-L274)): Captures 4 iterations (configurable via `--cg_unroll_factor`) in a single graph to reduce replay overhead:

```python
def _eval_main_loop_unroll(self, ...):
    for u in range(self.cg_unroll_factor):  # Unroll 4 iterations
        label_tensor, hidden, ... = self._eval_main_loop_stream(...)
    return ...
```

**3. Multi-stream state updates** ([decoder.py#L235-L261](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L235-L261)): Uses separate streams for concurrent state updates to maximize GPU utilization:

```python
# Within the graphed loop body
self.label_upd_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(self.label_upd_stream):
    current_label = current_label * ~non_blank_mask + k * non_blank_mask
    label_tensor[...] = label_tensor[...] * complete_mask + current_label * ~complete_mask

self.hidden_upd_stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(self.hidden_upd_stream):
    for i in range(2):
        hidden[i] = hidden[i] * ~expand_mask + hidden_prime[i] * expand_mask

# Synchronize before proceeding
torch.cuda.current_stream().wait_stream(self.label_upd_stream)
torch.cuda.current_stream().wait_stream(self.hidden_upd_stream)
```

**4. Async CPU-GPU pipeline** ([decoder.py#L492-L539](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/decoder.py#L492-L539)): Outer CPU loop replays graph with async completion check:

```python
def _greedy_decode_batch_replay_pipelined(self, x, out_len):
    # Copy encoder outputs to stashed tensors
    self.stashed_tensor[0][:x.size(0), :x.size(1)] = x

    # Initialize state tensors
    hidden = [torch.zeros((2, batch, pred_n_hid), ...)]*2
    complete_mask = time_idx >= out_len

    # Async CPU-GPU pipeline for completion check
    batch_complete_cpu = torch.tensor(False, device='cpu').pin_memory()
    copy_stream = torch.cuda.Stream()

    while True:
        # Replay graph (processes 4 unrolled iterations)
        ..., batch_complete = self.main_loop_cg(...)

        # Check completion on CPU from previous iteration
        copy_stream.synchronize()
        if torch.any(batch_complete_cpu):
            break

        # Async copy for next iteration check
        with torch.cuda.stream(copy_stream):
            batch_complete_cpu.copy_(batch_complete, non_blocking=True)
```

This design handles the key challenges of evaluation graphs: dynamic iteration counts, variable-length outputs per sequence, concurrent state updates, and CPU-side loop exit decisions without blocking GPU execution.

### 5. Other Constraints

Beyond the essential modifications described above, the implementation enforces additional constraints when CUDA graphs are enabled. These are primarily related to optimizer choice, batch handling, and fixed input sizes. The implementation validates these requirements at runtime ([train.py#L576-L580](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/train.py#L576-L580)) and raises errors if violated:

- **DistributedFusedLAMB optimizer required** ([train.py#L475](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/train.py#L475)): The implementation requires APEX's `DistributedFusedLAMB` optimizer for gradient synchronization (see "MLPerf Training RNN-T" section above for distributed training architecture). The optimizer step itself is NOT graphed - only the encoder and prediction networks are captured.
- **No batch splitting**: `batch_split_factor` must be 1 because the joint network (which remains in eager mode) must process the full batch.
- **Fixed batch size**: Batch size must be known at graph capture time for buffer allocation.

## Performance Impact

CUDA graphs provide significant performance improvements for both training and evaluation:

**Training**:

- **Reduced kernel launch overhead**: Batch submission of encoder/prediction network kernels instead of individual launches
- **Better GPU utilization**: Eliminates CPU-GPU synchronization gaps between operations
- **Multi-node scaling**: Launch overhead reduction compounds across distributed training nodes

**Evaluation**:

- **Faster greedy decoding**: Graph replay of the decoding step eliminates per-iteration kernel launch overhead
- **Improved throughput**: Async CPU-GPU pipeline keeps both processing units busy
- **Reduced latency**: Multi-stream execution enables concurrent state updates

The benefits are most pronounced in scenarios with:

- Smaller batch sizes (where launch overhead is more significant relative to compute)
- Multi-GPU/multi-node configurations (overhead reduction scales with number of GPUs)
- Long-running workloads (one-time graph capture cost amortized over many iterations)

## Key Lessons

This RNN-T implementation demonstrates a comprehensive set of CUDA graph optimizations for sequence-to-sequence models. The most important lessons:

1. **Sync-free execution is critical**: Eliminate all CPU-GPU synchronization within the graphed region. Operations requiring synchronization (loss computation, NaN checking, metric logging) must stay outside the graph. This is the foundational requirement for CUDA graph correctness.

2. **Warmup on the same stream as capture**: Run warmup iterations on the same side stream before capture to populate the stream-ordered memory pool. This prevents `cudaMalloc` operations from being captured and avoids cross-stream dependencies that cause capture failures (critical before CUDA 11.6+ relaxed mode).

3. **Bucketing for variable-length sequences**: Capture multiple graphs for different sequence length buckets rather than padding all sequences to maximum length. Without bucketing, padding wastes 50-75% of computation. Bucketing reduces waste to 15-30% while maintaining graph benefits (see [Dynamic Shapes - Bucketing](../torch-cuda-graph/handling-dynamic-patterns.md#example-1-variable-sequence-lengths-nlp)).

4. **Handle dynamic control flow via mask-based transformations**: Evaluation's greedy decoding has variable iteration counts and conditional branches (`if k == blank_idx`) incompatible with CUDA graphs. Solution: graph only one decoding step with mask-based operations (`non_blank_mask = (k != blank_idx)`), while the CPU-side loop handles iteration control. Unrolling multiple steps and async CPU-GPU pipelines maximize performance.

5. **Custom capture for fine-grained control**: RNN-T uses a custom graph capture function (early prototype of PyTorch's `make_graphed_callables()`) to enable separate bucketing for encoder and prediction networks. For new projects, use PyTorch's native [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html).

6. **Trade off capture range, performance, and effort**: Only the encoder and prediction networks are graphed for training; the joint network, loss, and optimizer remain in eager mode. This demonstrates how to balance performance gains against implementation complexity—you don't need to graph everything to see benefits.

7. **Feature flags enable incremental adoption**: The `--num_cg` flag allows easy comparison between eager and graphed execution, critical for validation and debugging.

## References

-   **Original Paper**: [Sequence Transduction with Recurrent Neural Networks (Alex Graves, 2012)](https://arxiv.org/pdf/1211.3711)
-   **Source Code**: [NVIDIA MLPerf Training v3.1 RNN-T Implementation](https://github.com/mlcommons/training_results_v3.1/tree/main/NVIDIA/benchmarks/rnnt/implementations/pytorch)

    -   [rnnt_graph.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/rnnt_graph.py) - CUDA graph bucketing implementation
    -   [function.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/function.py) - Custom graph capture utility
    -   [model.py](https://github.com/mlcommons/training_results_v3.1/blob/main/NVIDIA/benchmarks/rnnt/implementations/pytorch/rnnt/model.py) - RNNTEncode and RNNTPredict wrappers

-   **Related Documentation**:

    -   [Handling Dynamic Patterns](../torch-cuda-graph/handling-dynamic-patterns.md) - Solutions for dynamic shapes and scalars
    -   [Best Practices](../torch-cuda-graph/best-practices.md) - General CUDA graph adoption guidance

## What's Next?

Explore more examples:

- **[Stable Diffusion v2](stable-diffusion-v2.md)**: Optimizing diffusion models for image generation
- **[Llama 3.1 405B](llama-31-405b.md)**: Large-scale transformer training and inference

Continue learning:

- **[Troubleshooting](../troubleshooting/introduction.md)**: Debug common issues and failures
- **[PyTorch Integration](../torch-cuda-graph/torch-integration.md)**: Deep dive into PyTorch's CUDA Graph APIs
- **[Reference](../reference.md)**: Additional resources and documentation
