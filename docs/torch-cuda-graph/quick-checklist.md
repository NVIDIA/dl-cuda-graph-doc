<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Quick Checklist

:::{note}

A complete checklist to verify your PyTorch code is ready for CUDA Graph capture. Check each item before attempting to graph your workload.
:::

## Asynchronous Execution

- [ ] **No host-device synchronization** ([sync-free code](sync-free-code.md), [capture failures](../troubleshooting/capture-failures.md#synchronization-failures))

    - [ ] No explicit sync: `torch.cuda.synchronize()`, `stream.synchronize()`, `event.synchronize()` ([details](sync-free-code.md#explicit-synchronizations))
    - [ ] No blocking GPU→CPU transfers: `.item()`, `.cpu()`, `.numpy()`, `print(tensor)` ([details](sync-free-code.md#tensor-movement-between-devices))
    - [ ] No direct CUDA tensor creation from Python objects ([details](sync-free-code.md#tensor-creation-and-allocation))
    - [ ] No data-dependent control flow: `if tensor:`, `loss.item()` ([details](sync-free-code.md#data-dependent-control-flow))
    - [ ] No GPU tensor indexing with CPU tensors or Python lists ([details](sync-free-code.md#indexing-tensors))
    - [ ] No slicing with CUDA tensor bounds: `x[i:j]` where `i`, `j` are CUDA tensors ([details](sync-free-code.md#slicing-with-tensor-indices))

- [ ] **No default stream usage**

    - [ ] Tensors with gradient tape execute on side stream before capture ([details](../troubleshooting/capture-failures.md#gradient-tape-default-stream-conflict))
    - [ ] DDP/FSDP initialized on side stream ([details](../troubleshooting/capture-failures.md#distributeddataparallel-default-stream-conflict))
    - [ ] Ensure extensions/libraries use PyTorch's current stream, not default stream ([details](../troubleshooting/numerical-errors.md#library-stream-awareness-operations-on-wrong-stream))

- [ ] **No event/stream query**

    - [ ] No `stream.query()`, `event.query()` during capture ([details](../troubleshooting/capture-failures.md#direct-query-forbidden-operations))
    - [ ] No background thread queries ([details](../troubleshooting/capture-failures.md#background-thread-queries))
    - [ ] No pinned memory allocation during capture (triggers hidden event query) ([details](../troubleshooting/capture-failures.md#pinned-memory-allocation-hidden-event-queries))
    - [ ] DataLoader with `pin_memory=True`: use `thread_local` mode or disable pin_memory ([details](../troubleshooting/capture-failures.md#dataloader-pin-memory-thread))
    - [ ] NCCL watchdog handled (auto in PyTorch 2.2+) ([details](../troubleshooting/capture-failures.md#nccl-watchdog-thread))

## Static Graph

- [ ] **Static graph topology** ([details](handling-dynamic-patterns.md#dynamic-control-flow))

    - [ ] No dynamic control flow (`if/else` based on tensor values); use `torch.where()` or capture multiple graphs ([details](../troubleshooting/numerical-errors.md#dynamic-control-flow-frozen-execution-path))
    - [ ] Gradient clipping: use sync-free `clip_grad_norm_` (PyTorch 1.13+) ([details](handling-dynamic-patterns.md#example-1-gradient-clipping))
    - [ ] Early exit and adaptive inference: capture separate graphs per path ([details](handling-dynamic-patterns.md#example-2-early-exit-and-adaptive-inference))
    - [ ] Capture-aware code (`is_current_stream_capturing()`) doesn't change computation ([details](../troubleshooting/numerical-errors.md#capture-aware-code-behavioral-differences))

- [ ] **Static memory addresses** ([details](handling-dynamic-patterns.md#dynamic-tensors))

    - [ ] Static input tensors allocated before capture, updated via `.copy_()` ([details](../troubleshooting/numerical-errors.md#dynamic-input-tensors-stale-memory-addresses))
    - [ ] Global tensors used within graph are persistent ([details](handling-dynamic-patterns.md#example-1-global-statistics-tensor-for-normalization))
    - [ ] Grouped GEMM / pointer arrays: keep host pointer tensors alive ([details](handling-dynamic-patterns.md#example-2-grouped-gemm-in-moe-models))
    - [ ] AMP autocast cache disabled (`cache_enabled=False`) or capture autocast inside graph ([details](../troubleshooting/numerical-errors.md#dynamic-input-tensors-stale-memory-addresses))

- [ ] **Static scalars** ([details](handling-dynamic-patterns.md#dynamic-scalars))

    - [ ] CPU variable scalars converted to GPU tensors, update via `.fill_()` ([details](../troubleshooting/numerical-errors.md#dynamic-scalars-captured-constants))
    - [ ] Learning rate / global step: use capturable optimizer (e.g., APEX FusedAdam) ([details](handling-dynamic-patterns.md#example-2-learning-rate-and-global-step-in-apex-fusedadam))
    - [ ] Handling RNG state correctly

        - [ ] Custom generators registered with `graph.register_generator_state()` ([details](../troubleshooting/capture-failures.md#custom-generator-registration-required))
        - [ ] Use graph-safe APIs: `graphsafe_get_state()`, `graphsafe_set_state()` ([details](../troubleshooting/capture-failures.md#rng-state-operations-use-graph-safe-apis))
        - [ ] Activation checkpointing uses `preserve_rng_state=False` ([details](../troubleshooting/capture-failures.md#activation-checkpointing-with-rng-state-preservation))
        - [ ] Partial graphing uses `use_reentrant=False` ([details](../troubleshooting/capture-failures.md#partial-graphing-with-reentrant-activation-checkpointing))
        - [ ] `torch.compile` functions warmed up before capture ([details](../troubleshooting/capture-failures.md#torchcompile-during-graph-capture))

- [ ] **Static shapes** ([details](handling-dynamic-patterns.md#dynamic-shapes))

    - [ ] Tensor shapes fixed across replays; use padding or bucketing ([details](../troubleshooting/numerical-errors.md#dynamic-shapes-fixed-tensor-dimensions))
    - [ ] MoE with dynamic routing: graph only static parts ([details](handling-dynamic-patterns.md#example-2-moe-models-with-token-dropless))

## Self-Contained Stream Capture

- [ ] Side streams **fork from** capture stream via `side_stream.wait_stream(capture_stream)` ([details](../troubleshooting/capture-failures.md#inlet-stream-missing-entry-synchronization))
- [ ] Side streams **join back to** capture stream via `capture_stream.wait_stream(side_stream)` ([details](../troubleshooting/capture-failures.md#outlet-stream-missing-exit-synchronization))
- [ ] No dependency on external work, or use external events ([details](../troubleshooting/capture-failures.md#isolation-dependency-on-external-work))

## CPU Code Is Not Captured

- [ ] No host state mutation inside graph that affects code outside ([details](../troubleshooting/numerical-errors.md#host-state-mutation-frozen-cpu-state))
- [ ] CPU code requiring execution on every replay moved outside graph ([details](../troubleshooting/numerical-errors.md#cpu-code-eliminated-during-graph-replay))
- [ ] Use `cudaLaunchHostFunc()` for necessary CPU code

## Memory Requirements

- [ ] **No pinned memory alloc/free in global mode** ([details](../troubleshooting/capture-failures.md#pinned-memory-forbidden-in-global-mode))

- [ ] **Persistent graph input tensors**

    - [ ] CUDA input tensors not freed before graph replay ([details](../troubleshooting/numerical-errors.md#deconstructed-cuda-tensors))
    - [ ] CPU input tensors (for H2D copy) kept alive for graph lifetime ([details](../troubleshooting/numerical-errors.md#deconstructed-cpu-tensors))

- [ ] **No cross-iteration reuse of output tensor** without cloning ([details](../troubleshooting/numerical-errors.md#accumulating-graph-outputs))

- [ ] **Memory pool sharing** (if using shared pools):

    - [ ] Intermediate tensors exposed as output handled carefully ([details](../troubleshooting/numerical-errors.md#shared-memory-pool-corruption))
    - [ ] Replay order matches capture order ([details](../troubleshooting/numerical-errors.md#replay-order-mismatch))
    - [ ] No parallel replay of graphs sharing pools ([details](../troubleshooting/numerical-errors.md#parallel-replay-concurrent-pool-access))

- [ ] **Memory usage awareness** (for OOM prevention):

    - [ ] Reuse static input tensors across graphs when possible ([details](../troubleshooting/memory-issues.md#static-input-tensors-cant-be-freed))
    - [ ] Chain graph outputs as inputs to next graph ([details](../troubleshooting/memory-issues.md#static-input-tensors-cant-be-freed))
    - [ ] Be aware: intermediate tensors can't be reused across different pools ([details](../troubleshooting/memory-issues.md#intermediate-tensors-cant-be-reused-across-memory-pools))
    - [ ] Be aware: operations after capture can't reuse graph pool memory ([details](../troubleshooting/memory-issues.md#intermediate-tensors-after-capture-cant-reuse-graph-pool-memory))
    - [ ] Be aware: memory fragmentation across pools ([details](../troubleshooting/memory-issues.md#memory-fragmentation-across-pools))
    - [ ] Be aware: deferred memory recycling with multi-streams during capture ([details](../troubleshooting/memory-issues.md#deferred-memory-recycling))
    - [ ] Be aware: gradient accumulator cross-stream growth ([details](../troubleshooting/memory-issues.md#gradient-accumulator-cross-stream-memory-growth))
    - [ ] Be aware: `cudaFree` is suppressed during capture ([details](../troubleshooting/memory-issues.md#cudafree-is-suppressed-during-capture))

## Other Considerations

- [ ] **Warmup iterations** before capture on the same side stream ([details](torch-integration.md#mandatory-warmup-iterations))
- [ ] **Capture mode**: Use `global` mode unless specific multi-threading needs require `thread_local` or `relaxed` ([details](../cuda-graph-basics/constraints.md#capture-modes))
- [ ] **Module hooks**: Only top-level module hooks fire with `make_graphed_callables` ([details](../troubleshooting/numerical-errors.md#module-hooks-not-firing))
- [ ] **Deferred gradient hooks**: `make_graphed_callables` defers gradient accumulation and DDP hooks ([details](../troubleshooting/performance-issues.md#deferred-gradient-hooks-with-make_graphed_callables))
- [ ] **NCCL communicator lifecycle**: Destroy CUDA graphs before NCCL communicators ([details](../troubleshooting/process-hang.md#nccl-communicator-destruction-with-active-cuda-graphs))
- [ ] **Pinned memory race condition**: Synchronize before CPU writes to pinned memory ([details](../troubleshooting/numerical-errors.md#pinned-memory-race-condition))
- [ ] **Stream count**: Avoid too many streams to prevent channel serialization ([details](../troubleshooting/performance-issues.md#too-many-cuda-streams-channel-serialization))
- [ ] **NCCL buffer registration**: Set `NCCL_GRAPH_REGISTER=0` if using expandable segments with older NCCL ([details](../cuda-graph-basics/constraints.md#multi-gpu-operations))

## What's Next?

- **[Best Practices](best-practices.md)**: Systematic approach to adopting CUDA Graphs
- **[Writing Sync-Free Code](sync-free-code.md)**: Eliminate CPU-GPU synchronizations
- **[Handling Dynamic Patterns](handling-dynamic-patterns.md)**: Solutions for common obstacles
- **[Troubleshooting](../troubleshooting/introduction.md)**: Debug capture failures and issues
