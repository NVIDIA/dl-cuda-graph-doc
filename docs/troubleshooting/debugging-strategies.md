<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Debugging Strategies

:::{note}

This section provides systematic approaches to debug CUDA graph issues in PyTorch applications.
:::

## General Debugging Approach

When encountering CUDA graph problems, follow this systematic strategy:

**1. Make Sure Eager Mode Works**

Always verify your code executes correctly in eager mode before attempting to graph it. If your model fails in eager execution, those issues are unrelated to CUDA graphs and must be fixed first. Only proceed with graph capture once eager mode runs successfully and produces correct results. This baseline validation separates model-level bugs from graph-specific issues, saving significant debugging time.

**2. Isolate the Issue**

Use a combination of binary search and progressive simplification to identify the problematic code:

**Binary search approach** - Narrow down the location:

- **Start broad**: Try graphing your entire model
- **If it fails**: Graph only the first half
- **Still fails?**: Graph only the first quarter
- **Repeat until**: You find the smallest section that fails

**Progressive simplification** - Drop features gradually and reduce complexity, e.g.:

1. Disable pipeline parallelism (PP)
2. Remove dropout: `model.eval()`
3. Remove random operations temporarily
4. Test with smaller inputs
5. Reduce number of layers

Once the simplified version works, add complexity back one piece at a time. This combination quickly identifies both the location and nature of the incompatibility.

**3. Read Error Messages Carefully**

When CUDA graph capture fails, PyTorch CUDA graphs will throw informative error messages. Read these messages carefully to understand what might be wrong—they often point directly to the problematic operation or constraint violation. For more accurate error positions, set `CUDA_LAUNCH_BLOCKING=1` before running your application. This forces synchronous kernel launches, making it easier to pinpoint exactly which operation triggered the error.

For CUDA-level errors, enable CUDA driver logging to get detailed diagnostic information. Set the environment variable `CUDA_LOG_FILE` to capture logs (use `stdout` or `stderr` as special-case names to print directly):

```bash
CUDA_LOG_FILE=stderr python your_script.py
```

This logs CUDA driver-level errors and can help identify what failed at the CUDA level. You can also use the [CUDA Error Log Management APIs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/error-log-management.html) programmatically, including a callback option that lets you set a breakpoint or dump a stack trace when an error occurs. Note that these are CUDA driver APIs (C/C++); for PyTorch debugging, you'll need to write a C++ extension or use [cuda-python bindings](https://nvidia.github.io/cuda-python/cuda-bindings/latest/module/driver.html#cuda.bindings.driver.cuLogsRegisterCallback). This feature is available in CUDA 12.9+.

**4. Search This Documentation for Similar Issues**

This documentation already covers many common issues:

- **[Capture Failures](capture-failures.md)**: Common capture errors and solutions
- **[Numerical Errors](numerical-errors.md)**: Wrong results, NaN/Inf issues
- **[Memory Issues](memory-issues.md)**: OOM errors and memory management
- **[Process Hang](process-hang.md)**: Hanging or stuck processes
- **[Performance Issues](performance-issues.md)**: When graphs don't deliver speedup

Search for your specific error message or symptom—chances are it's already documented.

**5. Check Constraint Violations**

Verify your code doesn't violate CUDA graph constraints at both CUDA and PyTorch levels:

**CUDA-level constraints** (see [CUDA Graph Constraints](../cuda-graph-basics/constraints.md)), e.g.:

- No CPU-GPU synchronization (`.item()`, `.cpu()`, `torch.cuda.synchronize()`)
- No memory allocations during capture
- Static control flow (no data-dependent branches)
- No forbidden operations (CUDA event query, etc.)

**PyTorch-level constraints** (see [PyTorch Integration](../torch-cuda-graph/torch-integration.md)), e.g.:

- Static tensor shapes
- No graph input tensor reassignments (use `.copy_()` instead)
- DDP requires 11+ warmup iterations
- AMP requires `cache_enabled=False`
- RNG generators must be registered

**6. Dump and Inspect CUDA Graph Structure**

If the graph is not too large, dump it to verify the structure matches your expectations:

```python
# Enable debug mode and dump graph
g = torch.cuda.CUDAGraph()
g.enable_debug_mode()

with torch.cuda.graph(g):
    output = model(static_input)

# Dump graph for inspection
g.debug_dump("/tmp/graph_debug")
```

This generates a visualization showing captured operations, dependencies, and memory addresses. Check if:

- All expected operations are captured
- Dependencies are correct
- Memory addresses are as expected

**7. Compare with Eager Mode for Numerical Errors**

For numerical errors (wrong results without explicit failures), use a systematic comparison approach:

- **Enable determinism**: Set `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.deterministic = True`, and use a fixed random seed (e.g., `torch.manual_seed(42)`) to eliminate non-determinism as a variable.

- **Inspect key tensors**: Key tensors include parameters, gradients, and activations. Use a global list or dictionary to save graph-internal tensors for later inspection.

- **Compare with eager mode**: Run the same input through both eager mode and graph replay. Compare tensor statistics (max, min, norm, mean) and values to localize where divergence starts:

    ```python
    # Eager execution
    model.eval()
    eager_out = model(test_input.clone())

    # Graph execution
    static_input.copy_(test_input)
    graph.replay()
    torch.cuda.synchronize()

    # Compare
    if torch.allclose(eager_out, graph_out, rtol=1e-4, atol=1e-5):
        print("✓ Results match")
    else:
        diff = (eager_out - graph_out).abs()
        print(f"✗ Max diff: {diff.max():.2e}")
    ```

- **Bisect to find the issue**: If outputs differ, progressively reduce the graphed region until you find the operation causing the divergence. Common culprits include dynamic tensors, dynamic scalars, or operations after CUDA graph that depend on CPU state inside CUDA graph.

## Debugging Checklist

When debugging CUDA graph issues, systematically check:

- [ ] Does it work in eager mode?
- [ ] Did you move necessary CPU code outside the graph region (e.g., tokenizer)?
- [ ] Is there any CPU-GPU sync in the captured region (`.item()`, `.cpu()`)?
- [ ] Is control flow deterministic (no data-dependent branches)?
- [ ] Are all shapes truly static?
- [ ] Did you perform adequate warmup (3+ iterations)?
- [ ] Are you using `.copy_()` for input updates (not reassignment)?
- [ ] For DDP: Did you warmup 11+ iterations?
- [ ] For AMP: Is `cache_enabled=False`?
- [ ] Did you register custom random generators?

## When to Ask for Help

If you've tried the above and still stuck:

1. **Check existing issues**: Search PyTorch GitHub issues for similar problems
2. **Minimal reproducer**: Create the smallest code that reproduces the issue
3. **Gather information**: CUDA version, PyTorch version, GPU model, error messages
4. **Ask on forums**: PyTorch Discuss or GitHub with your minimal reproducer

Provide all relevant details—it helps others help you faster.

## What's Next?

- **[Capture Failures](capture-failures.md)**: Common capture errors and solutions
- **[Numerical Errors](numerical-errors.md)**: Debugging incorrect results, NaN/Inf, and precision problems
- **[Memory Issues](memory-issues.md)**: OOM and memory-related issues
- **[Process Hang](process-hang.md)**: Resolving hanging or stuck processes
- **[Performance Issues](performance-issues.md)**: When graphs don't speed things up
