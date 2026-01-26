<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Process Hang

:::{note}

This section covers debugging hanging or stuck processes when using CUDA graphs.
:::

## Process Hangs During Teardown

### NCCL Communicator Destruction with Active CUDA Graphs

**Symptom**: Process hangs indefinitely during `destroy_process_group()` or at program exit when using CUDA graphs with NCCL operations.

**What it means**: NCCL communicator destruction is blocked waiting for CUDA graphs to be released first.

**Why it happens**: Starting from NCCL 2.26, the communicator destruction logic [polls until all CUDA graphs referencing the communicator are destroyed](https://github.com/NVIDIA/nccl/blob/v2.28.3-1/src/init.cc#L2157-L2160):

```cpp
// NCCL source: src/init.cc
// And keep polling until all graphs referencing us die.
while (comm->localPersistentRefs != 0) {
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
}
```

This enforces a strict release order: **CUDA graphs must be destroyed before NCCL communicators**. The polling mechanism prevents launching CUDA graphs on an already-destroyed communicator, which would cause undefined behavior.

However, in complex PyTorch workloads, both CUDA graphs and NCCL communicators may be garbage-collected by Python, and there's no guarantee on the destruction order. If the NCCL communicator is destroyed first while a CUDA graph still holds a reference, the destruction blocks forever.

**Example 1**: The following code hangs at `destroy_process_group()`:

```python
# torchrun --nproc_per_node=2 035.nccl-hang.v1.py
import os
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

import torch

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(backend='nccl', device_id=local_rank)
torch.distributed.barrier()

x = torch.randn((32, 128), device="cuda")

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    torch.distributed.all_reduce(x)

graph.replay()

# Won't hang if you add `del graph` or `graph.reset()` here
torch.distributed.destroy_process_group()  # Hangs here!
```

**Example 2**: Global variables and class attributes are destroyed last during Python shutdown, after the NCCL communicator may have been finalized:

```python
# torchrun --nproc_per_node=2 035.nccl-hang.v2.py
import os
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

import torch

class CUDAGraphWrapper:
    graph = None  # Class attribute survives function scope

def test():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl', device_id=local_rank)
    torch.distributed.barrier()

    x = torch.randn((32, 128), device="cuda")

    CUDAGraphWrapper.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(CUDAGraphWrapper.graph):
        torch.distributed.all_reduce(x)

    CUDAGraphWrapper.graph.replay()
    # No explicit destroy_process_group()

if __name__ == "__main__":
    test()
    # Hangs at exit! The class attribute keeps graph alive
    # until after NCCL communicator destruction
```

:::{warning} DDP, FSDP, and Other Distributed Wrappers

The same issue affects any CUDA graph capturing NCCL operations from distributed wrappers like `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP), or tensor parallelism libraries. If your training loop captures all-reduce, all-gather, or reduce-scatter operations into a graph, ensure the graph is released before process group teardown.
:::

**How to fix**:

**Solution 1: Explicitly Release Graphs Before Communicator Destruction**

Ensure CUDA graphs are destroyed before calling `destroy_process_group()` or before program exit:

```python
# Release graph before destroying process group
del graph  # or graph.reset()
torch.distributed.destroy_process_group()  # Now works

# For global/class-level graphs
CUDAGraphWrapper.graph.reset()  # or = None
```

**Solution 2: Use `atexit` Handler for Cleanup**

Use an `atexit` handler to ensure graphs are released before Python shutdown finalizes NCCL:

```python
import atexit

graphs_to_cleanup = []

def cleanup_graphs():
    for g in graphs_to_cleanup:
        g.reset()
    graphs_to_cleanup.clear()

atexit.register(cleanup_graphs)

# Register graphs for cleanup
graph = torch.cuda.CUDAGraph()
graphs_to_cleanup.append(graph)
```

## Debugging Hanging Processes

When a process hangs, use GDB with Python debug symbols to inspect native call stacks:

```bash
# Install debug symbols (Ubuntu/Debian)
sudo apt install python3-dbg gdb

# Attach to hanging process
gdb -p <process_id>

# Inside GDB, get Python stack trace
(gdb) py-bt
```

This is particularly useful for debugging hangs in native CUDA or NCCL code.

## What's Next?

- **[Capture Failures](capture-failures.md)**: When capture fails with errors
- **[Numerical Errors](numerical-errors.md)**: Wrong results or NaN/Inf issues
- **[Debugging Strategies](debugging-strategies.md)**: Systematic debugging approach
- **[Performance Issues](performance-issues.md)**: When graphs don't provide expected speedup
