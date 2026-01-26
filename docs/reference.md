<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Reference

This page provides links to additional resources for learning about CUDA Graphs and related topics. These materials complement this guide and offer deeper technical details, alternative perspectives, and comprehensive API documentation.

## Official Documentation

### CUDA

- **[CUDA Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)**: The authoritative source on CUDA Graphs, covering all features, constraints, and low-level APIs
- **[CUDA C++ Programming Guide - CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)**: Legacy documentation (no longer updated as of CUDA 13.0), useful for older CUDA versions
- **[CUDA Runtime API - Graph Management](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)**: Complete API reference for CUDA graph functions
- **[Getting Started with CUDA Graphs (NVIDIA Blog)](https://developer.nvidia.com/blog/cuda-graphs/)**: Introductory blog post with examples and use cases

### PyTorch

- **[PyTorch CUDA Semantics - CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)**: PyTorch's official CUDA graph documentation with API details and usage patterns
- **[torch.cuda.CUDAGraph API](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html)**: Low-level CUDAGraph class reference
- **[torch.cuda.graph()](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html)**: Context manager for stream capture
- **[torch.cuda.make_graphed_callables()](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html)**: High-level API for graphing callables
- **[PyTorch Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)**: Guide for deterministic behavior, relevant for graph validation

### Frameworks

- **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**: NVIDIA's framework for training large language models with built-in CUDA graph support
- **[Megatron Core Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)**: Official Megatron Core developer guide
- **[CUDAGraph Trees](https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html)**: PyTorch compiler's automatic CUDA graph system
- **[NCCL User Guide - CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)**: Using NCCL collectives with CUDA graphs

## Technical Blogs and Articles

- **[Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)**: PyTorch blog post on CUDA graph integration
- **[Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/)**: NVIDIA blog post on conditional nodes (IF, WHILE, SWITCH) for dynamic control flow in CUDA graphs
- **[CUDA 10 Features Revealed](https://developer.nvidia.com/blog/cuda-10-features-revealed/)**: Original announcement of CUDA Graphs feature

## Tools and Profiling

- **[NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)**: GPU profiling tool for identifying performance bottlenecks and synchronizations
- **[NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)**: Detailed kernel-level profiling

## Research Papers

- **[Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)**: Paper introducing Megatron's parallelism strategies

## Community Resources

- **[PyTorch Discuss Forums](https://discuss.pytorch.org/)**: Community Q&A, search for "CUDA graphs" for discussions
- **[PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)**: Bug reports and feature requests, searchable for CUDA graph issues
- **[CUDA Zone](https://developer.nvidia.com/cuda-zone)**: NVIDIA's CUDA developer portal with tutorials and resources

## Related Topics

- **[PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)**: General performance optimization techniques
- **[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)**: Comprehensive CUDA optimization guide
- **[Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)**: Automatic mixed precision documentation

These resources provide deeper dives into specific topics covered in this guide and help you become proficient with CUDA Graphs and GPU optimization.
