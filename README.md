# CUDA Graph Best Practice For PyTorch

A comprehensive guide to using CUDA Graphs effectively with PyTorch, covering CUDA fundamentals, PyTorch integration, Megatron-LM implementations, and practical troubleshooting.

## 📚 Documentation

View the documentation at: **https://docs.nvidia.com/dl-cuda-graph/**

## 📖 What's Inside

- **CUDA Graph Basics**: Fundamentals, constraints, and quantitative benefits
- **PyTorch CUDA Graphs**: Integration, Transformer Engine & Megatron-LM, best practices, and handling dynamic patterns
- **Examples**: Real-world implementations from MLPerf Training (Llama 3.1 405B, GPT-3 175B, Stable Diffusion v2, RNN-T, and DeepSeek-V3 671B for dropless MoE)
- **Troubleshooting**: Capture failures, numerical errors, memory issues, and performance debugging

## 🚀 Quick Start

```bash
# Clone and serve locally (auto-installs dependencies)
git clone https://github.com/NVIDIA/dl-cuda-graph-doc.git
cd dl-cuda-graph-doc
./scripts/sphinx-serve.sh
```

Visit `http://127.0.0.1:8000` to view the documentation.

**Manual build:**

```bash
uv sync --group docs
uv run --group docs sphinx-build docs docs/_build/html
```

## 📝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project uses dual licensing:

- **Documentation** (Markdown files): [CC-BY-4.0](LICENSE-DOCS) - Creative Commons Attribution 4.0 International
- **Code** (Python, shell scripts, configuration): [MIT](LICENSE)

See [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for third-party dependencies and their licenses.

## 🔗 Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [Sphinx](https://www.sphinx-doc.org/) with [NVIDIA Sphinx Theme](https://pypi.org/project/nvidia-sphinx-theme/)
