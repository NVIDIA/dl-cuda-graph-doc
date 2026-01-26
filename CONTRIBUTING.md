<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: CC-BY-4.0
-->

# Contributing

Thank you for your interest in improving this CUDA Graph documentation! Your real-world experience with PyTorch CUDA Graphs helps make this guide more comprehensive and valuable for the entire community.

## Quick Start

### Reporting Issues

Found an error or have a suggestion? [Open an issue](https://github.com/NVIDIA/dl-cuda-graph-doc/issues) with:

- Check if the issue already exists in the issue tracker
- Clear description of the problem
- Steps to reproduce (if applicable)
- Suggested fix or improvement (optional)

### Making Changes

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Make your changes and test locally
4. Submit a pull request

## Development Setup

### Project Structure

```
dl-cuda-graph-doc/
├── docs/                          # Documentation source
│   ├── conf.py                    # Sphinx configuration
│   ├── index.md                   # Homepage with toctree
│   ├── cuda-graph-basics/         # CUDA Graph fundamentals
│   ├── torch-cuda-graph/          # PyTorch integration
│   ├── examples/                  # Real-world examples
│   ├── troubleshooting/           # Debugging guides
│   ├── assets/                    # Images and static files
│   ├── versions.json              # Version switcher configuration
│   └── project.json               # Project metadata
├── pyproject.toml                 # Python dependencies
└── README.md
```

### Local Development

```bash
# Clone and serve locally (auto-installs dependencies)
git clone https://github.com/NVIDIA/dl-cuda-graph-doc.git
cd dl-cuda-graph-doc
./scripts/sphinx-serve.sh
```

View at `http://127.0.0.1:8000`

**Manual commands:**

```bash
uv sync --group docs                                      # Install dependencies
uv run --group docs sphinx-build docs docs/_build/html    # Build
uv run --group docs sphinx-autobuild docs docs/_build/html --open-browser  # Serve
```

## Writing Documentation

### Adding a New Page

1. Create markdown file: `docs/section/page.md`
2. Add to the toctree in `docs/index.md`:
   ````markdown
   ```{toctree}
   :caption: Your Section
   :hidden:

   section/page
   ```
   ````
3. Write content using MyST Markdown
4. Preview with `./scripts/sphinx-serve.sh`

### Content Guidelines

- **Be clear and concise**: Avoid jargon when possible
- **Use examples**: Show, don't just tell
- **Be practical**: Focus on real-world use cases
- **Stay current**: Use latest PyTorch APIs

### Markdown Reference

| Feature | Syntax |
|---------|--------|
| Admonitions | `:::{note}`, `:::{tip}`, `:::{warning}` |
| Code blocks | ` ```python ... ``` ` |
| Tables | `\| Header \| ... \|` |
| Links | `[text](page.md)` or `[text](https://...)` |
| Task lists | `- [ ] item` or `- [x] item` |
| Mermaid | ` ```mermaid ... ``` ` |
| Grid cards | `::::{grid}` with `:::{grid-item-card}` |

**Admonitions example:**

```text
:::{note} Title
Content here
:::

:::{tip}
Helpful tip
:::

:::{warning}
Important warning
:::
```

**Grid cards example:**

```text
::::{grid} 2
:::{grid-item-card} Title
Content
:::
::::
```

## Pull Request Guidelines

1. **Create descriptive title**: e.g., "Add example for multi-GPU usage"
2. **Write clear description**: Explain what and why
3. **Link related issues**: Reference issue numbers
4. **Request review**: Tag relevant reviewers
5. **Address feedback**: Respond to review comments

## For Maintainers

### Releasing a New Version

1. **Update `docs/versions.json`**:
   ```json
   [
       {"preferred": true, "version": "v1.0", "url": "../v1.0"},
       {"version": "latest", "url": "../latest"}
   ]
   ```

2. **Update `docs/conf.py`**:
   ```python
   release = "v1.0"
   ```

3. **Build and deploy** to a versioned directory

## Community

- **Be respectful and inclusive**
- **Welcome newcomers**
- **Focus on constructive feedback**

Questions? [Open an issue](https://github.com/NVIDIA/dl-cuda-graph-doc/issues) with the "question" label.
