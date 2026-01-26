#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Serve the Sphinx documentation locally with live reload

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies
uv sync --group docs

# Serve with live reload (auto-refreshes on changes)
echo "Starting Sphinx autobuild server..."
echo "Documentation will be available at http://127.0.0.1:8000"
uv run --group docs sphinx-autobuild docs docs/_build/html --open-browser
