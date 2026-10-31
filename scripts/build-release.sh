#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Build the release HTML and stage the upload tree:
#   docs/_build/release/dl-cuda-graph/versions.json   -> /dl-cuda-graph/versions.json
#   docs/_build/release/dl-cuda-graph/latest/         -> /dl-cuda-graph/latest/
# and pack it into docs/_build/release/dl-cuda-graph.tar.gz.

set -euo pipefail

cd "$(dirname "$0")/.."

BUILD=docs/_build
STAGE="$BUILD/release/dl-cuda-graph"

rm -rf "$BUILD"
uv sync --group docs
# Keep doctrees out of the HTML output directory
uv run --group docs sphinx-build -W -b html -d "$BUILD/doctrees" docs "$BUILD/html"

mkdir -p "$STAGE"
cp -r "$BUILD/html" "$STAGE/latest"
rm -f "$STAGE/latest/.buildinfo"
cp docs/versions.json "$STAGE/versions.json"
tar -czf "$STAGE.tar.gz" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"

echo "Upload tree ready at $STAGE and $STAGE.tar.gz"
