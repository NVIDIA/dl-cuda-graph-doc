#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Build the release HTML for a pinned version and for "latest", and stage the upload tree:
#   docs/_build/release/dl-cuda-graph/versions.json   -> /dl-cuda-graph/versions.json
#   docs/_build/release/dl-cuda-graph/<version>/      -> /dl-cuda-graph/<version>/
#   docs/_build/release/dl-cuda-graph/latest/         -> /dl-cuda-graph/latest/
# and pack it into docs/_build/release/dl-cuda-graph.tar.gz.
#
# Usage: ./scripts/build-release.sh <version>    (e.g. 1.0)

set -euo pipefail

cd "$(dirname "$0")/.."

PINNED=${1:?usage: $0 <version>, e.g. 1.0}
BUILD=docs/_build
STAGE="$BUILD/release/dl-cuda-graph"

if ! grep -q "\"version\": \"$PINNED\"" docs/versions.json; then
    echo "error: version $PINNED is not listed in docs/versions.json" >&2
    exit 1
fi

rm -rf "$BUILD"
uv sync --group docs
mkdir -p "$STAGE"

for v in "$PINNED" latest; do
    # Keep doctrees out of the HTML output directory
    DOCS_VERSION="$v" uv run --group docs \
        sphinx-build -W -b html -d "$BUILD/doctrees/$v" docs "$STAGE/$v"
    rm -f "$STAGE/$v/.buildinfo"
    printf '{"name": "dl-cuda-graph-doc", "version": "%s"}\n' "$v" > "$STAGE/$v/project.json"
done

cp docs/versions.json "$STAGE/versions.json"
tar -czf "$STAGE.tar.gz" -C "$(dirname "$STAGE")" "$(basename "$STAGE")"

echo "Upload tree ready at $STAGE and $STAGE.tar.gz"
