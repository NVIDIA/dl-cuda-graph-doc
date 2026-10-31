# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "CUDA Graph Best Practices for PyTorch"
copyright = "2026, NVIDIA Corporation"
author = "NVIDIA Corporation"
version = "latest"  # Fills <meta name="docsearch:version">
release = "latest"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For Markdown docs
    "sphinx_copybutton",  # For copy button in code blocks
    "sphinx_design",  # For grid layouts and cards
    "sphinxcontrib.mermaid",  # For Mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "colon_fence",  # Enables ::: admonitions
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for sphinx_design cards
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5
myst_fence_as_directive = ["mermaid"]  # Treat ```mermaid blocks as directives

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/dl-cuda-graph-doc",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {
        "json_url": "../versions.json",
        "version_match": release,
    },
    # OneTrust consent banner and Adobe Analytics, required for public NVIDIA docs
    "public_docs_features": True,
}

# Static files (images, etc.)
html_static_path = ["assets"]

# Extra files to copy to the build output; versions.json is published one level up
# (see scripts/build-release.sh)
html_extra_path = ["project.json"]

# -- Linkcheck configuration -------------------------------------------------
linkcheck_ignore = []
