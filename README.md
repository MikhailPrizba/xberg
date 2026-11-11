# Xberg compact extraction fork

A Python/Rust extraction build for textractor. Upstream source commit:
`ff69566bc3757104fbf28c7afdededda6c1adaac`.

The checkout has shallow Git history and retains Python as its only language binding.
Core format parsers and the custom OCR plugin interface remain available. The build
keeps the formats used by textractor, including native PDF, DOC/DOCX/ODT, ODS/Excel,
HTML, presentations, ebooks, email, and the image input path.

## Build

Requires Rust 1.99.1, Python 3.11+, a C compiler and CMake. Use the minimal Rust
profile in rust-toolchain.toml; no WASM toolchain is installed.

Keep compilation artifacts outside this source checkout:

```sh
export CARGO_TARGET_DIR="$HOME/.cache/xberg-compact/target"
uv sync --frozen
uv run --frozen python -c 'import xberg; print(xberg.__file__)'
uv build --package xberg --out-dir /tmp/xberg-compact-wheels
```

The checked Cargo manifests select the compact profile. Its feature list and
retained workspace members are recorded in compact-profile.json. Release builds
use thin LTO and 16 codegen units. Cargo defaults to the Python package. Only the core, native PDF and Python crates
are workspace members. Dormant optional path crates are excluded from workspace
builds and retained only for manifest resolution. VS Code checks only xberg-py.
The default Cargo cache is ../.cache/xberg-compact/target, outside this checkout.

## Capabilities

OCR is supplied by a host-registered backend such as textractor's external OCR
plugin. Built-in Tesseract/Paddle/Candle engines, ONNX Runtime, embedding/reranker
models, LLM inference, transcription, tree-sitter grammar engines, HTTP/MCP servers,
HEIC, WordPerfect and PDFium are disabled in this build.

The generated Python options surface keeps lightweight configuration/result types
and a few pure-Rust helpers for compatibility. Those types do not enable models or
servers. Optional classify_chunks_owned, map_url and extract_with_layout_pages
entry points raise NotImplementedError in this extraction profile.

Do not enable upstream aggregate features such as full or mobile when building
this profile. Python binding adaptations and lightweight type gates are maintained
in this fork; review them when merging upstream generated bindings.

## Maintenance

The original source is available through origin; fetch additional history only
when needed. No Git history was rewritten on the remote. Native PDF tests and
fixtures are retained for parser fixes. License and attribution files remain.
Validation results and exact measured sizes are recorded in compact-verification.json
once the compact wheel has passed the textractor checks.
