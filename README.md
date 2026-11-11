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
uv build --out-dir /tmp/xberg-compact-wheels
```

The checked Cargo manifests select the compact profile. Its feature list and
retained workspace members are recorded in compact-profile.json. Release builds
use thin LTO and 16 codegen units. Cargo defaults to the Python package. The workspace contains the core, native PDF, Python and PDFium crates.
Only the Python crate is built by default. PDFium is available through the
optional pdf-pdfium feature and is a workspace member for maintenance checks.
Unused OCR/ML/HEIC/WordPerfect crates, model implementations, server transports and
upstream generation/release scripts have been removed. VS Code checks only xberg-py.
The default Cargo cache is ../.cache/xberg-compact/target, outside this checkout.

## Capabilities

OCR is supplied by a host-registered backend such as textractor's external OCR
plugin. Built-in Tesseract/Paddle/Candle engines, ONNX Runtime, embedding/reranker
models, LLM inference, transcription, tree-sitter grammar engines, HTTP/MCP servers,
HEIC, WordPerfect and PDFium are disabled in this build.

The generated Python options surface keeps lightweight configuration/result types
and a few pure-Rust helpers for compatibility. Paddle configuration DTOs live in
src/compat/paddle_ocr.rs inside the core crate; its engine code is removed. Those types do not enable models or
servers. Optional classify_chunks_owned, map_url and extract_with_layout_pages
entry points raise NotImplementedError in this extraction profile.

Unsupported upstream features, including full and mobile, are removed from Cargo
manifests. Source branches for removed features have been deleted, including unreachable
modules and private model/layout execution paths. Only retained Cargo features
and platform/test conditions remain. Python
binding adaptations and lightweight type gates are maintained in this fork; review
them when merging upstream generated bindings.

## Optional PDFium

The default wheel uses the native Rust PDF engine. For a future PDFium comparison:

```sh
uv run maturin build --release --locked --features pdf-pdfium --out /tmp/xberg-pdfium-wheels
```

This enables the backend without changing the extraction default. Select Pdfium
explicitly in PdfConfig when running a comparison. Set PDFIUM_DYNAMIC_LIB_PATH at runtime to the directory containing a compatible
libpdfium shared library, or install it on the system library search path.
The Python build configuration lives in the repository-root pyproject.toml.
Source archives explicitly include this optional path crate. PDFium uses checked-in
Rust FFI bindings; the unused bindgen generator and C headers are removed.
Compilation with this feature is checked; runtime PDFium extraction is not yet
validated on this host.

## Maintenance

The original source is available through origin; fetch additional history only
when needed. No Git history was rewritten on the remote. Native PDF tests and
fixtures are retained for parser fixes. License and attribution files remain.
Some upstream Rust tests additionally require external binary fixtures under
`test_documents`; fetch the needed paths using that repository's README.
Compile-time PDF/TSV fixtures used by retained core tests are kept under fixtures/repros.
Validation results and measured sizes are recorded in compact-verification.json.
