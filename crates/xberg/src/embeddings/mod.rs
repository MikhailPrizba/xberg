//! Embedding generation support for RAG (Retrieval-Augmented Generation) systems.
//!
//! This module provides text embedding generation using ONNX models via a vendored
//! text embedding inference engine. Embeddings can be generated for text chunks to
//! enable semantic search and RAG pipelines.
//!
//! # Features
//!
//! - Multiple pre-configured models optimized for different use cases
//! - Preset configurations for common RAG scenarios
//! - Full customization of model location and parameters
//! - Batch processing for efficient embedding generation
//! - Thread-safe inference without mutex contention
//! - Optional GPU acceleration via ONNX Runtime execution providers
//!
//! # ONNX Runtime Requirement
//!
//! **CRITICAL**: This module requires ONNX Runtime to be installed on the system.
//! The `embeddings` feature uses dynamic loading (`ort-load-dynamic`), which detects
//! the ONNX Runtime library at runtime.
//!
//! ## Installation Instructions
//!
//! - **macOS**: `brew install onnxruntime`
//! - **Linux (Ubuntu/Debian)**: `apt install libonnxruntime libonnxruntime-dev`
//! - **Linux (Fedora)**: `dnf install onnxruntime onnxruntime-devel`
//! - **Linux (Arch)**: `pacman -S onnxruntime`
//! - **Windows (MSVC)**: Download from <https://github.com/microsoft/onnxruntime/releases> and add to PATH
//!
//! Alternatively, set the `ORT_DYLIB_PATH` environment variable to the ONNX Runtime library path.
//!
//! For Docker/containers, install via package manager in your base image.
//! Verified packages: Ubuntu 22.04+, Fedora 38+, Arch Linux.
//!
//! ## Platform Limitations
//!
//! **Windows MinGW builds are not supported**. ONNX Runtime requires the MSVC toolchain on Windows.
//! Please use Windows MSVC builds or disable the embeddings feature.
//!
//! # Static (model2vec) Embeddings
//!
//! The `"lightweight"` preset (and any future `Static`-backend preset) runs
//! through a pure-Rust model2vec engine instead of ONNX Runtime, gated behind
//! the `static-embeddings` feature. It requires no native ONNX dependency and is
//! the only dense-embedding backend available on `no-ort-target` (WASM, Android
//! x86_64 emulator). Select it the same way as any other preset:
//! `EmbeddingConfig { model: EmbeddingModelType::Preset { name: "lightweight".into() }, .. }`.
//!
//! # Example
//!
//! ```rust,ignore
//! use xberg::{extract, ChunkingConfig, EmbeddingConfig, ExtractInput, ExtractionConfig};
//!
//! let config = ExtractionConfig {
//!     chunking: Some(ChunkingConfig {
//!         preset: Some("balanced".to_string()),
//!         embedding: Some(EmbeddingConfig::default()),
//!         ..Default::default()
//!     }),
//!     ..Default::default()
//! };
//!
//! let output = extract(ExtractInput::from_uri("document.pdf"), &config).await?;
//! let result = output.results.into_iter().next().expect("one input yields one result");
//! for chunk in result.chunks.unwrap() {
//!     if let Some(embedding) = chunk.embedding {
//!         println!("Chunk has {} dimension embedding", embedding.len());
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Inference backend that an [`EmbeddingPreset`] runs on.
///
/// `Onnx` presets require the `embeddings` feature (ONNX Runtime, not available on
/// WASM/Android x86_64 emulator). `Static` presets require `static-embeddings`
/// (pure-Rust model2vec inference, no ORT — the only dense-embedding backend
/// available on `no-ort-target`).
///
/// Defaults to `Onnx` via `#[serde(default)]` so every existing preset payload
/// (which predates this field) keeps deserializing without change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingBackend {
    /// ONNX Runtime transformer inference (the historical, default backend).
    #[default]
    Onnx,
    /// Pure-Rust static (model2vec) inference — no ONNX Runtime.
    Static,
}

/// Preset configurations for common RAG use cases.
///
/// Each preset combines chunk size, overlap, and embedding model
/// to provide an optimized configuration for specific scenarios.
///
/// All string fields are owned `String` for FFI compatibility — instances
/// are safe to clone and pass across language boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingPreset {
    /// Short identifier for this preset (e.g. `"balanced"`, `"fast"`, `"quality"`).
    pub name: String,
    /// Target chunk size in characters.
    pub chunk_size: usize,
    /// Overlap between consecutive chunks in characters.
    pub overlap: usize,
    /// HuggingFace repository name for the model.
    pub model_repo: String,
    /// Pooling strategy: "cls" or "mean". Static (model2vec) presets always mean-pool.
    pub pooling: String,
    /// Path to the model file within the repo (ONNX file for `Onnx`, model2vec
    /// `model.safetensors` for `Static`).
    pub model_file: String,
    /// Embedding vector dimension produced by this model.
    pub dimensions: usize,
    /// Human-readable description of the preset's intended use case.
    pub description: String,
    /// Which inference backend this preset runs on. Defaults to `Onnx` for
    /// back-compat with presets/payloads that predate this field.
    #[serde(default)]
    pub backend: EmbeddingBackend,
    /// Sibling files downloaded alongside `model_file`. Large fp32 ONNX exports
    /// (Qwen3-Embedding, Arctic-Embed-v2.0) store weights in an external-data
    /// `model.onnx.data` blob that ORT loads by relative path; single-file
    /// models leave this empty.
    #[serde(default)]
    pub additional_files: Vec<String>,
    /// Instruction prefix prepended to *query-side* text before encoding.
    /// Asymmetric retrieval models (Arctic-Embed, E5) are trained with a
    /// `"query: "`-style prefix on queries only; document text is never
    /// prefixed. `None` for symmetric models.
    #[serde(default)]
    pub query_prefix: Option<String>,
}

pub static EMBEDDING_PRESETS: LazyLock<Vec<EmbeddingPreset>> = LazyLock::new(|| {
    vec![
        EmbeddingPreset {
            name: "fast".to_string(),
            chunk_size: 512,
            overlap: 50,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "mean".to_string(),
            model_file: "all-MiniLM-L6-v2/model_quantized.onnx".to_string(),
            dimensions: 384,
            description: "Fast embedding with quantized model (384 dims, ~22M params). Best for: Quick prototyping, development, resource-constrained environments.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "balanced".to_string(),
            chunk_size: 1024,
            overlap: 100,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "cls".to_string(),
            model_file: "bge-base-en-v1.5/model.onnx".to_string(),
            dimensions: 768,
            description: "Balanced quality and speed (768 dims, ~109M params). Best for: General-purpose RAG, production deployments, English documents.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "quality".to_string(),
            chunk_size: 2000,
            overlap: 200,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "cls".to_string(),
            model_file: "bge-large-en-v1.5/model.onnx".to_string(),
            dimensions: 1024,
            description: "High quality with larger context (1024 dims, ~335M params). Best for: Complex documents, maximum accuracy, sufficient compute resources.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "multilingual".to_string(),
            chunk_size: 1024,
            overlap: 100,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "mean".to_string(),
            model_file: "multilingual-e5-base/model.onnx".to_string(),
            dimensions: 768,
            description: "Multilingual support (768 dims, 100+ languages). Best for: International documents, mixed-language content, global applications.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "gte-modernbert-base".to_string(),
            chunk_size: 1024,
            overlap: 100,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "cls".to_string(),
            model_file: "gte-modernbert-base/model.onnx".to_string(),
            dimensions: 768,
            description: "GTE ModernBERT base (768 dims, 2026-gen, 8192 context). Best for: general-purpose English RAG with long-context ModernBERT tokenization.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "lightweight".to_string(),
            chunk_size: 512,
            overlap: 50,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "mean".to_string(),
            model_file: "potion-base-8m/model.safetensors".to_string(),
            dimensions: 256,
            description: "Static (model2vec) embedding — pure Rust, no ONNX Runtime (256 dims, ~7.5M params). Best for: WASM, Android, and other no-ORT targets; extremely fast CPU-only inference.".to_string(),
            backend: EmbeddingBackend::Static,
            additional_files: Vec::new(),
            query_prefix: None,
        },
        EmbeddingPreset {
            name: "arctic-embed-m-v2.0".to_string(),
            chunk_size: 1024,
            overlap: 100,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "cls".to_string(),
            model_file: "arctic-embed-m-v2.0/model.onnx".to_string(),
            dimensions: 768,
            description: "Snowflake Arctic-Embed-M v2.0 (768 dims, multilingual, 2026-gen). Asymmetric retrieval: queries are prefixed with \"query: \". Best for: multilingual RAG where query/document roles are known.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: vec!["arctic-embed-m-v2.0/model.onnx.data".to_string()],
            query_prefix: Some("query: ".to_string()),
        },
        EmbeddingPreset {
            name: "qwen3-embedding-0.6b".to_string(),
            chunk_size: 2000,
            overlap: 200,
            model_repo: "xberg-io/embedding-models".to_string(),
            pooling: "last".to_string(),
            model_file: "qwen3-embedding-0.6b/model.onnx".to_string(),
            dimensions: 1024,
            description: "Qwen3-Embedding 0.6B (1024 dims, decoder-style last-token pooling, 32k context, multilingual, 2026-gen). Best for: highest-quality multilingual/long-context retrieval when compute allows.".to_string(),
            backend: EmbeddingBackend::Onnx,
            additional_files: vec!["qwen3-embedding-0.6b/model.onnx.data".to_string()],
            query_prefix: None,
        },
    ]
});

/// Get a preset by name (returns an owned clone for FFI compatibility).
pub(crate) fn get_preset(name: &str) -> Option<EmbeddingPreset> {
    EMBEDDING_PRESETS.iter().find(|p| p.name == name).cloned()
}

/// Query-side instruction prefix for the given embedding config, if the
/// resolved preset defines one.
///
/// Asymmetric retrieval models (e.g. Arctic-Embed) are trained with a
/// `"query: "`-style prefix on the query only — the RAG query path prepends
/// this before embedding, while document text is embedded verbatim. Returns
/// `None` for symmetric presets, custom models, and non-preset backends.

pub fn embedding_query_prefix(config: &crate::core::config::EmbeddingConfig) -> Option<String> {
    match &config.model {
        crate::core::config::EmbeddingModelType::Preset { name } => get_preset(name).and_then(|p| p.query_prefix),
        _ => None,
    }
}

/// List all available preset names (owned clones for FFI compatibility).
pub(crate) fn list_presets() -> Vec<String> {
    EMBEDDING_PRESETS.iter().map(|p| p.name.clone()).collect()
}

// The cache functions live in their own module so the feature gate sits on the module
// declaration: alef copies a feature named in a function's own `#[cfg]` into every binding
// manifest, and `static-embeddings` is not a binding feature. ~keep

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_preset() {
        assert!(get_preset("balanced").is_some());
        assert!(get_preset("fast").is_some());
        assert!(get_preset("quality").is_some());
        assert!(get_preset("multilingual").is_some());
        assert!(get_preset("gte-modernbert-base").is_some());
        assert!(get_preset("lightweight").is_some());
        assert!(get_preset("nonexistent").is_none());
    }

    #[test]
    fn test_list_presets() {
        let presets = list_presets();
        assert_eq!(presets.len(), 8, "expected exactly 8 presets, got: {presets:?}");
        assert!(presets.iter().any(|n| n == "fast"));
        assert!(presets.iter().any(|n| n == "balanced"));
        assert!(presets.iter().any(|n| n == "quality"));
        assert!(presets.iter().any(|n| n == "multilingual"));
        assert!(presets.iter().any(|n| n == "gte-modernbert-base"));
        assert!(presets.iter().any(|n| n == "lightweight"));
        assert!(presets.iter().any(|n| n == "arctic-embed-m-v2.0"));
        assert!(presets.iter().any(|n| n == "qwen3-embedding-0.6b"));
    }

    #[test]
    fn asymmetric_presets_carry_query_prefix_and_external_data() {
        let arctic = get_preset("arctic-embed-m-v2.0").expect("arctic preset must exist");
        assert_eq!(arctic.query_prefix.as_deref(), Some("query: "));
        assert_eq!(arctic.pooling, "cls");
        assert_eq!(arctic.dimensions, 768);
        assert_eq!(
            arctic.additional_files,
            vec!["arctic-embed-m-v2.0/model.onnx.data".to_string()]
        );

        let qwen3 = get_preset("qwen3-embedding-0.6b").expect("qwen3-embedding preset must exist");
        assert_eq!(qwen3.query_prefix, None);
        assert_eq!(qwen3.pooling, "last");
        assert_eq!(qwen3.dimensions, 1024);
        assert_eq!(
            qwen3.additional_files,
            vec!["qwen3-embedding-0.6b/model.onnx.data".to_string()]
        );
    }

    #[test]
    fn lightweight_preset_uses_static_backend() {
        let preset = get_preset("lightweight").expect("lightweight preset must exist");
        assert_eq!(preset.backend, EmbeddingBackend::Static);
        assert_eq!(preset.dimensions, 256);
        assert_eq!(preset.model_repo, "xberg-io/embedding-models");
    }

    #[test]
    fn every_onnx_preset_defaults_to_onnx_backend() {
        for preset in EMBEDDING_PRESETS.iter().filter(|p| p.name != "lightweight") {
            assert_eq!(
                preset.backend,
                EmbeddingBackend::Onnx,
                "preset '{}' should default to the Onnx backend",
                preset.name
            );
        }
    }

    #[test]
    fn embedding_backend_deserializes_missing_field_as_onnx() {
        // deserialize, defaulting to Onnx via #[serde(default)].
        let json = r#"{
            "name": "custom",
            "chunk_size": 512,
            "overlap": 50,
            "model_repo": "org/repo",
            "pooling": "mean",
            "model_file": "model.onnx",
            "dimensions": 384,
            "description": "test"
        }"#;
        let preset: EmbeddingPreset = serde_json::from_str(json).expect("should deserialize without backend field");
        assert_eq!(preset.backend, EmbeddingBackend::Onnx);
    }

    #[test]
    fn test_preset_dimensions() {
        let balanced = get_preset("balanced").unwrap();
        assert_eq!(balanced.dimensions, 768);

        let fast = get_preset("fast").unwrap();
        assert_eq!(fast.dimensions, 384);

        let quality = get_preset("quality").unwrap();
        assert_eq!(quality.dimensions, 1024);
    }

    #[test]
    fn test_preset_chunk_sizes() {
        let fast = get_preset("fast").unwrap();
        assert_eq!(fast.chunk_size, 512);
        assert_eq!(fast.overlap, 50);

        let quality = get_preset("quality").unwrap();
        assert_eq!(quality.chunk_size, 2000);
        assert_eq!(quality.overlap, 200);
    }

    #[test]
    fn test_preset_model_repos() {
        let fast = get_preset("fast").unwrap();
        assert_eq!(fast.model_repo, "xberg-io/embedding-models");
        assert_eq!(fast.pooling, "mean");
        assert_eq!(fast.model_file, "all-MiniLM-L6-v2/model_quantized.onnx");

        let balanced = get_preset("balanced").unwrap();
        assert_eq!(balanced.model_repo, "xberg-io/embedding-models");
        assert_eq!(balanced.pooling, "cls");
    }

    // uses `#[tokio::test]` throughout. A `static-embeddings`-only build (no
}
