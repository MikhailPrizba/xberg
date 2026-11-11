//! Sparse (SPLADE) learned embeddings.
//!
//! Produces high-dimensional, mostly-zero vocabulary vectors from a
//! `BertForMaskedLM` ONNX model, stored as parallel `(indices, values)` arrays.
//! These enable hybrid dense+sparse retrieval: a sparse arm captures exact
//! lexical term importance that dense vectors smear away.
//!
//! Built on the shared [`crate::onnx`] model-loading helpers. The engine math is
//! in [`engine`].
//!

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// A sparse learned embedding: vocabulary term indices and their weights.
///
/// `indices` are ascending vocabulary token ids; `values[i]` is the weight for
/// `indices[i]`. The two arrays always have equal length. Only strictly-positive
/// terms are retained, so the representation is genuinely sparse.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct SparseEmbedding {
    /// Vocabulary token ids with non-zero weight, ascending.
    pub indices: Vec<u32>,
    /// Weights parallel to [`SparseEmbedding::indices`].
    pub values: Vec<f32>,
}

/// Static metadata for a bundled SPLADE preset (WASM/Android-safe, no ORT).
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct SparseEmbeddingPreset {
    /// Stable preset name referenced from config.
    pub name: String,
    /// HuggingFace repository hosting the ONNX model.
    pub model_repo: String,
    /// Path to the ONNX file within the repo.
    pub model_file: String,
    /// Sibling files that must be downloaded alongside `model_file`.
    pub additional_files: Vec<String>,
    /// Maximum token sequence length.
    pub max_length: usize,
    /// Human-readable description.
    pub description: String,
}

pub static SPARSE_EMBEDDING_PRESETS: LazyLock<Vec<SparseEmbeddingPreset>> = LazyLock::new(|| {
    vec![
        SparseEmbeddingPreset {
            name: "splade".to_string(),
            model_repo: "xberg-io/sparse-embeddings".to_string(),
            model_file: "splade/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 256,
            description: "SPLADE++ EN v1 — English learned sparse retrieval (Apache-2.0).".to_string(),
        },
        SparseEmbeddingPreset {
            name: "opensearch-v3-distill".to_string(),
            model_repo: "xberg-io/sparse-embeddings".to_string(),
            model_file: "opensearch-v3-distill/model.onnx".to_string(),
            additional_files: vec!["opensearch-v3-distill/model.onnx.data".to_string()],
            max_length: 512,
            description: "OpenSearch neural-sparse v3 distill (DistilBERT MLM, 2026-gen, Apache-2.0). \
                Exported with its MLM head; 30522-dim SPLADE sparse vectors, 512 max-len."
                .to_string(),
        },
    ]
});

/// Look up a bundled SPLADE preset by exact name.
///
#[cfg(feature = "sparse-embedding-presets")]

pub fn get_preset(name: &str) -> Option<SparseEmbeddingPreset> {
    SPARSE_EMBEDDING_PRESETS.iter().find(|p| p.name == name).cloned()
}

/// List the names of all bundled SPLADE presets.
///
#[cfg(feature = "sparse-embedding-presets")]

pub fn list_presets() -> Vec<String> {
    SPARSE_EMBEDDING_PRESETS.iter().map(|p| p.name.clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_catalog_is_nonempty_and_lookup_works() {
        assert!(!SPARSE_EMBEDDING_PRESETS.is_empty());
        assert!(list_presets().contains(&"splade".to_string()));
        let p = get_preset("splade").expect("splade preset present");
        assert_eq!(p.model_repo, "xberg-io/sparse-embeddings");
        assert_eq!(p.model_file, "splade/model.onnx");
        assert!(get_preset("does-not-exist").is_none());
    }

    #[test]
    fn sparse_embedding_serde_roundtrip() {
        let se = SparseEmbedding {
            indices: vec![3, 17, 200],
            values: vec![0.5, 0.25, 0.1],
        };
        let json = serde_json::to_string(&se).unwrap();
        let back: SparseEmbedding = serde_json::from_str(&json).unwrap();
        assert_eq!(se, back);
    }
}
