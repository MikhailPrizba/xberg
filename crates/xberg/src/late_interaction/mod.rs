//! ColBERT late-interaction (multi-vector) embeddings.
//!
//! Produces a *sequence* of per-token vectors per document (rather than a
//! single pooled vector) from a ColBERT-style ONNX model. Retrieval scores
//! documents against a query via MaxSim — for every query token, take the
//! maximum similarity against any document token, then sum across query
//! tokens — which captures fine-grained term-level interaction that dense
//! single-vector embeddings smear away.
//!
//! Built on the shared [`crate::onnx`] model-loading helpers. The engine math
//! (tokenization, marker insertion, query augmentation, per-token
//! normalization) is in [`engine`].
//!

use std::sync::LazyLock;

use serde::{Deserialize, Serialize};

/// A ColBERT multi-vector embedding: one row per attention-live token.
///
/// `data` is a flat, row-major buffer of length `num_tokens * dim` — row `i`
/// (the embedding for token `i`) occupies `data[i*dim .. (i+1)*dim]`. Flat
/// storage keeps the type FFI-friendly across binding boundaries; use
/// [`MultiVectorEmbedding::rows`] internally to iterate per-token slices.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct MultiVectorEmbedding {
    /// Number of attention-live token rows (padding rows are dropped, not
    /// zeroed — see [`engine::normalize_tokens`]).
    pub num_tokens: u32,
    /// Dimensionality of each per-token vector.
    pub dim: u32,
    /// Flat row-major buffer, length `num_tokens * dim`.
    pub data: Vec<f32>,
}

impl MultiVectorEmbedding {
    /// Returns `true` if `data` holds exactly `num_tokens * dim` values — i.e.
    /// the flat buffer matches the declared shape.
    ///
    /// All fields are `pub` and the type is `Deserialize`, so a value coming
    /// from an untrusted source (FFI caller, JSON, a store row) may be
    /// malformed. [`max_sim_score`] guards on this so a length-mismatched
    /// buffer scores `0.0` rather than silently mis-scoring (a shorter `data`
    /// would make [`rows`](Self::rows)' `chunks_exact` drop a trailing partial
    /// chunk). Uses `checked_mul` so an overflowing `num_tokens * dim` is
    /// reported as malformed instead of wrapping.
    ///
    pub fn is_well_formed(&self) -> bool {
        (self.num_tokens as usize)
            .checked_mul(self.dim as usize)
            .is_some_and(|expected| expected == self.data.len())
    }

    /// Iterate over this embedding's per-token vectors as `&[f32]` slices.
    ///
    /// Internal helper for MaxSim scoring; not part of the FFI-facing surface.
    /// Panic-safe on `dim == 0` (`chunks_exact(0)` would panic): a zero-dim
    /// value yields an empty iterator. Callers should still validate via
    /// [`Self::is_well_formed`] to get a meaningful row count.
    pub(crate) fn rows(&self) -> impl Iterator<Item = &[f32]> {
        let dim = self.dim as usize;
        let (data, step): (&[f32], usize) = if dim == 0 { (&[], 1) } else { (&self.data, dim) };
        data.chunks_exact(step)
    }
}

/// Static metadata for a bundled ColBERT preset (WASM/Android-safe, no ORT).
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct LateInteractionPreset {
    /// Stable preset name referenced from config.
    pub name: String,
    /// HuggingFace repository hosting the ONNX model.
    pub model_repo: String,
    /// Path to the ONNX file within the repo.
    pub model_file: String,
    /// Sibling files that must be downloaded alongside `model_file`.
    pub additional_files: Vec<String>,
    /// Maximum document token sequence length.
    pub max_length: usize,
    /// Fixed padded query length (ColBERT query augmentation).
    pub query_max_length: usize,
    /// Per-token embedding dimensionality.
    pub dim: usize,
    /// Human-readable description.
    pub description: String,
}

pub static LATE_INTERACTION_PRESETS: LazyLock<Vec<LateInteractionPreset>> = LazyLock::new(|| {
    vec![
        LateInteractionPreset {
            name: "colbert".to_string(),
            model_repo: "xberg-io/late-interaction-models".to_string(),
            model_file: "colbert-small-v1/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 512,
            query_max_length: 32,
            dim: 96,
            description: "AnswerAI ColBERT small v1 — English multi-vector late-interaction retrieval (Apache-2.0)."
                .to_string(),
        },
        LateInteractionPreset {
            name: "gte-moderncolbert".to_string(),
            model_repo: "xberg-io/late-interaction-models".to_string(),
            model_file: "gte-moderncolbert-v1/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 512,
            query_max_length: 32,
            dim: 128,
            description: "LightOn GTE-ModernColBERT v1 — ModernBERT long-context multi-vector \
                late-interaction retrieval, 128-dim tokens (Apache-2.0)."
                .to_string(),
        },
    ]
});

/// Look up a bundled ColBERT preset by exact name.
///
#[cfg(feature = "late-interaction-presets")]

pub fn get_preset(name: &str) -> Option<LateInteractionPreset> {
    LATE_INTERACTION_PRESETS.iter().find(|p| p.name == name).cloned()
}

/// List the names of all bundled ColBERT presets.
///
#[cfg(feature = "late-interaction-presets")]

pub fn list_presets() -> Vec<String> {
    LATE_INTERACTION_PRESETS.iter().map(|p| p.name.clone()).collect()
}

/// A single document match returned by [`max_sim_rank`], with its position in
/// the input and MaxSim score.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct LateInteractionMatch {
    /// Position of this document in the original input slice.
    pub index: usize,
    /// MaxSim relevance score. Higher means more relevant to the query.
    pub score: f32,
}

/// Score a query against a document using ColBERT's MaxSim operator: for each
/// query token vector, take the maximum dot product against any document
/// token vector, then sum across query tokens.
///
/// Returns `0.0` if `query` and `doc` have mismatched dimensionality, if either
/// has zero tokens, or if either is not well-formed per
/// [`MultiVectorEmbedding::is_well_formed`] (its `data` length does not match
/// `num_tokens * dim`).
///
/// Pure CPU primitive — available without ONNX Runtime.
///
#[cfg(feature = "late-interaction-presets")]
pub fn max_sim_score(query: &MultiVectorEmbedding, doc: &MultiVectorEmbedding) -> f64 {
    if query.dim != doc.dim
        || query.dim == 0
        || query.num_tokens == 0
        || doc.num_tokens == 0
        || !query.is_well_formed()
        || !doc.is_well_formed()
    {
        return 0.0;
    }

    query
        .rows()
        .map(|q_row| {
            doc.rows()
                .map(|d_row| q_row.iter().zip(d_row.iter()).map(|(a, b)| a * b).sum::<f32>())
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum::<f32>() as f64
}

/// Rank a set of documents against a query by MaxSim score, descending.
///
/// Mirrors the sort/truncate shape of `crate::reranking`'s `build_results`,
/// minus top-k truncation (callers slice the returned `Vec` themselves).
///
/// Pure CPU primitive — available without ONNX Runtime.
///
#[cfg(feature = "late-interaction-presets")]
pub fn max_sim_rank(query: &MultiVectorEmbedding, docs: &[MultiVectorEmbedding]) -> Vec<LateInteractionMatch> {
    let mut results: Vec<LateInteractionMatch> = docs
        .iter()
        .enumerate()
        .map(|(index, doc)| LateInteractionMatch {
            index,
            score: max_sim_score(query, doc) as f32,
        })
        .collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_catalog_is_nonempty_and_lookup_works() {
        assert!(!LATE_INTERACTION_PRESETS.is_empty());
        assert!(list_presets().contains(&"colbert".to_string()));
        let p = get_preset("colbert").expect("colbert preset present");
        assert_eq!(p.model_repo, "xberg-io/late-interaction-models");
        assert_eq!(p.model_file, "colbert-small-v1/model.onnx");
        assert_eq!(p.query_max_length, 32);
        assert!(get_preset("does-not-exist").is_none());
    }

    #[test]
    fn multi_vector_embedding_serde_roundtrip() {
        let mv = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 3,
            data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        };
        let json = serde_json::to_string(&mv).unwrap();
        let back: MultiVectorEmbedding = serde_json::from_str(&json).unwrap();
        assert_eq!(mv, back);
    }

    #[test]
    fn rows_iterates_token_vectors() {
        let mv = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 2,
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let rows: Vec<&[f32]> = mv.rows().collect();
        assert_eq!(rows, vec![[1.0, 2.0].as_slice(), [3.0, 4.0].as_slice()]);
    }

    #[test]
    fn max_sim_score_sums_best_match_per_query_token() {
        let query = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 2,
            data: vec![1.0, 0.0, 0.0, 1.0],
        };
        let doc = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 2,
            data: vec![1.0, 0.0, 0.6, 0.8],
        };
        let score = max_sim_score(&query, &doc);
        assert!((score - 1.8).abs() < 1e-5, "expected 1.8, got {score}");
    }

    #[test]
    fn max_sim_score_zero_on_dim_mismatch() {
        let query = MultiVectorEmbedding {
            num_tokens: 1,
            dim: 2,
            data: vec![1.0, 0.0],
        };
        let doc = MultiVectorEmbedding {
            num_tokens: 1,
            dim: 3,
            data: vec![1.0, 0.0, 0.0],
        };
        assert_eq!(max_sim_score(&query, &doc), 0.0);
    }

    #[test]
    fn max_sim_score_zero_on_empty_tokens() {
        let query = MultiVectorEmbedding {
            num_tokens: 0,
            dim: 2,
            data: vec![],
        };
        let doc = MultiVectorEmbedding {
            num_tokens: 1,
            dim: 2,
            data: vec![1.0, 0.0],
        };
        assert_eq!(max_sim_score(&query, &doc), 0.0);
    }

    #[test]
    fn is_well_formed_checks_data_length_against_shape() {
        let ok = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 3,
            data: vec![0.0; 6],
        };
        assert!(ok.is_well_formed());

        let short = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 3,
            data: vec![0.0; 5],
        };
        assert!(!short.is_well_formed());

        let overflow = MultiVectorEmbedding {
            num_tokens: u32::MAX,
            dim: u32::MAX,
            data: vec![0.0; 1],
        };
        assert!(!overflow.is_well_formed());
    }

    #[test]
    fn max_sim_score_zero_on_malformed_buffer() {
        let query = MultiVectorEmbedding {
            num_tokens: 1,
            dim: 2,
            data: vec![1.0, 0.0],
        };
        let malformed_doc = MultiVectorEmbedding {
            num_tokens: 2,
            dim: 2,
            data: vec![1.0, 0.0, 0.6],
        };
        assert_eq!(max_sim_score(&query, &malformed_doc), 0.0);
    }

    #[test]
    fn max_sim_rank_sorts_descending_by_score() {
        let query = MultiVectorEmbedding {
            num_tokens: 1,
            dim: 2,
            data: vec![1.0, 0.0],
        };
        let docs = vec![
            MultiVectorEmbedding {
                num_tokens: 1,
                dim: 2,
                data: vec![0.5, 0.0],
            },
            MultiVectorEmbedding {
                num_tokens: 1,
                dim: 2,
                data: vec![1.0, 0.0],
            },
            MultiVectorEmbedding {
                num_tokens: 1,
                dim: 2,
                data: vec![0.1, 0.0],
            },
        ];
        let ranked = max_sim_rank(&query, &docs);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].index, 1);
        assert_eq!(ranked[1].index, 0);
        assert_eq!(ranked[2].index, 2);
        assert!(ranked[0].score >= ranked[1].score);
        assert!(ranked[1].score >= ranked[2].score);
    }
}
