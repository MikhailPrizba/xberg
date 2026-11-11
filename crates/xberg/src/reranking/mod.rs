//! Cross-encoder reranking support.
//!
//! This module provides `(query, document)` pair scoring using ONNX cross-encoder
//! models. Reranking is the standard "second pass" in retrieval pipelines:
//! a first-pass embedding search retrieves a candidate set cheaply; reranking
//! rescores and reorders those candidates by true relevance to the query.
//!
//! Three backend variants are supported:
//! - **Local ONNX** — cross-encoder models (ms-marco, bge-reranker) via ONNX Runtime.
//! - **liter-llm** — provider-hosted rerankers (Cohere, Jina, Voyage) via an API.
//! - **In-process plugin** — caller-supplied backends registered via
//!   [`crate::plugins::register_reranker_backend`].
//!
//! # Download/cache machinery
//!
//! The ONNX path downloads models, loads the tokenizer, and builds the ORT
//! session through the shared [`crate::onnx`] helpers (enabled by the
//! `onnx-runtime` feature that `reranker` pulls in), so no download/lock or ORT
//! diagnostic code is duplicated here — the cross-encoder pair encoding is the
//! only reranker-specific step, applied at inference time via `EncodeInput::Dual`.
//!

use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// A single document returned by the reranker, with its position in the input and score.
///
/// `index` maps back to the caller's original document list, so metadata arrays
/// (e.g. IDs, paths) can be reordered without passing them through the reranker.
///
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct RerankedDocument {
    /// Position of this document in the original input `documents` slice.
    pub index: usize,
    /// Relevance score in `[0, 1]`. Higher means more relevant to the query.
    pub score: f32,
    /// The document text.
    pub document: String,
}

/// Metadata for a bundled reranker preset.
///
/// All string fields are owned `String` for FFI compatibility — instances are
/// safe to clone and pass across language boundaries.
///
#[cfg(feature = "reranker-presets")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankerPreset {
    /// Short identifier (catalog name, e.g. `"bge-reranker-base"`).
    pub name: String,
    /// HuggingFace repository name for the model.
    pub model_repo: String,
    /// Path to the ONNX model file within the repo.
    pub model_file: String,
    /// Sibling files that must be downloaded alongside `model_file`.
    ///
    /// Empty for most presets. Used by repos that split the weight blob —
    /// e.g. `rozgo/bge-reranker-v2-m3` ships the model in `model.onnx` plus a
    /// co-located `model.onnx.data` payload.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_files: Vec<String>,
    /// Maximum token sequence length the model supports.
    pub max_length: usize,
    /// Human-readable description of the preset's intended use case.
    pub description: String,
    /// Scoring head for the ONNX model's output tensor.
    ///
    /// Defaults to [`crate::core::config::reranker::RerankerHead::CrossEncoder`]
    /// for all existing presets. Set to `Qwen3Generative` for Qwen3
    /// generative-reranker checkpoints.
    #[serde(default)]
    pub head: crate::core::config::reranker::RerankerHead,
}

pub static RERANKER_PRESETS: LazyLock<Vec<RerankerPreset>> = LazyLock::new(|| {
    vec![
        RerankerPreset {
            name: "bge-reranker-base".to_string(),
            model_repo: "xberg-io/reranker-models".to_string(),
            model_file: "bge-reranker-base/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 512,
            description: "BGE cross-encoder base (~278M params, EN + ZH). Best for: \
                general-purpose RAG, production deployments, English or Chinese documents."
                .to_string(),
            head: crate::core::config::reranker::RerankerHead::CrossEncoder,
        },
        RerankerPreset {
            name: "bge-reranker-v2-m3".to_string(),
            model_repo: "xberg-io/reranker-models".to_string(),
            model_file: "bge-reranker-v2-m3/model.onnx".to_string(),
            additional_files: vec!["bge-reranker-v2-m3/model.onnx.data".to_string()],
            max_length: 8192,
            description: "BGE cross-encoder v2 M3 (568M params, 100+ languages, 8192 max-len). \
                Best for: international documents, mixed-language retrieval. \
                Mirror of the official BAAI model; the weight is split into model.onnx + model.onnx.data."
                .to_string(),
            head: crate::core::config::reranker::RerankerHead::CrossEncoder,
        },
        RerankerPreset {
            name: "jina-reranker-v1-turbo-en".to_string(),
            model_repo: "xberg-io/reranker-models".to_string(),
            model_file: "jina-reranker-v1-turbo-en/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 8192,
            description: "Jina reranker v1 turbo English (~37M params, 8192 max-len). \
                Best for: low-latency reranking, English documents, long-context retrieval."
                .to_string(),
            head: crate::core::config::reranker::RerankerHead::CrossEncoder,
        },
        // NOTE: `jina-reranker-v2-base-multilingual` was REMOVED — its license is
        RerankerPreset {
            name: "qwen3-reranker-0.6b".to_string(),
            model_repo: "xberg-io/reranker-models".to_string(),
            model_file: "qwen3-reranker-0.6b/model.onnx".to_string(),
            additional_files: vec!["qwen3-reranker-0.6b/model.onnx.data".to_string()],
            max_length: 512,
            description: "Qwen3 generative reranker (0.6B params, multilingual). Best for: \
                instruction-aware relevance judgment via a causal-LM yes/no head, higher quality \
                at higher latency than classic cross-encoders."
                .to_string(),
            head: crate::core::config::reranker::RerankerHead::Qwen3Generative,
        },
        RerankerPreset {
            name: "ettin-reranker-150m".to_string(),
            model_repo: "xberg-io/reranker-models".to_string(),
            model_file: "ettin-reranker-150m/model.onnx".to_string(),
            additional_files: Vec::new(),
            max_length: 7999,
            description: "Ettin cross-encoder (150M params, ModernBERT long-context, EN). Best for: \
                high-quality English reranking with long documents at cross-encoder latency."
                .to_string(),
            head: crate::core::config::reranker::RerankerHead::CrossEncoder,
        },
    ]
});

/// Friendly aliases mapped to catalog short-names.
///
/// `(alias, catalog_name)` pairs let callers say `"balanced"` or
/// `"multilingual"` and get a stable, opinionated default that may evolve
/// across releases. The catalog name is the underlying primitive — alias
/// resolution is single-hop (aliases cannot point at other aliases).
///
#[cfg(feature = "reranker-presets")]
const PRESET_ALIASES: &[(&str, &str)] = &[
    ("fast", "jina-reranker-v1-turbo-en"),
    ("balanced", "ettin-reranker-150m"),
    ("quality", "bge-reranker-v2-m3"),
    ("multilingual", "bge-reranker-v2-m3"),
];

/// Get a preset by name (returns an owned clone for FFI compatibility).
///
/// Lookup is case-sensitive. Checks the catalog literals first; falls back
/// to the alias table for the documented friendly names
/// (`fast` / `balanced` / `quality` / `multilingual`).
///
#[cfg(feature = "reranker-presets")]
pub(crate) fn get_preset(name: &str) -> Option<RerankerPreset> {
    if let Some(preset) = RERANKER_PRESETS.iter().find(|p| p.name == name) {
        return Some(preset.clone());
    }
    let resolved = PRESET_ALIASES.iter().find(|(alias, _)| *alias == name)?.1;
    RERANKER_PRESETS.iter().find(|p| p.name == resolved).cloned()
}

/// List all available reranker preset names (owned clones for FFI compatibility).
///
/// Returns the catalog short-names followed by the friendly aliases, so
/// `list_presets()[..5]` is the catalog and `list_presets()[5..]` is aliases.
///
#[cfg(feature = "reranker-presets")]
pub(crate) fn list_presets() -> Vec<String> {
    let mut out: Vec<String> = RERANKER_PRESETS.iter().map(|p| p.name.clone()).collect();
    out.extend(PRESET_ALIASES.iter().map(|(alias, _)| alias.to_string()));
    out
}

/// Apply sigmoid to convert a raw logit to a `[0, 1]` score.
#[cfg(test)]
pub(crate) fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Build the sorted, optionally truncated result vector from raw logits.
///
/// Always applies sigmoid — used by callers (the Plugin backend path) that are
/// guaranteed to hand back raw cross-encoder-style logits. The local ONNX path
/// uses [`build_results_for_head`] instead, since its Qwen3 head already
/// returns a `[0, 1]` probability that must NOT be sigmoided a second time.
#[cfg(test)]
fn build_results(documents: &[String], logits: Vec<f32>, top_k: Option<usize>) -> Vec<RerankedDocument> {
    build_results_from_scores(documents, logits, top_k, sigmoid_f32)
}

/// Build the sorted, optionally truncated result vector from the local ONNX
/// engine's raw output, applying the score transform appropriate to `head`.
///
/// - [`crate::core::config::reranker::RerankerHead::CrossEncoder`] — the engine
///   returns raw logits; sigmoid is applied here to reach `[0, 1]`, exactly as
///   the original single-head implementation did.
/// - [`crate::core::config::reranker::RerankerHead::Qwen3Generative`] — the
///   engine's `qwen3_scores` already performs a softmax internally and returns
///   `P(yes)` in `[0, 1]`. Applying sigmoid again here would double-transform
///   an already-bounded probability, compressing the score distribution toward
///   0.5 and corrupting ranking. This path is a pass-through (identity) instead.
#[cfg(test)]
fn build_results_for_head(
    documents: &[String],
    raw_scores: Vec<f32>,
    top_k: Option<usize>,
    head: crate::core::config::reranker::RerankerHead,
) -> Vec<RerankedDocument> {
    match head {
        crate::core::config::reranker::RerankerHead::CrossEncoder => {
            build_results_from_scores(documents, raw_scores, top_k, sigmoid_f32)
        }
        crate::core::config::reranker::RerankerHead::Qwen3Generative => {
            build_results_from_scores(documents, raw_scores, top_k, |already_a_probability| {
                already_a_probability
            })
        }
    }
}

/// Shared sort/truncate core for both [`build_results`] and [`build_results_for_head`].
///
/// Applies `transform` to each raw value to produce the final `[0, 1]` score,
/// then sorts descending and truncates to `top_k`.
///
/// When `top_k` is smaller than the survivor count, this partitions with
/// [`slice::select_nth_unstable_by`] (`O(n)`) to isolate the top-k elements and
/// sorts only that k-slice (`O(k log k)`), instead of fully sorting all `n`
/// results (`O(n log n)`) before truncating. The returned order is identical
/// to a full sort followed by `truncate(k)` — same descending-by-score
/// ordering, same tie-breaking (`partial_cmp` with `Equal` fallback) applied
/// consistently by both the partition and the final sort.
#[cfg(test)]
fn build_results_from_scores(
    documents: &[String],
    raw_scores: Vec<f32>,
    top_k: Option<usize>,
    transform: impl Fn(f32) -> f32,
) -> Vec<RerankedDocument> {
    let mut results: Vec<RerankedDocument> = documents
        .iter()
        .enumerate()
        .zip(raw_scores.iter())
        .map(|((index, document), &raw)| RerankedDocument {
            index,
            score: transform(raw),
            document: document.clone(),
        })
        .collect();

    let cmp_desc =
        |a: &RerankedDocument, b: &RerankedDocument| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal);

    match top_k {
        Some(k) if k < results.len() => {
            if k > 0 {
                results.select_nth_unstable_by(k - 1, cmp_desc);
            }
            results.truncate(k);
            results.sort_by(cmp_desc);
        }
        _ => {
            results.sort_by(cmp_desc);
            if let Some(k) = top_k {
                results.truncate(k);
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_documents_returns_empty_vec() {
        let results = build_results(&[], vec![], None);
        assert!(results.is_empty());
    }

    #[test]
    fn build_results_sorts_descending_by_score() {
        let documents = vec!["doc0".to_string(), "doc1".to_string(), "doc2".to_string()];
        let logits = vec![-1.0_f32, 2.0_f32, 0.5_f32];
        let results = build_results(&documents, logits, None);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].index, 1);
        assert!(results[0].score > results[1].score, "Results must be sorted descending");
        assert!(results[1].score > results[2].score, "Results must be sorted descending");
    }

    #[test]
    fn top_k_truncation_applies() {
        let documents = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let logits = vec![1.0_f32, 2.0_f32, 0.5_f32, 1.5_f32];
        let results = build_results(&documents, logits, Some(2));

        assert_eq!(results.len(), 2, "top_k=2 should truncate to 2 results");
        assert!(results[0].score >= results[1].score);
    }

    #[test]
    fn top_k_zero_returns_empty() {
        let documents = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let logits = vec![1.0_f32, 2.0_f32, 0.5_f32];
        let results = build_results(&documents, logits, Some(0));
        assert!(results.is_empty(), "top_k=0 must return an empty vec");
    }

    #[test]
    fn top_k_larger_than_docs_returns_all() {
        let documents = vec!["a".to_string(), "b".to_string()];
        let logits = vec![1.0_f32, 0.5_f32];
        let results = build_results(&documents, logits, Some(100));
        assert_eq!(results.len(), 2, "top_k larger than docs.len() should return all docs");
    }

    #[test]
    fn build_results_preserves_document_text() {
        let documents = vec!["hello world".to_string(), "foo bar".to_string()];
        let logits = vec![0.0_f32, 1.0_f32];
        let results = build_results(&documents, logits, None);

        assert_eq!(results[0].document, "foo bar");
        assert_eq!(results[1].document, "hello world");
    }

    #[test]
    fn build_results_for_head_cross_encoder_applies_sigmoid() {
        use crate::core::config::reranker::RerankerHead;

        let documents = vec!["doc0".to_string(), "doc1".to_string()];
        let raw_logits = vec![0.0_f32, 2.0_f32];
        let results = build_results_for_head(&documents, raw_logits.clone(), None, RerankerHead::CrossEncoder);

        let expected = build_results(&documents, raw_logits, None);
        assert_eq!(results.len(), expected.len());
        for (a, b) in results.iter().zip(expected.iter()) {
            assert_eq!(a.index, b.index);
            assert!((a.score - b.score).abs() < 1e-9);
        }
        let doc0 = results.iter().find(|r| r.index == 0).unwrap();
        assert!(
            (doc0.score - 0.5).abs() < 1e-6,
            "cross-encoder head must sigmoid, got {}",
            doc0.score
        );
    }

    #[test]
    fn build_results_for_head_qwen3_does_not_double_sigmoid() {
        use crate::core::config::reranker::RerankerHead;

        let documents = vec!["doc0".to_string(), "doc1".to_string()];
        let already_probabilities = vec![0.9_f32, 0.1_f32];
        let results = build_results_for_head(
            &documents,
            already_probabilities.clone(),
            None,
            RerankerHead::Qwen3Generative,
        );

        let doc0 = results.iter().find(|r| r.index == 0).unwrap();
        let doc1 = results.iter().find(|r| r.index == 1).unwrap();
        assert!(
            (doc0.score - 0.9).abs() < 1e-6,
            "Qwen3 score must pass through unchanged, got {}",
            doc0.score
        );
        assert!(
            (doc1.score - 0.1).abs() < 1e-6,
            "Qwen3 score must pass through unchanged, got {}",
            doc1.score
        );

        let wrongly_double_sigmoided = sigmoid_f32(0.9);
        assert!(
            (doc0.score - wrongly_double_sigmoided).abs() > 0.01,
            "Qwen3 score must NOT be re-sigmoided"
        );

        assert!(doc0.score > doc1.score);
    }

    #[test]
    fn reranked_document_serde_roundtrip() {
        let doc = RerankedDocument {
            index: 3,
            score: 0.87,
            document: "test document".to_string(),
        };
        let json = serde_json::to_string(&doc).unwrap();
        let back: RerankedDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(back.index, doc.index);
        assert!((back.score - doc.score).abs() < 1e-6);
        assert_eq!(back.document, doc.document);
    }

    #[cfg(feature = "reranker-presets")]
    #[test]
    fn preset_list_exposes_catalog_plus_aliases() {
        let presets = list_presets();
        assert_eq!(presets.len(), RERANKER_PRESETS.len() + PRESET_ALIASES.len());
        assert!(presets.iter().any(|n| n == "bge-reranker-base"));
        assert!(presets.iter().any(|n| n == "bge-reranker-v2-m3"));
        assert!(presets.iter().any(|n| n == "jina-reranker-v1-turbo-en"));
        assert!(presets.iter().any(|n| n == "qwen3-reranker-0.6b"));
        assert!(!presets.iter().any(|n| n == "jina-reranker-v2-base-multilingual"));
        for (alias, _) in PRESET_ALIASES {
            assert!(presets.iter().any(|n| n == *alias), "missing alias: {alias}");
        }
    }

    #[cfg(feature = "reranker-presets")]
    #[test]
    fn qwen3_preset_uses_generative_head() {
        use crate::core::config::reranker::RerankerHead;

        let preset = get_preset("qwen3-reranker-0.6b").expect("qwen3 preset must exist");
        assert_eq!(preset.model_repo, "xberg-io/reranker-models");
        assert_eq!(preset.head, RerankerHead::Qwen3Generative);

        for name in ["bge-reranker-base", "bge-reranker-v2-m3", "jina-reranker-v1-turbo-en"] {
            let preset = get_preset(name).expect(name);
            assert_eq!(
                preset.head,
                RerankerHead::CrossEncoder,
                "{name} must remain on the original cross-encoder head"
            );
        }
    }

    #[cfg(feature = "reranker-presets")]
    #[test]
    fn get_preset_case_sensitive() {
        assert!(get_preset("bge-reranker-base").is_some());
        assert!(
            get_preset("BGE-Reranker-Base").is_none(),
            "Preset lookup must be case-sensitive"
        );
        assert!(get_preset("nonexistent").is_none());
    }

    #[cfg(feature = "reranker-presets")]
    #[test]
    fn aliases_resolve_to_catalog_entries() {
        for (alias, catalog_name) in PRESET_ALIASES {
            let preset = get_preset(alias).expect("alias must resolve");
            assert_eq!(
                preset.name, *catalog_name,
                "alias {alias} should resolve to catalog entry {catalog_name}"
            );
        }
    }

    #[cfg(feature = "reranker-presets")]
    #[test]
    fn catalog_paths_are_stable() {
        let by_name = |n: &str| get_preset(n).expect(n);

        let base = by_name("bge-reranker-base");
        assert_eq!(base.model_repo, "xberg-io/reranker-models");
        assert_eq!(base.model_file, "bge-reranker-base/model.onnx");
        assert!(base.additional_files.is_empty());

        let m3 = by_name("bge-reranker-v2-m3");
        assert_eq!(m3.model_repo, "xberg-io/reranker-models");
        assert_eq!(m3.model_file, "bge-reranker-v2-m3/model.onnx");
        assert_eq!(
            m3.additional_files,
            vec!["bge-reranker-v2-m3/model.onnx.data".to_string()]
        );

        let turbo = by_name("jina-reranker-v1-turbo-en");
        assert_eq!(turbo.model_repo, "xberg-io/reranker-models");
        assert_eq!(turbo.model_file, "jina-reranker-v1-turbo-en/model.onnx");

        let qwen3 = by_name("qwen3-reranker-0.6b");
        assert_eq!(qwen3.model_repo, "xberg-io/reranker-models");
        assert_eq!(qwen3.model_file, "qwen3-reranker-0.6b/model.onnx");
        assert_eq!(
            qwen3.additional_files,
            vec!["qwen3-reranker-0.6b/model.onnx.data".to_string()]
        );
        assert_eq!(qwen3.head, crate::core::config::reranker::RerankerHead::Qwen3Generative);
    }
}
