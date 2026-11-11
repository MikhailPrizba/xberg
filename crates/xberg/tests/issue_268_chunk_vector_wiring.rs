//! Wiring tests for #268: sparse (SPLADE) and late-interaction (ColBERT) vectors attached
//! to chunks via `ChunkingConfig::sparse_embedding` / `ChunkingConfig::late_interaction`.
//!
//! `sparse_and_late_interaction_configs_unset_leave_chunks_without_vectors` is a fast,
//! network-free regression guard: it never exercises the ONNX-backed producers, so it
//! must pass regardless of which optional embedding features are compiled in.
//!
//! The `*_matches_direct_producer_call` tests are live: they download real ONNX models
//! from Hugging Face and cross-check that the vector the extraction pipeline attaches to
//! a chunk is bit-for-bit identical to calling the standalone `xberg::embed_sparse` /
//! `xberg::embed_multi_vector` producer directly on the same chunk content. Opt out on
//! offline dev with `XBERG_SKIP_LIVE_HF=1`.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests

mod helpers;

#[cfg(feature = "chunking")]
mod vectors_unset {
    use crate::helpers::extract_bytes_document;
    use xberg::core::config::{ChunkingConfig, ExtractionConfig};

    /// No behaviour change for existing callers (#268): leaving both new config fields
    /// unset must leave every chunk's `sparse_embedding`/`late_interaction` as `None`.
    #[tokio::test]
    async fn sparse_and_late_interaction_configs_unset_leave_chunks_without_vectors() {
        let config = ExtractionConfig {
            chunking: Some(ChunkingConfig {
                max_characters: 50,
                overlap: 10,
                ..Default::default()
            }),
            ..Default::default()
        };

        let text = "This is a long text that should be split into multiple chunks. ".repeat(5);

        let result = extract_bytes_document(text.as_bytes(), "text/plain", &config)
            .await
            .expect("extraction should succeed");
        let chunks = result
            .chunks
            .expect("chunks should be produced when chunking is configured");
        assert!(!chunks.is_empty(), "chunking must produce at least one chunk");

        for (i, chunk) in chunks.iter().enumerate() {
            assert!(
                chunk.sparse_embedding.is_none(),
                "chunk[{i}] must have no sparse_embedding when sparse_embedding config is not set"
            );
            assert!(
                chunk.late_interaction.is_none(),
                "chunk[{i}] must have no late_interaction when late_interaction config is not set"
            );
        }
    }
}

// Gated to the union of its two callers below. Without this, a narrow-feature leg that
// enables neither `sparse-embeddings` nor `late-interaction` (CI's `--no-default-features
// --features "ocr,auto-rotate-tract"` clippy run) sees an uncalled function and fails on
// `-D warnings`. ~keep
