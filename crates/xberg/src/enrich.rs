//! Active post-extraction enrichment pipeline for an [`ExtractedDocument`].
//!
//! # Design
//!
//! Each enrichment stage is independently optional and feature-gated. Passing
//! `None` for a stage's config field skips that stage entirely. Stages run
//! sequentially so that later stages can see prior results without complex
//! dependency tracking.
//!
//! ## Stage order
//!
//! 1. Classification — operates on the per-page `content` (or the whole document when it
//!    has no pages); writes `page_classifications` and appends `llm_usage`
//! 2. Chunk classification — multi-labels each entry of `ExtractedDocument::chunks`
//!    in place; a no-op when the document has no chunks
//! 3. NER — operates on the full document text (`content`); writes `entities`
//! 4. Captioning — operates on images extracted into `ExtractedDocument::images`; writes
//!    each image's `caption` / `description` and appends `llm_usage`
//!
//! ## Results live on the document
//!
//! Every stage writes its output onto the corresponding [`ExtractedDocument`]
//! field. [`EnrichedResult`] also provides direct access to stage outputs.
//!
//! ## Transcription is not an enrichment stage
//!
//! Transcription turns audio/video *bytes* into text, which makes it extraction-time
//! work — it is implemented by `TranscriptionExtractor` and driven by
//! `ExtractionConfig::transcription`. By the time a document reaches `enrich` the source
//! bytes are gone, so there is nothing here to transcribe.
//!
//! # Example
//!
//! ```ignore
//! use xberg::{ExtractInput, ExtractionConfig, extract, enrich, EnrichmentConfig};
//!
//! # async fn run() -> xberg::Result<()> {
//! let output = extract(ExtractInput::from_uri("document.pdf"), &ExtractionConfig::default()).await?;
//! let extraction = output.results.into_iter().next().expect("one input yields one result");
//! let config = EnrichmentConfig::default();
//! let enriched = enrich(extraction, &config).await?;
//! assert!(enriched.entities.is_none()); // no NER config supplied
//! # Ok(())
//! # }
//! ```

use crate::types::ExtractedDocument;

/// Classification enrichment knob: how to label the document.
#[cfg(feature = "classification-types")]
pub struct ClassificationEnrichmentConfig {
    /// Label set and LLM settings for the classification stage.
    pub config: crate::core::config::PageClassificationConfig,
}

/// Chunk-classification enrichment knob: how to multi-label individual chunks.
///
/// Operates on `ExtractedDocument::chunks` in place — the caller must have
/// already produced chunks (e.g. via `ExtractionConfig::chunking`) for this
/// stage to have any effect; a document with no chunks is a no-op.
#[cfg(feature = "classification-types")]
pub struct ChunkClassificationEnrichmentConfig {
    /// Label-definition set and LLM/batching settings for the chunk-classification stage.
    pub config: crate::core::config::ChunkClassificationConfig,
}

/// Aggregated enrichment configuration.
///
/// Each field is feature-gated and independently optional. Set a field to
/// `Some(...)` to activate the corresponding stage; leave it `None` to skip.
///
/// `EnrichmentConfig::default()` produces a no-op config: all stages skipped,
/// and `enrich` returns an `EnrichedResult` with all enrichment fields `None`.
#[derive(Default)]
pub struct EnrichmentConfig {}

/// Extraction result with optional enrichment layers applied.
///
/// `extraction` is the enriched [`ExtractedDocument`]: every stage writes its output onto
/// the canonical field that already exists for it, so enrichment survives serialization,
/// the REST/API schema, and the language bindings — all of which see the document and
/// nothing else. The fields below mirror those writes for callers that want the stage
/// output directly; they are `None` when the stage was not configured or the feature was
/// compiled out.
///
/// | Stage | Written onto `extraction` |
/// |---|---|
/// | Classification | `page_classifications`, `llm_usage` |
/// | Chunk classification | `chunks[].classification` (in place) |
/// | NER | `entities` |
/// | Captioning | `images[].caption`, `images[].description`, `llm_usage` |
pub struct EnrichedResult {
    /// The extraction result with every configured stage's output applied.
    pub extraction: ExtractedDocument,
}

/// Apply enrichment stages to an extraction result.
///
/// Stages run sequentially: classification → chunk classification → NER → captioning.
/// On any error the partial result is dropped and the error is returned
/// immediately.
///
/// # Example
///
/// ```ignore
/// use xberg::{ExtractInput, ExtractionConfig, extract, enrich, EnrichmentConfig};
///
/// # async fn run() -> xberg::Result<()> {
/// let output = extract(ExtractInput::from_uri("report.pdf"), &ExtractionConfig::default()).await?;
/// let extraction = output.results.into_iter().next().expect("one input yields one result");
///
/// // Skip all stages — identity pass.
/// let enriched = enrich(extraction, &EnrichmentConfig::default()).await?;
/// assert!(enriched.entities.is_none());
/// assert!(enriched.classification.is_none());
/// assert!(enriched.captions.is_none());
/// # Ok(())
/// # }
/// ```
///
/// # Errors
///
/// - [`crate::XbergError::Validation`] when the classification config has an
///   empty label set (propagated from `classify_document_onto`).
/// - [`crate::XbergError::Other`] when the NER or captioning backends fail. A captioning
///   failure still leaves `extraction.images` intact and records the usage for the calls
///   that already succeeded before the error propagates.

#[allow(unused_mut)]
pub async fn enrich(mut extraction: ExtractedDocument, config: &EnrichmentConfig) -> crate::Result<EnrichedResult> {
    // read inside `#[cfg(...)]` branches that are all compiled out — silence

    let _ = config;

    Ok(EnrichedResult { extraction })
}
