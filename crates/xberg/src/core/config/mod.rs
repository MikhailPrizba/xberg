//! Configuration loading and management.
//!
//! This module provides utilities for loading extraction configuration from various
//! sources (TOML, YAML, JSON) and discovering configuration files in the project hierarchy.

pub mod acceleration;
pub mod captioning;
pub mod chunk_classification;
pub mod classification;
pub mod concurrency;
pub mod content_filter;
pub mod csv;
pub mod email;
pub mod extraction;
pub mod formats;
pub mod geojson;
#[cfg(feature = "html")]
pub mod html_output;
pub mod late_interaction;
pub mod layout;
pub mod llm;
pub mod merge;
pub mod ner;
pub mod ocr;
pub mod page;
pub mod pdf;
pub mod processing;
pub mod redaction;
pub mod reranker;
pub mod sparse_embedding;
pub mod summarization;
pub mod transcription;
pub mod translation;
#[cfg(feature = "tree-sitter-types")]
pub mod tree_sitter;

pub use acceleration::{AccelerationConfig, ExecutionProviderType};
pub use concurrency::ConcurrencyConfig;
pub use content_filter::ContentFilterConfig;
pub use csv::CsvConfig;
pub use email::EmailConfig;
pub use extraction::{
    ExtractInput, ExtractInputKind, ExtractInputWithOptions, ExtractionConfig, ExtractionErrorItem, ExtractionResult,
    ExtractionSummary, FileExtractionConfig, ImageExtractionConfig, LanguageDetectionConfig, MimeDetectionPolicy,
    TokenReductionOptions, UrlExtractionConfig, UrlExtractionMode,
};
pub use formats::{JupyterCellRendering, OutputFormat};
pub use geojson::GeoJsonExtractionConfig;
#[cfg(feature = "html")]
pub use html_output::{HtmlOutputConfig, HtmlTheme};
pub use late_interaction::{LateInteractionConfig, LateInteractionModelType};
#[cfg(feature = "layout-types")]
pub use layout::{LayoutDetectionConfig, LayoutDetectionModel, LayoutStrategy, TableModel};
pub use llm::{
    BedrockConfig, CallMode, CredentialProviderConfig, LlmBudgetConfig, LlmCacheConfig, LlmConfig, LlmProviderConfig,
    LlmRateLimitConfig, MergeMode, StructuredExtractionConfig,
};
pub use ocr::{
    DEFAULT_SCANNED_MIN_CONFIDENCE, OcrConfig, OcrPipelineConfig, OcrPipelineStage, OcrQualityThresholds, OcrStrategy,
    VlmFallbackPolicy,
};
// Crate-internal only: the best-effort pipeline selection policy is derived from `OcrConfig`
// and threaded through OCR internals, deliberately not part of the public/binding-facing API (#1341).
#[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
pub(crate) use ocr::OcrPipelineSelection;
pub use page::PageConfig;
#[cfg(feature = "pdf")]
pub use pdf::{HierarchyConfig, PdfBackend, PdfConfig};
pub use processing::{
    ChunkSizing, ChunkerType, ChunkingConfig, EmbeddingConfig, EmbeddingModelType, PostProcessorConfig,
    TableChunkingMode,
};
pub use reranker::{RerankerConfig, RerankerHead, RerankerModelType};
pub use sparse_embedding::{SparseEmbeddingConfig, SparseEmbeddingModelType};
#[cfg(feature = "tree-sitter-types")]
pub use tree_sitter::{CodeContentMode, TreeSitterConfig, TreeSitterProcessConfig};

pub use captioning::{CaptionAltTextMode, CaptioningConfig};
pub use chunk_classification::{ChunkClassificationConfig, ChunkClassificationDefinition};
pub use classification::PageClassificationConfig;
pub use ner::{NerBackendKind, NerConfig};
pub use redaction::{
    ExternalRedactionFinding, RedactionConfig, RedactionOffsetEncoding, RedactionPattern, RedactionTerm,
};
pub use summarization::SummarizationConfig;
pub use translation::TranslationConfig;

#[cfg(feature = "transcription-types")]
pub use transcription::{TranscriptionConfig, WhisperModel};
