/// UTF-8 validation and safe decoding helpers.
pub mod utf8_validation;

#[cfg(any(feature = "office", feature = "email"))]
/// Windows codepage number to `encoding_rs` encoding mapping.
pub(crate) mod windows_codepage;

#[cfg(feature = "quality")]
/// OCR quality scoring: noise detection, confidence aggregation, and artifact removal.
pub mod quality;

#[cfg(feature = "quality")]
/// String utilities: mojibake repair, encoding detection, safe truncation.
pub mod string_utils;

#[cfg(feature = "quality")]
/// Token-level text reduction pipeline for summarizing or compressing document content.
pub mod token_reduction;

#[cfg(feature = "quality")]
pub mod quality_processor;

#[cfg(feature = "quality")]
pub use quality_processor::QualityProcessor;

#[cfg(feature = "quality")]
pub use token_reduction::{ReductionLevel, TokenReductionConfig};



#[cfg(feature = "redaction")]
pub mod redaction;


#[cfg(feature = "markdown-footnotes")]
/// Markdown footnote and citation parsing.
pub mod markdown_footnotes;
