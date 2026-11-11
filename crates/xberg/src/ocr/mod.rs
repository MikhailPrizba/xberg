//! Shared OCR types for host-registered backends.
pub mod error;
pub mod types;
pub use error::OcrError;
pub use types::{BatchItemResult, ExtractionResult, PSMMode, Table, TesseractConfig};
