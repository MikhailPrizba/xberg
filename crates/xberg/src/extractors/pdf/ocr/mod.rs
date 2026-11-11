//! OCR functionality for PDF extraction.
//!
//! Handles text quality evaluation, OCR fallback decision logic, and OCR processing.
//!
//! Split into topical submodules: [`scoring`] (native-text/OCR-output quality evaluation and
//! the skip/fallback gate), [`plausibility`] (language/dictionary-plausibility detection for
//! wrong-mapped text layers, issue #1696), [`rendering`] (PDF page rasterization and
//! image-XObject recovery), [`document`] (bbox/geometry transforms and assembling OCR output
//! into page documents, including merging OCR pages back into a natively-extracted document),
//! and [`pipeline`] (the top-level mixed-OCR and per-page pipeline orchestrators). Items used
//! across submodule boundaries are `pub(super)`; items already reachable from outside `ocr`
//! keep their original visibility and are re-exported here so external call sites are
//! unaffected by the split. Unit tests live in `tests.rs`, `recognition_noise_tests.rs`,
//! `numeric_repair_tests.rs`, and `plausibility_tests.rs`, not inline here.

#[cfg(feature = "ocr-pipeline")]
mod cache_policy;
#[cfg(feature = "ocr-pipeline")]
pub(super) use cache_policy::resolve_ocr_config;

mod document;
#[cfg(feature = "ocr-pipeline")]
mod markdown;
#[cfg(feature = "ocr-pipeline")]
pub(crate) use markdown::strip_embedded_gfm_tables;
mod pipeline;
mod plausibility;
mod rendering;
mod scoring;
#[cfg(feature = "ocr-pipeline")]
pub(crate) mod stage_model_inputs;
#[cfg(feature = "ocr-pipeline")]
pub(crate) mod symbol_noise_lines;

#[cfg(feature = "ocr-pipeline")]
pub(crate) use plausibility::scan_text_plausibility;
#[cfg(feature = "ocr-pipeline")]
pub(crate) use scoring::{OcrGateOutcome, apply_flagged_pages, evaluate_ocr_skip_gate, evaluate_per_page_ocr};

// ~keep The standalone-image extractor builds the same `PageContent.ocr_confidence` summary
// from its own single OCR run (#1568), so these three leave `ocr` rather than staying
// `pub(super)`. They are pure readers/builders with no PDF dependency of their own.

// ~keep GH#1789's numeric-token repair has the same "no PDF dependency" shape as the three
// above: `extractors::image`'s standalone OCR route needs it too (the issue's own reproduction
// runs against a bare PNG, not a PDF), gated the same way `page_ocr_confidence` already is
// there (`all(feature = "ocr", feature = "pdf")`, since this module tree only exists under
// `feature = "pdf"`).

// ~keep These three are reachable only from `extractors::pdf`'s own `#[cfg(test)]` unit tests,
// so a non-test build of this crate never exercises the re-export itself (the underlying items
// are not dead code -- `scoring` uses `NativeTextStats` and `evaluate_native_text_for_ocr`
// internally too, and `OcrFallbackDecision` is `pub`). `#[allow(unused_imports)]` documents that
// on purpose rather than leaving an unexplained warning suppression.
#[cfg(feature = "ocr-pipeline")]
#[allow(unused_imports)]
pub(crate) use scoring::{NativeTextStats, OcrFallbackDecision, evaluate_native_text_for_ocr};

// ~keep Consumed only by `extractors::pdf::layout_runner`'s own tests, which additionally
// require `feature = "layout-detection"`; a `test, ocr, pdf` build without it compiles this
// re-export with no reader. `#[allow(unused_imports)]` documents that on purpose.

#[cfg(feature = "ocr-pipeline")]
pub(crate) use document::{
    boundaries_after_replacements, clean_native_page_layers, merge_ocr_pages_into_internal_document,
    merge_page_segmentation_modes, merge_source_strip_frames, merge_structured_ocr_pages_into_internal_document,
    valid_page_boundary,
};

#[cfg(all(feature = "ocr-pipeline", feature = "pdf"))]
pub(crate) use pipeline::{extract_mixed_ocr_native, failed_run_page_failures, route_level_ocr_page_failures};
// ~keep Must match the definition's own gate in `pipeline.rs`; `#[cfg(test)]` alone left
// this import unresolved on every leg without an OCR feature (GH#1951). The definition is
// dead on an `ocr-pipeline`-only test leg, so the re-export is unused there.
#[cfg(feature = "ocr-pipeline")]
pub(crate) use pipeline::{
    OcrRunResult, PageOcrHints, apply_numeric_repair_to_whole_document_ocr,
    extract_with_ocr_with_page_hints_and_evidence, numeric_repair_enabled, run_ocr_pipeline,
};

#[cfg(feature = "ocr-pipeline")]
pub(crate) use pipeline::estimate_png_encode_page_peak_bytes;

#[cfg(all(test, feature = "ocr-pipeline"))]
mod ligature_space_tests;
#[cfg(all(test, feature = "ocr-pipeline"))]
mod numeric_repair_tests;
#[cfg(all(test, feature = "ocr-pipeline"))]
mod photo_layer_tests;
#[cfg(all(test, feature = "ocr-pipeline"))]
mod plausibility_tests;
#[cfg(all(test, feature = "pdf", feature = "ocr-pipeline"))]
mod proven_read_tests;
#[cfg(all(test, feature = "ocr-pipeline"))]
mod recognition_noise_tests;

#[cfg(all(test, feature = "tokio-runtime", feature = "ocr-pipeline"))]
mod raw_transcript_tests;

#[cfg(all(test, feature = "pdf", feature = "ocr-pipeline"))]
mod combined_rotation_tests;

#[cfg(all(test, feature = "ocr-pipeline"))]
#[allow(unused_imports)]
pub(crate) use pipeline::extract_with_ocr_with_page_hints;
