//! PDF output quality integration tests.
//!
//! Regression tests verifying that extraction output is clean and free of
//! common noise patterns (figure-internal text, arXiv watermarks, reference
//! entries misclassified as headings, repeating conference headers).
//!
//! Benchmark documents:
//! - `docling.pdf` — academic paper with figures, tables, arXiv sidebar
//! - `multi_page.pdf` — clean multi-page document (no noise expected)

#![cfg(feature = "pdf")]

mod helpers;
use helpers::extract_uri_document_blocking;

use helpers::*;
use xberg::core::config::{ExtractionConfig, OutputFormat};

fn extract_markdown(relative_path: &str) -> String {
    let pdf_path = get_test_file_path(relative_path);
    if !pdf_path.exists() {
        panic!("Test document not found: {}", relative_path);
    }
    let config = ExtractionConfig {
        output_format: OutputFormat::Markdown,
        ..Default::default()
    };
    extract_uri_document_blocking(&pdf_path, None, &config)
        .expect("extraction should succeed")
        .content
}

#[test]
fn test_multipage_clean_output() {
    if !test_documents_available() {
        return;
    }
    let content = extract_markdown("pdf/multi_page.pdf");

    assert!(content.contains("Evolution of the Word Processor"), "Title not found");
    assert!(
        content.contains("Pre-Digital Era"),
        "Section 'Pre-Digital Era' not found"
    );
    assert!(content.contains("IBM MT/ST"), "'IBM MT/ST' not found");
}

#[test]
fn test_multipage_no_noise() {
    if !test_documents_available() {
        return;
    }
    let content = extract_markdown("pdf/multi_page.pdf");

    assert!(
        !content.contains("arXiv:"),
        "multipage.pdf should have no arXiv identifiers"
    );
}

/// Regression test for reading-order inversion caused by coordinate system mismatch.
///
/// xberg_native_pdf returns bounding boxes in PDF coordinates (y=0 at bottom of page, y increases
/// upward). Storing these directly in SegmentData.baseline_y and then sorting descending
/// in `assemble_page_elements_with_tables` produces correct top-to-bottom reading order.
///
/// The bug was an erroneous conversion `page_height - bbox.y - bbox.height` that turned
/// PDF coordinates into screen coordinates (y=0 at top). The descending sort then placed
/// bottom-of-page content first, completely reversing the output.
#[test]
fn test_pdf_structure_reading_order() {
    if !test_documents_available() {
        return;
    }
    let content = extract_markdown("vendored/pdfplumber/pdf/pdf_structure.pdf");

    let title_pos = content.find("Titre du document").expect("title not found in output");
    let tableau_pos = content.find("Tableau").expect("'Tableau' section not found in output");
    assert!(
        title_pos < tableau_pos,
        "'Titre du document' (top of page) must appear before 'Tableau' (bottom of page); \
         got title at byte {title_pos}, Tableau at byte {tableau_pos}. \
         This indicates a reading-order inversion — check PDF coordinate handling in \
         crates/xberg/src/pdf/native/hierarchy.rs"
    );
}

/// Regression test for yfedoseev/xberg_native_pdf#847: inter-word spaces were dropped
/// on TJ-positioned runs, so words in this pdfplumber fixture's tabular
/// rate-table header extracted glued (`Comparisonrate`, eight times) while the
/// same words in flowing prose were spaced correctly. xberg_native_pdf 0.3.74 accounts
/// for the `TJ` numeric adjustment that carries the space, so the header text
/// is spaced too.
#[test]
fn test_tj_positioned_runs_keep_interword_spaces() {
    if !test_documents_available() {
        return;
    }
    let path = get_test_file_path("vendored/pdfplumber/pdf/pr-138-example.pdf");
    if !path.exists() {
        return;
    }
    let content = extract_uri_document_blocking(&path, None, &ExtractionConfig::default())
        .expect("extraction should succeed")
        .content;
    assert!(
        !content.contains("Comparisonrate"),
        "TJ-positioned header still glues 'Comparisonrate' — xberg_native_pdf#847 regression"
    );
    assert!(
        content.contains("Comparison rate"),
        "expected the spaced phrase 'Comparison rate' in extracted text"
    );
}

/// Regression test for same-line word gaps being mistaken for kerning-run splits.
#[test]
fn test_681693_same_line_words_keep_spaces() {
    if !test_documents_available() {
        return;
    }
    let path = get_test_file_path("pdf/681693.pdf");
    if !path.exists() {
        return;
    }
    let content = extract_markdown("pdf/681693.pdf");
    assert!(
        content.contains("Your data is clean:"),
        "expected genuine same-line word gaps to remain spaces in 681693.pdf"
    );
    assert!(
        content.contains("See \\#182 for more infos."),
        "expected explicit source whitespace to survive overlapping text spans"
    );
    assert!(
        content.contains("MongoKit is a python module"),
        "expected the MongoKit introduction to retain its word boundary"
    );
    assert!(
        content.contains("## Recent Change Log\n\n### v0.9.1"),
        "expected the changelog section and adjacent semantic version to retain H2/H3 hierarchy"
    );
}
