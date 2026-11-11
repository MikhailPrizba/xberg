//! Ground truth-based PDF table detection and markdown quality tests.
//!
//! These tests establish baselines for table detection and markdown output quality.
//! Run after each substantial change to measure improvement or regression.
//!
//! Usage:
//!   # Non-OCR tests (fast, native path):
//!   cargo test -p xberg --features "pdf" --test pdf_table_ground_truth -- --nocapture
//!
//!   # Full tests including table detection (needs ocr feature for HocrWord):
//!   cargo test -p xberg --features "pdf,ocr" --test pdf_table_ground_truth -- --nocapture
//!
//!   # Comprehensive baseline snapshot:
//!   cargo test -p xberg --features "pdf,ocr" --test pdf_table_ground_truth -- --ignored --nocapture

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests
#![cfg(feature = "pdf")]

mod helpers;
use helpers::extract_uri_document_blocking;

use helpers::*;
use xberg::core::config::{ExtractionConfig, OutputFormat};

/// Compute word-level Jaccard similarity between two strings.
fn word_similarity(a: &str, b: &str) -> f64 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() && words_b.is_empty() {
        return 1.0;
    }
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    intersection as f64 / union as f64
}

/// Extract markdown from a PDF file (native path, no OCR).
fn extract_markdown(relative_path: &str) -> Option<xberg::types::ExtractedDocument> {
    let path = get_test_file_path(relative_path);
    if !path.exists() {
        return None;
    }

    let config = ExtractionConfig {
        output_format: OutputFormat::Markdown,
        ..Default::default()
    };

    extract_uri_document_blocking(&path, None, &config).ok()
}

#[test]
fn test_markdown_quality_fake_memo() {
    if skip_if_missing("pdf/fake_memo.pdf") {
        return;
    }

    let result = extract_markdown("pdf/fake_memo.pdf").expect("extraction should succeed");

    println!("=== fake_memo.pdf markdown quality ===");
    println!("Content length: {} chars", result.content.len());

    assert!(
        result.content.len() > 100,
        "fake_memo.pdf should produce >100 chars of markdown (got {})",
        result.content.len()
    );
}

#[test]
fn test_markdown_quality_simple() {
    if skip_if_missing("pdf/simple.pdf") {
        return;
    }

    let result = extract_markdown("pdf/simple.pdf").expect("extraction should succeed");

    println!("=== simple.pdf markdown quality ===");
    println!("Content length: {} chars", result.content.len());

    assert!(
        result.content.len() > 1000,
        "simple.pdf should produce >1000 chars of markdown (got {})",
        result.content.len()
    );
}

#[test]
fn test_markdown_quality_multi_page() {
    if skip_if_missing("pdf/multi_page.pdf") {
        return;
    }

    let result = extract_markdown("pdf/multi_page.pdf").expect("extraction should succeed");

    println!("=== multi_page.pdf markdown quality ===");
    println!("Content length: {} chars", result.content.len());

    assert!(
        result.content.len() > 5000,
        "multi_page.pdf should produce >5000 chars (got {})",
        result.content.len()
    );
}

#[test]
fn test_markdown_quality_vs_ground_truth_simple() {
    if skip_if_missing("pdf/table_document.pdf") {
        return;
    }

    let gt_path = get_test_file_path("ground_truth/pdf/pdf_tables.txt");
    if !gt_path.exists() {
        println!("Skipping: ground truth file not found");
        return;
    }

    let ground_truth = std::fs::read_to_string(&gt_path).expect("should read ground truth");
    let result = extract_markdown("pdf/table_document.pdf").expect("extraction should succeed");

    let similarity = word_similarity(&result.content, &ground_truth);

    println!("=== table_document.pdf vs ground truth ===");
    println!("Extraction length: {} chars", result.content.len());
    println!("Ground truth length: {} chars", ground_truth.len());
    println!("Word similarity: {:.1}%", similarity * 100.0);

    println!("NOTE: table_document.pdf is image-only; low similarity expected without OCR.");
}

#[test]
#[ignore]
fn test_comprehensive_table_detection_baseline() {
    if !test_documents_available() {
        println!("Skipping: test_documents not available");
        return;
    }

    let image_table_pdfs = [
        "table_document.pdf",
        "multi_page_tables.pdf",
        "embedded_images_tables.pdf",
    ];

    let text_table_pdfs = [
        "multi_page.pdf",
        "medium.pdf",
        "large.pdf",
        "a_comparison_of_programming_languages_in_economics_16_jun_2014.pdf",
        "tiny.pdf",
        "tatr.pdf",
    ];

    let non_table_pdfs = [
        "simple.pdf",
        "fake_memo.pdf",
        "google_doc_document.pdf",
        "searchable.pdf",
        "test_article.pdf",
        "code_and_formula.pdf",
    ];

    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║     Table Detection Baseline Snapshot             ║");
    println!("╚══════════════════════════════════════════════════╝\n");

    println!("--- Image-Only Table PDFs (need OCR) ---");
    for pdf in &image_table_pdfs {
        let rel = format!("pdf/{}", pdf);
        if skip_if_missing(&rel) {
            continue;
        }
        match extract_markdown(&rel) {
            Some(result) => {
                let table_count = result.tables.len();
                let status = if result.content.len() < 50 { "IMG" } else { "OK" };
                println!(
                    "  [{:4}] {:<55} tables={} md_len={}",
                    status,
                    pdf,
                    table_count,
                    result.content.len()
                );
            }
            None => println!("  [ERR ] {}", pdf),
        }
    }

    println!("\n--- Text-Based PDFs Expected to Have Tables ---");
    let mut true_positives = 0;
    let mut false_negatives = 0;
    for pdf in &text_table_pdfs {
        let rel = format!("pdf/{}", pdf);
        if skip_if_missing(&rel) {
            continue;
        }
        match extract_markdown(&rel) {
            Some(result) => {
                let table_count = result.tables.len();
                let status = if table_count > 0 {
                    true_positives += 1;
                    "OK"
                } else {
                    false_negatives += 1;
                    "MISS"
                };
                println!(
                    "  [{:4}] {:<55} tables={} md_len={}",
                    status,
                    pdf,
                    table_count,
                    result.content.len()
                );
            }
            None => println!("  [ERR ] {}", pdf),
        }
    }

    println!("\n--- Expected Non-Table PDFs ---");
    let mut true_negatives = 0;
    let mut false_positives = 0;
    for pdf in &non_table_pdfs {
        let rel = format!("pdf/{}", pdf);
        if skip_if_missing(&rel) {
            continue;
        }
        match extract_markdown(&rel) {
            Some(result) => {
                let table_count = result.tables.len();
                let status = if table_count == 0 {
                    true_negatives += 1;
                    "OK"
                } else {
                    false_positives += 1;
                    "FP"
                };
                println!(
                    "  [{:4}] {:<55} tables={} md_len={}",
                    status,
                    pdf,
                    table_count,
                    result.content.len()
                );
            }
            None => println!("  [ERR ] {}", pdf),
        }
    }

    println!("\n--- Summary ---");
    println!("True positives:  {}", true_positives);
    println!("False negatives: {}", false_negatives);
    println!("True negatives:  {}", true_negatives);
    println!("False positives: {}", false_positives);

    let precision = if true_positives + false_positives > 0 {
        true_positives as f64 / (true_positives + false_positives) as f64
    } else {
        0.0
    };
    let recall = if true_positives + false_negatives > 0 {
        true_positives as f64 / (true_positives + false_negatives) as f64
    } else {
        0.0
    };

    println!("Precision: {:.1}%", precision * 100.0);
    println!("Recall:    {:.1}%", recall * 100.0);
}
