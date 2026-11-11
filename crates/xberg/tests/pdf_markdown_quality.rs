//! PDF markdown quality smoke tests: verify extraction produces structural elements.
//!
//! These are lightweight assertions — detailed quality scoring and A/B comparisons
//! live in `tools/benchmark-harness` (subcommands: `compare`, `pipeline-benchmark`).
//!
//! Usage:
//!   cargo test -p xberg --features "pdf" \
//!     --test pdf_markdown_quality -- --nocapture

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests
#![cfg(feature = "pdf")]

mod helpers;
use helpers::extract_uri_document_blocking;

use helpers::*;
use xberg::core::config::{ExtractionConfig, OutputFormat};

/// Documents with markdown ground truth.
const MARKDOWN_GT_DOCS: &[(&str, &str)] = &[("docling", "pdf/docling.pdf")];

fn extract_markdown(pdf_path: &std::path::Path) -> String {
    let config = ExtractionConfig {
        output_format: OutputFormat::Markdown,
        ..Default::default()
    };
    extract_uri_document_blocking(pdf_path, None, &config)
        .expect("extraction should succeed")
        .content
}

/// Count structural elements in markdown content.
fn count_headings(md: &str) -> usize {
    md.lines().filter(|l| l.starts_with('#')).count()
}

fn count_table_rows(md: &str) -> usize {
    md.lines()
        .filter(|l| l.starts_with('|') && l.ends_with('|') && !l.contains("---"))
        .count()
}

fn count_list_items(md: &str) -> usize {
    md.lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("- ") || t.starts_with("* ") || t.starts_with("+ ")
        })
        .count()
}

fn has_code_blocks(md: &str) -> bool {
    md.contains("```")
}

#[test]
fn test_baseline_produces_structural_markdown() {
    if !test_documents_available() {
        println!("Skipping: test_documents not available");
        return;
    }

    for &(name, pdf_rel) in MARKDOWN_GT_DOCS {
        let pdf_path = get_test_file_path(pdf_rel);
        if !pdf_path.exists() {
            println!("Skipping {}: file not found", name);
            continue;
        }

        let content = extract_markdown(&pdf_path);

        assert!(
            !content.trim().is_empty(),
            "{}: extraction produced empty content",
            name
        );
        assert!(
            content.len() > 500,
            "{}: content too short ({} chars)",
            name,
            content.len()
        );
        assert!(count_headings(&content) > 0, "{}: expected at least one heading", name);

        println!(
            "{}: {} chars, {} headings, {} table rows, {} list items, code={}",
            name,
            content.len(),
            count_headings(&content),
            count_table_rows(&content),
            count_list_items(&content),
            has_code_blocks(&content),
        );
    }
}
