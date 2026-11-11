//! Regression coverage for native two-column PDF reading order.

#![cfg(feature = "pdf")]

mod helpers;
use helpers::extract_bytes_document_blocking;

use xberg::core::config::{ExtractionConfig, OutputFormat};
// Only the `layout-detection`-gated test builds a `PdfConfig`, so the import carries the same
// gate: a helper's cfg must equal the union of its users' cfgs, or a narrow feature leg fails on
// `unused_imports` under `-D warnings`. ~keep

const ISSUE_1484_PDF: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_documents/pdf/issue-1484-two-column-hanging-number-gutter.pdf"
);

fn make_two_column_pdf() -> Vec<u8> {
    let stream = "\
BT /F1 11 Tf 1 0 0 1 60 712 Tm (The committee reviewed the annual) Tj ET\n\
BT /F1 11 Tf 1 0 0 1 60 698 Tm (report and) Tj ET\n\
BT /F1 11 Tf 1 0 0 1 330 712 Tm (approved the budget for the) Tj ET\n\
BT /F1 11 Tf 1 0 0 1 330 698 Tm (coming fiscal year.) Tj ET\n";
    let mut pdf = Vec::new();

    macro_rules! push {
        ($value:expr) => {
            pdf.extend_from_slice($value.as_bytes())
        };
    }

    push!("%PDF-1.4\n");
    let catalog_offset = pdf.len();
    push!("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    let pages_offset = pdf.len();
    push!("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    let page_offset = pdf.len();
    push!(
        "3 0 obj\n\
         << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n\
         /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n\
         endobj\n"
    );
    let content_offset = pdf.len();
    push!(format!("4 0 obj\n<< /Length {} >>\nstream\n", stream.len()));
    push!(stream);
    push!("endstream\nendobj\n");
    let font_offset = pdf.len();
    push!("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n");
    let xref_offset = pdf.len();
    push!(format!(
        "xref\n0 6\n\
         0000000000 65535 f \r\n\
         {catalog_offset:010} 00000 n \r\n\
         {pages_offset:010} 00000 n \r\n\
         {page_offset:010} 00000 n \r\n\
         {content_offset:010} 00000 n \r\n\
         {font_offset:010} 00000 n \r\n\
         trailer\n<< /Size 6 /Root 1 0 R >>\n\
         startxref\n{xref_offset}\n%%EOF\n"
    ));
    pdf
}

fn normalized_content(config: &ExtractionConfig) -> String {
    extract_bytes_document_blocking(&make_two_column_pdf(), "application/pdf", config)
        .expect("two-column PDF extraction must succeed")
        .content
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn assert_text_appears_in_order(content: &str, expected: &[&str]) {
    let mut offset = 0;
    for text in expected {
        let relative = content[offset..]
            .find(text)
            .unwrap_or_else(|| panic!("expected {text:?} after byte {offset} in {content:?}"));
        offset += relative + text.len();
    }
}

#[test]
fn native_two_column_pdf_uses_column_block_reading_order() {
    let expected = "The committee reviewed the annual report and approved the budget for the coming fiscal year.";

    assert_eq!(normalized_content(&ExtractionConfig::default()), expected);
}

// `pdf_options.reading_order` is rejected by config validation without `layout-detection`
// ("requires the layout-detection feature"), so on a `pdf`-only build this test cannot pass --
// it is the only test in the file that sets the option, which is why the file-level
// `#![cfg(feature = "pdf")]` is not enough for it. ~keep

#[test]
fn markdown_two_column_pdf_uses_column_block_reading_order() {
    let config = ExtractionConfig {
        output_format: OutputFormat::Markdown,
        ..ExtractionConfig::default()
    };
    let expected = "The committee reviewed the annual report and approved the budget for the coming fiscal year.";

    assert_eq!(normalized_content(&config), expected);
}

/// End-to-end corpus coverage for the reported two-page document. The focused
/// span-level regression in `pdf::native::text` proves the gutter-snap path.
#[test]
fn issue_1484_corpus_pages_extract_in_column_block_order() {
    let pdf = std::fs::read(ISSUE_1484_PDF).expect("issue #1484 corpus fixture must exist");
    let content = extract_bytes_document_blocking(&pdf, "application/pdf", &ExtractionConfig::default())
        .expect("issue #1484 PDF extraction must succeed")
        .content;
    let second_page = content
        .match_indices("An agreement which, due to its nature")
        .nth(1)
        .map(|(index, _)| index)
        .expect("fixture must contain its unnumbered control page");

    assert_text_appears_in_order(
        &content[..second_page],
        &[
            "15.3 An agreement",
            "15.4 Client",
            "15.5 Either",
            "16.5 The exclusions",
            "16.6 The exclusions",
            "16.7 Unless",
        ],
    );
    assert_text_appears_in_order(
        &content[second_page..],
        &[
            "An agreement",
            "Client is not entitled",
            "Either party",
            "The exclusions and limitations of supplier liability",
            "The exclusions and limitations referred",
            "Unless performance",
        ],
    );
}
