//! Regression tests for issue #316: the remaining `.split("\n\n")` paragraph
//! boundaries outside the plain-text extractor fixed in #227.
//!
//! #227 fixed `extractors/text.rs`. The same defect — splitting on a bare `"\n\n"`
//! without first normalizing CRLF / lone-CR — survived at every other site whose
//! input reaches it from a producer that does not guarantee LF-only text:
//!
//! * `extractors/email.rs` — RFC 5322 mandates CRLF and `mail-parser` returns the
//!   decoded body verbatim, so *every* plain-text email collapsed into one paragraph.
//! * `extractors/pst.rs` — `PR_BODY` (0x1000) out of an Outlook message store, same shape.
//! * `pdf/structure/adapters.rs`, `extractors/pdf/{mod,ocr}.rs` — OCR backend output,
//!   which is passed through verbatim; the VLM backend returns model markdown straight
//!   out of an HTTP JSON body and third-party plugin backends return arbitrary strings.
//! * `extractors/djot_format/conversion.rs` — a public API over `ExtractedDocument`,
//!   whose `content` carries no LF-only guarantee (`render_plain` copies element text
//!   verbatim and the `pre_rendered_content` path bypasses rendering entirely).
//!
//! Each test asserts the exact paragraph count *and* the exact paragraph contents,
//! so a regression that merely produces "more than one" paragraph still fails.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests

mod helpers;

#[cfg(feature = "email")]
mod email_body {
    use super::helpers::extract_bytes_document;
    use xberg::ExtractionConfig;
    use xberg::types::ResultFormat;

    /// Header block emitted by `EmailExtractor::build_internal_document` before the
    /// body paragraphs. It is a `MetadataBlock`, which maps to `NarrativeText` in the
    /// element-based output, so it is part of the exact expected sequence.
    const EXPECTED_HEADER_BLOCK: &str = "Subject: CRLF Body Test\nFrom: sender@example.com\nTo: recipient@example.com";

    fn element_based_config() -> ExtractionConfig {
        ExtractionConfig {
            result_format: ResultFormat::ElementBased,
            use_cache: false,
            ..Default::default()
        }
    }

    fn narrative_texts(elements: &[xberg::types::Element]) -> Vec<&str> {
        elements
            .iter()
            .filter(|e| e.element_type == xberg::types::ElementType::NarrativeText)
            .map(|e| e.text.as_str())
            .collect()
    }

    #[tokio::test]
    async fn should_split_email_body_into_exact_paragraphs_when_body_uses_crlf() {
        let eml = b"From: sender@example.com\r\n\
To: recipient@example.com\r\n\
Subject: CRLF Body Test\r\n\
\r\n\
First paragraph line one.\r\n\
First paragraph line two.\r\n\
\r\n\
Second paragraph.\r\n\
\r\n\
Third paragraph.";

        let result = extract_bytes_document(eml, "message/rfc822", &element_based_config())
            .await
            .expect("CRLF email extraction should succeed");

        let elements = result.elements.expect("elements output should be populated");

        assert_eq!(
            narrative_texts(&elements),
            vec![
                EXPECTED_HEADER_BLOCK,
                "First paragraph line one.\nFirst paragraph line two.",
                "Second paragraph.",
                "Third paragraph.",
            ],
            "CRLF blank lines in an email body must be recognized as paragraph boundaries, \
             and \\r\\n must normalize to \\n within a paragraph"
        );
    }

    #[tokio::test]
    async fn should_split_email_body_into_exact_paragraphs_when_body_uses_lone_cr() {
        // Headers stay RFC-conformant CRLF; only the body uses classic-Mac lone CR.
        let eml = b"From: sender@example.com\r\n\
To: recipient@example.com\r\n\
Subject: CRLF Body Test\r\n\
\r\n\
Alpha line one.\rAlpha line two.\r\rBeta paragraph.\r\rGamma paragraph.";

        let result = extract_bytes_document(eml, "message/rfc822", &element_based_config())
            .await
            .expect("lone-CR email extraction should succeed");

        let elements = result.elements.expect("elements output should be populated");

        assert_eq!(
            narrative_texts(&elements),
            vec![
                EXPECTED_HEADER_BLOCK,
                "Alpha line one.\nAlpha line two.",
                "Beta paragraph.",
                "Gamma paragraph.",
            ],
            "a lone \\r blank line in an email body must be recognized as a paragraph boundary"
        );
    }

    #[tokio::test]
    async fn should_split_email_body_into_exact_paragraphs_when_body_uses_lf() {
        let eml = b"From: sender@example.com\r\n\
To: recipient@example.com\r\n\
Subject: CRLF Body Test\r\n\
\r\n\
One.\nStill one.\n\nTwo.\n\nThree.";

        let result = extract_bytes_document(eml, "message/rfc822", &element_based_config())
            .await
            .expect("LF email extraction should succeed");

        let elements = result.elements.expect("elements output should be populated");

        assert_eq!(
            narrative_texts(&elements),
            vec![EXPECTED_HEADER_BLOCK, "One.\nStill one.", "Two.", "Three."],
            "plain LF paragraph splitting in an email body must remain unchanged"
        );
    }
}

/// `extraction_result_to_djot` is a public API taking a caller-supplied
/// `ExtractedDocument`. Nothing in the pipeline guarantees `content` is LF-only:
/// `render_plain` copies each element's text verbatim, and extractors that set
/// `pre_rendered_content` bypass rendering altogether.
mod djot_conversion {
    use xberg::extractors::djot_format::extraction_result_to_djot;
    use xberg::types::ExtractedDocument;

    // `ExtractedDocument` has private fields, so a struct literal with `..Default::default()`
    // is rejected from an integration test (E0451). Build the default and assign instead. ~keep
    fn document_with_content(content: &str) -> ExtractedDocument {
        let mut document = ExtractedDocument::default();
        document.content = content.to_string();
        document
    }

    #[test]
    fn should_emit_exact_djot_paragraphs_when_content_uses_crlf() {
        let document = document_with_content("First paragraph.\r\n\r\nSecond paragraph.\r\n\r\nThird paragraph.");

        let djot = extraction_result_to_djot(&document).expect("djot conversion should succeed");

        assert_eq!(
            djot, "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n\n",
            "CRLF blank lines must produce three separate djot paragraphs"
        );
    }

    #[test]
    fn should_emit_exact_djot_paragraphs_when_content_uses_lone_cr() {
        let document = document_with_content("Alpha.\r\rBeta.\r\rGamma.");

        let djot = extraction_result_to_djot(&document).expect("djot conversion should succeed");

        assert_eq!(
            djot, "Alpha.\n\nBeta.\n\nGamma.\n\n",
            "a lone \\r blank line must produce three separate djot paragraphs"
        );
    }

    #[test]
    fn should_emit_exact_djot_paragraphs_when_content_uses_lf() {
        let document = document_with_content("Alpha.\n\nBeta.\n\nGamma.");

        let djot = extraction_result_to_djot(&document).expect("djot conversion should succeed");

        assert_eq!(
            djot, "Alpha.\n\nBeta.\n\nGamma.\n\n",
            "plain LF djot paragraph splitting must remain unchanged"
        );
    }
}
