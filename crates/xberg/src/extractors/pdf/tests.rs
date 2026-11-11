use super::*;

#[cfg(all(feature = "pdf", feature = "tokio-runtime", not(target_arch = "wasm32")))]
#[tokio::test(flavor = "current_thread")]
async fn native_pdf_blocking_phase_does_not_starve_the_timeout_driver() {
    let work = spawn_pdf_blocking(|| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        Ok::<_, crate::XbergError>(())
    });

    let result = tokio::time::timeout(std::time::Duration::from_millis(10), work).await;

    assert!(
        result.is_err(),
        "the timeout must fire while native PDF work is still running"
    );
}

#[cfg(feature = "pdf")]
#[test]
fn native_pdf_blocking_phase_stops_when_cancelled() {
    let token = crate::cancellation::CancellationToken::new();
    token.cancel();
    let config = ExtractionConfig {
        cancel_token: Some(token),
        ..Default::default()
    };

    let error = ensure_pdf_not_cancelled(&config).expect_err("cancelled native PDF work must stop");

    assert!(matches!(error, crate::XbergError::Cancelled));
}

#[cfg(feature = "pdf")]
#[test]
fn should_suppress_extracted_pdf_metadata_when_disabled() {
    let mut metadata = Metadata {
        output_format: Some("markdown".to_string()),
        ..Default::default()
    };
    let extracted = crate::pdf::metadata::PdfExtractionMetadata {
        title: Some("Document title".to_string()),
        subject: Some("Document subject".to_string()),
        authors: Some(vec!["Author".to_string()]),
        keywords: Some(vec!["keyword".to_string()]),
        created_at: Some("2026-01-01".to_string()),
        modified_at: Some("2026-01-02".to_string()),
        created_by: Some("Producer".to_string()),
        pdf_specific: crate::pdf::metadata::PdfMetadata {
            page_count: Some(2),
            ..Default::default()
        },
        page_structure: Some(crate::types::PageStructure {
            total_count: 2,
            unit_type: crate::types::PageUnitType::Page,
            boundaries: None,
            pages: None,
        }),
    };

    apply_extracted_pdf_metadata(&mut metadata, extracted, false);

    assert_eq!(metadata.output_format.as_deref(), Some("markdown"));
    assert!(metadata.title.is_none());
    assert!(metadata.subject.is_none());
    assert!(metadata.authors.is_none());
    assert!(metadata.keywords.is_none());
    assert!(metadata.created_at.is_none());
    assert!(metadata.modified_at.is_none());
    assert!(metadata.created_by.is_none());
    assert!(metadata.pages.is_none());
    assert!(metadata.format.is_none());
}

#[cfg(feature = "pdf")]
#[tokio::test]
async fn should_validate_direct_pdf_config_before_opening_document() {
    let config = ExtractionConfig {
        pdf_options: Some(crate::core::config::PdfConfig {
            top_margin_fraction: Some(f32::NAN),
            ..Default::default()
        }),
        ..Default::default()
    };

    let error = PdfExtractor::new()
        .extract_content(b"not a PDF", "application/pdf", &config)
        .await
        .expect_err("invalid direct PDF config must fail before parsing");

    assert!(matches!(error, crate::XbergError::Validation { .. }));
    assert!(error.to_string().contains("top_margin_fraction"));
}

fn coverage_native_text() -> String {
    (0..MIN_NATIVE_TOKENS_FOR_STRUCTURE_COVERAGE)
        .map(|index| format!("token{index}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `ExtractedDocument.pages` is derived from `element.page`, so a flat document with no
/// page tags yields `pages: None` -- which a caller cannot distinguish from "this
/// document has no pages". Reported against a 16-page scanned PDF where native mode
/// returned 16 pages and enabling OCR returned `None`, so `list(doc.pages)` raised
/// TypeError on the very mode that exists to give per-page OCR routing.
#[test]
fn should_tag_flat_document_paragraphs_with_their_page_numbers() {
    let text = "Page one body.\n\nPage two body.\n\nPage three body.";
    let first = text.find("Page two").expect("page two");
    let second = text.find("Page three").expect("page three");
    let boundaries = vec![
        crate::types::PageBoundary {
            byte_start: 0,
            byte_end: first,
            page_number: 1,
        },
        crate::types::PageBoundary {
            byte_start: first,
            byte_end: second,
            page_number: 2,
        },
        crate::types::PageBoundary {
            byte_start: second,
            byte_end: text.len(),
            page_number: 3,
        },
    ];

    let doc = flat_pdf_document(text, "application/pdf", Some(&boundaries));

    assert_eq!(doc.elements.len(), 3, "one paragraph per page");
    assert_eq!(
        doc.elements.iter().map(|e| e.page).collect::<Vec<_>>(),
        vec![Some(1), Some(2), Some(3)],
        "every paragraph must carry the page its content is on"
    );
}

/// Without boundaries the pre-existing untagged behaviour stands, so a caller that has
/// no page information is no worse off than before.
#[test]
fn should_leave_flat_document_untagged_when_no_boundaries_are_available() {
    let doc = flat_pdf_document("Only body.\n\nSecond para.", "application/pdf", None);

    assert_eq!(doc.elements.len(), 2);
    assert!(
        doc.elements.iter().all(|e| e.page.is_none()),
        "no boundaries means no page attribution, not a guessed one"
    );
}

#[test]
fn should_fall_back_to_native_text_when_structure_drops_substantial_content() {
    let native_text = coverage_native_text();
    let represented = native_text.split_whitespace().take(13).collect::<Vec<_>>().join(" ");
    let mut structured = InternalDocument::new("pdf");
    structured.push_element(InternalElement::text(ElementKind::Heading { level: 1 }, represented, 0));

    let (selected, is_structured) = select_native_pdf_document(&native_text, "application/pdf", Some(structured), None);

    assert!(!is_structured);
    assert_eq!(selected.elements.len(), 1);
    assert_eq!(selected.elements[0].text, native_text);
}

#[test]
fn should_keep_structure_when_native_token_coverage_meets_floor() {
    let native_text = coverage_native_text();
    let represented = native_text.split_whitespace().take(14).collect::<Vec<_>>().join(" ");
    let mut structured = InternalDocument::new("pdf");
    structured.push_element(InternalElement::text(ElementKind::Heading { level: 1 }, represented, 0));

    let (selected, is_structured) = select_native_pdf_document(&native_text, "application/pdf", Some(structured), None);

    assert!(is_structured);
    assert!(matches!(selected.elements[0].kind, ElementKind::Heading { level: 1 }));
}

#[test]
fn should_count_table_cells_as_structured_native_text() {
    let native_text = coverage_native_text();
    let mut structured = InternalDocument::new("pdf");
    structured.tables.push(crate::types::Table {
        cells: vec![native_text.split_whitespace().map(str::to_string).collect()],
        ..Default::default()
    });

    let (_, is_structured) = select_native_pdf_document(&native_text, "application/pdf", Some(structured), None);

    assert!(is_structured);
}

#[test]
fn should_not_apply_structure_coverage_gate_to_trivial_native_text() {
    let mut structured = InternalDocument::new("pdf");
    structured.push_element(InternalElement::text(ElementKind::Heading { level: 1 }, "Title", 0));

    let (selected, is_structured) =
        select_native_pdf_document("Title and body", "application/pdf", Some(structured), None);

    assert!(is_structured);
    assert!(matches!(selected.elements[0].kind, ElementKind::Heading { level: 1 }));
}

#[cfg(feature = "pdf")]
fn catalog(
    entries: impl IntoIterator<Item = (&'static str, xberg_native_pdf::object::Object)>,
) -> xberg_native_pdf::object::Object {
    xberg_native_pdf::object::Object::Dictionary(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    )
}

#[cfg(feature = "pdf")]
#[test]
fn lopdf_compatibility_pass_skips_pdf_without_relevant_signals() {
    let catalog = catalog([]);

    assert!(!raw_pdf_needs_lopdf_compatibility_pass(
        b"%PDF-1.7\nordinary content\n%%EOF",
    ));
    assert!(!catalog_needs_lopdf_compatibility_pass(Some(&catalog)));
}

#[cfg(feature = "pdf")]
#[test]
fn lopdf_compatibility_pass_keeps_raw_pdf_signals() {
    let catalog = catalog([]);

    assert!(raw_pdf_needs_lopdf_compatibility_pass(
        b"%PDF-1.7\ntrailer << /Prev 42 >>",
    ));
    assert!(raw_pdf_needs_lopdf_compatibility_pass(
        b"%PDF-1.7\n<< /Outlines 7 0 R >>",
    ));
    assert!(!catalog_needs_lopdf_compatibility_pass(Some(&catalog)));
}

#[cfg(feature = "pdf")]
#[test]
fn lopdf_compatibility_pass_keeps_parsed_catalog_outline() {
    let catalog = catalog([("Outlines", xberg_native_pdf::object::Object::Null)]);

    assert!(catalog_needs_lopdf_compatibility_pass(Some(&catalog)));
}

#[cfg(feature = "pdf")]
#[test]
fn lopdf_compatibility_pass_fails_open_without_catalog() {
    assert!(catalog_needs_lopdf_compatibility_pass(None));
}

#[cfg(feature = "pdf")]
fn pdf_test_document(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../test_documents/pdf/{name}"))
}

#[cfg(feature = "pdf")]
fn extraction_method(result: &crate::types::ExtractedDocument) -> Option<ExtractionMethod> {
    result.extraction_method
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn source_strip_frames_survive_plain_and_structured_document_selection() {
    let key = "ocr_source_strip_original_frames";
    let frame = serde_json::json!([{
        "page_number": 1, "rendered_width": 450, "rendered_height": 12000,
        "processed_width": 154, "processed_height": 4096,
    }]);
    let mut page = InternalDocument::new("pdf");
    page.push_element(InternalElement::text(ElementKind::Paragraph, "accepted OCR text", 0));
    page.metadata.additional.insert(key.into(), frame.clone());
    let pages = ahash::AHashMap::from([(1, page)]);
    let replacements = ahash::AHashMap::from([(1, "accepted OCR text".to_owned())]);
    let boundaries = [crate::types::PageBoundary {
        page_number: 1,
        byte_start: 0,
        byte_end: "accepted OCR text".len(),
    }];
    for format in [
        crate::core::config::OutputFormat::Plain,
        crate::core::config::OutputFormat::Markdown,
    ] {
        let (document, _, _) = select_pdf_document(
            ExtractionMethod::Mixed,
            "accepted OCR text",
            "application/pdf",
            None,
            None,
            Some(&replacements),
            Some(&pages),
            Some(&boundaries),
            &format,
        );
        assert_eq!(document.metadata.additional.get(key), Some(&frame));
        assert_eq!(
            document
                .metadata
                .additional
                .get(crate::ocr_metadata_keys::OCR_SOURCE_STRIP_ORIGINAL_FRAME_METADATA_KEY),
            Some(&frame[0]),
        );
        assert_eq!(crate::rendering::render_plain(&document).trim(), "accepted OCR text");
    }
    let (native, _, _) = select_pdf_document(
        ExtractionMethod::Native,
        "native text",
        "application/pdf",
        None,
        None,
        Some(&replacements),
        Some(&pages),
        None,
        &crate::core::config::OutputFormat::Plain,
    );
    assert!(!native.metadata.additional.contains_key(key));
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn should_preserve_tables_without_injecting_text_when_plain_mixed_skips_structure_merge() {
    let table_markdown = "| hidden OCR table |";
    let mut page = InternalDocument::new("pdf");
    page.tables.push(crate::types::Table {
        markdown: table_markdown.to_string(),
        page_number: 2,
        ..Default::default()
    });
    let structured_pages = ahash::AHashMap::from([(2, page)]);
    let replacements = ahash::AHashMap::from([(2, "accepted OCR text".to_string())]);

    let (mut plain, _, _) = select_pdf_document(
        ExtractionMethod::Mixed,
        "accepted OCR text",
        "application/pdf",
        None,
        None,
        Some(&replacements),
        Some(&structured_pages),
        None,
        &crate::core::config::OutputFormat::Plain,
    );
    assert!(plain.tables.is_empty());

    let tables = accepted_mixed_ocr_tables(&structured_pages);
    attach_unrepresented_tables(&mut plain, tables);
    inject_unrepresented_table_elements(&mut plain, false);

    assert_eq!(plain.tables.len(), 1);
    assert_eq!(plain.tables[0].page_number, 2);
    assert_eq!(plain.tables[0].markdown, table_markdown);
    assert!(!crate::rendering::render_plain(&plain).contains(table_markdown));
}

/// A single page with a non-origin, negative-origin `MediaBox [10 -100 622 692]` (GH#1653)
/// and two font sizes -- a 24pt heading line and a 10pt body paragraph, both at known raw
/// user-space positions -- so hierarchy clustering assigns a heading level to the first and
/// leaves the second as body text. `rotate` optionally sets `/Rotate` on the page (GH#1654).
#[cfg(feature = "pdf")]
fn coordinate_frame_test_pdf(rotate: Option<i32>) -> Vec<u8> {
    use lopdf::content::{Content, Operation};
    use lopdf::{Document, Object, Stream, dictionary};

    let mut document = Document::with_version("1.5");
    let pages_id = document.new_object_id();
    let font_id = document.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });

    let content = Content {
        operations: vec![
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 24.into()]),
            Operation::new("Td", vec![82.into(), 500.into()]),
            Operation::new("Tj", vec![Object::string_literal("Coordinate Frame Heading")]),
            Operation::new("ET", vec![]),
            Operation::new("BT", vec![]),
            Operation::new("Tf", vec!["F1".into(), 10.into()]),
            Operation::new("Td", vec![82.into(), 460.into()]),
            Operation::new(
                "Tj",
                vec![Object::string_literal(
                    "This is a body paragraph with enough ordinary words to cluster as body text below the heading.",
                )],
            ),
            Operation::new("ET", vec![]),
        ],
    };
    let content_id = document.add_object(Stream::new(
        dictionary! {},
        content.encode().expect("coordinate frame fixture content must encode"),
    ));

    let mut page_dict = dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => dictionary! { "Font" => dictionary! { "F1" => font_id } },
        "MediaBox" => vec![10.into(), (-100).into(), 622.into(), 692.into()],
    };
    if let Some(rotate) = rotate {
        page_dict.set("Rotate", rotate);
    }
    let page_id = document.add_object(page_dict);

    document.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );
    let catalog_id = document.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    document.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    document
        .save_to(&mut bytes)
        .expect("coordinate frame fixture PDF must serialize");
    bytes
}

#[cfg(feature = "pdf")]
fn coordinate_frame_extraction_config() -> ExtractionConfig {
    use crate::core::config::{HierarchyConfig, PdfConfig};

    ExtractionConfig {
        pdf_options: Some(PdfConfig {
            hierarchy: Some(HierarchyConfig {
                enabled: true,
                ..HierarchyConfig::default()
            }),
            ..PdfConfig::default()
        }),
        ..ExtractionConfig::default()
    }
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn pdf_page_coordinate_frame_reports_raw_media_box_origin_and_unswapped_extent() {
    let content = coordinate_frame_test_pdf(None);
    let config = coordinate_frame_extraction_config();
    let extractor = PdfExtractor::new();
    let result = extractor
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("coordinate frame fixture must extract");

    let frames = result
        .metadata
        .additional
        .get("pdf_page_coordinate_frames")
        .expect("pdf_page_coordinate_frames must be present when a page has hierarchy blocks")
        .as_array()
        .expect("pdf_page_coordinate_frames must be a JSON array");
    assert_eq!(frames.len(), 1, "exactly one page has hierarchy blocks");

    let frame = &frames[0];
    assert_eq!(frame["page_number"], serde_json::json!(1));
    assert_eq!(frame["origin_x"], serde_json::json!(10.0));
    assert_eq!(frame["origin_y"], serde_json::json!(-100.0));
    assert_eq!(frame["width"], serde_json::json!(612.0));
    assert_eq!(frame["height"], serde_json::json!(792.0));
    assert_eq!(frame["unit"], serde_json::json!("point"));
    assert_eq!(frame["origin"], serde_json::json!("bottom_left"));
    assert_eq!(frame["clockwise_rotation"], serde_json::json!(0));
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn pdf_page_coordinate_frame_reports_rotation_without_swapping_extent() {
    let content = coordinate_frame_test_pdf(Some(90));
    let config = coordinate_frame_extraction_config();
    let extractor = PdfExtractor::new();
    let result = extractor
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("rotated coordinate frame fixture must extract");

    let frames = result
        .metadata
        .additional
        .get("pdf_page_coordinate_frames")
        .expect("pdf_page_coordinate_frames must be present when a page has hierarchy blocks")
        .as_array()
        .expect("pdf_page_coordinate_frames must be a JSON array");
    assert_eq!(frames.len(), 1);

    let frame = &frames[0];
    assert_eq!(frame["clockwise_rotation"], serde_json::json!(90));
    // Deliberately un-swapped: this describes raw PDF user space, not the displayed frame.
    assert_eq!(frame["width"], serde_json::json!(612.0));
    assert_eq!(frame["height"], serde_json::json!(792.0));
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn pdf_page_coordinate_frame_omits_page_with_malformed_rotation() {
    let content = coordinate_frame_test_pdf(Some(135));
    let config = coordinate_frame_extraction_config();
    let extractor = PdfExtractor::new();
    let result = extractor
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("malformed-rotation fixture must still extract successfully");

    assert!(
        !result.elements.is_empty(),
        "extraction must otherwise be unaffected by the malformed /Rotate"
    );

    match result.metadata.additional.get("pdf_page_coordinate_frames") {
        None => {}
        Some(value) => {
            let frames = value.as_array().expect("pdf_page_coordinate_frames must be an array");
            assert!(
                frames.is_empty(),
                "page with malformed /Rotate must be omitted entirely, got {frames:?}"
            );
        }
    }
}

#[test]
fn flat_plain_text_retains_table_asset_without_duplicate_rendering() {
    const TABLE_TEXT: &str = "Account balance 42";

    let mut doc = InternalDocument::new("pdf");
    doc.push_element(InternalElement::text(ElementKind::Paragraph, TABLE_TEXT, 0));
    let table = crate::types::Table {
        cells: vec![vec!["Account balance".to_string(), "42".to_string()]],
        markdown: "| Account balance | 42 |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, false);

    assert_eq!(doc.tables.len(), 1, "table assets must remain available to callers");
    assert!(
        !doc.elements
            .iter()
            .any(|element| matches!(element.kind, ElementKind::Table { .. })),
        "flat plain text must not append a second renderable copy of native table text"
    );

    let result =
        crate::extraction::derive::derive_extraction_result(doc, true, crate::core::config::OutputFormat::Plain);
    assert_eq!(result.content.matches(TABLE_TEXT).count(), 1);
    assert_eq!(result.tables.len(), 1);
}

#[test]
fn table_element_injection_remains_available_for_structured_output() {
    let mut doc = InternalDocument::new("pdf");
    let table = crate::types::Table {
        cells: vec![vec!["Heading".to_string(), "Value".to_string()]],
        markdown: "| Heading | Value |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, true);

    assert_eq!(doc.tables.len(), 1);
    assert_eq!(
        doc.elements
            .iter()
            .filter(|element| matches!(element.kind, ElementKind::Table { .. }))
            .count(),
        1,
        "structured and OCR paths must still be able to render attached table assets"
    );
}

/// pdfa_031: the structure-coverage safeguard falls back to `flat_pdf_document`,
/// which carries no `Table` element, so a table reconstructed from prose that is
/// still in the flat text used to be rendered a second time.
#[test]
fn structured_output_skips_tables_already_present_in_the_element_stream() {
    let mut doc = InternalDocument::new("pdf");
    doc.push_element(InternalElement::text(
        ElementKind::Paragraph,
        "Persons committed to the custody of a sheriff shall be confined\nin the facilities designated by law.",
        0,
    ));
    let table = crate::types::Table {
        cells: vec![
            vec![
                "Persons committed to the custody".to_string(),
                "of a sheriff".to_string(),
            ],
            vec![
                "shall be confined in the facilities".to_string(),
                "designated by law".to_string(),
            ],
        ],
        markdown: "| Persons committed to the custody | of a sheriff |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, true);

    assert_eq!(doc.tables.len(), 1, "table assets must remain available to callers");
    assert!(
        !doc.elements
            .iter()
            .any(|element| matches!(element.kind, ElementKind::Table { .. })),
        "a table whose cells are already rendered must not be injected a second time"
    );
}

#[test]
fn structured_output_still_injects_tables_missing_from_the_element_stream() {
    let mut doc = InternalDocument::new("pdf");
    doc.push_element(InternalElement::text(
        ElementKind::Paragraph,
        "Narrative prose that shares none of the tabulated content.",
        0,
    ));
    let table = crate::types::Table {
        cells: vec![
            vec!["Region".to_string(), "Revenue".to_string(), "Growth".to_string()],
            vec!["North".to_string(), "1200".to_string(), "4 percent".to_string()],
            vec!["South".to_string(), "980".to_string(), "7 percent".to_string()],
        ],
        markdown: "| Region | Revenue | Growth |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, true);

    assert_eq!(
        doc.elements
            .iter()
            .filter(|element| matches!(element.kind, ElementKind::Table { .. }))
            .count(),
        1,
        "dropped table content must still be recovered by injection"
    );
}

/// The containment check abstains below `MIN_TABLE_TOKENS_FOR_CONTAINMENT`, so a
/// short table cannot be suppressed by an incidental token overlap.
#[test]
fn short_tables_are_injected_even_when_their_tokens_appear_in_the_text() {
    let mut doc = InternalDocument::new("pdf");
    doc.push_element(InternalElement::text(ElementKind::Paragraph, "total 42 balance", 0));
    let table = crate::types::Table {
        cells: vec![vec!["Total".to_string(), "42".to_string()]],
        markdown: "| Total | 42 |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, true);

    assert_eq!(
        doc.elements
            .iter()
            .filter(|element| matches!(element.kind, ElementKind::Table { .. }))
            .count(),
        1
    );
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn ocr_tables_replace_native_tables_and_are_sorted() {
    let table = |markdown: &str, page_number| crate::types::Table {
        cells: Vec::new(),
        markdown: markdown.to_string(),
        page_number,
        bounding_box: None,
        ..Default::default()
    };
    let mut tables = vec![table("native", 1)];

    replace_tables_with_ocr_output(&mut tables, vec![table("ocr-page-2", 2), table("ocr-page-1", 1)]);

    assert_eq!(tables.len(), 2);
    assert_eq!(tables[0].markdown, "ocr-page-1");
    assert_eq!(tables[1].markdown, "ocr-page-2");
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn empty_ocr_tables_preserve_native_tables() {
    let mut tables = vec![crate::types::Table {
        cells: Vec::new(),
        markdown: "native".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    }];

    replace_tables_with_ocr_output(&mut tables, Vec::new());

    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].markdown, "native");
}

/// GH#1670: a mixed-path document where only page 2 goes to OCR must keep the native
/// tables of every other page. The bug replaced the whole document's table list with only
/// page 2's OCR table, dropping pages 1 and 3 entirely.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn mixed_ocr_tables_replace_only_the_pages_ocr_produced_a_table_for() {
    let native_table = |page_number| crate::types::Table {
        cells: Vec::new(),
        markdown: format!("native-page-{page_number}"),
        page_number,
        bounding_box: None,
        ..Default::default()
    };
    let mut tables = vec![native_table(1), native_table(2), native_table(3)];

    let mut ocr_page = InternalDocument::new("pdf");
    ocr_page.tables.push(crate::types::Table {
        cells: Vec::new(),
        markdown: "ocr-page-2".to_string(),
        page_number: 2,
        bounding_box: None,
        ..Default::default()
    });
    let structured_pages = ahash::AHashMap::from([(2, ocr_page)]);

    merge_mixed_ocr_tables(&mut tables, &structured_pages);

    assert_eq!(
        tables.len(),
        3,
        "pages 1 and 3 were never sent to OCR; their native tables must survive"
    );
    assert_eq!(tables[0].markdown, "native-page-1");
    assert_eq!(
        tables[1].markdown, "ocr-page-2",
        "page 2 went to OCR; its native table is superseded"
    );
    assert_eq!(tables[2].markdown, "native-page-3");
}

/// A page that goes to OCR but whose OCR pass finds no table must keep its native table,
/// the same rule `replace_tables_with_ocr_output` applies when OCR finds nothing anywhere.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn a_page_ocr_ran_on_but_found_no_table_for_keeps_its_native_table() {
    let mut tables = vec![crate::types::Table {
        cells: Vec::new(),
        markdown: "native-page-1".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    }];

    // Page 1 went to OCR (it is a key in structured_pages) but produced no table there.
    let structured_pages = ahash::AHashMap::from([(1, InternalDocument::new("pdf"))]);

    merge_mixed_ocr_tables(&mut tables, &structured_pages);

    assert_eq!(tables.len(), 1);
    assert_eq!(tables[0].markdown, "native-page-1");
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn full_ocr_flat_fallback_never_selects_stale_native_document() {
    let mut native = InternalDocument::new("pdf");
    native.push_element(InternalElement::text(ElementKind::Paragraph, "stale native content", 0));

    let (doc, origin, structured) = select_pdf_document(
        ExtractionMethod::Ocr,
        "authoritative OCR content",
        "application/pdf",
        Some(native),
        None,
        None,
        None,
        None,
        &crate::core::config::OutputFormat::Markdown,
    );

    assert_eq!(origin, PdfDocumentOrigin::Ocr);
    assert!(!structured);
    assert!(
        doc.elements
            .iter()
            .any(|element| element.text == "authoritative OCR content")
    );
    assert!(
        !doc.elements
            .iter()
            .any(|element| element.text == "stale native content")
    );
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn mixed_origin_replaces_only_targeted_structured_page() {
    let mut native = InternalDocument::new("pdf");
    let table = crate::types::Table {
        cells: vec![vec!["native table".to_string()]],
        markdown: "| native table |".to_string(),
        page_number: 2,
        bounding_box: None,
        ..Default::default()
    };
    native.tables.push(table.clone());
    native.push_element(InternalElement::text(ElementKind::Paragraph, "native page one", 0).with_page(1));
    native.push_element(InternalElement::text(ElementKind::PageBreak, "", 0));
    native.push_element(InternalElement::text(ElementKind::Paragraph, "stale page two", 0).with_page(2));
    native.push_element(InternalElement::text(ElementKind::Table { table_index: 0 }, "", 0).with_page(2));
    let mut results = ahash::AHashMap::new();
    results.insert(2, "OCR page two".to_string());

    let (mut doc, origin, structured) = select_pdf_document(
        ExtractionMethod::Mixed,
        "flat text must not be selected",
        "application/pdf",
        Some(native),
        None,
        Some(&results),
        None,
        None,
        &crate::core::config::OutputFormat::Markdown,
    );

    assert_eq!(origin, PdfDocumentOrigin::Mixed);
    assert!(structured);
    assert!(doc.elements.iter().any(|element| element.text == "native page one"));
    assert!(doc.elements.iter().any(|element| element.text == "OCR page two"));
    assert!(!doc.elements.iter().any(|element| element.text == "stale page two"));
    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, false);
    assert_eq!(doc.tables.len(), 1, "structured table data must remain available");
    assert!(
        !doc.elements
            .iter()
            .any(|element| matches!(element.kind, ElementKind::Table { .. })),
        "target-page table markup must not be re-injected after OCR replacement"
    );
}

/// A fully scanned PDF has no native text layer at all, so
/// `extract_document_structure_from_segments` (native pipeline) runs over zero
/// segments, produces zero elements, and its caller collapses that to
/// `pre_rendered_doc: None` (see `extraction.rs`'s `Ok(_) => None` arm). Before this
/// fix, `ExtractionMethod::Mixed`'s `None` branch ignored `structured_ocr_pages`
/// entirely and returned `flat_pdf_document`, which only ever emits `Paragraph`
/// elements -- discarding every heading the document-global OCR heuristic recovered
/// (matches the `ordinance_2197_scanned.pdf` field observation: `blocks_to_paragraphs`
/// found ~71 headings, but the returned document, and its rendered markdown, had zero).
/// Fails on unfixed code: `doc.elements` has no `Heading` (only the flat `Paragraph`
/// from `text`), and the rendered markdown has no `#` line.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn mixed_without_native_document_still_renders_ocr_structure_as_markdown() {
    let mut results = ahash::AHashMap::new();
    results.insert(1u32, "ANNUAL REPORT OVERVIEW\n\nReport body text.".to_string());

    let mut structured_page = InternalDocument::new("pdf");
    structured_page.push_element(
        InternalElement::text(ElementKind::Heading { level: 1 }, "ANNUAL REPORT OVERVIEW", 0).with_page(1),
    );
    structured_page.push_element(InternalElement::text(ElementKind::Paragraph, "Report body text.", 0).with_page(1));
    let mut structured_pages = ahash::AHashMap::new();
    structured_pages.insert(1u32, structured_page);

    let text = "ANNUAL REPORT OVERVIEW\n\nReport body text.";
    let boundaries = vec![crate::types::PageBoundary {
        byte_start: 0,
        byte_end: text.len(),
        page_number: 1,
    }];

    let (doc, origin, structured) = select_pdf_document(
        ExtractionMethod::Mixed,
        text,
        "application/pdf",
        None, // no native document -- the fully-scanned-PDF case
        None,
        Some(&results),
        Some(&structured_pages),
        Some(&boundaries),
        &crate::core::config::OutputFormat::Markdown,
    );

    assert_eq!(origin, PdfDocumentOrigin::Mixed);
    assert!(
        structured,
        "a document assembled from OCR-recovered structure must be reported as structured"
    );
    assert!(
        doc.elements
            .iter()
            .any(|element| matches!(element.kind, ElementKind::Heading { .. })),
        "expected a Heading element to survive the merge when there is no native document; \
             got element kinds: {:?}",
        doc.elements.iter().map(|element| &element.kind).collect::<Vec<_>>()
    );

    let markdown = crate::rendering::render_markdown(&doc);
    assert!(
        markdown.contains("# ANNUAL REPORT OVERVIEW"),
        "OCR document-global heading structure must reach the rendered markdown even when \
             native xberg_native_pdf extraction produced no structured document: {markdown:?}"
    );
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn full_ocr_structured_without_ocr_tables_injects_native_fallback_once() {
    let ocr_doc = InternalDocument::new("pdf");
    let table = crate::types::Table {
        cells: vec![vec!["native fallback".to_string()]],
        markdown: "| native fallback |".to_string(),
        page_number: 1,
        bounding_box: None,
        ..Default::default()
    };
    let (mut doc, origin, structured) = select_pdf_document(
        ExtractionMethod::Ocr,
        "OCR content",
        "application/pdf",
        None,
        Some(ocr_doc),
        None,
        None,
        None,
        &crate::core::config::OutputFormat::Markdown,
    );
    let allow_injection = !structured || (origin == PdfDocumentOrigin::Ocr && doc.tables.is_empty());

    attach_unrepresented_tables(&mut doc, vec![table]);
    inject_unrepresented_table_elements(&mut doc, allow_injection);

    assert_eq!(doc.tables.len(), 1);
    assert_eq!(
        doc.elements
            .iter()
            .filter(|element| matches!(element.kind, ElementKind::Table { .. }))
            .count(),
        1
    );
}

#[test]
fn test_pdf_extractor_plugin_interface() {
    let extractor = PdfExtractor::new();
    assert_eq!(extractor.name(), "pdf-extractor");
    assert!(extractor.initialize().is_ok());
    assert!(extractor.shutdown().is_ok());
}

/// The condition under which a failed automatic OCR fallback must propagate instead of
/// returning an empty success. Observed on a real scanned document whose pages exceeded
/// `security_limits.max_content_size`: extraction returned `ok`, empty content,
/// `extraction_method: native`, `ocr_used: false`, `quality_score: 0.0`, and the caller had
/// no way to distinguish that from a genuinely empty PDF. ~keep
#[cfg(feature = "ocr-pipeline")]
#[test]
fn a_failed_ocr_fallback_with_no_native_text_is_a_total_loss() {
    assert!(
        failed_ocr_fallback_is_total_loss(""),
        "no native text at all is a total loss"
    );
    assert!(
        failed_ocr_fallback_is_total_loss("\n\n  \t "),
        "whitespace-only native text is as total a loss as none: it carries no content"
    );
    assert!(
        failed_ocr_fallback_is_total_loss("\u{FFFD}\u{FFFD} \u{FFFD}"),
        "text of only lost-glyph markers carries no content either"
    );
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn a_failed_ocr_fallback_with_native_text_still_returns_that_text() {
    assert!(
        !failed_ocr_fallback_is_total_loss("Ordinance No. 2024-17"),
        "native text below the OCR trigger is still worth returning with a warning"
    );
    assert!(
        !failed_ocr_fallback_is_total_loss("  x  "),
        "a single meaningful character is not a total loss"
    );
}

#[test]
fn test_pdf_extractor_supported_mime_types() {
    let extractor = PdfExtractor::new();
    let mime_types = extractor.supported_mime_types();
    assert_eq!(mime_types.len(), 1);
    assert!(mime_types.contains(&"application/pdf"));
}

/// GH#1752. Leaving `ocr_scanned_page_quality_gate` unset must reproduce the derived
/// answer for every shape of `ocr` block.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn scanned_page_quality_gate_unset_reproduces_the_presence_of_an_ocr_block() {
    use crate::core::config::OcrConfig;

    for ocr in [None, Some(OcrConfig::default())] {
        let config = ExtractionConfig {
            ocr: ocr.clone(),
            ..Default::default()
        };
        assert_eq!(
            config.ocr_scanned_page_quality_gate, None,
            "the setting must default to None"
        );
        assert_eq!(
            scanned_page_quality_gate_enabled(&config),
            ocr.is_some(),
            "with the setting unset the gate must follow the presence of an `ocr` block"
        );
    }
}

/// GH#1752 F1: mirrors `near_empty_fallback_unset_reproduces_the_1338_carve_out`'s oracle
/// pattern below, which already compares `near_empty_ocr_fallback_applies`'s `Some(true)`
/// arm against `automatic_ocr_backend_is_registered()` directly rather than forcing a
/// particular build's registry state. Switched on without an `ocr` block,
/// `scanned_page_quality_gate_enabled` must require a registered automatic backend the
/// same way; before a fix it returned `true` unconditionally, which let
/// `scanned_pages_to_ocr` enter the per-page gate (and its `whole_doc_failure` branch,
/// selecting every page) with no backend able to run OCR on any of them.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn scanned_page_quality_gate_on_without_an_ocr_block_requires_a_registered_backend() {
    let config = ExtractionConfig {
        ocr: None,
        ocr_scanned_page_quality_gate: Some(true),
        ..Default::default()
    };
    assert_eq!(
        scanned_page_quality_gate_enabled(&config),
        automatic_ocr_backend_is_registered(),
        "switched on without an `ocr` block, the gate must require a registered automatic \
             backend, exactly like its sibling `near_empty_ocr_fallback_applies`'s `Some(true)` arm"
    );
}

/// GH#1752 / #1338. `ocr_near_empty_fallback` unset must reproduce the derived condition:
/// always with an `ocr` block, and without one only for genuinely absent native text.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn near_empty_fallback_unset_reproduces_the_1338_carve_out() {
    use crate::core::config::OcrConfig;

    let with_block = ExtractionConfig {
        ocr: Some(OcrConfig::default()),
        ..Default::default()
    };
    let without_block = ExtractionConfig::default();

    assert_eq!(with_block.ocr_near_empty_fallback, None);
    assert_eq!(without_block.ocr_near_empty_fallback, None);

    assert!(
        near_empty_ocr_fallback_applies(&with_block, "a page with plenty of real native text"),
        "an explicit `ocr` block keeps its full per-page gate behaviour"
    );
    assert!(
        !near_empty_ocr_fallback_applies(&without_block, "a page with plenty of real native text"),
        "without a block, a legitimately sparse or short PDF must stay native (#1338)"
    );
    assert_eq!(
        near_empty_ocr_fallback_applies(&without_block, "   \n  "),
        automatic_ocr_backend_is_registered(),
        "without a block, genuinely absent native text reaches OCR iff a backend is registered"
    );
}

/// Text made only of U+FFFD holds no content: every glyph of the page lost its Unicode value.
/// Without an `ocr` block the fallback must treat it like empty text, or a page that is
/// fully undecodable stops reaching OCR once the extractor keeps the U+FFFD. A page with
/// real text beside the U+FFFD stays native.
#[cfg(feature = "ocr-pipeline")]
#[test]
#[serial_test::serial]
fn near_empty_fallback_treats_undecodable_only_text_as_absent_text() {
    use crate::plugins::{OcrBackend, OcrBackendType, Plugin};
    use std::sync::Arc;

    struct StandInBackend(String);
    impl Plugin for StandInBackend {
        fn name(&self) -> &str {
            &self.0
        }
        fn version(&self) -> String {
            "1.0.0".to_string()
        }
        fn initialize(&self) -> crate::Result<()> {
            Ok(())
        }
        fn shutdown(&self) -> crate::Result<()> {
            Ok(())
        }
    }
    #[async_trait::async_trait]
    impl OcrBackend for StandInBackend {
        fn backend_type(&self) -> OcrBackendType {
            OcrBackendType::Custom
        }
        fn supports_language(&self, _: &str) -> bool {
            true
        }
        async fn process_image(
            &self,
            _: &[u8],
            _: &crate::core::config::OcrConfig,
        ) -> crate::Result<crate::types::ExtractedDocument> {
            Ok(crate::types::ExtractedDocument::default())
        }
    }

    let name = crate::core::config::OcrConfig::default().backend;
    let registered_here = !automatic_ocr_backend_is_registered()
        && crate::plugins::register_ocr_backend(Arc::new(StandInBackend(name.clone()))).is_ok();
    let without_block = ExtractionConfig::default();

    let lost = near_empty_ocr_fallback_applies(&without_block, "\u{FFFD}\u{FFFD} \n\u{FFFD}");
    let mixed = near_empty_ocr_fallback_applies(&without_block, "A real word \u{FFFD}");
    let backend_present = automatic_ocr_backend_is_registered();
    if registered_here {
        crate::plugins::unregister_ocr_backend(&name).unwrap();
    }

    assert!(backend_present, "the test needs an automatic backend to be registered");
    assert!(lost, "text of only U+FFFD reaches OCR like empty text");
    assert!(!mixed, "text with real words beside U+FFFD stays native");
}

/// Switched on, the fallback stops requiring *completely* empty text -- that is the whole
/// point: a scanned page carrying only a page label or scanner stamp must reach OCR. It
/// still consults the backend registry exactly as the derived condition does.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn near_empty_fallback_on_without_an_ocr_block_drops_the_empty_text_requirement() {
    let opted_in = ExtractionConfig {
        ocr: None,
        ocr_near_empty_fallback: Some(true),
        ..Default::default()
    };
    let derived = ExtractionConfig::default();

    assert_eq!(
        near_empty_ocr_fallback_applies(&opted_in, "Page 3 of 412"),
        automatic_ocr_backend_is_registered(),
        "the stamp-only page must now be judged by the quality gate, not by the empty-text test"
    );
    assert_eq!(
        near_empty_ocr_fallback_applies(&opted_in, "Page 3 of 412"),
        near_empty_ocr_fallback_applies(&derived, ""),
        "and the backend requirement is unchanged: only the text test widened"
    );
}

/// Switched off, the fallback never runs, even with an explicit `ocr` block.
#[cfg(feature = "ocr-pipeline")]
#[test]
fn near_empty_fallback_off_suppresses_the_branch_even_with_an_ocr_block() {
    use crate::core::config::OcrConfig;

    let config = ExtractionConfig {
        ocr: Some(OcrConfig::default()),
        ocr_near_empty_fallback: Some(false),
        ..Default::default()
    };

    assert!(!near_empty_ocr_fallback_applies(&config, ""));
    assert!(!near_empty_ocr_fallback_applies(&config, "real native text"));
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_pdf_batch_mode_validates_page_config_enabled() {
    use crate::core::config::PageConfig;

    let extractor = PdfExtractor::new();

    let config = ExtractionConfig {
        pages: Some(PageConfig {
            extract_pages: true,
            insert_page_markers: false,
            marker_format: "<!-- PAGE {page_num} -->".to_string(),
            keep_native_content: false,
        }),
        ..Default::default()
    };

    let pdf_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_documents/pdf/google_doc_document.pdf");
    if let Ok(content) = std::fs::read(pdf_path) {
        let result = extractor.extract_content(&content, "application/pdf", &config).await;
        assert!(
            result.is_ok(),
            "Failed to extract PDF with page config: {:?}",
            result.err()
        );

        let extraction_result = result.unwrap();
        let extraction_result = crate::extraction::derive::derive_extraction_result(
            extraction_result,
            true,
            crate::core::config::OutputFormat::Plain,
        );
        assert!(
            !extraction_result.content.is_empty(),
            "Content should be extracted from PDF"
        );
    }
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_pdf_batch_mode_derives_pages_with_page_config_absent() {
    let extractor = PdfExtractor::new();
    let config = ExtractionConfig::default();

    let pdf_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_documents/pdf/google_doc_document.pdf");
    if let Ok(content) = std::fs::read(pdf_path) {
        let result = extractor.extract_content(&content, "application/pdf", &config).await;
        assert!(
            result.is_ok(),
            "Failed to extract PDF without page config: {:?}",
            result.err()
        );

        let extraction_result = result.unwrap();
        let extraction_result = crate::extraction::derive::derive_extraction_result(
            extraction_result,
            true,
            crate::core::config::OutputFormat::Plain,
        );
        assert!(
            extraction_result.pages.is_some(),
            "Pages should always be derived: the default config leaves annotation extraction \
                 off, which forces per-page tracking for the annotation fallback regardless of an \
                 explicit pages config"
        );
    }
}

/// GH CI E2E `test_pdf_hierarchy_config`: `pdf_options.hierarchy.enabled` used to be a silent
/// no-op. A `PageHierarchy` can only hang off a `PageContent`, and `page_contents` is produced
/// only when `pages.extract_pages` is set -- which `PageConfig::default()` leaves `false` and
/// nothing in the hierarchy config points at. Headings were detected, then dropped.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_pdf_hierarchy_enabled_populates_page_hierarchy_without_explicit_page_config() {
    use crate::core::config::{HierarchyConfig, PdfConfig};

    let pdf_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_documents/pdf/embedded_images_tables.pdf");
    let Ok(content) = std::fs::read(pdf_path) else {
        return;
    };

    let config = ExtractionConfig {
        pdf_options: Some(PdfConfig {
            hierarchy: Some(HierarchyConfig {
                enabled: true,
                ..HierarchyConfig::default()
            }),
            ..PdfConfig::default()
        }),
        ..ExtractionConfig::default()
    };

    let extractor = PdfExtractor::new();
    let result = extractor
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("hierarchy extraction should succeed");
    let result =
        crate::extraction::derive::derive_extraction_result(result, true, crate::core::config::OutputFormat::Plain);

    let pages = result.pages.as_ref().expect("hierarchy request must produce pages");
    let hierarchy = pages[0]
        .hierarchy
        .as_ref()
        .expect("pages[0].hierarchy must be populated when hierarchy.enabled is true");
    assert!(
        !hierarchy.blocks.is_empty(),
        "hierarchy must carry at least one block; got block_count={}",
        hierarchy.block_count
    );
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_pdf_page_marker_validation() {
    use crate::core::config::PageConfig;

    let extractor = PdfExtractor::new();

    let config = ExtractionConfig {
        pages: Some(PageConfig {
            extract_pages: true,
            insert_page_markers: true,
            marker_format: "\n\n<!-- PAGE {page_num} -->\n\n".to_string(),
            keep_native_content: false,
        }),
        ..Default::default()
    };

    let pdf_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_documents/pdf/multi_page.pdf");
    if let Ok(content) = std::fs::read(pdf_path) {
        let result = extractor.extract_content(&content, "application/pdf", &config).await;
        assert!(
            result.is_ok(),
            "Failed to extract PDF with page markers: {:?}",
            result.err()
        );

        let extraction_result = result.unwrap();
        let extraction_result = crate::extraction::derive::derive_extraction_result(
            extraction_result,
            true,
            crate::core::config::OutputFormat::Plain,
        );
        let marker_placeholder = "<!-- PAGE ";
        if extraction_result.content.len() > 100 {
            assert!(
                extraction_result.content.contains(marker_placeholder),
                "Page markers should be inserted when configured and document has multiple pages"
            );
        }
    }
}

/// #1451: `max_pages` must reject a document once its page count is known, before
/// per-page work (OCR/layout/rendering) starts. Against the unfixed extractor there is
/// no `max_pages` field to set, so this test fails to compile; once the field exists but
/// enforcement is missing, `extract_content` would return `Ok` here instead of the
/// expected `SecurityError::TooManyPages`.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn should_reject_document_exceeding_max_pages() {
    let pdf_path = pdf_test_document("multi_page.pdf");
    let Ok(content) = std::fs::read(&pdf_path) else {
        return;
    };
    let real_page_count = xberg_native_pdf::PdfDocument::from_bytes(content.clone())
        .expect("fixture must parse")
        .page_count()
        .expect("fixture must expose a page count");
    assert!(
        real_page_count > 1,
        "fixture must have more than one page to exercise the limit"
    );

    let extractor = PdfExtractor::new();
    let config = ExtractionConfig {
        security_limits: Some(crate::extractors::security::SecurityLimits {
            max_pages: Some(real_page_count - 1),
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = extractor.extract_content(&content, "application/pdf", &config).await;
    let error = result.expect_err("a document over max_pages must be rejected");
    let message = error.to_string();
    assert!(
        message.contains("too many pages") || message.contains("max_pages"),
        "error must name the limit that was hit: {message}"
    );
}

/// A base64-encoded, 3-page, AES-256 (R6) encrypted PDF whose user password is the
/// obviously-fake literal `xberg-test-fake-password-1451`. No corpus fixture requires a
/// real user password to open: every encrypted PDF under `test_documents/pdf/` (including
/// `password_protected.pdf`, despite the name) authenticates with the empty password and
/// only restricts copy/edit permissions, so none of them exercise the password-required
/// path this test targets. Built with `qpdf --encrypt ... 256` from a minimal 3-page
/// document and verified with `qpdf --check` (fails with no password or the wrong one,
/// succeeds and reports 3 pages with the correct one) before being embedded here.
#[cfg(feature = "pdf")]
const ENCRYPTED_THREE_PAGE_PDF_BASE64: &str = concat!(
    "JVBERi0xLjcKJb/3ov4KMSAwIG9iago8PCAvRXh0ZW5zaW9ucyA8PCAvQURCRSA8PCAvQmFzZVZlcnNpb24gLzEuNyAv",
    "RXh0ZW5zaW9uTGV2ZWwgOCA+PiA+PiAvUGFnZXMgMiAwIFIgL1R5cGUgL0NhdGFsb2cgPj4KZW5kb2JqCjIgMCBvYmoK",
    "PDwgL0NvdW50IDMgL0tpZHMgWyAzIDAgUiA0IDAgUiA1IDAgUiBdIC9UeXBlIC9QYWdlcyA+PgplbmRvYmoKMyAwIG9i",
    "ago8PCAvQ29udGVudHMgNiAwIFIgL01lZGlhQm94IFsgMCAwIDIwMCAyMDAgXSAvUGFyZW50IDIgMCBSIC9SZXNvdXJj",
    "ZXMgPDwgL0ZvbnQgPDwgL0YxIDcgMCBSID4+ID4+IC9UeXBlIC9QYWdlID4+CmVuZG9iago0IDAgb2JqCjw8IC9Db250",
    "ZW50cyA4IDAgUiAvTWVkaWFCb3ggWyAwIDAgMjAwIDIwMCBdIC9QYXJlbnQgMiAwIFIgL1Jlc291cmNlcyA8PCAvRm9u",
    "dCA8PCAvRjEgNyAwIFIgPj4gPj4gL1R5cGUgL1BhZ2UgPj4KZW5kb2JqCjUgMCBvYmoKPDwgL0NvbnRlbnRzIDkgMCBS",
    "IC9NZWRpYUJveCBbIDAgMCAyMDAgMjAwIF0gL1BhcmVudCAyIDAgUiAvUmVzb3VyY2VzIDw8IC9Gb250IDw8IC9GMSA3",
    "IDAgUiA+PiA+PiAvVHlwZSAvUGFnZSA+PgplbmRvYmoKNiAwIG9iago8PCAvRmlsdGVyIC9GbGF0ZURlY29kZSAvTGVu",
    "Z3RoIDgwID4+CnN0cmVhbQq/uprLnpDr1L1WYuji7aoo8FC/AmNCG9HIRQLXttVpBBc7g+Wbs7gcxVkaZNEjLHKAT0E4",
    "zK38JxdAJjfL7XkpkCbptuaCZCGvvfQ0dyrxlWVuZHN0cmVhbQplbmRvYmoKNyAwIG9iago8PCAvQmFzZUZvbnQgL0hl",
    "bHZldGljYSAvU3VidHlwZSAvVHlwZTEgL1R5cGUgL0ZvbnQgPj4KZW5kb2JqCjggMCBvYmoKPDwgL0ZpbHRlciAvRmxh",
    "dGVEZWNvZGUgL0xlbmd0aCA4MCA+PgpzdHJlYW0K4JE0iPOQNpJRjCvrfuVuZ8G+7e3bbkuci4qETQEnA0RfeOqEZe5q",
    "L81EqqU/h7+KsfI+uoIT6tBO4uAdnj/i0054F0Q6VoQWb0PkpOiMLy1lbmRzdHJlYW0KZW5kb2JqCjkgMCBvYmoKPDwg",
    "L0ZpbHRlciAvRmxhdGVEZWNvZGUgL0xlbmd0aCA4MCA+PgpzdHJlYW0KKqDRR85TmYv59t1N7OdBCttbZfgU8lsnHRFY",
    "MueyVRSwbBDvnujXiRLztgcPFiS4XexkTW/ikEXnPHq9uFq1VDpPVQNpaRQlZKf9xlAUveVlbmRzdHJlYW0KZW5kb2Jq",
    "CjEwIDAgb2JqCjw8IC9DRiA8PCAvU3RkQ0YgPDwgL0F1dGhFdmVudCAvRG9jT3BlbiAvQ0ZNIC9BRVNWMyAvTGVuZ3Ro",
    "IDMyID4+ID4+IC9GaWx0ZXIgL1N0YW5kYXJkIC9MZW5ndGggMjU2IC9PIDw0ZmQzOTFhNjY0MzdlZjFkYzEwNDMxOGVj",
    "OTZhNDY1OTM5OWNhMzY2YTY1MTBmZjhkNTI1N2FjMmJiOGRmMGNkMGRiZjkzYTc4MDgzYTkzOGRhYzBlMjcwMzliMTMw",
    "ZmM+IC9PRSA8NDkwZTZlNzI0OGM2NmZkMjZjOTdjMzE3MWVlNjVlNjc0Yjk1YWM2ZWI1MTI1ZjVlZmVkMDBmOTI3NmQw",
    "Y2YxMj4gL1AgLTQgL1Blcm1zIDw1NmI5YWRkMWJiMWI2YjUwZGYyMjg3NWM1Njc5YTJmMz4gL1IgNiAvU3RtRiAvU3Rk",
    "Q0YgL1N0ckYgL1N0ZENGIC9VIDw0NTNlNjI2Y2JmZTI1MGE5N2FmNDllZWEzMmYwOWFmOWMyZDk4Y2MzMWM2ZGQ1NDdj",
    "NjkwYzU1NTMwZDhjN2ViYjA2MzdlN2JjZmZmMDdkZThmYjc2NmFmNWQ4ZDM1NWY+IC9VRSA8MTA0YWEyZjAzNDBiMTEw",
    "ZGEzZTFmMGQ0ODQ3NmQyOWMyZTExNmYxOTRlZTBjMjFmYWFiMzkzNDFiZmU2NjMyZT4gL1YgNSA+PgplbmRvYmoKeHJl",
    "ZgowIDExCjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAxNSAwMDAwMCBuIAowMDAwMDAwMTMwIDAwMDAwIG4gCjAw",
    "MDAwMDAyMDEgMDAwMDAgbiAKMDAwMDAwMDMyOSAwMDAwMCBuIAowMDAwMDAwNDU3IDAwMDAwIG4gCjAwMDAwMDA1ODUg",
    "MDAwMDAgbiAKMDAwMDAwMDczNSAwMDAwMCBuIAowMDAwMDAwODA1IDAwMDAwIG4gCjAwMDAwMDA5NTUgMDAwMDAgbiAK",
    "MDAwMDAwMTEwNSAwMDAwMCBuIAp0cmFpbGVyIDw8IC9Sb290IDEgMCBSIC9TaXplIDExIC9JRCBbPGU4ODhiODFiZjdj",
    "YzRhODE3NzQ3YmM4ZDZlZTgxYzgxPjxmYzI0MzUwMTY1NzA4NjdhNDYzNTZiYTFiYWE5ZGVhND5dIC9FbmNyeXB0IDEw",
    "IDAgUiA+PgpzdGFydHhyZWYKMTY1MwolJUVPRgo=",
);

#[cfg(feature = "pdf")]
const ENCRYPTED_THREE_PAGE_PDF_PASSWORD: &str = "xberg-test-fake-password-1451";

/// An obviously-fake literal that is not this fixture's password, and is not empty.
#[cfg(feature = "pdf")]
const WRONG_PDF_PASSWORD: &str = "xberg-test-fake-wrong-password-1879";

/// `PdfDocument::authenticate` reports a rejected password as `Ok(false)`, not as an `Err`,
/// so `open_pdf_document` -- which mapped the error and dropped the bool -- accepted every
/// wrong password. That is why no `pdf_page_count` fixture could prove a binding forwards
/// its password argument: the page tree reads unauthenticated, and authentication itself
/// never failed either. The extraction path (`open_bytes_with_passwords`) always read the
/// bool and is unaffected. GH#1879.
#[test]
#[cfg(feature = "pdf")]
fn should_reject_a_wrong_pdf_password_rather_than_ignore_it() {
    use base64::Engine as _;

    let content = base64::engine::general_purpose::STANDARD
        .decode(ENCRYPTED_THREE_PAGE_PDF_BASE64)
        .expect("fixture must be valid base64");

    let error = crate::pdf_page_count(&content, Some(WRONG_PDF_PASSWORD))
        .expect_err("a wrong non-empty user password must not authenticate");
    assert!(
        error.to_string().contains("authenticate"),
        "the error must name authentication as the cause: {error}"
    );

    assert_eq!(
        crate::pdf_page_count(&content, Some(ENCRYPTED_THREE_PAGE_PDF_PASSWORD))
            .expect("the correct user password must authenticate"),
        3,
        "the correct password must still open the document"
    );
    assert_eq!(
        crate::pdf_page_count(&content, None).expect("an unauthenticated page-tree read must still succeed"),
        3,
        "the page tree is structure, not encrypted data, so omitting the password is not an error"
    );
}

/// The corpus fixture whose user password is empty is still the right document for a
/// binding-level wrong-password test: the empty password authenticates, and any non-empty
/// password is rejected. So the e2e fixture needs no new corpus object. GH#1879.
#[test]
#[cfg(feature = "pdf")]
fn should_reject_a_non_empty_password_on_a_pdf_whose_user_password_is_empty() {
    let Some(content) = crate::utils::read_test_fixture("pdf/password_protected.pdf") else {
        return;
    };

    assert_eq!(
        crate::pdf_page_count(&content, None).expect("an encrypted page tree reads without a password"),
        2
    );
    assert_eq!(
        crate::pdf_page_count(&content, Some("")).expect("the empty user password authenticates"),
        2
    );
    let error = crate::pdf_page_count(&content, Some(WRONG_PDF_PASSWORD))
        .expect_err("a non-empty password is neither the user nor the owner password");
    assert!(
        error.to_string().contains("authenticate"),
        "the error must name authentication as the cause: {error}"
    );
}

/// An encrypted document is subject to `max_pages` like any other. This is a
/// characterization test, NOT a regression test — it passes against the pre-fix code
/// too, and it is kept because the behaviour is worth pinning, not because it catches
/// anything.
///
/// It was originally written believing `xberg_native_pdf` refuses to count an unauthenticated
/// document's pages, so an encrypted PDF would skip the cap. That is false, and measuring
/// it is what settled the design: a PDF's page tree is structure, not string or stream
/// data, so `xberg_native_pdf::PdfDocument::from_bytes(..).page_count()` returns 3 for this
/// AES-256/R6 fixture with no password at all. A sweep of all 488 PDFs in
/// `test_documents/` found exactly one document the unauthenticated count cannot handle
/// (`corrupt_truncated.pdf`), which `lopdf` also cannot read and which fails extraction
/// anyway. So `enforce_page_limit` needs no password handling.
///
/// The fixture: 3 pages, AES-256/R6, built with `qpdf --encrypt ... 256` and verified
/// with `qpdf --check` before embedding. Its password is the obviously-fake literal
/// `xberg-test-fake-password-1451`. No corpus fixture requires a real user password —
/// every encrypted PDF under `test_documents/pdf/` (including `password_protected.pdf`,
/// despite the name) opens with the empty password and only restricts copy/edit.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn encrypted_document_over_max_pages_is_rejected_like_any_other() {
    use crate::core::config::pdf::PdfConfig;
    use base64::Engine as _;

    let content = base64::engine::general_purpose::STANDARD
        .decode(ENCRYPTED_THREE_PAGE_PDF_BASE64)
        .expect("fixture must be valid base64");

    let extractor = PdfExtractor::new();
    let config = ExtractionConfig {
        security_limits: Some(crate::extractors::security::SecurityLimits {
            max_pages: Some(2),
            ..Default::default()
        }),
        pdf_options: Some(PdfConfig {
            passwords: Some(vec![ENCRYPTED_THREE_PAGE_PDF_PASSWORD.to_string()]),
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = extractor.extract_content(&content, "application/pdf", &config).await;
    let error = result.expect_err("an encrypted document over max_pages must be rejected like any other");
    let message = error.to_string();
    assert!(
        message.contains("too many pages") || message.contains("max_pages"),
        "error must name the limit that was hit: {message}"
    );
}

/// A document exactly at the configured `max_pages` ceiling must extract in full — a
/// limit that rejects the boundary case too is not the fix the issue asked for.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn should_extract_fully_when_page_count_is_at_max_pages() {
    let pdf_path = pdf_test_document("multi_page.pdf");
    let Ok(content) = std::fs::read(&pdf_path) else {
        return;
    };
    let real_page_count = xberg_native_pdf::PdfDocument::from_bytes(content.clone())
        .expect("fixture must parse")
        .page_count()
        .expect("fixture must expose a page count");

    let extractor = PdfExtractor::new();
    let config = ExtractionConfig {
        security_limits: Some(crate::extractors::security::SecurityLimits {
            max_pages: Some(real_page_count),
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = extractor.extract_content(&content, "application/pdf", &config).await;
    let doc = result.expect("a document exactly at max_pages must extract fully, not be rejected");
    assert!(
        !doc.elements.is_empty(),
        "extraction at the boundary must still produce content, not an empty truncated result"
    );
}

/// The default `SecurityLimits` (no override) must extract a multi-page document exactly
/// as before #1451: `max_pages` defaulting to anything other than `None` (unlimited) would
/// silently start rejecting existing callers' documents.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn should_extract_normally_with_default_security_limits() {
    let pdf_path = pdf_test_document("multi_page.pdf");
    let Ok(content) = std::fs::read(&pdf_path) else {
        return;
    };

    let extractor = PdfExtractor::new();
    let config = ExtractionConfig::default();

    let result = extractor.extract_content(&content, "application/pdf", &config).await;
    assert!(
        result.is_ok(),
        "default security limits must not reject a normal multi-page document: {:?}",
        result.err()
    );
}

/// Proves the `PdfBackend::Pdfium` dispatch branch in `extract_core` (#702):
/// selecting pdfium must fail with a specific, actionable error naming the
/// backend and telling the caller what to do, rather than the native parser's
/// generic "corrupt file" error. Content is deliberately not a valid PDF --
/// dispatch must happen before any parsing, so a real pdfium rejection never
/// reaches the parser at all.
///
/// Fails against unfixed code: `extract_core` unconditionally calls
/// `extract_core_native` and never reads `PdfConfig::backend`, so `content`
/// gets handed straight to `xberg_native_pdf`. The native parser rejects the garbage
/// bytes with an `XbergError::Parsing` whose message describes a corrupt/invalid
/// PDF and never contains the word "pdfium" -- the
/// `message.contains("pdfium")` assertion below fails.
///
/// Gated on `not(feature = "pdf-pdfium")`: this exercises the *stub* rejection
/// wording. With `pdf-pdfium` enabled the real engine runs instead (see
/// `pdfium_engine`), whose error text this assertion does not describe.
#[tokio::test]
#[cfg(all(feature = "pdf", not(feature = "pdf-pdfium")))]
async fn pdfium_backend_is_rejected_with_actionable_error_not_generic_parse_failure() {
    let extractor = PdfExtractor::new();
    let mut config = ExtractionConfig::default();
    config.pdf_options = Some(crate::core::config::PdfConfig {
        backend: crate::core::config::PdfBackend::Pdfium,
        ..Default::default()
    });

    let result = extractor
        .extract_content(b"not a real pdf", "application/pdf", &config)
        .await;

    let error = result.expect_err("selecting pdfium must fail, not silently extract via xberg_native_pdf");
    let message = error.to_string();
    assert!(
        message.contains("pdfium"),
        "error must name the requested backend 'pdfium', got: {message}"
    );
    assert!(
        message.contains("native"),
        "error must tell the caller what to do (use native instead), got: {message}"
    );
}

/// Proves there is no silent fallback (#702's core danger): given a document
/// `xberg_native_pdf` can extract perfectly well, selecting `PdfBackend::Pdfium` must
/// still fail rather than quietly returning `xberg_native_pdf`'s output mislabeled as
/// pdfium. Uses a real, valid multi-page PDF specifically to rule out "it failed
/// because the input was bad" as an explanation for the error.
///
/// Fails against unfixed code: `extract_core` ignores `PdfConfig::backend`
/// entirely and always calls `extract_core_native`, so `result.is_ok()` is `true`
/// pre-fix (the exact silent-fallback failure mode #702 exists to close) instead
/// of the `false` asserted here.
///
/// Gated on `not(feature = "pdf-pdfium")`: with the real engine enabled, a valid
/// document legitimately extracting via pdfium is the whole point (see
/// `pdfium_engine_serializes_concurrent_extractions_when_the_library_is_available` below), not
/// a silent fallback -- this test's premise (pdfium selection must always fail)
/// only holds for the stub.
#[tokio::test]
#[cfg(all(feature = "pdf", not(feature = "pdf-pdfium")))]
async fn pdfium_backend_never_silently_falls_back_to_xberg_native_pdf_output() {
    let pdf_path = pdf_test_document("multi_page.pdf");
    let Ok(content) = std::fs::read(&pdf_path) else {
        return;
    };

    let extractor = PdfExtractor::new();
    let mut config = ExtractionConfig::default();
    config.pdf_options = Some(crate::core::config::PdfConfig {
        backend: crate::core::config::PdfBackend::Pdfium,
        ..Default::default()
    });

    let result = extractor.extract_content(&content, "application/pdf", &config).await;
    assert!(
        result.is_err(),
        "pdfium selected on a document xberg_native_pdf could extract fine must still error, \
             not silently return xberg_native_pdf's output under the caller's pdfium selection"
    );
}

/// Proves the `pdf-pdfium` engine (#702) actually extracts text and metadata
/// through pdfium rather than reusing `xberg_native_pdf` output. `multi_page.pdf` is a
/// 5-page document whose first page's text is known ahead of time (verified
/// with `pdftotext` against the fixture). Four requests run concurrently to
/// regression-test serialization of Pdfium's process-global C API.
///
/// Requires a real `libpdfium` on the system library search path (or
/// `PDFIUM_DYNAMIC_LIB_PATH` pointing at one) -- this repository provisions
/// neither today, so this test skips (rather than fails) when
/// `pdfium_engine::bind_once` cannot load the library. That is an environment
/// gap, not a code defect; asserting the error is specifically
/// `XbergError::MissingDependency` (rather than string-matching the message)
/// still proves the failure came from library loading and not from the stub
/// rejection or some other error being swallowed.
///
/// Fails against unfixed code: the stub `extract_core_pdfium` unconditionally
/// returns `XbergError::Validation` (never attempting to load pdfium), so on a
/// machine without `libpdfium` installed the `matches!(err,
/// XbergError::MissingDependency(_))` assertion below fails -- the error variant
/// is `Validation`, not `MissingDependency` -- and on a machine *with*
/// `libpdfium` installed the stub still returns `Validation` unconditionally, so
/// `result.is_ok()` never becomes `true` and the `.expect`-equivalent match's
/// `Ok` arm is never reached; either way the real extraction assertions further
/// down never run against unfixed code.
#[tokio::test]
#[cfg(all(feature = "pdf", feature = "pdf-pdfium"))]
async fn pdfium_engine_serializes_concurrent_extractions_when_the_library_is_available() {
    // Skip-on-absence is right for a developer without the bucket-fetched corpus, but it must
    // never be how CI passes. `XBERG_REQUIRE_PDFIUM` (set by the `pdfium-engine` job) turns
    // every skip in this test into a failure, following the same truthy-env convention as
    // `XBERG_REQUIRE_MODELS` in tests/candle_backends.rs. Without it this test reports success
    // whenever libpdfium cannot be loaded -- which is exactly how the engine ended up with no
    // executed coverage at all. ~keep
    let require_pdfium = matches!(
        std::env::var("XBERG_REQUIRE_PDFIUM").as_deref(),
        Ok("1" | "true" | "yes")
    );

    let pdf_path = pdf_test_document("multi_page.pdf");
    let content = match std::fs::read(&pdf_path) {
        Ok(content) => content,
        Err(err) => {
            assert!(
                !require_pdfium,
                "XBERG_REQUIRE_PDFIUM is set but the fixture at {} could not be read ({err}); \
                     a fixture-fetch failure is a checkout bug, never a reason to report success",
                pdf_path.display()
            );
            return;
        }
    };

    let extractor = PdfExtractor::new();
    let mut config = ExtractionConfig::default();
    config.pdf_options = Some(crate::core::config::PdfConfig {
        backend: crate::core::config::PdfBackend::Pdfium,
        ..Default::default()
    });

    let (first, second, third, fourth) = tokio::join!(
        extractor.extract_content(&content, "application/pdf", &config),
        extractor.extract_content(&content, "application/pdf", &config),
        extractor.extract_content(&content, "application/pdf", &config),
        extractor.extract_content(&content, "application/pdf", &config),
    );
    let results = [first, second, third, fourth];

    if let Some(err) = results.iter().find_map(|result| result.as_ref().err()) {
        if matches!(err, crate::XbergError::MissingDependency(_)) {
            assert!(
                !require_pdfium,
                "XBERG_REQUIRE_PDFIUM is set, so a real libpdfium was supposed to be loadable, \
                     but binding failed: {err:?}. Returning here would report success without ever \
                     exercising the engine."
            );
            return;
        }

        panic!("concurrent pdfium extraction failed: {err:?}");
    }

    for document in results.into_iter().flatten() {
        let result = crate::extraction::derive::derive_extraction_result(
            document,
            true,
            crate::core::config::OutputFormat::Plain,
        );

        assert!(
            result.content.contains("The Evolution of the Word Processor"),
            "expected the fixture's real first-page heading in pdfium output, got: {:?}",
            result.content
        );
        assert!(
            result.content.contains("Christopher Latham Sholes"),
            "expected real body text from the fixture in pdfium output, got: {:?}",
            result.content
        );

        let page_count = result.metadata.format.as_ref().and_then(|format| match format {
            crate::types::FormatMetadata::Pdf(pdf) => pdf.page_count,
            _ => None,
        });
        assert_eq!(page_count, Some(5), "multi_page.pdf has 5 pages");
    }
}

#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_pdf_exposes_native_extraction_method() {
    let extractor = PdfExtractor::new();
    let config = ExtractionConfig::default();
    let pdf_path = pdf_test_document("google_doc_document.pdf");

    if let Ok(content) = std::fs::read(pdf_path) {
        let result = extractor
            .extract_content(&content, "application/pdf", &config)
            .await
            .expect("native PDF extraction should succeed");
        let result =
            crate::extraction::derive::derive_extraction_result(result, true, crate::core::config::OutputFormat::Plain);

        assert_eq!(extraction_method(&result), Some(ExtractionMethod::Native));
    }
}

/// GH#1668 end to end: `include_document_structure` alone, with `output_format`
/// left at its `Plain` default and no hierarchy config, must trigger the
/// structured native pass. Before the fix, this config silently took the flat
/// path (`flat_pdf_document`, paragraph-only elements), so the structure tree
/// this flag asks for came back holding nothing but `paragraph` nodes even
/// though `embedded_images_tables.pdf` has a native table `counts.tables`
/// reports correctly either way.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_include_document_structure_alone_triggers_the_structured_path() {
    use crate::core::config::OutputFormat;

    let extractor = PdfExtractor::new();
    let pdf_path = pdf_test_document("embedded_images_tables.pdf");
    assert!(
        pdf_path.exists(),
        "missing test fixture: {pdf_path:?} — add embedded_images_tables.pdf to test_documents/pdf/"
    );
    let content = std::fs::read(&pdf_path).expect("failed to read embedded_images_tables.pdf");

    let config = crate::core::config::ExtractionConfig {
        include_document_structure: true,
        ..Default::default()
    };

    let result = extractor
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("native extraction with include_document_structure should succeed");

    let result = crate::extraction::derive::derive_extraction_result(result, true, OutputFormat::Plain);

    let structure = result
        .document
        .as_ref()
        .expect("include_document_structure=true must populate the document field");
    assert!(
        structure.node_types.iter().any(|kind| kind != "paragraph"),
        "structure must hold a non-paragraph node kind on a document with a native table; got {:?}",
        structure.node_types
    );
}

/// GH#1668 review follow-up: `test_include_document_structure_alone_triggers_the_structured_path`
/// only asserts `node_types`, never `content`, so it cannot catch a change that keeps the
/// structure tree correct while reverting the plain-text rendering the changelog also
/// promises. `mod.rs`'s `select_pdf_document` Mixed-branch merge already gates a similar
/// merge on `output_format != Plain` (see the comment there); a later change of that shape
/// applied to the native path would silently revert this fix without failing the existing
/// test. This test pins the plain-text side directly: with `include_document_structure` set,
/// `Plain` output must still come from the structured document, so each table row renders as
/// its own line (`render_table_plain` is `row.join(" ")` per row), and that output must differ
/// from the flag left off.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn plain_output_with_document_structure_renders_each_table_row_on_its_own_line() {
    use crate::core::config::OutputFormat;

    let extractor = PdfExtractor::new();
    let pdf_path = pdf_test_document("embedded_images_tables.pdf");
    assert!(
        pdf_path.exists(),
        "missing test fixture: {pdf_path:?} — add embedded_images_tables.pdf to test_documents/pdf/"
    );
    let content = std::fs::read(&pdf_path).expect("failed to read embedded_images_tables.pdf");

    let flat_config = crate::core::config::ExtractionConfig {
        include_document_structure: false,
        ..Default::default()
    };
    let flat_result = extractor
        .extract_content(&content, "application/pdf", &flat_config)
        .await
        .expect("native extraction without include_document_structure should succeed");
    let flat_result = crate::extraction::derive::derive_extraction_result(flat_result, true, OutputFormat::Plain);

    let structured_config = crate::core::config::ExtractionConfig {
        include_document_structure: true,
        ..Default::default()
    };
    let structured_result = extractor
        .extract_content(&content, "application/pdf", &structured_config)
        .await
        .expect("native extraction with include_document_structure should succeed");
    let structured_result =
        crate::extraction::derive::derive_extraction_result(structured_result, true, OutputFormat::Plain);

    assert_eq!(
        flat_result.tables.len(),
        1,
        "fixture must carry exactly one native table with include_document_structure off"
    );
    assert_eq!(
        structured_result.tables.len(),
        1,
        "fixture must carry exactly one native table with include_document_structure on"
    );

    let table = &structured_result.tables[0];
    for row in &table.cells {
        let rendered_row = row.join(" ");
        assert!(
            structured_result.content.lines().any(|line| line == rendered_row),
            "expected table row {rendered_row:?} to render as its own line in the structured Plain content; got {:?}",
            structured_result.content
        );
    }

    assert_ne!(
        flat_result.content, structured_result.content,
        "include_document_structure must change the rendered Plain content for a document with a native table"
    );
}

/// Tests form field extraction from a PDF.
/// Uses an existing test PDF rather than creating one programmatically.
/// This is a simple smoke test to verify that form field extraction works
/// and doesn't panic or crash the extraction pipeline.
/// Path to the vendored fillable-form fixture (AcroForm with text, button,
/// and choice fields).
#[cfg(feature = "pdf")]
fn form_test_document() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_documents/vendored/pdfium-render/form-test.pdf")
}

/// End-to-end: a real AcroForm PDF yields populated, correctly-typed
/// `form_fields` on the extractor's `InternalDocument` carrier.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_form_field_extraction_enabled() {
    let content = std::fs::read(form_test_document()).expect("read form-test.pdf fixture");

    let config = crate::core::config::ExtractionConfig {
        pdf_options: Some(crate::core::config::pdf::PdfConfig {
            extract_form_fields: true,
            ..Default::default()
        }),
        ..Default::default()
    };

    let internal_doc = PdfExtractor::new()
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("extraction must not fail");

    assert!(
        !internal_doc.form_fields.is_empty(),
        "AcroForm PDF must yield form fields, got none"
    );
    assert!(
        internal_doc.form_fields.iter().all(|f| !f.full_name.is_empty()),
        "every extracted field must have a full_name"
    );
    assert!(
        internal_doc
            .form_fields
            .iter()
            .any(|f| f.field_type == crate::types::FormFieldType::Text),
        "fixture has text fields; at least one must map to FormFieldType::Text"
    );
}

/// With `extract_form_fields = false`, the same AcroForm PDF yields no fields.
#[tokio::test]
#[cfg(feature = "pdf")]
async fn test_form_field_extraction_disabled() {
    let content = std::fs::read(form_test_document()).expect("read form-test.pdf fixture");

    let config = crate::core::config::ExtractionConfig {
        pdf_options: Some(crate::core::config::pdf::PdfConfig {
            extract_form_fields: false,
            ..Default::default()
        }),
        ..Default::default()
    };

    let internal_doc = PdfExtractor::new()
        .extract_content(&content, "application/pdf", &config)
        .await
        .expect("extraction must not fail");

    assert!(
        internal_doc.form_fields.is_empty(),
        "form fields must be empty when extract_form_fields is disabled"
    );
}

/// Tests that form fields are properly extracted and carried through the pipeline
/// using the InternalDocument.form_fields carrier (not metadata.additional).
/// This test manually constructs an InternalDocument with form_fields to verify
/// the carrier pattern works end-to-end, without relying on a complex PDF fixture.
#[test]
#[cfg(feature = "pdf")]
fn test_form_fields_carrier_via_internal_document() {
    let mut doc = InternalDocument::new("pdf");
    doc.mime_type = "application/pdf".to_string();

    doc.form_fields = vec![crate::types::PdfFormField {
        name: "full_name".to_string(),
        full_name: "form.full_name".to_string(),
        field_type: crate::types::FormFieldType::Text,
        value: Some("Ada Lovelace".to_string()),
        default_value: Some("Default Name".to_string()),
        flags: 0,
        page: None,
        bbox: None,
        max_length: None,
        tooltip: None,
    }];

    assert_eq!(doc.form_fields.len(), 1);
    let field = &doc.form_fields[0];
    assert_eq!(field.name, "full_name");
    assert_eq!(field.value, Some("Ada Lovelace".to_string()));
    assert_eq!(field.field_type, crate::types::FormFieldType::Text);

    assert!(
        doc.metadata.additional.get("_pdf_form_fields").is_none(),
        "metadata.additional should not contain _pdf_form_fields (leak check)"
    );

    let result =
        crate::extraction::derive::derive_extraction_result(doc, false, crate::core::config::OutputFormat::Plain);

    assert_eq!(result.form_fields.len(), 1);
    assert_eq!(result.form_fields[0].name, "full_name");
    assert_eq!(result.form_fields[0].value, Some("Ada Lovelace".to_string()));
}

/// Tests that with form_fields extraction disabled, the result contains no form fields.
#[test]
#[cfg(feature = "pdf")]
fn test_form_fields_disabled_no_leak() {
    let mut doc = InternalDocument::new("pdf");
    doc.mime_type = "application/pdf".to_string();

    assert!(doc.form_fields.is_empty());

    assert!(doc.metadata.additional.get("_pdf_form_fields").is_none());

    let result =
        crate::extraction::derive::derive_extraction_result(doc, false, crate::core::config::OutputFormat::Plain);

    assert!(result.form_fields.is_empty());
}

/// Two stroked boxes joined by a line that ends in a filled arrowhead, with a
/// label centred in each box. Mirrors the fixture in
/// `tests/diagram_dot_pdf.rs`, kept local so this unit test can inspect
/// `InternalDocument::diagrams` directly rather than through the renderer.
#[cfg(feature = "pdf")]
fn two_boxes_and_an_arrow_pdf() -> Vec<u8> {
    use lopdf::{Document, Object, Stream, dictionary};

    let content = b"1 w 0 0 0 RG
10 150 80 30 re S
10 20 80 30 re S
50 150 m 50 56 l S
46 56 m 54 56 l 50 50 l h f
BT /F1 12 Tf 30 160 Td (Alpha) Tj ET
BT /F1 12 Tf 30 30 Td (Beta) Tj ET
"
    .to_vec();

    let mut document = Document::with_version("1.5");
    let pages_id = document.new_object_id();
    let font_id = document.add_object(dictionary! {
        "Type" => "Font",
        "Subtype" => "Type1",
        "BaseFont" => "Helvetica",
    });
    let content_id = document.add_object(Stream::new(dictionary! {}, content));
    let page_id = document.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "Resources" => dictionary! { "Font" => dictionary! { "F1" => font_id } },
        "MediaBox" => vec![0.into(), 0.into(), 200.into(), 200.into()],
    });
    document.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );
    let catalog_id = document.add_object(dictionary! {
        "Type" => "Catalog",
        "Pages" => pages_id,
    });
    document.trailer.set("Root", catalog_id);

    let mut bytes = Vec::new();
    document.save_to(&mut bytes).expect("fixture PDF must save");
    bytes
}

/// #579 defect: recovery ran on every PDF regardless of the requested
/// output format, so an ordinary document paid for a second full text
/// pass on any page that happened to look graph-shaped. `diagrams` is
/// internal, so this has to be a unit test here rather than in the
/// integration suite, which can only see the rendered `content` string —
/// and `content` looks identical whether recovery ran and was discarded
/// or never ran at all, so it cannot tell the two apart.
#[cfg(feature = "pdf")]
#[tokio::test]
async fn recovery_is_skipped_unless_dot_output_is_requested() {
    let bytes = two_boxes_and_an_arrow_pdf();
    let extractor = PdfExtractor::new();

    let plain_doc = extractor
        .extract_content(&bytes, "application/pdf", &ExtractionConfig::default())
        .await
        .expect("plain extraction must succeed");
    assert_eq!(
        plain_doc.diagrams.len(),
        0,
        "diagram recovery must not run for non-dot output"
    );

    let dot_config = ExtractionConfig {
        output_format: crate::core::config::OutputFormat::Custom("dot".to_string()),
        ..Default::default()
    };
    let dot_doc = extractor
        .extract_content(&bytes, "application/pdf", &dot_config)
        .await
        .expect("dot extraction must succeed");
    assert_eq!(dot_doc.diagrams.len(), 1, "diagram recovery must run for dot output");
}

#[cfg(feature = "ocr-pipeline")]
#[test]
fn should_attach_distinct_preprocessing_metadata_to_each_pdf_page() {
    fn preprocessing(target_dpi: i32, final_dpi: i32) -> crate::types::ImagePreprocessingMetadata {
        crate::types::ImagePreprocessingMetadata {
            original_dimensions: (100, 200).into(),
            original_dpi: (72.0, 72.0).into(),
            target_dpi,
            scale_factor: 1.0,
            auto_adjusted: false,
            final_dpi,
            new_dimensions: None,
            resample_method: "LANCZOS3".to_string(),
            dimension_clamped: false,
            calculated_dpi: None,
            skipped_resize: true,
            resize_error: None,
        }
    }

    let mut pages = Some(vec![
        crate::types::PageContent {
            page_number: 1,
            content: "first".to_string(),
            tables: Vec::new(),
            image_indices: Vec::new(),
            image_preprocessing: None,
            hierarchy: None,
            is_blank: Some(false),
            layout_regions: None,
            speaker_notes: None,
            section_name: None,
            sheet_name: None,
            ocr_confidence: None,
            native_content: None,
        },
        crate::types::PageContent {
            page_number: 2,
            content: "second".to_string(),
            tables: Vec::new(),
            image_indices: Vec::new(),
            image_preprocessing: None,
            hierarchy: None,
            is_blank: Some(false),
            layout_regions: None,
            speaker_notes: None,
            section_name: None,
            sheet_name: None,
            ocr_confidence: None,
            native_content: None,
        },
    ]);
    let by_page = ahash::AHashMap::from([(1, preprocessing(150, 150)), (2, preprocessing(300, 300))]);
    let root_metadata = super::attach_pdf_preprocessing_metadata(&mut pages, &by_page);

    let pages = pages.expect("PDF OCR pages must remain present");
    assert_eq!(
        pages[0]
            .image_preprocessing
            .as_ref()
            .map(|metadata| metadata.target_dpi),
        Some(150)
    );
    assert_eq!(
        pages[1]
            .image_preprocessing
            .as_ref()
            .map(|metadata| metadata.target_dpi),
        Some(300)
    );
    assert_eq!(
        root_metadata.as_ref().map(|metadata| metadata.target_dpi),
        Some(150),
        "the singular compatibility field must deterministically report the first preprocessed page"
    );
}

/// A minimal 2-page PDF carrying real text content, for tests that only need page
/// structure and must not depend on the `test_documents/` corpus (see `read_test_fixture`).
#[cfg(feature = "pdf")]
fn build_two_page_text_pdf() -> Vec<u8> {
    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets: Vec<usize> = Vec::new();

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>\nendobj\n");

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
              /Contents 5 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n",
    );

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"4 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
              /Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>\nendobj\n",
    );

    let stream1 = "BT /F1 12 Tf 72 700 Td (Page one has real text content here.) Tj ET\n";
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "5 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
            stream1.len(),
            stream1
        )
        .as_bytes(),
    );

    let stream2 = "BT /F1 12 Tf 72 700 Td (Page two has different text content here.) Tj ET\n";
    offsets.push(pdf.len());
    pdf.extend_from_slice(
        format!(
            "6 0 obj\n<< /Length {} >>\nstream\n{}\nendstream\nendobj\n",
            stream2.len(),
            stream2
        )
        .as_bytes(),
    );

    offsets.push(pdf.len());
    pdf.extend_from_slice(
        b"7 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica \
              /Encoding /WinAnsiEncoding >>\nendobj\n",
    );

    let xref_pos = pdf.len();
    let total_objs = offsets.len() + 1;
    pdf.extend_from_slice(format!("xref\n0 {}\n", total_objs).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f\r\n");
    for &off in &offsets {
        pdf.extend_from_slice(format!("{off:010} 00000 n\r\n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            total_objs, xref_pos
        )
        .as_bytes(),
    );
    pdf
}

/// GH#1752: `metadata.pages` must be populated when `ocr_near_empty_fallback = Some(true)`
/// is set with no `ocr` block, at parity with what an `ocr` block alone already produced.
///
/// Before the fix, `page_structure_was_implicitly_tracked` in `extraction.rs` discarded the
/// page structure whenever `config.ocr` was `None`, regardless of the two settings GH#1752
/// gave the near-empty-fallback and scanned-page-quality-gate behaviours. Both configs here
/// set `pdf_options: None` and `pages: None` so `force_annotation_page_tracking` is `true`,
/// the only condition under which this site is reachable. Config X (an `ocr` block) is
/// checked first as a control -- asserting it is `Some` before checking config Y rules out a
/// fixture that simply never produces page structure at all, which would make the Y
/// assertion pass vacuously.
#[cfg(feature = "pdf")]
#[tokio::test]
async fn metadata_pages_populated_for_near_empty_fallback_without_ocr_block() {
    use crate::core::config::OcrConfig;

    let pdf = build_two_page_text_pdf();
    let extractor = PdfExtractor::new();

    let config_x = ExtractionConfig {
        ocr: Some(OcrConfig::default()),
        pdf_options: None,
        pages: None,
        ..ExtractionConfig::default()
    };
    let result_x = extractor
        .extract_content(&pdf, "application/pdf", &config_x)
        .await
        .expect("extraction with an `ocr` block must succeed");
    let result_x =
        crate::extraction::derive::derive_extraction_result(result_x, false, crate::core::config::OutputFormat::Plain);
    let pages_x = result_x
        .metadata
        .pages
        .as_ref()
        .expect("control: an `ocr` block alone must populate metadata.pages");

    let config_y = ExtractionConfig {
        ocr: None,
        ocr_near_empty_fallback: Some(true),
        pdf_options: None,
        pages: None,
        ..ExtractionConfig::default()
    };
    let result_y = extractor
        .extract_content(&pdf, "application/pdf", &config_y)
        .await
        .expect("extraction with ocr_near_empty_fallback must succeed");
    let result_y =
        crate::extraction::derive::derive_extraction_result(result_y, false, crate::core::config::OutputFormat::Plain);
    let pages_y = result_y.metadata.pages.as_ref().expect(
        "ocr_near_empty_fallback = Some(true) without an `ocr` block must also populate \
             metadata.pages (GH#1752)",
    );

    assert_eq!(
        pages_y.total_count, pages_x.total_count,
        "page counts must match between the two configurations"
    );
}
