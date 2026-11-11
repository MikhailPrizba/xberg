use super::{ocr::OcrGateOutcome, with_native_page_acceptance};
use crate::types::PageBoundary;

fn boundary(page_number: u32, byte_start: usize, byte_end: usize) -> PageBoundary {
    PageBoundary {
        page_number,
        byte_start,
        byte_end,
    }
}

#[test]
fn every_selected_page_keeps_the_existing_page_acceptance_route() {
    let bounds = [boundary(1, 0, 3), boundary(2, 5, 8)];
    assert_eq!(
        with_native_page_acceptance(
            OcrGateOutcome::RunFallback,
            &[1, 2],
            Some(&bounds),
            "one\n\ntwo",
            Some(2),
            &[]
        ),
        OcrGateOutcome::RunFallbackOnPages(vec![1, 2])
    );
}

#[test]
fn one_page_with_utf8_text_can_keep_native_text_after_ocr() {
    let bounds = [boundary(1, 1, 3)];
    assert_eq!(
        with_native_page_acceptance(OcrGateOutcome::RunFallback, &[1], Some(&bounds), " é\n", Some(1), &[]),
        OcrGateOutcome::RunFallbackOnPages(vec![1])
    );
}

#[test]
fn aggregate_or_incomplete_selection_keeps_whole_document_fallback() {
    let bounds = [boundary(1, 0, 3), boundary(2, 4, 7)];
    for (pages, count) in [(&[][..], Some(2)), (&[1][..], Some(2)), (&[1, 2][..], None)] {
        assert_eq!(
            with_native_page_acceptance(OcrGateOutcome::RunFallback, pages, Some(&bounds), "one two", count, &[]),
            OcrGateOutcome::RunFallback
        );
    }
    for bounds in [None, Some(&[][..])] {
        assert_eq!(
            with_native_page_acceptance(OcrGateOutcome::RunFallback, &[1], bounds, "one", Some(1), &[]),
            OcrGateOutcome::RunFallback
        );
    }
}

#[test]
fn no_native_text_keeps_whole_document_failure_handling() {
    assert_eq!(
        with_native_page_acceptance(
            OcrGateOutcome::RunFallback,
            &[1],
            Some(&[boundary(1, 0, 0)]),
            "",
            Some(1),
            &[]
        ),
        OcrGateOutcome::RunFallback
    );
}

#[test]
fn invalid_or_incomplete_boundaries_keep_whole_document_fallback() {
    let cases = [
        [boundary(1, 0, 4), boundary(2, 3, 7)],
        [boundary(1, 0, 3), boundary(2, 4, 8)],
        [boundary(1, 3, 0), boundary(2, 4, 7)],
        [boundary(2, 0, 3), boundary(1, 4, 7)],
        [boundary(1, 1, 3), boundary(2, 4, 7)],
        [boundary(1, 0, 3), boundary(2, 5, 7)],
        [boundary(1, 0, 3), boundary(2, 4, 6)],
    ];
    for bounds in cases {
        assert_eq!(
            with_native_page_acceptance(
                OcrGateOutcome::RunFallback,
                &[1, 2],
                Some(&bounds),
                "one two",
                Some(2),
                &[]
            ),
            OcrGateOutcome::RunFallback,
            "{bounds:?}"
        );
    }
    assert_eq!(
        with_native_page_acceptance(
            OcrGateOutcome::RunFallback,
            &[1],
            Some(&[boundary(1, 0, 1)]),
            "é",
            Some(1),
            &[]
        ),
        OcrGateOutcome::RunFallback
    );
}

#[test]
fn existing_skips_and_partial_page_selection_are_unchanged() {
    for (outcome, expected) in [
        (OcrGateOutcome::SkipNonText, OcrGateOutcome::SkipNonText),
        (OcrGateOutcome::SkipSubstantive, OcrGateOutcome::SkipSubstantive),
        (OcrGateOutcome::UseNative, OcrGateOutcome::UseNative),
        (
            OcrGateOutcome::RunFallbackOnPages(vec![1]),
            OcrGateOutcome::RunFallbackOnPages(vec![1]),
        ),
    ] {
        let actual = with_native_page_acceptance(outcome, &[1], Some(&[boundary(1, 0, 3)]), "one", Some(1), &[]);
        assert_eq!(actual, expected);
    }
}

#[test]
fn native_khmer_number_candidates_keep_the_full_document_route() {
    for digit in '\u{17e0}'..='\u{17e9}' {
        let native = format!("native {digit} text");
        let bounds = [boundary(1, 0, native.len())];
        assert_eq!(
            with_native_page_acceptance(OcrGateOutcome::RunFallback, &[1], Some(&bounds), &native, Some(1), &[]),
            OcrGateOutcome::RunFallback,
            "{digit}"
        );
    }
    for text in ["native 0123456789 text", "native \u{17df} \u{17ea} text"] {
        let bounds = [boundary(1, 0, text.len())];
        assert_eq!(
            with_native_page_acceptance(OcrGateOutcome::RunFallback, &[1], Some(&bounds), text, Some(1), &[]),
            OcrGateOutcome::RunFallbackOnPages(vec![1]),
            "{text}"
        );
    }
}

fn recognized_page(page_number: u32) -> crate::types::internal::InternalDocument {
    use crate::types::internal::{ElementKind, InternalDocument, InternalElement};
    let mut page = InternalDocument::new("pdf");
    let mut element = InternalElement::text(
        ElementKind::OcrText {
            level: crate::types::OcrElementLevel::Line,
        },
        "accepted OCR text",
        0,
    );
    element.page = Some(page_number);
    element.ocr_geometry = Some(crate::types::OcrBoundingGeometry::Rectangle {
        left: 12,
        top: 34,
        width: 120,
        height: 20,
    });
    element.ocr_confidence = Some(crate::types::OcrConfidence {
        detection: Some(0.95),
        recognition: 0.93,
    });
    page.push_element(element);
    page.ocr_coordinate_frame = Some(crate::types::internal::OcrPageCoordinateFrame::new(
        page_number,
        640,
        480,
    ));
    page.metadata.additional.insert(
        crate::ocr_metadata_keys::OCR_PAGE_SEGMENTATION_MODES_METADATA_KEY.into(),
        serde_json::json!([{"page_number": page_number, "psm": 6}]),
    );
    page
}

#[test]
fn plain_page_acceptance_keeps_only_accepted_segmentation_modes_in_page_order() {
    let pages = ahash::AHashMap::from([
        (3, recognized_page(3)),
        (2, recognized_page(2)),
        (1, recognized_page(1)),
    ]);
    let replacements = ahash::AHashMap::from([(2, "second".into()), (1, "first".into())]);
    let bounds = [boundary(1, 0, 5), boundary(2, 7, 13)];
    let (document, _, _) = super::select_pdf_document(
        super::ExtractionMethod::Mixed,
        "first\n\nsecond",
        "application/pdf",
        None,
        None,
        Some(&replacements),
        Some(&pages),
        Some(&bounds),
        &crate::OutputFormat::Plain,
    );
    assert_eq!(crate::rendering::render_plain(&document).trim(), "first\n\nsecond");
    assert_eq!(
        document
            .metadata
            .additional
            .get(crate::ocr_metadata_keys::OCR_PAGE_SEGMENTATION_MODES_METADATA_KEY),
        Some(&serde_json::json!([{"page_number": 1, "psm": 6}, {"page_number": 2, "psm": 6}]))
    );
}

#[test]
fn accepted_ocr_text_without_a_prebuilt_sidecar_keeps_its_derived_spatial_element() {
    let page = recognized_page(2);
    let control = crate::extraction::derive::derive_extraction_result(page.clone(), false, crate::OutputFormat::Plain);
    assert_eq!(control.ocr_elements.as_ref().unwrap().len(), 1);
    let pages = ahash::AHashMap::from([(2, page)]);
    let replacements = ahash::AHashMap::from([(2, "accepted OCR text".into())]);
    let (mut document, _, _) = super::select_pdf_document(
        super::ExtractionMethod::Mixed,
        "accepted OCR text",
        "application/pdf",
        None,
        None,
        Some(&replacements),
        Some(&pages),
        None,
        &crate::OutputFormat::Plain,
    );
    document.prebuilt_ocr_elements = Some(super::accepted_mixed_ocr_elements(&pages));
    let actual = crate::extraction::derive::derive_extraction_result(document, false, crate::OutputFormat::Plain);
    assert_eq!(actual.content, control.content);
    assert_eq!(
        serde_json::to_value(&actual.ocr_elements).unwrap(),
        serde_json::to_value(&control.ocr_elements).unwrap()
    );
    let frames = super::accepted_mixed_ocr_coordinate_frames(&pages);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].page_number, 2);
    assert_eq!((frames[0].width, frames[0].height), (640, 480));
}

#[test]
fn an_explicit_empty_ocr_sidecar_does_not_reconstruct_filtered_elements() {
    let mut page = recognized_page(1);
    page.prebuilt_ocr_elements = Some(Vec::new());
    let pages = ahash::AHashMap::from([(1, page)]);
    assert!(super::accepted_mixed_ocr_elements(&pages).is_empty());
    assert!(super::accepted_mixed_ocr_coordinate_frames(&pages).is_empty());
}

#[test]
fn mixed_rejected_page_keeps_native_tables_and_excludes_its_ocr_payload() {
    let bounds = [boundary(1, 0, 11), boundary(2, 13, 21)];
    let mut native = super::flat_pdf_document("native text\n\nold text", "application/pdf", Some(&bounds));
    native.tables.push(crate::types::Table {
        markdown: "| native table |".into(),
        page_number: 1,
        ..Default::default()
    });
    let mut rejected = recognized_page(1);
    rejected.elements[0].text = "rejected OCR page one".into();
    rejected.prebuilt_ocr_elements = Some(vec![crate::types::OcrElement {
        text: "rejected OCR sidecar".into(),
        page_number: 1,
        ..Default::default()
    }]);
    let mut accepted = recognized_page(2);
    accepted.metadata.additional.clear();
    let pages = ahash::AHashMap::from([(1, rejected), (2, accepted)]);
    let replacements = ahash::AHashMap::from([(2, "accepted OCR text".into())]);
    let (document, origin, _) = super::select_pdf_document(
        super::ExtractionMethod::Mixed,
        "native text\n\naccepted OCR text",
        "application/pdf",
        Some(native),
        None,
        Some(&replacements),
        Some(&pages),
        Some(&bounds),
        &crate::OutputFormat::Plain,
    );
    assert_eq!(origin, super::PdfDocumentOrigin::Mixed);
    assert_eq!(document.tables[0].markdown, "| native table |");
    assert!(
        !document
            .metadata
            .additional
            .contains_key(crate::ocr_metadata_keys::OCR_PAGE_SEGMENTATION_MODES_METADATA_KEY)
    );
    let actual = crate::extraction::derive::derive_extraction_result(document, false, crate::OutputFormat::Plain);
    let elements = actual
        .ocr_elements
        .as_ref()
        .expect("the accepted page carries OCR text");
    assert_eq!(elements.len(), 1);
    assert_eq!(elements[0].page_number, 2);
    assert_eq!(elements[0].text, "accepted OCR text");
    assert!(actual.content.contains("native text"));
    assert!(!actual.content.contains("rejected OCR"));
}

#[test]
fn non_plain_partial_mixed_output_keeps_fallback_elements_and_frames() {
    let bounds = [boundary(1, 0, 11), boundary(2, 13, 21)];
    let native = super::flat_pdf_document("native text\n\nold text", "application/pdf", Some(&bounds));
    let pages = ahash::AHashMap::from([(2, recognized_page(2))]);
    let replacements = ahash::AHashMap::from([(2, "accepted OCR text".into())]);
    let (document, origin, _) = super::select_pdf_document(
        super::ExtractionMethod::Mixed,
        "native text\n\naccepted OCR text",
        "application/pdf",
        Some(native),
        None,
        Some(&replacements),
        Some(&pages),
        Some(&bounds),
        &crate::OutputFormat::Markdown,
    );
    assert_eq!(origin, super::PdfDocumentOrigin::Mixed);
    let actual = crate::extraction::derive::derive_extraction_result(document, false, crate::OutputFormat::Markdown);
    assert!(actual.content.contains("native text"));
    let expected = super::accepted_mixed_ocr_elements(&pages);
    assert_eq!(expected.len(), 1);
    assert_eq!(expected[0].page_number, 2);
    assert_eq!(
        serde_json::to_value(&actual.ocr_elements).unwrap(),
        serde_json::to_value(Some(&expected)).unwrap()
    );
    let frames = super::accepted_mixed_ocr_coordinate_frames(&pages);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].page_number, 2);
    assert_eq!((frames[0].width, frames[0].height), (640, 480));
}

#[test]
fn fabricated_native_mappings_keep_the_whole_document_outcome() {
    let bounds = [boundary(1, 0, 3), boundary(2, 5, 8)];
    for fabricated in [&[1][..], &[2][..], &[37][..]] {
        assert_eq!(
            with_native_page_acceptance(
                OcrGateOutcome::RunFallback,
                &[1, 2],
                Some(&bounds),
                "one\n\ntwo",
                Some(2),
                fabricated
            ),
            OcrGateOutcome::RunFallback,
            "fabricated pages {fabricated:?} must retain the original document route",
        );
    }
    assert_eq!(
        with_native_page_acceptance(
            OcrGateOutcome::RunFallback,
            &[1, 2],
            Some(&bounds),
            "one\n\ntwo",
            Some(2),
            &[]
        ),
        OcrGateOutcome::RunFallbackOnPages(vec![1, 2]),
    );
}
