//! A replaced page keeps its readable text layer first when its OCR text is judged recognition
//! noise, or when its OCR text confirms a layer that passes the per-page quality check.

use super::document::{clean_native_page_layers, layer_lines_absent_from_ocr};
use super::scoring::native_layer_is_clean;
use crate::core::config::OcrQualityThresholds;
use crate::types::PageBoundary;

#[path = "native_retention_stress_tests.rs"]
mod native_retention_stress_tests;

#[path = "photo_layer_tests/review_2505.rs"]
mod review_2505;

#[test]
fn should_call_a_short_readable_title_a_clean_layer() {
    assert!(native_layer_is_clean("Meeting Notes", &OcrQualityThresholds::default()));
}

#[test]
fn should_not_call_empty_or_symbol_only_text_a_clean_layer() {
    let thresholds = OcrQualityThresholds::default();
    assert!(!native_layer_is_clean("   \n ", &thresholds));
    assert!(!native_layer_is_clean("-- | ~ ..", &thresholds));
}

#[test]
fn should_require_complete_source_rows_before_omitting_native_lines_without_geometry() {
    let ocr = "intro\nMEETING  notes!\nbody";
    assert_eq!(
        layer_lines_absent_from_ocr("Meeting Notes", ocr).as_deref(),
        Some("Meeting Notes")
    );
    assert_eq!(layer_lines_absent_from_ocr("Meeting Notes", "MEETING   notes"), None);
    assert_eq!(
        layer_lines_absent_from_ocr("Meeting Notes\nBudget 2026", ocr).as_deref(),
        Some("Meeting Notes\nBudget 2026")
    );
}

/// The letters of a layer word that occur in the OCR text only across a word boundary, or in a
/// longer word, are not evidence that the OCR read the word. ~keep
#[test]
fn should_keep_a_layer_word_that_the_ocr_text_holds_only_across_a_word_boundary() {
    assert_eq!(
        layer_lines_absent_from_ocr("Quillmarrow", "the quill marrow road").as_deref(),
        Some("Quillmarrow")
    );
    assert_eq!(
        layer_lines_absent_from_ocr("Quill", "the quillmarrow road").as_deref(),
        Some("Quill")
    );
    assert_eq!(
        layer_lines_absent_from_ocr("Quill Marrow", "the quill marrow road").as_deref(),
        Some("Quill Marrow")
    );
    assert_eq!(
        layer_lines_absent_from_ocr("Plot 100", "plot 1OO").as_deref(),
        Some("Plot 100")
    );
}

#[test]
fn should_collect_layers_only_for_requested_pages_with_one_boundary() {
    let text = "Meeting Notes--|~..Agenda Items";
    let boundaries = [
        PageBoundary {
            byte_start: 0,
            byte_end: 13,
            page_number: 1,
        },
        PageBoundary {
            byte_start: 13,
            byte_end: 19,
            page_number: 2,
        },
        PageBoundary {
            byte_start: 19,
            byte_end: 31,
            page_number: 3,
        },
        PageBoundary {
            byte_start: 19,
            byte_end: 31,
            page_number: 3,
        },
    ];
    let thresholds = OcrQualityThresholds::default();
    let all = clean_native_page_layers(text, &boundaries, None, &thresholds);
    assert_eq!(all.get(&1).map(String::as_str), Some("Meeting Notes"));
    assert!(!all.contains_key(&2), "a symbol-only layer is not clean");
    assert!(!all.contains_key(&3), "a page with two boundaries is skipped");
    let only_two = clean_native_page_layers(
        text,
        &boundaries,
        Some(&std::collections::HashSet::from([2])),
        &thresholds,
    );
    assert!(only_two.is_empty(), "page 1 is not requested and page 2 is not clean");
}

#[test]
fn should_put_the_layer_before_the_structured_page_and_shift_relationships() {
    use super::document::{flat_ocr_page_document, insert_layer_into_document};
    use crate::types::internal::{Relationship, RelationshipKind, RelationshipTarget};

    let mut page = flat_ocr_page_document("first block\n\nsecond block");
    page.relationships.push(Relationship {
        source: 0,
        target: RelationshipTarget::Index(1),
        kind: RelationshipKind::InternalLink,
    });
    insert_layer_into_document(&mut page, None, "Meeting Notes");
    let texts: Vec<&str> = page.elements.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(texts, ["Meeting Notes", "first block", "second block"]);
    assert_eq!(page.relationships[0].source, 1);
    assert!(matches!(page.relationships[0].target, RelationshipTarget::Index(2)));
}

#[test]
fn should_put_the_layer_first_on_the_noise_page_of_a_whole_document_run_only() {
    use super::document::{flat_ocr_page_document, put_layers_first_in_run};
    use super::scoring::OcrPageNoiseVerdict;
    use crate::core::config::ExtractionConfig;
    use crate::types::internal::{Relationship, RelationshipKind, RelationshipTarget};

    let verdict = |page_index: usize, discarded: bool| OcrPageNoiseVerdict {
        page_index,
        low_confidence: true,
        fragmented_noise: false,
        dictionary_noise: false,
        fragmented_word_ratio: 0.0,
        word_count: 3,
        mean_confidence: Some(0.2),
        dict_invalid_word_ratio: None,
        discarded,
    };
    let mut page_texts = vec![
        "page one text".to_string(),
        "xq zv wj".to_string(),
        "page three text".to_string(),
    ];
    let mut text = page_texts.join("\n\n");
    let mut doc = flat_ocr_page_document("page one text");
    for (page, body) in [(2u32, "xq zv wj"), (3, "page three text")] {
        let mut part = flat_ocr_page_document(body);
        part.elements[0].page = Some(page);
        doc.elements.append(&mut part.elements);
    }
    doc.elements[0].page = Some(1);
    doc.relationships.push(Relationship {
        source: 2,
        target: RelationshipTarget::Index(0),
        kind: RelationshipKind::InternalLink,
    });
    let layers = ahash::AHashMap::from([
        (1, "Page One".to_string()),
        (2, "Meeting Notes".to_string()),
        (3, "Page Three".to_string()),
    ]);
    // Page 1 has no verdict, page 3's verdict was discarded: only page 2 gets its layer.
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        Some(&mut doc),
        super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[verdict(1, false), verdict(2, true)],
        0,
        &ExtractionConfig::default(),
    );
    assert_eq!(page_texts[1], "Meeting Notes\n\nxq zv wj");
    assert_eq!(page_texts[0], "page one text");
    assert_eq!(page_texts[2], "page three text");
    assert_eq!(text, "page one text\n\nMeeting Notes\n\nxq zv wj\n\npage three text");
    let texts: Vec<&str> = doc.elements.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(texts, ["page one text", "Meeting Notes", "xq zv wj", "page three text"]);
    assert_eq!(doc.elements[1].page, Some(2));
    assert_eq!(
        doc.relationships[0].source, 3,
        "a relationship past the insertion point shifts"
    );
    assert!(
        matches!(doc.relationships[0].target, RelationshipTarget::Index(0)),
        "one before it stays"
    );
}

/// The two existing text judgements (language plausibility and the recognition-noise ratio)
/// both abstain on a one-line layer, so neither can tell invented words from a real title.
/// The layer gate therefore cannot reuse them to reject a short invented layer. ~keep
#[test]
fn should_show_that_the_existing_judgements_abstain_on_a_one_line_layer() {
    use super::plausibility::{PlausibilityVerdict, evaluate_text_plausibility};
    use super::scoring::is_ocr_recognition_noise;

    let thresholds = OcrQualityThresholds::default();
    for line in ["Meeting Notes", "Qxzv Kjwq Zxcv Wqjx Vkzq Jqxz"] {
        assert_eq!(
            evaluate_text_plausibility(line, &thresholds),
            PlausibilityVerdict::NotEvaluated
        );
        assert!(!is_ocr_recognition_noise(line, &thresholds));
    }
}

/// Two pages with identical OCR text and a layer only on the later page: the layer goes on page 2
/// in the flat text, the page texts and the document alike, never on page 1. ~keep
#[test]
fn should_put_the_layer_on_the_later_of_two_identical_ocr_pages() {
    use super::document::{flat_ocr_page_document, put_layers_first_in_run};
    use super::scoring::OcrPageNoiseVerdict;
    use crate::core::config::ExtractionConfig;

    let verdict = |page_index: usize| OcrPageNoiseVerdict {
        page_index,
        low_confidence: true,
        fragmented_noise: false,
        dictionary_noise: false,
        fragmented_word_ratio: 0.0,
        word_count: 3,
        mean_confidence: Some(0.2),
        dict_invalid_word_ratio: None,
        discarded: false,
    };
    let mut page_texts = vec!["xq zv wj".to_string(), "xq zv wj".to_string()];
    let mut text = page_texts.join("\n\n");
    let mut doc = flat_ocr_page_document("xq zv wj");
    doc.elements[0].page = Some(1);
    let mut second = flat_ocr_page_document("xq zv wj");
    second.elements[0].page = Some(2);
    doc.elements.append(&mut second.elements);
    let layers = ahash::AHashMap::from([(2, "Meeting Notes".to_string())]);
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        Some(&mut doc),
        super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[verdict(0), verdict(1)],
        0,
        &ExtractionConfig::default(),
    );
    assert_eq!(page_texts, ["xq zv wj", "Meeting Notes\n\nxq zv wj"]);
    assert_eq!(text, "xq zv wj\n\nMeeting Notes\n\nxq zv wj");
    let texts: Vec<&str> = doc.elements.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(texts, ["xq zv wj", "Meeting Notes", "xq zv wj"]);

    let mut marked_pages = vec!["xq zv wj".to_string(), "xq zv wj".to_string()];
    let marked_config = ExtractionConfig {
        pages: Some(crate::core::config::PageConfig {
            insert_page_markers: true,
            marker_format: "<p{page_num}>".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut marked = "<p1>xq zv wj<p2>xq zv wj".to_string();
    put_layers_first_in_run(
        &mut marked,
        &mut marked_pages,
        None,
        super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[verdict(1)],
        0,
        &marked_config,
    );
    assert_eq!(marked, "<p1>xq zv wj<p2>Meeting Notes\n\nxq zv wj");
}

/// Six lines of plain prose: a layer that passes the per-page quality check. ~keep
const HARBOUR_LAYER: [&str; 6] = [
    "The committee met on Tuesday to review the annual budget for the harbour district.",
    "Members agreed that the proposal was sound and asked for a revised timetable.",
    "The treasurer will prepare the final report before the next quarterly meeting.",
    "Repairs to the north pier start in March and finish before the summer season.",
    "A public notice about the closure goes to every household along the coast road.",
    "The next meeting takes place in the spring at the library on Market Street.",
];

/// Shift every ASCII letter three places, as a wrong `/ToUnicode` map does: the shape of the
/// text stays clean and no word is a word of the true text. ~keep
fn rot3(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            'a'..='z' => (((c as u8 - b'a' + 3) % 26) + b'a') as char,
            'A'..='Z' => (((c as u8 - b'A' + 3) % 26) + b'A') as char,
            _ => c,
        })
        .collect()
}

/// A line with its longest word misread: one letter added. ~keep
fn misread_longest_word(line: &str) -> String {
    let stem = |word: &str| word.trim_end_matches(|c: char| !c.is_alphanumeric()).to_string();
    let longest = line
        .split(' ')
        .map(stem)
        .max_by_key(|word| word.len())
        .unwrap_or_default();
    line.replacen(&longest, &format!("{longest}x"), 1)
}

fn layer_first_fixture(layer: &str) -> super::document::LayerFirst {
    super::document::LayerFirst {
        lines: layer.to_string(),
    }
}

fn row_box(row: usize) -> super::document::LineBox {
    let top = row as f32 * 12.0;
    super::document::LineBox {
        left: 0.0,
        top,
        right: 500.0,
        bottom: top + 10.0,
    }
}

/// The non-empty lines of a text, each on the row of `rows` with its index. ~keep
fn lines_at_rows(text: &str, rows: &[usize]) -> Vec<super::document::LineAt> {
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    assert_eq!(lines.len(), rows.len(), "one row per non-empty line");
    lines
        .iter()
        .zip(rows)
        .map(|(line, row)| super::document::LineAt::new(line, row_box(*row)))
        .collect()
}

/// The boxes of a page whose layer lines sit on rows 0.. in order, and whose OCR lines are given. ~keep
fn page_boxes(layer: &[&str], ocr: &[super::document::LineAt]) -> super::document::PageLineBoxes {
    super::document::PageLineBoxes {
        layer: layer
            .iter()
            .enumerate()
            .map(|(row, line)| super::document::LineAt::new(line, row_box(row)))
            .collect(),
        ocr: ocr.to_vec(),
    }
}

/// Six lines of numbers: a layer that passes the per-page quality check and has no word with a
/// letter. ~keep
const LEDGER_LAYER: [&str; 6] = [
    "1042 2087 3315 4471 5530 6694",
    "7120 8243 9356 1468 2579 3681",
    "4792 5804 6915 7026 8137 9248",
    "1359 2460 3571 4682 5793 6804",
    "7915 8026 9137 1248 2359 3460",
    "4571 5682 6793 7804 8915 9026",
];

/// Eight lines of invented minutes: the layer of the ownership tests. ~keep
const MINUTES: [&str; 8] = [
    "Harbour committee minutes",
    "Members present at the meeting",
    "Repairs to the north pier",
    "Total 100",
    "Treasurer report accepted",
    "Library hours extended",
    "Public notice posted",
    "Next meeting in spring",
];

/// The merge of one confirmed page whose OCR lines sit on `ocr_rows`, as the page text. ~keep
fn merge_at_rows(layer: &[&str], ocr: &str, ocr_rows: &[usize]) -> String {
    use super::document::put_layers_first_in_run;
    let mut page_texts = vec![ocr.to_string()];
    let mut text = ocr.to_string();
    let layers = ahash::AHashMap::from([(1u32, layer.join("\n"))]);
    let boxes = ahash::AHashMap::from([(1u32, page_boxes(layer, &lines_at_rows(ocr, ocr_rows)))]);
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        None,
        super::document::PageLayers {
            lines: &layers,
            boxes: &boxes,
        },
        &[],
        0,
        &crate::core::config::ExtractionConfig::default(),
    );
    page_texts.remove(0)
}

fn count_line(page: &str, needle: &str) -> usize {
    page.lines().filter(|line| line.trim() == needle).count()
}

#[path = "photo_layer_tests/confirmed_layers.rs"]
mod confirmed_layers;

#[path = "photo_layer_tests/occurrence_retention.rs"]
mod occurrence_retention;
