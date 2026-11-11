use super::*;

/// The OCR text of a page that the noise gate discards is empty. In a whole-document run the
/// layer does not go in front of that empty text: the page stays empty. Another page of the same
/// run, not discarded, gets its layer, so the layer words differ from the OCR text and the
/// whole-word skip cannot hide the discarded case. ~keep
#[test]
fn should_not_put_a_layer_on_a_page_the_noise_gate_discarded() {
    use super::super::document::put_layers_first_in_run;
    use super::super::scoring::OcrPageNoiseVerdict;
    use crate::core::config::ExtractionConfig;

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
    let mut page_texts = vec![String::new(), "xq zv wj".to_string()];
    let mut text = page_texts.join("\n\n");
    let layers = ahash::AHashMap::from([(1, "Meeting Notes".to_string()), (2, "Budget Report".to_string())]);
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        None,
        super::super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[verdict(0, true), verdict(1, false)],
        0,
        &ExtractionConfig::default(),
    );
    assert_eq!(page_texts, ["", "Budget Report\n\nxq zv wj"]);
    assert_eq!(text, "\n\nBudget Report\n\nxq zv wj");
}

#[test]
fn should_confirm_a_layer_when_the_ocr_text_reads_most_of_its_words() {
    use super::super::document::ocr_confirms_layer;

    let thresholds = OcrQualityThresholds::default();
    let layer = HARBOUR_LAYER.join("\n");
    let partial = [HARBOUR_LAYER[0], HARBOUR_LAYER[1], HARBOUR_LAYER[2], HARBOUR_LAYER[4]].join("\n");
    assert!(ocr_confirms_layer(&layer, &partial, &thresholds));
    assert!(
        !ocr_confirms_layer(&rot3(&layer), &layer, &thresholds),
        "a wrong-letter layer shares no word with the OCR read"
    );
    assert!(
        !ocr_confirms_layer(&layer, HARBOUR_LAYER[5], &thresholds),
        "one line of six is not most of the words"
    );
    assert!(
        !ocr_confirms_layer("Meeting Notes", "Meeting Notes", &thresholds),
        "a layer that fails the per-page quality check is never confirmed"
    );
}

/// A whole-document run (#2439): a page with no noise verdict whose OCR text confirms its layer
/// gets the layer lines the OCR text lacks first. A wrong-letter layer and a layer the OCR text
/// does not confirm stay as the OCR text. ~keep
#[test]
fn should_put_a_confirmed_layer_first_on_a_page_of_a_whole_document_run() {
    use super::super::document::put_layers_first_in_run;
    use crate::core::config::ExtractionConfig;

    let layer = HARBOUR_LAYER.join("\n");
    let partial = [HARBOUR_LAYER[0], HARBOUR_LAYER[1], HARBOUR_LAYER[2], HARBOUR_LAYER[4]].join("\n");
    let mut page_texts = vec![partial.clone(), layer.clone(), HARBOUR_LAYER[5].to_string()];
    let mut text = page_texts.join("\n\n");
    let layers = ahash::AHashMap::from([(1, layer.clone()), (2, rot3(&layer)), (3, layer.clone())]);
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        None,
        super::super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[],
        0,
        &ExtractionConfig::default(),
    );
    let first = format!("{}\n{}\n\n{partial}", HARBOUR_LAYER[3], HARBOUR_LAYER[5]);
    assert_eq!(page_texts, [first.clone(), layer.clone(), HARBOUR_LAYER[5].to_string()]);
    assert_eq!(text, format!("{first}\n\n{layer}\n\n{}", HARBOUR_LAYER[5]));
}

/// A whole-document run (#2439): a page whose noise verdict discarded its OCR text keeps no layer,
/// even when text sits on it later and that text would confirm the layer. ~keep
#[test]
fn should_not_put_a_confirmed_layer_on_a_page_the_noise_gate_discarded() {
    use super::super::document::put_layers_first_in_run;
    use super::super::scoring::OcrPageNoiseVerdict;
    use crate::core::config::ExtractionConfig;

    let partial = [HARBOUR_LAYER[0], HARBOUR_LAYER[1], HARBOUR_LAYER[2], HARBOUR_LAYER[4]].join("\n");
    let mut page_texts = vec![partial.clone()];
    let mut text = partial.clone();
    let layers = ahash::AHashMap::from([(1, HARBOUR_LAYER.join("\n"))]);
    let discarded = OcrPageNoiseVerdict {
        page_index: 0,
        low_confidence: true,
        fragmented_noise: false,
        dictionary_noise: false,
        fragmented_word_ratio: 0.0,
        word_count: 3,
        mean_confidence: Some(0.2),
        dict_invalid_word_ratio: None,
        discarded: true,
    };
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        None,
        super::super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[discarded],
        0,
        &ExtractionConfig::default(),
    );
    assert_eq!(page_texts, [partial.as_str()]);
    assert_eq!(text, partial);
}

#[test]
fn should_restore_native_rows_without_deleting_unproved_ocr_alternatives() {
    use super::super::document::{layer_lines_to_put_first, text_with_layer_first};
    let layer = HARBOUR_LAYER.join("\n");
    let misread: Vec<String> = HARBOUR_LAYER.iter().map(|line| misread_longest_word(line)).collect();
    let read = misread.join("\n");
    let boxes = page_boxes(&HARBOUR_LAYER, &lines_at_rows(&read, &[0, 1, 2, 3, 4, 5]));
    let first = layer_lines_to_put_first(&layer, &read, false, &OcrQualityThresholds::default(), Some(&boxes))
        .expect("all source rows lack complete literal OCR coverage");
    assert_eq!(first.lines, layer);
    assert_eq!(text_with_layer_first(&read, &first), format!("{layer}\n\n{read}"));
}

/// A whole-document run (#2439): a layer of numbers only. The OCR text reads three of the six
/// lines, so it confirms the layer and the other three lines come back. Numbers that the OCR text
/// does not hold, and too few numbers, never confirm a layer. ~keep
#[test]
fn should_confirm_a_number_only_layer_against_the_ocr_text() {
    use super::super::document::{ocr_confirms_layer, put_layers_first_in_run};
    use crate::core::config::ExtractionConfig;

    let thresholds = OcrQualityThresholds::default();
    let layer = LEDGER_LAYER.join("\n");
    let read = LEDGER_LAYER[..3].join("\n");
    assert!(
        ocr_confirms_layer(&layer, &read, &thresholds),
        "the OCR text holds half the numbers"
    );
    let other = "9911 8822 7733 6644 5555 4466\n3377 2288 1199 9900 8811 7722";
    assert!(
        !ocr_confirms_layer(&layer, other, &thresholds),
        "garbage shares no number"
    );
    assert!(
        !ocr_confirms_layer(&layer, &LEDGER_LAYER[5].replace("4571", "1111"), &thresholds),
        "one line of six is not half of the numbers"
    );
    assert!(
        !ocr_confirms_layer("2024 17", "2024 17", &thresholds),
        "two numbers are too few to confirm a layer"
    );

    let mut page_texts = vec![read.clone()];
    let mut text = read.clone();
    let layers = ahash::AHashMap::from([(1, layer)]);
    put_layers_first_in_run(
        &mut text,
        &mut page_texts,
        None,
        super::super::document::PageLayers {
            lines: &layers,
            boxes: &ahash::AHashMap::default(),
        },
        &[],
        0,
        &ExtractionConfig::default(),
    );
    assert_eq!(
        text,
        format!(
            "{}\n{}\n{}\n\n{read}",
            LEDGER_LAYER[3], LEDGER_LAYER[4], LEDGER_LAYER[5]
        )
    );
}

/// A layer of numbers with only two distinct numbers on six full lines: it passes the per-page
/// quality check, and the OCR text holds both numbers by chance. Two numbers never confirm a layer
/// (#2439), and three do. ~keep
#[test]
fn should_need_three_distinct_numbers_to_confirm_a_number_only_layer() {
    use super::super::document::ocr_confirms_layer;

    let thresholds = OcrQualityThresholds::default();
    let line = |numbers: &str| [numbers; 6].join(" ");
    let two = vec![line("1042 2087"); 6].join("\n");
    let three = vec![line("1042 2087 3315"); 6].join("\n");
    assert!(
        !ocr_confirms_layer(&two, &two, &thresholds),
        "two distinct numbers are too few to confirm a layer"
    );
    assert!(
        ocr_confirms_layer(&three, &three, &thresholds),
        "three distinct numbers confirm a layer"
    );
}

/// another layer line that differs from it by one word. The layer holds a line and a near twin of
/// it; the OCR text reads the first exactly and lacks the twin. The twin comes back, and the exact
/// OCR line stays. ~keep
#[test]
fn should_not_drop_an_exact_ocr_line_as_the_copy_of_its_near_twin_in_the_layer() {
    use super::super::document::layer_lines_to_put_first;

    let thresholds = OcrQualityThresholds::default();
    let twin = misread_longest_word(HARBOUR_LAYER[0]);
    let layer = format!("{}\n{twin}", HARBOUR_LAYER.join("\n"));
    let ocr_text = HARBOUR_LAYER[..4].join("\n");
    // The twin sits on the first line's row: it owns the exact OCR line's place. ~keep
    let mut boxes = page_boxes(&HARBOUR_LAYER, &lines_at_rows(&ocr_text, &[0, 1, 2, 3]));
    boxes.layer.push(super::super::document::LineAt::new(&twin, row_box(0)));
    let first = layer_lines_to_put_first(&layer, &ocr_text, false, &thresholds, Some(&boxes))
        .expect("the OCR text confirms the layer, and the twin and two more lines are absent from it");
    assert!(
        first.lines.lines().any(|line| line == twin),
        "the twin comes back: {:?}",
        first.lines
    );
    assert!(super::super::document::text_with_layer_first(&ocr_text, &first).ends_with(&ocr_text));
}
