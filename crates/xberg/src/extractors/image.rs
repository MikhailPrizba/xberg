//! Image extractors for various image formats.

use crate::Result;
use crate::core::config::ExtractionConfig;
use crate::extraction::image::extract_image_metadata_with_security_limits;
use crate::plugins::{InternalDocumentExtractor, Plugin};
use crate::types::internal::InternalDocument;
use crate::types::internal_builder::InternalDocumentBuilder;
use crate::types::metadata::Metadata;
use async_trait::async_trait;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const WHOLE_IMAGE_TESSERACT_PSM: i32 = 11;

// Tesseract's automatic layout modes can hang under the single-threaded WASM
// runtime, so retain the existing single-block default there. ~keep
#[cfg(all(target_arch = "wasm32", feature = "ocr-pipeline"))]
const WHOLE_IMAGE_TESSERACT_PSM: i32 = 6;

#[cfg(feature = "ocr-pipeline")]
const VERTICAL_BLOCK_TESSERACT_PSM: i32 = 5;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const SPARSE_IMAGE_OCR_WORD_LIMIT: usize = 20;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const SPARSE_IMAGE_OCR_FALLBACK_PSM: i32 = 3;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_SPARSE_IMAGE_OCR_UNIQUE_TOKEN_LIMIT: usize = 32;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_SPARSE_IMAGE_OCR_TOTAL_TOKEN_LIMIT: usize = 48;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_SPARSE_IMAGE_OCR_FALLBACK_PSM: i32 = 6;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_IDENTIFIER_RECOVERY_MAX_CROPS: usize = 6;
#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_IDENTIFIER_RECOVERY_MIN_CONFIDENCE: f64 = 0.25;
#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_IDENTIFIER_RECOVERY_MAX_CROP_PIXELS: u64 = 500_000;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_SPARSE_IMAGE_OCR_MAX_CONFIDENCE_DROP: f64 = 0.10;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const RTL_SPARSE_IMAGE_OCR_MIN_PRIMARY_TOKEN_RETENTION: f64 = 0.80;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const SPARSE_IMAGE_OCR_MIN_WORD_CONFIDENCE: f64 = 0.30;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const SPARSE_IMAGE_OCR_MAX_LOW_CONFIDENCE_RATIO: f64 = 0.30;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
const SPARSE_IMAGE_OCR_CONFIDENCE_PERCENTILE: f64 = 0.10;

#[cfg(test)]
const MIN_LAYOUT_OCR_ALPHANUMERIC_TOKEN_RETENTION: f64 = 0.80;

#[cfg(feature = "ocr-pipeline")]
const NORMALIZED_PNG_ENCODE_BYTES_PER_PIXEL: u64 = 4;
#[cfg(feature = "ocr-pipeline")]
const NORMALIZED_PNG_ENCODE_FIXED_BYTES: u64 = 256 * 1024;

// ~keep The cfg is the union of both call sites': `whole_image_ocr_coordinate_transform` here
// (`layout-detection` + `ocr`) and `pdf::ocr::document::ocr_render_scale_is_trustworthy`
// (`layout-detection` + `ocr`/`ocr-wasm`). One constant, because the two routes must apply the
// same bar to the same question.

#[cfg(test)]
fn internal_document_text(doc: &InternalDocument) -> String {
    doc.elements
        .iter()
        .filter_map(|element| {
            let text = element.text.trim();
            (!text.is_empty()).then_some(text)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
fn alphanumeric_tokens(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

#[cfg(test)]
fn alphanumeric_token_retention(layout_text: &str, whole_image_text: &str) -> f64 {
    let whole_image_tokens = alphanumeric_tokens(whole_image_text);
    if whole_image_tokens.is_empty() {
        return 1.0;
    }

    let mut layout_token_counts = std::collections::HashMap::<String, usize>::new();
    for token in alphanumeric_tokens(layout_text) {
        *layout_token_counts.entry(token).or_default() += 1;
    }

    let retained = whole_image_tokens
        .iter()
        .filter(|token| {
            let Some(count) = layout_token_counts.get_mut(*token) else {
                return false;
            };
            if *count == 0 {
                return false;
            }
            *count -= 1;
            true
        })
        .count();

    retained as f64 / whole_image_tokens.len() as f64
}

#[cfg(test)]
fn image_ocr_quality_score(text: &str) -> f64 {
    #[cfg(feature = "quality")]
    {
        crate::text::quality::calculate_quality_score(text, None)
    }

    #[cfg(not(feature = "quality"))]
    {
        let _ = text;
        0.0
    }
}

#[cfg(test)]
fn select_image_ocr_result(
    layout_doc: InternalDocument,
    whole_image_result: Result<InternalDocument>,
) -> InternalDocument {
    let whole_image_doc = match whole_image_result {
        Ok(doc) => doc,
        Err(error) => {
            tracing::warn!(%error, "Whole-image OCR quality comparison failed; retaining layout-region OCR");
            return layout_doc;
        }
    };
    let layout_text = internal_document_text(&layout_doc);
    let whole_image_text = internal_document_text(&whole_image_doc);
    let layout_score = image_ocr_quality_score(&layout_text);
    let whole_image_score = image_ocr_quality_score(&whole_image_text);
    let token_retention = alphanumeric_token_retention(&layout_text, &whole_image_text);

    if layout_score < whole_image_score || token_retention < MIN_LAYOUT_OCR_ALPHANUMERIC_TOKEN_RETENTION {
        tracing::debug!(
            layout_score,
            whole_image_score,
            token_retention,
            "Whole-image OCR retained because layout-region OCR reduced text quality"
        );
        whole_image_doc
    } else {
        layout_doc
    }
}

/// Applies `psm` whenever the caller has not explicitly set one, whether or not a
/// `TesseractConfig` is present. Keyed on the `psm` field itself, not on struct
/// presence (#1573) — an explicitly materialised `TesseractConfig` that leaves `psm`
/// unset still gets the automatic default, and any other field the caller did set is
/// preserved rather than overwritten.
#[cfg(feature = "ocr-pipeline")]
fn apply_default_tesseract_psm(config: &mut crate::core::config::OcrConfig, psm: i32) {
    if config.backend != "tesseract" {
        return;
    }

    let tesseract_config = config
        .tesseract_config
        .get_or_insert_with(|| crate::types::TesseractConfig {
            language: config.language.clone(),
            ..Default::default()
        });
    if tesseract_config.psm.is_none() {
        tesseract_config.psm = Some(psm);
    }
}

/// Also applied by the PDF OCR routes to a page that is one full-page raster, so a scanned
/// page and the same page OCR'd as an image use the same segmentation mode (#1786).
#[cfg(feature = "ocr-pipeline")]
pub(crate) fn apply_default_whole_image_tesseract_psm(config: &mut crate::core::config::OcrConfig) {
    let psm = if has_vertical_tesseract_language(config) {
        VERTICAL_BLOCK_TESSERACT_PSM
    } else {
        WHOLE_IMAGE_TESSERACT_PSM
    };
    apply_default_tesseract_psm(config, psm);
}

/// Checks the same reconciled language `config_to_tesseract` resolves (#1572), so a
/// `jpn_vert` set only on `tesseract_config.language` still selects the vertical PSM.
#[cfg(feature = "ocr-pipeline")]
fn has_vertical_tesseract_language(config: &crate::core::config::OcrConfig) -> bool {
    config
        .effective_tesseract_language()
        .iter()
        .flat_map(|language| language.split('+'))
        .any(|language| language.trim().to_ascii_lowercase().ends_with("_vert"))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn usable_word_confidences(result: &crate::types::ExtractedDocument) -> Vec<f64> {
    result
        .ocr_elements
        .iter()
        .flatten()
        .filter(|element| {
            element.level == crate::types::OcrElementLevel::Word
                && !element.text.trim().is_empty()
                && element.confidence.recognition.is_finite()
        })
        .map(|element| element.confidence.recognition)
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn should_retry_sparse_image_ocr(
    config: &crate::core::config::OcrConfig,
    result: &crate::types::ExtractedDocument,
) -> bool {
    is_implicit_horizontal_tesseract(config)
        && usable_word_confidences(result).len() <= SPARSE_IMAGE_OCR_WORD_LIMIT
        && !has_robust_word_confidence_distribution(result)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn is_implicit_horizontal_tesseract(config: &crate::core::config::OcrConfig) -> bool {
    config.backend == "tesseract"
        && config.tesseract_config.as_ref().and_then(|c| c.psm).is_none()
        && !has_vertical_tesseract_language(config)
}

// The table claim uses a word-count floor. Page text can use the row rendering only if it
// preserves every literal token occurrence, including repeated values. ~keep

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn has_robust_word_confidence_distribution(result: &crate::types::ExtractedDocument) -> bool {
    let mut confidences = usable_word_confidences(result);
    if confidences.is_empty() {
        return false;
    }
    confidences.sort_by(f64::total_cmp);
    let percentile_index = ((confidences.len() as f64 - 1.0) * SPARSE_IMAGE_OCR_CONFIDENCE_PERCENTILE).floor() as usize;
    let low_confidence_count = confidences
        .iter()
        .filter(|confidence| **confidence < SPARSE_IMAGE_OCR_MIN_WORD_CONFIDENCE)
        .count();
    let low_confidence_ratio = low_confidence_count as f64 / confidences.len() as f64;

    confidences[percentile_index] >= SPARSE_IMAGE_OCR_MIN_WORD_CONFIDENCE
        && low_confidence_ratio <= SPARSE_IMAGE_OCR_MAX_LOW_CONFIDENCE_RATIO
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn sparse_image_ocr_fallback_config(
    whole_image_config: &crate::core::config::OcrConfig,
) -> crate::core::config::OcrConfig {
    let mut fallback_config = whole_image_config.clone();
    let tesseract_config = fallback_config.tesseract_config.get_or_insert_default();
    tesseract_config.psm = Some(SPARSE_IMAGE_OCR_FALLBACK_PSM);
    let preprocessing = crate::types::ImagePreprocessingConfig {
        deskew: false,
        contrast_enhance: true,
        binarization_method: "none".to_string(),
        ..Default::default()
    };
    tesseract_config.preprocessing = Some(preprocessing);
    fallback_config
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn is_implicit_strong_rtl_tesseract(config: &crate::core::config::OcrConfig) -> bool {
    is_implicit_horizontal_tesseract(config)
        && config
            .effective_tesseract_language()
            .iter()
            .any(|language| crate::core::config::ocr::is_strong_rtl_tesseract_language(language))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn rtl_sparse_image_ocr_fallback_config(
    whole_image_config: &crate::core::config::OcrConfig,
) -> crate::core::config::OcrConfig {
    let mut fallback_config = whole_image_config.clone();
    fallback_config.tesseract_config.get_or_insert_default().psm = Some(RTL_SPARSE_IMAGE_OCR_FALLBACK_PSM);
    fallback_config
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn is_strong_rtl_letter(character: char) -> bool {
    character.is_alphabetic()
        && matches!(
            character as u32,
            0x0590..=0x05ff | 0x0600..=0x06ff | 0x0750..=0x077f | 0x08a0..=0x08ff
                | 0xfb1d..=0xfdff | 0xfe70..=0xfeff
        )
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn normalized_ocr_tokens(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(|token| {
            let normalized = token
                .trim_matches(|character: char| !character.is_alphanumeric() && !":/_@.-".contains(character))
                .to_lowercase();
            (!normalized.is_empty()).then_some(normalized)
        })
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn normalized_identifier_slot(token: &str) -> Option<String> {
    let token = token.trim_matches(|character: char| !character.is_alphanumeric());
    let characters = token.chars().collect::<Vec<_>>();
    let leading_digits = characters.iter().take_while(|character| character.is_numeric()).count();
    let attached_rtl_list_item = (1..=2).contains(&leading_digits)
        && characters
            .get(leading_digits)
            .is_some_and(|character| matches!(*character, '.' | ')'))
        && characters
            .get(leading_digits + 1)
            .is_some_and(|character| character.is_alphabetic())
        && characters[leading_digits + 1..]
            .iter()
            .any(|character| is_strong_rtl_letter(*character));
    if attached_rtl_list_item {
        return None;
    }
    let alphanumeric = characters
        .iter()
        .filter(|character| character.is_alphanumeric())
        .count();
    if alphanumeric < 3 || token.contains('\u{fffd}') {
        return None;
    }
    let has_internal_separator = characters.iter().enumerate().any(|(index, character)| {
        !character.is_alphanumeric()
            && characters[..index].iter().any(|character| character.is_alphanumeric())
            && characters[index + 1..]
                .iter()
                .any(|character| character.is_alphanumeric())
    });
    let digits = characters.iter().filter(|character| character.is_ascii_digit()).count();
    let letters = characters
        .iter()
        .filter(|character| character.is_ascii_alphabetic())
        .count();
    let is_long_numeric = digits >= 4 && letters == 0;
    let is_compact_identifier = digits > 0 && letters > 0 && alphanumeric >= 6;
    (has_internal_separator || is_long_numeric || is_compact_identifier).then(|| token.to_lowercase())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn identifier_slot_shape(token: &str) -> String {
    let mut shape = String::new();
    let mut previous_class = None;
    for character in token.chars() {
        let class = if character.is_alphabetic() {
            Some('a')
        } else if character.is_numeric() {
            Some('0')
        } else {
            None
        };
        if let Some(class) = class {
            if previous_class != Some(class) {
                shape.push(class);
            }
            previous_class = Some(class);
        } else {
            shape.push(character);
            previous_class = None;
        }
    }
    shape
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn normalized_edit_distance(left: &str, right: &str) -> usize {
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_character) in left.chars().enumerate() {
        let mut current = Vec::with_capacity(right.len() + 1);
        current.push(left_index + 1);
        for (right_index, right_character) in right.iter().enumerate() {
            current.push(std::cmp::min(
                std::cmp::min(current[right_index] + 1, previous[right_index + 1] + 1),
                previous[right_index] + usize::from(left_character != *right_character),
            ));
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn identifier_slots(content: &str) -> Vec<String> {
    content
        .split_whitespace()
        .filter_map(normalized_identifier_slot)
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn identifier_slots_are_preserved(primary: &str, candidate: &str) -> bool {
    fn compatible(primary: &str, candidate: &str) -> bool {
        if identifier_slot_shape(primary) != identifier_slot_shape(candidate) {
            return false;
        }
        let length = primary.chars().count().max(candidate.chars().count());
        normalized_edit_distance(primary, candidate) <= length.div_ceil(5).clamp(1, 2)
    }

    fn assign(
        primary_index: usize,
        compatibility: &[Vec<bool>],
        seen: &mut [bool],
        candidate_owner: &mut [Option<usize>],
    ) -> bool {
        for candidate_index in 0..candidate_owner.len() {
            if !compatibility[primary_index][candidate_index] || seen[candidate_index] {
                continue;
            }
            seen[candidate_index] = true;
            let can_assign = match candidate_owner[candidate_index] {
                Some(owner) => assign(owner, compatibility, seen, candidate_owner),
                None => true,
            };
            if can_assign {
                candidate_owner[candidate_index] = Some(primary_index);
                return true;
            }
        }
        false
    }

    let primary = identifier_slots(primary);
    let candidate = identifier_slots(candidate);
    if primary.len() != candidate.len() {
        return false;
    }
    let compatibility = primary
        .iter()
        .map(|primary| {
            candidate
                .iter()
                .map(|candidate| compatible(primary, candidate))
                .collect()
        })
        .collect::<Vec<Vec<_>>>();
    let mut candidate_owner = vec![None; candidate.len()];
    (0..primary.len()).all(|primary_index| {
        let mut seen = vec![false; candidate.len()];
        assign(primary_index, &compatibility, &mut seen, &mut candidate_owner)
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn is_rtl_identifier_element(element: &crate::types::OcrElement) -> bool {
    element.level == crate::types::OcrElementLevel::Word
        && normalized_identifier_slot(&element.text).is_some()
        && element
            .backend_metadata
            .get("word_language")
            .and_then(serde_json::Value::as_str)
            .is_some_and(crate::core::config::ocr::is_strong_rtl_tesseract_language)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn identifier_slots_are_preserved_after_recovery(
    primary: &crate::types::ExtractedDocument,
    candidate: &crate::types::ExtractedDocument,
) -> bool {
    if identifier_slots_are_preserved(&primary.content, &candidate.content) {
        return true;
    }
    let mut primary_slots = identifier_slots(&primary.content);
    let mut candidate_slots = identifier_slots(&candidate.content);
    for recovered in candidate.ocr_elements.iter().flatten().filter(|element| {
        element
            .backend_metadata
            .get("recovered_from")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|source| compatible_identifier_recovery(source, &element.text))
    }) {
        let Some(candidate_slot) = normalized_identifier_slot(&recovered.text) else {
            return false;
        };
        if primary_slots.contains(&candidate_slot) {
            continue;
        }
        let mut primary_matches = primary
            .ocr_elements
            .iter()
            .flatten()
            .filter(|element| compatible_identifier_recovery(&element.text, &recovered.text));
        let Some(primary_element) = primary_matches.next() else {
            return false;
        };
        if primary_matches.next().is_some() {
            return false;
        }
        let Some(candidate_index) = candidate_slots.iter().position(|slot| slot == &candidate_slot) else {
            return false;
        };
        if let Some(primary_slot) = normalized_identifier_slot(&primary_element.text) {
            let Some(primary_index) = primary_slots.iter().position(|slot| slot == &primary_slot) else {
                return false;
            };
            primary_slots.remove(primary_index);
        }
        candidate_slots.remove(candidate_index);
    }
    identifier_slots_are_preserved(&primary_slots.join(" "), &candidate_slots.join(" "))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
pub(crate) fn is_mixed_ltr_recovery_eligible_config(config: &crate::core::config::OcrConfig) -> bool {
    let languages = config
        .effective_tesseract_language()
        .into_iter()
        .flat_map(|language| {
            language
                .split('+')
                .map(|part| part.trim().to_string())
                .collect::<Vec<_>>()
        })
        .collect::<std::collections::HashSet<_>>();
    is_implicit_strong_rtl_tesseract(config)
        && languages.contains("eng")
        && languages
            .iter()
            .any(|language| crate::core::config::ocr::is_strong_rtl_tesseract_language(language))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn internal_rtl_recovery_config(config: &crate::core::config::OcrConfig) -> crate::core::config::OcrConfig {
    let mut config = config.clone();
    enable_image_ocr_elements(&mut config, true);
    config.element_config.get_or_insert_default().min_confidence = 0.0;
    config
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn apply_rtl_capture_public_policy(
    config: &crate::core::config::OcrConfig,
    mut document: crate::types::ExtractedDocument,
) -> crate::types::ExtractedDocument {
    if is_mixed_ltr_recovery_eligible_config(config) {
        document.ocr_elements = config.select_public_elements(document.ocr_elements.take());
    }
    document
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn compatible_identifier_recovery(source: &str, candidate: &str) -> bool {
    let source = source.trim_matches(|character: char| !character.is_ascii_alphanumeric());
    let candidate = candidate.trim_matches(|character: char| !character.is_ascii_alphanumeric());
    let alnum = |text: &str| {
        text.chars()
            .filter(char::is_ascii_alphanumeric)
            .map(|character| character.to_ascii_lowercase())
            .collect::<String>()
    };
    if alnum(source) == alnum(candidate) {
        return source != candidate;
    }
    candidate.char_indices().any(|(index, character)| {
        if !character.is_ascii_digit() {
            return false;
        }
        let suffix_index = candidate[..index]
            .char_indices()
            .next_back()
            .filter(|(_, character)| "-._/".contains(*character))
            .map_or(index, |(separator_index, _)| separator_index);
        let suffix = &candidate[suffix_index..];
        if suffix.len() < 3
            || suffix.chars().filter(char::is_ascii_digit).count() < 2
            || !suffix.chars().any(|character| "-._/".contains(character))
            || !source.ends_with(suffix)
        {
            return false;
        }
        let source_prefix = source[..source.len() - suffix.len()]
            .chars()
            .filter(|character| !"-._/".contains(*character))
            .collect::<String>();
        let candidate_prefix = candidate[..suffix_index]
            .chars()
            .filter(|character| !"-._/".contains(*character))
            .collect::<String>();
        (1..=6).contains(&source_prefix.len())
            && (1..=5).contains(&candidate_prefix.len())
            && source_prefix.chars().all(|character| character.is_ascii_digit())
            && candidate_prefix
                .chars()
                .all(|character| character.is_ascii_alphabetic())
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn token_occurrences(text: &str, token: &str) -> Vec<usize> {
    text.match_indices(token)
        .filter_map(|(start, _)| {
            let end = start + token.len();
            (text[..start]
                .chars()
                .next_back()
                .is_none_or(|character| !character.is_ascii_alphanumeric())
                && text[end..]
                    .chars()
                    .next()
                    .is_none_or(|character| !character.is_ascii_alphanumeric()))
            .then_some(start)
        })
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn replace_unique_token(text: &mut String, source: &str, replacement: &str) -> std::result::Result<bool, ()> {
    let occurrences = token_occurrences(text, source);
    match occurrences.as_slice() {
        [] => Ok(false),
        [offset] => {
            text.replace_range(*offset..*offset + source.len(), replacement);
            Ok(true)
        }
        _ => Err(()),
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn update_recovery_tables(
    tables: &mut [crate::types::Table],
    source: &str,
    replacement: &str,
) -> std::result::Result<bool, ()> {
    let cell_occurrences = tables
        .iter()
        .flat_map(|table| table.cells.iter().flatten())
        .map(|cell| token_occurrences(cell, source).len())
        .sum::<usize>();
    let column_occurrences = tables
        .iter()
        .flat_map(|table| table.columns.iter().flatten())
        .map(|column| token_occurrences(column, source).len())
        .sum::<usize>();
    if cell_occurrences > 1 || column_occurrences > 1 {
        return Err(());
    }
    let mut updated = false;
    for table in tables {
        let markdown_contained_source = !token_occurrences(&table.markdown, source).is_empty();
        let mut table_updated = false;
        for row in &mut table.cells {
            for cell in row {
                table_updated |= replace_unique_token(cell, source, replacement)?;
            }
        }
        if let Some(columns) = table.columns.as_mut() {
            for column in columns {
                table_updated |= replace_unique_token(column, source, replacement)?;
            }
        }
        if markdown_contained_source && !table_updated {
            return Err(());
        }
        if table_updated {
            table.markdown = crate::rendering::common::render_table_markdown(&table.cells);
            updated = true;
        }
    }
    Ok(updated)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn apply_identifier_recovery_transaction(
    selected: &mut crate::types::ExtractedDocument,
    index: usize,
    source: &str,
    evidence: &crate::types::OcrElement,
) -> bool {
    let source_core = source.trim_matches(|character: char| !character.is_ascii_alphanumeric());
    let evidence_core = evidence
        .text
        .trim_matches(|character: char| !character.is_ascii_alphanumeric());
    let prefix_len = source.find(source_core).unwrap_or(0);
    let replacement_element = format!(
        "{}{}{}",
        &source[..prefix_len],
        evidence_core,
        &source[prefix_len + source_core.len()..]
    );
    let mut updated = selected.clone();
    if replace_unique_token(&mut updated.content, source_core, evidence_core) != Ok(true) {
        return false;
    }
    let Some(element) = updated
        .ocr_elements
        .as_mut()
        .and_then(|elements| elements.get_mut(index))
    else {
        return false;
    };
    if element.text != source {
        return false;
    }
    element.text = replacement_element;
    element.confidence = evidence.confidence.clone();
    element.backend_metadata = evidence.backend_metadata.clone();
    element
        .backend_metadata
        .insert("recovered_from".to_string(), serde_json::json!(source));

    if update_recovery_tables(&mut updated.tables, source_core, evidence_core).is_err() {
        return false;
    }
    if let Some(internal) = updated.ocr_internal_document.as_mut() {
        let mut element_updates = 0;
        for element in &mut internal.elements {
            match replace_unique_token(&mut element.text, source_core, evidence_core) {
                Ok(true) => element_updates += 1,
                Ok(false) => {}
                Err(()) => return false,
            }
        }
        if element_updates > 1 || update_recovery_tables(&mut internal.tables, source_core, evidence_core).is_err() {
            return false;
        }
    }
    *selected = updated;
    true
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn unique_compatible_recovery_evidence<'a>(
    source: &str,
    elements: impl Iterator<Item = &'a crate::types::OcrElement>,
) -> Option<&'a crate::types::OcrElement> {
    let mut matches = elements.filter(|element| {
        element.level == crate::types::OcrElementLevel::Word
            && element.confidence.recognition >= RTL_IDENTIFIER_RECOVERY_MIN_CONFIDENCE
            && compatible_identifier_recovery(
                source,
                element
                    .text
                    .trim_matches(|character: char| !character.is_ascii_alphanumeric()),
            )
    });
    let evidence = matches.next()?;
    matches.next().is_none().then_some(evidence)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn is_rtl_recovery_suspect(element: &crate::types::OcrElement, has_lost_prefix: bool) -> bool {
    if element.level != crate::types::OcrElementLevel::Word || !element.text.is_ascii() || !has_lost_prefix {
        return false;
    }
    let identifier = element
        .text
        .trim_matches(|character: char| !character.is_ascii_alphanumeric());
    let parts = identifier.split('-').collect::<Vec<_>>();
    let iso_date = parts.len() == 3
        && [4, 2, 2]
            .into_iter()
            .zip(&parts)
            .all(|(length, part)| part.len() == length && part.chars().all(|character| character.is_ascii_digit()));
    if iso_date {
        return false;
    }
    let Some(language) = element
        .backend_metadata
        .get("word_language")
        .and_then(serde_json::Value::as_str)
    else {
        return false;
    };
    let language_is_rtl = crate::core::config::ocr::is_strong_rtl_tesseract_language(language);
    let no_letters = !identifier.chars().any(|character| character.is_ascii_alphabetic());
    let separators = identifier
        .chars()
        .filter(|character| "-._/".contains(*character))
        .count();
    let lost_prefix_fragment = language_is_rtl && no_letters && separators >= 2;
    let short_numeric_fragment = identifier.len() <= 5
        && no_letters
        && separators >= 1
        && identifier.chars().any(|character| character.is_ascii_digit());
    let compact_unseparated_fragment = identifier.len() <= 6
        && identifier.chars().all(|character| character.is_ascii_alphanumeric())
        && identifier.chars().any(|character| character.is_ascii_alphabetic())
        && identifier.chars().any(|character| character.is_ascii_digit());
    (lost_prefix_fragment || short_numeric_fragment || compact_unseparated_fragment)
        && (normalized_identifier_slot(&element.text).is_some() || compact_unseparated_fragment)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn rtl_recovery_suspects(
    selected: &crate::types::ExtractedDocument,
) -> Vec<(usize, String, crate::types::OcrBoundingGeometry)> {
    let elements = selected.ocr_elements.as_deref().unwrap_or_default();
    let has_lost_prefix = elements.iter().any(|element| {
        is_rtl_identifier_element(element)
            && !element.text.chars().any(|character| character.is_ascii_alphabetic())
            && element
                .text
                .chars()
                .filter(|character| "-._/".contains(*character))
                .count()
                >= 2
    });
    elements
        .iter()
        .enumerate()
        .filter(|(_, element)| is_rtl_recovery_suspect(element, has_lost_prefix))
        .take(RTL_IDENTIFIER_RECOVERY_MAX_CROPS)
        .map(|(index, element)| (index, element.text.clone(), element.geometry.clone()))
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn build_rtl_recovery_crops_blocking(
    image_bytes: &[u8],
    suspects: Vec<(usize, String, crate::types::OcrBoundingGeometry)>,
    processed_width: Option<u32>,
    processed_height: Option<u32>,
    security_limits: &crate::extractors::security::SecurityLimits,
) -> crate::Result<Vec<(usize, String, Vec<u8>)>> {
    let image = crate::extraction::image::decode_image_to_rgb8_with_security_limits(image_bytes, security_limits)?;
    let scale_x = image.width() as f64 / f64::from(processed_width.unwrap_or(image.width()));
    let scale_y = image.height() as f64 / f64::from(processed_height.unwrap_or(image.height()));
    let mut crops = Vec::with_capacity(suspects.len());
    for (index, source, geometry) in suspects {
        let crate::types::OcrBoundingGeometry::Rectangle {
            left,
            top,
            width,
            height,
        } = geometry
        else {
            continue;
        };
        let left = (f64::from(left) * scale_x).round() as u32;
        let top = (f64::from(top) * scale_y).round() as u32;
        let width = (f64::from(width) * scale_x).round().max(1.0) as u32;
        let height = (f64::from(height) * scale_y).round().max(1.0) as u32;
        let x = left.saturating_sub(width / 2);
        let y = top.saturating_sub(height);
        let right = left.saturating_add(width + width / 2).min(image.width());
        let bottom = top.saturating_add(height.saturating_mul(2)).min(image.height());
        if right <= x
            || bottom <= y
            || u64::from(right - x) * u64::from(bottom - y) > RTL_IDENTIFIER_RECOVERY_MAX_CROP_PIXELS
        {
            continue;
        }
        let crop = image::imageops::crop_imm(&image, x, y, right - x, bottom - y).to_image();
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(crop)
            .write_to(&mut encoded, image::ImageFormat::Png)
            .map_err(|error| crate::XbergError::Other(format!("Failed to encode OCR recovery crop: {error}")))?;
        crops.push((index, source, encoded.into_inner()));
    }
    Ok(crops)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
async fn build_rtl_recovery_crops(
    image_bytes: std::sync::Arc<Vec<u8>>,
    suspects: Vec<(usize, String, crate::types::OcrBoundingGeometry)>,
    processed_width: Option<u32>,
    processed_height: Option<u32>,
    security_limits: crate::extractors::security::SecurityLimits,
) -> crate::Result<Vec<(usize, String, Vec<u8>)>> {
    #[cfg(feature = "tokio-runtime")]
    {
        tokio::task::spawn_blocking(move || {
            build_rtl_recovery_crops_blocking(
                image_bytes.as_slice(),
                suspects,
                processed_width,
                processed_height,
                &security_limits,
            )
        })
        .await
        .map_err(|error| crate::XbergError::Other(format!("OCR recovery crop worker failed: {error}")))?
    }
    #[cfg(not(feature = "tokio-runtime"))]
    {
        build_rtl_recovery_crops_blocking(
            image_bytes.as_slice(),
            suspects,
            processed_width,
            processed_height,
            &security_limits,
        )
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn rtl_identifier_recovery_config(fallback_config: &crate::core::config::OcrConfig) -> crate::core::config::OcrConfig {
    let mut config = fallback_config.clone();
    config.language = vec!["eng".to_string()];
    let tesseract = config.tesseract_config.get_or_insert_default();
    tesseract.language = vec!["eng".to_string()];
    tesseract.psm = Some(7);
    enable_image_ocr_elements(&mut config, true);
    config.element_config.get_or_insert_default().min_confidence = 0.0;
    config
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
async fn recover_selected_rtl_identifiers(
    backend: &std::sync::Arc<dyn crate::plugins::OcrBackend>,
    image_bytes: std::sync::Arc<Vec<u8>>,
    selected: &mut crate::types::ExtractedDocument,
    fallback_config: &crate::core::config::OcrConfig,
    cancel_token: Option<&crate::cancellation::CancellationToken>,
) -> crate::Result<()> {
    if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
        return Err(crate::XbergError::Cancelled);
    }
    let suspects = rtl_recovery_suspects(selected);
    if suspects.is_empty() {
        return Ok(());
    }
    let processed_dimension = |key: &str| {
        selected
            .metadata
            .additional
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| u32::try_from(value).ok())
            .filter(|value| *value > 0)
    };
    let processed_width = processed_dimension("ocr_processed_image_width");
    let processed_height = processed_dimension("ocr_processed_image_height");
    let config = rtl_identifier_recovery_config(fallback_config);
    let security_limits = fallback_config.security_limits.clone().unwrap_or_default();
    let crops = build_rtl_recovery_crops(
        image_bytes,
        suspects,
        processed_width,
        processed_height,
        security_limits,
    )
    .await?;
    if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
        return Err(crate::XbergError::Cancelled);
    }
    let mut recoveries = Vec::with_capacity(crops.len());
    for (index, source, encoded) in crops {
        recoveries.push((index, source, backend.process_image(&encoded, &config).await?));
        if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
            return Err(crate::XbergError::Cancelled);
        }
    }
    for (index, source, evidence) in recoveries {
        let Some(evidence) = unique_compatible_recovery_evidence(&source, evidence.ocr_elements.iter().flatten())
        else {
            continue;
        };
        apply_identifier_recovery_transaction(selected, index, &source, evidence);
    }
    Ok(())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
async fn recover_selected_rtl_identifiers_fail_open(
    backend: &std::sync::Arc<dyn crate::plugins::OcrBackend>,
    image_bytes: std::sync::Arc<Vec<u8>>,
    selected: &mut crate::types::ExtractedDocument,
    fallback_config: &crate::core::config::OcrConfig,
    cancel_token: Option<&crate::cancellation::CancellationToken>,
) -> crate::Result<()> {
    match recover_selected_rtl_identifiers(backend, image_bytes, selected, fallback_config, cancel_token).await {
        Ok(()) => Ok(()),
        Err(error @ crate::XbergError::Cancelled) => Err(error),
        Err(_) if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) => {
            Err(crate::XbergError::Cancelled)
        }
        Err(error) => {
            tracing::warn!(
                backend = backend.name(),
                stage = "rtl_identifier_recovery",
                %error,
                "optional RTL identifier OCR recovery failed"
            );
            Ok(())
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn normalized_tesseract_mean_confidence(result: &crate::types::ExtractedDocument) -> Option<f64> {
    if let Some(value) = result.metadata.additional.get("mean_text_conf")
        && let Some(confidence) = value
            .as_f64()
            .or_else(|| value.as_i64().map(|value| value as f64))
            .or_else(|| value.as_u64().map(|value| value as f64))
            .filter(|value| value.is_finite() && (0.0..=100.0).contains(value))
    {
        return Some(confidence / 100.0);
    }

    let confidences = usable_word_confidences(result)
        .into_iter()
        .filter(|confidence| (0.0..=1.0).contains(confidence))
        .collect::<Vec<_>>();
    (!confidences.is_empty()).then(|| confidences.iter().sum::<f64>() / confidences.len() as f64)
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn has_new_pathological_line_repetition(primary: &str, candidate: &str) -> bool {
    fn lines(content: &str) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        for line in content.lines().filter_map(|line| {
            let normalized = normalized_ocr_tokens(line).join(" ");
            (!normalized.is_empty()).then_some(normalized)
        }) {
            *counts.entry(line).or_default() += 1;
        }
        counts
    }

    let primary_lines = lines(primary);
    lines(candidate)
        .into_iter()
        .any(|(line, count)| count >= 3 && count > primary_lines.get(&line).copied().unwrap_or_default())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn has_new_pathological_trigram_repetition(primary: &str, candidate: &str) -> bool {
    fn trigrams(content: &str) -> std::collections::HashMap<[String; 3], usize> {
        let mut counts = std::collections::HashMap::new();
        for line in content.lines() {
            let tokens = normalized_ocr_tokens(line);
            for words in tokens.windows(3) {
                *counts
                    .entry([words[0].clone(), words[1].clone(), words[2].clone()])
                    .or_default() += 1;
            }
        }
        counts
    }

    let primary_trigrams = trigrams(primary);
    trigrams(candidate)
        .into_iter()
        .any(|(trigram, count)| count >= 3 && count > primary_trigrams.get(&trigram).copied().unwrap_or_default())
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn rtl_retry_candidate_improves(
    primary: &crate::types::ExtractedDocument,
    candidate: &crate::types::ExtractedDocument,
) -> bool {
    let primary_tokens = normalized_ocr_tokens(&primary.content);
    let candidate_tokens = normalized_ocr_tokens(&candidate.content);
    let mut primary_unique = primary_tokens.iter().collect::<std::collections::HashSet<_>>();
    let mut candidate_unique = candidate_tokens.iter().collect::<std::collections::HashSet<_>>();
    for (result, tokens) in [(primary, &mut primary_unique), (candidate, &mut candidate_unique)] {
        for element in result
            .ocr_elements
            .iter()
            .flatten()
            .filter(|element| is_rtl_identifier_element(element))
        {
            for token in normalized_ocr_tokens(&element.text) {
                tokens.remove(&token);
            }
        }
    }
    let primary_rtl = primary_unique
        .iter()
        .filter(|token| token.chars().any(is_strong_rtl_letter))
        .count();
    let candidate_rtl = candidate_unique
        .iter()
        .filter(|token| token.chars().any(is_strong_rtl_letter))
        .count();
    let retained_primary = primary_unique.intersection(&candidate_unique).count();
    let primary_retention = if primary_unique.is_empty() {
        1.0
    } else {
        retained_primary as f64 / primary_unique.len() as f64
    };
    if candidate_rtl <= primary_rtl
        || candidate_unique.len() < primary_unique.len()
        || primary_retention < RTL_SPARSE_IMAGE_OCR_MIN_PRIMARY_TOKEN_RETENTION
    {
        return false;
    }

    if has_new_pathological_line_repetition(&primary.content, &candidate.content)
        || has_new_pathological_trigram_repetition(&primary.content, &candidate.content)
    {
        return false;
    }

    match (
        normalized_tesseract_mean_confidence(primary),
        normalized_tesseract_mean_confidence(candidate),
    ) {
        (Some(primary), Some(candidate)) => candidate + RTL_SPARSE_IMAGE_OCR_MAX_CONFIDENCE_DROP >= primary,
        (Some(_), None) => false,
        _ => true,
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn merge_primary_warnings_into_selected(
    primary: &crate::types::ExtractedDocument,
    selected: &mut crate::types::ExtractedDocument,
) {
    let selected_warnings = std::mem::take(&mut selected.processing_warnings);
    selected.processing_warnings = primary.processing_warnings.clone();
    for warning in selected_warnings {
        crate::core::diagnostics::push_warning_deduped(&mut selected.processing_warnings, warning);
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
fn should_retry_sparse_rtl_ocr(
    original_config: &crate::core::config::OcrConfig,
    effective_config: &crate::core::config::OcrConfig,
    primary: &crate::types::ExtractedDocument,
) -> bool {
    let tokens = normalized_ocr_tokens(&primary.content);
    let unique_tokens = tokens.iter().collect::<std::collections::HashSet<_>>().len();
    let suspect_identifier = primary.ocr_elements.iter().flatten().any(|element| {
        is_rtl_identifier_element(element)
            && !element.text.chars().any(|character| character.is_ascii_alphabetic())
            && element
                .text
                .chars()
                .filter(|character| "-._/".contains(*character))
                .count()
                >= 2
    });
    is_implicit_strong_rtl_tesseract(original_config)
        && effective_config.tesseract_config.as_ref().and_then(|config| config.psm) == Some(WHOLE_IMAGE_TESSERACT_PSM)
        && (suspect_identifier
            || (unique_tokens <= RTL_SPARSE_IMAGE_OCR_UNIQUE_TOKEN_LIMIT
                && tokens.len() <= RTL_SPARSE_IMAGE_OCR_TOTAL_TOKEN_LIMIT))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
async fn select_rtl_retry_result(
    backend: &std::sync::Arc<dyn crate::plugins::OcrBackend>,
    image_bytes: std::sync::Arc<Vec<u8>>,
    configs: (&crate::core::config::OcrConfig, &crate::core::config::OcrConfig),
    primary: crate::types::ExtractedDocument,
    fallback: crate::types::ExtractedDocument,
    cancel_token: Option<&crate::cancellation::CancellationToken>,
) -> crate::Result<crate::types::ExtractedDocument> {
    let (original_config, fallback_config) = configs;
    let select_fallback = rtl_retry_candidate_improves(&primary, &fallback);
    let recover_identifiers = is_mixed_ltr_recovery_eligible_config(original_config);
    if !select_fallback && !recover_identifiers {
        return Ok(apply_rtl_capture_public_policy(original_config, primary));
    }
    let mut selected = if select_fallback { fallback } else { primary.clone() };
    if select_fallback {
        merge_primary_warnings_into_selected(&primary, &mut selected);
    }
    if recover_identifiers {
        recover_selected_rtl_identifiers_fail_open(backend, image_bytes, &mut selected, fallback_config, cancel_token)
            .await?;
    }
    if select_fallback && !identifier_slots_are_preserved_after_recovery(&primary, &selected) {
        return Ok(apply_rtl_capture_public_policy(original_config, primary));
    }
    Ok(apply_rtl_capture_public_policy(original_config, selected))
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
#[cfg(feature = "pdf")]
#[path = "image_ocr_budget.rs"]
mod image_ocr_budget;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
#[cfg(feature = "pdf")]
pub(crate) use image_ocr_budget::reserve_retained_ocr_raster;

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
/// Which sparse-output retries [`process_whole_image_ocr_with_sparse_retry`] may attempt.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SparseRetryPolicy {
    pub(crate) whole_page_raster: bool,
    pub(crate) allow_standard_retry: bool,
}

#[cfg(any(feature = "pdf", test))]
#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
pub(crate) async fn process_whole_image_ocr_with_sparse_retry(
    backend: &std::sync::Arc<dyn crate::plugins::OcrBackend>,
    image_bytes: std::sync::Arc<Vec<u8>>,
    original_config: &crate::core::config::OcrConfig,
    effective_config: &crate::core::config::OcrConfig,
    retry_policy: SparseRetryPolicy,
    cancel_token: Option<&crate::cancellation::CancellationToken>,
) -> crate::Result<crate::types::ExtractedDocument> {
    process_whole_image_ocr(
        backend,
        image_bytes,
        (original_config, effective_config),
        retry_policy,
        cancel_token,
    )
    .await
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
async fn process_whole_image_ocr(
    backend: &std::sync::Arc<dyn crate::plugins::OcrBackend>,
    image_bytes: std::sync::Arc<Vec<u8>>,
    configs: (&crate::core::config::OcrConfig, &crate::core::config::OcrConfig),
    retry_policy: SparseRetryPolicy,
    cancel_token: Option<&crate::cancellation::CancellationToken>,
) -> crate::Result<crate::types::ExtractedDocument> {
    let (original_config, effective_config) = configs;
    let SparseRetryPolicy {
        whole_page_raster,
        allow_standard_retry,
    } = retry_policy;
    if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
        return Err(crate::XbergError::Cancelled);
    }
    let effective_config = if is_mixed_ltr_recovery_eligible_config(original_config) {
        std::borrow::Cow::Owned(internal_rtl_recovery_config(effective_config))
    } else {
        std::borrow::Cow::Borrowed(effective_config)
    };
    let primary_bytes = std::sync::Arc::clone(&image_bytes);

    let primary = backend
        .process_image_owned(primary_bytes, effective_config.as_ref())
        .await?;
    let rtl_retry = should_retry_sparse_rtl_ocr(original_config, effective_config.as_ref(), &primary);
    if !whole_page_raster && !rtl_retry {
        return Ok(apply_rtl_capture_public_policy(original_config, primary));
    }

    let standard_retry = allow_standard_retry && !rtl_retry && should_retry_sparse_image_ocr(original_config, &primary);
    if !rtl_retry && !standard_retry {
        return Ok(apply_rtl_capture_public_policy(original_config, primary));
    }
    if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
        return Err(crate::XbergError::Cancelled);
    }

    let fallback_config = if rtl_retry {
        rtl_sparse_image_ocr_fallback_config(effective_config.as_ref())
    } else {
        sparse_image_ocr_fallback_config(effective_config.as_ref())
    };
    let fallback_bytes = std::sync::Arc::clone(&image_bytes);

    let fallback_result = backend.process_image_owned(fallback_bytes, &fallback_config).await;
    if cancel_token.is_some_and(crate::cancellation::CancellationToken::is_cancelled) {
        return Err(crate::XbergError::Cancelled);
    }
    let fallback = match fallback_result {
        Ok(fallback) => fallback,
        Err(error @ crate::XbergError::Cancelled) => return Err(error),
        Err(error) => {
            tracing::warn!(%error, "sparse whole-image OCR fallback failed");
            return Ok(apply_rtl_capture_public_policy(original_config, primary));
        }
    };
    if rtl_retry {
        return select_rtl_retry_result(
            backend,
            image_bytes,
            (original_config, &fallback_config),
            primary,
            fallback,
            cancel_token,
        )
        .await;
    }
    if !has_robust_word_confidence_distribution(&fallback) {
        return Ok(apply_rtl_capture_public_policy(original_config, primary));
    }
    let mut selected = fallback;
    merge_primary_warnings_into_selected(&primary, &mut selected);
    Ok(apply_rtl_capture_public_policy(original_config, selected))
}

/// Resize/re-DPI raw image bytes for OCR using `ExtractionConfig::images`
/// (`ImageExtractionConfig`) before handing them to an OCR backend.
///
/// OCR backends only ever see `OcrConfig` (via the `OcrBackend` trait), and
/// `OcrConfig` has no field that traces back to `ExtractionConfig::images` — so
/// `target_dpi`, `max_image_dimension`, `auto_adjust_dpi`, `min_dpi`, and
/// `max_dpi` were parsed into config but silently dropped before reaching any
/// backend (issue #209). Normalizing the bytes once, here, at the extractor
/// boundary makes the setting effective for every backend without touching the
/// backend trait or its config types.
///
/// Falls back to the original bytes unchanged if decoding or normalization
/// fails; OCR should still be attempted on the original image rather than
/// aborting the extraction.
#[cfg(feature = "ocr-pipeline")]
struct NormalizedOcrImage {
    bytes: Vec<u8>,
    metadata: Option<crate::types::ImagePreprocessingMetadata>,
}

#[cfg(feature = "ocr-pipeline")]
fn unchanged_ocr_image(content: &[u8]) -> NormalizedOcrImage {
    NormalizedOcrImage {
        bytes: content.to_vec(),
        metadata: None,
    }
}

#[cfg(feature = "ocr-pipeline")]
fn normalize_image_bytes_for_ocr(
    content: &[u8],
    images_config: &crate::core::config::ImageExtractionConfig,
    security_limits: &crate::extractors::security::SecurityLimits,
    ocr_config: &crate::core::config::OcrConfig,
) -> Result<NormalizedOcrImage> {
    let rgb = match crate::extraction::image::decode_image_to_rgb8_with_security_limits(content, security_limits) {
        Ok(decoded) => decoded,
        Err(error @ crate::XbergError::Validation { .. }) => return Err(error),
        Err(_) => return Ok(unchanged_ocr_image(content)),
    };
    let (width, height) = rgb.dimensions();
    let dpi_config = crate::types::ImageDpiConfig::from(images_config);
    // GH#1630: an explicit `OcrConfig.backend_options["source_dpi"]` hint wins, then embedded
    // PNG `pHYs` density, then `None` (a resolution guessed from the text size). Shared with
    // `ocr::processor::execution::perform_ocr` (GH#1621: no second density scanner).
    let explicit_source_dpi = crate::extraction::image::explicit_source_dpi_from_ocr_config(ocr_config);
    let known_source_dpi =
        crate::extraction::image::resolve_known_source_dpi(explicit_source_dpi, content, width, height);
    let (planned_width, planned_height) =
        crate::image::preprocessing::normalized_image_dimensions(width, height, &dpi_config, known_source_dpi);
    let encoded_source_bytes = u64::try_from(content.len())
        .map_err(|_| crate::extraction::image_decode::image_dimension_error(width, height, u64::MAX, u64::MAX))?;
    let current_bytes = u64::try_from(rgb.as_raw().len())
        .ok()
        .and_then(|bytes| bytes.checked_add(encoded_source_bytes))
        .ok_or_else(|| crate::extraction::image_decode::image_dimension_error(width, height, u64::MAX, u64::MAX))?;
    let planned_bytes = crate::extraction::image_decode::decoded_byte_count(planned_width, planned_height, 3)?;
    if (planned_width, planned_height) != (width, height) {
        crate::extraction::image_decode::validate_image_live_bytes(
            width,
            height,
            current_bytes,
            planned_bytes,
            security_limits,
        )?;
    }
    let result = match crate::image::preprocessing::normalize_image_dpi_owned(
        rgb.into_raw(),
        width as usize,
        height as usize,
        &dpi_config,
        known_source_dpi,
    ) {
        Ok(result) => result,
        Err((error, _)) => {
            tracing::warn!(%error, "failed to normalize OCR image; using original image");
            return Ok(unchanged_ocr_image(content));
        }
    };
    let (new_width, new_height) = result.dimensions;
    let new_width = u32::try_from(new_width)
        .map_err(|_| crate::extraction::image_decode::image_dimension_error(u32::MAX, u32::MAX, u64::MAX, u64::MAX))?;
    let new_height = u32::try_from(new_height)
        .map_err(|_| crate::extraction::image_decode::image_dimension_error(new_width, u32::MAX, u64::MAX, u64::MAX))?;
    let normalized_bytes = u64::try_from(result.rgb_data.len())
        .ok()
        .and_then(|bytes| bytes.checked_add(encoded_source_bytes))
        .ok_or_else(|| {
            crate::extraction::image_decode::image_dimension_error(new_width, new_height, u64::MAX, u64::MAX)
        })?;
    let encode_live_bytes = crate::extraction::image_decode::decoded_byte_count(
        new_width,
        new_height,
        NORMALIZED_PNG_ENCODE_BYTES_PER_PIXEL,
    )?
    .checked_add(NORMALIZED_PNG_ENCODE_FIXED_BYTES)
    .ok_or_else(|| crate::extraction::image_decode::image_dimension_error(new_width, new_height, u64::MAX, u64::MAX))?;
    crate::extraction::image_decode::validate_image_live_bytes(
        new_width,
        new_height,
        normalized_bytes,
        encode_live_bytes,
        security_limits,
    )?;
    let bytes = match encode_rgb_as_png(&result.rgb_data, new_width, new_height) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(%error, "failed to encode normalized OCR image; using original image");
            return Ok(unchanged_ocr_image(content));
        }
    };
    Ok(NormalizedOcrImage {
        bytes,
        metadata: Some(result.metadata),
    })
}

#[cfg(feature = "ocr-pipeline")]
fn encode_rgb_as_png(rgb_data: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    use image::ImageEncoder;

    let mut png = std::io::Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(rgb_data, width, height, image::ExtendedColorType::Rgb8)
        .map_err(|error| crate::XbergError::Other(format!("Failed to encode normalized image as PNG: {error}")))?;
    Ok(png.into_inner())
}

#[cfg(feature = "ocr-pipeline")]
fn enable_image_ocr_elements(config: &mut crate::core::config::OcrConfig, include_words: bool) {
    let element_config = config
        .element_config
        .get_or_insert_with(crate::types::OcrElementConfig::default);
    element_config.include_elements = true;
    if include_words {
        element_config.min_level = crate::types::OcrElementLevel::Word;
    }
}

#[cfg(feature = "ocr-pipeline")]
fn apply_public_image_ocr_element_policy(document: &mut InternalDocument, config: &crate::core::config::OcrConfig) {
    document.prebuilt_ocr_elements = config.select_public_elements(document.prebuilt_ocr_elements.take());
}

/// Image extractor for various image formats.
///
/// Supports: PNG, JPEG, WebP, BMP, TIFF, GIF.
/// Extracts dimensions, format, and EXIF metadata.
/// Optionally runs OCR when configured.
/// When layout detection is also enabled, uses per-region OCR with
/// markdown formatting based on detected layout classes.
pub struct ImageExtractor;

impl ImageExtractor {
    /// Create a new image extractor.
    pub(crate) fn new() -> Self {
        Self
    }

    fn mark_ocr_extraction(doc: &mut InternalDocument) {
        doc.metadata.ocr_used = true;
        doc.metadata.additional.insert(
            std::borrow::Cow::Borrowed("extraction_method"),
            serde_json::Value::String(crate::types::ExtractionMethod::Ocr.as_str().to_string()),
        );
    }

    /// Extract text from image using OCR with optional page tracking for multi-frame TIFFs.
    ///
    /// A standalone picture is read with the sparse-text default first. When that pass finds no
    /// table but its word boxes share column edges across rows, and the caller chose no PSM, the
    /// picture is read once more as one uniform block (PSM 6), whose row recovery keeps
    /// dot-leader table rows whole (#2503, #2432). The second read is adopted only if it finds a
    /// table and keeps enough letters and digits; see [`image_table_reread::second_read_is_usable`].
    /// A cancelled second read ends the extraction.
    #[cfg(feature = "ocr-pipeline")]
    async fn extract_with_ocr(
        &self,
        content: &[u8],
        mime_type: &str,
        config: &ExtractionConfig,
    ) -> Result<InternalDocument> {
        let first = self.extract_with_ocr_pass(content, mime_type, config).await?;
        self.reread_standalone_table(content, mime_type, config, first).await
    }

    async fn reread_standalone_table(
        &self,
        _content: &[u8],
        _mime_type: &str,
        _config: &ExtractionConfig,
        first: InternalDocument,
    ) -> Result<InternalDocument> {
        Ok(first)
    }

    #[cfg(feature = "ocr-pipeline")]
    async fn extract_with_ocr_pass(
        &self,
        content: &[u8],
        mime_type: &str,
        config: &ExtractionConfig,
    ) -> Result<InternalDocument> {
        use crate::plugins::registry::get_ocr_backend_registry;

        let default_ocr_config;
        let ocr_config = match config.ocr.as_ref() {
            Some(c) => c,
            None => {
                default_ocr_config = crate::core::config::OcrConfig::default();
                &default_ocr_config
            }
        };

        // `vlm_fallback` and explicit multi-stage pipelines were only honoured by the
        // PDF extractor, so a bare image upload always used a single backend and never
        // fell back to the VLM (issue #1339). When such a policy is configured, route
        // the image through the same shared pipeline runner. Gated on `pdf` because the
        // runner lives in that module; a build without `pdf` keeps the single-backend
        // path unchanged. The default (no fallback, no explicit pipeline) also keeps the
        // fast path, so ordinary image OCR is unaffected.
        #[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
        {
            let wants_pipeline = ocr_config.vlm_fallback != crate::core::config::VlmFallbackPolicy::Disabled
                || ocr_config.pipeline.is_some();
            if wants_pipeline && let Some(pipeline) = ocr_config.effective_pipeline() {
                return self.extract_with_ocr_pipeline(content, config, &pipeline).await;
            }
        }

        let backend = {
            crate::plugins::ensure_ocr_backends_initialized();
            let registry = get_ocr_backend_registry();
            let registry = registry.read();
            let backend = registry.get(&ocr_config.backend)?;
            backend
        };

        let mut ocr_config_with_format = ocr_config.clone();
        apply_default_whole_image_tesseract_psm(&mut ocr_config_with_format);
        ocr_config_with_format.output_format = Some(config.output_format.clone());
        ocr_config_with_format.acceleration = config.acceleration.clone();
        // GH#1651: the sibling injection the embedded-image route already does
        // (`extraction::image_ocr`). Without it a standalone image's OCR decode never sees
        // the caller's `ExtractionConfig::security_limits`. Assigned conditionally so an
        // `OcrConfig::security_limits` the caller set directly is not silently replaced by
        // `None`; `ExtractionConfig` still wins whenever it carries a value. ~keep
        if let Some(security_limits) = config.security_limits.clone() {
            ocr_config_with_format.security_limits = Some(security_limits);
        }
        // #1787: the top-level `ExtractionConfig.use_cache = false` only ever bypassed the
        // whole-extraction cache (`extract_file_with_extractor`); the OCR backend's own result
        // cache is governed independently by `TesseractConfig.use_cache` (default `true`), which
        // `OcrBackend::process_image` never sees `ExtractionConfig` to read. Propagate through
        // `backend_options["use_cache"]`, the channel `TesseractBackend::config_to_tesseract`
        // already reads for exactly this purpose. Only the "off" direction is forwarded: a
        // caller who leaves the top-level flag at its default keeps whatever OCR-specific cache
        // setting they configured directly, unmodified. ~keep
        if !config.use_cache {
            let mut options = ocr_config_with_format
                .backend_options
                .take()
                .and_then(|value| match value {
                    serde_json::Value::Object(map) => Some(map),
                    _ => None,
                })
                .unwrap_or_default();
            options.insert("use_cache".to_string(), serde_json::Value::Bool(false));
            ocr_config_with_format.backend_options = Some(serde_json::Value::Object(options));
        }

        let include_words = false;
        enable_image_ocr_elements(&mut ocr_config_with_format, include_words);
        #[cfg(not(target_arch = "wasm32"))]
        if is_implicit_horizontal_tesseract(ocr_config) {
            enable_image_ocr_elements(&mut ocr_config_with_format, true);
        }

        // OCR backends only see `OcrConfig`, which has no route back to
        // `ExtractionConfig::images`, so DPI/dimension normalization from
        // `ImageExtractionConfig` has to happen here, once, before any backend
        // ever sees the bytes (issue #209). ~keep
        #[cfg(feature = "ocr-pipeline")]
        let default_security_limits = crate::extractors::security::SecurityLimits::default();
        #[cfg(feature = "ocr-pipeline")]
        let security_limits = config.security_limits.as_ref().unwrap_or(&default_security_limits);
        #[cfg(feature = "ocr-pipeline")]
        let normalized_ocr_bytes = config
            .images
            .as_ref()
            .map(|images_config| normalize_image_bytes_for_ocr(content, images_config, security_limits, ocr_config))
            .transpose()?;
        #[cfg(feature = "ocr-pipeline")]
        let ocr_input: &[u8] = normalized_ocr_bytes
            .as_ref()
            .map(|normalized| normalized.bytes.as_slice())
            .unwrap_or(content);
        #[cfg(not(feature = "ocr-pipeline"))]
        let ocr_input: &[u8] = content;

        #[cfg(not(target_arch = "wasm32"))]
        let ocr_bytes = std::sync::Arc::new(ocr_input.to_vec());
        #[cfg(not(target_arch = "wasm32"))]
        let ocr_result = process_whole_image_ocr(
            &backend,
            ocr_bytes,
            (ocr_config, &ocr_config_with_format),
            SparseRetryPolicy {
                whole_page_raster: true,
                allow_standard_retry: true,
            },
            config.cancel_token.as_ref(),
        )
        .await?;
        #[cfg(target_arch = "wasm32")]
        let ocr_result = backend.process_image(ocr_input, &ocr_config_with_format).await?;
        let ocr_result = {
            let mut result = ocr_result;
            crate::extraction::derive::adopt_public_ocr_document(
                &mut result,
                config.security_limits.as_ref().unwrap_or(&Default::default()),
            )?;
            result
        };
        #[cfg(feature = "ocr-pipeline")]
        let ocr_result = {
            let mut ocr_result = ocr_result;
            if let Some(metadata) = normalized_ocr_bytes.and_then(|normalized| normalized.metadata) {
                ocr_result.metadata.image_preprocessing = Some(metadata);
            }
            ocr_result
        };

        // Only the GH#1789 numeric repair below mutates this, and that is gated on
        // `ocr` + `pdf`; an unconditional `mut` is a dead `mut` on every narrower leg,
        // which the formula-recognition+pdf clippy gate rejects. Same shape as
        // `ocr_tables`/`ocr_internal_document` just below. ~keep

        let ocr_content = ocr_result.content;
        let ocr_metadata = ocr_result.metadata;
        let ocr_elements = ocr_result.ocr_elements;
        let ocr_formulas = ocr_result.formulas;
        let ocr_llm_usage = ocr_result.llm_usage;

        let processing_warnings = ocr_result.processing_warnings;
        // `ExtractedDocument::tables` (tables reconstructed by OCR table
        // detection) used to be dropped here entirely: nothing on the
        // standalone-image path ever read it, so a table found in a bare
        // image never reached the output.
        // Only the GH#1789 repair below mutates these two, and it is gated on `ocr` + `pdf`,
        // so binding them `mut` under `ocr` alone is a dead `mut` on the ocr-without-pdf leg. ~keep

        let ocr_internal_document = ocr_result.ocr_internal_document;

        // GH#1789: opt-in, same gate and rationale as the PDF mixed-OCR route
        // (`extractors::pdf::ocr::pipeline::numeric_repair_enabled`). See
        // `apply_numeric_repair_to_standalone_image_ocr`'s own doc comment for why this has to
        // touch three separate string representations of the same OCR run, not just
        // `ocr_content`.

        // #2312: a page flagged as symbol noise keeps only its word-shaped lines. The page
        // verdict and photo-title recovery above judged the unpruned run.

        // ~keep The whole image is one OCR run, so its confidence describes page 1 and only
        // page 1. The multi-frame TIFF branch below splits that single run's text by byte
        // offset into several pages with no per-frame OCR of their own, so those pages keep
        // `ocr_confidence: None` rather than repeating a figure that was never measured for
        // them (#1568). Gated on `pdf` because the shared builder lives in that module.

        {
            let _ = mime_type;
            let mut doc = build_image_ocr_document(ocr_internal_document, &ocr_content);
            doc.metadata = ocr_metadata;
            doc.formulas = ocr_formulas;
            doc.llm_usage = ocr_llm_usage;
            doc.processing_warnings = processing_warnings;
            Self::mark_ocr_extraction(&mut doc);
            doc.prebuilt_ocr_elements = ocr_elements;
            let text = ocr_content.trim().to_string();
            if !text.is_empty() {
                doc.prebuilt_pages = Some(vec![crate::types::PageContent {
                    page_number: 1,
                    content: text,
                    tables: vec![],
                    image_indices: vec![],
                    image_preprocessing: None,
                    hierarchy: None,
                    is_blank: None,
                    layout_regions: None,
                    speaker_notes: None,
                    section_name: None,
                    sheet_name: None,
                    ocr_confidence: None,
                    native_content: None,
                }]);
            }
            Ok(doc)
        }
    }

    /// Route a single image through the multi-stage OCR pipeline (issue #1339).
    ///
    /// Decodes the image and reuses the PDF extractor's pipeline runner so that
    /// `vlm_fallback` / explicit `pipeline` policies apply to bare images the same
    /// way they do to PDF pages. Gated on `pdf` because the runner lives there.
    #[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
    async fn extract_with_ocr_pipeline(
        &self,
        content: &[u8],
        config: &ExtractionConfig,
        pipeline: &crate::core::config::OcrPipelineConfig,
    ) -> Result<InternalDocument> {
        let default_security_limits = crate::extractors::security::SecurityLimits::default();
        let security_limits = config.security_limits.as_ref().unwrap_or(&default_security_limits);
        let (image, original_width, original_height, scale) = decode_pipeline_image(content, security_limits)?;
        let source_dpi = crate::extraction::image::resolve_known_source_dpi(
            config
                .ocr
                .as_ref()
                .and_then(crate::extraction::image::explicit_source_dpi_from_ocr_config),
            content,
            original_width,
            original_height,
        )
        .map(|dpi| dpi / f64::from(scale));
        let rgb_bytes = u64::try_from(image.as_raw().len()).map_err(|_| {
            crate::extraction::image_decode::image_dimension_error(image.width(), image.height(), u64::MAX, u64::MAX)
        })?;
        crate::extraction::image_decode::validate_image_live_bytes(
            image.width(),
            image.height(),
            rgb_bytes,
            rgb_bytes,
            security_limits,
        )?;
        let image = image::DynamicImage::ImageRgb8(image);
        let images = [image];

        let (
            text,
            tables,
            ocr_elements,
            pipeline_doc,
            llm_usage,
            _page_texts,
            _rasters,
            formulas,
            _,
            mut pipeline_ocr_confidence,
        ) = Box::pin(crate::extractors::pdf::ocr::run_ocr_pipeline(
            None,
            Some(&images),
            config,
            pipeline,
            None,
            Some(crate::extractors::pdf::ocr::PageOcrHints {
                source_dpi,
                known_full_page_scan: true,
                single_block_pages: None,
                native_page_layers: None,
                ..Default::default()
            }),
        ))
        .await?;

        // Build a clean image document from the pipeline text (keeping the "image"
        // doc type and shape the rest of the image path produces), then carry over
        // the pipeline's elements, formulas, usage, and — crucially — any processing
        // warnings, so a fallback backend that failed is visible to the caller
        // rather than silently swapped for the classical result (issue #1339).
        let prose = crate::extractors::pdf::ocr::strip_embedded_gfm_tables(&text, tables.iter());
        let mut pipeline_doc = pipeline_doc;
        let mut doc = if pipeline_doc.as_ref().is_some_and(|doc| !doc.elements.is_empty()) {
            build_image_ocr_document(pipeline_doc.take(), &prose)
        } else {
            build_image_internal_document(Some(&prose), None)
        };
        Self::mark_ocr_extraction(&mut doc);
        if !ocr_elements.is_empty() {
            doc.prebuilt_ocr_elements = Some(ocr_elements);
        }
        if !formulas.is_empty() {
            doc.formulas = formulas;
        }
        if !llm_usage.is_empty() {
            doc.llm_usage = Some(llm_usage);
        }
        if let Some(pipeline_doc) = pipeline_doc {
            doc.processing_warnings.extend(pipeline_doc.processing_warnings);
            doc.ocr_page_failures.extend(pipeline_doc.ocr_page_failures);
        }

        // ~keep The pipeline document owns the final table set and reading order after
        // normalization, stitching and authoritative public-node replacement (#2226).
        let page_tables: Vec<std::sync::Arc<crate::types::Table>> =
            doc.tables.iter().cloned().map(std::sync::Arc::new).collect();
        if !page_tables.is_empty() {
            // ~keep Page content is re-rendered from the elements tagged with its page number;
            // untagged elements leave the page as plain text, which would omit the tables.
            for element in doc
                .elements
                .iter_mut()
                .filter(|element| !matches!(element.kind, crate::types::internal::ElementKind::Image { .. }))
            {
                element.page.get_or_insert(1);
            }
        }

        let trimmed = prose.trim().to_string();
        if !trimmed.is_empty() || !page_tables.is_empty() {
            doc.prebuilt_pages = Some(vec![crate::types::PageContent {
                page_number: 1,
                content: trimmed,
                tables: page_tables,
                image_indices: vec![],
                image_preprocessing: None,
                hierarchy: None,
                is_blank: None,
                layout_regions: None,
                speaker_notes: None,
                section_name: None,
                sheet_name: None,
                // ~keep The lone image was handed to the runner as page 0, so the winning
                // stage keys its summary at page 1 -- the same page this builds (#1568).
                ocr_confidence: pipeline_ocr_confidence.remove(&1),
                native_content: None,
            }]);
        }

        Ok(doc)
    }
}

/// Push a placeholder `Image` element carrying no binary data.
fn push_image_placeholder(builder: &mut InternalDocumentBuilder) {
    use crate::types::document_structure::ContentLayer;
    use crate::types::internal::{ElementKind, InternalElement, InternalElementId};

    let kind = ElementKind::Image { image_index: 0 };
    let id = InternalElementId::generate(kind.discriminant(), "", None, 0);
    builder.push_element(InternalElement {
        id,
        kind,
        text: String::new(),
        depth: 0,
        page: None,
        bbox: None,
        layer: ContentLayer::Body,
        annotations: Vec::new(),
        attributes: None,
        anchor: None,
        ocr_geometry: None,
        ocr_confidence: None,
        ocr_rotation: None,
    });
}

fn metadata_for_ocr_route(
    content: &[u8],
    limits: &crate::extractors::security::SecurityLimits,
    config: &ExtractionConfig,
) -> Result<crate::extraction::image::ExtractedImageMetadata> {
    #[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
    if !config.disable_ocr
        && config.ocr.as_ref().is_some_and(|ocr| {
            (ocr.vlm_fallback != crate::core::config::VlmFallbackPolicy::Disabled || ocr.pipeline.is_some())
                && ocr.effective_pipeline().is_some()
        })
    {
        return crate::extraction::image::extract_image_metadata_for_processing(content, limits, |width, height| {
            pipeline_encode_fits(width, height, limits)
        });
    }
    #[cfg(not(all(feature = "pdf", feature = "ocr-pipeline")))]
    let _ = config;
    extract_image_metadata_with_security_limits(content, limits)
}

#[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
fn pipeline_encode_fits(width: u32, height: u32, limits: &crate::extractors::security::SecurityLimits) -> bool {
    crate::extractors::pdf::ocr::estimate_png_encode_page_peak_bytes(width, height)
        .is_ok_and(|peak| peak <= limits.max_content_size as u64)
}

/// Decode the image the OCR pipeline encodes once per stage, with its original size and the reduction
/// denominator. Each stage holds the PNG encode working set on top of the raster, so a JPEG whose
/// encode does not fit `max_content_size` is decoded at the finest reduced scale whose encode fits.
#[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
fn decode_pipeline_image(
    content: &[u8],
    security_limits: &crate::extractors::security::SecurityLimits,
) -> Result<(image::RgbImage, u32, u32, u8)> {
    {
        let image = crate::extraction::image::decode_image_to_rgb8_with_security_limits(content, security_limits)?;
        let (width, height) = image.dimensions();
        Ok((image, width, height, 1))
    }
}

/// Keep the standalone source-image envelope while adopting the OCR contribution.
#[cfg(feature = "ocr-pipeline")]
fn build_image_ocr_document(document: Option<InternalDocument>, transcript: &str) -> InternalDocument {
    let Some(mut doc) = document else {
        return build_image_internal_document(Some(transcript), None);
    };
    doc.source_format = "image".into();
    if !doc.elements.iter().any(|element| {
        matches!(
            element.kind,
            crate::types::internal::ElementKind::Image { image_index: 0 }
        )
    }) {
        doc.elements.extend(build_image_internal_document(None, None).elements);
    }
    doc
}

/// Build image paragraphs and the source-image reference, optionally with its bytes.
fn build_image_internal_document(
    ocr_text: Option<&str>,
    image_data: Option<crate::types::ExtractedImage>,
) -> InternalDocument {
    let mut builder = InternalDocumentBuilder::new("image");
    if let Some(text) = ocr_text
        && !text.trim().is_empty()
    {
        for paragraph in split_ocr_paragraphs(text) {
            builder.push_paragraph(&paragraph, vec![], None, None);
        }
    }
    if let Some(img) = image_data {
        builder.push_image(None, img, None, None);
    } else {
        push_image_placeholder(&mut builder);
    }
    builder.build()
}

fn split_ocr_paragraphs(text: &str) -> Vec<String> {
    let mut paragraphs = Vec::new();
    let mut current = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            if !current.is_empty() {
                paragraphs.push(current.join("\n"));
                current.clear();
            }
        } else {
            current.push(line);
        }
    }
    if !current.is_empty() {
        paragraphs.push(current.join("\n"));
    }
    paragraphs
}

impl Default for ImageExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for ImageExtractor {
    fn name(&self) -> &str {
        "image-extractor"
    }

    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn initialize(&self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    fn description(&self) -> &str {
        "Extracts dimensions, format, and EXIF data from images (PNG, JPEG, WebP, BMP, TIFF, GIF)"
    }

    fn author(&self) -> &str {
        "Xberg Team"
    }
}

/// Guards the one-time "max_pages is unenforced for TIFF" warning so it fires
/// at most once per process, not once per extraction — same idiom as
/// `DEFAULT_CAP_WARNED` in `core::config::concurrency`.

static UNENFORCED_TIFF_PAGE_LIMIT_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Emit the "`max_pages` is unenforced for TIFF in this build" warning at most
/// once per `already_warned` guard.
///
/// The guard is a parameter rather than a direct read of
/// [`UNENFORCED_TIFF_PAGE_LIMIT_WARNED`] so the once-per-process property can be
/// tested against a guard the test owns, following
/// `warn_default_thread_cap_once` in `core::config::concurrency`: asserting on
/// the real global from more than one test would be unfixably racy.

fn warn_unenforced_tiff_page_limit_once(already_warned: &std::sync::atomic::AtomicBool, max_pages: usize) {
    if already_warned
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::SeqCst,
            std::sync::atomic::Ordering::SeqCst,
        )
        .is_ok()
    {
        tracing::warn!(
            max_pages,
            "security_limits.max_pages is set but cannot be enforced on TIFF images in this \
             build: counting TIFF frames requires the `ocr` feature, which this build does not \
             have enabled. TIFF page limits are silently unenforced until the binary is rebuilt \
             with `--features ocr`."
        );
    }
}

/// No-op stub for builds without the `ocr` feature: there is no `tiff` crate
/// dependency available to count frames, so `max_pages` is not enforced on TIFF
/// images at all in these builds (see `SecurityLimits::max_pages`'s doc
/// comment). A one-shot warning surfaces the gap instead of silently ignoring
/// a limit the caller explicitly set.

fn enforce_image_page_limit(_content: &[u8], mime_type: &str, config: &ExtractionConfig) -> Result<()> {
    enforce_image_page_limit_stub_with_guard(mime_type, config, &UNENFORCED_TIFF_PAGE_LIMIT_WARNED)
}

/// Pure core of the `not(feature = "ocr")` [`enforce_image_page_limit`], parameterized on the
/// one-shot warning guard so tests can exercise it without touching process-global state.

fn enforce_image_page_limit_stub_with_guard(
    mime_type: &str,
    config: &ExtractionConfig,
    already_warned: &std::sync::atomic::AtomicBool,
) -> Result<()> {
    let max_pages = config.security_limits.as_ref().and_then(|limits| limits.max_pages);
    let Some(max_pages) = max_pages else {
        return Ok(());
    };
    if !mime_type.to_lowercase().contains("tiff") {
        return Ok(());
    }
    warn_unenforced_tiff_page_limit_once(already_warned, max_pages);
    Ok(())
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InternalDocumentExtractor for ImageExtractor {
    async fn extract_content(
        &self,
        content: &[u8],
        mime_type: &str,
        config: &ExtractionConfig,
    ) -> Result<InternalDocument> {
        tracing::debug!(format = "image", size_bytes = content.len(), "extraction starting");
        enforce_image_page_limit(content, mime_type, config)?;
        let default_security_limits = crate::extractors::security::SecurityLimits::default();
        let security_limits = config.security_limits.as_ref().unwrap_or(&default_security_limits);
        let extraction_metadata = metadata_for_ocr_route(content, security_limits, config)?;
        // Computed against the original bytes (before any HEIC->PNG rebinding
        // below) so it reflects the same input `extract_image_metadata` saw. ~keep
        let exif_warning = crate::extraction::exif::extract_exif_warning(content);

        let format_str = extraction_metadata.format;
        let image_metadata = crate::types::ImageMetadata {
            width: extraction_metadata.width,
            height: extraction_metadata.height,
            format: format_str.clone(),
            exif: extraction_metadata.exif_data,
        };

        let (image_kind, kind_confidence) =
            crate::extraction::image_kind::classify(crate::extraction::image_kind::ImageClassifyInput {
                bytes: content,
                format: &format_str,
                width: Some(extraction_metadata.width),
                height: Some(extraction_metadata.height),
                colorspace: None,
                bits_per_component: None,
                is_mask: false,
            });

        let extracted_image = crate::types::ExtractedImage {
            data: bytes::Bytes::copy_from_slice(content),
            format: std::borrow::Cow::Owned(format_str),
            image_index: 0,
            page_number: None,
            width: Some(extraction_metadata.width),
            height: Some(extraction_metadata.height),
            colorspace: None,
            bits_per_component: None,
            is_mask: false,
            description: None,
            ocr_result: None,
            bounding_box: None,
            source_path: None,
            image_kind: Some(image_kind),
            kind_confidence: Some(kind_confidence),
            cluster_id: None,
            caption: None,
            qr_codes: None,
            data_base64: None,
        };

        if config.effective_disable_ocr() {
            let attach_image = if config.wants_own_bytes_in_result() {
                Some(extracted_image)
            } else {
                None
            };
            let mut doc = build_image_internal_document(None, attach_image);
            doc.metadata = Metadata {
                format: Some(crate::types::FormatMetadata::Image(image_metadata)),
                ..Default::default()
            };
            doc.mime_type = mime_type.to_string();
            if let Some(warning) = exif_warning.clone() {
                doc.processing_warnings.push(warning);
            }
            tracing::debug!(
                format = "image",
                "OCR disabled via disable_ocr, returning metadata only"
            );
            return Ok(doc);
        }

        {
            #[cfg(feature = "ocr-pipeline")]
            {
                let mut doc = self.extract_with_ocr(content, mime_type, config).await?;
                if let Some(ocr_config) = config.ocr.as_ref() {
                    apply_public_image_ocr_element_policy(&mut doc, ocr_config);
                }
                Self::mark_ocr_extraction(&mut doc);
                doc.metadata.format = Some(crate::types::FormatMetadata::Image(image_metadata));
                doc.mime_type = mime_type.to_string();
                if config.wants_own_bytes_in_result() {
                    doc.images.push(extracted_image);
                }
                if let Some(warning) = exif_warning.clone() {
                    doc.processing_warnings.push(warning);
                }
                return Ok(doc);
            }
        }

        #[cfg(not(feature = "ocr-pipeline"))]
        {
            let mut doc = build_image_internal_document(None, Some(extracted_image));
            doc.metadata = Metadata {
                format: Some(crate::types::FormatMetadata::Image(image_metadata)),
                ..Default::default()
            };
            doc.mime_type = mime_type.to_string();
            if let Some(warning) = exif_warning.clone() {
                doc.processing_warnings.push(warning);
            }

            tracing::debug!(
                element_count = doc.elements.len(),
                format = "image",
                "extraction complete"
            );
            Ok(doc)
        }
    }

    fn supported_mime_types(&self) -> &[&str] {
        &[
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/pjpeg",
            "image/webp",
            "image/bmp",
            "image/x-bmp",
            "image/x-ms-bmp",
            "image/tiff",
            "image/x-tiff",
            "image/gif",
            "image/jp2",
            "image/j2c",
            "image/x-jbig2",
            "image/x-portable-anymap",
            "image/x-portable-bitmap",
            "image/x-portable-graymap",
            "image/x-portable-pixmap",
            "image/heic",
            "image/heic-sequence",
            "image/heif",
            "image/heif-sequence",
            "image/avif",
            "image/avcs",
        ]
    }

    fn priority(&self) -> i32 {
        50
    }
}

#[cfg(all(test, feature = "pdf", feature = "ocr-pipeline"))]
#[path = "image_table_tests.rs"]
mod image_table_tests;

#[cfg(all(test, not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
#[path = "image_sparse_retry_cancellation_tests.rs"]
mod image_sparse_retry_cancellation_tests;

#[cfg(test)]
mod tests {

    use super::*;

    #[tokio::test]
    async fn rejects_declared_image_dimensions_over_configured_security_budget() {
        let content = crate::extraction::image_decode::bmp_with_declared_dimensions(100, 100);
        let config = ExtractionConfig {
            disable_ocr: true,
            security_limits: Some(crate::extractors::security::SecurityLimits {
                max_content_size: 1024,
                ..Default::default()
            }),
            ..Default::default()
        };

        let error = ImageExtractor::new()
            .extract_content(&content, "image/bmp", &config)
            .await
            .expect_err("the extractor must apply the configured budget before decoding pixels");

        assert!(matches!(error, crate::XbergError::Validation { .. }));
        assert!(error.to_string().contains("100x100"));
        assert!(error.to_string().contains("security_limits.max_content_size"));
    }

    /// #860: `normalize_image_bytes_for_ocr` applies `ImageExtractionConfig`'s
    /// `max_image_dimension` / DPI settings at the extractor boundary, because OCR
    /// backends only ever see `OcrConfig` and have no route back to them (#209).
    ///
    /// It used to be gated on `feature = "ocr"`, while `extract_with_ocr` is gated on
    /// `ocr` OR `ocr-wasm` OR `ocr-pipeline`. Every `candle-*` backend — and any
    /// `liter-llm` VLM-only build — implies `ocr-pipeline` but NOT `ocr`, so those
    /// builds took the `#[cfg(not(feature = "ocr"))]` arm and passed the raw bytes
    /// through with the settings silently dropped.
    ///
    /// Run this under a candle-only feature set to see the fix: before it, the
    /// function did not exist in such a build at all.
    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn normalizes_a_standalone_image_to_the_configured_maximum_dimension() {
        let oversized = image::RgbImage::from_pixel(1200, 600, image::Rgb([255, 255, 255]));
        let png = encode_rgb_as_png(oversized.as_raw(), 1200, 600).expect("encode the fixture");

        let images_config = crate::core::config::ImageExtractionConfig {
            max_image_dimension: 300,
            auto_adjust_dpi: false,
            ..Default::default()
        };
        let normalized = normalize_image_bytes_for_ocr(
            &png,
            &images_config,
            &crate::extractors::security::SecurityLimits::default(),
            &crate::core::config::OcrConfig::default(),
        )
        .expect("normal image should remain within the default decode budget");

        assert_ne!(
            normalized.bytes, png,
            "the oversized image should not pass through unchanged"
        );
        let decoded = image::load_from_memory(&normalized.bytes).expect("decode the normalized image");
        assert_eq!(
            (decoded.width(), decoded.height()),
            (300, 150),
            "the long edge should be clamped to max_image_dimension, preserving the 2:1 aspect ratio"
        );
        let metadata = normalized
            .metadata
            .expect("successful standalone-image normalization must retain metadata");
        assert_eq!(metadata.original_dimensions.width, 1200);
        assert_eq!(metadata.original_dimensions.height, 600);
        assert_eq!(
            metadata.new_dimensions.as_ref().map(|dimensions| dimensions.width),
            Some(300)
        );
        assert_eq!(
            metadata.new_dimensions.as_ref().map(|dimensions| dimensions.height),
            Some(150)
        );
        assert!(metadata.dimension_clamped);
    }

    // GH#1630: standalone image OCR discarded PNG `pHYs` density and always assumed 72 DPI, so
    // a genuine 300 DPI scan was resampled even when the requested `target_dpi` was already 300.
    // This is the actual site the reporter's repro exercised: `extract_with_ocr` normalizes the
    // bytes here, at the extractor boundary, before any backend (including `perform_ocr`, fixed
    // separately) ever sees them.
    mod png_source_dpi_at_extractor_boundary {
        use super::*;

        /// Standard PNG CRC-32 (polynomial 0xEDB88320), needed to splice a well-formed `pHYs`
        /// chunk into a real encoded PNG. Test-fixture construction only — the parser being
        /// exercised, `extraction::image::png_pixel_density_dpi`, is defined once and shared;
        /// see GH#1621.
        fn png_crc32(data: &[u8]) -> u32 {
            let mut crc: u32 = 0xFFFF_FFFF;
            for &byte in data {
                crc ^= u32::from(byte);
                for _ in 0..8 {
                    let mask = (crc & 1).wrapping_neg();
                    crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
                }
            }
            !crc
        }

        fn png_chunk(chunk_type: &[u8; 4], data: &[u8]) -> Vec<u8> {
            let mut body = Vec::with_capacity(4 + data.len());
            body.extend_from_slice(chunk_type);
            body.extend_from_slice(data);
            let mut out = Vec::with_capacity(8 + body.len());
            out.extend_from_slice(&(u32::try_from(data.len()).unwrap()).to_be_bytes());
            out.extend_from_slice(&body);
            out.extend_from_slice(&png_crc32(&body).to_be_bytes());
            out
        }

        /// Splice a `pHYs` chunk expressing `ppu` pixels-per-metre (both axes, unit = meter)
        /// into a real encoded PNG, immediately after IHDR (signature 8 bytes + IHDR chunk 25
        /// bytes) and always before IDAT.
        fn png_with_phys_density(width: u32, height: u32, ppu: u32) -> Vec<u8> {
            const SIGNATURE_AND_IHDR_LEN: usize = 8 + 25;
            let img = image::RgbImage::from_pixel(width, height, image::Rgb([255, 255, 255]));
            let png = encode_rgb_as_png(img.as_raw(), width, height).expect("encode fixture PNG");

            let mut phys_data = Vec::with_capacity(9);
            phys_data.extend_from_slice(&ppu.to_be_bytes());
            phys_data.extend_from_slice(&ppu.to_be_bytes());
            phys_data.push(1); // unit = meter
            let phys_chunk = png_chunk(b"pHYs", &phys_data);

            let mut out = Vec::with_capacity(png.len() + phys_chunk.len());
            out.extend_from_slice(&png[..SIGNATURE_AND_IHDR_LEN]);
            out.extend_from_slice(&phys_chunk);
            out.extend_from_slice(&png[SIGNATURE_AND_IHDR_LEN..]);
            out
        }

        /// 11811 px/m is the reporter's fixture value: 11811 * 0.0254 = 299.9994 DPI, not an
        /// exact 300 — every assertion below tolerates that rather than requiring an exact
        /// integer match.
        const REPORTER_FIXTURE_PPU: u32 = 11_811;
        const EXPECTED_DPI_FROM_REPORTER_FIXTURE: f64 = 299.999_4;
        const DPI_TOLERANCE: f64 = 0.001;

        fn images_config_targeting(target_dpi: i32) -> crate::core::config::ImageExtractionConfig {
            crate::core::config::ImageExtractionConfig {
                target_dpi,
                auto_adjust_dpi: false,
                ..Default::default()
            }
        }

        #[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
        #[tokio::test]
        async fn standalone_image_pipeline_keeps_whole_image_scan_hints() {
            use crate::core::config::{OcrConfig, OcrPipelineConfig, OcrPipelineStage, OcrQualityThresholds};
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};
            use crate::types::ExtractedDocument;
            use std::sync::{Arc, Mutex};

            struct TesseractHintCapturingBackend {
                config: Mutex<Option<OcrConfig>>,
            }

            #[async_trait::async_trait]
            impl OcrBackend for TesseractHintCapturingBackend {
                fn backend_type(&self) -> OcrBackendType {
                    OcrBackendType::Custom
                }
                fn supports_language(&self, _: &str) -> bool {
                    true
                }
                async fn process_image(&self, _: &[u8], config: &OcrConfig) -> crate::Result<ExtractedDocument> {
                    *self.config.lock().unwrap() = Some(config.clone());
                    Ok(ExtractedDocument {
                        content: "standalone image text".to_string(),
                        ..Default::default()
                    })
                }
            }

            impl Plugin for TesseractHintCapturingBackend {
                fn name(&self) -> &str {
                    "tesseract"
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

            let backend = Arc::new(TesseractHintCapturingBackend {
                config: Mutex::new(None),
            });

            let pipeline = OcrPipelineConfig {
                stages: vec![OcrPipelineStage {
                    backend: "tesseract".to_string(),
                    priority: 100,
                    language: None,
                    tesseract_config: None,
                    paddle_ocr_config: None,
                    paddle_ocr_settings: None,
                    vlm_config: None,
                    backend_options: None,
                }],
                quality_thresholds: OcrQualityThresholds {
                    pipeline_min_quality: 0.0,
                    ..Default::default()
                },
            };
            let config = ExtractionConfig {
                ocr: Some(OcrConfig {
                    pipeline: Some(pipeline.clone()),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let png = png_with_phys_density(16, 16, REPORTER_FIXTURE_PPU);

            crate::plugins::registry::with_test_ocr_backend(
                backend.clone(),
                ImageExtractor::new().extract_with_ocr_pipeline(&png, &config, &pipeline),
            )
            .await
            .expect("standalone image pipeline succeeds");

            let captured = backend.config.lock().unwrap().clone().expect("backend was called");
            let source_dpi = captured
                .backend_options
                .as_ref()
                .and_then(|options| options.get("source_dpi"))
                .and_then(serde_json::Value::as_f64);
            assert!(
                source_dpi.is_some_and(|dpi| (dpi - EXPECTED_DPI_FROM_REPORTER_FIXTURE).abs() < DPI_TOLERANCE),
                "pipeline must preserve the standalone image's embedded source DPI; got {source_dpi:?}"
            );
            assert_eq!(
                captured
                    .backend_options
                    .as_ref()
                    .and_then(|options| options.get("known_full_page_scan"))
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
            assert!(
                captured
                    .tesseract_config
                    .as_ref()
                    .and_then(|config| config.psm)
                    .is_some()
            );
        }

        /// Unit-level check on `normalize_image_bytes_for_ocr` directly: a `target_dpi=300`
        /// request against a genuine ~300 DPI PNG must not resize, and must report the true
        /// embedded density rather than the fabricated 72 assumption.
        #[cfg(feature = "ocr-pipeline")]
        #[test]
        fn does_not_resize_when_target_dpi_matches_known_png_density() {
            let png = png_with_phys_density(16, 16, REPORTER_FIXTURE_PPU);
            let images_config = images_config_targeting(300);

            let normalized = normalize_image_bytes_for_ocr(
                &png,
                &images_config,
                &crate::extractors::security::SecurityLimits::default(),
                &crate::core::config::OcrConfig::default(),
            )
            .expect("normalization must succeed");

            let metadata = normalized.metadata.expect("preprocessing metadata is recorded");
            assert!(
                metadata.skipped_resize,
                "a 300 target against a ~300 detected source must not resize"
            );
            assert!(
                (metadata.original_dpi.horizontal - EXPECTED_DPI_FROM_REPORTER_FIXTURE).abs() < DPI_TOLERANCE,
                "expected ~{EXPECTED_DPI_FROM_REPORTER_FIXTURE} DPI, got {}",
                metadata.original_dpi.horizontal
            );
        }

        /// Control: a PNG with no density metadata must still default to 72 DPI and still
        /// resize against a 300 target — the pre-fix, and still-correct-for-this-case, behaviour.
        #[cfg(feature = "ocr-pipeline")]
        #[test]
        fn still_resizes_and_assumes_72_dpi_when_png_has_no_density_metadata() {
            let img = image::RgbImage::from_pixel(16, 16, image::Rgb([255, 255, 255]));
            let png = encode_rgb_as_png(img.as_raw(), 16, 16).expect("encode fixture PNG");
            let images_config = images_config_targeting(300);

            let normalized = normalize_image_bytes_for_ocr(
                &png,
                &images_config,
                &crate::extractors::security::SecurityLimits::default(),
                &crate::core::config::OcrConfig::default(),
            )
            .expect("normalization must succeed");

            let metadata = normalized.metadata.expect("preprocessing metadata is recorded");
            assert!(
                !metadata.skipped_resize,
                "a 300 target against an assumed-72 source must resize"
            );
            assert_eq!(
                metadata.original_dpi,
                crate::types::ImageDpi::from((72.0, 72.0)),
                "no embedded metadata must leave the historical 72 assumption unchanged"
            );
        }
    }

    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn should_apply_vertical_block_psm_to_default_vertical_tesseract_config() {
        let mut ocr_config = crate::core::config::OcrConfig {
            language: vec!["jpn_vert".to_string()],
            ..Default::default()
        };

        apply_default_whole_image_tesseract_psm(&mut ocr_config);

        let tesseract_config = ocr_config
            .tesseract_config
            .as_ref()
            .expect("whole-image OCR must materialize Tesseract configuration");
        assert_eq!(tesseract_config.psm, Some(VERTICAL_BLOCK_TESSERACT_PSM));
        assert_eq!(tesseract_config.language, vec!["jpn_vert"]);
    }

    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn should_apply_default_whole_image_psm_to_horizontal_tesseract_config() {
        let mut ocr_config = crate::core::config::OcrConfig {
            language: vec!["eng".to_string()],
            ..Default::default()
        };

        apply_default_whole_image_tesseract_psm(&mut ocr_config);

        let tesseract_config = ocr_config
            .tesseract_config
            .as_ref()
            .expect("whole-image OCR must materialize Tesseract configuration");
        assert_eq!(tesseract_config.psm, Some(WHOLE_IMAGE_TESSERACT_PSM));
        assert_eq!(tesseract_config.language, vec!["eng"]);
    }

    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn should_preserve_explicit_whole_image_tesseract_psm() {
        let mut ocr_config = crate::core::config::OcrConfig {
            language: vec!["jpn_vert".to_string()],
            tesseract_config: Some(crate::types::TesseractConfig {
                language: vec!["jpn_vert".to_string()],
                psm: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        };

        apply_default_whole_image_tesseract_psm(&mut ocr_config);

        let tesseract_config = ocr_config
            .tesseract_config
            .as_ref()
            .expect("explicit config must remain");
        assert_eq!(tesseract_config.psm, Some(4));
        assert_eq!(tesseract_config.language, vec!["jpn_vert"]);
    }

    // Regression test for #1573: `TesseractConfig()` with every field left at its
    // default must produce the SAME effective PSM as no `TesseractConfig` at all, not
    // fall back to the internal engine default (PSM 3 native / 6 wasm). Previously
    // `apply_default_tesseract_psm` keyed off `tesseract_config.is_some()`, so a caller
    // who materialized the struct (even with every field default) silently lost the
    // whole-image PSM.
    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn should_apply_same_default_psm_whether_or_not_tesseract_config_struct_is_present() {
        let mut without_struct = crate::core::config::OcrConfig {
            language: vec!["eng".to_string()],
            ..Default::default()
        };
        let mut with_default_struct = crate::core::config::OcrConfig {
            language: vec!["eng".to_string()],
            tesseract_config: Some(crate::types::TesseractConfig::default()),
            ..Default::default()
        };

        apply_default_whole_image_tesseract_psm(&mut without_struct);
        apply_default_whole_image_tesseract_psm(&mut with_default_struct);

        let psm_without_struct = without_struct
            .tesseract_config
            .expect("whole-image OCR must materialize Tesseract configuration")
            .psm;
        let psm_with_default_struct = with_default_struct
            .tesseract_config
            .expect("struct was already present")
            .psm;

        assert_eq!(psm_without_struct, Some(WHOLE_IMAGE_TESSERACT_PSM));
        assert_eq!(
            psm_without_struct, psm_with_default_struct,
            "TesseractConfig::default() must resolve to the same effective PSM as no TesseractConfig at all"
        );
    }

    // An explicitly set `psm`, alongside another explicitly set field, must still be
    // honoured and not overwritten by the automatic default (#1573).
    #[cfg(feature = "ocr-pipeline")]
    #[test]
    fn should_not_overwrite_explicit_psm_when_another_field_is_also_set() {
        let mut ocr_config = crate::core::config::OcrConfig {
            language: vec!["eng".to_string()],
            tesseract_config: Some(crate::types::TesseractConfig {
                psm: Some(7),
                enable_table_detection: false,
                ..Default::default()
            }),
            ..Default::default()
        };

        apply_default_whole_image_tesseract_psm(&mut ocr_config);

        let tesseract_config = ocr_config.tesseract_config.expect("explicit config must remain");
        assert_eq!(tesseract_config.psm, Some(7));
        assert!(!tesseract_config.enable_table_detection);
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "ocr-pipeline"))]
    mod sparse_image_ocr_fallback_tests {
        use super::*;

        fn result_with_word_confidences(confidences: &[f64]) -> crate::types::ExtractedDocument {
            let ocr_elements = confidences
                .iter()
                .enumerate()
                .map(|(index, confidence)| crate::types::OcrElement {
                    text: format!("word-{index}"),
                    confidence: crate::types::OcrConfidence {
                        recognition: *confidence,
                        ..Default::default()
                    },
                    level: crate::types::OcrElementLevel::Word,
                    ..Default::default()
                })
                .collect();
            crate::types::ExtractedDocument {
                ocr_elements: Some(ocr_elements),
                ..Default::default()
            }
        }

        fn result_with_content_and_confidence(content: &str, confidence: f64) -> crate::types::ExtractedDocument {
            let mut result = crate::types::ExtractedDocument {
                content: content.to_string(),
                ..Default::default()
            };
            result
                .metadata
                .additional
                .insert("mean_text_conf".into(), serde_json::json!(confidence));
            result
        }

        #[test]
        fn should_retry_implicit_horizontal_tesseract_when_words_are_sparse() {
            let config = crate::core::config::OcrConfig::default();
            let confidences = vec![0.10; SPARSE_IMAGE_OCR_WORD_LIMIT];
            let result = result_with_word_confidences(&confidences);

            assert!(should_retry_sparse_image_ocr(&config, &result));
        }

        #[test]
        fn should_not_retry_implicit_horizontal_tesseract_when_words_are_dense() {
            let config = crate::core::config::OcrConfig::default();
            let confidences = vec![0.10; SPARSE_IMAGE_OCR_WORD_LIMIT + 1];
            let result = result_with_word_confidences(&confidences);

            assert!(!should_retry_sparse_image_ocr(&config, &result));
        }

        #[test]
        fn should_not_retry_sparse_primary_with_robust_word_confidences() {
            let config = crate::core::config::OcrConfig::default();
            let result = result_with_word_confidences(&[0.90, 0.95]);

            assert!(!should_retry_sparse_image_ocr(&config, &result));
        }

        #[test]
        fn should_exclude_explicit_psm_and_vertical_tesseract_from_sparse_retry() {
            let result = result_with_word_confidences(&[0.10]);
            let explicit_psm_config = crate::core::config::OcrConfig {
                tesseract_config: Some(crate::types::TesseractConfig {
                    psm: Some(4),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let vertical_config = crate::core::config::OcrConfig {
                language: vec!["jpn_vert".to_string()],
                ..Default::default()
            };
            let other_backend_config = crate::core::config::OcrConfig {
                backend: "paddle-ocr".to_string(),
                ..Default::default()
            };

            assert!(!should_retry_sparse_image_ocr(&explicit_psm_config, &result));
            assert!(!should_retry_sparse_image_ocr(&vertical_config, &result));
            assert!(!should_retry_sparse_image_ocr(&other_backend_config, &result));
        }

        // Regression test for #1573: a `TesseractConfig` may be present for a reason
        // unrelated to `psm` (e.g. table detection toggled off). The sparse-text retry
        // must still trigger as long as `psm` itself is unset — struct presence alone
        // no longer disables it. Previously `is_implicit_horizontal_tesseract` keyed off
        // `tesseract_config.is_none()`, so this case never retried.
        #[test]
        fn should_retry_sparse_image_ocr_when_tesseract_config_present_but_psm_unset() {
            let config = crate::core::config::OcrConfig {
                tesseract_config: Some(crate::types::TesseractConfig {
                    enable_table_detection: false,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let confidences = vec![0.10; SPARSE_IMAGE_OCR_WORD_LIMIT];
            let result = result_with_word_confidences(&confidences);

            assert!(should_retry_sparse_image_ocr(&config, &result));
        }

        #[test]
        fn should_reject_high_mean_fallback_with_bad_tenth_percentile() {
            let mut confidences = vec![0.95; 16];
            confidences.extend([0.14, 0.14]);
            let mean = confidences.iter().sum::<f64>() / confidences.len() as f64;
            let result = result_with_word_confidences(&confidences);

            assert!(mean > 0.80, "fixture must model misleadingly high mean confidence");
            assert!(!has_robust_word_confidence_distribution(&result));
        }

        #[test]
        fn should_select_fallback_with_robust_word_confidences() {
            let mut confidences = vec![0.95; 9];
            confidences.push(SPARSE_IMAGE_OCR_MIN_WORD_CONFIDENCE);
            let result = result_with_word_confidences(&confidences);

            assert!(has_robust_word_confidence_distribution(&result));
        }

        /// #2699: a re-read that returns no usable words must not be kept over the first pass. ~keep
        #[test]
        fn should_reject_fallback_that_read_no_usable_words() {
            assert!(!has_robust_word_confidence_distribution(&result_with_word_confidences(
                &[]
            )));
            assert!(!has_robust_word_confidence_distribution(
                &crate::types::ExtractedDocument::default()
            ));
            let mut whitespace_only = result_with_word_confidences(&[0.95, 0.95]);
            for element in whitespace_only.ocr_elements.iter_mut().flatten() {
                element.text = "  ".to_string();
            }
            assert!(!has_robust_word_confidence_distribution(&whitespace_only));
        }

        #[test]
        fn robust_word_confidence_gate_decides_at_the_percentile_and_confidence_thresholds() {
            let with_low = |good: usize, low: usize| {
                let mut confidences = vec![0.95; good];
                confidences.extend(vec![0.2; low]);
                confidences
            };
            let cases: [(&str, Vec<f64>, bool); 6] = [
                ("a single confident word", vec![0.95], true),
                (
                    "a word exactly at the minimum confidence is not low",
                    vec![0.30; 10],
                    true,
                ),
                (
                    "one low word in ten sits on the tenth percentile",
                    with_low(9, 1),
                    false,
                ),
                (
                    "one low word in twenty stays under the tenth percentile",
                    with_low(19, 1),
                    true,
                ),
                (
                    "two low words in twenty reach the tenth percentile",
                    with_low(18, 2),
                    false,
                ),
                ("every word low", with_low(0, 12), false),
            ];
            for (name, confidences, expected) in cases {
                assert_eq!(
                    has_robust_word_confidence_distribution(&result_with_word_confidences(&confidences)),
                    expected,
                    "{name}"
                );
            }
        }

        #[test]
        fn should_build_psm3_fallback_with_explicit_grayscale_enhancement() {
            let mut whole_image_config = crate::core::config::OcrConfig::default();
            apply_default_whole_image_tesseract_psm(&mut whole_image_config);

            let fallback_config = sparse_image_ocr_fallback_config(&whole_image_config);
            let tesseract_config = fallback_config
                .tesseract_config
                .expect("fallback must materialize Tesseract configuration");

            assert_eq!(tesseract_config.psm, Some(SPARSE_IMAGE_OCR_FALLBACK_PSM));
            assert_eq!(
                tesseract_config
                    .preprocessing
                    .as_ref()
                    .expect("fallback must materialize preprocessing")
                    .binarization_method,
                "none"
            );
            let preprocessing = tesseract_config.preprocessing.unwrap();
            assert!(!preprocessing.deskew);
            assert!(preprocessing.contrast_enhance);
        }

        #[test]
        fn should_normalize_tesseract_metadata_confidence_from_zero_to_one() {
            for (raw, expected) in [(0.0, 0.0), (1.0, 0.01), (91.0, 0.91), (100.0, 1.0)] {
                let result = result_with_content_and_confidence("نص", raw);
                assert_eq!(normalized_tesseract_mean_confidence(&result), Some(expected));
            }
        }

        #[test]
        fn should_select_rtl_retry_only_when_grounded_tokens_increase() {
            let primary = result_with_content_and_confidence("تقرير عربي REQ-2026-104 42", 91.0);
            let candidate = result_with_content_and_confidence("تقرير عربي كامل REQ-2026-104 42", 84.0);

            assert!(rtl_retry_candidate_improves(&primary, &candidate));
        }

        #[test]
        fn should_reject_unrecovered_rtl_retry_that_loses_or_duplicates_primary_identifiers() {
            let primary = result_with_content_and_confidence("تقرير عربي REQ-2026-104 42", 91.0);
            let lost = result_with_content_and_confidence("تقرير عربي كامل 42", 92.0);
            let duplicated = result_with_content_and_confidence("تقرير عربي كامل REQ-2026-104 REQ-2026-104 42", 92.0);

            assert!(!identifier_slots_are_preserved_after_recovery(&primary, &lost));
            assert!(!identifier_slots_are_preserved_after_recovery(&primary, &duplicated));
        }

        #[test]
        fn should_match_identifier_slots_with_close_ocr_substitution() {
            assert!(identifier_slots_are_preserved(
                "طلب REQ-2026-104 OCR-v2.1",
                "1. طلب REO-2026-104 OCR-v2.1 42"
            ));
            assert!(!identifier_slots_are_preserved(
                "طلب REQ-2026-104 OCR-v2.1",
                "طلب OCR-v2.1"
            ));
            assert!(!identifier_slots_are_preserved(
                "طلب REQ-2026-104 OCR-v2.1",
                "طلب REO-2026-104 REO-2026-104 OCR-v2.1"
            ));
            assert_eq!(
                identifier_slots("1.افتح 3.احتفظ REQ-2026-104 1,250.75 2026"),
                ["req-2026-104", "1,250.75", "2026"]
            );
        }

        #[test]
        fn should_require_compatible_identifier_evidence() {
            assert!(compatible_identifier_recovery("A17", "A-17"));
            assert!(compatible_identifier_recovery("880-2026-104", "REQ-2026-104"));
            assert!(compatible_identifier_recovery("8-09", "B-09"));
            assert!(compatible_identifier_recovery("001-2.1", "OCR-v2.1"));
            assert!(!compatible_identifier_recovery("BAD-2026", "GOOD-2026"));
            assert!(!compatible_identifier_recovery("1234567-2026-104", "REQ-2026-104"));
            assert!(!compatible_identifier_recovery("2026-10-08", "REQ-2026-104"));

            let mixed = crate::core::config::OcrConfig {
                language: vec!["ara".to_string(), "eng".to_string()],
                ..Default::default()
            };
            assert!(is_mixed_ltr_recovery_eligible_config(&mixed));
            assert!(!is_mixed_ltr_recovery_eligible_config(
                &crate::core::config::OcrConfig::default()
            ));

            let word = |text: &str, language: Option<&str>| {
                let mut element = crate::types::OcrElement {
                    text: text.to_string(),
                    level: crate::types::OcrElementLevel::Word,
                    ..Default::default()
                };
                if let Some(language) = language {
                    element
                        .backend_metadata
                        .insert("word_language".to_string(), serde_json::json!(language));
                }
                element
            };
            assert!(is_rtl_recovery_suspect(&word("880-2026-104", Some("ara")), true));
            assert!(is_rtl_recovery_suspect(&word("A17", Some("eng")), true));
            assert!(!is_rtl_recovery_suspect(&word("A17", None), true));
            assert!(!is_rtl_recovery_suspect(&word("2026-10-08", Some("ara")), true));
            assert!(!is_rtl_recovery_suspect(&word("ordinary", Some("ara")), true));
            assert!(!is_rtl_recovery_suspect(&word("A17", Some("eng")), false));

            let mut evidence = word("A-17", Some("eng"));
            evidence.confidence.recognition = 0.9;
            let duplicate = evidence.clone();
            assert!(unique_compatible_recovery_evidence("2-17", [&evidence].into_iter()).is_some());
            assert!(unique_compatible_recovery_evidence("2-17", [&evidence, &duplicate].into_iter()).is_none());
        }

        #[test]
        fn should_apply_identifier_recovery_transactionally_to_all_representations() {
            let source = "(2-17),";
            let mut selected = crate::types::ExtractedDocument {
                content: "code (2-17),".to_string(),
                ocr_elements: Some(vec![crate::types::OcrElement {
                    text: source.to_string(),
                    level: crate::types::OcrElementLevel::Word,
                    ..Default::default()
                }]),
                tables: vec![crate::types::Table {
                    cells: vec![vec!["(2-17),".to_string()]],
                    markdown: "| (2-17), |".to_string(),
                    columns: Some(vec!["(2-17),".to_string()]),
                    ..Default::default()
                }],
                ..Default::default()
            };
            selected.ocr_internal_document = Some(crate::types::internal::InternalDocument {
                elements: vec![crate::types::internal::InternalElement::text(
                    crate::types::internal::ElementKind::Paragraph,
                    "code (2-17),",
                    0,
                )],
                tables: selected.tables.clone(),
                ..Default::default()
            });
            let evidence = crate::types::OcrElement {
                text: "A-17".to_string(),
                confidence: crate::types::OcrConfidence {
                    recognition: 0.9,
                    ..Default::default()
                },
                level: crate::types::OcrElementLevel::Word,
                ..Default::default()
            };

            assert!(apply_identifier_recovery_transaction(
                &mut selected,
                0,
                source,
                &evidence
            ));
            assert_eq!(selected.content, "code (A-17),");
            assert_eq!(selected.ocr_elements.as_ref().unwrap()[0].text, "(A-17),");
            assert_eq!(selected.tables[0].cells[0][0], "(A-17),");
            assert_eq!(selected.tables[0].columns.as_ref().unwrap()[0], "(A-17),");
            assert!(selected.tables[0].markdown.contains("(A-17),"));
            let internal = selected.ocr_internal_document.as_ref().unwrap();
            assert_eq!(internal.elements[0].text, "code (A-17),");
            assert_eq!(internal.tables[0].cells[0][0], "(A-17),");

            let mut ambiguous = selected.clone();
            ambiguous.content = "2-17 and 2-17".to_string();
            ambiguous.ocr_elements.as_mut().unwrap()[0].text = "2-17".to_string();
            let before = ambiguous.clone();
            assert!(!apply_identifier_recovery_transaction(
                &mut ambiguous,
                0,
                "2-17",
                &evidence
            ));
            assert_eq!(ambiguous.content, before.content);
            assert_eq!(
                ambiguous.ocr_elements.unwrap()[0].text,
                before.ocr_elements.unwrap()[0].text
            );
        }

        #[test]
        fn should_capture_internal_words_below_public_confidence_threshold() {
            let public = crate::core::config::OcrConfig {
                language: vec!["ara".to_string(), "eng".to_string()],
                element_config: Some(crate::types::OcrElementConfig {
                    include_elements: true,
                    min_level: crate::types::OcrElementLevel::Word,
                    min_confidence: 0.95,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let internal = internal_rtl_recovery_config(&public);
            assert_eq!(internal.element_config.as_ref().unwrap().min_confidence, 0.0);
            let low_confidence = crate::types::OcrElement {
                text: "2-17".to_string(),
                confidence: crate::types::OcrConfidence {
                    recognition: 0.5,
                    ..Default::default()
                },
                level: crate::types::OcrElementLevel::Word,
                ..Default::default()
            };
            assert!(public.select_public_elements(Some(vec![low_confidence])).is_none());
        }

        #[tokio::test]
        async fn should_apply_public_confidence_only_after_internal_recovery_capture() {
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};

            struct CapturingBackend(std::sync::Mutex<Option<crate::core::config::OcrConfig>>);

            #[async_trait::async_trait]
            impl OcrBackend for CapturingBackend {
                fn backend_type(&self) -> OcrBackendType {
                    OcrBackendType::Custom
                }
                fn supports_language(&self, _: &str) -> bool {
                    true
                }
                async fn process_image(
                    &self,
                    _: &[u8],
                    config: &crate::core::config::OcrConfig,
                ) -> crate::Result<crate::types::ExtractedDocument> {
                    *self.0.lock().unwrap() = Some(config.clone());
                    Ok(crate::types::ExtractedDocument {
                        content: std::iter::repeat_n("كلمة", 60).collect::<Vec<_>>().join(" "),
                        ocr_elements: Some(vec![crate::types::OcrElement {
                            text: "كلمة".to_string(),
                            confidence: crate::types::OcrConfidence {
                                recognition: 0.5,
                                ..Default::default()
                            },
                            level: crate::types::OcrElementLevel::Word,
                            ..Default::default()
                        }]),
                        ..Default::default()
                    })
                }
            }

            impl Plugin for CapturingBackend {
                fn name(&self) -> &str {
                    "capturing-recovery"
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

            let original = crate::core::config::OcrConfig {
                language: vec!["ara".to_string(), "eng".to_string()],
                element_config: Some(crate::types::OcrElementConfig {
                    include_elements: true,
                    min_level: crate::types::OcrElementLevel::Word,
                    min_confidence: 0.95,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut effective = original.clone();
            apply_default_whole_image_tesseract_psm(&mut effective);
            let backend = std::sync::Arc::new(CapturingBackend(std::sync::Mutex::new(None)));
            let erased: std::sync::Arc<dyn OcrBackend> = backend.clone();
            let result = process_whole_image_ocr_with_sparse_retry(
                &erased,
                std::sync::Arc::new(vec![1]),
                &original,
                &effective,
                SparseRetryPolicy {
                    whole_page_raster: true,
                    allow_standard_retry: false,
                },
                None,
            )
            .await
            .expect("capture must succeed");

            assert_eq!(
                backend
                    .0
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .element_config
                    .as_ref()
                    .unwrap()
                    .min_confidence,
                0.0
            );
            assert!(result.ocr_elements.is_none());
        }

        #[tokio::test]
        async fn should_rollback_rtl_fallback_when_identifier_corruption_has_no_verified_recovery() {
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};

            struct SequenceBackend(std::sync::Mutex<std::collections::VecDeque<crate::types::ExtractedDocument>>);

            #[async_trait::async_trait]
            impl OcrBackend for SequenceBackend {
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
                    self.0
                        .lock()
                        .unwrap()
                        .pop_front()
                        .ok_or_else(|| crate::XbergError::Other("unexpected OCR call".to_string()))
                }
            }

            impl Plugin for SequenceBackend {
                fn name(&self) -> &str {
                    "sequence"
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

            let result = |content: &str, identifier: &str, language: &str| {
                let mut word = crate::types::OcrElement {
                    text: identifier.to_string(),
                    level: crate::types::OcrElementLevel::Word,
                    geometry: crate::types::OcrBoundingGeometry::Rectangle {
                        left: 10,
                        top: 10,
                        width: 40,
                        height: 10,
                    },
                    ..Default::default()
                };
                word.backend_metadata
                    .insert("word_language".to_string(), serde_json::json!(language));
                let mut result = crate::types::ExtractedDocument {
                    content: content.to_string(),
                    ocr_elements: Some(vec![word]),
                    ..Default::default()
                };
                result
                    .metadata
                    .additional
                    .insert("mean_text_conf".into(), serde_json::json!(92.0));
                result
            };
            let primary = result("تقرير 880-2026-104", "880-2026-104", "ara");
            let fallback = result("تقرير عربي كامل BAD-2026-104", "BAD-2026-104", "eng");
            let backend: std::sync::Arc<dyn OcrBackend> = std::sync::Arc::new(SequenceBackend(std::sync::Mutex::new(
                std::collections::VecDeque::from([primary.clone(), fallback]),
            )));
            let original = crate::core::config::OcrConfig {
                language: vec!["ara".to_string(), "eng".to_string()],
                ..Default::default()
            };
            let mut effective = original.clone();
            apply_default_whole_image_tesseract_psm(&mut effective);

            let selected = process_whole_image_ocr_with_sparse_retry(
                &backend,
                std::sync::Arc::new(vec![1]),
                &original,
                &effective,
                SparseRetryPolicy {
                    whole_page_raster: true,
                    allow_standard_retry: false,
                },
                None,
            )
            .await
            .expect("fallback evaluation must succeed");

            assert_eq!(selected.content, primary.content);
            assert!(!selected.content.contains("BAD-2026-104"));
        }

        #[tokio::test]
        async fn should_recover_primary_identifier_when_rtl_fallback_is_rejected() {
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};

            struct SequenceBackend(std::sync::Mutex<std::collections::VecDeque<crate::types::ExtractedDocument>>);

            #[async_trait::async_trait]
            impl OcrBackend for SequenceBackend {
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
                    self.0
                        .lock()
                        .unwrap()
                        .pop_front()
                        .ok_or_else(|| crate::XbergError::Other("unexpected OCR call".to_string()))
                }
            }

            impl Plugin for SequenceBackend {
                fn name(&self) -> &str {
                    "primary-recovery-sequence"
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

            let mut identifier = crate::types::OcrElement {
                text: "880-2026-104".to_string(),
                level: crate::types::OcrElementLevel::Word,
                geometry: crate::types::OcrBoundingGeometry::Rectangle {
                    left: 10,
                    top: 10,
                    width: 40,
                    height: 10,
                },
                ..Default::default()
            };
            identifier
                .backend_metadata
                .insert("word_language".to_string(), serde_json::json!("ara"));
            let primary = crate::types::ExtractedDocument {
                content: "تقرير 880-2026-104".to_string(),
                ocr_elements: Some(vec![identifier]),
                ..Default::default()
            };
            let fallback = primary.clone();
            let evidence = crate::types::ExtractedDocument {
                ocr_elements: Some(vec![crate::types::OcrElement {
                    text: "REQ-2026-104".to_string(),
                    confidence: crate::types::OcrConfidence {
                        recognition: 0.9,
                        ..Default::default()
                    },
                    level: crate::types::OcrElementLevel::Word,
                    ..Default::default()
                }]),
                ..Default::default()
            };
            let backend: std::sync::Arc<dyn OcrBackend> = std::sync::Arc::new(SequenceBackend(std::sync::Mutex::new(
                std::collections::VecDeque::from([primary, fallback, evidence]),
            )));
            let original = crate::core::config::OcrConfig {
                language: vec!["ara".to_string(), "eng".to_string()],
                ..Default::default()
            };
            let mut effective = original.clone();
            apply_default_whole_image_tesseract_psm(&mut effective);
            let mut encoded = std::io::Cursor::new(Vec::new());
            image::DynamicImage::new_rgb8(100, 50)
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image must encode");

            let selected = process_whole_image_ocr_with_sparse_retry(
                &backend,
                std::sync::Arc::new(encoded.into_inner()),
                &original,
                &effective,
                SparseRetryPolicy {
                    whole_page_raster: true,
                    allow_standard_retry: false,
                },
                None,
            )
            .await
            .expect("primary recovery must succeed");

            assert_eq!(selected.content, "تقرير REQ-2026-104");
        }

        #[tokio::test]
        async fn should_preserve_selected_ocr_when_optional_identifier_recovery_fails() {
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};

            struct FailingRecoveryBackend;

            #[async_trait::async_trait]
            impl OcrBackend for FailingRecoveryBackend {
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
                    Err(crate::XbergError::Other("recovery failed".to_string()))
                }
            }

            impl Plugin for FailingRecoveryBackend {
                fn name(&self) -> &str {
                    "failing-recovery"
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

            let mut word = crate::types::OcrElement {
                text: "880-2026-104".to_string(),
                level: crate::types::OcrElementLevel::Word,
                geometry: crate::types::OcrBoundingGeometry::Rectangle {
                    left: 1,
                    top: 1,
                    width: 8,
                    height: 8,
                },
                ..Default::default()
            };
            word.backend_metadata
                .insert("word_language".to_string(), serde_json::json!("ara"));
            let mut selected = crate::types::ExtractedDocument {
                content: "880-2026-104".to_string(),
                ocr_elements: Some(vec![word]),
                ..Default::default()
            };
            let mut encoded = std::io::Cursor::new(Vec::new());
            image::DynamicImage::new_rgb8(10, 10)
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image must encode");
            let image = encoded.into_inner();
            let backend: std::sync::Arc<dyn OcrBackend> = std::sync::Arc::new(FailingRecoveryBackend);

            recover_selected_rtl_identifiers_fail_open(
                &backend,
                std::sync::Arc::new(image.clone()),
                &mut selected,
                &crate::core::config::OcrConfig::default(),
                None,
            )
            .await
            .expect("optional recovery failure must remain non-fatal");

            assert_eq!(selected.content, "880-2026-104");

            let cancelled = crate::cancellation::CancellationToken::new();
            cancelled.cancel();
            assert!(matches!(
                recover_selected_rtl_identifiers_fail_open(
                    &backend,
                    std::sync::Arc::new(image.clone()),
                    &mut selected,
                    &crate::core::config::OcrConfig::default(),
                    Some(&cancelled),
                )
                .await,
                Err(crate::XbergError::Cancelled)
            ));

            struct CancellingRecoveryBackend(crate::cancellation::CancellationToken);

            #[async_trait::async_trait]
            impl OcrBackend for CancellingRecoveryBackend {
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
                    self.0.cancel();
                    Ok(crate::types::ExtractedDocument::default())
                }
            }

            impl Plugin for CancellingRecoveryBackend {
                fn name(&self) -> &str {
                    "cancelling-recovery"
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

            let cancellation = crate::cancellation::CancellationToken::new();
            let backend: std::sync::Arc<dyn OcrBackend> =
                std::sync::Arc::new(CancellingRecoveryBackend(cancellation.clone()));
            assert!(matches!(
                recover_selected_rtl_identifiers_fail_open(
                    &backend,
                    std::sync::Arc::new(image),
                    &mut selected,
                    &crate::core::config::OcrConfig::default(),
                    Some(&cancellation),
                )
                .await,
                Err(crate::XbergError::Cancelled)
            ));
        }

        #[tokio::test]
        async fn recover_selected_rtl_identifiers_keeps_cancellation_when_recovery_errors() {
            use crate::plugins::{OcrBackend, OcrBackendType, Plugin};

            struct CancellingFailingRecoveryBackend(crate::cancellation::CancellationToken);

            #[async_trait::async_trait]
            impl OcrBackend for CancellingFailingRecoveryBackend {
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
                    self.0.cancel();
                    Err(crate::XbergError::Other("recovery interrupted".to_string()))
                }
            }

            impl Plugin for CancellingFailingRecoveryBackend {
                fn name(&self) -> &str {
                    "cancelling-failing-recovery"
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

            let mut word = crate::types::OcrElement {
                text: "880-2026-104".to_string(),
                level: crate::types::OcrElementLevel::Word,
                geometry: crate::types::OcrBoundingGeometry::Rectangle {
                    left: 1,
                    top: 1,
                    width: 8,
                    height: 8,
                },
                ..Default::default()
            };
            word.backend_metadata
                .insert("word_language".to_string(), serde_json::json!("ara"));
            let mut selected = crate::types::ExtractedDocument {
                content: "880-2026-104".to_string(),
                ocr_elements: Some(vec![word]),
                ..Default::default()
            };
            let mut encoded = std::io::Cursor::new(Vec::new());
            image::DynamicImage::new_rgb8(10, 10)
                .write_to(&mut encoded, image::ImageFormat::Png)
                .expect("test image must encode");
            let cancellation = crate::cancellation::CancellationToken::new();
            let backend: std::sync::Arc<dyn OcrBackend> =
                std::sync::Arc::new(CancellingFailingRecoveryBackend(cancellation.clone()));

            let outcome = recover_selected_rtl_identifiers_fail_open(
                &backend,
                std::sync::Arc::new(encoded.into_inner()),
                &mut selected,
                &crate::core::config::OcrConfig::default(),
                Some(&cancellation),
            )
            .await;

            assert!(cancellation.is_cancelled(), "the recovery OCR call must run");
            assert!(
                matches!(outcome, Err(crate::XbergError::Cancelled)),
                "a recovery error after cancellation must stay a cancellation, got {outcome:?}"
            );
        }

        #[test]
        fn should_reject_longer_rtl_retry_with_repeated_lines_or_low_confidence() {
            let primary = result_with_content_and_confidence("تقرير عربي REQ-2026-104", 91.0);
            let repeated =
                result_with_content_and_confidence("تقرير عربي REQ-2026-104\nكلام زائد\nكلام زائد\nكلام زائد", 92.0);
            let repeated_phrase = result_with_content_and_confidence(
                "تقرير عربي REQ-2026-104 كلام زائد جدا كلام زائد جدا كلام زائد جدا",
                92.0,
            );
            let low_confidence = result_with_content_and_confidence("تقرير عربي كامل REQ-2026-104", 80.0);

            assert!(!rtl_retry_candidate_improves(&primary, &repeated));
            assert!(!rtl_retry_candidate_improves(&primary, &repeated_phrase));
            assert!(!rtl_retry_candidate_improves(&primary, &low_confidence));
        }

        #[test]
        fn should_reject_rtl_retry_that_replaces_too_many_primary_tokens() {
            let primary =
                result_with_content_and_confidence("واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة", 91.0);
            let candidate =
                result_with_content_and_confidence("واحد اثنان ثلاثة أربعة خمسة ستة سبعة جديد إضافي كامل مفيد", 92.0);

            assert!(!rtl_retry_candidate_improves(&primary, &candidate));
        }

        #[test]
        fn should_bound_rtl_retry_by_unique_and_total_content_tokens() {
            let original = crate::core::config::OcrConfig {
                language: vec!["ara".to_string()],
                ..Default::default()
            };
            let mut effective = original.clone();
            apply_default_whole_image_tesseract_psm(&mut effective);
            for (unique, total, expected) in [(32, 48, true), (33, 33, false), (32, 49, false)] {
                let tokens = (0..total)
                    .map(|index| format!("كلمة{}", index % unique))
                    .collect::<Vec<_>>();
                let primary = crate::types::ExtractedDocument {
                    content: tokens.join(" "),
                    ..Default::default()
                };
                assert_eq!(should_retry_sparse_rtl_ocr(&original, &effective, &primary), expected);
            }
        }

        #[test]
        fn should_retry_only_implicit_strong_rtl_tesseract_with_psm11_primary() {
            let implicit_arabic = crate::core::config::OcrConfig {
                language: vec!["ara".to_string()],
                ..Default::default()
            };
            let explicit_arabic = crate::core::config::OcrConfig {
                language: vec!["ara".to_string()],
                tesseract_config: Some(crate::types::TesseractConfig {
                    psm: Some(11),
                    language: vec!["ara".to_string()],
                    ..Default::default()
                }),
                ..Default::default()
            };
            let implicit_english = crate::core::config::OcrConfig::default();

            assert!(is_implicit_strong_rtl_tesseract(&implicit_arabic));
            assert!(!is_implicit_strong_rtl_tesseract(&explicit_arabic));
            assert!(!is_implicit_strong_rtl_tesseract(&implicit_english));
        }

        #[cfg(feature = "pdf")]
        #[tokio::test]
        #[ignore = "requires local Tesseract Arabic traineddata"]
        async fn should_preserve_recovered_identifiers_in_scanned_pdf_content() {
            let pdf = crate::utils::read_test_fixture("pdf_scanned/rtl_arabic_scanned.pdf")
                .expect("required scanned Arabic PDF fixture is missing");
            let languages = vec!["ara".to_string(), "eng".to_string()];
            let config = crate::core::config::ExtractionConfig {
                force_ocr: true,
                use_cache: false,
                ocr: Some(crate::core::config::OcrConfig {
                    language: languages.clone(),
                    tesseract_config: Some(crate::types::TesseractConfig {
                        language: languages,
                        use_cache: false,
                        ..Default::default()
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            };
            let mut primary_config = config.clone();
            primary_config
                .ocr
                .as_mut()
                .and_then(|ocr| ocr.tesseract_config.as_mut())
                .expect("primary config must contain Tesseract settings")
                .psm = Some(WHOLE_IMAGE_TESSERACT_PSM);
            let primary = crate::extract(
                crate::core::config::ExtractInput::from_bytes(pdf.clone(), "application/pdf", None),
                &primary_config,
            )
            .await
            .expect("explicit primary PDF extraction must succeed");
            let primary_content = &primary
                .results
                .first()
                .expect("one primary PDF result is required")
                .content;
            let input = crate::core::config::ExtractInput::from_bytes(pdf, "application/pdf", None);
            let result = crate::extract(input, &config)
                .await
                .expect("scanned PDF extraction must succeed");
            let content = &result.results.first().expect("one PDF result is required").content;
            let ground_truth = String::from_utf8(
                crate::utils::read_test_fixture("ground_truth/images/rtl_arabic_300dpi.txt")
                    .expect("required Arabic ground truth is missing"),
            )
            .expect("Arabic ground truth must be UTF-8");
            let actual = normalized_ocr_tokens(content);
            let actual_unique = actual.iter().collect::<std::collections::HashSet<_>>();
            let primary_unique = normalized_ocr_tokens(primary_content)
                .into_iter()
                .collect::<std::collections::HashSet<_>>();
            let correct = normalized_ocr_tokens(&ground_truth)
                .iter()
                .filter(|token| actual_unique.contains(token))
                .count();
            let primary_correct = normalized_ocr_tokens(&ground_truth)
                .iter()
                .filter(|token| primary_unique.contains(*token))
                .count();
            assert!(actual.len() >= 41, "scanned PDF must preserve output coverage");
            assert!(
                correct >= primary_correct,
                "scanned PDF correct-token count must not regress: {primary_correct} -> {correct}: {content}"
            );

            for identifier in ["REQ-2026-104", "OCR-v2.1", "A-17", "B-09"] {
                assert!(content.contains(identifier), "missing {identifier}: {content}");
            }
        }

        /// Regression test for the standalone-image-OCR variant of the `TesseractConfig`
        /// duplication hazard: `apply_default_whole_image_tesseract_psm` materializes
        /// `crate::types::TesseractConfig::default()` (the public struct) whenever the
        /// caller never set `tesseract_config` explicitly, which is the default path for
        /// every plain image (PNG/JPG/TIFF/...) OCR request. That materialized value is
        /// later converted into `crate::ocr::types::TesseractConfig` (the internal,
        /// engine-facing struct) via `From`, which carries the value across as-is — it does
        /// NOT fall back to the internal struct's own `Default`. So if the public struct's
        /// default ever drifts from the internal struct's default, standalone image OCR
        /// silently keeps using the stale public value while other callers that reach the
        /// internal `Default` directly (e.g. PDF-embedded OCR, which never calls this
        /// function) get the current one.
        ///
        /// Against the unfixed code, `crate::types::TesseractConfig::default()` still has
        /// `language_model_ngram_on: false`, so this assertion fails with `false` even
        /// though `crate::ocr::types::TesseractConfig::default()` already has `true`.
        #[test]
        fn should_apply_public_ngram_default_to_whole_image_tesseract_config() {
            let mut whole_image_config = crate::core::config::OcrConfig::default();
            apply_default_whole_image_tesseract_psm(&mut whole_image_config);

            let tesseract_config = whole_image_config
                .tesseract_config
                .expect("default whole-image OCR must materialize a Tesseract configuration");

            assert!(
                tesseract_config.language_model_ngram_on,
                "standalone image OCR must use the same language_model_ngram_on default as the \
                 internal TesseractConfig, not a stale value baked in from the public struct's \
                 own Default impl"
            );
        }
    }

    fn image_ocr_document(text: &str) -> InternalDocument {
        build_image_internal_document(Some(text), None)
    }

    #[test]
    fn should_mark_metadata_when_standalone_image_ocr_succeeds() {
        let mut doc = image_ocr_document("recognized text");

        ImageExtractor::mark_ocr_extraction(&mut doc);
        let result =
            crate::extraction::derive::derive_extraction_result(doc, false, crate::core::config::OutputFormat::Plain);

        assert!(result.metadata.ocr_used);
        assert_eq!(result.extraction_method, Some(crate::types::ExtractionMethod::Ocr));
    }

    #[tokio::test]
    async fn should_not_mark_metadata_when_standalone_image_ocr_is_disabled() {
        let mut png = std::io::Cursor::new(Vec::new());
        image::ImageBuffer::<image::Rgb<u8>, _>::from_pixel(1, 1, image::Rgb([255u8, 255, 255]))
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("failed to encode test PNG");
        let config = ExtractionConfig {
            disable_ocr: true,
            ..Default::default()
        };

        let doc = ImageExtractor::new()
            .extract_content(&png.into_inner(), "image/png", &config)
            .await
            .expect("metadata-only image extraction must succeed");
        let result =
            crate::extraction::derive::derive_extraction_result(doc, false, crate::core::config::OutputFormat::Plain);

        assert!(!result.metadata.ocr_used);
        assert_eq!(result.extraction_method, None);
    }

    #[test]
    fn image_ocr_preserves_blank_line_paragraph_boundaries() {
        let doc = image_ocr_document("\nfirst line\nwrapped line\n\n\nsecond paragraph\n");
        let paragraphs = doc
            .elements
            .iter()
            .filter(|element| matches!(element.kind, crate::types::internal::ElementKind::Paragraph))
            .map(|element| element.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(paragraphs, ["first line\nwrapped line", "second paragraph"]);
    }

    #[test]
    fn image_ocr_keeps_single_newline_wrapping_in_one_paragraph() {
        let doc = image_ocr_document("first line\nwrapped line");
        let paragraphs = doc
            .elements
            .iter()
            .filter(|element| matches!(element.kind, crate::types::internal::ElementKind::Paragraph))
            .map(|element| element.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(paragraphs, ["first line\nwrapped line"]);
    }

    #[test]
    fn should_use_whole_image_ocr_when_invoice_regions_drop_fields() {
        let layout = image_ocr_document("Invoice 1042 Acme Total 1250 USD");
        let mut whole = image_ocr_document("Invoice 1042 Acme Corporation Date July 31 Total 1250 USD");
        whole.metadata.additional.insert(
            std::borrow::Cow::Borrowed("ocr_candidate"),
            serde_json::Value::String("whole-image".to_string()),
        );
        whole.processing_warnings.push(crate::types::ProcessingWarning {
            source: std::borrow::Cow::Borrowed("ocr"),
            message: std::borrow::Cow::Borrowed("whole-image warning"),
        });

        let selected = select_image_ocr_result(layout, Ok(whole));

        assert_eq!(
            internal_document_text(&selected),
            "Invoice 1042 Acme Corporation Date July 31 Total 1250 USD"
        );
        assert_eq!(
            selected.metadata.additional.get("ocr_candidate"),
            Some(&serde_json::Value::String("whole-image".to_string()))
        );
        assert_eq!(selected.processing_warnings.len(), 1);
        assert_eq!(selected.processing_warnings[0].message, "whole-image warning");
    }

    #[test]
    fn should_keep_layout_ocr_for_complete_simple_line() {
        let mut layout = image_ocr_document("The quick brown fox jumps over the lazy dog");
        layout.metadata.additional.insert(
            std::borrow::Cow::Borrowed("ocr_candidate"),
            serde_json::Value::String("layout".to_string()),
        );
        let whole = image_ocr_document("The quick brown fox jumps over the lazy dog");

        let selected = select_image_ocr_result(layout, Ok(whole));

        assert_eq!(
            selected.metadata.additional.get("ocr_candidate"),
            Some(&serde_json::Value::String("layout".to_string()))
        );
    }

    #[cfg(feature = "quality")]
    #[test]
    fn should_use_whole_image_ocr_when_complex_layout_text_scores_lower() {
        let layout =
            image_ocr_document("Quarterly   revenue   increased   while   operating   expenses   remained   stable.");
        let whole = image_ocr_document("Quarterly revenue increased while operating expenses remained stable.");

        assert_eq!(
            alphanumeric_token_retention(&internal_document_text(&layout), &internal_document_text(&whole)),
            1.0
        );
        let selected = select_image_ocr_result(layout, Ok(whole));

        assert_eq!(
            internal_document_text(&selected),
            "Quarterly revenue increased while operating expenses remained stable."
        );
    }

    #[test]
    fn should_count_duplicate_alphanumeric_tokens_individually() {
        let retention = alphanumeric_token_retention("invoice total", "invoice invoice total");

        assert!((retention - (2.0 / 3.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn should_accept_exact_minimum_alphanumeric_token_retention() {
        let retention = alphanumeric_token_retention("one two three four", "one two three four five");

        assert!((retention - MIN_LAYOUT_OCR_ALPHANUMERIC_TOKEN_RETENTION).abs() < f64::EPSILON);
    }

    #[test]
    fn should_keep_layout_ocr_when_whole_image_comparison_fails() {
        let mut layout = image_ocr_document("Invoice 1042 Total 1250 USD");
        layout.metadata.additional.insert(
            std::borrow::Cow::Borrowed("ocr_candidate"),
            serde_json::Value::String("layout".to_string()),
        );
        layout.processing_warnings.push(crate::types::ProcessingWarning {
            source: std::borrow::Cow::Borrowed("layout-ocr"),
            message: std::borrow::Cow::Borrowed("layout warning"),
        });

        let selected = select_image_ocr_result(
            layout,
            Err(crate::XbergError::Other("comparison OCR failed".to_string())),
        );

        assert_eq!(
            selected.metadata.additional.get("ocr_candidate"),
            Some(&serde_json::Value::String("layout".to_string()))
        );
        assert_eq!(selected.processing_warnings.len(), 1);
        assert_eq!(selected.processing_warnings[0].message, "layout warning");
    }

    #[tokio::test]
    async fn test_image_extractor_invalid_image() {
        let extractor = ImageExtractor::new();
        let invalid_bytes = vec![0, 1, 2, 3, 4, 5];
        let config = ExtractionConfig::default();

        let result = extractor.extract_content(&invalid_bytes, "image/png", &config).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_image_plugin_interface() {
        let extractor = ImageExtractor::new();
        assert_eq!(extractor.name(), "image-extractor");
        assert_eq!(extractor.version(), env!("CARGO_PKG_VERSION"));
        assert!(extractor.supported_mime_types().contains(&"image/png"));
        assert!(extractor.supported_mime_types().contains(&"image/jpeg"));
        assert!(extractor.supported_mime_types().contains(&"image/webp"));
        assert_eq!(extractor.priority(), 50);
    }

    #[test]
    fn test_image_extractor_default() {
        let extractor = ImageExtractor;
        assert_eq!(extractor.name(), "image-extractor");
    }

    #[test]
    fn test_image_extractor_supports_alias_mime_types() {
        let extractor = ImageExtractor::new();
        let supported = extractor.supported_mime_types();
        assert!(supported.contains(&"image/pjpeg"));
        assert!(supported.contains(&"image/x-bmp"));
        assert!(supported.contains(&"image/x-ms-bmp"));
        assert!(supported.contains(&"image/x-tiff"));
        assert!(supported.contains(&"image/x-portable-anymap"));
    }

    /// REV-C6 regression for the #1662 review: OCR alone must NOT put a standalone
    /// image's own bytes in the result. `needs_image_data` gained an OCR disjunct so
    /// containers would read embedded images for OCR (#1662); `wants_own_bytes_in_result`
    /// is the narrower predicate `extractors/image.rs` uses instead, so a standalone
    /// image's public `images` output stays exactly as it was before #1662 unless the
    /// caller also asked for image extraction, captioning, or QR codes. `doc.images` must
    /// stay empty here, in contrast to
    /// `test_extract_with_ocr_populates_images_for_captioning` immediately above, whose
    /// only difference is a `captioning` config.
    #[tokio::test]
    async fn test_extract_with_ocr_only_does_not_populate_images() {
        use crate::core::config::OcrConfig;
        use crate::plugins::{OcrBackend, OcrBackendType, Plugin, register_ocr_backend, unregister_ocr_backend};
        use crate::types::ExtractedDocument;

        let mut png_buf = std::io::Cursor::new(Vec::new());
        image::ImageBuffer::<image::Rgb<u8>, _>::from_pixel(1, 1, image::Rgb([255u8, 255, 255]))
            .write_to(&mut png_buf, image::ImageFormat::Png)
            .expect("failed to encode test PNG");
        let png_1x1 = png_buf.into_inner();

        struct EmptyOcrBackend1662;

        #[async_trait::async_trait]
        impl OcrBackend for EmptyOcrBackend1662 {
            fn backend_type(&self) -> OcrBackendType {
                OcrBackendType::Custom
            }
            fn supports_language(&self, _: &str) -> bool {
                true
            }
            async fn process_image(&self, _: &[u8], _config: &OcrConfig) -> crate::Result<ExtractedDocument> {
                Ok(ExtractedDocument {
                    content: String::new(),
                    ..Default::default()
                })
            }
        }

        impl Plugin for EmptyOcrBackend1662 {
            fn name(&self) -> &str {
                "empty-ocr-1662"
            }
            fn version(&self) -> String {
                "0.0.0".to_string()
            }
            fn initialize(&self) -> crate::Result<()> {
                Ok(())
            }
            fn shutdown(&self) -> crate::Result<()> {
                Ok(())
            }
        }

        register_ocr_backend(std::sync::Arc::new(EmptyOcrBackend1662)).unwrap();
        struct BackendGuard(&'static str);
        impl Drop for BackendGuard {
            fn drop(&mut self) {
                let _ = unregister_ocr_backend(self.0);
            }
        }
        let _guard = BackendGuard("empty-ocr-1662");

        let config = ExtractionConfig {
            ocr: Some(OcrConfig {
                backend: "empty-ocr-1662".to_string(),
                ..Default::default()
            }),
            ..Default::default()
        };

        let extractor = ImageExtractor::new();
        let doc = extractor.extract_content(&png_1x1, "image/png", &config).await.unwrap();

        assert!(
            doc.images.is_empty(),
            "OCR alone must not echo a standalone image's bytes into the public result: {:?}",
            doc.images
        );
    }

    // -- enforce_image_page_limit stub (not(feature = "ocr")) -------------------
    //
    // The `ocr`-enabled sibling counts real TIFF frames and is exercised
    // elsewhere; these tests cover the no-op stub compiled when the `tiff`
    // crate isn't available, which instead emits a one-shot warning so a
    // caller-set `max_pages` going unenforced is discoverable (#1451-adjacent).
    //
    // `warn_unenforced_tiff_page_limit_once` takes its "already warned" guard
    // as a parameter rather than reading `UNENFORCED_TIFF_PAGE_LIMIT_WARNED`
    // directly, mirroring `warn_default_thread_cap_once` in
    // `core::config::concurrency`: a shared process-global guard can only be
    // asserted from a single test without the assertion racing every other
    // test in the binary that reaches the same code path. Three of the four
    // tests below use a guard they own, so they are independent of each other
    // and of test execution order; only the "fires" test below exercises the
    // real global-backed entry point (`enforce_image_page_limit`), and it is
    // the only test in this binary that does so.

    mod enforce_image_page_limit_stub {
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, Mutex};

        use tracing_subscriber::layer::SubscriberExt as _;
        use tracing_subscriber::{EnvFilter, Layer};

        use super::super::{ExtractionConfig, enforce_image_page_limit, enforce_image_page_limit_stub_with_guard};
        use crate::extractors::security::SecurityLimits;

        /// A tracing `Layer` that records the level of every emitted event.
        #[derive(Clone, Default)]
        struct EventCapture {
            levels: Arc<Mutex<Vec<tracing::Level>>>,
        }

        impl<S> Layer<S> for EventCapture
        where
            S: tracing::Subscriber,
        {
            fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
                self.levels.lock().unwrap().push(*event.metadata().level());
            }
        }

        fn warn_event_count(capture: &EventCapture) -> usize {
            capture
                .levels
                .lock()
                .unwrap()
                .iter()
                .filter(|level| **level == tracing::Level::WARN)
                .count()
        }

        fn config_with_max_pages(max_pages: usize) -> ExtractionConfig {
            ExtractionConfig {
                security_limits: Some(SecurityLimits {
                    max_pages: Some(max_pages),
                    ..Default::default()
                }),
                ..Default::default()
            }
        }

        /// Changing this to `_content.is_empty()` (as opposed to inspecting
        /// `mime_type`) would make this test pass for the wrong reason; the
        /// content bytes are irrelevant to the stub, which is the point.
        const TIFF_MIME: &str = "image/tiff";

        /// The real, process-global-backed entry point warns exactly once
        /// across repeated calls when `max_pages` is set on a TIFF. Bundled
        /// into a single test (rather than a separate "fires" and "fires
        /// once" test) because both assertions read the same global guard;
        /// splitting them would make the pass/fail outcome depend on which
        /// test the harness happens to run first.
        ///
        /// Fails if: the `if !mime_type...contains("tiff")` check in
        /// `enforce_image_page_limit_stub_with_guard` is removed or inverted;
        /// the `let Some(max_pages) = max_pages else { return Ok(()) };` guard
        /// is removed; `warn_unenforced_tiff_page_limit_once` is changed to
        /// unconditionally warn (no longer once); or the call to
        /// `warn_unenforced_tiff_page_limit_once` is deleted altogether.
        #[test]
        fn fires_exactly_once_for_tiff_with_max_pages_set_across_repeated_calls() {
            let capture = EventCapture::default();
            let subscriber = tracing_subscriber::registry()
                .with(EnvFilter::new("warn"))
                .with(capture.clone());
            let config = config_with_max_pages(3);

            tracing::subscriber::with_default(subscriber, || {
                for _ in 0..5 {
                    let result = enforce_image_page_limit(b"unused", TIFF_MIME, &config);
                    assert!(result.is_ok(), "the stub warns but never rejects extraction");
                }
            });

            assert_eq!(
                warn_event_count(&capture),
                1,
                "expected exactly one WARN across repeated calls, got {:?}",
                capture.levels.lock().unwrap()
            );
        }

        /// Fails if: the `let Some(max_pages) = max_pages else { return
        /// Ok(()) };` early return in `enforce_image_page_limit_stub_with_guard`
        /// is removed (or its condition inverted), causing an unset
        /// `max_pages` to warn anyway.
        #[test]
        fn does_not_fire_when_max_pages_is_unset() {
            let already_warned = AtomicBool::new(false);
            let capture = EventCapture::default();
            let subscriber = tracing_subscriber::registry()
                .with(EnvFilter::new("warn"))
                .with(capture.clone());
            let config = ExtractionConfig::default();
            assert!(config.security_limits.is_none());

            let result = tracing::subscriber::with_default(subscriber, || {
                enforce_image_page_limit_stub_with_guard(TIFF_MIME, &config, &already_warned)
            });

            assert!(result.is_ok());
            assert_eq!(warn_event_count(&capture), 0);
        }

        /// Fails if: the `if !mime_type.to_lowercase().contains("tiff") {
        /// return Ok(()); }` guard in `enforce_image_page_limit_stub_with_guard`
        /// is removed (or its condition inverted), causing a non-TIFF mime type
        /// to warn.
        #[test]
        fn does_not_fire_for_non_tiff_mime() {
            let already_warned = AtomicBool::new(false);
            let capture = EventCapture::default();
            let subscriber = tracing_subscriber::registry()
                .with(EnvFilter::new("warn"))
                .with(capture.clone());
            let config = config_with_max_pages(3);

            let result = tracing::subscriber::with_default(subscriber, || {
                enforce_image_page_limit_stub_with_guard("image/png", &config, &already_warned)
            });

            assert!(result.is_ok());
            assert_eq!(warn_event_count(&capture), 0);
        }

        /// A second, independently-guarded call for a TIFF with `max_pages`
        /// set still succeeds and still warns — proving the "does not fire"
        /// tests above are silent because their inputs don't qualify, not
        /// because the warning path is broken. Fails under the same mutations
        /// as `fires_exactly_once_for_tiff_with_max_pages_set_across_repeated_calls`.
        #[test]
        fn fires_when_max_pages_is_set_and_mime_is_tiff_with_a_fresh_guard() {
            let already_warned = AtomicBool::new(false);
            let capture = EventCapture::default();
            let subscriber = tracing_subscriber::registry()
                .with(EnvFilter::new("warn"))
                .with(capture.clone());
            let config = config_with_max_pages(3);

            let result = tracing::subscriber::with_default(subscriber, || {
                enforce_image_page_limit_stub_with_guard(TIFF_MIME, &config, &already_warned)
            });

            assert!(result.is_ok());
            assert_eq!(warn_event_count(&capture), 1);
        }
    }
}
