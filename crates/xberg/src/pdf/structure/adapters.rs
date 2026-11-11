//! OCR-to-structure adapters: convert xberg internal types into the PDF
//! structure pipeline's paragraph representation.
// `types` is used by the OCR conversion helpers (`feature = "ocr"`) and by the
// unused when only `ocr-pipeline` is on without `layout-detection`, as in the
// WASM `ocr-wasm` feature set.
#[cfg(feature = "ocr-pipeline")]
use super::types;

/// Separate points-per-pixel scale factors for [`resolve_ocr_font_size_pt`]'s two
/// geometric fallback branches.
///
/// `element.bbox` and `element.ocr_geometry` can be in different units by the time
/// font-size resolution runs, depending on the caller's route:
///
/// - On the mixed OCR route (`extractors::pdf::ocr::assemble_mixed_ocr_page_document`),
///   `element.bbox` has already been rescaled into PDF points by that route's own
///   `rescale_ocr_bboxes_to_page_points` before this runs, so the `block_bbox`-height
///   fallback needs no further scaling. `element.ocr_geometry` is untouched by that
///   rescale -- it stays raster-pixel space, because
///   `extraction::derive::OcrElement::geometry` documents that field as public,
///   raster-pixel-space API and rescaling it in place would both corrupt that contract
///   and, since its point type is `(u32, u32)`, round away sub-pixel precision. The
///   quad-edge fallback (sceptre/paddle's `Quadrilateral` geometry) therefore still
///   needs the real points-per-pixel ratio.
/// - On the pure-OCR route (`extract_with_ocr_for_page`), both fields are still in raw
///   OCR raster pixels when font-size resolution runs, so both branches need the same
///   real points-per-pixel ratio.
///
/// A single scalar cannot express this: collapsing to one number either corrupts the
/// mixed route's bbox branch (multiplying an already-in-points value again) or leaves
/// its geometry branch unconverted -- inflating sceptre/paddle font sizes by the raster
/// DPI's points-per-pixel ratio (~2.08x at 150 DPI) and pushing ordinary intra-line
/// ascender/descender variance over the heading heuristic's `MIN_HEADING_FONT_GAP`,
/// breaking paragraphs mid physical line.
#[cfg(feature = "ocr-pipeline")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct OcrFontSizeScale {
    bbox_points_per_pixel: f32,
    geometry_points_per_pixel: f32,
}

/// Scale to apply to a value that is already in the target unit (PDF points): a
/// genuine no-op, not a magic number picked for effect.
#[cfg(feature = "ocr-pipeline")]
const NO_OP_POINTS_PER_PIXEL: f32 = 1.0;

#[cfg(feature = "ocr-pipeline")]
impl OcrFontSizeScale {
    /// `element.bbox` and `element.ocr_geometry` are both still in the same unit
    /// space (raw OCR raster pixels on the pure-OCR route, or already-uniform test
    /// fixtures): apply one real points-per-pixel ratio to both branches.
    // Every caller sits behind an OCR-document assembly path that `layout-detection` without
    // `ocr`/`ocr-wasm` compiles out, while `ocr-pipeline` still brings this type in. ~keep

    pub(crate) fn uniform(points_per_pixel: f32) -> Self {
        Self {
            bbox_points_per_pixel: points_per_pixel,
            geometry_points_per_pixel: points_per_pixel,
        }
    }

    /// The mixed OCR route's shape: `element.bbox` was already rescaled into PDF
    /// points by the caller, so the bbox-height fallback is a no-op scale, while
    /// `element.ocr_geometry` is still raw raster pixels and needs the real
    /// points-per-pixel ratio for this page.
    pub(crate) fn bbox_already_in_points(geometry_points_per_pixel: f32) -> Self {
        Self {
            bbox_points_per_pixel: NO_OP_POINTS_PER_PIXEL,
            geometry_points_per_pixel,
        }
    }
}

/// Convert an OCR-produced [`crate::types::internal::InternalDocument`] into a vec of [`types::PdfParagraph`]s
/// for the structure assembly pipeline.
///
/// Coordinates are in image-space (y=0 at top) and are flipped to PDF-space
/// (y=0 at bottom) using `page_height_px`.
#[cfg(feature = "ocr-pipeline")]

pub(crate) fn ocr_doc_to_paragraphs(
    doc: &crate::types::internal::InternalDocument,
    page_height_px: u32,
    font_size_scale: OcrFontSizeScale,
) -> Vec<types::PdfParagraph> {
    use crate::types::internal::ElementKind;
    let page_h = page_height_px as f32;
    let mut result = Vec::new();
    let mut previous_block_id = None;
    let block_font_sizes = ocr_block_median_font_sizes(doc, page_h, font_size_scale);

    for element in &doc.elements {
        if !matches!(element.kind, ElementKind::OcrText { .. }) || element.text.trim().is_empty() {
            previous_block_id = None;
            continue;
        }
        let block_id = hocr_block_id(element);
        let block_median_pt = block_id.and_then(|block_id| block_font_sizes.get(block_id).copied());
        // A block that split into several list-item paragraphs (see
        // `make_ocr_block_paragraphs`'s doc comment, #713) only offers its *first*
        // segment for the same-block-id merge below: that segment carries the block's
        // leading edge and is what a continuing block would have merged into before
        // this split existed. Later segments are already-separated list items and must
        // not be re-merged into the block that precedes them.
        for (segment_index, paragraph) in make_ocr_block_paragraphs(element, page_h, font_size_scale, block_median_pt)
            .into_iter()
            .enumerate()
        {
            if segment_index == 0 && block_id.is_some() && block_id == previous_block_id {
                if let Some(current) = result.last_mut() {
                    merge_ocr_block_paragraph(current, paragraph);
                } else {
                    result.push(paragraph);
                }
            } else {
                result.push(paragraph);
            }
        }
        previous_block_id = block_id;
    }

    trace_conversion("ocr_doc_to_paragraphs", doc, &result);
    result
}

/// hOCR `x_fsize` attribute key, mirroring `ocr::hocr_parser::HOCR_FONT_SIZE_ATTRIBUTE`
/// (not imported directly to keep this module's feature-gate surface self-contained).
#[cfg(feature = "ocr-pipeline")]
const HOCR_FONT_SIZE_ATTRIBUTE: &str = "x_fsize";

/// hOCR bold/italic fraction attribute keys, mirroring
/// `ocr::hocr_parser::HOCR_BOLD_FRACTION_ATTRIBUTE` / `HOCR_ITALIC_FRACTION_ATTRIBUTE`
/// (not imported directly, for the same self-containment reason as
/// [`HOCR_FONT_SIZE_ATTRIBUTE`]). Only `ocr::hocr_parser::parse_hocr_to_internal_document`
/// (the Tesseract hOCR block-parsing path) ever writes these; sceptre and paddle elements
/// never carry them, so [`resolve_ocr_style_flags`] falls back to `(false, false)` for both.
///
/// Each value is the fraction (0.0-1.0) of words in the hOCR block that Tesseract
/// reported as bold/italic, not a boolean -- a block is a mix of words, not a single
/// style.
#[cfg(feature = "ocr-pipeline")]
const HOCR_BOLD_FRACTION_ATTRIBUTE: &str = "x_bold_fraction";
#[cfg(feature = "ocr-pipeline")]
const HOCR_ITALIC_FRACTION_ATTRIBUTE: &str = "x_italic_fraction";

/// A block counts as bold/italic once more than half its words carry that style,
/// mirroring the majority-vote convention already used for native-PDF segments
/// (`pdf::structure::pipeline`'s `is_bold = ... .count() > segments.len() / 2`).
#[cfg(feature = "ocr-pipeline")]
const STYLE_MAJORITY_FRACTION: f32 = 0.5;

/// Resolve block-level bold/italic flags for an OCR-produced text element from its
/// hOCR fraction attributes, defaulting to `(false, false)` when the attribute is
/// absent (sceptre, paddle) or unparseable (defensive: never panics on a malformed
/// value).
#[cfg(feature = "ocr-pipeline")]
fn resolve_ocr_style_flags(element: &crate::types::internal::InternalElement) -> (bool, bool) {
    let attrs = element.attributes.as_ref();
    let is_majority = |attribute: &str| {
        attrs
            .and_then(|attrs| attrs.get(attribute))
            .and_then(|value| value.parse::<f32>().ok())
            .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
            .is_some_and(|fraction| fraction > STYLE_MAJORITY_FRACTION)
    };
    (
        is_majority(HOCR_BOLD_FRACTION_ATTRIBUTE),
        is_majority(HOCR_ITALIC_FRACTION_ATTRIBUTE),
    )
}

/// Fallback font size (points) when neither an `x_fsize` attribute nor a usable
/// bbox height is available. Matches the constant every call site hardcoded before.
#[cfg(feature = "ocr-pipeline")]
const DEFAULT_OCR_FONT_SIZE_PT: f32 = 12.0;

/// A/B switch for the block-median font-size resolution described on
/// [`ocr_block_median_font_sizes`] (#712).
///
/// **Off.** Measured a net regression over GT-scored full-text F1 across 3
/// backends x 7 scanned fixtures: better on 2 files, worse on 10, tied on 2.
/// The mechanism is over-fusion -- taking a block-wide median collapses
/// legitimately distinct fragments into one font size, and the max block size
/// grows in every affected file (ordinance/paddle 327 words vs 70;
/// nougat_007/sceptre 549 vs 111), which drags word-gluing artefacts up with
/// it. It does deliver one real win -- fabricated mid-sentence headings
/// (`### storage.`, `### groundwork for future developments. Over time,`) went
/// 14 -> 0 on multi_page/paddle and 9 -> 0 on nougat_007/sceptre -- but that
/// win is now covered directly at the heading gate by
/// `SUPPRESS_LOWERCASE_START_HEADINGS` (#712), without paying for it in
/// full-text fusion. For this to come back, the median would need to be scoped
/// tighter than "whole hOCR block" -- e.g. per physical line rather than per
/// block -- so it stops fusing fragments that legitimately belong to different
/// lines.
///
/// **This is the single edit that neutralises the whole feature**: flip it to
/// `true` and [`ocr_block_median_font_sizes`] resumes producing entries, so
/// every element goes back through the block-median path -- no other line of
/// this module needs to change, and no call site does either.
#[cfg(feature = "ocr-pipeline")]
const RESOLVE_OCR_FONT_SIZE_PER_BLOCK: bool = false;

/// A block needs at least this many measurable fragments before a median is worth
/// taking. At one sample the median is the sample, so the entry would be an exact
/// identity; requiring two keeps "nothing to stabilise" structurally distinct from
/// "stabilised" instead of relying on that identity holding.
#[cfg(feature = "ocr-pipeline")]
const MIN_BLOCK_FRAGMENTS_FOR_MEDIAN: usize = 2;

/// The hOCR `x_fsize` font size (points) an element carries, if any.
///
/// Only `ocr::hocr_parser` (the Tesseract path) ever writes this attribute, and it
/// writes one value per `ocr_par`, so every fragment of a Tesseract block already
/// reports the *same* number. That is exactly the within-block stability the median
/// below manufactures for the geometric backends.
#[cfg(feature = "ocr-pipeline")]
fn hocr_font_size_pt(element: &crate::types::internal::InternalElement) -> Option<f32> {
    element
        .attributes
        .as_ref()
        .and_then(|attrs| attrs.get(HOCR_FONT_SIZE_ATTRIBUTE))
        .and_then(|value| value.parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

/// The geometric font-size proxy (PDF points) for a single OCR fragment, or `None`
/// when the element carries neither usable quadrilateral geometry nor a usable bbox.
///
/// Split out of [`resolve_ocr_font_size_pt`] unchanged so that
/// [`ocr_block_median_font_sizes`] can sample exactly the same quantity the
/// per-fragment path would have produced -- the median is taken over resolved point
/// sizes, not over raw pixel heights, so the two are directly comparable and the
/// `None` case is excluded from the sample set rather than entering it as
/// [`DEFAULT_OCR_FONT_SIZE_PT`].
#[cfg(feature = "ocr-pipeline")]
fn geometric_ocr_font_size_pt(
    element: &crate::types::internal::InternalElement,
    block_bbox: Option<(f32, f32, f32, f32)>,
    line_count: usize,
    font_size_scale: OcrFontSizeScale,
) -> Option<f32> {
    element
        .ocr_geometry
        .as_ref()
        .and_then(quad_edge_height_px)
        .map(|height_px| height_px * font_size_scale.geometry_points_per_pixel)
        .or_else(|| block_bbox.map(|(_, bottom, _, top)| (top - bottom) * font_size_scale.bbox_points_per_pixel))
        .map(|height| height / line_count.max(1) as f32)
        .filter(|value| value.is_finite() && *value > 0.0)
}

/// One font size (PDF points) per hOCR block id, taken as the **median** over every
/// measurable fragment of that block (#712).
///
/// `SegmentData::font_size` carries two incompatible physical quantities depending
/// on backend. Tesseract's `x_fsize` is a typographic point size and is a per-
/// `ocr_par` constant. Sceptre and PaddleOCR write no `x_fsize`, so their font size
/// is the *height of a detection box* -- a geometric quantity that varies with the
/// glyph mix (ascenders, descenders, punctuation) *within one physical line*.
/// Measured on the sceptre regression fixture, six fragments of a single body line
/// resolved to 15.4-17.3pt: a 1.12 ratio inside one line, against
/// `MIN_HEADING_FONT_RATIO = 1.15`, and a 1.9pt spread against the 1.5pt absolute
/// `font_change` paragraph break -- so paragraphs split mid-line, between two words
/// of the same sentence, and the heading heuristic had ~80% of its headroom eaten
/// before any real size difference was considered.
///
/// **Median, not mean.** The samples are detection-box heights, and OCR detection
/// routinely emits a badly fragmented box: a single stray character, a piece of a
/// rule mis-detected as text, a box that swallowed part of the line above. Those are
/// outliers of arbitrary magnitude in *both* directions, and a mean moves with them
/// proportionally to how extreme they are -- one 4x-too-tall fragment in a five-
/// fragment block drags the mean 60% above the true line height. The median ignores
/// magnitude entirely and only asks how many fragments fall on each side, so it
/// still reports the type size the majority of the block actually has.
///
/// Blocks whose fragments carry `x_fsize` are skipped entirely (never sampled, never
/// keyed), so the Tesseract path cannot be perturbed even in principle -- see
/// [`resolve_ocr_font_size_pt`], which returns on `x_fsize` before it ever consults
/// this map. Elements with no `hocr_block_id` (and blocks under
/// [`MIN_BLOCK_FRAGMENTS_FOR_MEDIAN`] measurable fragments) produce no entry, and
/// their callers fall back to the unchanged per-fragment resolution.
#[cfg(feature = "ocr-pipeline")]
fn ocr_block_median_font_sizes(
    doc: &crate::types::internal::InternalDocument,
    page_height: f32,
    font_size_scale: OcrFontSizeScale,
) -> ahash::AHashMap<String, f32> {
    use crate::types::internal::ElementKind;

    if !RESOLVE_OCR_FONT_SIZE_PER_BLOCK {
        return ahash::AHashMap::new();
    }

    let mut samples: ahash::AHashMap<String, Vec<f32>> = ahash::AHashMap::new();
    for element in &doc.elements {
        if !matches!(element.kind, ElementKind::OcrText { .. }) || element.text.trim().is_empty() {
            continue;
        }
        if hocr_font_size_pt(element).is_some() {
            continue;
        }
        let Some(block_id) = hocr_block_id(element) else {
            continue;
        };
        let line_count = element.text.split('\n').count().max(1);
        let block_bbox = pdf_block_bbox(element, page_height);
        let Some(font_size) = geometric_ocr_font_size_pt(element, block_bbox, line_count, font_size_scale) else {
            continue;
        };
        samples.entry(block_id.to_owned()).or_default().push(font_size);
    }

    samples
        .into_iter()
        .filter(|(_, values)| values.len() >= MIN_BLOCK_FRAGMENTS_FOR_MEDIAN)
        .filter_map(|(block_id, mut values)| median_font_size_pt(&mut values).map(|median| (block_id, median)))
        .collect()
}

/// Median of a non-empty slice of font sizes, averaging the two central values for
/// an even count. Sorted with `f32::total_cmp` rather than `partial_cmp`: the input
/// is already filtered to finite positives, and `total_cmp` is a total order, so
/// this cannot panic on a stray NaN reaching it through a future edit.
#[cfg(feature = "ocr-pipeline")]
fn median_font_size_pt(values: &mut [f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    (median.is_finite() && median > 0.0).then_some(median)
}

/// Resolve a font size in PDF points for an OCR-produced text element.
///
/// Prefers the hOCR `x_fsize` attribute when present: it is already reported in
/// points by the backend that populated it (tesseract's per-block average, set by
/// `ocr::hocr_parser`), so it needs no unit conversion. That branch returns
/// *before* `block_median_pt` is consulted, so the Tesseract path is bit-identical
/// to what it was before #712 -- not "provably a no-op through the median", but
/// never routed through it at all, which is a stronger and cheaper guarantee.
///
/// Otherwise prefers `block_median_pt`, the median over every measurable fragment
/// of this element's hOCR block (see [`ocr_block_median_font_sizes`]), which is what
/// removes the intra-line detection-box variance that sceptre and PaddleOCR would
/// otherwise feed into the heading and paragraph-break thresholds.
///
/// Falls back last to a proxy derived from the element's own line-height
/// (`quad_edge_height_px` when the backend reports a quadrilateral, else the raw
/// bbox height), scaled from OCR raster pixels to PDF points via `font_size_scale`.
/// Each branch is scaled by the factor appropriate to *its own* unit space -- see
/// [`OcrFontSizeScale`] -- because `element.bbox` and `element.ocr_geometry` are not
/// always in the same unit space by the time this runs. The scaling matters either
/// way: the document-level heading heuristic
/// (`pdf::structure::pipeline::extract_document_structure_from_segments`) compares
/// font sizes against `MIN_HEADING_FONT_GAP`, an **absolute-points** constant: a
/// pixel-space value would either swamp that constant into irrelevance (high-DPI
/// rasters) or make it dominate spuriously (low-DPI rasters), silently mis-tuning
/// heading promotion either way.
#[cfg(feature = "ocr-pipeline")]
fn resolve_ocr_font_size_pt(
    element: &crate::types::internal::InternalElement,
    block_bbox: Option<(f32, f32, f32, f32)>,
    line_count: usize,
    font_size_scale: OcrFontSizeScale,
    block_median_pt: Option<f32>,
) -> f32 {
    if let Some(font_size) = hocr_font_size_pt(element) {
        return font_size;
    }
    if let Some(median) = block_median_pt {
        return median;
    }
    geometric_ocr_font_size_pt(element, block_bbox, line_count, font_size_scale).unwrap_or(DEFAULT_OCR_FONT_SIZE_PT)
}

/// Rotation-robust line-height proxy for a 4-point OCR quadrilateral, in raster
/// pixels.
///
/// `resolve_ocr_font_size_pt`'s previous fallback took the axis-aligned bounding
/// box height (`top - bottom` of `block_bbox`) as a font-size proxy. For a
/// perfectly horizontal quad that is the line height; for a quad with *any* skew
/// -- unavoidable on real scanned pages -- the AABB height also picks up
/// `line_width * sin(skew_angle)`, a term that grows with how much *text* is on
/// the line rather than with its *point size*. Measured on
/// `test_documents/pdf_scanned/ordinance_2197_scanned.pdf` (sceptre backend,
/// non-layout-detection route): resolved font sizes for `blocks_to_paragraphs`
/// ranged from 8.6 to 472.3 and tracked word count (13-word title: 470.4;
/// 2-word "EXHIBIT B-3": 74.9; 1-word "H.": 13.4) rather than the two visibly
/// different type sizes actually on the page, so the heading/body clusters
/// were inseparable and every page logged `headings=0`.
///
/// This computes the two side-edge lengths of the quad (`top_left`-`bottom_left`
/// and `top_right`-`bottom_right`) and averages them instead. A side edge runs
/// *across* the line, not *along* it, so it measures the true glyph-height band
/// regardless of the line's length or skew.
///
/// Returns `None` for `OcrBoundingGeometry::Rectangle` (already axis-aligned by
/// construction; Tesseract's proxy path already used `block_bbox` directly and is
/// unaffected) so callers fall back to the existing `block_bbox`-height proxy.
#[cfg(feature = "ocr-pipeline")]
fn quad_edge_height_px(geometry: &crate::types::OcrBoundingGeometry) -> Option<f32> {
    let crate::types::OcrBoundingGeometry::Quadrilateral { points } = geometry else {
        return None;
    };
    let [top_left, top_right, bottom_right, bottom_left] = points.as_slice() else {
        return None;
    };
    let left_edge = point_distance((*top_left).into(), (*bottom_left).into());
    let right_edge = point_distance((*top_right).into(), (*bottom_right).into());
    let height = (left_edge + right_edge) / 2.0;
    (height.is_finite() && height > 0.0).then_some(height)
}

#[cfg(feature = "ocr-pipeline")]
fn point_distance(a: (u32, u32), b: (u32, u32)) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;
    dx.hypot(dy)
}

/// Harvest [`crate::pdf::hierarchy::SegmentData`] out of already-built OCR paragraphs,
/// one page at a time, for the document-level heading/list heuristic
/// (`pdf::structure::pipeline::extract_document_structure_from_segments`).
///
/// Reuses the geometry `make_ocr_pdf_line` already computed instead of re-deriving it
/// from the raw `InternalDocument`: every [`types::PdfLine`] built by the functions in
/// this module carries exactly one `SegmentData` (see `make_ocr_pdf_line`), so this is
/// a plain extraction, not a second conversion path.
#[cfg(feature = "ocr-pipeline")]
pub(crate) fn segments_from_ocr_pages(
    pages: &[Vec<types::PdfParagraph>],
) -> Vec<Vec<crate::pdf::hierarchy::SegmentData>> {
    pages
        .iter()
        .map(|paragraphs| {
            paragraphs
                .iter()
                .flat_map(|paragraph| paragraph.lines.iter())
                .flat_map(|line| line.segments.iter().cloned())
                .filter(|segment| !segment.text.trim().is_empty())
                .collect()
        })
        .collect()
}

#[cfg(feature = "ocr-pipeline")]
fn hocr_block_id(element: &crate::types::internal::InternalElement) -> Option<&str> {
    const HOCR_BLOCK_ID_ATTRIBUTE: &str = "hocr_block_id";
    const MAX_HOCR_BLOCK_FRAGMENT_LINES: usize = 6;

    if element.text.lines().count() > MAX_HOCR_BLOCK_FRAGMENT_LINES {
        return None;
    }

    element
        .attributes
        .as_ref()?
        .get(HOCR_BLOCK_ID_ATTRIBUTE)
        .map(String::as_str)
        .filter(|block_id| !block_id.is_empty())
}

#[cfg(feature = "ocr-pipeline")]
fn merge_ocr_block_paragraph(current: &mut types::PdfParagraph, next: types::PdfParagraph) {
    current.text.push('\n');
    current.text.push_str(&next.text);
    current.lines.extend(next.lines);
    current.block_bbox = match (current.block_bbox, next.block_bbox) {
        (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))),
        (bbox @ Some(_), None) | (None, bbox @ Some(_)) => bbox,
        (None, None) => None,
    };
    current.word_count = types::PdfParagraph::compute_word_count(&current.text, &current.lines);
}

/// Convert unstructured OCR text into page paragraphs without inventing geometry.
///
/// Line endings are normalized first: OCR page text is not guaranteed to be LF-only.
/// The VLM backend returns the model's markdown verbatim out of an HTTP JSON body
/// (`crate::llm::vlm_ocr`), which routinely carries `\r\n`, and nothing on the way
/// here rewrites it. Splitting raw would fold a whole page into one paragraph (#316).
#[cfg(feature = "ocr-pipeline")]
pub(crate) fn ocr_text_to_paragraphs(text: &str) -> Vec<types::PdfParagraph> {
    crate::extraction::transform::normalize_line_endings(text)
        .split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .map(|paragraph| make_ocr_paragraph(paragraph.to_string(), Vec::new(), None, DEFAULT_OCR_FONT_SIZE_PT))
        .collect()
}

#[cfg(feature = "ocr-pipeline")]
fn trace_conversion(
    route: &'static str,
    doc: &crate::types::internal::InternalDocument,
    result: &[types::PdfParagraph],
) {
    // GH#793 A/B instrumentation. Off by default (tracing target must be enabled to
    // see it): the two call sites (`ocr_doc_to_paragraphs` -- non-layout -- and
    // `ocr_doc_to_layout_paragraphs` -- layout) pass a distinct `route` so the exact
    // same fields can be diffed side by side to find where the two arms' word counts
    // diverge for a single document. `furniture_paragraphs`/`furniture_words` count
    // paragraphs this call marked `is_page_furniture` -- the only route-exclusive
    // classification of the two (the non-layout route never runs a layout hint, so
    // it is structurally always 0 there). A furniture paragraph still SURVIVES this
    // function (it is a normal entry in `result`, text intact) -- it is excluded
    // later, at render time, by `rendering::common::is_body_element` keeping only
    // `ContentLayer::Body` -- so a non-zero count here does not yet prove the text
    // was lost, only that it is now at risk of being layered out of the default
    // (Body-only) text/markdown output. Correlate against the final rendered output
    // to confirm.
    let input_elements = doc
        .elements
        .iter()
        .filter(|element| matches!(element.kind, crate::types::internal::ElementKind::OcrText { .. }));
    let input_word_count: usize = input_elements
        .clone()
        .map(|element| element.text.split_whitespace().count())
        .sum();
    let output_word_count: usize = result.iter().map(|paragraph| paragraph.word_count).sum();
    let furniture_paragraphs = result.iter().filter(|paragraph| paragraph.is_page_furniture).count();
    let furniture_words: usize = result
        .iter()
        .filter(|paragraph| paragraph.is_page_furniture)
        .map(|paragraph| paragraph.word_count)
        .sum();
    tracing::debug!(
        target: "xberg::pdf::structure::adapters::ocr_conversion",
        route,
        input_elements = input_elements.count(),
        input_word_count,
        output_paragraphs = result.len(),
        output_word_count,
        furniture_paragraphs,
        furniture_words,
        total_text_chars = result.iter().map(|paragraph| paragraph.text.len()).sum::<usize>(),
        "ocr paragraph conversion"
    );
}

/// Build one paragraph per hOCR block -- or several, when the block's own OCR lines
/// contain at least [`MIN_LIST_MARKERS_TO_SPLIT`] independent list-marker openings.
///
/// Tesseract (and other backends) group physically adjacent lines into one hOCR
/// block by layout proximity alone, with no notion of "list item" -- a tightly
/// spaced list can end up as a single block spanning every item. Before this, the
/// non-layout OCR route (`ocr_doc_to_paragraphs`) turned that whole block into one
/// paragraph unconditionally, so only the block's first line was ever visible to a
/// downstream text-marker classifier; items 2..N were swallowed into item 1's body
/// (#713 -- the same problem `push_body_group`/`split_body_group_at_list_markers`
/// already solve for the ML-layout route, reused here rather than re-implemented).
///
/// Below the threshold this returns exactly the single paragraph the old
/// `make_ocr_block_paragraph` built, byte-for-byte: same `text` (the raw
/// `element.text`, not a rejoin), same block-level (not per-segment) font size. Only
/// once a real split happens do segments switch to the `split_body_group_at_list_markers`
/// / `build_body_paragraph` construction already used by the layout route (including its
/// marker-led space-join -- see `join_body_segment_text`'s doc comment for why that
/// matters for rendering).
#[cfg(feature = "ocr-pipeline")]
fn make_ocr_block_paragraphs(
    element: &crate::types::internal::InternalElement,
    page_height: f32,
    font_size_scale: OcrFontSizeScale,
    block_median_pt: Option<f32>,
) -> Vec<types::PdfParagraph> {
    let block_bbox = pdf_block_bbox(element, page_height);
    let line_paragraphs = make_ocr_line_paragraphs(element, page_height, font_size_scale, block_median_pt);

    let marker_line_count = line_paragraphs
        .iter()
        .filter(|paragraph| super::pipeline::looks_like_list_item(&paragraph.text))
        .count();
    if marker_line_count < MIN_LIST_MARKERS_TO_SPLIT {
        let lines = line_paragraphs
            .into_iter()
            .flat_map(|paragraph| paragraph.lines)
            .collect();
        let text_line_count = element.text.split('\n').count().max(1);
        let font_size =
            resolve_ocr_font_size_pt(element, block_bbox, text_line_count, font_size_scale, block_median_pt);
        return vec![make_ocr_paragraph(element.text.clone(), lines, block_bbox, font_size)];
    }

    split_body_group_at_list_markers(line_paragraphs)
        .into_iter()
        .filter_map(build_body_paragraph)
        .collect()
}

#[cfg(feature = "ocr-pipeline")]
fn make_ocr_line_paragraphs(
    element: &crate::types::internal::InternalElement,
    page_height: f32,
    font_size_scale: OcrFontSizeScale,
    block_median_pt: Option<f32>,
) -> Vec<types::PdfParagraph> {
    let block_bbox = pdf_block_bbox(element, page_height);
    let text_lines = element.text.split('\n').collect::<Vec<_>>();
    let line_count = text_lines.len().max(1);
    // Resolved once per element (an hOCR block, or a single OCR line for
    // backends that emit line-level elements): `x_fsize` when the backend
    // provides one is itself a block/element-level average, so re-deriving a
    // per-line value would add false precision without a real per-line signal.
    // For the geometric backends, `block_median_pt` widens that "once" from the
    // element to the whole hOCR block -- an element there is a single detection
    // box, so per-element resolution is precisely the per-fragment variance #712
    // is about.
    let font_size = resolve_ocr_font_size_pt(element, block_bbox, line_count, font_size_scale, block_median_pt);
    // Same reasoning applies to bold/italic: `x_bold_fraction`/`x_italic_fraction`
    // are block-level averages, resolved once and applied uniformly to every line.
    let (is_bold, is_italic) = resolve_ocr_style_flags(element);

    text_lines
        .into_iter()
        .enumerate()
        .map(|(line_index, text)| {
            make_ocr_line_paragraph(text, line_index, line_count, block_bbox, font_size, is_bold, is_italic)
        })
        .collect()
}

#[cfg(feature = "ocr-pipeline")]
fn pdf_block_bbox(element: &crate::types::internal::InternalElement, page_height: f32) -> Option<(f32, f32, f32, f32)> {
    element.bbox.as_ref().map(|bbox| {
        (
            bbox.x0 as f32,
            page_height - bbox.y1 as f32,
            bbox.x1 as f32,
            page_height - bbox.y0 as f32,
        )
    })
}

#[cfg(feature = "ocr-pipeline")]
fn make_ocr_line_paragraph(
    text: &str,
    line_index: usize,
    line_count: usize,
    block_bbox: Option<(f32, f32, f32, f32)>,
    font_size: f32,
    is_bold: bool,
    is_italic: bool,
) -> types::PdfParagraph {
    const DEFAULT_LINE_WIDTH: f32 = 100.0;

    let line_height = block_bbox
        .map(|(_, bottom, _, top)| (top - bottom) / line_count as f32)
        .unwrap_or(DEFAULT_OCR_FONT_SIZE_PT);
    let line_bbox = block_bbox.map(|(left, _bottom, right, top)| {
        let line_top = top - line_index as f32 * line_height;
        (left, line_top - line_height, right, line_top)
    });
    let (x, baseline_y, width) = line_bbox
        .map(|(left, bottom, right, _)| (left, bottom, right - left))
        .unwrap_or((0.0, 0.0, DEFAULT_LINE_WIDTH));
    let lines = if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![make_ocr_pdf_line(
            text,
            x,
            baseline_y,
            width,
            line_height,
            font_size,
            is_bold,
            is_italic,
        )]
    };
    make_ocr_paragraph(text.to_string(), lines, line_bbox, font_size)
}

/// A merged body group is only split into separate paragraphs once it contains at
/// least this many marker-opening lines. A single marker-opening line is not
/// enough signal: ordinary prose regularly *starts* with something that parses as
/// a list marker (e.g. a lead-in sentence "1. Overview covers..."), and splitting
/// on that alone would carve one such paragraph into a spurious empty lead segment
/// plus itself. Two or more independent marker openings in the same merged group
/// is the point where "this is a list" stops being a guess.
///
/// Shared by both the layout route (`push_body_group`, below) and the non-layout
/// route (`make_ocr_block_paragraphs`): the layout-detection feature requirement was
/// incidental, not deliberate -- this predicate is pure text, needs no layout input,
/// and #713 found the non-layout OCR routes had no marker-splitting at all.
#[cfg(feature = "ocr-pipeline")]
const MIN_LIST_MARKERS_TO_SPLIT: usize = 2;

/// Split `lines` into segments at each line for which `looks_like_list_item` is
/// true, provided there are at least [`MIN_LIST_MARKERS_TO_SPLIT`] such lines.
/// Lines before the first marker (e.g. a lead-in sentence) form their own leading
/// segment rather than being absorbed into the first item; a wrapped continuation
/// line carries no marker of its own and so stays in the segment it already
/// belongs to. Below the threshold, `lines` is returned unsplit as the sole
/// segment. Segments are built forward into a `Vec<Vec<_>>` in original order --
/// no reversal, no `split_off`.
#[cfg(feature = "ocr-pipeline")]
fn split_body_group_at_list_markers(lines: Vec<types::PdfParagraph>) -> Vec<Vec<types::PdfParagraph>> {
    let marker_indices = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| super::pipeline::looks_like_list_item(&line.text))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();

    if marker_indices.len() < MIN_LIST_MARKERS_TO_SPLIT {
        return vec![lines];
    }

    let mut boundaries = marker_indices;
    if boundaries.first() != Some(&0) {
        boundaries.insert(0, 0);
    }

    let mut slots = lines.into_iter().map(Some).collect::<Vec<_>>();
    boundaries
        .iter()
        .enumerate()
        .map(|(window_index, &start)| {
            let end = boundaries.get(window_index + 1).copied().unwrap_or(slots.len());
            slots[start..end]
                .iter_mut()
                .filter_map(Option::take)
                .collect::<Vec<_>>()
        })
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// Join a body segment's lines into paragraph text.
///
/// A segment whose first line satisfies `looks_like_list_item` is marker-led: its
/// remaining lines are wrapped continuations of one logical item, so they join
/// with a single space, not a newline. Newline-joining a marker-led segment looks
/// harmless in isolation, but `apply_ocr_text_list_fallback`
/// (`extractors::pdf::ocr`) flags any paragraph whose text starts with a list
/// marker as `is_list_item = true` regardless of how many lines it has, and the
/// markdown renderer then treats an embedded `"\n"` inside a list-item paragraph
/// as a block break -- so a wrapped continuation line came out as its own
/// standalone paragraph in production markdown. Non-marker-led segments (plain
/// prose) keep the long-standing `"\n"` join, unchanged.
///
/// Each line is trimmed before joining, and empty lines are dropped, so a line
/// with trailing/leading whitespace (or a rare interior blank line) can't turn
/// into a double space.
#[cfg(feature = "ocr-pipeline")]
fn join_body_segment_text(lines: &[types::PdfParagraph]) -> String {
    let marker_led = lines
        .first()
        .is_some_and(|line| super::pipeline::looks_like_list_item(&line.text));
    if marker_led {
        lines
            .iter()
            .map(|line| line.text.trim())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        lines
            .iter()
            .map(|line| line.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Build one merged body paragraph from a (possibly split) run of OCR-line
/// paragraphs -- the construction `push_body_group` always used, extracted so it
/// applies identically to every segment `split_body_group_at_list_markers`
/// produces: `pdf_lines` is collected only from this segment's lines, `block_bbox`
/// unions only this segment's boxes, and `layout_class` takes the first non-`None`
/// value within this segment (not the whole original group).
#[cfg(feature = "ocr-pipeline")]
fn build_body_paragraph(lines: Vec<types::PdfParagraph>) -> Option<types::PdfParagraph> {
    let text = join_body_segment_text(&lines);
    if text.trim().is_empty() {
        // GH#793 instrumentation: a body segment whose lines were all already blank
        // (see `make_ocr_line_paragraph`'s empty-text branch). Shared by both routes
        // -- the layout route via `push_body_group`, the non-layout route via
        // `make_ocr_block_paragraphs`'s >= MIN_LIST_MARKERS_TO_SPLIT path -- so a
        // nonzero count here is not route-exclusive and is not expected to explain a
        // layout-only loss; recorded for completeness of the site enumeration.
        tracing::trace!(
            target: "xberg::pdf::structure::adapters::ocr_conversion",
            dropped_lines = lines.len(),
            "ocr paragraph conversion: body segment empty after join"
        );
        return None;
    }
    let bbox = union_bboxes(&lines);
    let layout_class = lines.iter().find_map(|line| line.layout_class);
    let font_size = lines.iter().map(|line| line.dominant_font_size).fold(0.0_f32, f32::max);
    let font_size = if font_size > 0.0 {
        font_size
    } else {
        DEFAULT_OCR_FONT_SIZE_PT
    };
    let pdf_lines = lines.into_iter().flat_map(|line| line.lines).collect();
    let mut paragraph = make_ocr_paragraph(text, pdf_lines, bbox, font_size);
    paragraph.layout_class = layout_class;
    Some(paragraph)
}

// Both call sites -- `push_body_group` and `merge_structural_group` -- are gated on
// `layout-detection` as well, so the looser `any(ocr, ocr-pipeline)` gate this used to carry
// left the function compiled with zero callers on the `ocr`-without-`layout-detection` leg,
// which is one of the narrow feature legs CI now checks independently (a167989b2e). ~keep

#[cfg(feature = "ocr-pipeline")]
fn union_bboxes(lines: &[types::PdfParagraph]) -> Option<(f32, f32, f32, f32)> {
    lines
        .iter()
        .filter_map(|line| line.block_bbox)
        .reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
}

/// Reattach a genuinely UNCLASSIFIED bare OCR list marker -- `is_list_item ==
/// false`, because no ML layout hint ever fired on it -- to the body paragraph it
/// belongs to. (#729)
///
/// Delegates to the native pipeline's own
/// [`super::pipeline::reattach_detached_list_markers`], passing
/// [`super::pipeline::DetachedMarkerFrame::OcrOnPage`] with this page's PDF
/// `/Rotate` value (#760): that function's own geometry test is expressed
/// relative to font size specifically so it "behaves identically" on OCR's
/// raster-derived paragraph geometry and native PDF points, but on a *rotated*
/// OCR page the raw baseline/advance geometry sits in the raster frame, not the
/// upright one, and OCR segments cannot carry the correction on
/// `rotation_degrees` the way native rotated text does -- see
/// `DetachedMarkerFrame`'s doc comment for why. Nothing on any OCR call path
/// actually invoked this function before #729 -- its only real-world
/// reachability from OCR was through `heuristically_restructured_ocr_pages`,
/// whose document-wide "already structured" gate is tripped by the very ML
/// classifications that make list-marker recovery necessary in the first place,
/// so in practice it ran natively only.
///
/// This is the mirror image of [`reattach_ocr_layout_list_markers`], not a
/// duplicate of it. That function's marker-side test (`ocr_detached_list_marker`)
/// requires `is_list_item == true` -- proof an ML hint already fired on the marker
/// -- and pairs a whole marker run against a whole body run by position. This
/// function's marker-side test (`pipeline::detached_list_marker`) requires the
/// OPPOSITE, `is_list_item == false`, and pairs a single marker to a single body by
/// baseline instead: it covers the shape `reattach_ocr_layout_list_markers` cannot
/// reach at all -- no hint ever fired on the marker (missing detection, confidence
/// below 0.8, or `layout-detection` disabled entirely) -- so the marker paragraph
/// is not `is_list_item == true` by construction the way that function's
/// precondition requires. Without this pass such a marker stays a bare, unmerged
/// paragraph and renders as literal marker text (e.g. `"(a)"`) in the output.
///
/// Never touches `heading_level` (`pipeline::reattach_detached_list_markers` only
/// ever sets `is_list_item`), so -- like `apply_ocr_text_list_fallback` in
/// `extractors::pdf::ocr` -- it cannot trip
/// `heuristically_restructured_ocr_pages`'s "already structured" gate or regress
/// heading detection: callers run it entirely outside that gate, as a fallback
/// pass alongside `apply_ocr_text_list_fallback`, never before it.
#[cfg(feature = "ocr-pipeline")]
pub(crate) fn reattach_detached_ocr_list_markers(
    paragraphs: &mut Vec<types::PdfParagraph>,
    page_rotation_degrees: u32,
) {
    super::pipeline::reattach_detached_list_markers(
        paragraphs,
        super::pipeline::DetachedMarkerFrame::OcrOnPage(page_rotation_degrees),
    );
}

#[cfg(feature = "ocr-pipeline")]
fn make_ocr_paragraph(
    text: String,
    lines: Vec<types::PdfLine>,
    block_bbox: Option<(f32, f32, f32, f32)>,
    font_size: f32,
) -> types::PdfParagraph {
    types::PdfParagraph {
        word_count: types::PdfParagraph::compute_word_count(&text, &lines),
        text,
        lines,
        dominant_font_size: font_size,
        heading_level: None,
        is_bold: false,
        is_list_item: false,
        is_code_block: false,
        is_formula: false,
        is_page_furniture: false,
        layout_class: None,
        layout_region_path: None,
        caption_for: None,
        block_bbox,
    }
}

#[cfg(feature = "ocr-pipeline")]
#[allow(clippy::too_many_arguments)]
fn make_ocr_pdf_line(
    text: &str,
    x: f32,
    baseline_y: f32,
    width: f32,
    line_height: f32,
    font_size: f32,
    is_bold: bool,
    is_italic: bool,
) -> types::PdfLine {
    let segment = crate::pdf::hierarchy::SegmentData {
        text: text.to_string(),
        x,
        y: baseline_y,
        width,
        height: line_height,
        font_size,
        is_bold,
        is_italic,
        is_monospace: false,
        baseline_y,
        rotation_degrees: 0.0,
        assigned_role: None,
    };
    types::PdfLine {
        native_scripts: Vec::new(),
        segments: vec![segment],
        baseline_y,
        is_monospace: false,
    }
}
