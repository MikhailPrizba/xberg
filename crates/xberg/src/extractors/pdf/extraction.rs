//! Core PDF extraction functionality.
//!
//! Handles document loading, text extraction, metadata parsing, and table detection.

use crate::Result;
use crate::core::config::{ExtractionConfig, OutputFormat};
use crate::types::{PageBoundary, PageContent, PdfAnnotation};

#[cfg(test)]
#[path = "extraction/native_table_metadata_tests.rs"]
mod native_table_metadata_tests;

#[cfg(feature = "pdf")]
use crate::types::Table;

#[cfg(feature = "pdf")]
pub(crate) type PdfExtractionPhaseResult = (
    crate::pdf::metadata::PdfExtractionMetadata,
    String,
    Vec<Table>,
    Option<Vec<PageContent>>,
    Option<Vec<PageBoundary>>,
    Option<crate::types::internal::InternalDocument>,
    bool,
    Option<AnnotationTextFallback>,
    Option<Vec<PdfAnnotation>>,
    Option<Vec<crate::types::ExtractedImage>>,
    Vec<crate::types::PdfFormField>,
    Vec<crate::types::ProcessingWarning>,
    Option<Vec<String>>,
    Vec<crate::types::internal::PdfPageCoordinateFrame>,
    Vec<u32>,
);

#[cfg(feature = "pdf")]
#[derive(Clone)]
pub(crate) struct AnnotationTextFallback {
    native_pages: Vec<String>,
    annotation_pages: Vec<Vec<String>>,
}

#[cfg(feature = "pdf")]
pub(crate) struct AnnotationFallbackTarget<'a> {
    pub(crate) text: &'a mut String,
    pub(crate) page_contents: &'a mut Option<Vec<PageContent>>,
    pub(crate) boundaries: &'a mut Option<Vec<PageBoundary>>,
    pub(crate) pdf_metadata: &'a mut crate::pdf::metadata::PdfExtractionMetadata,
    pub(crate) page_config: Option<&'a crate::core::config::PageConfig>,
}

#[derive(Default)]
struct NativeTableMetadata {
    ruled: bool,
    grid: Option<crate::types::TableGrid>,
    scripts: Option<crate::pdf::native::table::scripts::TableScripts>,
    complete_leading_header: bool,
    source_members: Option<crate::pdf::native::table::source_members::TableSources>,
}

impl NativeTableMetadata {
    fn retain_column_labels(&self, table: &mut Table) {
        if table.columns.is_none()
            && self.complete_leading_header
            && self
                .grid
                .as_ref()
                .is_some_and(|grid| crate::extraction::derive::native_grid_matches_table(table, grid))
            && let Some(labels) = table.cells.first()
            && labels.iter().all(|label| !label.trim().is_empty())
        {
            table.columns = Some(labels.clone());
        }
    }
}

struct NativeHierarchySegments {
    pages: Vec<crate::pdf::native::hierarchy::script_evidence::NativePageSegments>,
    used_structure_tree: bool,
}

type TableExtractionResult = (
    Vec<Table>,
    Vec<NativeTableMetadata>,
    Option<NativeHierarchySegments>,
    Vec<crate::types::ProcessingWarning>,
);

fn attach_native_table_grids(
    document: &mut crate::types::internal::InternalDocument,
    tables: &[Table],
    grids: &[NativeTableMetadata],
) -> Result<()> {
    let sidecars: Vec<_> = tables
        .iter()
        .zip(grids)
        .filter_map(|(table, metadata)| {
            metadata
                .grid
                .as_ref()
                .map(|grid| crate::pdf::structure::table_grid::TableGridSidecar {
                    table: table.clone(),
                    grid: grid.clone(),
                })
        })
        .collect();
    crate::pdf::structure::table_grid::attach_ruled_table_grids(document, &sidecars)
        .map_err(|error| crate::XbergError::parsing(format!("Failed to retain native table geometry: {error}")))?;
    for element in &mut document.elements {
        let crate::types::internal::ElementKind::Table { table_index } = element.kind else {
            continue;
        };
        let index = table_index as usize;
        let Some((table, metadata)) = tables.get(index).zip(grids.get(index)) else {
            continue;
        };
        if let Some(scripts) = &metadata.scripts
            && scripts.cells == table.cells
            && document
                .tables
                .get(index)
                .is_some_and(|current| current.cells == scripts.cells)
        {
            let encoded = serde_json::to_string(&scripts.annotations)
                .map_err(|error| crate::XbergError::parsing(format!("Failed to retain PDF table scripts: {error}")))?;
            element.attributes.get_or_insert_with(Default::default).insert(
                crate::rendering::table_scripts::TABLE_CELL_SCRIPTS_ATTRIBUTE.to_string(),
                encoded,
            );
        }
    }
    Ok(())
}

fn retain_headerless_financial_tables(
    tables: &mut [Table],
    metadata: &[NativeTableMetadata],
    financial_rows: &crate::pdf::native::text::NativeFinancialRows,
    document: Option<&mut crate::types::internal::InternalDocument>,
) {
    use crate::rendering::common::render_headerless_table_markdown;
    use crate::types::internal::{ElementKind, NATIVE_PDF_HEADERLESS_TABLE_ATTRIBUTE};

    if financial_rows.is_empty() {
        return;
    }
    let mut page_counts = ahash::AHashMap::<u32, usize>::new();
    for table in tables
        .iter()
        .filter(|table| financial_rows.contains_key(&table.page_number))
    {
        *page_counts.entry(table.page_number).or_default() += 1;
    }
    let mut proven = ahash::AHashMap::new();
    for (table, metadata) in tables.iter_mut().zip(metadata) {
        if page_counts.get(&table.page_number) == Some(&1)
            && financial_rows.get(&table.page_number) == Some(&table.cells)
            && table.columns.is_none()
            && !metadata.complete_leading_header
            && metadata
                .grid
                .as_ref()
                .is_none_or(|grid| grid.cells.iter().all(|cell| !cell.is_header))
        {
            table.markdown = render_headerless_table_markdown(&table.cells);
            proven.insert(table.page_number, table);
        }
    }
    let Some(document) = document else {
        return;
    };
    for element in &mut document.elements {
        let ElementKind::Table { table_index } = element.kind else {
            continue;
        };
        let Some(table) = document.tables.get_mut(table_index as usize) else {
            continue;
        };
        if let Some(source) = proven.get(&table.page_number)
            && table.cells == source.cells
            && table.bounding_box == source.bounding_box
            && table.columns.is_none()
        {
            table.markdown.clone_from(&source.markdown);
            element
                .attributes
                .get_or_insert_with(Default::default)
                .insert(NATIVE_PDF_HEADERLESS_TABLE_ATTRIBUTE.to_owned(), String::new());
        }
    }
}

/// Whether the PDF pipeline must retain the full native hierarchy/geometry pass
/// rather than take the cheaper flat-text path.
///
/// `Custom(_)` is deliberately excluded: `OutputFormat::from_str` never rejects an
/// unknown string (any typo becomes `Custom(other)`), so treating every `Custom`
/// value as structure-hungry ran the full hierarchy pass — `extract_all_segments`
/// plus `extract_document_structure_from_segments`, including k-means heading
/// clustering — for a typo'd or unregistered format, only to discard the result
/// and fall back to plain text during derivation
/// (`core/pipeline/format.rs`'s `custom_fallback_to_plain`). `DocTags` is a real,
/// always-registered built-in renderer that needs the same geometry and headings
/// as Markdown/Djot/HTML, so it gets its own explicit arm instead. So do the
/// built-in binary formats, DOCX and PDF, which are built from the Markdown rendering.
///
/// `include_document_structure` triggers it directly (GH#1668): a caller who set
/// only that flag, with `output_format` left at its `Plain` default, used to get
/// `flat_pdf_document`'s paragraph-only elements. `include_document_structure`
/// then dutifully built a structure tree from them, correctly, since the tree
/// builder is not what was broken -- it had nothing but paragraphs to build
/// from. `counts.tables` looked unaffected because it reads `doc.tables`
/// directly, never the element tree, so the document looked like it knew about
/// its own tables while the structure asking for them came back flat.
fn needs_structured_extraction(
    hierarchy_enabled: bool,
    include_document_structure: bool,
    output_format: &OutputFormat,
    ocr_inline_images: bool,
    content_filter_configured: bool,
) -> bool {
    hierarchy_enabled
        || include_document_structure
        || matches!(
            output_format,
            OutputFormat::Markdown | OutputFormat::Djot | OutputFormat::Html | OutputFormat::DocTags
        )
        || matches!(output_format, OutputFormat::Custom(name) if crate::plugins::registry::renders_from_markdown(name))
        || ocr_inline_images
        || content_filter_configured
}

fn hierarchy_cluster_count(config: &ExtractionConfig) -> usize {
    config
        .pdf_options
        .as_ref()
        .and_then(|options| options.hierarchy.as_ref())
        .map_or_else(
            || crate::core::config::HierarchyConfig::default().k_clusters,
            |hierarchy| hierarchy.k_clusters,
        )
}

/// What an image element of a PDF document stands for.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ImageAnchor {
    /// The document gets no image elements.
    None,
    /// The placeholder of the picture, with its OCR text where a renderer adds it.
    Placeholder,
}

/// The one answer to "does this PDF document get image elements, and for what".
///
/// A picture that is sent to OCR needs an element whether or not the caller wants a
/// placeholder: the element is the only anchor for the recognized text (GH#2068). Both
/// sites that make image elements call this; neither derives it again. ~keep
pub(super) fn image_anchor(config: &ExtractionConfig) -> ImageAnchor {
    let extract_images = config.images.as_ref().is_some_and(|images| images.extract_images)
        || config
            .pdf_options
            .as_ref()
            .is_some_and(|options| options.extract_images);
    if extract_images && config.images.as_ref().is_some_and(|images| images.inject_placeholders) {
        return ImageAnchor::Placeholder;
    }
    ImageAnchor::None
}

/// Report a table-extraction failure that took out a whole detector pass, not just one page.
///
/// The per-page warnings in `pdf::native::table` cannot cover these: a stage that fails or
/// panics outright returns no pages at all, so its own warnings never reach the caller and
/// the document comes back with an empty `tables` list indistinguishable from a PDF that
/// genuinely has none. ~keep
///
/// Module-scoped (rather than nested inside `extract_all_from_native_document`) so
/// `mod tests` below can exercise its message format directly — see
/// `table_stage_failure_warning_formats_the_message_it_promises`.
fn table_stage_failure_warning(stage: &str, error: &impl std::fmt::Display) -> crate::types::ProcessingWarning {
    crate::types::ProcessingWarning {
        source: std::borrow::Cow::Borrowed("pdf_tables"),
        message: std::borrow::Cow::Owned(format!(
            "{stage} table extraction failed for the document: {error}; \
             tables from this pass were skipped"
        )),
    }
}

#[cfg(feature = "pdf")]
fn extract_annotations_with_empty_body_fallback(
    doc: &mut crate::pdf::native::NativeDocument,
    requested: bool,
    native_text: &str,
    page_contents: Option<&[PageContent]>,
    boundaries: Option<&[PageBoundary]>,
) -> (
    Option<Vec<PdfAnnotation>>,
    Option<AnnotationTextFallback>,
    Vec<crate::types::ProcessingWarning>,
) {
    if requested {
        let (annotations, warnings) = crate::pdf::native::annotations::extract_annotations(doc);
        return ((!annotations.is_empty()).then_some(annotations), None, warnings);
    }

    let page_count = match doc.doc.page_count() {
        Ok(page_count) => page_count,
        Err(error) => {
            return (
                None,
                None,
                vec![crate::pdf::native::annotations::page_count_failure_warning(&error)],
            );
        }
    };
    let native_pages = native_page_texts(native_text, page_contents, boundaries, page_count);
    if native_pages.len() != page_count {
        return (None, None, Vec::new());
    }
    let eligible_pages: Vec<bool> = native_pages.iter().map(|page| page.trim().is_empty()).collect();
    if !eligible_pages.iter().any(|eligible| *eligible) {
        return (None, None, Vec::new());
    }
    let (annotations, mut warnings) =
        crate::pdf::native::annotations::extract_visible_free_text_annotations(doc, &eligible_pages);
    let (annotation_pages, recovered_count) = annotation_text_by_page(&annotations, page_count);
    if recovered_count > 0 {
        warnings.push(crate::types::ProcessingWarning {
            source: std::borrow::Cow::Borrowed("pdf_annotations"),
            message: std::borrow::Cow::Owned(format!(
                "native PDF page text was empty; recovered {} text-bearing annotation(s) as document content",
                recovered_count
            )),
        });
    }

    let fallback = (recovered_count > 0).then_some(AnnotationTextFallback {
        native_pages,
        annotation_pages,
    });
    (None, fallback, warnings)
}

#[cfg(feature = "pdf")]
fn native_page_texts(
    native_text: &str,
    page_contents: Option<&[PageContent]>,
    boundaries: Option<&[PageBoundary]>,
    page_count: usize,
) -> Vec<String> {
    if let Some(pages) = page_contents.filter(|pages| pages.len() == page_count) {
        return pages.iter().map(|page| page.content.clone()).collect();
    }
    if let Some(boundaries) = boundaries.filter(|boundaries| boundaries.len() == page_count) {
        return boundaries
            .iter()
            .map(|boundary| {
                native_text
                    .get(boundary.byte_start..boundary.byte_end)
                    .unwrap_or_default()
                    .to_string()
            })
            .collect();
    }
    if native_text.trim().is_empty() {
        return vec![String::new(); page_count];
    }
    vec![native_text.to_string()]
}

#[cfg(feature = "pdf")]
fn annotation_text_by_page(annotations: &[PdfAnnotation], page_count: usize) -> (Vec<Vec<String>>, usize) {
    let mut pages = vec![Vec::new(); page_count];
    let mut recovered_count = 0;
    for annotation in annotations {
        let Some(content) = annotation
            .content
            .as_deref()
            .map(str::trim)
            .filter(|content| !content.is_empty())
        else {
            continue;
        };
        let Some(page) = annotation
            .page_number
            .checked_sub(1)
            .and_then(|page| pages.get_mut(page as usize))
        else {
            continue;
        };
        page.push(content.to_string());
        recovered_count += 1;
    }
    (pages, recovered_count)
}

#[cfg(feature = "pdf")]
pub(crate) fn apply_annotation_text_fallback(
    fallback: &AnnotationTextFallback,
    ocr_page_texts: Option<&[String]>,
    ocr_results: Option<&ahash::AHashMap<u32, String>>,
    full_ocr: bool,
    target: AnnotationFallbackTarget<'_>,
) {
    let mut native_pages = fallback.native_pages.clone();
    let single_string_ocr = native_pages.len() > 1 && ocr_page_texts.is_some_and(|pages| pages.len() == 1);
    if full_ocr && single_string_ocr {
        let authoritative_text = ocr_page_texts
            .and_then(|pages| pages.first())
            .cloned()
            .unwrap_or_else(|| target.text.clone());
        native_pages.fill(String::new());
        if let Some(first_page) = native_pages.first_mut() {
            *first_page = authoritative_text;
        }
    } else if full_ocr {
        if let Some(ocr_pages) = ocr_page_texts.filter(|pages| pages.len() == native_pages.len()) {
            native_pages.clone_from_slice(ocr_pages);
        } else {
            let authoritative_text = target.text.clone();
            native_pages.fill(String::new());
            if let Some(first_page) = native_pages.first_mut() {
                *first_page = authoritative_text;
            }
        }
    } else if let Some(ocr_results) = ocr_results {
        for (page_number, content) in ocr_results {
            if let Some(page) = page_number
                .checked_sub(1)
                .and_then(|page| native_pages.get_mut(page as usize))
            {
                page.clone_from(content);
            }
        }
    }
    for (page, annotations) in native_pages.iter_mut().zip(&fallback.annotation_pages) {
        for annotation in annotations {
            if page_has_exact_text_block(page, annotation) {
                continue;
            }
            if !page.is_empty() {
                page.push('\n');
            }
            page.push_str(annotation);
        }
    }

    if let Some(pages) = target.page_contents.as_mut() {
        for page in pages.iter_mut() {
            if let Some(content) = native_pages.get(page.page_number.saturating_sub(1) as usize) {
                page.content.clone_from(content);
                page.is_blank = Some(crate::extraction::blank_detection::is_page_text_blank(content));
            }
        }
    }

    let (rebuilt_text, rebuilt_boundaries) = join_pages_with_boundaries(&native_pages, target.page_config);
    *target.text = rebuilt_text;
    if target.boundaries.is_some() || target.page_contents.is_some() || target.pdf_metadata.page_structure.is_some() {
        *target.boundaries = Some(rebuilt_boundaries.clone());
    }
    if let Some(page_structure) = target.pdf_metadata.page_structure.as_mut() {
        page_structure.boundaries = Some(rebuilt_boundaries);
        if let Some(pages) = page_structure.pages.as_mut() {
            for page in pages {
                page.is_blank = native_pages
                    .get(page.number.saturating_sub(1) as usize)
                    .map(|content| crate::extraction::blank_detection::is_page_text_blank(content));
            }
        }
    }
}

fn page_has_exact_text_block(page: &str, annotation: &str) -> bool {
    let annotation_lines: Vec<_> = annotation.trim().lines().map(str::trim).collect();
    if annotation_lines.is_empty() {
        return false;
    }

    let page_lines: Vec<_> = page.lines().map(str::trim).collect();
    page_lines
        .windows(annotation_lines.len())
        .any(|lines| lines == annotation_lines)
}

#[cfg(feature = "pdf")]
pub(crate) fn append_annotation_fallback_elements(
    fallback: &AnnotationTextFallback,
    document: &mut crate::types::internal::InternalDocument,
) {
    for (page_index, annotations) in fallback.annotation_pages.iter().enumerate() {
        for annotation in annotations {
            if document.elements.iter().any(|element| {
                element.page == Some(page_index as u32 + 1) && page_has_exact_text_block(&element.text, annotation)
            }) {
                continue;
            }
            document.push_element(
                crate::types::internal::InternalElement::text(
                    crate::types::internal::ElementKind::Paragraph,
                    annotation,
                    0,
                )
                .with_page(page_index as u32 + 1),
            );
        }
    }
}

#[cfg(all(feature = "pdf", test))]
fn retain_segments_inside_page_margins(
    segments: &mut Vec<crate::pdf::hierarchy::SegmentData>,
    page_bottom: f32,
    page_top: f32,
    margins: crate::pdf::native::text::PageMarginFractions,
) {
    segments.retain(|segment| {
        crate::pdf::native::text::baseline_is_inside_page_margins(segment.baseline_y, page_bottom, page_top, margins)
    });
}

#[cfg(feature = "pdf")]
fn map_native_text_error(error: crate::pdf::error::PdfError) -> crate::XbergError {
    match error {
        crate::pdf::error::PdfError::Cancelled => crate::XbergError::Cancelled,
        error => crate::XbergError::Parsing {
            message: format!("xberg_native_pdf text extraction failed: {error}"),
            source: None,
        },
    }
}

/// Extract text, metadata, tables, and annotations from a PDF document using the xberg_native_pdf backend.
///
/// Accepts an authenticated `NativeDocument`, then delegates to each native extraction module.
/// The return type is `PdfExtractionPhaseResult` so callers can switch transparently between
/// backends.
///
/// # Notes
///
/// - With the `layout-detection` feature, layout images, per-page layout results, the configured
///   table model and table-overlap preference, and the resolved layout acceleration are threaded
///   into `SegmentStructureConfig`; layout hints additionally drive reading-order reordering.
/// - When output format is Markdown/Djot/HTML, the native hierarchy module extracts font
///   metrics and feeds them into the backend-agnostic structure pipeline for heading detection.
/// - Font encoding issue detection is not available; the flag is always `false`.
#[cfg(feature = "pdf")]
pub(crate) fn extract_all_from_native_document(
    mut doc: crate::pdf::native::NativeDocument,
    config: &ExtractionConfig,
    outline_entries: &[crate::pdf::bookmarks::PdfOutlineEntry],
) -> Result<PdfExtractionPhaseResult> {
    let layout_hints = None::<&[Vec<crate::pdf::structure::types::LayoutHint>]>;
    let _span = tracing::debug_span!("extract_xberg_native_pdf").entered();
    let margins = crate::pdf::native::text::PageMarginFractions::from_extraction_config(Some(config));
    let artifacts =
        crate::pdf::native::hierarchy::ArtifactRetention::from_content_filter(config.content_filter.as_ref());
    let annotation_fallback_requested = !config
        .pdf_options
        .as_ref()
        .is_some_and(|options| options.extract_annotations);
    let force_annotation_page_tracking = annotation_fallback_requested && config.pages.is_none();
    let hierarchy_enabled = config
        .pdf_options
        .as_ref()
        .is_some_and(|options| options.hierarchy.as_ref().is_some_and(|hierarchy| hierarchy.enabled));
    // ~keep A `PageHierarchy` can only be hung off a `PageContent` (`pages.rs`'s
    // `assign_hierarchy_to_pages`), and `page_contents` is produced only when
    // `pages.extract_pages` is set. `PageConfig::default()` leaves that `false`, so
    // `pdf_options.hierarchy.enabled` was a silent no-op unless the caller also set an
    // unrelated flag nothing in the hierarchy config points at: headings were detected, then
    // dropped for want of a page to attach them to (CI E2E `test_pdf_hierarchy_config`).
    // Asking for the hierarchy is the opt-in for the per-page tracking it requires.
    let force_hierarchy_page_tracking =
        hierarchy_enabled && !config.pages.as_ref().is_some_and(|pages| pages.extract_pages);
    let mut tracked_config;
    let text_config = if force_annotation_page_tracking || force_hierarchy_page_tracking {
        tracked_config = config.clone();
        let mut page_config = tracked_config.pages.take().unwrap_or_default();
        if force_hierarchy_page_tracking {
            page_config.extract_pages = true;
        }
        tracked_config.pages = Some(page_config);
        &tracked_config
    } else {
        config
    };
    let page_structure_was_implicitly_tracked =
        force_annotation_page_tracking && !crate::pdf::native::text::page_boundaries_required(config);

    #[allow(unused_mut)]
    let (mut native_text, mut boundaries, mut page_contents, mut pdf_metadata, financial_rows) =
        crate::pdf::native::text::extract_text_and_metadata(&mut doc, Some(text_config))
            .map_err(map_native_text_error)?;

    let _ = layout_hints;

    let ocr_inline_images = config
        .pdf_options
        .as_ref()
        .map(|options| options.ocr_inline_images)
        .unwrap_or(false);
    let needs_structured = needs_structured_extraction(
        hierarchy_enabled,
        config.include_document_structure,
        &config.output_format,
        ocr_inline_images,
        config.content_filter.is_some(),
    );
    let retain_hierarchy_segments = needs_structured && !config.force_ocr;

    let mut extraction_warnings: Vec<crate::types::ProcessingWarning> = Vec::new();

    let extract_tables_flag = config.pdf_options.as_ref().is_none_or(|opts| opts.extract_tables);
    let allow_single_column = config
        .pdf_options
        .as_ref()
        .is_some_and(|o| o.allow_single_column_tables);
    let (mut tables, native_table_grids, mut extracted_hierarchy_segments, table_warnings) = if extract_tables_flag {
        crate::pdf::native::guard_native_panic(
            || -> Result<TableExtractionResult> {
                let ruled = crate::pdf::native::table::extract_ruled_tables(&doc).unwrap_or_else(|error| {
                    tracing::warn!("xberg_native_pdf ruled table extraction failed: {error}");
                    crate::pdf::native::table::RuledTableExtraction {
                        tables: Vec::new(),
                        warnings: vec![
                            table_stage_failure_warning("native", &error),
                            table_stage_failure_warning("bordered", &error),
                        ],
                        recovered_only: Default::default(),
                    }
                });
                let covered_pages = ruled.covered_pages();
                let hierarchy_segments = crate::pdf::native::table::extract_tables_heuristic(
                    &mut doc,
                    allow_single_column,
                    &covered_pages,
                    artifacts,
                );
                let hierarchy_segments = hierarchy_segments.map(|mut extraction| {
                    extraction.tables.retain(|table| ruled.allows_heuristic_table(table));
                    extraction
                });
                let mut warnings = ruled.warnings;
                let (mut combined, mut native_grids): (Vec<_>, Vec<_>) = ruled
                    .tables
                    .into_iter()
                    .map(|entry| {
                        (
                            entry.table,
                            NativeTableMetadata {
                                ruled: true,
                                grid: entry.grid,
                                scripts: entry.scripts,
                                complete_leading_header: entry.complete_leading_header,
                                source_members: entry.source_members,
                            },
                        )
                    })
                    .unzip();
                let hierarchy_segments = match hierarchy_segments {
                    Ok(extraction) => {
                        native_grids
                            .resize_with(combined.len() + extraction.tables.len(), NativeTableMetadata::default);
                        combined.extend(extraction.tables);
                        drop((
                            extraction.hierarchy_segments.pages,
                            extraction.hierarchy_segments.used_structure_tree,
                        ));
                        if retain_hierarchy_segments {
                            let (pages, used_structure_tree) =
                                crate::pdf::native::hierarchy::script_evidence::extract_all_segment_carriers(
                                    &mut doc, artifacts,
                                )
                                .map_err(|error| {
                                    crate::XbergError::parsing(format!("PDF hierarchy extraction failed: {error}"))
                                })?;
                            Some(NativeHierarchySegments {
                                pages,
                                used_structure_tree,
                            })
                        } else {
                            None
                        }
                    }
                    Err(error) => {
                        tracing::warn!("xberg_native_pdf heuristic table extraction failed, skipping tables: {error}");
                        warnings.push(table_stage_failure_warning("heuristic", &error));
                        None
                    }
                };
                let repaired_tables = combined
                    .iter_mut()
                    .map(crate::pdf::table_normalize::repair_consistently_merged_numeric_column)
                    .filter(|repaired| *repaired)
                    .count();
                if repaired_tables > 0 {
                    tracing::debug!(repaired_tables, "repaired collapsed PDF table columns");
                }
                Ok((combined, native_grids, hierarchy_segments, warnings))
            },
            |panic| crate::error::XbergError::Parsing {
                message: format!("xberg_native_pdf panicked during table extraction: {panic}"),
                source: None,
            },
        )
        .unwrap_or_else(|e| {
            tracing::warn!("xberg_native_pdf table extraction panicked, skipping tables: {e}");
            (
                Vec::new(),
                Vec::new(),
                None,
                vec![table_stage_failure_warning("whole-document", &e)],
            )
        })
    } else {
        (Vec::new(), Vec::new(), None, Vec::new())
    };
    extraction_warnings.extend(table_warnings);

    let (annotations, annotation_text_fallback, annotation_warnings) = extract_annotations_with_empty_body_fallback(
        &mut doc,
        config
            .pdf_options
            .as_ref()
            .is_some_and(|options| options.extract_annotations),
        &native_text,
        page_contents.as_deref(),
        boundaries.as_deref(),
    );
    extraction_warnings.extend(annotation_warnings);
    if page_structure_was_implicitly_tracked {
        pdf_metadata.page_structure = None;
    }

    let images_extraction_enabled =
        config.needs_image_data() || config.pdf_options.as_ref().map(|p| p.extract_images).unwrap_or(false);

    // Non-empty only when OCR is the SOLE reason `images_extraction_enabled` is true, so a
    // full-page image on an already-text-bearing page can skip its decode entirely instead of
    // being decoded, PNG re-encoded, and thrown away by `drop_ocr_only_images` (GH#1732). Any
    // other consumer of image bytes leaves this empty and every image decodes as before. ~keep

    let ocr_skip_candidate_pages: std::collections::HashMap<u32, (f64, f64)> = std::collections::HashMap::new();

    let (images, image_positions) = if images_extraction_enabled || ocr_inline_images {
        let max_images = config.images.as_ref().and_then(|i| i.max_images_per_page);
        let (extracted, image_warnings) = crate::pdf::native::images::extract_images_with_data(
            &mut doc,
            max_images,
            config.cancel_token.as_ref(),
            &ocr_skip_candidate_pages,
        )
        .map_err(|e| crate::error::XbergError::Parsing {
            message: format!("xberg_native_pdf image extraction failed: {e}"),
            source: None,
        })?;
        extraction_warnings.extend(image_warnings);

        let positions: Vec<(u32, u32)> = extracted
            .iter()
            .map(|img| (img.page_number.unwrap_or(1), img.image_index))
            .collect();
        (Some(extracted), positions)
    } else {
        (None, Vec::new())
    };

    if config.cancel_token.as_ref().is_some_and(|t| t.is_cancelled()) {
        return Err(crate::error::XbergError::Cancelled);
    }

    // GH#1653 + GH#1654: one raw-MediaBox coordinate frame per page, computed alongside the
    // margin-filtering media-box read below while the native document is still in scope. Not
    // yet filtered to pages that actually end up with hierarchy blocks -- `mod.rs` does that
    // once `assign_hierarchy_to_pages` has run, since that happens after this function
    // returns and the `NativeDocument` this loop reads from is dropped. ~keep
    let mut pdf_page_coordinate_frames: Vec<crate::types::internal::PdfPageCoordinateFrame> = Vec::new();
    let mut pre_rendered_doc = if needs_structured && !config.force_ocr {
        let k = hierarchy_cluster_count(config);

        // `HierarchyConfig::include_bbox` (default `true`): when explicitly disabled,
        // bounding-box info is stripped from every structural element below so
        // `assign_hierarchy_to_pages` (extractors/pdf/pages.rs) — which maps
        // `element.bbox` straight onto each `HierarchicalBlock.bbox` — omits it too. ~keep
        let include_bbox = config
            .pdf_options
            .as_ref()
            .and_then(|opts| opts.hierarchy.as_ref())
            .map(|h| h.include_bbox)
            .unwrap_or(true);

        let (strip_repeating_text, include_headers, include_footers, include_footnotes, include_watermarks) = config
            .content_filter
            .as_ref()
            .map(|cf| {
                (
                    cf.strip_repeating_text,
                    cf.include_headers,
                    cf.include_footers,
                    cf.include_footnotes,
                    cf.include_watermarks,
                )
            })
            .unwrap_or((true, false, false, false, false));

        let (mut native_pages, used_structure_tree) = match extracted_hierarchy_segments.take() {
            Some(segments) => (segments.pages, segments.used_structure_tree),
            None => crate::pdf::native::hierarchy::script_evidence::extract_all_segment_carriers(&mut doc, artifacts)
                .map_err(|e| crate::error::XbergError::Parsing {
                message: format!("xberg_native_pdf hierarchy extraction failed: {e}"),
                source: None,
            })?,
        };

        if config.pdf_options.as_ref().is_some_and(|options| options.reading_order) {
            for page in &mut native_pages {
                page.clear_prose_rows();
            }
        }

        for (page_index, page) in native_pages.iter_mut().enumerate() {
            let (llx, lower_y, urx, upper_y) =
                doc.doc
                    .get_page_media_box(page_index)
                    .map_err(|error| crate::error::XbergError::Parsing {
                        message: format!(
                            "xberg_native_pdf failed to read page {} media box for margin filtering: {error}",
                            page_index + 1
                        ),
                        source: None,
                    })?;
            page.retain_segments(|segment| {
                crate::pdf::native::text::baseline_is_inside_page_margins(
                    segment.baseline_y,
                    lower_y.min(upper_y),
                    lower_y.max(upper_y),
                    margins,
                )
            });

            // Fail closed by omission (GH#1654): a page whose `/Rotate` is present but
            // malformed gets no record at all rather than a silently-degraded
            // `clockwise_rotation: 0`, which would be indistinguishable from an honestly
            // absent `/Rotate`. ~keep
            match doc.doc.get_page_rotation_status(page_index) {
                Ok(xberg_native_pdf::PageRotation::Absent) => {
                    pdf_page_coordinate_frames.extend(crate::types::internal::PdfPageCoordinateFrame::new(
                        (page_index + 1) as u32,
                        llx,
                        lower_y,
                        urx,
                        upper_y,
                        0,
                    ));
                }
                Ok(xberg_native_pdf::PageRotation::Valid(rotation)) => {
                    pdf_page_coordinate_frames.extend(crate::types::internal::PdfPageCoordinateFrame::new(
                        (page_index + 1) as u32,
                        llx,
                        lower_y,
                        urx,
                        upper_y,
                        rotation,
                    ));
                }
                Ok(xberg_native_pdf::PageRotation::Malformed) | Err(_) => {}
            }
        }

        let total_segs: usize = native_pages.iter().map(|page| page.segments.len()).sum();
        tracing::debug!(
            total_segs,
            k,
            used_structure_tree,
            "native structure: extracted segments for heading detection"
        );

        let image_anchor = image_anchor(config);
        let inject_placeholders = images_extraction_enabled && image_anchor != ImageAnchor::None;

        match crate::pdf::structure::pipeline::extract_document_structure_from_native_pages(
            native_pages,
            &native_table_grids
                .iter()
                .filter_map(|metadata| metadata.source_members.as_ref())
                .collect::<Vec<_>>(),
            native_table_grids.iter().take_while(|metadata| metadata.ruled).count(),
            crate::pdf::structure::SegmentStructureConfig {
                k_clusters: k,
                tables: &tables,
                outline_entries,
                strip_repeating_text,
                include_headers,
                include_footers,
                include_footnotes,
                include_watermarks,
                used_structure_tree,
                ocr_text: false,
                image_positions: &image_positions,
                images: images.as_deref(),
                inject_placeholders,
                layout_hints,
                allow_single_column,
                cancel_token: config.cancel_token.as_ref(),
            },
        ) {
            Ok(mut structured_doc) if !structured_doc.elements.is_empty() => {
                extraction_warnings.append(&mut structured_doc.processing_warnings);
                tracing::debug!(
                    elements = structured_doc.elements.len(),
                    has_headings = structured_doc
                        .elements
                        .iter()
                        .any(|e| matches!(e.kind, crate::types::internal::ElementKind::Heading { .. })),
                    "native structure: render succeeded"
                );
                if !include_bbox {
                    for element in &mut structured_doc.elements {
                        element.bbox = None;
                    }
                }
                Some(structured_doc)
            }
            Ok(mut structured_doc) => {
                extraction_warnings.append(&mut structured_doc.processing_warnings);
                tracing::warn!("native structure: rendering produced empty output, falling back to plain text");
                None
            }
            Err(e) => {
                tracing::warn!(
                    "native structure: rendering failed: {:?}, falling back to plain text",
                    e
                );
                None
            }
        }
    } else {
        None
    };

    // Column labels describe the retained table, never guide repairs or stitching. ~keep
    for (table, metadata) in tables.iter_mut().zip(&native_table_grids) {
        metadata.retain_column_labels(table);
    }
    if let Some(document) = &mut pre_rendered_doc {
        attach_native_table_grids(document, &tables, &native_table_grids)?;
    }
    retain_headerless_financial_tables(
        &mut tables,
        &native_table_grids,
        &financial_rows,
        pre_rendered_doc.as_mut(),
    );

    let outlined_ocr_pages = Vec::new();

    let has_font_encoding_issues = false;

    let form_fields = if config.pdf_options.as_ref().is_none_or(|opts| opts.extract_form_fields) {
        let (fields, form_warnings) = crate::pdf::native::forms::extract_form_fields(&mut doc);
        extraction_warnings.extend(form_warnings);
        fields
    } else {
        Vec::new()
    };

    // Issue #66: `/PageLabels` (roman-numeral front matter, per-section
    // numbering, ...). `None` when the document defines none, which is the
    // common case.
    let page_labels = crate::pdf::native::metadata::extract_page_labels_all(&mut doc).unwrap_or_else(|e| {
        tracing::debug!("page label extraction failed: {e}");
        None
    });

    Ok((
        pdf_metadata,
        native_text,
        tables,
        page_contents,
        boundaries,
        pre_rendered_doc,
        has_font_encoding_issues,
        annotation_text_fallback,
        annotations,
        images,
        form_fields,
        extraction_warnings,
        page_labels,
        pdf_page_coordinate_frames,
        outlined_ocr_pages,
    ))
}

/// Join per-page texts, recording each page's byte range in the combined
/// string, faithful to how `extract_text_from_native_document` assembles it:
/// a rendered page marker before each page when `insert_page_markers` is on,
/// otherwise `PAGE_SEPARATOR` between pages. Markers and separators belong
/// to no page.
fn join_pages_with_boundaries(
    pages: &[String],
    page_config: Option<&crate::core::config::PageConfig>,
) -> (String, Vec<crate::types::PageBoundary>) {
    let markers = page_config.filter(|c| c.insert_page_markers);
    let mut content = String::new();
    let mut boundaries = Vec::with_capacity(pages.len());
    for (idx, page_text) in pages.iter().enumerate() {
        if let Some(config) = markers {
            let marker = config.marker_format.replace("{page_num}", &(idx + 1).to_string());
            content.push_str(&marker);
        } else if idx > 0 {
            content.push_str(crate::pdf::native::text::PAGE_SEPARATOR);
        }
        let byte_start = content.len();
        content.push_str(page_text);
        boundaries.push(crate::types::PageBoundary {
            byte_start,
            byte_end: content.len(),
            page_number: idx as u32 + 1,
        });
    }
    (content, boundaries)
}

#[cfg(test)]
mod tests {
    use super::{
        hierarchy_cluster_count, map_native_text_error, needs_structured_extraction, page_has_exact_text_block,
        retain_segments_inside_page_margins, table_stage_failure_warning,
    };
    use crate::core::config::OutputFormat;

    #[test]
    fn native_text_cancellation_remains_a_typed_cancellation() {
        let error = map_native_text_error(crate::pdf::error::PdfError::Cancelled);

        assert!(matches!(error, crate::XbergError::Cancelled));
    }

    // GH#1732: `only_reason_images_were_requested_is_ocr` must return `true` only when no
    // other consumer -- extract_images, captioning, QR codes, inline-image OCR, page rasters,
    // `pdf_options.extract_images`, or styled-HTML rendering -- would ever read image bytes.

    #[test]
    fn should_match_multiline_annotation_as_contiguous_lines_inside_page_text() {
        assert!(page_has_exact_text_block(
            "HEADER\nLINE 1\nLINE 2\nFOOTER",
            "LINE 1\nLINE 2"
        ));
        assert!(!page_has_exact_text_block(
            "HEADER\nLINE 1\nUNRELATED\nLINE 2\nFOOTER",
            "LINE 1\nLINE 2"
        ));
    }

    #[test]
    fn should_use_public_hierarchy_default_for_implicit_structure() {
        let config = crate::core::config::ExtractionConfig::default();

        assert_eq!(
            hierarchy_cluster_count(&config),
            crate::core::config::HierarchyConfig::default().k_clusters
        );
    }

    #[test]
    fn should_filter_reused_hierarchy_segments_by_page_margins() {
        let segment = |text: &str, baseline_y: f32| crate::pdf::hierarchy::SegmentData {
            text: text.to_string(),
            x: 20.0,
            y: baseline_y,
            width: 100.0,
            height: 10.0,
            font_size: 10.0,
            is_bold: false,
            is_italic: false,
            is_monospace: false,
            baseline_y,
            rotation_degrees: 0.0,
            assigned_role: None,
        };
        let mut segments = vec![
            segment("header", 1000.01),
            segment("top boundary", 1000.0),
            segment("body", 500.0),
            segment("bottom boundary", 150.0),
            segment("footer", 149.99),
        ];

        retain_segments_inside_page_margins(
            &mut segments,
            100.0,
            1100.0,
            crate::pdf::native::text::PageMarginFractions {
                top: 0.10,
                bottom: 0.05,
            },
        );

        assert_eq!(
            segments.iter().map(|segment| segment.text.as_str()).collect::<Vec<_>>(),
            ["top boundary", "body", "bottom boundary"]
        );
    }

    /// Characterises `table_stage_failure_warning`'s message format only — it constructs the
    /// error directly rather than driving a real xberg_native_pdf failure, so it does NOT prove any
    /// table-extraction pass can fail or panic. That coverage used to live in
    /// `crates/xberg/tests/issue_603_pdf_table_stage_failure_warning.rs` as
    /// `should_emit_pdf_tables_warning_when_whole_document_table_pass_panics`, driven through the
    /// public API against `test_documents/pdf/total_order_panic_1198_tables_path.pdf`. That
    /// fixture stopped panicking once the `xberg-native` fork picked up upstream commit
    /// `9b0f9c99` ("Fix reading-order sort panics on scanned/malformed PDFs (#807)"), which
    /// replaced the non-transitive pairwise tategaki column comparator with
    /// `sort_vertical_tategaki`, a genuine total order — `9b0f9c99` is an ancestor of the
    /// `xberg-native` 1.0.1 version this crate now consumes. No fixture in `test_documents/`
    /// still reaches a real `xberg_native_pdf` panic through the table-detection stage, so the old test
    /// asserted a panic-recovery warning against code that no longer panics. Do not re-add a
    /// test that asserts a panic on this fixture.
    #[test]
    fn table_stage_failure_warning_formats_the_message_it_promises() {
        let cause = "xberg_native_pdf panicked during table extraction: boom";
        let warning = table_stage_failure_warning("whole-document", &cause);

        assert_eq!(warning.source, "pdf_tables");
        let message = warning.message.as_ref();
        assert_eq!(
            message,
            "whole-document table extraction failed for the document: xberg_native_pdf panicked during table \
             extraction: boom; tables from this pass were skipped"
        );
    }

    /// Regression test for the defective `Custom(_)` catch-all shipped in #1388:
    /// `OutputFormat::FromStr` never rejects an unknown string, so a typo like
    /// `Custom("markdwon")` must not trigger the expensive native hierarchy pass —
    /// it can only ever fall back to plain text (`custom_fallback_to_plain` in
    /// `core/pipeline/format.rs`), so running the full structure pass for it was
    /// pure waste.
    #[test]
    fn should_not_trigger_structured_extraction_for_unregistered_custom_format() {
        let output_format = OutputFormat::Custom("markdwon".to_string());
        assert!(!needs_structured_extraction(false, false, &output_format, false, false));
    }

    /// `DocTags` is a real, always-registered built-in renderer (see
    /// `plugins::registry::renderer::RendererRegistry`) and needs the same
    /// geometry/headings as Markdown, Djot, and HTML.
    #[test]
    fn should_trigger_structured_extraction_for_doctags_format() {
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::DocTags,
            false,
            false
        ));
    }

    /// DOCX is built from the Markdown rendering, so without the structured path
    /// a PDF converted to it has no headings or tables.
    #[cfg(feature = "office")]
    #[test]
    fn should_trigger_structured_extraction_for_docx_format() {
        let output_format = OutputFormat::Custom("docx".to_string());
        assert!(needs_structured_extraction(false, false, &output_format, false, false));
    }

    /// PDF output is laid out from the Markdown rendering too.
    #[test]
    fn should_trigger_structured_extraction_for_pdf_format() {
        let output_format = OutputFormat::Custom("pdf".to_string());
        assert!(needs_structured_extraction(false, false, &output_format, false, false));
    }

    /// The pre-existing markup formats must keep triggering the structured path.
    #[test]
    fn should_trigger_structured_extraction_for_markdown_djot_and_html() {
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::Markdown,
            false,
            false
        ));
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::Djot,
            false,
            false
        ));
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::Html,
            false,
            false
        ));
    }

    /// `Plain` and `Json` must not trigger the structured path on their own.
    #[test]
    fn should_not_trigger_structured_extraction_for_plain_or_json() {
        assert!(!needs_structured_extraction(
            false,
            false,
            &OutputFormat::Plain,
            false,
            false
        ));
        assert!(!needs_structured_extraction(
            false,
            false,
            &OutputFormat::Json,
            false,
            false
        ));
    }

    /// GH#1668: `include_document_structure` alone, with everything else left at
    /// its default (`Plain` output, no hierarchy, no inline OCR, no content
    /// filter), must trigger the structured path on its own. Before this fix a
    /// caller who set only this flag silently got `flat_pdf_document`'s
    /// paragraph-only elements, and the structure tree it asked for came back
    /// holding nothing else.
    #[test]
    fn should_trigger_structured_extraction_for_include_document_structure_alone() {
        assert!(needs_structured_extraction(
            false,
            true,
            &OutputFormat::Plain,
            false,
            false
        ));
    }

    /// Hierarchy, inline-image OCR, and explicit content filtering require the
    /// structured path regardless of output format.
    #[test]
    fn should_trigger_structured_extraction_when_structure_dependent_options_are_enabled() {
        assert!(needs_structured_extraction(
            true,
            false,
            &OutputFormat::Plain,
            false,
            false
        ));
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::Plain,
            true,
            false
        ));
        assert!(needs_structured_extraction(
            false,
            false,
            &OutputFormat::Plain,
            false,
            true
        ));
    }

    #[test]
    fn test_bounding_box_coordinate_conversion() {
        let page_height = 800.0_f64;

        let img_left = 50.0_f64;
        let img_top = 100.0_f64;
        let img_right = 300.0_f64;
        let img_bottom = 150.0_f64;

        let bbox = crate::types::BoundingBox {
            x0: img_left,
            y0: page_height - img_bottom,
            x1: img_right,
            y1: page_height - img_top,
        };

        assert_eq!(bbox.x0, 50.0);
        assert_eq!(bbox.y0, 650.0);
        assert_eq!(bbox.x1, 300.0);
        assert_eq!(bbox.y1, 700.0);
        assert!(bbox.y1 > bbox.y0);
    }

    #[test]
    fn test_bounding_box_coordinate_conversion_different_scales() {
        let page_height = 1000.0_f64;

        let img_left = 100.0_f64;
        let img_top = 50.0_f64;
        let img_right = 600.0_f64;
        let img_bottom = 400.0_f64;

        let bbox = crate::types::BoundingBox {
            x0: img_left,
            y0: page_height - img_bottom,
            x1: img_right,
            y1: page_height - img_top,
        };

        assert_eq!(bbox.x0, 100.0);
        assert_eq!(bbox.y0, 600.0);
        assert_eq!(bbox.x1, 600.0);
        assert_eq!(bbox.y1, 950.0);
        assert_eq!(bbox.y1 - bbox.y0, 350.0);
    }

    #[test]
    fn test_bounding_box_coordinate_conversion_preserves_width() {
        let page_height = 595.0_f64;

        let img_left = 72.0_f64;
        let img_right = 522.0_f64;
        let img_top = 36.0_f64;
        let img_bottom = 300.0_f64;

        let bbox = crate::types::BoundingBox {
            x0: img_left,
            y0: page_height - img_bottom,
            x1: img_right,
            y1: page_height - img_top,
        };

        let expected_width = img_right - img_left;
        let actual_width = bbox.x1 - bbox.x0;
        assert_eq!(actual_width, expected_width);
        assert_eq!(actual_width, 450.0);
    }

    #[test]
    fn test_bounding_box_serialization_round_trip() {
        let original = crate::types::BoundingBox {
            x0: 10.5,
            y0: 20.25,
            x1: 100.75,
            y1: 200.5,
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: crate::types::BoundingBox = serde_json::from_str(&json).unwrap();

        assert_eq!(original, deserialized);
        assert_eq!(deserialized.x0, 10.5);
        assert_eq!(deserialized.y0, 20.25);
        assert_eq!(deserialized.x1, 100.75);
        assert_eq!(deserialized.y1, 200.5);
    }
}
