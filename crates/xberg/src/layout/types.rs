use serde::{Deserialize, Serialize};
use std::fmt;

/// Bounding box in original image coordinates (x1, y1) top-left, (x2, y2) bottom-right.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    /// Left edge (x-coordinate of the top-left corner).
    pub x1: f32,
    /// Top edge (y-coordinate of the top-left corner).
    pub y1: f32,
    /// Right edge (x-coordinate of the bottom-right corner).
    pub x2: f32,
    /// Bottom edge (y-coordinate of the bottom-right corner).
    pub y2: f32,
}

impl fmt::Display for BBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.1}, {:.1}, {:.1}, {:.1}]", self.x1, self.y1, self.x2, self.y2)
    }
}

/// The 18 canonical document layout classes.
///
/// All model backends (RT-DETR, YOLO, etc.) map their native class IDs
/// to this shared set. Models with fewer classes (DocLayNet: 11, PubLayNet: 5)
/// map to the closest equivalent.
///
/// Wire format is snake_case in all serializers (JSON, TOML, YAML).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LayoutClass {
    /// Figure or table caption text.
    #[default]
    Caption,
    /// Chart or graph visualization.
    Chart,
    /// Footnote or endnote text.
    Footnote,
    /// Mathematical formula or equation.
    Formula,
    /// A single item in a bulleted or numbered list.
    ListItem,
    /// Running footer at the bottom of a page.
    PageFooter,
    /// Running header at the top of a page.
    PageHeader,
    /// Image, chart, or other graphical element.
    Picture,
    /// Section heading.
    SectionHeader,
    /// Data table.
    Table,
    /// Body text paragraph.
    Text,
    /// Document or chapter title.
    Title,
    /// Table of contents or index.
    DocumentIndex,
    /// Source code block.
    Code,
    /// Checkbox in selected state.
    CheckboxSelected,
    /// Checkbox in unselected state.
    CheckboxUnselected,
    /// Form field or form element.
    Form,
    /// Key-value pair region (e.g. label + value in a form).
    KeyValueRegion,
}

impl LayoutClass {
    /// Snake_case wire-format name (matches serde output).
    ///
    /// Equivalent to `format!("{self}")` but returns a `&'static str` slice
    /// for callers that need to avoid allocations.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Caption => "caption",
            Self::Chart => "chart",
            Self::Footnote => "footnote",
            Self::Formula => "formula",
            Self::ListItem => "list_item",
            Self::PageFooter => "page_footer",
            Self::PageHeader => "page_header",
            Self::Picture => "picture",
            Self::SectionHeader => "section_header",
            Self::Table => "table",
            Self::Text => "text",
            Self::Title => "title",
            Self::DocumentIndex => "document_index",
            Self::Code => "code",
            Self::CheckboxSelected => "checkbox_selected",
            Self::CheckboxUnselected => "checkbox_unselected",
            Self::Form => "form",
            Self::KeyValueRegion => "key_value_region",
        }
    }
}

impl fmt::Display for LayoutClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single layout detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDetection {
    /// Detected layout class (e.g. `Table`, `Text`, `Title`).
    pub class_name: LayoutClass,
    /// Detection confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Bounding box in image pixel coordinates.
    pub bbox: BBox,
}

impl LayoutDetection {
    /// Deprecated: use the `class_name` field directly.
    #[deprecated(since = "1.1.0", note = "Use `class_name` field instead")]
    pub fn class(&self) -> LayoutClass {
        self.class_name
    }
}

impl fmt::Display for LayoutDetection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:20} conf={:.3}  bbox={}",
            self.class_name.as_str(),
            self.confidence,
            self.bbox
        )
    }
}

/// Pre-computed table markdown for a table detection region.
///
/// Produced by the TATR-based table structure recognizer and surfaced as part of
/// layout-aware OCR results.  The struct lives here (under `layout-types`, pure-Rust)
/// so that consumers who do not enable `layout-detection` (ORT) can still reference
/// the type in their own code.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecognizedTable {
    /// Detection bbox that this table corresponds to (for matching).
    pub detection_bbox: BBox,
    /// Table cells as a 2D vector (rows × columns).
    pub cells: Vec<Vec<String>>,
    /// Rendered markdown table.
    pub markdown: String,
}

/// Page-level detection result containing all detections and page metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Page width in pixels (as seen by the model).
    pub page_width: u32,
    /// Page height in pixels (as seen by the model).
    pub page_height: u32,
    /// All layout detections on this page after postprocessing.
    pub detections: Vec<LayoutDetection>,
}
