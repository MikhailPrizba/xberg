//! Unified OCR element types for structured output.
//!
//! This module provides a unified representation of OCR results that preserves
//! all spatial and confidence information from both Tesseract and PaddleOCR backends.
//!
//! # Design Goals
//!
//! - **Full fidelity preservation**: Keep all data from both backends (bounding boxes, confidence scores, rotation)
//! - **Unified API**: Same types work for both Tesseract and PaddleOCR
//! - **Format flexibility**: Support text, markdown, djot, and structured output formats
//! - **Table detection support**: Enable table reconstruction from element geometry

use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;

fn deserialize_quadrilateral_points<'de, D>(deserializer: D) -> Result<Vec<OcrPoint>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let points = Vec::<OcrPoint>::deserialize(deserializer)?;
    if points.len() != 4 {
        return Err(serde::de::Error::invalid_length(
            points.len(),
            &"exactly four quadrilateral points",
        ));
    }
    Ok(points)
}

/// A point in OCR raster pixel coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct OcrPoint {
    /// Horizontal coordinate.
    pub x: u32,
    /// Vertical coordinate.
    pub y: u32,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OcrPointWire {
    Positional((u32, u32)),
    Named { x: u32, y: u32 },
}

// ~keep Deserialize stays hand-written (not derived) so a legacy positional `[x, y]` array from
// pre-migration callers still parses; only Serialize now emits the named object.
impl<'de> Deserialize<'de> for OcrPoint {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(match OcrPointWire::deserialize(deserializer)? {
            OcrPointWire::Positional(point) => point.into(),
            OcrPointWire::Named { x, y } => Self { x, y },
        })
    }
}

impl From<(u32, u32)> for OcrPoint {
    fn from((x, y): (u32, u32)) -> Self {
        Self { x, y }
    }
}

impl From<OcrPoint> for (u32, u32) {
    fn from(point: OcrPoint) -> Self {
        (point.x, point.y)
    }
}

/// Bounding geometry for an OCR element.
///
/// Supports both axis-aligned rectangles (from Tesseract) and 4-point quadrilaterals
/// (from PaddleOCR and rotated text detection).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]

pub enum OcrBoundingGeometry {
    /// Axis-aligned bounding box (typical for Tesseract output).
    Rectangle {
        /// Left x-coordinate in pixels
        left: u32,
        /// Top y-coordinate in pixels
        top: u32,
        /// Width in pixels
        width: u32,
        /// Height in pixels
        height: u32,
    },
    /// 4-point quadrilateral for rotated/skewed text (PaddleOCR).
    ///
    /// Points are in clockwise order starting from top-left:
    /// `[top_left, top_right, bottom_right, bottom_left]`
    Quadrilateral {
        /// Exactly four corner points in clockwise order.
        #[serde(deserialize_with = "deserialize_quadrilateral_points")]
        points: Vec<OcrPoint>,
    },
}

impl Default for OcrBoundingGeometry {
    fn default() -> Self {
        OcrBoundingGeometry::Rectangle {
            left: 0,
            top: 0,
            width: 0,
            height: 0,
        }
    }
}

/// Confidence scores for an OCR element.
///
/// Separates detection confidence (how confident that text exists at this location)
/// from recognition confidence (how confident about the actual text content).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]

pub struct OcrConfidence {
    /// Detection confidence: how confident the OCR engine is that text exists here.
    ///
    /// PaddleOCR provides this as `box_score`, Tesseract doesn't have a direct equivalent.
    /// Range: 0.0 to 1.0 (or None if not available).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub detection: Option<f64>,

    /// Recognition confidence: how confident about the text content.
    ///
    /// Range: 0.0 to 1.0.
    pub recognition: f64,
}

/// Rotation information for an OCR element.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]

pub struct OcrRotation {
    /// Rotation angle in degrees (0, 90, 180, 270 for PaddleOCR).
    pub angle_degrees: f64,

    /// Confidence score for the rotation detection.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Hierarchical level of an OCR element.
///
/// Maps to Tesseract's page segmentation hierarchy and provides
/// equivalent semantics for PaddleOCR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OcrElementLevel {
    /// Individual word
    Word,
    /// Line of text (default for PaddleOCR)
    #[default]
    Line,
    /// Paragraph or text block
    Block,
    /// Page-level element
    Page,
}

/// A unified OCR element representing detected text with full metadata.
///
/// This is the primary type for structured OCR output, preserving all information
/// from both Tesseract and PaddleOCR backends.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]

pub struct OcrElement {
    /// The recognized text content.
    pub text: String,

    /// Bounding geometry (rectangle or quadrilateral).
    pub geometry: OcrBoundingGeometry,

    /// Confidence scores for detection and recognition.
    pub confidence: OcrConfidence,

    /// Hierarchical level (word, line, block, page).
    #[serde(default)]
    pub level: OcrElementLevel,

    /// Rotation information (if detected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<OcrRotation>,

    /// Page number (1-indexed).
    #[serde(default = "default_page_number")]
    pub page_number: u32,

    /// Parent element ID for hierarchical relationships.
    ///
    /// ~keep When hierarchy output is enabled, this resolves to another emitted element's
    /// `backend_metadata["element_id"]` value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,

    /// Backend-specific metadata that doesn't fit the unified schema.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub backend_metadata: HashMap<String, serde_json::Value>,
}

const OCR_ELEMENT_ID_METADATA_KEY: &str = "element_id";

impl OcrElement {
    /// Return this element's hierarchy ID when hierarchy output was requested.
    ///
    /// ~keep The ID is stored in `backend_metadata["element_id"]` for compatibility with the
    /// existing serialized and binding-safe shape of [`OcrElement`].

    pub fn element_id(&self) -> Option<&str> {
        self.backend_metadata
            .get(OCR_ELEMENT_ID_METADATA_KEY)
            .and_then(serde_json::Value::as_str)
    }
}

fn default_page_number() -> u32 {
    1
}

/// Configuration for OCR element extraction.
///
/// Controls how OCR elements are extracted and filtered.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct OcrElementConfig {
    /// Whether to include OCR elements in the extraction result.
    ///
    /// When true, the `ocr_elements` field in `ExtractedDocument` will be populated.
    #[serde(default)]
    pub include_elements: bool,

    /// Minimum hierarchical level to include.
    ///
    /// Elements below this level (e.g., words when min_level is Line) will be excluded.
    #[serde(default)]
    pub min_level: OcrElementLevel,

    /// Minimum recognition confidence threshold (0.0-1.0).
    ///
    /// Elements with confidence below this threshold will be filtered out.
    #[serde(default)]
    pub min_confidence: f64,

    /// Whether to build hierarchical relationships between elements.
    ///
    /// ~keep When true, emitted elements receive an `element_id` metadata value and `parent_id`
    /// references are populated only when a spatially containing parent is also emitted.
    #[serde(default)]
    pub build_hierarchy: bool,
}

#[cfg(feature = "ocr-pipeline")]
impl OcrElementConfig {
    pub(crate) fn select_elements(&self, elements: &[OcrElement]) -> Vec<OcrElement> {
        if !self.include_elements {
            return Vec::new();
        }

        let minimum_rank = element_level_rank(self.min_level);
        let mut selected = elements
            .iter()
            .filter(|element| element.confidence.recognition >= self.min_confidence)
            .filter(|element| element_level_rank(element.level) >= minimum_rank)
            .cloned()
            .collect::<Vec<_>>();

        for element in &mut selected {
            element.parent_id = None;
            element.backend_metadata.remove(OCR_ELEMENT_ID_METADATA_KEY);
        }
        if !self.build_hierarchy {
            return selected;
        }

        let element_ids = selected
            .iter()
            .enumerate()
            .map(|(index, element)| format!("ocr-p{}-e{}", element.page_number, index + 1))
            .collect::<Vec<_>>();
        for (element, element_id) in selected.iter_mut().zip(&element_ids) {
            element.backend_metadata.insert(
                OCR_ELEMENT_ID_METADATA_KEY.to_string(),
                serde_json::Value::String(element_id.clone()),
            );
        }
        for child_index in 0..selected.len() {
            if let Some(parent_index) = hierarchy_parent_index(&selected, child_index) {
                selected[child_index].parent_id = Some(element_ids[parent_index].clone());
            }
        }
        selected
    }
}

#[cfg(any(feature = "ocr-pipeline", feature = "heuristics"))]
pub(crate) fn element_level_rank(level: OcrElementLevel) -> u8 {
    match level {
        OcrElementLevel::Word => 0,
        OcrElementLevel::Line => 1,
        OcrElementLevel::Block => 2,
        OcrElementLevel::Page => 3,
    }
}

#[cfg(feature = "ocr-pipeline")]
fn hierarchy_parent_index(elements: &[OcrElement], child_index: usize) -> Option<usize> {
    let child = &elements[child_index];
    let child_rank = element_level_rank(child.level);
    let child_bounds = geometry_bounds(&child.geometry)?;
    elements
        .iter()
        .enumerate()
        .filter(|(_, candidate)| candidate.page_number == child.page_number)
        .filter_map(|(index, candidate)| {
            let candidate_rank = element_level_rank(candidate.level);
            if candidate_rank <= child_rank {
                return None;
            }
            let candidate_bounds = geometry_bounds(&candidate.geometry)?;
            bounds_contain(candidate_bounds, child_bounds).then_some((
                index,
                candidate_rank - child_rank,
                candidate_bounds.2 * candidate_bounds.3,
            ))
        })
        .min_by_key(|(index, rank_distance, area)| (*rank_distance, *area, *index))
        .map(|(index, _, _)| index)
}

#[cfg(feature = "ocr-pipeline")]
fn geometry_bounds(geometry: &OcrBoundingGeometry) -> Option<(u64, u64, u64, u64)> {
    match geometry {
        OcrBoundingGeometry::Rectangle {
            left,
            top,
            width,
            height,
        } => (*width > 0 && *height > 0).then_some((
            u64::from(*left),
            u64::from(*top),
            u64::from(*width),
            u64::from(*height),
        )),
        OcrBoundingGeometry::Quadrilateral { points } => {
            let left = u64::from(points.iter().map(|point| point.x).min()?);
            let top = u64::from(points.iter().map(|point| point.y).min()?);
            let right = u64::from(points.iter().map(|point| point.x).max()?);
            let bottom = u64::from(points.iter().map(|point| point.y).max()?);
            (right > left && bottom > top).then_some((left, top, right - left, bottom - top))
        }
    }
}

#[cfg(feature = "ocr-pipeline")]
fn bounds_contain(parent: (u64, u64, u64, u64), child: (u64, u64, u64, u64)) -> bool {
    parent.0 <= child.0
        && parent.1 <= child.1
        && parent.0 + parent.2 >= child.0 + child.2
        && parent.1 + parent.3 >= child.1 + child.3
}

#[cfg(test)]
mod binding_value_serde_tests {
    use super::{OcrBoundingGeometry, OcrPoint};
    use serde_json::json;

    #[test]
    fn should_still_accept_legacy_positional_array_for_ocr_point() {
        let legacy = json!([10, 20]);
        let positional: OcrPoint = serde_json::from_value(legacy).expect("legacy OCR point must deserialize");

        assert_eq!(positional, OcrPoint { x: 10, y: 20 });
    }

    #[test]
    fn should_serialize_ocr_point_as_named_object() {
        let named: OcrPoint =
            serde_json::from_value(json!({"x": 10, "y": 20})).expect("named OCR point must deserialize");

        assert_eq!(named, OcrPoint { x: 10, y: 20 });
        assert_eq!(
            serde_json::to_value(named).expect("named OCR point must serialize"),
            json!({"x": 10, "y": 20})
        );
    }

    #[test]
    fn should_still_accept_legacy_positional_array_for_quadrilateral_points() {
        let legacy = json!({
            "type": "quadrilateral",
            "points": [[10, 20], [100, 22], [98, 70], [8, 68]]
        });
        let geometry: OcrBoundingGeometry = serde_json::from_value(legacy).expect("legacy geometry must deserialize");
        let OcrBoundingGeometry::Quadrilateral { points } = &geometry else {
            panic!("expected quadrilateral geometry");
        };

        assert_eq!(points[0], OcrPoint { x: 10, y: 20 });
        assert_eq!(points[1], OcrPoint { x: 100, y: 22 });
        assert_eq!(points[2], OcrPoint { x: 98, y: 70 });
        assert_eq!(points[3], OcrPoint { x: 8, y: 68 });
    }

    #[test]
    fn should_serialize_quadrilateral_points_as_named_objects() {
        let geometry: OcrBoundingGeometry = serde_json::from_value(json!({
            "type": "quadrilateral",
            "points": [
                {"x": 10, "y": 20},
                {"x": 100, "y": 22},
                {"x": 98, "y": 70},
                {"x": 8, "y": 68}
            ]
        }))
        .expect("named geometry must deserialize");

        assert_eq!(
            serde_json::to_value(geometry).expect("geometry must serialize"),
            json!({
                "type": "quadrilateral",
                "points": [
                    {"x": 10, "y": 20},
                    {"x": 100, "y": 22},
                    {"x": 98, "y": 70},
                    {"x": 8, "y": 68}
                ]
            })
        );
    }

    #[test]
    fn should_reject_quadrilateral_with_invalid_point_count() {
        let error = serde_json::from_value::<OcrBoundingGeometry>(json!({
            "type": "quadrilateral",
            "points": [[0, 0], [10, 0], [10, 10]]
        }))
        .expect_err("quadrilateral with three points must be rejected");

        assert!(
            error.to_string().contains("exactly four quadrilateral points"),
            "unexpected deserialization error: {error}"
        );
    }
}
