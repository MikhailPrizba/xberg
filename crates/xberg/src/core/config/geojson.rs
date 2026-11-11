//! GeoJSON extraction configuration.

use serde::{Deserialize, Serialize};

/// Configuration for GeoJSON extraction.
///
/// GeoJSON coordinates can dominate extraction output and duplicate large geometry
/// payloads in rendered content and metadata. The default therefore emits a bounded
/// aggregate summary. Set [`Self::include_full_coordinates`] only when callers need
/// every coordinate in the extracted text and accept output proportional to the input.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]

pub struct GeoJsonExtractionConfig {
    /// Include every coordinate in rendered content and `flattened_fields` metadata.
    ///
    /// Defaults to `false`. When false, extraction reports feature, property, geometry,
    /// position-count, bounds, truncation, and discarded-category metadata and emits a
    /// `ProcessingWarning` for every GeoJSON input.
    pub include_full_coordinates: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_coordinates_are_disabled_by_default_and_in_serde() {
        assert!(!GeoJsonExtractionConfig::default().include_full_coordinates);
        let config: GeoJsonExtractionConfig = serde_json::from_str("{}").unwrap();
        assert!(!config.include_full_coordinates);
    }

    #[test]
    fn extraction_config_deserializes_explicit_full_coordinate_opt_in() {
        let config: crate::ExtractionConfig =
            serde_json::from_str(r#"{"geojson":{"include_full_coordinates":true}}"#).unwrap();

        assert!(
            config
                .geojson
                .expect("GeoJSON configuration must be reachable from ExtractionConfig")
                .include_full_coordinates
        );
    }
}
