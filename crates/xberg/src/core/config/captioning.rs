//! VLM image-captioning configuration.
//!
//! When `ExtractionConfig::captioning` is `Some`, the captioning post-processor runs at
//! the Middle stage, iterates `ExtractedDocument::images`, and populates
//! [`ExtractedImage::caption`](crate::types::ExtractedImage::caption) for each image whose
//! pixel area exceeds `min_image_area`.

use serde::{Deserialize, Serialize};

/// ~keep: How a generated image caption interacts with existing alternate text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]

pub enum CaptionAltTextMode {
    /// ~keep: Keep existing alternate text, using the caption only when it is absent.
    #[default]
    Preserve,
    /// ~keep: Join existing alternate text and the generated caption with `: `.
    Combine,
    /// ~keep: Use the generated caption even when alternate text is present.
    Replace,
}

/// Configuration for the VLM captioning post-processor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]

pub struct CaptioningConfig {
    /// LLM configuration used for the VLM call.
    pub llm: super::llm::LlmConfig,
    /// Optional custom caption prompt. `None` uses the default `RegionKind::Caption`
    /// prompt that ships with `crate::llm::region_extractor`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Skip images whose `width * height` is below this threshold (in pixels).
    /// Default `1_000` filters out icons and decorations.
    #[serde(default = "CaptioningConfig::default_min_image_area")]
    pub min_image_area: u32,
    /// ~keep: Controls whether generated captions preserve, combine with, or replace
    /// existing image alternate text.

    #[serde(default)]
    pub alt_text: CaptionAltTextMode,
}

impl CaptioningConfig {
    /// Default [`Self::min_image_area`]: 1000 px.
    ///
    /// Public and on the type rather than a free private `fn` because generated bindings
    /// have to call it to reproduce the default, and a private one is out of their reach.
    pub fn default_min_image_area() -> u32 {
        1_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alt_text_defaults_to_preserve_when_omitted() {
        let config: CaptioningConfig = serde_json::from_value(serde_json::json!({
            "llm": { "model": "openai/gpt-4o-mini" },
            "prompt": null,
            "min_image_area": 1000
        }))
        .expect("captioning config should deserialize");

        assert_eq!(config.alt_text, CaptionAltTextMode::Preserve);
    }

    #[test]
    fn alt_text_accepts_all_supported_modes() {
        for (wire_value, expected) in [
            ("preserve", CaptionAltTextMode::Preserve),
            ("combine", CaptionAltTextMode::Combine),
            ("replace", CaptionAltTextMode::Replace),
        ] {
            let config: CaptioningConfig = serde_json::from_value(serde_json::json!({
                "llm": { "model": "openai/gpt-4o-mini" },
                "alt_text": wire_value
            }))
            .expect("supported alt_text mode should deserialize");

            assert_eq!(config.alt_text, expected);
        }
    }
}
