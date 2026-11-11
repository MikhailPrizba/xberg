//! Redaction & anonymisation configuration.
//!
//! When `ExtractionConfig::redaction` is `Some`, the redaction post-processor runs
//! as the Late stage of the pipeline and rewrites `content`, `formatted_content`,
//! every chunk's text, and the textual fields of `entities` / `summary` /
//! `translation` / `page_classifications` using the configured strategy. The
//! original text never appears in the returned `ExtractedDocument`.

use crate::Result;
use crate::types::redaction::{PiiCategory, RedactionStrategy};
use serde::de;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;

/// Configuration for the redaction post-processor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]

pub struct RedactionConfig {
    /// Categories to redact. Empty means "every category supported by the engine."
    #[serde(default)]
    pub categories: HashSet<PiiCategory>,
    /// Strategy applied to every match.
    #[serde(default)]
    pub strategy: RedactionStrategy,
    /// Optional NER backend — required to redact PERSON / ORGANIZATION / LOCATION
    /// categories (the pure-Rust pattern engine only covers regex-detectable PII).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ner: Option<super::ner::NerConfig>,
    /// When `true`, chunk byte ranges are kept consistent with the rewritten content by
    /// adjusting `byte_start` / `byte_end` after replacement. When `false`, chunk byte
    /// ranges still refer to the *original* content offsets — useful when downstream
    /// consumers want to map findings back to the original document.
    #[serde(default = "default_preserve_offsets")]
    pub preserve_offsets: bool,
    /// Arbitrary user-supplied literal terms to redact.
    ///
    /// Each term is treated as a regex hit against the document, surfacing as
    /// `PiiCategory::Custom(label)` in [`RedactionFinding`](crate::types::redaction::RedactionFinding)
    /// where `label` is the per-term label (defaulting to the literal value itself).
    /// Case-insensitive by default; set [`RedactionTerm::case_sensitive`] for exact match.
    ///
    /// Use this when you need to redact tenant-specific tokens (employee IDs,
    /// project codes, internal product names) without writing a custom plugin.
    #[serde(default)]
    pub custom_terms: Vec<RedactionTerm>,
    /// Arbitrary user-supplied regex patterns to redact.
    ///
    /// Same surfacing semantics as [`custom_terms`](Self::custom_terms): each
    /// hit becomes a `PiiCategory::Custom(label)` finding. Patterns are validated
    /// at config-construction time via [`RedactionConfig::validate`].
    #[serde(default)]
    pub custom_patterns: Vec<RedactionPattern>,
    /// Findings supplied inline by an external inspection engine. ~keep
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub findings: Vec<ExternalRedactionFinding>,
    /// Minimum accepted confidence for scored external findings. Findings without a score remain eligible. ~keep
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,
    /// JSON array or JSON Lines file containing external findings. ~keep
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub findings_path: Option<PathBuf>,
    /// Unit used by configured findings that derive their text from offsets. ~keep
    #[serde(default)]
    pub findings_offset_encoding: RedactionOffsetEncoding,
}

/// One finding reported by an external content-inspection engine.
///
/// Unknown fields are ignored, so an engine's raw output can be passed as is.
/// Presidio, AWS Comprehend, Azure Language, and GCP DLP payload fields are
/// accepted as aliases. GCP's ordered likelihood buckets are normalized to
/// evenly spaced scores from `0.0` through `1.0`; `LIKELIHOOD_UNSPECIFIED`
/// maps to `0.5`, matching GCP's documented `POSSIBLE` default. ~keep
#[derive(Debug, Clone, PartialEq, Default, Serialize)]

pub struct ExternalRedactionFinding {
    /// Engine category, surfaced as `PiiCategory::Custom(label)`.
    pub label: String,
    /// Literal value to redact. When absent, it is read from `content` at
    /// `start..end` under the requested [`RedactionOffsetEncoding`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Start offset (inclusive) into `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u32>,
    /// End offset (exclusive) into `content`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<u32>,
    /// Engine confidence in `[0.0, 1.0]`; filtered by [`RedactionConfig::min_score`] when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
}

#[derive(Deserialize)]
struct ExternalRedactionFindingWire {
    #[serde(default, alias = "entity_type", alias = "Type", alias = "category")]
    label: Option<String>,
    #[serde(default, alias = "Text")]
    text: Option<String>,
    #[serde(default, alias = "BeginOffset", alias = "offset")]
    start: Option<VendorOffset>,
    #[serde(default, alias = "EndOffset")]
    end: Option<VendorOffset>,
    #[serde(default, alias = "Score", alias = "confidenceScore")]
    score: Option<f32>,
    #[serde(default)]
    length: Option<VendorOffset>,
    #[serde(default, rename = "infoType")]
    info_type: Option<GcpInfoType>,
    #[serde(default)]
    location: Option<GcpLocation>,
    #[serde(default)]
    likelihood: Option<GcpLikelihood>,
}

#[derive(Deserialize)]
struct GcpInfoType {
    name: Option<String>,
}

#[derive(Deserialize)]
struct GcpLocation {
    #[serde(default, rename = "codepointRange")]
    codepoint_range: Option<GcpCodepointRange>,
}

#[derive(Deserialize)]
struct GcpCodepointRange {
    start: Option<VendorOffset>,
    end: Option<VendorOffset>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum VendorOffset {
    Number(u64),
    String(String),
}

impl VendorOffset {
    fn into_u32<E>(self, field: &str) -> std::result::Result<u32, E>
    where
        E: de::Error,
    {
        let value = match self {
            Self::Number(value) => value,
            Self::String(value) => value
                .parse::<u64>()
                .map_err(|_| E::custom(format!("{field} must be a non-negative integer")))?,
        };
        u32::try_from(value).map_err(|_| E::custom(format!("{field} exceeds the supported u32 range")))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GcpLikelihood {
    LikelihoodUnspecified,
    VeryUnlikely,
    Unlikely,
    Possible,
    Likely,
    VeryLikely,
}

impl GcpLikelihood {
    fn normalized_score(self) -> f32 {
        match self {
            Self::VeryUnlikely => 0.0,
            Self::Unlikely => 0.25,
            Self::LikelihoodUnspecified | Self::Possible => 0.5,
            Self::Likely => 0.75,
            Self::VeryLikely => 1.0,
        }
    }
}

impl<'de> Deserialize<'de> for ExternalRedactionFinding {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let ExternalRedactionFindingWire {
            label,
            text,
            start,
            end,
            score,
            length,
            info_type,
            location,
            likelihood,
        } = ExternalRedactionFindingWire::deserialize(deserializer)?;
        let (nested_start, nested_end) = location
            .and_then(|location| location.codepoint_range)
            .map(|range| (range.start, range.end))
            .unwrap_or_default();

        let label = label
            .or_else(|| info_type.and_then(|info_type| info_type.name))
            .ok_or_else(|| de::Error::missing_field("label"))?;
        let start = start
            .or(nested_start)
            .map(|offset| offset.into_u32::<D::Error>("start"))
            .transpose()?;
        let mut end = end.map(|offset| offset.into_u32::<D::Error>("end")).transpose()?;

        if end.is_none()
            && let Some(length) = length
        {
            let length = length.into_u32::<D::Error>("length")?;
            let offset = start.ok_or_else(|| de::Error::custom("length requires offset"))?;
            end = Some(
                offset
                    .checked_add(length)
                    .ok_or_else(|| de::Error::custom("offset + length exceeds the supported u32 range"))?,
            );
        }
        if end.is_none() {
            end = nested_end
                .map(|offset| offset.into_u32::<D::Error>("end"))
                .transpose()?;
        }

        Ok(Self {
            label,
            text,
            start,
            end,
            score: score.or_else(|| likelihood.map(GcpLikelihood::normalized_score)),
        })
    }
}

impl ExternalRedactionFinding {
    pub(crate) fn validate(&self, location: &str) -> Result<()> {
        let invalid = |reason: &str| Err(crate::XbergError::validation(format!("{location}: {reason}")));
        if self.label.trim().is_empty() {
            return invalid("label is empty");
        }
        if let (Some(start), Some(end)) = (self.start, self.end)
            && start >= end
        {
            return invalid(&format!("start {start} is not before end {end}"));
        }
        match &self.text {
            Some(text) if text.trim().is_empty() => return invalid("text is empty"),
            Some(_) => {}
            None if self.start.is_none() || self.end.is_none() => {
                return invalid("needs either text or both start and end");
            }
            None => {}
        }
        if let Some(score) = self.score
            && !(score.is_finite() && (0.0..=1.0).contains(&score))
        {
            return invalid(&format!("score must be between 0.0 and 1.0, got {score}"));
        }
        Ok(())
    }
}

/// Unit that an external finding's `start` / `end` offsets count in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedactionOffsetEncoding {
    /// UTF-8 byte offsets.
    Utf8Bytes,
    /// Unicode scalar value (code point) offsets, as Presidio reports them.
    #[default]
    UnicodeCodePoints,
    /// UTF-16 code unit offsets.
    Utf16CodeUnits,
}

impl FromStr for RedactionOffsetEncoding {
    type Err = crate::XbergError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "utf8_bytes" => Ok(Self::Utf8Bytes),
            "unicode_code_points" => Ok(Self::UnicodeCodePoints),
            "utf16_code_units" => Ok(Self::Utf16CodeUnits),
            _ => Err(crate::XbergError::validation(format!(
                "unsupported redaction offset encoding `{value}`; expected utf8_bytes, unicode_code_points, or utf16_code_units"
            ))),
        }
    }
}

fn default_preserve_offsets() -> bool {
    true
}

fn default_case_sensitive() -> bool {
    false
}

/// One user-supplied literal term to redact.
///
/// Matched as a regex-escaped substring (so callers do not need to escape
/// metacharacters themselves). Case-insensitive by default — set
/// [`Self::case_sensitive`] to `true` for exact byte-match semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]

pub struct RedactionTerm {
    /// Custom category label surfaced in [`RedactionFinding::category`](crate::types::redaction::RedactionFinding::category).
    pub label: String,
    /// Literal value to match. Regex metacharacters are escaped automatically.
    pub value: String,
    /// When `true`, match the value as-is; otherwise match ASCII-case-insensitively.
    #[serde(default = "default_case_sensitive")]
    pub case_sensitive: bool,
}

impl RedactionTerm {
    /// Build a term whose label is the literal value itself (case-insensitive).
    pub fn literal(value: impl Into<String>) -> Self {
        let v = value.into();
        Self {
            label: v.clone(),
            value: v,
            case_sensitive: false,
        }
    }

    /// Build a term with a custom label.
    pub fn labeled(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: value.into(),
            case_sensitive: false,
        }
    }
}

/// One user-supplied regex pattern to redact.
///
/// The pattern is compiled with the Rust `regex` crate (no look-around). Case
/// sensitivity is encoded in the pattern via the `(?i)` inline flag when
/// [`Self::case_sensitive`] is `false`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]

pub struct RedactionPattern {
    /// Custom category label surfaced in [`RedactionFinding::category`](crate::types::redaction::RedactionFinding::category).
    pub label: String,
    /// Regex pattern (Rust `regex` crate dialect — no look-around).
    pub pattern: String,
    /// When `true`, match case-sensitively; otherwise prepend `(?i)` to the regex.
    #[serde(default = "default_case_sensitive")]
    pub case_sensitive: bool,
}

impl RedactionPattern {
    /// Build a pattern with the given label (case-insensitive by default).
    pub fn labeled(label: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            pattern: pattern.into(),
            case_sensitive: false,
        }
    }
}

impl Default for RedactionConfig {
    fn default() -> Self {
        Self {
            categories: HashSet::new(),
            strategy: RedactionStrategy::default(),
            ner: None,
            preserve_offsets: true,
            custom_terms: Vec::new(),
            custom_patterns: Vec::new(),
            findings: Vec::new(),
            min_score: None,
            findings_path: None,
            findings_offset_encoding: RedactionOffsetEncoding::default(),
        }
    }
}

impl RedactionConfig {
    /// Validate user-supplied terms and patterns at config-construction time.
    ///
    /// Compiles every [`RedactionPattern::pattern`] (with the case-insensitive
    /// inline flag where applicable) and returns the first compilation error so
    /// the caller can reject the config before the redaction pipeline runs.
    /// Pure terms (regex-escaped) cannot fail to compile, but the function
    /// still rejects empty values to avoid degenerate zero-length matches.
    pub fn validate(&self) -> Result<()> {
        if let Some(min_score) = self.min_score
            && !(min_score.is_finite() && (0.0..=1.0).contains(&min_score))
        {
            return Err(crate::XbergError::validation(format!(
                "RedactionConfig.min_score must be between 0.0 and 1.0, got {min_score}"
            )));
        }
        for term in &self.custom_terms {
            if term.value.is_empty() {
                return Err(crate::XbergError::validation(format!(
                    "RedactionConfig.custom_terms[{}]: value is empty",
                    term.label
                )));
            }
        }
        for pattern in &self.custom_patterns {
            if pattern.pattern.is_empty() {
                return Err(crate::XbergError::validation(format!(
                    "RedactionConfig.custom_patterns[{}]: pattern is empty",
                    pattern.label
                )));
            }
            let compiled = if pattern.case_sensitive {
                regex::Regex::new(&pattern.pattern)
            } else {
                regex::Regex::new(&format!("(?i){}", pattern.pattern))
            };
            if let Err(err) = compiled {
                return Err(crate::XbergError::validation(format!(
                    "RedactionConfig.custom_patterns[{}]: invalid regex: {err}",
                    pattern.label
                )));
            }
        }
        for (index, finding) in self.findings.iter().enumerate() {
            finding.validate(&format!("RedactionConfig.findings[{index}]"))?;
        }
        #[cfg(target_arch = "wasm32")]
        if self.findings_path.is_some() {
            return Err(crate::XbergError::validation(
                "RedactionConfig.findings_path is not supported on wasm32; pass findings inline".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::ExternalRedactionFinding;

    #[test]
    fn should_deserialize_raw_azure_language_finding() {
        let finding: ExternalRedactionFinding = serde_json::from_str(
            r#"{
                "category": "Person",
                "text": "Ada",
                "offset": 7,
                "length": 3,
                "confidenceScore": 0.97,
                "warnings": []
            }"#,
        )
        .expect("raw Azure Language finding should deserialize");

        assert_eq!(finding.label, "Person");
        assert_eq!(finding.text.as_deref(), Some("Ada"));
        assert_eq!(finding.start, Some(7));
        assert_eq!(finding.end, Some(10));
        assert_eq!(finding.score, Some(0.97));
        finding
            .validate("Azure finding")
            .expect("Azure finding should validate");
    }

    #[test]
    fn should_deserialize_raw_gcp_dlp_finding() {
        let finding: ExternalRedactionFinding = serde_json::from_str(
            r#"{
                "infoType": {"name": "EMAIL_ADDRESS"},
                "likelihood": "VERY_LIKELY",
                "location": {
                    "codepointRange": {"start": "4", "end": "21"},
                    "byteRange": {"start": "4", "end": "21"}
                },
                "quote": "ada@example.test"
            }"#,
        )
        .expect("raw GCP DLP finding should deserialize");

        assert_eq!(finding.label, "EMAIL_ADDRESS");
        assert_eq!(finding.text, None);
        assert_eq!(finding.start, Some(4));
        assert_eq!(finding.end, Some(21));
        assert_eq!(finding.score, Some(1.0));
        finding.validate("GCP finding").expect("GCP finding should validate");
    }

    #[test]
    fn should_normalize_gcp_likelihood_buckets_monotonically() {
        let cases = [
            ("VERY_UNLIKELY", 0.0),
            ("UNLIKELY", 0.25),
            ("POSSIBLE", 0.5),
            ("LIKELIHOOD_UNSPECIFIED", 0.5),
            ("LIKELY", 0.75),
            ("VERY_LIKELY", 1.0),
        ];

        for (likelihood, expected) in cases {
            let json = format!(
                r#"{{"infoType":{{"name":"PERSON_NAME"}},"likelihood":"{likelihood}","location":{{"codepointRange":{{"start":"0","end":"3"}}}}}}"#
            );
            let finding: ExternalRedactionFinding =
                serde_json::from_str(&json).expect("known GCP likelihood should deserialize");

            assert_eq!(finding.score, Some(expected), "likelihood {likelihood}");
        }
    }

    #[test]
    fn should_preserve_canonical_external_finding_serialization() {
        let finding: ExternalRedactionFinding =
            serde_json::from_str(r#"{"label":"PERSON","text":"Ada","start":2,"end":5,"score":0.8}"#)
                .expect("canonical finding should deserialize");

        assert_eq!(
            serde_json::to_value(finding).expect("canonical finding should serialize"),
            serde_json::json!({
                "label": "PERSON",
                "text": "Ada",
                "start": 2,
                "end": 5,
                "score": 0.8_f32
            })
        );
    }

    #[test]
    fn should_reject_azure_offset_length_overflow() {
        let error =
            serde_json::from_str::<ExternalRedactionFinding>(r#"{"category":"PERSON","offset":4294967295,"length":1}"#)
                .expect_err("overflowing Azure range should fail");

        assert!(
            error
                .to_string()
                .contains("offset + length exceeds the supported u32 range"),
            "unexpected error: {error}"
        );
    }
}
