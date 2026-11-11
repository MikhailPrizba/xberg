//! Serializable text-enrichment transport types.
//!
//! This module defines request, result, and lifecycle schemas for external
//! enrichment workers. It does not execute enrichment. Use [`crate::enrich`]
//! to run Xberg's post-extraction enrichment stages.
//!
//! # Types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`EnrichOptions`] | Which enrichment passes to run on a piece of text |
//! | [`EnrichResult`] | Structured output from a completed enrichment pass |
//! | [`EnrichStatus`] | Tagged-union status for async enrichment pipelines |
//!

use serde::{Deserialize, Serialize};

use crate::types::entity::Entity;

/// Which enrichment passes to run on a piece of text.
///
/// All fields default to `false` / empty so callers can opt in precisely.
///
/// # Examples
///
/// ```
/// use xberg::enrichment::EnrichOptions;
///
/// let opts = EnrichOptions {
///     keywords: true,
///     ..Default::default()
/// };
/// assert!(opts.keywords);
/// assert!(!opts.entities);
/// assert!(opts.labels.is_empty());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]

pub struct EnrichOptions {
    /// Run keyword extraction on the input text.
    ///
    /// When `true`, the enrichment backend identifies the most salient terms
    /// and returns them in [`EnrichResult::keywords`].
    #[serde(default)]
    pub keywords: bool,

    /// Run named-entity recognition (NER) on the input text.
    ///
    /// When `true`, the enrichment backend identifies named entities
    /// (persons, organisations, locations, etc.) and returns them in
    /// [`EnrichResult::entities`].
    #[serde(default)]
    pub entities: bool,

    /// Custom labels to pass through to the result without modification.
    ///
    /// These are caller-supplied tags that the enrichment pipeline
    /// propagates verbatim into [`EnrichResult::labels`]. Useful for
    /// attaching project- or document-level metadata to every enrichment
    /// result.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

/// Structured output produced by a completed enrichment pass.
///
/// Fields are populated only when the corresponding [`EnrichOptions`] flag was set.
///
/// # Examples
///
/// ```
/// use xberg::enrichment::EnrichResult;
///
/// let result = EnrichResult::default();
/// assert!(result.keywords.is_empty());
/// assert!(result.entities.is_empty());
/// assert!(result.labels.is_empty());
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]

pub struct EnrichResult {
    /// Salient terms extracted from the text.
    ///
    /// Populated when [`EnrichOptions::keywords`] was `true`. The ordering is
    /// backend-defined (typically by descending relevance score).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,

    /// Named entities found in the text.
    ///
    /// Populated when [`EnrichOptions::entities`] was `true`. Uses the shared
    /// OSS entity schema ([`Entity`] / [`EntityCategory`](crate::types::entity::EntityCategory))
    /// so consumers can pattern-match on entity categories without JSON gymnastics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<Entity>,

    /// Caller-supplied labels echoed from [`EnrichOptions::labels`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<String>,
}

/// Async lifecycle status for an enrichment job.
///
/// Intended for use with any polling or event-driven pipeline that needs
/// to track whether enrichment has completed, succeeded, or failed.
///
/// # Serialisation
///
/// Uses an internally-tagged `"status"` field with `snake_case` variants:
///
/// ```json
/// { "status": "pending" }
/// { "status": "completed", "result": { ... } }
/// { "status": "failed", "error": "text too large" }
/// ```
///
/// # Examples
///
/// ```
/// use xberg::enrichment::{EnrichStatus, EnrichResult};
///
/// let s = EnrichStatus::Pending;
/// let json = serde_json::to_value(&s).unwrap();
/// assert_eq!(json["status"], "pending");
///
/// let s = EnrichStatus::Completed { result: EnrichResult::default() };
/// let json = serde_json::to_value(&s).unwrap();
/// assert_eq!(json["status"], "completed");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EnrichStatus {
    /// Job submitted; processing has not yet started or is in progress.
    Pending,

    /// Processing completed successfully.
    Completed {
        /// The enrichment output.
        result: EnrichResult,
    },

    /// Processing failed.
    Failed {
        /// Human-readable error message describing the failure reason.
        error: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::entity::{Entity, EntityCategory};

    #[test]
    fn enrich_options_defaults_are_all_off() {
        let opts = EnrichOptions::default();
        assert!(!opts.keywords, "keywords should default to false");
        assert!(!opts.entities, "entities should default to false");
        assert!(opts.labels.is_empty(), "labels should default to empty");
    }

    #[test]
    fn enrich_options_roundtrip_all_fields() {
        let opts = EnrichOptions {
            keywords: true,
            entities: true,
            labels: vec!["tag-a".to_string(), "tag-b".to_string()],
        };
        let json = serde_json::to_string(&opts).expect("serialize");
        let decoded: EnrichOptions = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(opts, decoded);
    }

    #[test]
    fn enrich_options_labels_omitted_when_empty() {
        let opts = EnrichOptions {
            keywords: true,
            entities: false,
            labels: vec![],
        };
        let json = serde_json::to_value(&opts).expect("serialize");
        assert!(json.get("labels").is_none(), "empty labels should be omitted from JSON");
    }

    #[test]
    fn enrich_result_defaults_are_empty() {
        let r = EnrichResult::default();
        assert!(r.keywords.is_empty());
        assert!(r.entities.is_empty());
        assert!(r.labels.is_empty());
    }

    #[test]
    fn enrich_result_roundtrip_with_entities() {
        let result = EnrichResult {
            keywords: vec!["rust".to_string(), "serde".to_string()],
            entities: vec![Entity {
                category: EntityCategory::Person,
                text: "Alice".to_string(),
                start: 0,
                end: 5,
                confidence: Some(0.95),
            }],
            labels: vec!["doc-type:invoice".to_string()],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        let decoded: EnrichResult = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(result, decoded);
    }

    #[test]
    fn enrich_result_empty_vecs_omitted_from_json() {
        let result = EnrichResult::default();
        let json = serde_json::to_value(&result).expect("serialize");
        assert!(json.get("keywords").is_none());
        assert!(json.get("entities").is_none());
        assert!(json.get("labels").is_none());
    }

    #[test]
    fn enrich_status_pending_serialises_tag() {
        let json = serde_json::to_value(EnrichStatus::Pending).expect("serialize");
        assert_eq!(json["status"], "pending");
        assert_eq!(
            json.as_object().unwrap().len(),
            1,
            "Pending should only have the status tag"
        );
    }

    #[test]
    fn enrich_status_completed_roundtrip() {
        let status = EnrichStatus::Completed {
            result: EnrichResult {
                keywords: vec!["hello".to_string()],
                entities: vec![],
                labels: vec![],
            },
        };
        let json = serde_json::to_string(&status).expect("serialize");
        let decoded: EnrichStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, decoded);
    }

    #[test]
    fn enrich_status_completed_tag_value() {
        let status = EnrichStatus::Completed {
            result: EnrichResult::default(),
        };
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json["status"], "completed");
    }

    #[test]
    fn enrich_status_failed_roundtrip() {
        let status = EnrichStatus::Failed {
            error: "text too large".to_string(),
        };
        let json = serde_json::to_string(&status).expect("serialize");
        let decoded: EnrichStatus = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(status, decoded);
    }

    #[test]
    fn enrich_status_failed_tag_and_error_field() {
        let status = EnrichStatus::Failed {
            error: "timeout".to_string(),
        };
        let json = serde_json::to_value(&status).expect("serialize");
        assert_eq!(json["status"], "failed");
        assert_eq!(json["error"], "timeout");
    }

    #[test]
    fn enrich_status_pending_deserialises_from_json() {
        let json = r#"{"status":"pending"}"#;
        let status: EnrichStatus = serde_json::from_str(json).expect("deserialize");
        assert_eq!(status, EnrichStatus::Pending);
    }
}
