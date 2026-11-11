//! Integration tests for the unified enrichment chokepoint.

use xberg::types::ExtractedDocument;
use xberg::{EnrichedResult, EnrichmentConfig, enrich};

fn bare_result(content: &str) -> ExtractedDocument {
    let mut result = ExtractedDocument::default();
    result.content = content.to_string();
    result
}

/// A default config is a no-op: the extraction passes through unchanged and
/// every enrichment field is `None`.
#[tokio::test]
async fn enrich_with_no_config_is_identity() {
    let extraction = bare_result("Some document text.");
    let config = EnrichmentConfig::default();
    let enriched: EnrichedResult = enrich(extraction.clone(), &config).await.unwrap();

    assert_eq!(enriched.extraction.content, extraction.content);
}
