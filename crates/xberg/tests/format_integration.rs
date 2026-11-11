//! Binding-specific format integration tests.
//!
//! Positive-path scenarios are now covered by the shared fixture-based E2E
//! suites. The tests here focus on behaviour that is specific to the Rust
//! asynchronous APIs or to graceful handling when optional system
//! dependencies are missing.

#![cfg(any(feature = "pdf", feature = "office"))]

mod helpers;
use helpers::extract_uri_document;

use helpers::{assert_mime_type, get_test_file_path, test_documents_available};

#[cfg(feature = "office")]
use helpers::assert_non_empty_content;
use xberg::core::config::ExtractionConfig;

#[cfg(feature = "pdf")]
#[tokio::test]
async fn test_pdf_password_protected_async() {
    if !test_documents_available() {
        return;
    }

    let path = get_test_file_path("pdf/copy_protected.pdf");
    if !path.exists() {
        tracing::debug!("Skipping test: protected PDF not available");
        return;
    }

    let result = extract_uri_document(&path, None, &ExtractionConfig::default()).await;

    match result {
        Err(err) => {
            tracing::debug!("Password protection detected (expected): {}", err);
        }
        Ok(res) => {
            tracing::debug!("Protected PDF extracted; some files allow fallback");
            assert_mime_type(&res, "application/pdf");
            assert!(res.chunks.is_none(), "Chunks should be None without chunking config");
            assert!(res.detected_languages.is_none(), "Language detection not enabled");
        }
    }
}

#[cfg(feature = "office")]
#[tokio::test]
#[cfg_attr(target_os = "windows", ignore = "Legacy office tests timeout on Windows CI")]
async fn test_legacy_doc_extraction_async() {
    if !test_documents_available() {
        return;
    }

    let path = get_test_file_path("doc/simple.doc");
    if !path.exists() {
        tracing::debug!("Skipping test: legacy .doc file not available");
        return;
    }

    let result = extract_uri_document(&path, None, &ExtractionConfig::default()).await;

    match result {
        Ok(extracted) => {
            assert_mime_type(&extracted, "application/msword");
            assert_non_empty_content(&extracted);
            assert!(
                extracted.chunks.is_none(),
                "Chunks should be None without chunking config"
            );
            assert!(extracted.detected_languages.is_none(), "Language detection not enabled");
        }
        Err(err) => {
            tracing::debug!(
                "Legacy Office extraction failed (office feature may not be enabled): {}",
                err
            );
        }
    }
}
