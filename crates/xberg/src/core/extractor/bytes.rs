//! Byte array extraction operations.
//!
//! This module handles extraction from in-memory byte arrays, including:
//! - MIME type validation
//! - Legacy format conversion (DOC, PPT)
//! - Extraction pipeline orchestration

use crate::Result;
#[cfg(not(feature = "office"))]
use crate::XbergError;
use crate::core::config::ExtractionConfig;
use crate::core::mime::{LEGACY_POWERPOINT_MIME_TYPE, LEGACY_WORD_MIME_TYPE};
use crate::types::ExtractedDocument;

pub(crate) async fn extract_bytes(
    content: &[u8],
    mime_type: &str,
    config: &ExtractionConfig,
) -> Result<ExtractedDocument> {
    // `token.cancel()` below needs a token to signal, but `config.cancel_token` is
    // `None` on every binding-driven and CLI-driven call (see
    // `ExtractionConfig::ensure_cancel_token`) — install an internal fallback so a
    // timeout actually stops the extraction instead of merely returning
    // `Err(Timeout)` while the spawned work keeps running. A caller-supplied token
    // is always left untouched. Gated identically to the timeout block below: on
    // wasm32 / without `tokio-runtime` there is no timeout to enforce, so no token
    // is ever needed.
    #[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
    let owned_config_with_cancel_token;
    #[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
    let config: &ExtractionConfig = if config.extraction_timeout_secs.is_some() && config.cancel_token.is_none() {
        let mut owned = config.clone();
        owned.ensure_cancel_token();
        owned_config_with_cancel_token = owned;
        &owned_config_with_cancel_token
    } else {
        config
    };

    let extraction_future = Box::pin(run_byte_extraction(content, mime_type, config));

    // without a JS/WASI shim), which aborts the whole module with an uncatchable
    // `unreachable` trap. `tokio-runtime` can be enabled transitively on that target
    // (e.g. by `layout-tract` inside `wasm-target`), so gating on the feature alone is
    // not enough — explicitly exclude wasm32 here, matching `run_timed_extraction` in
    #[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
    let result = if let Some(secs) = config.extraction_timeout_secs {
        let start = std::time::Instant::now();
        match tokio::time::timeout(std::time::Duration::from_secs(secs), extraction_future).await {
            Ok(inner) => inner,
            Err(_elapsed) => {
                if let Some(ref token) = config.cancel_token {
                    token.cancel();
                }
                Err(crate::XbergError::Timeout {
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    limit_ms: secs * 1000,
                })
            }
        }
    } else {
        extraction_future.await
    };

    #[cfg(any(not(feature = "tokio-runtime"), target_arch = "wasm32"))]
    let result = {
        // Without a usable tokio timer (no 'tokio-runtime' feature, or the WASM build,
        // where `std::time::Instant::now()` panics) there is no timer to enforce a
        // timeout, but the default ExtractionConfig sets extraction_timeout_secs, so
        // erroring here would reject every default call. Ignore the unenforceable
        // limit and run the extraction instead. ~keep
        if config.extraction_timeout_secs.is_some() {
            tracing::debug!(
                "extraction_timeout_secs is ignored on this target (no usable tokio timer); running without a timeout"
            );
        }
        extraction_future.await
    };

    result
}

/// Resolve the MIME type extraction should run under.
///
/// A caller-supplied `application/octet-stream` carries no information, so the
/// bytes are sniffed instead; anything else is validated as given.
fn resolve_validated_mime(content: &[u8], mime_type: &str, config: &ExtractionConfig) -> Result<String> {
    use crate::core::mime;

    let validated_mime = if mime::is_generic_mime_type(mime_type) {
        {
            let _ = config;
            mime::detect_mime_type_from_bytes(content)?
        }
    } else {
        mime::validate_mime_type(mime_type)?
    };

    Ok(validated_mime)
}

/// Validate the configuration, resolve the MIME type, and run the extractor chain.
///
/// Split out of [`extract_bytes`] so the timeout and cancellation wrapper there stays
/// readable; the returned future is still only polled at the original await point.
async fn run_byte_extraction(content: &[u8], mime_type: &str, config: &ExtractionConfig) -> Result<ExtractedDocument> {
    if config.force_ocr && config.effective_disable_ocr() {
        return Err(crate::XbergError::Validation {
            message: "force_ocr and disable_ocr cannot both be true".to_string(),
            source: None,
        });
    }

    if matches!(
        config.ocr_strategy,
        crate::core::config::OcrStrategy::ScannedPages { .. }
    ) && config.effective_disable_ocr()
    {
        return Err(crate::XbergError::Validation {
            message: "ocr_strategy selects scanned pages for OCR, but disable_ocr is true".to_string(),
            source: None,
        });
    }

    let validated_mime = resolve_validated_mime(content, mime_type, config)?;

    #[cfg(not(feature = "office"))]
    match validated_mime.as_str() {
        LEGACY_WORD_MIME_TYPE => {
            return Err(XbergError::UnsupportedFormat(
                "Legacy Word extraction requires the `office` feature".to_string(),
            ));
        }
        LEGACY_POWERPOINT_MIME_TYPE => {
            return Err(XbergError::UnsupportedFormat(
                "Legacy PowerPoint extraction requires the `office` feature".to_string(),
            ));
        }
        _ => {}
    }

    #[cfg(feature = "office")]
    {
        let _ = LEGACY_WORD_MIME_TYPE;
        let _ = LEGACY_POWERPOINT_MIME_TYPE;
    }

    Box::pin(super::file::extract_bytes_with_extractor(
        content,
        &validated_mime,
        config,
    ))
    .await
}

#[cfg(test)]
mod declared_type_tests;
