//! Helper functions and utilities for extraction operations.
//!
//! This module provides shared utilities used across extraction modules.

use crate::Result;
use crate::plugins::InternalDocumentExtractor;
use std::sync::Arc;

/// Get an extractor from the registry.
///
/// This function acquires the registry read lock and retrieves the appropriate
/// extractor for the given MIME type.
///
/// # Performance
///
/// RwLock read + HashMap lookup is ~100ns, fast enough without caching.
/// Removed thread-local cache to avoid Tokio work-stealing scheduler issues.
pub(in crate::core::extractor) fn get_extractor(mime_type: &str) -> Result<(Arc<dyn InternalDocumentExtractor>, bool)> {
    let registry = crate::plugins::registry::get_document_extractor_registry();
    let registry_read = registry.read();
    let extractor = registry_read.get_registered(mime_type)?;
    let is_builtin = extractor.is_builtin();
    let extractor: Arc<dyn InternalDocumentExtractor> = Arc::new(extractor);

    Ok((extractor, is_builtin))
}
