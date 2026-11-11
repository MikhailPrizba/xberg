//! Startup validation for plugin registries.
//!
//! This module provides diagnostics and health checks for plugins
//! at server startup, helping operators diagnose issues in containerized
//! environments like Kubernetes.

/// Plugin health status information.
///
/// Contains diagnostic information about registered plugins for each type.

#[derive(Debug, Clone)]
pub struct PluginHealthStatus {
    /// Number of registered OCR backends
    pub ocr_backends_count: usize,
    /// Names of registered OCR backends
    pub ocr_backends: Vec<String>,
    /// Number of registered document extractors
    pub extractors_count: usize,
    /// Names of registered document extractors
    pub extractors: Vec<String>,
    /// Number of registered post-processors
    pub post_processors_count: usize,
    /// Names of registered post-processors
    pub post_processors: Vec<String>,
    /// Number of registered validators
    pub validators_count: usize,
    /// Names of registered validators
    pub validators: Vec<String>,
}
