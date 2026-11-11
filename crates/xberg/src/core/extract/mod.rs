//! Unified public extraction API.
//!
//! These functions are the stable, binding-generated public surface. Their
//! signatures must remain byte-identical (they are scanned by the alef binding
//! generator). The implementation delegates to a process-global default
//! [`crate::engine::Engine`]; the extraction internals live in
//! [`crate::engine`] and are a pure refactor of what previously lived here.

use std::sync::LazyLock;

use crate::Result;
use crate::core::config::{ExtractInput, ExtractInputWithOptions, ExtractionConfig, ExtractionResult};

/// Process-global default engine backing the free `extract` / `extract_batch`
/// functions. Construction is cheap and side-effect free.
static DEFAULT_ENGINE: LazyLock<crate::engine::Engine> = LazyLock::new(crate::engine::Engine::new_default);

/// Extract content from a single bytes or URI input.
pub async fn extract(input: ExtractInput, config: &ExtractionConfig) -> Result<ExtractionResult> {
    DEFAULT_ENGINE.extract(input, config).await
}

/// Extract one bytes input and redact a JSON array or JSON Lines payload from an external inspection engine.
///
/// The payload is parsed in Rust so vendor aliases and nested fields remain intact across language bindings.
/// `offset_encoding` defaults to `unicode_code_points` and `max_findings` defaults to 10,000 when omitted.
/// Unknown encodings return a validation error.
#[cfg(feature = "redaction")]

pub async fn extract_with_external_redaction(
    input: ExtractInput,
    config: &ExtractionConfig,
    findings_json: &str,
    offset_encoding: Option<&str>,
    max_findings: Option<u32>,
) -> Result<ExtractionResult> {
    let offset_encoding = offset_encoding.unwrap_or("unicode_code_points").parse()?;
    let requested_limit = max_findings.unwrap_or(crate::text::redaction::external::DEFAULT_MAX_FINDINGS);
    let default_limits = crate::extractors::security::SecurityLimits::default();
    let security_limit = crate::text::redaction::external::security_finding_limit(
        config.security_limits.as_ref().unwrap_or(&default_limits),
    );
    let effective_limit = u32::try_from(security_limit.min(requested_limit as usize)).map_err(|_| {
        crate::XbergError::validation("effective redaction finding limit exceeds the supported u32 range".to_string())
    })?;
    let findings = crate::text::redaction::parse_external_findings_bounded(findings_json, effective_limit)?;
    DEFAULT_ENGINE
        .extract_with_external_redaction(input, config, findings, offset_encoding, max_findings)
        .await
}

/// Extract content from multiple bytes or URI inputs.
pub async fn extract_batch(inputs: Vec<ExtractInput>, config: &ExtractionConfig) -> Result<ExtractionResult> {
    DEFAULT_ENGINE.extract_batch(inputs, config).await
}

/// Extract a batch whose inputs carry additional per-input options.
pub async fn extract_batch_with_input_options(
    inputs: Vec<ExtractInputWithOptions>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    DEFAULT_ENGINE.extract_batch_with_input_options(inputs, config).await
}

/// Extract a batch from a JSON array of per-input objects.
///
/// Uses the same fields and semantics as [`ExtractInputWithOptions`]. This is
/// useful for callers that pass input values through a text-based boundary.
/// Swift exposes this as a top-level async function.
///
/// # Examples
///
/// ```swift
/// import Xberg
///
/// let inputs = """
/// [
///   {"input":{"kind":"uri","uri":"invoice.pdf"},"use_layout_for_markdown":true},
///   {"input":{"kind":"uri","uri":"notes.pdf"},"use_layout_for_markdown":false}
/// ]
/// """
/// let result = try await extractBatchWithInputOptionsJson(inputs, "{}")
/// ```
///
/// ```csharp
/// using Xberg;
///
/// var inputs = """
/// [
///   {"input":{"kind":"uri","uri":"invoice.pdf"},"use_layout_for_markdown":true},
///   {"input":{"kind":"uri","uri":"notes.pdf"},"use_layout_for_markdown":false}
/// ]
/// """;
/// var result = await XbergConverter.ExtractBatchWithInputOptionsJsonAsync(inputs, new ExtractionConfig());
/// ```
pub async fn extract_batch_with_input_options_json(
    inputs_json: &str,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    let inputs = serde_json::from_str(inputs_json)
        .map_err(|error| crate::XbergError::validation(format!("Invalid batch inputs: {error}")))?;
    extract_batch_with_input_options(inputs, config).await
}
