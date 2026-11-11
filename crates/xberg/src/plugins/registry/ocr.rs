//! OCR backend registry.

use crate::Result;
#[cfg(feature = "ocr-pipeline")]
use crate::XbergError;
use crate::plugins::OcrBackend;
use ahash::AHashMap;
use std::sync::Arc;

/// Resolves a caller-supplied backend name to its registered spelling.
///
/// Public capability queries and internal dispatch must agree on what `"PaddleOCR"`
/// means; two copies of this mapping would let them drift, so both go through here.
/// Ungated on purpose — `OcrBackendRegistry::get` is gated behind the OCR features
/// while the public `ocr_backend_supports_language` is not. ~keep
pub(crate) fn canonical_ocr_backend_name(name: &str) -> String {
    let normalized = name.to_ascii_lowercase();
    match normalized.as_str() {
        "paddleocr" => "paddle-ocr".to_string(),
        _ => normalized,
    }
}

pub(crate) fn builtin_ocr_backend_names() -> Vec<&'static str> {
    Vec::new()
}

/// Registry for OCR backend plugins.
///
/// Manages OCR backends with backend type and language-based selection.
///
/// # Thread Safety
///
/// The registry is thread-safe and can be accessed concurrently from multiple threads.
///
/// # Example
///
/// ```rust,no_run
/// use xberg::plugins::registry::OcrBackendRegistry;
/// use std::sync::Arc;
///
/// let registry = OcrBackendRegistry::new();
/// // Register OCR backends
/// // registry.register(Arc::new(TesseractBackend::new()));
/// ```

pub struct OcrBackendRegistry {
    pub(super) backends: AHashMap<String, Arc<dyn OcrBackend>>,
}

impl OcrBackendRegistry {
    /// Create a new OCR backend registry with default backends.
    ///
    /// Registers the Tesseract backend by default if the "ocr" feature is enabled,
    /// PaddleOCR if the "paddle-ocr" feature is enabled, and Sceptre if the
    /// "sceptre-ocr" feature is enabled.
    ///
    /// If a backend fails to initialize or register it is skipped with a warning,
    /// allowing the process to continue with whichever backends are available.
    #[tracing::instrument(name = "ocr_backend_registry_init")]
    pub fn new() -> Self {
        let mut registry = Self::new_empty();
        registry.register_defaults();
        registry
    }

    /// Register the built-in OCR backends into this registry.
    ///
    /// Registers whichever backends the active feature set enables — Tesseract
    /// (`ocr`/`ocr-wasm`), PaddleOCR (`paddle-ocr`), Sceptre (`sceptre-ocr`),
    /// and the VLM backend (`liter-llm`). Each backend is registered independently:
    /// if one fails to initialize it is skipped with a warning so the remaining
    /// backends still register.
    ///
    /// This is invoked by [`OcrBackendRegistry::new`] at construction and reused
    /// by the self-healing initialization path so a registry emptied via
    /// [`clear`](Self::clear) can be re-seeded with the defaults.
    pub fn register_defaults(&mut self) {
        self.register_default_engine_backends();
        self.register_default_candle_backends();
    }

    /// Register the built-in non-Candle backends: Tesseract (native and WASM),
    /// PaddleOCR, Sceptre, and the VLM backend.
    fn register_default_engine_backends(&mut self) {}

    /// Register the built-in Candle-backed backends: TrOCR, PaddleOCR-VL,
    /// GLM-OCR, and DeepSeek-OCR.
    fn register_default_candle_backends(&mut self) {}

    /// Create a new empty OCR backend registry without default backends.
    ///
    /// This is useful for testing or when you want full control over backend registration.
    pub fn new_empty() -> Self {
        Self {
            backends: AHashMap::new(),
        }
    }

    /// Register an OCR backend.
    ///
    /// # Arguments
    ///
    /// * `backend` - The OCR backend to register
    ///
    /// # Returns
    ///
    /// - `Ok(())` if registration succeeded
    /// - `Err(...)` if initialization failed
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use xberg::plugins::registry::OcrBackendRegistry;
    /// # use std::sync::Arc;
    /// let mut registry = OcrBackendRegistry::new();
    /// // let backend = Arc::new(MyOcrBackend::new());
    /// // registry.register(backend)?;
    /// # Ok::<(), xberg::XbergError>(())
    /// ```
    #[tracing::instrument(skip(self, backend), fields(backend_name))]
    pub fn register(&mut self, backend: Arc<dyn OcrBackend>) -> Result<()> {
        let name = backend.name().to_string();
        tracing::Span::current().record("backend_name", name.as_str());

        super::validate_plugin_name(&name)?;

        backend.initialize()?;

        tracing::info!(backend = %name, "OCR backend registered");
        self.backends.insert(name, backend);
        Ok(())
    }

    /// Get an OCR backend by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Backend name
    ///
    /// # Returns
    ///
    /// The backend if found, or an error if not registered.
    #[cfg(feature = "ocr-pipeline")]
    #[tracing::instrument(skip(self), fields(registered_backends = ?self.backends.keys().collect::<Vec<_>>()))]
    pub(crate) fn get(&self, name: &str) -> Result<Arc<dyn OcrBackend>> {
        if let Some(backend) = self.backends.get(name) {
            return Ok(Arc::clone(backend));
        }

        let canonical = canonical_ocr_backend_name(name);
        self.backends.get(canonical.as_str()).cloned().ok_or_else(|| {
            tracing::error!(
                backend = name,
                available = ?self.backends.keys().collect::<Vec<_>>(),
                "OCR backend not found in registry"
            );
            XbergError::Plugin {
                message: format!(
                    "OCR backend '{}' not registered. Available backends: {:?}",
                    name,
                    self.backends.keys().collect::<Vec<_>>()
                ),
                plugin_name: name.to_string(),
            }
        })
    }

    /// List all registered backend names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<_> = self.backends.keys().cloned().collect();
        names.sort_unstable();
        names
    }

    pub(crate) fn registered_snapshot(&self) -> Vec<(String, Arc<dyn OcrBackend>)> {
        let mut backends: Vec<_> = self
            .backends
            .iter()
            .map(|(name, backend)| (name.clone(), Arc::clone(backend)))
            .collect();
        backends.sort_unstable_by(|left, right| left.0.cmp(&right.0));
        backends
    }

    /// Remove a backend from the registry.
    ///
    /// Calls `shutdown()` on the backend before removing.
    pub fn remove(&mut self, name: &str) -> Result<()> {
        if let Some(backend) = self.backends.remove(name) {
            backend.shutdown()?;
        }
        Ok(())
    }

    /// Shutdown all backends and clear the registry.
    pub fn shutdown_all(&mut self) -> Result<()> {
        let names: Vec<_> = self.backends.keys().cloned().collect();
        for name in names {
            self.remove(&name)?;
        }
        Ok(())
    }

    /// Drain the registry. Alias for `shutdown_all` used by alef trait-bridge codegen.
    pub fn clear(&mut self) -> Result<()> {
        self.shutdown_all()
    }

    /// True when the registry lacks a usable built-in default and should be
    /// re-seeded.
    ///
    /// This is the case when the registry is completely empty, or — when a
    /// Tesseract built-in is compiled in — when the built-in default backend
    /// (`tesseract`, the fallback dispatch target) is absent. The latter can
    /// happen after [`clear`](Self::clear) followed by registering a *different*
    /// backend: the registry is non-empty, yet default-config OCR has no usable
    /// backend.
    #[cfg(feature = "ocr-pipeline")]
    pub(crate) fn is_missing_default_backend(&self) -> bool {
        const DEFAULT: Option<&str> = None;

        self.backends.is_empty() || DEFAULT.is_some_and(|name| !self.backends.contains_key(name))
    }

    /// Non-destructively re-seed the built-in default backends when they are
    /// missing.
    ///
    /// Keeps any user-registered backends; a no-op when the built-in default is
    /// already present. Used by the self-healing dispatch path so a registry
    /// left without a default (via clear + register-other) heals itself.
    #[cfg(feature = "ocr-pipeline")]
    pub(crate) fn ensure_defaults(&mut self) {
        if self.is_missing_default_backend() {
            self.register_defaults();
        }
    }
}

impl Default for OcrBackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod capability_manifest_tests {
    use super::*;

    #[test]
    fn builtin_capability_manifest_is_empty_in_compact_build() {
        assert!(builtin_ocr_backend_names().is_empty());
    }

    #[test]
    fn registered_backend_snapshot_is_sorted() {
        let registry = OcrBackendRegistry::new();
        let names: Vec<_> = registry
            .registered_snapshot()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }
}
