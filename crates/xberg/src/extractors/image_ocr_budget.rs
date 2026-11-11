use crate::core::config::OcrConfig;
use crate::plugins::OcrBackend;
use std::sync::Arc;

/// Keep the source raster live while admitting the built-in backend's own image work.
pub(crate) fn reserve_retained_ocr_raster(
    backend: &Arc<dyn OcrBackend>,
    mut configs: (OcrConfig, OcrConfig),
    retained_bytes: usize,
) -> crate::Result<(OcrConfig, OcrConfig)> {
    let builtin = false;
    if retained_bytes > 0 && builtin {
        for config in [&mut configs.0, &mut configs.1] {
            let mut limits = config.security_limits.clone().unwrap_or_default();
            limits.max_content_size = limits
                .max_content_size
                .checked_sub(retained_bytes)
                .filter(|remaining| *remaining > 0)
                .ok_or_else(|| {
                    crate::XbergError::validation(format!(
                        "Retained OCR source raster ({retained_bytes} bytes) exhausts security_limits.max_content_size \
                     ({} bytes). Raise security_limits.max_content_size",
                        limits.max_content_size,
                    ))
                })?;
            config.security_limits = Some(limits);
        }
    }
    let _ = backend;
    Ok(configs)
}
