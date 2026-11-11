//! Layout probe fallback for builds without a layout inference engine.

use super::DoctorCheck;
use crate::core::config::ExtractionConfig;

pub(super) fn probe_layout(config: &ExtractionConfig) -> Vec<DoctorCheck> {
    // Without `layout-types` the config cannot even express layout detection.
    #[cfg(not(feature = "layout-types"))]
    {
        let _ = config;
        vec![DoctorCheck::skip(
            "layout",
            "layout detection configuration is not available in this build",
        )]
    }
    #[cfg(feature = "layout-types")]
    {
        if config.layout.is_none() {
            return vec![DoctorCheck::skip("layout", "layout detection is not configured")];
        }
        #[cfg(target_arch = "wasm32")]
        let reason = "layout inference is not available on wasm32";
        #[cfg(not(target_arch = "wasm32"))]
        let reason = "layout detection is not compiled in (enable `layout-detection` or `layout-tract`)";
        vec![DoctorCheck::fail("layout", reason)]
    }
}
