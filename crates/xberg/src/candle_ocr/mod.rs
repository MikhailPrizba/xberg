//! Configuration DTOs retained for Python API compatibility.
mod config;
pub use config::{
    CandleDeepseekOcrDtype, CandleDevicePreference, CandleModelId, CandleOcrConfig, CandleTrocrVariant,
    DeepseekOcrBackendOptions, GlmOcrBackendOptions, GlmOcrLayoutMode, GlmOcrTaskKind, PaddleOcrVlBackendOptions,
    PaddleOcrVlTaskKind, TrocrBackendOptions,
};
