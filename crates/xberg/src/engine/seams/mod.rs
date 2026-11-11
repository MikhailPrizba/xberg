//! Rust-only engine extension seams.
//!
//! Cache and progress are injectable through [`EngineBuilder`](super::EngineBuilder).
//! The remaining traits expose standalone Rust primitives for structured extraction,
//! preset lookup, LLM calls, and model resolution; they are not Engine configuration.
//!
//! These are deliberately **not** part of the language-binding surface: the
//! whole `engine` module is a bare `pub mod engine;` in `lib.rs` whose files are
//! not listed in `alef.toml` `sources`, so the binding generator emits nothing
//! for them. The trait names and public seam types are also listed in
//! `alef.toml` `[crates.exclude] types` as belt-and-suspenders.
//!

mod cache;
mod progress;

#[cfg(feature = "presets")]
mod preset_resolver;
#[cfg(feature = "heuristics")]
mod structured_policy;

pub use cache::{CacheBackend, NoopCache};
// ~keep Liveness is configuration-dependent: the only consumers are compiled out on
// narrow feature legs, so `-D dead-code` fires there and nowhere else. A hand-kept
// union-of-consumers `cfg` is what drifted here and failed the 1.3.0 publish (GH#1951).
#[cfg(any(test, feature = "ocr-pipeline"))]
#[allow(unused_imports)]
pub(crate) use progress::emit_ocr_page;
#[cfg(all(feature = "tokio-runtime", feature = "ocr-pipeline"))]
#[allow(unused_imports)]
pub(crate) use progress::inherit_progress;
#[cfg(feature = "tokio-runtime")]
pub(crate) use progress::scope_progress;
pub use progress::{NoopProgressSink, ProgressEvent, ProgressSink};

#[cfg(feature = "presets")]
pub use preset_resolver::{CorePresetResolver, PresetResolver};
#[cfg(feature = "heuristics")]
pub use structured_policy::{DefaultStructuredPolicy, StructuredPolicy};
