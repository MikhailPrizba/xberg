//! Equivalence tests: each in-core seam default must reproduce the pre-seam
//! behavior of the code path it wraps.
//!
//! This is the verification anchor for the P3 seam phase. The seams establish
//! injection points; their defaults must be byte-for-byte equivalent to calling
//! the underlying functions directly, so the default [`Engine`] behaves exactly
//! as it did before the seams existed.

#![cfg(feature = "tokio-runtime")]

use xberg::engine::Engine;
use xberg::engine::seams::{CacheBackend, NoopCache, NoopProgressSink, ProgressEvent, ProgressSink};

#[tokio::test]
async fn noop_cache_get_returns_none() {
    let cache = NoopCache;
    assert_eq!(cache.get("any-key").await, None, "NoopCache must never hit");
}

#[tokio::test]
async fn noop_cache_put_is_inert() {
    let cache = NoopCache;
    cache.put("k", b"v".to_vec(), None).await;
    assert_eq!(cache.get("k").await, None, "NoopCache::put must store nothing");
}

#[test]
fn noop_progress_sink_emit_is_inert() {
    let sink = NoopProgressSink;
    sink.emit(ProgressEvent {
        stage: "stage".to_string(),
        message: Some("detail".to_string()),
        fraction: Some(0.5),
    });
}

#[tokio::test]
async fn engine_new_default_cache_is_noop() {
    let engine = Engine::new_default();
    assert_eq!(
        engine.cache_backend().get("k").await,
        None,
        "default engine cache must be a NoopCache"
    );
}

#[cfg(feature = "heuristics")]
mod structured_policy {
    use xberg::core::config::MergeMode;
    use xberg::engine::seams::{DefaultStructuredPolicy, StructuredPolicy};
    use xberg::heuristics::{StructuredInput, StructuredThresholds, choose_call_mode};

    fn input(mime_type: &str) -> StructuredInput {
        StructuredInput {
            mime_type: mime_type.to_string(),
            page_count: 3,
            text_coverage: 0.95,
            avg_chars_per_page: 500.0,
            embedded_image_count: 0,
            user_force_vision: false,
        }
    }

    #[test]
    fn choose_call_mode_matches_free_function() {
        let policy = DefaultStructuredPolicy::default();
        let thresholds = StructuredThresholds::default();
        for mime_type in [
            "image/png",
            "image/jpeg",
            "application/pdf",
            "text/html",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/octet-stream",
        ] {
            let signals = input(mime_type);
            assert_eq!(
                policy.choose_call_mode(&signals),
                choose_call_mode(&signals, &thresholds),
                "default policy must match heuristics::choose_call_mode for {mime_type}"
            );
        }
    }

    #[test]
    fn defaults_match_in_core_defaults() {
        let policy = DefaultStructuredPolicy::default();
        let expected = serde_json::to_value(StructuredThresholds::default()).expect("serialize thresholds");
        let actual = serde_json::to_value(policy.thresholds()).expect("serialize policy thresholds");
        assert_eq!(
            actual, expected,
            "default policy thresholds must be StructuredThresholds::default()"
        );
        assert_eq!(
            policy.merge_mode(),
            MergeMode::default(),
            "default merge mode must be MergeMode::default()"
        );
    }
}

#[cfg(feature = "presets")]
mod preset_resolver {
    use std::collections::BTreeMap;

    use xberg::engine::seams::{CorePresetResolver, PresetResolver};
    use xberg::presets::{Registry, resolve};

    #[test]
    fn resolves_builtin_identically_to_free_function() {
        let registry = Registry::global();
        let preset_id = registry.iter().next().expect("at least one built-in preset").id.clone();

        let resolver = CorePresetResolver;
        let resolver_preset = resolver.get(&preset_id).expect("resolver returns the built-in preset");
        let registry_preset = registry.get(&preset_id).expect("registry has the preset");

        let context = BTreeMap::new();
        let via_seam = resolver
            .resolve(&resolver_preset, None, &context)
            .expect("seam resolve succeeds");
        let via_free = resolve(registry_preset, None, &context).expect("free resolve succeeds");

        assert_eq!(
            serde_json::to_value(&via_seam).expect("serialize seam result"),
            serde_json::to_value(&via_free).expect("serialize free result"),
            "CorePresetResolver must resolve a built-in preset identically to presets::resolve"
        );
    }

    #[test]
    fn get_unknown_preset_returns_none() {
        let resolver = CorePresetResolver;
        assert!(resolver.get("definitely-not-a-real-preset-id").is_none());
    }
}
