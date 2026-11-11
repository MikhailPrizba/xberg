use super::*;

#[test]
fn engine_batch_execution_plan_clamps_explicit_zero_to_one() {
    let config = ExtractionConfig {
        max_concurrent_extractions: Some(0),
        ..Default::default()
    };

    assert_eq!(
        resolve_engine_batch_execution_plan_for(&config, LayoutBatchWorkload::None, 8).workers,
        1
    );
}

#[test]
fn engine_batch_execution_plan_without_layout_respects_input_count() {
    let config = ExtractionConfig {
        concurrency: Some(crate::core::config::ConcurrencyConfig {
            max_threads: Some(4),
            max_concurrent_ocr: None,
        }),
        ..Default::default()
    };
    let inputs = vec![ExtractInput::default()];

    assert_eq!(resolve_engine_batch_execution_plan(&config, &inputs).workers, 1);
}

#[test]
fn downloaded_specific_http_mime_remains_authoritative_under_every_policy() {
    for policy in [
        crate::MimeDetectionPolicy::PreferContent,
        crate::MimeDetectionPolicy::TrustExtension,
        crate::MimeDetectionPolicy::ContentOnly,
    ] {
        let config = ExtractionConfig {
            mime_detection_policy: policy,
            ..Default::default()
        };
        let resolved = resolve_bytes_mime_type(
            Some("application/pdf"),
            Some("document.txt"),
            br#"{"kind":"json"}"#,
            &config,
        )
        .unwrap();
        assert_eq!(resolved, "application/pdf", "unexpected MIME for {policy:?}");
    }
}

#[test]
fn downloaded_octet_stream_uses_policy_with_derived_filename() {
    let cases = [
        (crate::MimeDetectionPolicy::PreferContent, "application/json"),
        (crate::MimeDetectionPolicy::TrustExtension, "text/plain"),
        (crate::MimeDetectionPolicy::ContentOnly, "application/json"),
    ];
    for (policy, expected) in cases {
        let config = ExtractionConfig {
            mime_detection_policy: policy,
            ..Default::default()
        };
        let resolved = resolve_bytes_mime_type(
            Some("application/octet-stream"),
            Some("download.txt"),
            br#"{"kind":"json"}"#,
            &config,
        )
        .unwrap();
        assert_eq!(resolved, expected, "unexpected MIME for {policy:?}");
    }
}

#[test]
fn prefer_content_bytes_falls_back_from_unsupported_specialized_extension_to_plain_text() {
    let config = ExtractionConfig {
        mime_detection_policy: crate::MimeDetectionPolicy::PreferContent,
        ..Default::default()
    };

    let resolved = resolve_bytes_mime_type(None, Some("feed.atom"), b"ordinary prose without markup", &config).unwrap();

    assert_eq!(resolved, "text/plain");
}

#[test]
fn content_only_bytes_ignores_a_supported_filename_extension() {
    let config = ExtractionConfig {
        mime_detection_policy: crate::MimeDetectionPolicy::ContentOnly,
        ..Default::default()
    };

    let resolved = resolve_bytes_mime_type(None, Some("document.txt"), br#"{"kind":"content"}"#, &config).unwrap();

    assert_eq!(resolved, "application/json");
}
