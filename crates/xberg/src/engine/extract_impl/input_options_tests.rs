use super::*;
use crate::core::config::ExtractInputWithOptions;

fn paired(flag: Option<bool>) -> ExtractInputWithOptions {
    let mut item = ExtractInputWithOptions::new(ExtractInput::from_bytes(
        b"%PDF-1.7\n".to_vec(),
        "application/pdf",
        None,
    ));
    item.use_layout_for_markdown = flag;
    item
}

#[test]
fn paired_omitted_options_preserve_legacy_borrowed_config() {
    let base = ExtractionConfig::default();
    let legacy = ExtractInput::from_bytes(b"hello".to_vec(), "text/plain", None);
    let old = effective_input_config(&legacy, &base);
    assert!(matches!(old, Cow::Borrowed(_)));
    assert!(std::ptr::eq(old.as_ref(), &base));
    drop(old);
    let item = ExtractInputWithOptions::new(legacy);
    let actual = effective_input_config(&item, &base);
    assert!(matches!(actual, Cow::Borrowed(_)));
    assert!(std::ptr::eq(actual.as_ref(), &base));
    drop(actual);
    #[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
    {
        let base = Arc::new(base);
        assert!(Arc::ptr_eq(&resolve_input_config_arc(&item, &base), &base));
    }
}

#[test]
fn paired_effective_config_resolves_the_complete_layout_matrix() {
    let mut actual = Vec::new();
    for base_flag in [false, true] {
        let base = ExtractionConfig {
            use_layout_for_markdown: base_flag,
            max_concurrent_extractions: Some(7),
            ..Default::default()
        };
        for flag in [None, Some(true), Some(false)] {
            let item = paired(flag);
            let resolved = effective_input_config(&item, &base);
            assert_eq!(resolved.max_concurrent_extractions, Some(7));
            assert_eq!(base.use_layout_for_markdown, base_flag);
            actual.push(resolved.use_layout_for_markdown);
        }
    }
    assert_eq!(actual, [false, true, false, true, true, false]);
}

#[test]
fn paired_layout_override_keeps_existing_file_override_precedence() {
    let base = ExtractionConfig {
        use_layout_for_markdown: true,
        enable_quality_processing: true,
        strict_parse: false,
        max_concurrent_extractions: Some(7),
        ..Default::default()
    };
    let mut item = paired(Some(false));
    item.input.config = Some(crate::core::config::FileExtractionConfig {
        enable_quality_processing: Some(false),
        strict_parse: Some(true),
        ..Default::default()
    });
    let resolved = effective_input_config(&item, &base);
    assert!(!resolved.enable_quality_processing);
    assert!(resolved.strict_parse);
    assert_eq!(resolved.max_concurrent_extractions, Some(7));
    assert!(base.enable_quality_processing);
    assert!(!base.strict_parse);
    assert!(base.use_layout_for_markdown);
    assert!(!resolved.use_layout_for_markdown);
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
#[test]
fn paired_pending_reorder_preserves_option_and_payload_identity() {
    let pending: VecDeque<_> = [Some(true), Some(false), None]
        .into_iter()
        .enumerate()
        .map(|(index, flag)| (index, paired(flag), format!("input-{index}")))
        .collect();
    let identities = pending
        .iter()
        .map(|(index, item, source)| {
            (
                *index,
                (
                    item.use_layout_for_markdown,
                    item.input.bytes.as_ref().unwrap().as_ptr(),
                    source.as_str(),
                ),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let expected = identities
        .iter()
        .map(|(index, (flag, pointer, source))| (*index, (*flag, *pointer, source.to_string())))
        .collect::<std::collections::BTreeMap<_, _>>();
    drop(identities);
    let actual = reorder_pending_batch_items(pending, &[Some(1), Some(3), Some(2)]);
    assert_eq!(actual.iter().map(|(index, _, _)| *index).collect::<Vec<_>>(), [1, 2, 0]);
    for (index, item, source) in actual {
        let (flag, pointer, expected_source) = &expected[&index];
        assert_eq!(item.use_layout_for_markdown, *flag);
        assert_eq!(item.input.bytes.as_ref().unwrap().as_ptr(), *pointer);
        assert_eq!(source, *expected_source);
    }
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
#[test]
fn paired_actual_classifier_without_backend_keeps_layout_work_disabled() {
    let base = ExtractionConfig::default();
    let inputs = [paired(Some(true)), paired(Some(false))];
    assert_eq!(
        classify_layout_batch(&base, &inputs),
        crate::core::config::concurrency::LayoutBatchWorkload::None
    );
}
