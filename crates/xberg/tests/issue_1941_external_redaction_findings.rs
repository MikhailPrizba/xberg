//! Regressions for external content-inspection findings (GH#1941).

#![cfg(all(feature = "redaction", feature = "tokio-runtime"))]

use std::borrow::Cow;
use std::collections::HashSet;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use xberg::engine::Engine;
use xberg::engine::seams::CacheBackend;
use xberg::extractors::security::SecurityLimits;
use xberg::plugins::PostProcessor;
use xberg::plugins::processor::builtin::redaction::RedactionProcessor;
use xberg::text::redaction::{parse_external_findings, parse_external_findings_bounded, redact_with_entities};
use xberg::types::redaction::PiiCategory;
use xberg::types::tables::Table;
use xberg::types::{Chunk, ChunkMetadata, ChunkType};
use xberg::{
    ExternalRedactionFinding, ExtractInput, ExtractedDocument, ExtractionConfig, ExtractionResult, RedactionConfig,
    RedactionOffsetEncoding, extract, extract_with_external_redaction as extract_with_external_redaction_json,
    redact_external as redact_external_json,
};

const MASK: &str = "[REDACTED]";

fn run_extraction_test<T, F>(future: F) -> T
where
    T: Send + 'static,
    F: Future<Output = T> + Send + 'static,
{
    std::thread::Builder::new()
        .name("external-redaction-test".to_string())
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(future)
        })
        .expect("test thread")
        .join()
        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

fn document(content: &str) -> ExtractedDocument {
    let mut document = ExtractedDocument::default();
    document.content = content.to_string();
    document.mime_type = Cow::Borrowed("text/plain");
    document
}

fn text_finding(label: &str, text: &str) -> ExternalRedactionFinding {
    ExternalRedactionFinding {
        label: label.to_string(),
        text: Some(text.to_string()),
        ..Default::default()
    }
}

fn span_finding(label: &str, start: u32, end: u32) -> ExternalRedactionFinding {
    ExternalRedactionFinding {
        label: label.to_string(),
        start: Some(start),
        end: Some(end),
        ..Default::default()
    }
}

fn with_findings(findings: Vec<ExternalRedactionFinding>) -> RedactionConfig {
    RedactionConfig {
        findings,
        ..Default::default()
    }
}

async fn extract_with_external_redaction(
    input: ExtractInput,
    config: &ExtractionConfig,
    findings: Vec<ExternalRedactionFinding>,
    offset_encoding: Option<&str>,
    max_findings: Option<u32>,
) -> xberg::Result<ExtractionResult> {
    let findings_json = serde_json::to_string(&findings).expect("serialize typed findings");
    extract_with_external_redaction_json(input, config, &findings_json, offset_encoding, max_findings).await
}

async fn redact_external(
    document: ExtractedDocument,
    config: RedactionConfig,
    findings: Vec<ExternalRedactionFinding>,
    offset_encoding: Option<&str>,
    max_findings: Option<u32>,
) -> xberg::Result<ExtractedDocument> {
    let findings_json = serde_json::to_string(&findings).expect("serialize typed findings");
    redact_external_json(document, config, &findings_json, offset_encoding, max_findings).await
}

fn categories(document: &ExtractedDocument) -> HashSet<PiiCategory> {
    document
        .redaction_report
        .as_ref()
        .expect("redaction report")
        .findings
        .iter()
        .map(|finding| finding.category.clone())
        .collect()
}

fn chunk(content: &str) -> Chunk {
    Chunk {
        content: content.to_string(),
        chunk_type: ChunkType::Unknown,
        embedding: None,
        sparse_embedding: None,
        late_interaction: None,
        metadata: ChunkMetadata {
            byte_start: 0,
            byte_end: content.len(),
            token_count: None,
            chunk_index: 0,
            total_chunks: 1,
            first_page: None,
            last_page: None,
            heading_context: None,
            heading_path: Vec::new(),
            image_indices: Vec::new(),
            node_ids: Vec::new(),
            page_spans: Vec::new(),
            classifications: Vec::new(),
        },
    }
}

#[tokio::test]
async fn should_redact_presidio_offsets_through_the_owned_request_api() {
    let redacted = redact_external_json(
        document("Zoë Quorlim called alice@example.com."),
        RedactionConfig::default(),
        r#"[{"entity_type":"PERSON","start":0,"end":11,"score":0.85,"analysis_explanation":{"recognizer":"spacy"}}]"#,
        Some("unicode_code_points"),
        Some(10),
    )
    .await
    .expect("external redaction must succeed");

    assert_eq!(redacted.content, format!("{MASK} called {MASK}."));
}

#[tokio::test]
async fn should_default_the_limit_and_enforce_explicit_zero() {
    let redacted = redact_external(
        document("Zarnak"),
        RedactionConfig::default(),
        vec![text_finding("PERSON", "Zarnak")],
        None,
        None,
    )
    .await
    .expect("the documented default must accept one finding");
    assert_eq!(redacted.content, MASK);

    let error = redact_external(
        document("Zarnak"),
        RedactionConfig::default(),
        vec![text_finding("PERSON", "Zarnak")],
        Some("unicode_code_points"),
        Some(0),
    )
    .await
    .expect_err("an explicit zero limit must reject one finding");
    assert!(error.to_string().contains("maximum of 0"), "{error}");
}

#[tokio::test]
async fn should_enforce_the_fixed_ceiling_before_compiling_direct_findings() {
    let error = redact_external(
        document("Zarnak"),
        RedactionConfig::default(),
        vec![ExternalRedactionFinding::default(); 10_001],
        None,
        Some(10_001),
    )
    .await
    .expect_err("the caller limit must not raise the fixed safety ceiling");

    assert!(error.to_string().contains("maximum of 10000"), "{error}");
}

#[tokio::test]
async fn should_default_to_unicode_code_point_offsets_and_reject_unknown_encodings() {
    let redacted = redact_external(
        document("Zoë Quorlim"),
        RedactionConfig::default(),
        vec![span_finding("PERSON", 0, 3)],
        None,
        Some(10),
    )
    .await
    .expect("omitted encoding must use Unicode code-point offsets");
    assert_eq!(redacted.content, format!("{MASK} Quorlim"));

    let error = redact_external(
        document("Zarnak"),
        RedactionConfig::default(),
        vec![text_finding("PERSON", "Zarnak")],
        Some("bytes-ish"),
        Some(10),
    )
    .await
    .expect_err("unknown offset encodings must fail validation");
    assert!(
        error.to_string().contains("unsupported redaction offset encoding"),
        "{error}"
    );
}

#[test]
fn should_accept_aws_comprehend_aliases() {
    let findings = parse_external_findings(
        r#"[{"Score":0.99,"Type":"NAME","Text":"Zarnak Quorlim","BeginOffset":6,"EndOffset":20}]"#,
    )
    .expect("AWS Comprehend output must parse");

    assert_eq!(findings[0].label, "NAME");
    assert_eq!(findings[0].text.as_deref(), Some("Zarnak Quorlim"));
    assert_eq!((findings[0].start, findings[0].end), (Some(6), Some(20)));
    assert_eq!(findings[0].score, Some(0.99));
}

#[test]
fn should_derive_utf16_spans_and_reject_surrogate_boundaries() {
    let content = "\u{1F600} Zarnak smiled.";
    let mut valid = document(content);
    let valid_config = RedactionConfig {
        findings: vec![span_finding("PERSON", 3, 9)],
        findings_offset_encoding: RedactionOffsetEncoding::Utf16CodeUnits,
        ..Default::default()
    };
    redact_with_entities(&mut valid, &valid_config, &[]).expect("UTF-16 code-unit span must resolve");
    assert_eq!(valid.content, format!("\u{1F600} {MASK} smiled."));

    let mut invalid = document(content);
    let invalid_config = RedactionConfig {
        findings: vec![span_finding("PERSON", 1, 9)],
        findings_offset_encoding: RedactionOffsetEncoding::Utf16CodeUnits,
        ..Default::default()
    };
    let error =
        redact_with_entities(&mut invalid, &invalid_config, &[]).expect_err("a span inside a surrogate pair must fail");
    assert!(error.to_string().contains("RedactionConfig.findings[0]"), "{error}");
    assert_eq!(invalid.content, content);
}

#[test]
fn should_parse_json_lines_and_report_a_malformed_line() {
    let findings = parse_external_findings(
        "{\"entity_type\":\"PERSON\",\"text\":\"Zarnak\"}\n\n{\"Type\":\"CITY\",\"Text\":\"Quorlim\"}\n",
    )
    .expect("JSON Lines findings must parse");
    assert_eq!(findings.len(), 2);

    let error = parse_external_findings("{\"entity_type\":\"PERSON\",\"text\":\"Zarnak\"}\nnot-json\n")
        .expect_err("malformed JSON Lines must fail");
    assert!(error.to_string().contains("line 2"), "{error}");
}

#[test]
fn should_validate_every_configured_finding() {
    for finding in [
        text_finding(" ", "Zarnak"),
        text_finding("PERSON", "  "),
        ExternalRedactionFinding {
            label: "PERSON".to_string(),
            start: Some(4),
            ..Default::default()
        },
        span_finding("PERSON", 8, 8),
        ExternalRedactionFinding {
            score: Some(1.5),
            ..text_finding("PERSON", "Zarnak")
        },
    ] {
        assert!(
            with_findings(vec![finding.clone()]).validate().is_err(),
            "must reject {finding:?}"
        );
    }
}

#[test]
fn should_redact_every_text_field_and_populate_the_report() {
    let name = "Zarnak Quorlim";
    let mut output = document(&format!("{name} signed. Witness: {name}."));
    output.formatted_content = Some(format!("# {name}\n\nSigned by {name}."));
    output.chunks = Some(vec![chunk(&format!("{name} signed."))]);
    output.tables = vec![Table {
        cells: vec![vec!["Signatory".into(), name.into()]],
        markdown: format!("| Signatory | {name} |"),
        page_number: 1,
        ..Default::default()
    }];
    output.metadata.subject = Some(format!("Agreement with {name}"));

    redact_with_entities(&mut output, &with_findings(vec![text_finding("PERSON", name)]), &[])
        .expect("configured finding must redact");

    assert_eq!(output.content, format!("{MASK} signed. Witness: {MASK}."));
    assert_eq!(
        output.formatted_content.as_deref(),
        Some(format!("# {MASK}\n\nSigned by {MASK}.").as_str())
    );
    assert_eq!(
        output.chunks.as_ref().expect("chunks")[0].content,
        format!("{MASK} signed.")
    );
    assert_eq!(output.tables[0].cells[0][1], MASK);
    assert_eq!(output.tables[0].markdown, format!("| Signatory | {MASK} |"));
    assert_eq!(output.metadata.subject, Some(format!("Agreement with {MASK}")));
    assert_eq!(
        categories(&output),
        HashSet::from([PiiCategory::Custom("PERSON".to_string())])
    );
    assert_eq!(output.redaction_report.as_ref().expect("report").total_redacted, 8);
}

#[test]
fn should_read_findings_from_config_json() {
    let config: ExtractionConfig = serde_json::from_str(
        r#"{"redaction":{"findings":[{"entity_type":"PERSON","text":"Zarnak","score":0.92,"vendor":{}}],"min_score":0.8,"findings_offset_encoding":"unicode_code_points"}}"#,
    )
    .expect("external findings must parse from ExtractionConfig JSON");
    let redaction = config.redaction.expect("redaction config");
    assert_eq!(
        redaction.findings,
        vec![ExternalRedactionFinding {
            score: Some(0.92),
            ..text_finding("PERSON", "Zarnak")
        }]
    );
    assert_eq!(
        redaction.findings_offset_encoding,
        RedactionOffsetEncoding::UnicodeCodePoints
    );
    assert_eq!(redaction.min_score, Some(0.8));
}

#[test]
fn should_load_json_and_json_lines_findings_files() {
    let directory = tempfile::tempdir().expect("tempdir");
    let array = directory.path().join("findings.json");
    std::fs::write(&array, r#"[{"entity_type":"PERSON","start":0,"end":14}]"#).expect("write array");
    let lines = directory.path().join("findings.jsonl");
    std::fs::write(
        &lines,
        "{\"entity_type\":\"PERSON\",\"start\":0,\"end\":14}\n{\"Type\":\"CITY\",\"Text\":\"Quorlim City\"}\n",
    )
    .expect("write JSON Lines");

    for (path, expected) in [
        (array, format!("{MASK} moved to Quorlim City.")),
        (lines, format!("{MASK} moved to {MASK}.")),
    ] {
        let mut output = document("Zarnak Quorlim moved to Quorlim City.");
        let config = RedactionConfig {
            findings_path: Some(path),
            ..Default::default()
        };
        redact_with_entities(&mut output, &config, &[]).expect("file findings must redact");
        assert_eq!(output.content, expected);
    }
}

#[tokio::test]
async fn should_enforce_the_security_limit_across_configured_findings() {
    let config = ExtractionConfig {
        redaction: Some(with_findings(vec![
            text_finding("PERSON", "Zarnak"),
            text_finding("PERSON", "Quorlim"),
        ])),
        security_limits: Some(SecurityLimits {
            max_iterations: 1,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut output = document("Zarnak Quorlim");
    let error = RedactionProcessor
        .process(&mut output, &config)
        .await
        .expect_err("the security limit must cap configured findings");
    assert!(
        error.to_string().contains("effective redaction finding limit (1)"),
        "{error}"
    );
    assert_eq!(output.content, "Zarnak Quorlim");
}

#[tokio::test]
async fn should_accept_configured_findings_at_the_iteration_limit() {
    let config = ExtractionConfig {
        redaction: Some(with_findings(vec![
            text_finding("PERSON", "Zarnak"),
            text_finding("PERSON", "Quorlim"),
        ])),
        security_limits: Some(SecurityLimits {
            max_iterations: 2,
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut output = document("Zarnak Quorlim");
    RedactionProcessor
        .process(&mut output, &config)
        .await
        .expect("the exact iteration limit must be accepted");
    assert_eq!(output.content, format!("{MASK} {MASK}"));
}

#[test]
fn should_enforce_the_request_limit_across_configured_and_request_findings() {
    run_extraction_test(async {
        let config = ExtractionConfig {
            redaction: Some(with_findings(vec![text_finding("PERSON", "Zarnak")])),
            security_limits: Some(SecurityLimits {
                max_iterations: 10,
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Quorlim")],
            None,
            Some(1),
        )
        .await
        .expect_err("the request limit must cap configured and request findings together");

        assert!(
            error.to_string().contains("effective redaction finding limit (1)"),
            "{error}"
        );
    });
}

#[test]
fn should_accept_configured_and_request_findings_at_the_iteration_limit() {
    run_extraction_test(async {
        let config = ExtractionConfig {
            redaction: Some(with_findings(vec![text_finding("PERSON", "Zarnak")])),
            security_limits: Some(SecurityLimits {
                max_iterations: 10,
                ..Default::default()
            }),
            ..Default::default()
        };
        let output = extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Quorlim")],
            None,
            Some(2),
        )
        .await
        .expect("the exact iteration limit must be accepted");

        assert_eq!(output.results[0].content, format!("{MASK} {MASK}"));
    });
}

#[test]
fn should_stop_parsing_at_the_finding_limit_before_a_later_malformed_entry() {
    let error = parse_external_findings_bounded(
        r#"[
            {"entity_type":"PERSON","text":"Zarnak"},
            {"entity_type":"CITY","text":"Quorlim"},
            not-json
        ]"#,
        1,
    )
    .expect_err("the second finding must exceed the limit before the third is parsed");

    assert!(error.to_string().contains("exceed"), "{error}");
    assert!(error.to_string().contains("1"), "{error}");
}

#[tokio::test]
async fn should_reject_an_unresolvable_span() {
    let error = redact_external(
        document("Zoë Quorlim called."),
        RedactionConfig::default(),
        vec![span_finding("PERSON", 0, 11)],
        Some("utf8_bytes"),
        Some(10),
    )
    .await
    .expect_err("a byte span cutting a word must fail");

    assert!(error.to_string().contains("offset_encoding"), "{error}");
    assert!(
        !error.to_string().contains("Quorl"),
        "document text leaked in error: {error}"
    );
}

#[test]
fn should_apply_only_external_matchers_when_base_redaction_is_absent() {
    run_extraction_test(async {
        let output = extract_with_external_redaction(
            ExtractInput::from_bytes(
                b"Zarnak Quorlim emailed alice@example.com.".to_vec(),
                "text/plain",
                None,
            ),
            &ExtractionConfig::default(),
            vec![text_finding("PERSON", "Zarnak Quorlim")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect("extraction must succeed");

        assert_eq!(output.results[0].content, format!("{MASK} emailed alice@example.com."));
    });
}

#[test]
fn should_combine_external_findings_with_per_input_redaction() {
    run_extraction_test(async {
        let mut input = ExtractInput::from_bytes(b"Zarnak Quorlim met Blorp Nazzle.".to_vec(), "text/plain", None);
        input.config = Some(xberg::FileExtractionConfig {
            redaction: Some(RedactionConfig {
                custom_terms: vec![xberg::RedactionTerm::labeled("inspection_term", "Blorp Nazzle")],
                ..Default::default()
            }),
            ..Default::default()
        });

        let output = extract_with_external_redaction(
            input,
            &ExtractionConfig::default(),
            vec![text_finding("PERSON", "Zarnak Quorlim")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect("extraction must succeed");

        assert_eq!(output.results[0].content, format!("{MASK} met {MASK}."));
    });
}

#[test]
fn should_isolate_concurrent_external_redaction_scopes() {
    run_extraction_test(async {
        let first = tokio::spawn(async {
            extract_with_external_redaction(
                ExtractInput::from_bytes(b"Zarnak Quorlim met Blorp Nazzle.".to_vec(), "text/plain", None),
                &ExtractionConfig::default(),
                vec![text_finding("PERSON", "Zarnak Quorlim")],
                Some("unicode_code_points"),
                Some(10),
            )
            .await
        });
        let second = tokio::spawn(async {
            extract_with_external_redaction(
                ExtractInput::from_bytes(b"Zarnak Quorlim met Blorp Nazzle.".to_vec(), "text/plain", None),
                &ExtractionConfig::default(),
                vec![text_finding("PERSON", "Blorp Nazzle")],
                Some("unicode_code_points"),
                Some(10),
            )
            .await
        });

        let (first, second) = tokio::join!(first, second);
        assert_eq!(
            first.expect("first task").expect("first extraction").results[0].content,
            format!("{MASK} met Blorp Nazzle.")
        );
        assert_eq!(
            second.expect("second task").expect("second extraction").results[0].content,
            format!("Zarnak Quorlim met {MASK}.")
        );

        let ordinary = extract(
            ExtractInput::from_bytes(b"Zarnak Quorlim met Blorp Nazzle.".to_vec(), "text/plain", None),
            &ExtractionConfig::default(),
        )
        .await
        .expect("ordinary extraction");
        assert_eq!(ordinary.results[0].content, "Zarnak Quorlim met Blorp Nazzle.");
    });
}

#[derive(Default)]
struct CountingCache {
    gets: AtomicUsize,
    puts: AtomicUsize,
    cached: Option<Vec<u8>>,
}

#[async_trait]
impl CacheBackend for CountingCache {
    async fn get(&self, _key: &str) -> Option<Vec<u8>> {
        self.gets.fetch_add(1, Ordering::SeqCst);
        self.cached.clone()
    }

    async fn put(&self, _key: &str, _value: Vec<u8>, _ttl: Option<std::time::Duration>) {
        self.puts.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct MemoryCache {
    values: Mutex<std::collections::HashMap<String, Vec<u8>>>,
}

#[async_trait]
impl CacheBackend for MemoryCache {
    async fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.values.lock().expect("memory cache lock").get(key).cloned()
    }

    async fn put(&self, key: &str, value: Vec<u8>, _ttl: Option<std::time::Duration>) {
        self.values
            .lock()
            .expect("memory cache lock")
            .insert(key.to_string(), value);
    }
}

fn findings_file_config(path: &std::path::Path) -> ExtractionConfig {
    ExtractionConfig {
        redaction: Some(RedactionConfig {
            findings_path: Some(path.to_path_buf()),
            ..Default::default()
        }),
        ..Default::default()
    }
}

#[test]
fn should_not_reuse_single_input_cache_when_findings_file_changes() {
    run_extraction_test(async {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("findings.json");
        let cache = Arc::new(MemoryCache::default());
        let engine = Engine::builder().with_cache_backend(cache).build();
        let config = findings_file_config(&path);

        std::fs::write(&path, r#"[{"label":"PERSON","text":"Zarnak"}]"#).expect("first findings");
        let first = engine
            .extract(
                ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
                &config,
            )
            .await
            .expect("first extraction");
        assert_eq!(first.results[0].content, format!("{MASK} Quorlim"));

        std::fs::write(&path, r#"[{"label":"PERSON","text":"Quorlim"}]"#).expect("second findings");
        let second = engine
            .extract(
                ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
                &config,
            )
            .await
            .expect("second extraction");
        assert_eq!(second.results[0].content, format!("Zarnak {MASK}"));
    });
}

#[test]
fn should_not_reuse_batch_cache_when_findings_file_changes() {
    run_extraction_test(async {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("findings.json");
        let cache = Arc::new(MemoryCache::default());
        let engine = Engine::builder().with_cache_backend(cache).build();
        let config = findings_file_config(&path);
        let inputs = || vec![ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None)];

        std::fs::write(&path, r#"[{"label":"PERSON","text":"Zarnak"}]"#).expect("first findings");
        let first = engine.extract_batch(inputs(), &config).await.expect("first batch");
        assert_eq!(first.results[0].content, format!("{MASK} Quorlim"));

        std::fs::write(&path, r#"[{"label":"PERSON","text":"Quorlim"}]"#).expect("second findings");
        let second = engine.extract_batch(inputs(), &config).await.expect("second batch");
        assert_eq!(second.results[0].content, format!("Zarnak {MASK}"));
    });
}

#[test]
fn should_bypass_engine_and_extraction_caches_for_scoped_findings() {
    run_extraction_test(async {
        let cached = xberg::ExtractionResult::single(document("CACHED UNREDACTED RESULT"));
        let cache = Arc::new(CountingCache {
            cached: Some(serde_json::to_vec(&cached).expect("cached result JSON")),
            ..Default::default()
        });
        let engine = Engine::builder().with_cache_backend(cache.clone()).build();

        let output = engine
            .extract_with_external_redaction(
                ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
                &ExtractionConfig::default(),
                vec![text_finding("PERSON", "Zarnak Quorlim")],
                RedactionOffsetEncoding::UnicodeCodePoints,
                Some(10),
            )
            .await
            .expect("extraction must succeed");

        assert_eq!(output.results[0].content, MASK);
        assert_eq!(cache.gets.load(Ordering::SeqCst), 0);
        assert_eq!(cache.puts.load(Ordering::SeqCst), 0);

        let second = engine
            .extract_with_external_redaction(
                ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
                &ExtractionConfig::default(),
                vec![text_finding("PERSON", "Quorlim")],
                RedactionOffsetEncoding::UnicodeCodePoints,
                Some(10),
            )
            .await
            .expect("second extraction must succeed");
        assert_eq!(second.results[0].content, "Zarnak [REDACTED]");
        assert_eq!(cache.gets.load(Ordering::SeqCst), 0);
        assert_eq!(cache.puts.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn should_fail_closed_when_the_late_processor_is_disabled() {
    run_extraction_test(async {
        let config = ExtractionConfig {
            postprocessor: Some(xberg::PostProcessorConfig {
                enabled: false,
                ..Default::default()
            }),
            ..Default::default()
        };
        let error = extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak Quorlim".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Zarnak Quorlim")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect_err("an unconsumed request must fail");

        assert!(
            error.to_string().contains("did not reach the Late processor"),
            "{error}"
        );
    });
}

#[test]
fn should_reject_uri_input_and_leave_no_scope_after_an_error() {
    run_extraction_test(async {
        let error = extract_with_external_redaction(
            ExtractInput::from_uri("document.txt"),
            &ExtractionConfig::default(),
            vec![text_finding("PERSON", "Zarnak")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect_err("URI input must be rejected");
        assert!(error.to_string().contains("one bytes input"), "{error}");

        let ordinary = extract(
            ExtractInput::from_bytes(b"Zarnak".to_vec(), "text/plain", None),
            &ExtractionConfig::default(),
        )
        .await
        .expect("ordinary extraction after error");
        assert_eq!(ordinary.results[0].content, "Zarnak");
    });
}

#[test]
fn should_leave_no_scope_after_cancellation() {
    run_extraction_test(async {
        let token = xberg::cancellation::CancellationToken::new();
        token.cancel();
        let config = ExtractionConfig {
            cancel_token: Some(token),
            ..Default::default()
        };
        extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Zarnak")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect_err("cancelled extraction must fail");

        let ordinary = extract(
            ExtractInput::from_bytes(b"Zarnak".to_vec(), "text/plain", None),
            &ExtractionConfig::default(),
        )
        .await
        .expect("ordinary extraction after cancellation");
        assert_eq!(ordinary.results[0].content, "Zarnak");
    });
}

#[test]
fn should_expose_send_futures_and_send_sync_request_types() {
    fn assert_send<T: Send>(_value: T) {}
    fn assert_send_sync<T: Send + Sync>() {}

    let config = ExtractionConfig::default();
    assert_send(extract_with_external_redaction(
        ExtractInput::from_bytes(b"Zarnak".to_vec(), "text/plain", None),
        &config,
        vec![text_finding("PERSON", "Zarnak")],
        Some("unicode_code_points"),
        Some(10),
    ));
    assert_send_sync::<ExternalRedactionFinding>();
    assert_send_sync::<RedactionOffsetEncoding>();
}

#[cfg(feature = "office")]
#[test]
fn should_redact_before_docx_output_is_encoded() {
    use base64::Engine as _;
    use std::io::Read;
    use xberg::OutputFormat;

    let config = ExtractionConfig {
        output_format: OutputFormat::Custom("docx".to_string()),
        ..Default::default()
    };
    let output = run_extraction_test(async move {
        extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak Quorlim signed.".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Zarnak Quorlim")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect("DOCX extraction must succeed")
    });
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&output.results[0].content)
        .expect("base64 DOCX");
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).expect("DOCX archive");
    let mut xml = String::new();
    archive
        .by_name("word/document.xml")
        .expect("document XML")
        .read_to_string(&mut xml)
        .expect("read document XML");

    assert!(!xml.contains("Zarnak Quorlim"), "secret remained in DOCX XML");
    assert!(xml.contains(MASK), "redaction token missing from DOCX XML");
}

#[cfg(feature = "pdf")]
#[test]
fn should_redact_before_pdf_output_is_encoded() {
    use base64::Engine as _;
    use xberg::OutputFormat;

    let extracted = run_extraction_test(async {
        let config = ExtractionConfig {
            output_format: OutputFormat::Custom("pdf".to_string()),
            ..Default::default()
        };
        let rendered = extract_with_external_redaction(
            ExtractInput::from_bytes(b"Zarnak Quorlim signed.".to_vec(), "text/plain", None),
            &config,
            vec![text_finding("PERSON", "Zarnak Quorlim")],
            Some("unicode_code_points"),
            Some(10),
        )
        .await
        .expect("PDF rendering");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&rendered.results[0].content)
            .expect("base64 PDF");
        extract(
            ExtractInput::from_bytes(bytes, "application/pdf", None),
            &ExtractionConfig::default(),
        )
        .await
        .expect("PDF render and extraction")
    });

    assert!(!extracted.results[0].content.contains("Zarnak Quorlim"));
    assert!(extracted.results[0].content.contains(MASK));
}
