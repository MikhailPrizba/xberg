//! Comprehensive concurrency and parallelism stress tests.
//!
//! Validates that the Xberg core handles concurrent operations correctly:
//! - Parallel extractions don't interfere with each other
//! - OCR processing is thread-safe and efficient
//! - Pipeline processing works correctly under concurrent load
//! - Cache access is safe with multiple readers/writers
//! - Registry access is thread-safe
//!
//! These tests ensure production workloads with high concurrency work correctly.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)]
// ~keep: test/bench binaries print by design; org logging policy exempts tests
// `tokio::spawn`ing an extraction future requires proving it `Send`. Under `full`
// features that proof walks into h2's `PollMessage` -> `slab::Entry` -> `PhantomData`
// chain (reqwest/hyper, pulled in by api/mcp/liter-llm), which exceeds the default
// 128 limit. The chain is entirely third-party — no xberg type appears in it — and
// rustc's own diagnostic recommends exactly this. ~keep
#![recursion_limit = "256"]

use async_trait::async_trait;
use std::sync::Arc;
use xberg::Result;
use xberg::core::config::{ExtractInput, ExtractionConfig, PostProcessorConfig};
use xberg::core::pipeline::run_pipeline;
use xberg::internal::{ElementKind, InternalDocument, InternalElement};
use xberg::plugins::registry::{get_document_extractor_registry, get_post_processor_registry};
use xberg::plugins::{DocumentExtractor, Plugin, PostProcessor, ProcessingStage};
use xberg::types::ExtractedDocument;

use std::time::Duration;
use tokio::time::timeout;

mod helpers;
use helpers::extract_bytes_document;
#[cfg(feature = "tokio-runtime")]
use helpers::{BytesInput, extract_bytes_documents};

fn trim_trailing_newlines(value: &str) -> &str {
    value.trim_end_matches(['\n', '\r'])
}

fn assert_text_content(actual: &str, expected: &str) {
    assert_eq!(
        trim_trailing_newlines(actual),
        expected,
        "Content mismatch after trimming trailing newlines"
    );
}

/// Test many concurrent extractions of different MIME types.
///
/// Validates that:
/// - Registry lookups don't block each other unnecessarily
/// - Different extractors can run in parallel
/// - No data races or corruption
#[tokio::test]
async fn test_concurrent_extractions_mixed_formats() {
    let config = ExtractionConfig::default();

    #[allow(unused_mut)]
    let mut test_cases = vec![
        (b"Plain text content" as &[u8], "text/plain"),
        (b"{\"key\": \"value\"}", "application/json"),
        (b"# Markdown\n\nContent here", "text/markdown"),
    ];

    #[cfg(feature = "xml")]
    test_cases.push((b"<root><item>XML content</item></root>" as &[u8], "application/xml"));

    let mut handles = vec![];
    for _ in 0..10 {
        for (data, mime_type) in &test_cases {
            let config = config.clone();
            let data = data.to_vec();
            let mime_type = mime_type.to_string();

            handles.push(tokio::spawn(async move {
                extract_bytes_document(&data, &mime_type, &config).await
            }));
        }
    }

    let results = timeout(Duration::from_secs(30), async {
        let mut results = vec![];
        for handle in handles {
            results.push(handle.await.expect("Task should not panic"));
        }
        results
    })
    .await
    .expect("All extractions should complete within 30s");

    for result in results {
        assert!(
            result.is_ok(),
            "Concurrent extraction should succeed: {:?}",
            result.err()
        );
    }
}

/// Test concurrent batch extractions.
///
/// Validates that batch processing correctly handles parallelism internally.
// Without Xberg's Tokio runtime feature, batch extraction uses the sequential
// path and intentionally erases its future without Send, so it cannot be spawned.
#[cfg(feature = "tokio-runtime")]
#[tokio::test]
async fn test_concurrent_batch_extractions() {
    let config = ExtractionConfig::default();

    let contents: Vec<Vec<u8>> = (0..20).map(|i| format!("Content {}", i).into_bytes()).collect();

    let mut handles = vec![];
    for _ in 0..5 {
        let config = config.clone();
        let contents_clone = contents.clone();

        handles.push(tokio::spawn(async move {
            let owned_data: Vec<BytesInput> = contents_clone
                .iter()
                .map(|c| BytesInput {
                    content: c.to_vec(),
                    mime_type: "text/plain".to_string(),
                    config: None,
                })
                .collect();
            extract_bytes_documents(owned_data, &config).await
        }));
    }

    for handle in handles {
        let results = handle.await.expect("Task should not panic");
        assert!(results.is_ok(), "Batch extraction should succeed");
        let results = results.expect("Operation failed");
        assert_eq!(results.len(), 20, "Should return all results");
    }
}

/// Test concurrent extractions with caching enabled.
///
/// Validates that:
/// - Cache reads/writes are thread-safe
/// - No cache corruption under concurrent access
/// - Cache hits work correctly across threads
#[tokio::test]
async fn test_concurrent_extractions_with_cache() {
    let config = ExtractionConfig {
        use_cache: true,
        postprocessor: Some(PostProcessorConfig {
            enabled: false,
            enabled_processors: None,
            disabled_processors: None,
            enabled_set: None,
            disabled_set: None,
        }),
        ..Default::default()
    };

    let test_data = b"Cached content for concurrent access test";

    let _ = extract_bytes_document(test_data, "text/plain", &config)
        .await
        .expect("Async operation failed");

    let mut handles = vec![];
    for _ in 0..100 {
        let config = config.clone();
        let data = test_data.to_vec();

        handles.push(tokio::spawn(async move {
            extract_bytes_document(&data, "text/plain", &config).await
        }));
    }

    let expected_content = "Cached content for concurrent access test";
    for handle in handles {
        let result = handle.await.expect("Task should not panic");
        assert!(result.is_ok(), "Cache read should succeed");
        let extraction = result.expect("Operation failed");
        assert_text_content(&extraction.content, expected_content);
    }
}

/// Test concurrent pipeline processing.
///
/// Validates that:
/// - Pipeline can process multiple results in parallel
/// - Processors don't interfere with each other
/// - Registry reads are thread-safe
///
/// Note: This test is flaky due to timing-dependent concurrent operations.
#[tokio::test]
#[ignore = "flaky concurrency test - timing dependent on system load"]
async fn test_concurrent_pipeline_processing() {
    struct ConcurrentTestProcessor;

    impl Plugin for ConcurrentTestProcessor {
        fn name(&self) -> &str {
            "concurrent-test"
        }
        fn version(&self) -> String {
            "1.0.0".to_string()
        }
        fn initialize(&self) -> Result<()> {
            Ok(())
        }
        fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl PostProcessor for ConcurrentTestProcessor {
        async fn process(&self, result: &mut ExtractedDocument, _: &ExtractionConfig) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            result.content.push_str("[processed]");
            Ok(())
        }

        fn processing_stage(&self) -> ProcessingStage {
            ProcessingStage::Early
        }
    }

    let registry = get_post_processor_registry();
    {
        let mut reg = registry.write();
        let processor = Arc::new(ConcurrentTestProcessor);
        let _ = reg.remove("concurrent-test");
        reg.register(processor).expect("Should register processor");
    }

    let config = ExtractionConfig {
        postprocessor: Some(PostProcessorConfig {
            enabled: true,
            enabled_processors: Some(vec!["concurrent-test".to_string()]),
            disabled_processors: None,
            enabled_set: None,
            disabled_set: None,
        }),
        ..Default::default()
    };

    let mut handles = vec![];
    for i in 0..50 {
        let config = config.clone();

        handles.push(tokio::spawn(async move {
            let mut doc = InternalDocument::new("text");
            doc.mime_type = "text/plain".to_string();
            doc.elements.push(InternalElement::text(
                ElementKind::Paragraph,
                format!("Content {}", i),
                0,
            ));

            run_pipeline(doc, &config).await
        }));
    }

    for handle in handles {
        let result = handle.await.expect("Task should not panic");
        assert!(result.is_ok(), "Pipeline should succeed");
        let processed = result.expect("Operation failed");
        assert!(processed.content.contains("[processed]"), "Processor should run");
    }

    {
        let mut reg = registry.write();
        let _ = reg.remove("concurrent-test");
    }
}

/// Test concurrent registry reads don't block unnecessarily.
///
/// Validates that:
/// - Multiple readers can access registry simultaneously
/// - Every reader resolves the same extractor the uncontended lookup does
#[tokio::test]
async fn test_concurrent_registry_reads() {
    const READERS: usize = 200;
    // A MIME type owned by this test alone. The global registry is process-wide and
    // keyed by MIME type, so a private key cannot change what any sibling test's
    // lookup returns. It has to be registered here because the global document
    // extractor registry holds no `text/plain` entry in this binary -- the earlier
    // version of this test looked one up and discarded the `Err`, so it asserted
    // nothing about the registry's contents. ~keep
    const MIME: &str = "application/x-xberg-gh1923";

    struct ConcurrentReadExtractor;

    impl Plugin for ConcurrentReadExtractor {
        fn name(&self) -> &str {
            "gh1923-concurrent-read"
        }
        fn version(&self) -> String {
            "1.0.0".to_string()
        }
        fn initialize(&self) -> Result<()> {
            Ok(())
        }
        fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl DocumentExtractor for ConcurrentReadExtractor {
        async fn extract(&self, _input: ExtractInput, _config: &ExtractionConfig) -> Result<ExtractedDocument> {
            Ok(ExtractedDocument::default())
        }

        fn supported_mime_types(&self) -> &[&str] {
            &[MIME]
        }
    }

    let registry = get_document_extractor_registry();
    {
        let mut reg = registry.write();
        reg.register(Arc::new(ConcurrentReadExtractor))
            .expect("Registration should succeed");
    }

    // The uncontended answer, taken before any reader is spawned, is the oracle the
    // concurrent reads are compared against.
    //
    // This replaced a `max_duration < 10ms` wall-clock bound, which measured the tokio
    // scheduler's queueing delay rather than the lock and failed on any loaded host
    // (GH#1923: `max duration: 17.042897ms` on a CI runner that never touched the
    // registry). A timing bound cannot distinguish a slow lock from a busy machine;
    // agreement across readers can, and is what a reader-writer lock actually promises. ~keep
    let oracle = {
        let reg = registry.read();
        reg.get(MIME).ok().map(|extractor| extractor.name().to_string())
    };
    assert_eq!(
        oracle.as_deref(),
        Some("gh1923-concurrent-read"),
        "the uncontended lookup must resolve, or the agreement assertion below compares None against None"
    );

    let mut handles = vec![];
    for _ in 0..READERS {
        let registry_clone = Arc::clone(&registry);
        handles.push(tokio::spawn(async move {
            let reg = registry_clone.read();
            reg.get(MIME).ok().map(|extractor| extractor.name().to_string())
        }));
    }

    let mut observed = Vec::with_capacity(READERS);
    for handle in handles {
        observed.push(handle.await.expect("Task should not panic"));
    }

    {
        let mut reg = registry.write();
        let _ = reg.remove("gh1923-concurrent-read");
    }

    assert_eq!(
        observed.len(),
        READERS,
        "every spawned reader must report; a short count means the comparison below examined less than it claims"
    );
    let disagreeing = observed.iter().filter(|name| **name != oracle).count();
    assert_eq!(
        disagreeing, 0,
        "{disagreeing} of {READERS} concurrent readers disagreed with the uncontended lookup {oracle:?}"
    );
}

/// Test that extraction throughput scales with concurrency.
///
/// Validates that:
/// - Parallel extractions are actually running in parallel
/// - No global bottlenecks limiting throughput
///
/// Note: This is a performance benchmark that can be flaky based on system load,
/// CPU availability, and other factors. Marked as #[ignore] to run only on demand.
#[tokio::test]
#[ignore]
async fn test_extraction_throughput_scales() {
    let config = ExtractionConfig::default();
    let test_data = b"Throughput test content";

    let sequential_start = std::time::Instant::now();
    for _ in 0..20 {
        let _ = extract_bytes_document(test_data, "text/plain", &config)
            .await
            .expect("Async operation failed");
    }
    let sequential_duration = sequential_start.elapsed();

    let parallel_start = std::time::Instant::now();
    let mut handles = vec![];
    for _ in 0..20 {
        let config = config.clone();
        let data = test_data.to_vec();

        handles.push(tokio::spawn(async move {
            extract_bytes_document(&data, "text/plain", &config).await
        }));
    }

    for handle in handles {
        let _ = handle.await.expect("Task should not panic");
    }
    let parallel_duration = parallel_start.elapsed();

    println!(
        "Sequential: {:?}, Parallel: {:?}, Speedup: {:.2}x",
        sequential_duration,
        parallel_duration,
        sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64()
    );

    let speedup = sequential_duration.as_secs_f64() / parallel_duration.as_secs_f64();

    assert!(
        speedup > 0.5,
        "Parallel execution should not be significantly slower than sequential. Sequential: {:?}, Parallel: {:?}, Speedup: {:.2}x",
        sequential_duration,
        parallel_duration,
        speedup
    );
}

/// High-load stress test with many concurrent operations.
///
/// Validates system stability under sustained concurrent load.
#[tokio::test]
async fn test_high_concurrency_stress() {
    let config = ExtractionConfig {
        use_cache: true,
        ..Default::default()
    };

    #[allow(unused_mut)]
    let mut formats = vec![
        (b"Text content" as &[u8], "text/plain"),
        (b"{\"json\": true}", "application/json"),
        (b"# Markdown\n\nContent", "text/markdown"),
    ];

    #[cfg(feature = "xml")]
    formats.push((b"<xml><item>content</item></xml>" as &[u8], "application/xml"));

    let mut handles = vec![];
    for _ in 0..100 {
        for (data, mime_type) in &formats {
            let config = config.clone();
            let data = data.to_vec();
            let mime_type = mime_type.to_string();

            handles.push(tokio::spawn(async move {
                extract_bytes_document(&data, &mime_type, &config).await
            }));
        }
    }

    let results = timeout(Duration::from_secs(60), async {
        let mut results = vec![];
        for handle in handles {
            results.push(handle.await.expect("Task should not panic"));
        }
        results
    })
    .await
    .expect("High-load stress test should complete within 60s");

    let expected_successes = 100 * formats.len();
    let success_count = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(
        success_count, expected_successes,
        "All extractions should succeed under stress, got {} successes",
        success_count
    );
}
