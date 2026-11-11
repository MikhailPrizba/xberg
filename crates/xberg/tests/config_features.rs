//! Configuration features integration tests.
//!
//! Tests for chunking, language detection, caching, token reduction, and quality processing.
//! Validates that configuration options work correctly end-to-end.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests

#[cfg(feature = "chunking")]
use xberg::core::config::ChunkingConfig;
use xberg::core::config::ExtractionConfig;
#[cfg(feature = "language-detection")]
use xberg::core::config::LanguageDetectionConfig;
use xberg::core::config::TokenReductionOptions;

mod helpers;
use helpers::extract_bytes_document;

/// Test chunking enabled - text split into chunks.
#[tokio::test]
#[cfg(feature = "chunking")]
async fn test_chunking_enabled() {
    let config = ExtractionConfig {
        chunking: Some(ChunkingConfig {
            max_characters: 50,
            overlap: 10,
            ..Default::default()
        }),
        ..Default::default()
    };

    let text = "This is a long text that should be split into multiple chunks. ".repeat(10);
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_some(), "Chunks should be present");
    let chunks = result.chunks.expect("Operation failed");
    assert!(chunks.len() > 1, "Should have multiple chunks");

    for chunk in &chunks {
        assert!(!chunk.content.is_empty(), "Chunk should not be empty");
        assert!(
            chunk.content.len() <= 50 + 10,
            "Chunk length {} exceeds max_chars + overlap",
            chunk.content.len()
        );
    }
}

/// Test chunking with overlap - overlap preserved between chunks.
#[tokio::test]
#[cfg(feature = "chunking")]
async fn test_chunking_with_overlap() {
    let config = ExtractionConfig {
        chunking: Some(ChunkingConfig {
            max_characters: 100,
            overlap: 20,
            ..Default::default()
        }),
        ..Default::default()
    };

    let text = "a".repeat(250);
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_some(), "Chunks should be present");
    let chunks = result.chunks.expect("Operation failed");
    assert!(chunks.len() >= 2, "Should have at least 2 chunks");

    if chunks.len() >= 2 {
        let chunk1 = &chunks[0];
        let chunk2 = &chunks[1];

        let chunk1_end = &chunk1.content[chunk1.content.len().saturating_sub(20)..];
        assert!(
            chunk2.content.starts_with(chunk1_end)
                || chunk1_end.starts_with(&chunk2.content[..chunk1_end.len().min(chunk2.content.len())]),
            "Chunks should have overlap"
        );
    }
}

/// Test chunking with custom sizes - custom chunk size and overlap.
#[tokio::test]
#[cfg(feature = "chunking")]
async fn test_chunking_custom_sizes() {
    let config = ExtractionConfig {
        chunking: Some(ChunkingConfig {
            max_characters: 200,
            overlap: 50,
            ..Default::default()
        }),
        ..Default::default()
    };

    let text = "Custom chunk test. ".repeat(50);
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_some(), "Chunks should be present");
    let chunks = result.chunks.expect("Operation failed");
    assert!(!chunks.is_empty(), "Should have at least 1 chunk");

    for chunk in &chunks {
        assert!(
            chunk.content.len() <= 200 + 50,
            "Chunk length {} exceeds custom max_chars + overlap",
            chunk.content.len()
        );
    }
}

/// Test chunking disabled - no chunking when disabled.
#[tokio::test]
async fn test_chunking_disabled() {
    let config = ExtractionConfig {
        chunking: None,
        ..Default::default()
    };

    let text = "This is a long text that should NOT be split into chunks. ".repeat(10);
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_none(), "Should not have chunks when chunking disabled");

    assert!(!result.content.is_empty(), "Content should be extracted");
    assert!(result.content.contains("long text"), "Should contain original text");
}

/// Test language detection for single language document.
#[tokio::test]
#[cfg(feature = "language-detection")]
async fn test_language_detection_single() {
    let config = ExtractionConfig {
        language_detection: Some(LanguageDetectionConfig {
            enabled: true,
            min_confidence: 0.8,
            detect_multiple: false,
        }),
        ..Default::default()
    };

    let text = "Hello world! This is English text. It should be detected as English language.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.detected_languages.is_some(), "Should detect language");
    let languages = result.detected_languages.expect("Operation failed");
    assert!(!languages.is_empty(), "Should detect at least one language");
    assert_eq!(languages[0], "eng", "Should detect English");
}

/// Test language detection for multi-language document.
#[cfg_attr(coverage, ignore = "coverage instrumentation affects multi-language heuristics")]
#[tokio::test]
#[cfg(feature = "language-detection")]
async fn test_language_detection_multiple() {
    let config = ExtractionConfig {
        language_detection: Some(LanguageDetectionConfig {
            enabled: true,
            min_confidence: 0.5,
            detect_multiple: true,
        }),
        ..Default::default()
    };

    // Use lexically rich passages rather than short repetitive phrases: whatlang
    // scores terse strings like "Hello world! This is English." near the 0.7
    // boundary, so the old assertion failed whenever no chunk cleared the
    // threshold. Longer sentences clear 0.5 deterministically for both languages.
    let text = format!(
        "{}{}",
        "The global economy has been experiencing significant changes in recent years. \
         International cooperation is essential for addressing climate change. "
            .repeat(5),
        "La economía global ha estado experimentando cambios significativos en los últimos años. \
         La cooperación internacional es esencial para abordar el cambio climático. "
            .repeat(5)
    );
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.detected_languages.is_some(), "Should detect languages");
    let languages = result.detected_languages.expect("Operation failed");
    assert!(!languages.is_empty(), "Should detect at least one language");
}

/// Test language detection with confidence threshold.
#[tokio::test]
#[cfg(feature = "language-detection")]
async fn test_language_detection_confidence() {
    let config = ExtractionConfig {
        language_detection: Some(LanguageDetectionConfig {
            enabled: true,
            min_confidence: 0.9,
            detect_multiple: false,
        }),
        ..Default::default()
    };

    let text = "This is clear English text that should have high confidence.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    if let Some(languages) = result.detected_languages {
        assert!(!languages.is_empty());
    }
}

/// Test language detection disabled.
#[tokio::test]
#[cfg(feature = "language-detection")]
async fn test_language_detection_disabled() {
    let config = ExtractionConfig {
        language_detection: Some(LanguageDetectionConfig {
            enabled: false,
            min_confidence: 0.8,
            detect_multiple: false,
        }),
        ..Default::default()
    };

    let text = "Hello world! This is English text.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(
        result.detected_languages.is_none(),
        "Should not detect language when disabled"
    );
}

/// Test cache hit behavior - second extraction from cache.
#[tokio::test]
async fn test_cache_hit_behavior() {
    let config = ExtractionConfig {
        use_cache: true,
        ..Default::default()
    };

    let text = "Test text for caching behavior.";
    let text_bytes = text.as_bytes();

    let result1 = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("First extraction should succeed");

    let result2 = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Second extraction should succeed");

    assert_eq!(result1.content, result2.content);
}

/// Test cache miss and invalidation.
#[tokio::test]
async fn test_cache_miss_invalidation() {
    let config = ExtractionConfig {
        use_cache: true,
        ..Default::default()
    };

    let text1 = "First text for cache test.";
    let text2 = "Second different text.";

    let result1 = extract_bytes_document(text1.as_bytes(), "text/plain", &config)
        .await
        .expect("First extraction should succeed");

    let result2 = extract_bytes_document(text2.as_bytes(), "text/plain", &config)
        .await
        .expect("Second extraction should succeed");

    assert_ne!(result1.content, result2.content);
}

/// Test custom cache directory (Note: OCR cache uses hardcoded directory).
#[tokio::test]
async fn test_custom_cache_directory() {
    let config = ExtractionConfig {
        use_cache: true,
        ..Default::default()
    };

    let text = "Test text for cache directory test.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(!result.content.is_empty());
}

/// Test cache disabled - bypass cache.
#[tokio::test]
async fn test_cache_disabled() {
    let config = ExtractionConfig {
        use_cache: false,
        ..Default::default()
    };

    let text = "Test text without caching.";
    let text_bytes = text.as_bytes();

    let result1 = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("First extraction should succeed");

    let result2 = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Second extraction should succeed");

    assert_eq!(result1.content, result2.content);
}

/// Test token reduction in aggressive mode.
#[tokio::test]
async fn test_token_reduction_aggressive() {
    let config = ExtractionConfig {
        token_reduction: Some(TokenReductionOptions {
            mode: "aggressive".to_string(),
            preserve_important_words: true,
        }),
        ..Default::default()
    };

    let text = "This is a very long sentence with many unnecessary words that could be reduced. ".repeat(5);
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(!result.content.is_empty());
}

/// Test token reduction in conservative mode.
#[tokio::test]
async fn test_token_reduction_conservative() {
    let config = ExtractionConfig {
        token_reduction: Some(TokenReductionOptions {
            mode: "light".to_string(),
            preserve_important_words: true,
        }),
        ..Default::default()
    };

    let text = "Conservative token reduction test with moderate text length.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(!result.content.is_empty());
}

/// Test token reduction disabled.
#[tokio::test]
async fn test_token_reduction_disabled() {
    let config = ExtractionConfig {
        token_reduction: Some(TokenReductionOptions {
            mode: "off".to_string(),
            preserve_important_words: false,
        }),
        ..Default::default()
    };

    let text = "Text without token reduction applied.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.content.contains("without token reduction"));
}

/// Test quality processing enabled - quality scoring applied.
#[tokio::test]
#[cfg(feature = "quality")]
async fn test_quality_processing_enabled() {
    let config = ExtractionConfig {
        enable_quality_processing: true,
        ..Default::default()
    };

    let text = "This is well-structured text. It has multiple sentences. And proper punctuation.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    if let Some(score_value) = result.quality_score {
        assert!((0.0..=1.0).contains(&score_value));
    }

    assert!(!result.content.is_empty());
}

/// Test quality processing calculates score for different text quality.
#[tokio::test]
#[cfg(feature = "quality")]
async fn test_quality_threshold_filtering() {
    let config = ExtractionConfig {
        enable_quality_processing: true,
        ..Default::default()
    };

    let high_quality = "This is a well-structured document. It has proper sentences. And good formatting.";
    let result_high = extract_bytes_document(high_quality.as_bytes(), "text/plain", &config)
        .await
        .expect("Should extract successfully");

    let low_quality = "a  b  c  d  ....... word123mixed .  . ";
    let result_low = extract_bytes_document(low_quality.as_bytes(), "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result_high.quality_score.is_some(), "High quality should have score");
    assert!(result_low.quality_score.is_some(), "Low quality should have score");

    let score_high = result_high.quality_score.expect("High quality should have score");
    let score_low = result_low.quality_score.expect("Low quality should have score");

    assert!((0.0..=1.0).contains(&score_high));
    assert!((0.0..=1.0).contains(&score_low));
}

/// Test quality processing disabled.
#[tokio::test]
async fn test_quality_processing_disabled() {
    let config = ExtractionConfig {
        enable_quality_processing: false,
        ..Default::default()
    };

    let text = "Text without quality processing.";
    let text_bytes = text.as_bytes();

    let result = extract_bytes_document(text_bytes, "text/plain", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.quality_score.is_none());
    assert!(!result.content.is_empty());
}

/// Test markdown chunker populates heading context.
#[tokio::test]
#[cfg(feature = "chunking")]
async fn test_markdown_chunker_heading_context() {
    let markdown = r#"# Title

Some intro text.

## Section One

Content in section one with enough text to create a chunk.

## Section Two

Content in section two with enough text to create another chunk.

### Subsection

More detailed content here in the subsection.
"#;

    let config = ExtractionConfig {
        chunking: Some(ChunkingConfig {
            max_characters: 80,
            overlap: 10,
            chunker_type: xberg::ChunkerType::Markdown,
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = extract_bytes_document(markdown.as_bytes(), "text/markdown", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_some(), "Chunks should be present");
    let chunks = result.chunks.expect("Should have chunks");
    assert!(chunks.len() >= 2, "Should have at least 2 chunks");

    let has_heading = chunks.iter().any(|c| c.metadata.heading_context.is_some());
    assert!(has_heading, "At least one chunk should have heading_context");

    for chunk in &chunks {
        if let Some(ref ctx) = chunk.metadata.heading_context {
            for heading in &ctx.headings {
                assert!(heading.level >= 1 && heading.level <= 6, "Heading level should be 1-6");
                assert!(!heading.text.is_empty(), "Heading text should not be empty");
            }
        }
    }
}

/// Test that chunk_type is populated for markdown chunks.
#[tokio::test]
#[cfg(feature = "chunking")]
async fn test_chunk_type_populated() {
    let markdown = r#"# Introduction

This section introduces the document with some content.

## Code Example

```rust
fn hello() {
    println!("Hello, world!");
}
```

## Summary

A brief summary of the document.
"#;

    let config = ExtractionConfig {
        chunking: Some(ChunkingConfig {
            max_characters: 200,
            overlap: 0,
            chunker_type: xberg::ChunkerType::Markdown,
            ..Default::default()
        }),
        ..Default::default()
    };

    let result = extract_bytes_document(markdown.as_bytes(), "text/markdown", &config)
        .await
        .expect("Should extract successfully");

    assert!(result.chunks.is_some(), "Chunks should be present");
    let chunks = result.chunks.expect("Should have chunks");
    assert!(!chunks.is_empty(), "Should have at least one chunk");

    for chunk in &chunks {
        assert!(!chunk.content.is_empty(), "Chunk should not be empty");
        let _ = &chunk.chunk_type;
    }

    let has_heading_context = chunks.iter().any(|c| c.metadata.heading_context.is_some());
    assert!(
        has_heading_context,
        "At least one chunk should have heading context from markdown structure"
    );
}
