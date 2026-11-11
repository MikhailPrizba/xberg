//! Decompression-path, page-raster and OCR-only image extraction regressions.

use super::*;

/// Trace-span assertion for #985: `extract_images_with_data` must NOT be entered
/// when `extract_images` is false (the default).
///
/// This directly proves the decompression code path was skipped — complementing
/// `test_no_decompression_when_images_disabled` which only observes the output.
/// An event with target `xberg::pdf::native::images` and field
/// `event = "decompression_started"` is emitted at the top of
/// `extract_images_with_data`; absence of that event is structural proof the
/// function was not called.
#[test]
fn test_no_decompression_trace_when_images_disabled() {
    use std::sync::{Arc, Mutex};
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt as _};

    #[allow(clippy::type_complexity)]
    #[derive(Clone, Default)]
    struct EventCapture {
        events: Arc<Mutex<Vec<(String, Option<String>)>>>,
    }

    impl<S> tracing_subscriber::Layer<S> for EventCapture
    where
        S: tracing::Subscriber,
    {
        fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
            let target = event.metadata().target().to_owned();

            if target != "xberg::pdf::native::images" {
                return;
            }

            struct FieldVisitor(Option<String>);
            impl tracing::field::Visit for FieldVisitor {
                fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                    if field.name() == "event" {
                        self.0 = Some(value.to_owned());
                    }
                }
                fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                    if field.name() == "event" {
                        self.0 = Some(format!("{value:?}"));
                    }
                }
            }

            let mut visitor = FieldVisitor(None);
            event.record(&mut visitor);

            self.events.lock().unwrap().push((target, visitor.0));
        }
    }

    let path = test_documents_dir().join("pdf/embedded_images_tables.pdf");
    assert!(path.exists(), "missing fixture: {}", path.display());

    let capture = EventCapture::default();
    let capture_clone = capture.clone();

    // Enable DEBUG so the tracing event would be visible if the function ran.
    let filter = EnvFilter::new("debug");
    let subscriber = tracing_subscriber::registry().with(filter).with(capture_clone);

    let result = tracing::subscriber::with_default(subscriber, || {
        let config = ExtractionConfig::default();
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(extract_uri_document(&path, None, &config))
            .expect("extraction must succeed")
    });

    assert!(
        result.images.as_ref().is_none_or(|v| v.is_empty()),
        "images must be absent when extract_images=false"
    );

    let events = capture.events.lock().unwrap();
    let decompression_events: Vec<_> = events
        .iter()
        .filter(|(target, event_field)| {
            target == "xberg::pdf::native::images" && event_field.as_deref() == Some("decompression_started")
        })
        .collect();

    assert!(
        decompression_events.is_empty(),
        "extract_images_with_data must not be entered when extract_images=false; \
         got {} decompression_started event(s)",
        decompression_events.len()
    );
}

/// Regression test for #1077: PDFs containing embedded images (common in Acrobat Sign /
/// Word exports) must extract without "image dimension probe failed" errors. Every image
/// must be probeable by load_image_for_ocr, extract_image_metadata, and VLM pipelines —
/// raw pixel buffers are re-encoded to PNG, DCT images pass through as JPEG, and no image
/// may escape as a headerless buffer with an undeterminable format.
///
/// Fixture: user_reports/mp_axmp_rec_en.pdf — the original bug reproducer.
#[test]
fn test_regression_1077_raw_pdf_images_re_encoded_as_png() {
    use xberg::core::config::ImageExtractionConfig;

    let path = test_documents_dir().join("user_reports/mp_axmp_rec_en.pdf");
    if !path.exists() {
        eprintln!("SKIP: user_reports/mp_axmp_rec_en.pdf not present");
        return;
    }

    let config = ExtractionConfig {
        images: Some(ImageExtractionConfig {
            extract_images: true,
            ..Default::default()
        }),
        use_cache: false,
        ..Default::default()
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt
        .block_on(extract_uri_document(&path, None, &config))
        .expect("extraction of mp_axmp_rec_en.pdf must succeed without probe errors");

    let images = result
        .images
        .as_ref()
        .expect("images must be Some when extract_images=true");

    assert!(
        !images.is_empty(),
        "mp_axmp_rec_en.pdf must yield at least one extracted image"
    );

    for img in images.iter() {
        let magic = &img.data[..8.min(img.data.len())];
        let magic_matches = match img.format.as_ref() {
            "png" => img.data.starts_with(b"\x89PNG\r\n\x1a\n"),
            "jpeg" => img.data.starts_with(b"\xff\xd8\xff"),
            "gif" => img.data.starts_with(b"GIF8"),
            "tiff" => img.data.starts_with(b"II") || img.data.starts_with(b"MM"),
            "bmp" => img.data.starts_with(b"BM"),
            "webp" => img.data.len() >= 12 && &img.data[0..4] == b"RIFF" && &img.data[8..12] == b"WEBP",
            other => panic!(
                "image at index {} has non-probeable format {other:?} (the #1077 regression: \
                 headerless/unrecognized image buffer); first 8 bytes: {magic:02x?}",
                img.image_index
            ),
        };
        assert!(
            magic_matches,
            "image at index {} declares format={:?} but data does not start with the matching \
             magic bytes; first 8 bytes: {magic:02x?}",
            img.image_index, img.format
        );
    }
}

/// `image_indices` on chunks must be empty when image extraction is disabled.
#[cfg(feature = "chunking")]
#[test]
fn test_chunk_image_indices_empty_when_images_disabled() {
    use xberg::core::config::{ChunkingConfig, ImageExtractionConfig};
    let path = test_documents_dir().join("pdf/embedded_images_tables.pdf");
    let config = ExtractionConfig {
        output_format: OutputFormat::Markdown,
        images: Some(ImageExtractionConfig {
            extract_images: false,
            ..Default::default()
        }),
        chunking: Some(ChunkingConfig {
            max_characters: 500,
            ..Default::default()
        }),
        ..Default::default()
    };
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(extract_uri_document(&path, None, &config)).unwrap();

    if let Some(chunks) = result.chunks.as_ref() {
        for chunk in chunks {
            assert!(
                chunk.metadata.image_indices.is_empty(),
                "chunk must have no image_indices when extract_images=false"
            );
        }
    }
}
