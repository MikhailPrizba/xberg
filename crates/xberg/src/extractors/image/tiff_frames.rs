//! Multi-frame TIFF decoding with the same image decoder and budgets as standalone images.
use super::ImageExtractor;
use crate::core::config::ExtractionConfig;
use crate::extractors::security::SecurityLimits;
use crate::types::internal::InternalDocument;
use crate::types::internal_builder::InternalDocumentBuilder;
use crate::{Result, XbergError};
use image::ImageDecoder;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::sync::Arc;

fn tiff_error(error: impl std::fmt::Display) -> XbergError {
    XbergError::parsing(format!("TIFF extraction failed: {error}"))
}

pub(super) fn offsets(content: &[u8], limits: &SecurityLimits) -> Result<Vec<u64>> {
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(content)).map_err(tiff_error)?;
    let mut decoder_limits = tiff::decoder::Limits::default();
    decoder_limits.decoding_buffer_size = limits.max_content_size;
    decoder_limits.ifd_value_size = limits.max_content_size.min(1024 * 1024);
    decoder_limits.intermediate_buffer_size = limits.max_content_size;
    decoder = decoder.with_limits(decoder_limits);
    let mut offsets = Vec::new();
    loop {
        let offset = decoder
            .ifd_pointer()
            .ok_or_else(|| tiff_error("missing frame directory"))?
            .0;
        offsets.push(offset);
        crate::extractors::security::enforce_page_count(offsets.len(), limits.max_pages)
            .map_err(|e| XbergError::validation(e.to_string()))?;
        if !decoder.more_images() {
            return Ok(offsets);
        }
        decoder.next_image().map_err(tiff_error)?;
    }
}

/// A seekable view with a different first-IFD pointer. The TIFF decoder follows absolute
/// offsets into the original bytes, so compressed raster data is never copied or rewritten.
struct FrameReader {
    source: Arc<[u8]>,
    position: u64,
    header: [u8; 16],
    header_len: usize,
}
impl Read for FrameReader {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        let start = usize::try_from(self.position).unwrap_or(usize::MAX);
        if start >= self.source.len() {
            return Ok(0);
        }
        let count = output.len().min(self.source.len() - start);
        output[..count].copy_from_slice(&self.source[start..start + count]);
        if start < self.header_len {
            let n = count.min(self.header_len - start);
            output[..n].copy_from_slice(&self.header[start..start + n]);
        }
        self.position += count as u64;
        Ok(count)
    }
}
impl Seek for FrameReader {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        let value = match position {
            SeekFrom::Start(n) => i128::from(n),
            SeekFrom::Current(n) => i128::from(self.position) + i128::from(n),
            SeekFrom::End(n) => self.source.len() as i128 + i128::from(n),
        };
        self.position = u64::try_from(value)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid TIFF seek"))?;
        Ok(self.position)
    }
}
fn png_frame(source: Arc<[u8]>, offset: u64, limits: &SecurityLimits) -> Result<Vec<u8>> {
    let mut header = [0u8; 16];
    let header_len = source.len().min(16);
    header[..header_len].copy_from_slice(&source[..header_len]);
    let little = source.starts_with(b"II");
    let big = if little { header[2] == 43 } else { header[3] == 43 };
    if big {
        header[8..16].copy_from_slice(&if little {
            offset.to_le_bytes()
        } else {
            offset.to_be_bytes()
        });
    } else {
        let offset = u32::try_from(offset).map_err(tiff_error)?;
        header[4..8].copy_from_slice(&if little {
            offset.to_le_bytes()
        } else {
            offset.to_be_bytes()
        });
    }
    let reader = FrameReader {
        source,
        position: 0,
        header,
        header_len,
    };
    let mut decoder = image::codecs::tiff::TiffDecoder::new(std::io::BufReader::new(reader)).map_err(tiff_error)?;
    let (width, height) = decoder.dimensions();
    crate::extraction::image_decode::ImageDecodeBudget::from_security_limits(limits).validate(
        width,
        height,
        decoder.total_bytes(),
    )?;
    let peak = decoder
        .total_bytes()
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(64 * 1024))
        .ok_or_else(|| XbergError::validation("TIFF decode memory estimate overflow"))?;
    crate::extraction::image_decode::ImageDecodeBudget::from_security_limits(limits).validate(width, height, peak)?;
    let mut image_limits = image::Limits::default();
    image_limits.max_alloc = Some(limits.max_content_size as u64);
    decoder.set_limits(image_limits).map_err(tiff_error)?;
    let image = image::DynamicImage::from_decoder(decoder).map_err(tiff_error)?;
    let mut png = Cursor::new(Vec::new());
    image.write_to(&mut png, image::ImageFormat::Png).map_err(tiff_error)?;
    let bytes = png.into_inner();
    if bytes.len() > limits.max_content_size {
        return Err(XbergError::validation("TIFF frame PNG exceeds content limit"));
    }
    Ok(bytes)
}

pub(super) async fn extract(
    extractor: &ImageExtractor,
    content: &[u8],
    config: &ExtractionConfig,
    frames: Vec<u64>,
) -> Result<InternalDocument> {
    let source: Arc<[u8]> = Arc::from(content);
    let limits = config.security_limits.clone().unwrap_or_default();
    let mut builder = InternalDocumentBuilder::new("image");
    let mut pages = Vec::with_capacity(frames.len());
    let mut warnings = Vec::new();
    let mut all_tables = Vec::new();
    let mut total = 0usize;
    for (index, offset) in frames.into_iter().enumerate() {
        let image_source = Arc::clone(&source);
        let frame_limits = limits.clone();
        let png = tokio::task::spawn_blocking(move || png_frame(image_source, offset, &frame_limits))
            .await
            .map_err(tiff_error)??;
        let frame = extractor.extract_with_ocr(&png, "image/png", config).await?;
        let result = crate::extraction::derive::derive_extraction_result(frame, false, config.output_format.clone());
        total = total.saturating_add(result.content.len());
        if total > limits.max_content_size {
            return Err(XbergError::validation("TIFF transcript exceeds content limit"));
        }
        let number = (index + 1) as u32;
        builder.push_paragraph(&result.content, vec![], Some(number), None);
        warnings.extend(result.processing_warnings);
        let tables: Vec<_> = result
            .tables
            .into_iter()
            .map(|mut table| {
                table.page_number = number;
                Arc::new(table)
            })
            .collect();
        all_tables.extend(tables.iter().map(|table| table.as_ref().clone()));
        pages.push(crate::types::PageContent {
            page_number: number,
            is_blank: Some(result.content.trim().is_empty()),
            content: result.content,
            tables,
            image_indices: Vec::new(),
            image_preprocessing: None,
            hierarchy: None,
            layout_regions: None,
            speaker_notes: None,
            section_name: None,
            sheet_name: None,
            ocr_confidence: None,
            native_content: None,
        });
    }
    let mut doc = builder.build();
    doc.metadata.pages = Some(crate::types::PageStructure {
        total_count: pages.len() as u32,
        unit_type: crate::types::PageUnitType::Page,
        boundaries: None,
        pages: None,
    });
    doc.ocr_text_pages = pages.iter().map(|p| p.page_number).collect();
    doc.tables = all_tables;
    doc.prebuilt_pages = Some(pages);
    doc.processing_warnings = warnings;
    doc.metadata.ocr_used = true;
    doc.mime_type = "image/tiff".into();
    Ok(doc)
}
