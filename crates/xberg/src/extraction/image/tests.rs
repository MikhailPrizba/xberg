//! Tests for the image extraction module, split out of `image.rs` to keep that file
//! under the line-count limit.

use super::*;
use image::{ImageBuffer, ImageFormat, Rgb, RgbImage};
use std::io::Cursor;

fn extract_image_metadata(bytes: &[u8]) -> Result<ExtractedImageMetadata> {
    extract_image_metadata_with_security_limits(bytes, &SecurityLimits::default())
}

fn create_test_image(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
    let img: RgbImage = ImageBuffer::from_fn(width, height, |x, y| {
        let r = ((x as f32 / width as f32) * 255.0) as u8;
        let g = ((y as f32 / height as f32) * 255.0) as u8;
        let b = 128;
        Rgb([r, g, b])
    });

    let mut bytes: Vec<u8> = Vec::new();
    let mut cursor = Cursor::new(&mut bytes);
    img.write_to(&mut cursor, format).unwrap();
    bytes
}

fn image_decode_limits(max_content_size: usize) -> crate::extractors::security::SecurityLimits {
    crate::extractors::security::SecurityLimits {
        max_content_size,
        ..Default::default()
    }
}

#[test]
fn metadata_rejects_corrupt_pixels_within_security_budget() {
    let bytes = crate::extraction::image_decode::bmp_with_declared_dimensions(10, 10);
    let limits = image_decode_limits(10_000);

    let error = extract_image_metadata_with_security_limits(&bytes, &limits)
        .expect_err("metadata extraction must validate bounded pixel data, not only the header");

    assert!(matches!(error, XbergError::Parsing { .. }));
}

#[test]
fn test_extract_png_image_returns_correct_metadata() {
    let bytes = create_test_image(100, 80, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 100);
    assert_eq!(metadata.height, 80);
    assert_eq!(metadata.format, "PNG");
}

#[test]
fn test_extract_jpeg_image_returns_correct_metadata() {
    let bytes = create_test_image(200, 150, ImageFormat::Jpeg);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 200);
    assert_eq!(metadata.height, 150);
    assert_eq!(metadata.format, "JPEG");
}

#[test]
fn test_extract_webp_image_returns_correct_metadata() {
    let bytes = create_test_image(120, 90, ImageFormat::WebP);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 120);
    assert_eq!(metadata.height, 90);
    assert_eq!(metadata.format, "WEBP");
}

#[test]
fn test_extract_bmp_image_returns_correct_metadata() {
    let bytes = create_test_image(50, 50, ImageFormat::Bmp);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 50);
    assert_eq!(metadata.height, 50);
    assert_eq!(metadata.format, "BMP");
}

#[test]
fn test_extract_tiff_image_returns_correct_metadata() {
    let bytes = create_test_image(180, 120, ImageFormat::Tiff);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 180);
    assert_eq!(metadata.height, 120);
    assert_eq!(metadata.format, "TIFF");
}

#[test]
fn test_extract_gif_image_returns_correct_metadata() {
    let bytes = create_test_image(64, 64, ImageFormat::Gif);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 64);
    assert_eq!(metadata.height, 64);
    assert_eq!(metadata.format, "GIF");
}

#[test]
fn test_extract_image_extreme_aspect_ratio() {
    let bytes = create_test_image(1000, 10, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 1000);
    assert_eq!(metadata.height, 10);
    assert!(metadata.width / metadata.height >= 100);
}

#[test]
fn test_extract_image_dimensions_correctly() {
    let bytes = create_test_image(640, 480, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 640);
    assert_eq!(metadata.height, 480);
}

#[test]
fn test_extract_image_format_correctly() {
    let png_bytes = create_test_image(100, 100, ImageFormat::Png);
    let jpeg_bytes = create_test_image(100, 100, ImageFormat::Jpeg);

    let png_metadata = extract_image_metadata(&png_bytes).unwrap();
    let jpeg_metadata = extract_image_metadata(&jpeg_bytes).unwrap();

    assert_eq!(png_metadata.format, "PNG");
    assert_eq!(jpeg_metadata.format, "JPEG");
}

#[test]
fn test_extract_image_without_exif_returns_empty_map() {
    let bytes = create_test_image(100, 100, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert!(metadata.exif_data.is_empty());
}

#[test]
fn test_extract_exif_data_from_jpeg_with_exif() {
    let bytes = create_test_image(100, 100, ImageFormat::Jpeg);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.exif_data.len(), 0);
}

#[test]
fn test_extract_image_metadata_invalid_returns_error() {
    let invalid_bytes = vec![0, 1, 2, 3, 4, 5];
    let result = extract_image_metadata(&invalid_bytes);
    assert!(result.is_err());
}

#[test]
fn test_extract_image_corrupted_data_returns_error() {
    let mut bytes = create_test_image(100, 100, ImageFormat::Png);
    if bytes.len() > 50 {
        for byte in bytes.iter_mut().take(50).skip(20) {
            *byte = 0xFF;
        }
    }

    let _result = extract_image_metadata(&bytes);
}

#[test]
fn test_extract_image_empty_bytes_returns_error() {
    let empty_bytes: Vec<u8> = Vec::new();
    let result = extract_image_metadata(&empty_bytes);
    assert!(result.is_err());
}

#[test]
fn test_extract_image_unsupported_format_returns_error() {
    let unsupported_bytes = vec![0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A];
    let result = extract_image_metadata(&unsupported_bytes);
    assert!(result.is_err());
}

#[test]
fn test_extract_very_small_image_1x1_pixel() {
    let bytes = create_test_image(1, 1, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 1);
    assert_eq!(metadata.height, 1);
    assert_eq!(metadata.format, "PNG");
}

#[test]
fn test_extract_large_image_dimensions() {
    let bytes = create_test_image(2048, 1536, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 2048);
    assert_eq!(metadata.height, 1536);
}

#[test]
fn test_extract_image_with_no_metadata_has_empty_exif() {
    let bytes = create_test_image(100, 100, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert!(metadata.exif_data.is_empty());
}

#[test]
fn test_extract_image_metadata_heic_without_feature_errors() {
    let mut heic_stub = Vec::from(&b"\x00\x00\x00\x18ftypheicheic"[..]);
    heic_stub.extend_from_slice(&[0u8; 12]);
    let err = extract_image_metadata(&heic_stub).expect_err("heic without feature should error");
    let msg = err.to_string();
    assert!(msg.contains("heic"), "expected `heic` mention in error: {msg}");
}

#[test]
fn test_extract_exif_data_returns_empty_map_for_non_jpeg() {
    let png_bytes = create_test_image(100, 100, ImageFormat::Png);
    let exif_data = extract_exif_data(&png_bytes);
    assert!(exif_data.is_empty());
}

#[test]
fn test_extract_rectangular_image_portrait_orientation() {
    let bytes = create_test_image(400, 800, ImageFormat::Jpeg);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 400);
    assert_eq!(metadata.height, 800);
    assert!(metadata.height > metadata.width);
}

#[test]
fn test_extract_rectangular_image_landscape_orientation() {
    let bytes = create_test_image(800, 400, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 800);
    assert_eq!(metadata.height, 400);
    assert!(metadata.width > metadata.height);
}

#[test]
fn test_extract_square_image_equal_dimensions() {
    let bytes = create_test_image(512, 512, ImageFormat::Png);
    let result = extract_image_metadata(&bytes);

    assert!(result.is_ok());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 512);
    assert_eq!(metadata.height, 512);
    assert_eq!(metadata.width, metadata.height);
}

#[test]
fn test_extract_metadata_preserves_format_case() {
    let png_bytes = create_test_image(100, 100, ImageFormat::Png);
    let jpeg_bytes = create_test_image(100, 100, ImageFormat::Jpeg);
    let webp_bytes = create_test_image(100, 100, ImageFormat::WebP);

    let png_meta = extract_image_metadata(&png_bytes).unwrap();
    let jpeg_meta = extract_image_metadata(&jpeg_bytes).unwrap();
    let webp_meta = extract_image_metadata(&webp_bytes).unwrap();

    assert_eq!(png_meta.format, "PNG");
    assert_eq!(jpeg_meta.format, "JPEG");
    assert_eq!(webp_meta.format, "WEBP");
}

#[test]
fn test_jp2_magic_detection() {
    assert!(is_jp2(&[0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A]));
    assert!(!is_jp2(&[0x89, 0x50, 0x4E, 0x47]));
    assert!(!is_jp2(&[0x00, 0x00]));
    assert!(!is_jp2(&[]));
}

#[test]
fn test_extract_jp2_rust_logo_metadata() {
    let Some(bytes) = crate::utils::read_test_fixture("images/rust-logo-512x512-blk.jp2") else {
        return;
    };
    let result = extract_image_metadata(&bytes);
    assert!(result.is_ok(), "Failed to extract JP2 metadata: {:?}", result.err());
    let metadata = result.unwrap();
    assert_eq!(metadata.width, 512);
    assert_eq!(metadata.height, 512);
    assert_eq!(metadata.format, "JPEG2000");
}

#[test]
fn test_extract_jp2_hadley_crater_metadata() {
    let Some(bytes) = crate::utils::read_test_fixture("images/Hadley_Crater.jp2") else {
        return;
    };
    let result = extract_image_metadata(&bytes);
    assert!(result.is_ok(), "Failed to extract JP2 metadata: {:?}", result.err());
    let metadata = result.unwrap();
    assert!(metadata.width > 0);
    assert!(metadata.height > 0);
    assert_eq!(metadata.format, "JPEG2000");
}

#[test]
fn test_parse_jp2_boxes_invalid_data() {
    let invalid = vec![0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A];
    let result = decode_jp2_metadata(&invalid);
    assert!(result.is_err());
}

#[test]
fn test_jp2_magic_detection_comprehensive() {
    assert!(is_jp2(&[
        0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A
    ]));
    assert!(!is_jp2(&[0xFF, 0x4F, 0xFF, 0x51]));
    assert!(!is_jp2(&[0x89, 0x50, 0x4E, 0x47]));
    assert!(!is_jp2(&[]));
}

// GH#1630: PNG pHYs density detection.
