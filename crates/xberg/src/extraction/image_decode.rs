use std::io::Cursor;

#[cfg(any(test, feature = "ocr-pipeline"))]
use image::ColorType;
use image::{ImageDecoder, ImageFormat, ImageReader};

use crate::error::{Result, XbergError};
use crate::extractors::security::SecurityLimits;

#[derive(Clone, Copy)]
pub(crate) struct ImageDecodeBudget {
    max_decoded_bytes: u64,
}

impl ImageDecodeBudget {
    pub(crate) fn from_security_limits(limits: &SecurityLimits) -> Self {
        Self {
            max_decoded_bytes: u64::try_from(limits.max_content_size).unwrap_or(u64::MAX),
        }
    }

    /// Reject an image whose dimensions are degenerate or whose live
    /// image-processing footprint exceeds the budget.
    ///
    /// `decoded_bytes` and the budget are both byte counts. GH#1761: the
    /// pixel count `width * height` used to be compared against the byte
    /// budget as well — different units. It decided nothing, because every
    /// caller passes a byte count at least as large as the pixel count and
    /// the byte clause therefore always rejected first, but it would have
    /// become wrong the moment either side changed. Callers that only have
    /// a byte count to check pass `1, 1` for the dimensions. ~keep
    pub(crate) fn validate(self, width: u32, height: u32, decoded_bytes: u64) -> Result<()> {
        // A backstop for a future widening of the dimension types: with
        // u32 extents widened to u64 the product cannot overflow, since
        // u32::MAX squared is just under u64::MAX. ~keep
        u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| image_dimension_error(width, height, decoded_bytes, self.max_decoded_bytes))?;
        if width == 0 || height == 0 || decoded_bytes > self.max_decoded_bytes {
            return Err(image_dimension_error(
                width,
                height,
                decoded_bytes,
                self.max_decoded_bytes,
            ));
        }
        Ok(())
    }
}

pub(crate) fn image_dimension_error(width: u32, height: u32, live_bytes: u64, max_decoded_bytes: u64) -> XbergError {
    XbergError::Validation {
        message: format!(
            "Image dimensions {width}x{height} require {live_bytes} live image-processing bytes, exceeding or invalid under \
             security_limits.max_content_size ({max_decoded_bytes} bytes)"
        ),
        source: None,
    }
}

pub(crate) fn decoded_byte_count(width: u32, height: u32, bytes_per_pixel: u64) -> Result<u64> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .ok_or_else(|| image_dimension_error(width, height, u64::MAX, u64::MAX))
}

fn image_decode_limits(budget: ImageDecodeBudget) -> image::Limits {
    let mut limits = image::Limits::default();
    limits.max_alloc = Some(budget.max_decoded_bytes);
    limits
}

#[derive(Clone, Copy)]
struct StandardImageProbe {
    width: u32,
    height: u32,
    format: ImageFormat,
    #[cfg(any(test, feature = "ocr-pipeline"))]
    color_type: ColorType,
    decoded_bytes: u64,
}

fn map_image_decode_error(error: image::ImageError) -> XbergError {
    if matches!(error, image::ImageError::Limits(_)) {
        XbergError::Validation {
            message: format!("Image exceeds security_limits.max_content_size while decoding: {error}"),
            source: Some(Box::new(error)),
        }
    } else {
        XbergError::parsing(format!("Failed to decode image: {error}"))
    }
}

fn probe_standard_image(
    bytes: &[u8],
    budget: ImageDecodeBudget,
    format: Option<ImageFormat>,
) -> Result<StandardImageProbe> {
    let encoded_bytes = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    budget.validate(1, 1, encoded_bytes)?;
    let mut reader = match format {
        Some(format) => ImageReader::with_format(Cursor::new(bytes), format),
        None => ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|error| XbergError::parsing(format!("Failed to read image format: {error}")))?,
    };
    let format = reader
        .format()
        .ok_or_else(|| XbergError::parsing("Could not determine image format".to_string()))?;
    reader.limits(image_decode_limits(budget));
    let decoder = reader.into_decoder().map_err(map_image_decode_error)?;
    let (width, height) = decoder.dimensions();
    let decoded_bytes = decoder.total_bytes();
    budget.validate(width, height, decoded_bytes)?;
    Ok(StandardImageProbe {
        width,
        height,
        format,
        #[cfg(any(test, feature = "ocr-pipeline"))]
        color_type: decoder.color_type(),
        decoded_bytes,
    })
}

#[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
pub(crate) fn probe_standard_image_with_security_limits(
    bytes: &[u8],
    limits: &SecurityLimits,
) -> Result<(u32, u32, image::ImageFormat)> {
    let probe = probe_standard_image(bytes, ImageDecodeBudget::from_security_limits(limits), None)?;
    Ok((probe.width, probe.height, probe.format))
}

#[cfg(feature = "image-encode")]
pub(crate) fn decode_standard_image_with_format_and_security_limits(
    bytes: &[u8],
    format: ImageFormat,
    limits: &SecurityLimits,
) -> Result<image::DynamicImage> {
    decode_standard_image(bytes, limits, Some(format))
}

// Callers live behind their own feature gates -- `core::image_encode` (`image-encode`),
// `extractors::pdf::ocr::rendering` and `engine::structured::rasterize` (`pdf`), and
// `extraction::image` (`ocr`/`ocr-wasm`/`ocr-pipeline`) -- so an ungated definition is dead code
// in any combination that selects none of them (e.g. CI's `--no-default-features --features
// layout-tract`, which denies warnings). `test` keeps `decode_for_encode_under_test` compiling. ~keep
#[cfg(any(test, feature = "image-encode", feature = "pdf", feature = "ocr-pipeline"))]
pub(crate) fn decode_standard_image_with_security_limits(
    bytes: &[u8],
    limits: &SecurityLimits,
) -> Result<image::DynamicImage> {
    decode_standard_image(bytes, limits, None)
}

// Same reasoning as `decode_standard_image_with_security_limits` above; this one is additionally
// reached from `clone_dynamic_image_to_rgb8_with_security_limits`, whose own gate
// (`layout-detection` + `pdf` + `ocr`/`ocr-wasm`) is already covered by the `pdf` arm here. ~keep
#[cfg(any(test, feature = "image-encode", feature = "pdf", feature = "ocr-pipeline"))]
pub(crate) fn validate_dynamic_image_additional_live_bytes(
    image: &image::DynamicImage,
    limits: &SecurityLimits,
    additional_bytes_per_pixel: u64,
    fixed_additional_bytes: u64,
) -> Result<()> {
    let width = image.width();
    let height = image.height();
    let live_bytes =
        u64::try_from(image.as_bytes().len()).map_err(|_| image_dimension_error(width, height, u64::MAX, u64::MAX))?;
    let additional_bytes = decoded_byte_count(width, height, additional_bytes_per_pixel)?
        .checked_add(fixed_additional_bytes)
        .ok_or_else(|| image_dimension_error(width, height, u64::MAX, u64::MAX))?;
    validate_image_live_bytes(width, height, live_bytes, additional_bytes, limits)
}

pub(crate) fn validate_image_live_bytes(
    width: u32,
    height: u32,
    current_live_bytes: u64,
    additional_live_bytes: u64,
    limits: &SecurityLimits,
) -> Result<()> {
    let peak_bytes = current_live_bytes
        .checked_add(additional_live_bytes)
        .ok_or_else(|| image_dimension_error(width, height, u64::MAX, u64::MAX))?;
    ImageDecodeBudget::from_security_limits(limits).validate(width, height, peak_bytes)
}

// Private helper behind both public decode wrappers, so it is live exactly when either is. ~keep
#[cfg(any(test, feature = "image-encode", feature = "pdf", feature = "ocr-pipeline"))]
fn decode_standard_image(
    bytes: &[u8],
    limits: &SecurityLimits,
    format: Option<ImageFormat>,
) -> Result<image::DynamicImage> {
    let budget = ImageDecodeBudget::from_security_limits(limits);
    let probe = probe_standard_image(bytes, budget, format)?;
    let encoded_bytes =
        u64::try_from(bytes.len()).map_err(|_| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?;
    let peak_bytes = probe
        .decoded_bytes
        .checked_add(encoded_bytes)
        .ok_or_else(|| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?;
    budget.validate(probe.width, probe.height, peak_bytes)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), probe.format);
    reader.limits(image_decode_limits(budget));
    reader.decode().map_err(map_image_decode_error)
}

#[cfg(any(test, feature = "ocr-pipeline"))]
fn conversion_peak_bytes(
    probe: StandardImageProbe,
    target: ColorType,
    encoded_live_bytes: u64,
    additional_live_bytes: u64,
) -> Result<u64> {
    let target_bytes = decoded_byte_count(probe.width, probe.height, u64::from(target.bytes_per_pixel()))?;
    let conversion_peak = if probe.color_type == target {
        target_bytes
    } else {
        probe
            .decoded_bytes
            .checked_add(target_bytes)
            .ok_or_else(|| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?
    };
    let post_conversion_peak = target_bytes
        .checked_add(additional_live_bytes)
        .ok_or_else(|| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?;
    conversion_peak
        .max(post_conversion_peak)
        .checked_add(encoded_live_bytes)
        .ok_or_else(|| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))
}

#[cfg(any(test, feature = "ocr-pipeline"))]
fn decode_standard_rgb8(bytes: &[u8], limits: &SecurityLimits, additional_live_bytes: u64) -> Result<image::RgbImage> {
    let budget = ImageDecodeBudget::from_security_limits(limits);
    let probe = probe_standard_image(bytes, budget, None)?;
    let encoded_live_bytes =
        u64::try_from(bytes.len()).map_err(|_| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?;
    let peak_bytes = conversion_peak_bytes(probe, ColorType::Rgb8, encoded_live_bytes, additional_live_bytes)?;
    budget.validate(probe.width, probe.height, peak_bytes)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), probe.format);
    reader.limits(image_decode_limits(budget));
    reader
        .decode()
        .map_err(map_image_decode_error)
        .map(image::DynamicImage::into_rgb8)
}

#[cfg(any(test, feature = "ocr-pipeline"))]
pub(crate) fn decode_standard_rgb8_with_security_limits(
    bytes: &[u8],
    limits: &SecurityLimits,
) -> Result<image::RgbImage> {
    decode_standard_rgb8(bytes, limits, 0)
}

#[cfg(all(feature = "pdf", feature = "ocr-pipeline"))]
pub(crate) fn decode_standard_rgb8_with_additional_live_bytes_and_security_limits(
    bytes: &[u8],
    limits: &SecurityLimits,
    additional_live_bytes: u64,
) -> Result<image::RgbImage> {
    decode_standard_rgb8(bytes, limits, additional_live_bytes)
}

#[cfg(any(all(feature = "pdf", feature = "ocr-pipeline"), test))]
pub(crate) fn decode_standard_luma8_with_security_limits(
    bytes: &[u8],
    limits: &SecurityLimits,
) -> Result<image::GrayImage> {
    let budget = ImageDecodeBudget::from_security_limits(limits);
    let probe = probe_standard_image(bytes, budget, None)?;
    let encoded_live_bytes =
        u64::try_from(bytes.len()).map_err(|_| image_dimension_error(probe.width, probe.height, u64::MAX, u64::MAX))?;
    let peak_bytes = conversion_peak_bytes(probe, ColorType::L8, encoded_live_bytes, 0)?;
    budget.validate(probe.width, probe.height, peak_bytes)?;
    let mut reader = ImageReader::with_format(Cursor::new(bytes), probe.format);
    reader.limits(image_decode_limits(budget));
    reader
        .decode()
        .map_err(map_image_decode_error)
        .map(image::DynamicImage::into_luma8)
}

#[cfg(test)]
pub(crate) fn bmp_with_declared_dimensions(width: u32, height: u32) -> Vec<u8> {
    use image::ImageEncoder;

    let mut bytes = Vec::new();
    image::codecs::bmp::BmpEncoder::new(&mut bytes)
        .write_image(&[255_u8, 255, 255], 1, 1, image::ExtendedColorType::Rgb8)
        .expect("encode the BMP control");
    bytes[18..22].copy_from_slice(&width.to_le_bytes());
    bytes[22..26].copy_from_slice(&height.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests;
