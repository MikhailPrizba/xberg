//! HEIF / HEIC / AVIF detection and decoding.
//!
//! The sniffer (`is_heif_container`) is always compiled — it's a 12-byte magic
//! check used by `extract_image_metadata` to dispatch. Actual pixel decoding
//! (`decode_heic_to_png`) lives behind the `heic` Cargo feature because it
//! pulls in the C `libheif` dependency via `xberg-libheif`.

/// Detect a HEIF-family container (HEIC / HEIF / AVIF / HEICS / AVCS) by
/// sniffing the `ftyp` box brand at offset 4..8 with one of the known major
/// brands at 8..12.
///
/// The function is always compiled (12-byte magic check, zero deps), but every
/// caller lives behind one of the OCR features. Reranker-only builds without
/// any OCR feature would surface this as `dead_code`; the `#[allow]` keeps the
/// unconditional definition stance documented in the module doc.

pub(crate) fn is_heif_container(bytes: &[u8]) -> bool {
    if bytes.len() < 12 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    matches!(
        &bytes[8..12],
        b"heic"
            | b"heix"
            | b"heim"
            | b"heis"
            | b"hevc"
            | b"hevm"
            | b"hevs"
            | b"mif1"
            | b"msf1"
            | b"avif"
            | b"avis"
            | b"avcs"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_known_heif_brands() {
        let make = |brand: &[u8; 4]| {
            let mut buf = Vec::from(&b"\x00\x00\x00\x18"[..]);
            buf.extend_from_slice(b"ftyp");
            buf.extend_from_slice(brand);
            buf.extend_from_slice(&[0u8; 12]);
            buf
        };
        for brand in [b"heic", b"heix", b"avif", b"avcs", b"mif1", b"avis"] {
            assert!(
                is_heif_container(&make(brand)),
                "brand {:?} should sniff as HEIF",
                std::str::from_utf8(brand).unwrap()
            );
        }
    }

    #[test]
    fn rejects_non_heif() {
        assert!(!is_heif_container(b""));
        assert!(!is_heif_container(b"hello world"));
        assert!(!is_heif_container(&[0u8; 4]));
        assert!(!is_heif_container(b"\x89PNG\r\n\x1a\n0000"));
        assert!(!is_heif_container(b"\x00\x00\x00\x18ftypxxxxRESERVED___"));
    }
}
