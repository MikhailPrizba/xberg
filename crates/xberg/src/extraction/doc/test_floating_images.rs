//! Floating pictures: a main-document `\x08` anchor whose `PlcSpaMom` shape names a blip.

use super::test_images::{
    IN_LIST, IN_TABLE, WHITE, change_stream, image_config, picture_doc, png, with_paragraph_property,
};
use super::tests::{
    TestPiece, build_doc_ole, build_fib, build_plc_pcd, build_table_stream, compressed_fc, test_rg_fc_lcb_offset,
    write_u32,
};
use crate::core::config::ExtractionConfig;
use crate::extraction::office_art::{
    MSO_PROPERTY_PIB, MSOFBT_BSE, MSOFBT_BSTORE_CONTAINER, MSOFBT_DG_CONTAINER, MSOFBT_DGG_CONTAINER, MSOFBT_FSP,
    MSOFBT_OPT, MSOFBT_SP_CONTAINER, MSOFBT_SPGR_CONTAINER, RT_BLIP_PNG,
};
use crate::extractors::doc::DocExtractor;
use crate::plugins::InternalDocumentExtractor;

fn office_art_record(packed: u16, kind: u16, body: &[u8]) -> Vec<u8> {
    let mut record = packed.to_le_bytes().to_vec();
    record.extend_from_slice(&kind.to_le_bytes());
    record.extend_from_slice(&(body.len() as u32).to_le_bytes());
    record.extend_from_slice(body);
    record
}

/// One piece of a test document's main text.
enum TextPiece {
    /// UTF-16 code units, two bytes each.
    Utf16(Vec<u16>),
    /// CP1252 bytes, one byte each.
    Cp1252(Vec<u8>),
}

fn utf16(text: &str) -> TextPiece {
    TextPiece::Utf16(text.encode_utf16().collect())
}

/// Main-document floating shapes, one anchored at each `\x08` of the text: `PlcSpaMom`
/// names each shape, whose `pib` selects one of `images`, each a BSE in the drawing
/// group's BStore. `delayed` stores each blip in the WordDocument stream at its BSE's
/// `foDelay`, as Word does; otherwise the blip follows its BSE inside the BStore.
struct FloatingDoc {
    pieces: Vec<TextPiece>,
    images: Vec<Vec<u8>>,
    delayed: bool,
    /// The `pib` of the shape at each anchor, in text order; `None` for a shape
    /// without a picture.
    pibs: Vec<Option<u32>>,
    /// The shapes of a header drawing (`dgglbl` 1) put before the main document's,
    /// with the same spids; empty for none.
    header_pibs: Vec<Option<u32>>,
}

impl FloatingDoc {
    fn new(text: &str) -> Self {
        Self {
            pieces: vec![utf16(text)],
            images: vec![png(WHITE)],
            delayed: true,
            pibs: vec![Some(1)],
            header_pibs: Vec::new(),
        }
    }

    /// The spid of the shape at the first anchor; the next anchors' follow it.
    const SPID: u32 = 1026;

    fn build(&self) -> Vec<u8> {
        let (mut word, pieces, anchors, ccp) = self.main_text();
        assert_eq!(anchors.len(), self.pibs.len(), "one shape per anchor");
        let mut table = build_table_stream(&mut word, &build_plc_pcd(&pieces));
        let content = self.office_art_content(&mut word);
        let dgg_info = table.len();
        table.extend_from_slice(&content);
        write_u32(&mut word, test_rg_fc_lcb_offset() + 50 * 8, dgg_info as u32);
        write_u32(&mut word, test_rg_fc_lcb_offset() + 50 * 8 + 4, content.len() as u32);

        let plc_spa = table.len();
        for cp in anchors.iter().chain([&ccp]) {
            table.extend_from_slice(&(*cp as u32).to_le_bytes());
        }
        for index in 0..anchors.len() {
            let mut spa = vec![0u8; 26];
            write_u32(&mut spa, 0, Self::SPID + index as u32);
            table.extend_from_slice(&spa);
        }
        write_u32(&mut word, test_rg_fc_lcb_offset() + 40 * 8, plc_spa as u32);
        write_u32(
            &mut word,
            test_rg_fc_lcb_offset() + 40 * 8 + 4,
            (table.len() - plc_spa) as u32,
        );
        build_doc_ole(&word, &table)
    }

    /// The WordDocument stream holding the main text, its pieces, the anchors' CPs and
    /// the text's length in CPs.
    fn main_text(&self) -> (Vec<u8>, Vec<TestPiece>, Vec<usize>, usize) {
        let ccp = self
            .pieces
            .iter()
            .map(|piece| match piece {
                TextPiece::Utf16(units) => units.len(),
                TextPiece::Cp1252(bytes) => bytes.len(),
            })
            .sum::<usize>();
        let mut word = build_fib(2048, ccp as u32, 0, 0, 0);
        let (mut fc, mut cp, mut anchors) = (1024, 0, Vec::new());
        let mut pieces = Vec::new();
        for piece in &self.pieces {
            let (bytes, units, fc_raw) = match piece {
                TextPiece::Utf16(units) => (
                    units.iter().flat_map(|unit| unit.to_le_bytes()).collect::<Vec<_>>(),
                    units.clone(),
                    fc as u32,
                ),
                TextPiece::Cp1252(bytes) => (
                    bytes.clone(),
                    bytes.iter().map(|&byte| u16::from(byte)).collect(),
                    compressed_fc(fc as u32),
                ),
            };
            anchors.extend(
                (0..units.len())
                    .filter(|&index| units[index] == 0x08)
                    .map(|index| cp + index),
            );
            word[fc..fc + bytes.len()].copy_from_slice(&bytes);
            pieces.push(TestPiece {
                cp_start: cp as u32,
                cp_end: (cp + units.len()) as u32,
                fc_raw,
            });
            fc += bytes.len();
            cp += units.len();
        }
        (word, pieces, anchors, ccp)
    }

    /// `OfficeArtContent`: the drawing group, one BSE per image, then the header
    /// drawing if any and the main document's drawing. A delayed blip goes at the end
    /// of `word`.
    fn office_art_content(&self, word: &mut Vec<u8>) -> Vec<u8> {
        let mut blocks = Vec::new();
        for image in &self.images {
            let blip = office_art_record(0x6e00, RT_BLIP_PNG, &[vec![0u8; 17].as_slice(), image].concat());
            let mut bse = vec![0u8; 36];
            bse[0] = 6;
            bse[1] = 6;
            write_u32(&mut bse, 20, blip.len() as u32);
            write_u32(&mut bse, 24, 1);
            if self.delayed {
                write_u32(&mut bse, 28, word.len() as u32);
                word.extend_from_slice(&blip);
            } else {
                bse.extend_from_slice(&blip);
            }
            blocks.extend(office_art_record(0x0062, MSOFBT_BSE, &bse));
        }
        let count = self.images.len() as u16;
        let bstore = office_art_record((count << 4) | 0xf, MSOFBT_BSTORE_CONTAINER, &blocks);
        let mut content = office_art_record(0x000f, MSOFBT_DGG_CONTAINER, &bstore);
        if !self.header_pibs.is_empty() {
            content.push(1);
            content.extend(Self::drawing(&self.header_pibs));
        }
        content.push(0);
        content.extend(Self::drawing(&self.pibs));
        content
    }

    /// An `OfficeArtDgContainer` with one shape per entry of `pibs`, from `SPID` on.
    fn drawing(pibs: &[Option<u32>]) -> Vec<u8> {
        let mut shapes = Vec::new();
        for (index, pib) in pibs.iter().enumerate() {
            let fsp = [(Self::SPID + index as u32).to_le_bytes(), 0x0a00u32.to_le_bytes()].concat();
            let mut shape = office_art_record(0x0002, MSOFBT_FSP, &fsp);
            if let Some(pib) = pib {
                let property = [(MSO_PROPERTY_PIB | 0x4000).to_le_bytes().as_slice(), &pib.to_le_bytes()].concat();
                shape.extend(office_art_record(0x0013, MSOFBT_OPT[0], &property));
            }
            shapes.extend(office_art_record(0x000f, MSOFBT_SP_CONTAINER, &shape));
        }
        let group = office_art_record(0x000f, MSOFBT_SPGR_CONTAINER, &shapes);
        office_art_record(0x000f, MSOFBT_DG_CONTAINER, &group)
    }
}

fn floating_png_doc(text: &str, delayed: bool, pib: Option<u32>) -> (Vec<u8>, Vec<u8>) {
    let doc = FloatingDoc {
        delayed,
        pibs: vec![pib],
        ..FloatingDoc::new(text)
    };
    (doc.build(), doc.images[0].clone())
}

async fn extract_floating(bytes: &[u8]) -> crate::types::internal::InternalDocument {
    DocExtractor::new()
        .extract_content(bytes, "application/msword", &image_config())
        .await
        .expect("DOC")
}

fn texts(doc: &crate::types::internal::InternalDocument) -> Vec<&str> {
    doc.elements
        .iter()
        .filter(|element| !matches!(element.kind, crate::types::internal::ElementKind::Image { .. }))
        .map(|element| element.text.as_str())
        .collect()
}

fn assert_picture_between_paragraphs(doc: &crate::types::internal::InternalDocument, context: &str) {
    assert_eq!(
        doc.processing_warnings.len(),
        0,
        "{context}: {:?}",
        doc.processing_warnings
    );
    assert_eq!(doc.images.len(), 1, "{context}");
    assert_eq!(doc.elements.len(), 3, "{context}");
    assert_eq!(doc.elements[0].text, "BEFORE", "{context}");
    assert!(
        matches!(
            doc.elements[1].kind,
            crate::types::internal::ElementKind::Image { image_index: 0 }
        ),
        "{context}"
    );
    assert_eq!(doc.elements[2].text, "AFTER", "{context}");
}

#[tokio::test]
async fn should_extract_floating_doc_pictures_between_native_paragraphs() {
    for delayed in [true, false] {
        let (bytes, png) = floating_png_doc("BEFORE\r\x08\rAFTER\r", delayed, Some(1));
        let doc = extract_floating(&bytes).await;
        assert_picture_between_paragraphs(&doc, &format!("delayed {delayed}"));
        assert_eq!(doc.images[0].format, "png");
        assert_eq!(doc.images[0].data.as_ref(), png.as_slice());
    }
}

#[tokio::test]
async fn should_place_floating_doc_pictures_by_cp_across_pieces() {
    // An unpaired surrogate takes a CP but decodes to no text. ~keep
    let lone = |before: &str, after: &str| {
        TextPiece::Utf16(
            before
                .encode_utf16()
                .chain([0xd800])
                .chain(after.encode_utf16())
                .collect(),
        )
    };
    let cases = [
        (
            "anchor in a later piece",
            vec![utf16("BEFORE\r"), utf16("\x08\rAFTER\r")],
        ),
        (
            "dropped unit before the anchor",
            vec![lone("BEFORE\r", "\x08\rAFTER\r")],
        ),
        (
            "compressed piece after a dropped unit",
            vec![lone("BEFORE\r", ""), TextPiece::Cp1252(b"\x08\rAFTER\r".to_vec())],
        ),
    ];
    for (case, pieces) in cases {
        let bytes = FloatingDoc {
            pieces,
            ..FloatingDoc::new("")
        }
        .build();
        assert_picture_between_paragraphs(&extract_floating(&bytes).await, case);
    }
}

#[tokio::test]
async fn should_extract_the_floating_doc_picture_its_pib_selects() {
    for delayed in [true, false] {
        let source = FloatingDoc {
            images: vec![png(WHITE), png([0, 0, 0])],
            delayed,
            pibs: vec![Some(2)],
            ..FloatingDoc::new("BEFORE\r\x08\rAFTER\r")
        };
        let doc = extract_floating(&source.build()).await;
        assert_picture_between_paragraphs(&doc, &format!("delayed {delayed}"));
        assert_eq!(
            doc.images[0].data.as_ref(),
            source.images[1].as_slice(),
            "delayed {delayed}"
        );
    }
}

#[tokio::test]
async fn should_read_floating_doc_picture_pib_from_a_secondary_property_table() {
    let (bytes, png) = floating_png_doc("BEFORE\r\x08\rAFTER\r", true, Some(1));
    // A secondary FOPT holds `pib`, with `fBid` clear in its `opid`. ~keep
    let bytes = change_stream(bytes, "/1Table", |table| {
        let fopt = [0x13, 0x00, 0x0b, 0xf0, 6, 0, 0, 0, 0x04, 0x41];
        let at = table
            .windows(fopt.len())
            .position(|window| window == fopt)
            .expect("FOPT record");
        table[at + 2..at + 4].copy_from_slice(&MSOFBT_OPT[1].to_le_bytes());
        table[at + 8..at + 10].copy_from_slice(&MSO_PROPERTY_PIB.to_le_bytes());
    });
    let doc = extract_floating(&bytes).await;
    assert_eq!(doc.processing_warnings.len(), 0, "{:?}", doc.processing_warnings);
    assert_eq!(doc.images.len(), 1);
    assert_eq!(doc.images[0].data.as_ref(), png.as_slice());
}

#[tokio::test]
async fn should_skip_floating_doc_shapes_without_a_picture_silently() {
    let (bytes, _) = floating_png_doc("BEFORE\r\x08\rAFTER\r", true, None);
    let doc = extract_floating(&bytes).await;
    assert_eq!(doc.images.len(), 0);
    assert_eq!(doc.processing_warnings.len(), 0);
    assert_eq!(texts(&doc), ["BEFORE", "AFTER"]);
}

#[tokio::test]
async fn should_read_floating_doc_shapes_from_the_main_document_drawing() {
    // The header drawing comes first and gives the same spid another picture. ~keep
    let source = FloatingDoc {
        images: vec![png(WHITE), png([0, 0, 0])],
        header_pibs: vec![Some(2)],
        ..FloatingDoc::new("BEFORE\r\x08\rAFTER\r")
    };
    let doc = extract_floating(&source.build()).await;
    assert_picture_between_paragraphs(&doc, "header drawing first");
    assert_eq!(doc.images[0].data.as_ref(), source.images[0].as_slice());
}

#[tokio::test]
async fn should_warn_for_floating_doc_pictures_in_table_cells_and_lists() {
    let text = "BEFORE\x08\r";
    for property in [IN_TABLE.as_slice(), IN_LIST.as_slice()] {
        // The paragraph property rewrites the WordDocument stream past the text, where
        // a delayed blip would go. ~keep
        let bytes = FloatingDoc {
            delayed: false,
            ..FloatingDoc::new(text)
        }
        .build();
        let bytes = with_paragraph_property(bytes, text, property);
        let doc = extract_floating(&bytes).await;
        assert_eq!(doc.images.len(), 0, "{property:?}");
        assert_eq!(doc.processing_warnings.len(), 1, "{property:?}");
        assert!(
            doc.processing_warnings[0]
                .message
                .contains("table cells or automatic lists"),
            "{property:?}"
        );
    }
}

#[tokio::test]
async fn should_count_only_floating_doc_shapes_that_are_pictures_towards_the_file_limit() {
    let mut config = image_config();
    config.security_limits = Some(crate::extractors::security::SecurityLimits {
        max_files_in_archive: 1,
        ..Default::default()
    });
    let extract = |pibs: Vec<Option<u32>>| {
        let text = format!("BEFORE\r{}\rAFTER\r", "\x08".repeat(pibs.len()));
        let bytes = FloatingDoc {
            pibs,
            ..FloatingDoc::new(&text)
        }
        .build();
        let config = config.clone();
        async move {
            DocExtractor::new()
                .extract_content(&bytes, "application/msword", &config)
                .await
        }
    };

    let doc = extract(vec![None, None, Some(1)]).await.expect("one picture");
    assert_eq!(doc.processing_warnings.len(), 0, "{:?}", doc.processing_warnings);
    assert_eq!(doc.images.len(), 1);
    let error = extract(vec![Some(1), Some(1)]).await.expect_err("two pictures");
    assert!(
        error.to_string().contains("DOC embedded picture count exceeds"),
        "{error}"
    );
}

#[tokio::test]
async fn should_warn_about_grouped_floating_doc_pictures_only() {
    for pib in [Some(1), None] {
        let (bytes, _) = floating_png_doc("BEFORE\r\x08\rAFTER\r", true, pib);
        let bytes = change_stream(bytes, "/1Table", |table| {
            let fsp = [FloatingDoc::SPID.to_le_bytes(), 0x0a00u32.to_le_bytes()].concat();
            let at = table
                .windows(fsp.len())
                .position(|window| window == fsp)
                .expect("FSP record");
            write_u32(table, at + 4, 0x0a01);
        });
        let doc = extract_floating(&bytes).await;
        assert_eq!(doc.images.len(), 0, "pib {pib:?}");
        assert_eq!(doc.processing_warnings.len(), usize::from(pib.is_some()), "pib {pib:?}");
        assert!(
            doc.processing_warnings
                .iter()
                .all(|warning| warning.message.contains("Grouped floating pictures"))
        );
        assert_eq!(texts(&doc), ["BEFORE", "AFTER"], "pib {pib:?}");
    }
}

#[tokio::test]
async fn should_warn_that_a_floating_doc_picture_outside_the_delay_stream_holds_no_image_data() {
    let (bytes, _) = floating_png_doc("BEFORE\r\x08\rAFTER\r", true, Some(1));
    // `foDelay` 0xFFFFFFFF says the blip is not in the delay stream. ~keep
    let bytes = change_stream(bytes, "/1Table", |table| {
        let header = [0x0062u16.to_le_bytes(), MSOFBT_BSE.to_le_bytes()].concat();
        let bse = table
            .windows(header.len())
            .position(|window| window == header)
            .expect("BSE record");
        write_u32(table, bse + 8 + 28, u32::MAX);
    });
    let doc = extract_floating(&bytes).await;
    assert_eq!(doc.images.len(), 0);
    assert_eq!(doc.processing_warnings.len(), 1);
    assert!(doc.processing_warnings[0].message.contains("no image data"));
    assert_eq!(texts(&doc), ["BEFORE", "AFTER"]);
}

#[tokio::test]
async fn should_warn_and_preserve_native_text_when_floating_picture_records_are_malformed() {
    let text = "BEFORE\r\x08\rAFTER\r";
    let unsupported = "unsupported image format";
    let unresolved = "floating shape has no drawing record";
    // Give the one container record of type `from` the type `to`. ~keep
    let retype = |table: &mut Vec<u8>, from: u16, to: u16| {
        let header = [0x000fu16.to_le_bytes(), from.to_le_bytes()].concat();
        let at = table
            .windows(header.len())
            .position(|window| window == header)
            .expect("container record");
        table[at + 2..at + 4].copy_from_slice(&to.to_le_bytes());
    };
    let mut expected = vec![unsupported; 2];
    expected.resize(8, unresolved);
    for (mutation, expected) in expected.into_iter().enumerate() {
        let (bytes, _) = floating_png_doc(text, true, Some(if mutation == 0 { 2 } else { 1 }));
        let bytes = match mutation {
            0 => bytes,
            1 => change_stream(bytes, "/WordDocument", |word| word.truncate(2060)),
            2 => change_stream(bytes, "/1Table", |table| {
                let spa = table.len() - 26;
                write_u32(table, spa, 7);
            }),
            3 => change_stream(bytes, "/1Table", |table| {
                let plc_spa = table.len() - 34;
                write_u32(table, plc_spa, 3);
            }),
            4 => change_stream(bytes, "/WordDocument", |word| {
                write_u32(word, test_rg_fc_lcb_offset() + 50 * 8 + 4, u32::MAX)
            }),
            5 => change_stream(bytes, "/1Table", |table| {
                retype(table, MSOFBT_DGG_CONTAINER, MSOFBT_BSTORE_CONTAINER)
            }),
            6 => change_stream(bytes, "/1Table", |table| {
                retype(table, MSOFBT_DG_CONTAINER, MSOFBT_SPGR_CONTAINER)
            }),
            // A `PlcSpaMom` one byte longer than its whole entries, the byte present. ~keep
            _ => change_stream(
                change_stream(bytes, "/1Table", |table| table.push(0)),
                "/WordDocument",
                |word| write_u32(word, test_rg_fc_lcb_offset() + 40 * 8 + 4, 35),
            ),
        };
        let doc = extract_floating(&bytes).await;
        assert_eq!(doc.images.len(), 0, "mutation {mutation}");
        assert_eq!(doc.processing_warnings.len(), 1, "mutation {mutation}");
        assert!(
            doc.processing_warnings[0].message.contains(expected),
            "mutation {mutation}: {}",
            doc.processing_warnings[0].message
        );
        assert_eq!(texts(&doc), ["BEFORE", "AFTER"], "mutation {mutation}");
    }
}

#[tokio::test]
async fn should_reject_floating_doc_picture_when_content_budget_is_exceeded() {
    for delayed in [true, false] {
        let (bytes, _) = floating_png_doc("\x08\r", delayed, Some(1));
        let mut config = image_config();
        config.security_limits = Some(crate::extractors::security::SecurityLimits {
            max_content_size: 16,
            ..Default::default()
        });
        let result = DocExtractor::new()
            .extract_content(&bytes, "application/msword", &config)
            .await;
        assert!(
            matches!(result, Err(crate::XbergError::Validation { .. })),
            "delayed {delayed}"
        );
    }
}

#[tokio::test]
async fn should_count_every_floating_doc_picture_copy_against_the_content_budget() {
    let picture = png(WHITE);
    let mut config = image_config();
    // Room for one copy of the picture, not for two. ~keep
    config.security_limits = Some(crate::extractors::security::SecurityLimits {
        max_content_size: picture.len() * 3 / 2,
        ..Default::default()
    });
    for delayed in [true, false] {
        for anchors in [1, 2] {
            let text = format!("BEFORE\r{}\rAFTER\r", "\x08".repeat(anchors));
            let bytes = FloatingDoc {
                delayed,
                pibs: vec![Some(1); anchors],
                ..FloatingDoc::new(&text)
            }
            .build();
            let result = DocExtractor::new()
                .extract_content(&bytes, "application/msword", &config)
                .await;
            let context = format!("delayed {delayed}, {anchors} anchors");
            match anchors {
                1 => assert_eq!(result.expect(&context).images.len(), 1, "{context}"),
                _ => assert!(matches!(result, Err(crate::XbergError::Validation { .. })), "{context}"),
            }
        }
    }
}

#[tokio::test]
async fn should_warn_about_unresolved_floating_doc_shapes_only_when_requested() {
    let (bytes, _) = picture_doc("BEFORE\r\x08\rAFTER\r", false);
    for requested in [false, true] {
        let config = if requested {
            image_config()
        } else {
            ExtractionConfig::default()
        };
        let doc = DocExtractor::new()
            .extract_content(&bytes, "application/msword", &config)
            .await
            .expect("DOC");
        assert_eq!(doc.images.len(), 0);
        assert_eq!(
            doc.processing_warnings
                .iter()
                .filter(|warning| warning.message.contains("floating shape"))
                .count(),
            usize::from(requested)
        );
        assert_eq!(
            doc.elements
                .iter()
                .map(|element| element.text.as_str())
                .collect::<Vec<_>>(),
            ["BEFORE", "AFTER"]
        );
    }
}
