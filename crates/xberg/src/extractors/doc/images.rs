use super::*;

pub(super) fn push_doc_image(doc: &mut InternalDocument, picture: &DocPicture, visible: bool) {
    let _ = visible;
    doc.push_element(InternalElement::text(
        ElementKind::Image {
            image_index: picture.image.image_index,
        },
        "",
        0,
    ));
}

pub(super) fn push_paragraph_runs(
    doc: &mut InternalDocument,
    kind: ElementKind,
    text: &str,
    annotations: &[crate::types::TextAnnotation],
    pictures: &[DocPicture],
    visible_images: bool,
) {
    let mut previous = 0;
    for picture in pictures.iter().filter(|picture| !picture.standalone) {
        let offset = picture.text_offset.min(text.len());
        if let Some(run) = text.get(previous..offset).filter(|run| !run.is_empty()) {
            push_annotated_run(doc, kind, run, annotations, previous);
        }
        push_doc_image(doc, picture, visible_images);
        previous = offset;
    }
    if let Some(run) = text.get(previous..).filter(|run| !run.is_empty()) {
        push_annotated_run(doc, kind, run, annotations, previous);
    }
}

fn push_annotated_run(
    doc: &mut InternalDocument,
    kind: ElementKind,
    text: &str,
    annotations: &[crate::types::TextAnnotation],
    offset: usize,
) {
    let mut element = InternalElement::text(kind, text, 0);
    let first = annotations.partition_point(|annotation| annotation.end as usize <= offset);
    for annotation in &annotations[first..] {
        if annotation.start as usize >= offset + text.len() {
            break;
        }
        let start = (annotation.start as usize).max(offset);
        let end = (annotation.end as usize).min(offset + text.len());
        if start < end {
            let mut annotation = annotation.clone();
            annotation.start = (start - offset) as u32;
            annotation.end = (end - offset) as u32;
            element.annotations.push(annotation);
        }
    }
    doc.push_element(element);
}

pub(super) fn parse_doc(
    content: &[u8],
    limits: &crate::extractors::security::SecurityLimits,
    images: bool,
) -> Result<DocWithImages> {
    extract_doc_with_limits(content, images.then_some(limits), limits)
}
