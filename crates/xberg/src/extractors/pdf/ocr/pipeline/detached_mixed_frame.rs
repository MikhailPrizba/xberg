use super::RunPageParagraphs;

impl RunPageParagraphs {
    /// Detached paragraph geometry is carried separately from the already assembled document.
    pub(super) fn place_detached_in_points(&mut self) -> bool {
        let Some((sx, sy)) = self.detached_ocr_scale.take() else {
            return false;
        };
        for paragraph in &mut self.paragraphs {
            if let Some((left, bottom, right, top)) = paragraph.block_bbox {
                paragraph.block_bbox = Some((left * sx, bottom * sy, right * sx, top * sy));
            }
            for line in &mut paragraph.lines {
                line.baseline_y *= sy;
                for segment in &mut line.segments {
                    segment.x *= sx;
                    segment.width *= sx;
                    segment.y *= sy;
                    segment.height *= sy;
                    segment.baseline_y *= sy;
                }
            }
        }
        true
    }
}
