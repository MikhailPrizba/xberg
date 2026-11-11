//! Table structure recognition for native PDF pages (TATR + SLANeXT backends).

use super::super::geometry::Rect;

/// Compute intersection-over-word-area between an HocrWord and a rectangular region.
///
/// Both word and region must be in the same coordinate space (image coords).
pub(in crate::pdf::structure) fn word_hint_iow(
    w: &crate::pdf::table_reconstruct::HocrWord,
    region_left: f32,
    region_top: f32,
    region_right: f32,
    region_bottom: f32,
) -> f32 {
    let word_rect = Rect::from_xywh(w.left as f32, w.top as f32, w.width as f32, w.height as f32);
    let region_rect = Rect::from_ltrb(region_left, region_top, region_right, region_bottom);
    if word_rect.area() <= 0.0 {
        return if region_rect.contains_point(word_rect.center_x(), word_rect.center_y()) {
            1.0
        } else {
            0.0
        };
    }
    word_rect.intersection_over_self(&region_rect)
}
