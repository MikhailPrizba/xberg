//! Extraction output: literal paragraph whitespace and Markdown tables in source order.
//! Unlike typesetting renderers this does not reflow tabs or hard line breaks.
use super::common::push_escaped_cell;
use crate::types::internal::{ElementKind, InternalDocument};

pub(crate) fn render_literal_markdown(doc: &InternalDocument) -> String {
    let mut blocks = Vec::new();
    let mut lists: Vec<(u16, u32)> = Vec::new();
    for elem in &doc.elements {
        // The extraction pipeline has already applied the requested content filter.
        // Keep retained headers, footers and footnotes as well as body paragraphs.
        match elem.kind {
            ElementKind::ListStart { .. } => {
                lists.push((elem.depth, 0));
                continue;
            }
            ElementKind::ListEnd => {
                lists.pop();
                continue;
            }
            ElementKind::Table { table_index } => {
                if let Some(table) = doc.tables.get(table_index as usize) {
                    let width = table.cells.iter().map(Vec::len).max().unwrap_or(0);
                    if width == 0 {
                        continue;
                    }
                    let mut text = String::new();
                    for (i, row) in table.cells.iter().enumerate() {
                        text.push('|');
                        for col in 0..width {
                            let content = row.get(col).map(String::as_str).unwrap_or("");
                            if i == 0 && content.is_empty() {
                                text.push_str(&format!("Col{}", col + 1));
                            } else {
                                push_escaped_cell(&mut text, content);
                            }
                            text.push('|');
                        }
                        text.push('\n');
                        if i == 0 {
                            text.push('|');
                            for _ in 0..width {
                                text.push_str("---|");
                            }
                            text.push('\n');
                        }
                    }
                    blocks.push(text.trim_end_matches('\n').to_owned());
                }
            }
            ElementKind::ListItem { ordered } => {
                let count = if let Some((_, counter)) = lists.last_mut() {
                    *counter += 1;
                    *counter
                } else {
                    1
                };
                let fallback = if ordered { format!("{count}.") } else { "•".to_owned() };
                let marker = elem
                    .list_item_source_label()
                    .or_else(|| {
                        elem.attributes
                            .as_ref()
                            .and_then(|a| a.get("list_marker"))
                            .map(String::as_str)
                    })
                    .unwrap_or(&fallback);
                if !elem.text.trim().is_empty() {
                    blocks.push(format!("{marker}\t{}", elem.text));
                }
            }
            ElementKind::Image { .. }
            | ElementKind::PageBreak
            | ElementKind::GroupStart
            | ElementKind::GroupEnd
            | ElementKind::QuoteStart
            | ElementKind::QuoteEnd
            | ElementKind::MetadataBlock
            | ElementKind::FootnoteRef
            | ElementKind::CommentRef
            | ElementKind::CommentDefinition => {}
            _ => {
                if !elem.text.trim().is_empty() {
                    blocks.push(elem.text.clone());
                }
            }
        }
    }
    let mut text = blocks.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::internal_builder::InternalDocumentBuilder;
    #[test]
    fn preserves_literal_whitespace_and_table_position() {
        let mut b = InternalDocumentBuilder::new("docx");
        b.push_paragraph("Before\tvalue\ncontinued\n", vec![], None, None);
        b.push_table_from_cells(
            &[vec!["Header".into(), "".into()], vec!["a|b".into(), "x\ny".into()]],
            None,
            None,
        );
        b.push_paragraph("After", vec![], None, None);
        assert_eq!(
            render_literal_markdown(&b.build()),
            "Before\tvalue\ncontinued\n\n|Header|Col2|\n|---|---|\n|a\\|b|x<br>y|\nAfter\n"
        );
    }
}
