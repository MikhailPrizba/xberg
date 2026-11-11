//! Layout-guided extraction utilities.
//!
//! Provides table recognition, layout validation, and heading classification
//! helpers used by the PDF structure pipeline.

mod geometric_tables;
mod heading;
pub(super) mod table_recognition;
pub(crate) mod tables;

pub(super) use geometric_tables::detect_geometric_table_hints;
pub(super) use heading::{looks_like_bare_url, looks_like_figure_label};
pub(super) use tables::extract_tables_from_layout_hints;
