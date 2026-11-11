#[cfg(test)]
use crate::types::Table;
#[cfg(test)]
use crate::types::TableGrid;
#[cfg(test)]
pub(crate) use crate::types::internal::table_grid::attach_table_grids;
#[cfg(test)]
use crate::types::internal::table_grid::retained_grid;
pub(crate) use crate::types::internal::table_grid::{TableGridSidecar, attach_ruled_table_grids};
#[cfg(test)]
use crate::types::internal::{ElementKind, InternalDocument};

#[cfg(test)]
#[path = "table_grid/tests.rs"]
mod tests;
