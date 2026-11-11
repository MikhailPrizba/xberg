//! Integration tests for Apple iWork format extractors.
//!
//! These tests verify that .pages, .numbers, and .key files can be
//! opened, parsed, and produce non-empty text output.

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests

mod helpers;
