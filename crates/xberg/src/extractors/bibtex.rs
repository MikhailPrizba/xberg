//! BibTeX bibliography extractor.
//!
//! Extracts and parses BibTeX bibliography files, providing structured access
//! to bibliography entries, metadata, and author information.

use crate::Result;
use crate::core::config::ExtractionConfig;
use crate::plugins::{InternalDocumentExtractor, Plugin};
use crate::types::internal::InternalDocument;
use crate::types::internal_builder::InternalDocumentBuilder;
use crate::types::metadata::{BibtexMetadata, FormatMetadata, Metadata, YearRange};
use crate::types::uri::ExtractedUri;
use ahash::AHashMap;
use ahash::AHashSet;
use async_trait::async_trait;
use std::borrow::Cow;
use std::collections::BTreeMap;

#[cfg(feature = "office")]
use crate::types::document_structure::{AnnotationKind, TextAnnotation};
#[cfg(feature = "office")]
use biblatex::{Bibliography, ChunksExt};

/// BibTeX bibliography extractor.
///
/// Parses BibTeX files and extracts structured bibliography data including
/// entries, authors, publication years, and entry type distribution.
pub struct BibtexExtractor;

#[cfg(feature = "office")]
#[derive(Default)]
struct ParsedBibliography {
    entries: Vec<(String, serde_json::Map<String, serde_json::Value>)>,
    authors: AHashSet<String>,
    years: AHashSet<u32>,
    entry_types: AHashMap<String, i32>,
    formatted_entries: String,
}

#[cfg(feature = "office")]
fn collect_entry_fields(entry: &biblatex::Entry, parsed: &mut ParsedBibliography) -> AHashMap<String, String> {
    let mut fields = AHashMap::new();
    fields.insert("entry_type".to_string(), entry.entry_type.to_string());
    parsed
        .formatted_entries
        .push_str(&format!("@{}{{{},\n", entry.entry_type, entry.key));

    for (field_name, field_chunks) in &entry.fields {
        let field_text = field_chunks.format_verbatim();
        parsed
            .formatted_entries
            .push_str(&format!("  {} = {{{}}},\n", field_name, field_text));
        fields.insert(field_name.to_lowercase(), field_text.clone());

        match field_name.to_ascii_lowercase().as_str() {
            "author" => parsed.authors.extend(
                field_text
                    .split(" and ")
                    .map(str::trim)
                    .filter(|author| !author.is_empty())
                    .map(str::to_string),
            ),
            "year" => {
                if let Ok(year) = field_text.parse::<u32>() {
                    parsed.years.insert(year);
                }
            }
            _ => {}
        }
    }
    parsed.formatted_entries.push_str("}\n\n");
    fields
}

#[cfg(feature = "office")]
fn add_entry_links(
    builder: &mut InternalDocumentBuilder,
    fields: &AHashMap<String, String>,
    label: &str,
) -> Vec<TextAnnotation> {
    let mut annotations = Vec::new();
    for (field, citation) in [("url", false), ("doi", true)] {
        let Some(value) = fields.get(field).filter(|value| !value.is_empty()) else {
            continue;
        };
        let annotation_url = if citation && !value.starts_with("http") {
            format!("https://doi.org/{value}")
        } else {
            value.clone()
        };
        if citation {
            builder.push_uri(ExtractedUri::citation(
                format!("https://doi.org/{value}"),
                Some(label.to_string()),
            ));
        } else {
            builder.push_uri(ExtractedUri::hyperlink(value, Some(label.to_string())));
        }
        annotations.push(TextAnnotation {
            start: 0,
            end: 0,
            kind: AnnotationKind::Link {
                url: annotation_url,
                title: Some(label.to_string()),
            },
        });
    }
    annotations
}

#[cfg(feature = "office")]
fn append_bibliography_entry(
    entry: &biblatex::Entry,
    parsed: &mut ParsedBibliography,
    builder: &mut InternalDocumentBuilder,
) {
    let entry_start = parsed.formatted_entries.len();
    let mut fields = collect_entry_fields(entry, parsed);
    let label = fields
        .get("title")
        .filter(|title| !title.is_empty())
        .cloned()
        .unwrap_or_else(|| entry.key.clone());
    let citation_text = parsed.formatted_entries[entry_start..].trim().to_string();
    let citation_index = builder.push_citation(&citation_text, &entry.key, None);
    let mut annotations = add_entry_links(builder, &fields, &label);
    for annotation in &mut annotations {
        annotation.end = citation_text.len() as u32;
    }
    if !annotations.is_empty() {
        builder.set_annotations(citation_index, annotations);
    }

    let fields_json = fields
        .iter()
        .map(|(key, value)| (key.clone(), serde_json::json!(value)))
        .collect();
    builder.set_attributes(citation_index, std::mem::take(&mut fields));
    *parsed
        .entry_types
        .entry(entry.entry_type.to_string().to_lowercase())
        .or_insert(0) += 1;
    parsed.entries.push((entry.key.clone(), fields_json));
}

#[cfg(feature = "office")]
fn parse_bibliography(input: &str, builder: &mut InternalDocumentBuilder) -> ParsedBibliography {
    let mut parsed = ParsedBibliography::default();
    match Bibliography::parse(input) {
        Ok(bibliography) => {
            for entry in bibliography.iter() {
                append_bibliography_entry(entry, &mut parsed, builder);
            }
        }
        Err(_error) => {
            builder.add_warning(crate::core::diagnostics::warning(
                "bibtex",
                "BibTeX parsing failed; returning raw text as a fallback",
            ));
            parsed.formatted_entries = input.to_string();
            builder.push_code(&parsed.formatted_entries, None, None, None);
        }
    }
    parsed
}

impl BibtexExtractor {
    /// Create a new BibTeX extractor.
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Default for BibtexExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for BibtexExtractor {
    fn name(&self) -> &str {
        "bibtex-extractor"
    }

    fn version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn initialize(&self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    fn description(&self) -> &str {
        "Extracts and parses BibTeX bibliography files with structured metadata"
    }

    fn author(&self) -> &str {
        "Xberg Team"
    }
}

#[cfg(feature = "office")]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InternalDocumentExtractor for BibtexExtractor {
    async fn extract_content(
        &self,
        content: &[u8],
        mime_type: &str,
        _config: &ExtractionConfig,
    ) -> Result<InternalDocument> {
        let bibtex_str = String::from_utf8_lossy(content);
        let mut builder = InternalDocumentBuilder::new("bibtex");
        let parsed = parse_bibliography(&bibtex_str, &mut builder);
        let citation_keys = parsed.entries.iter().map(|(key, _)| key.clone()).collect();
        let mut authors_list: Vec<String> = parsed.authors.into_iter().collect();
        authors_list.sort();
        let year_range = if parsed.years.is_empty() {
            None
        } else {
            let min_year = parsed.years.iter().min().copied();
            let max_year = parsed.years.iter().max().copied();
            let mut years: Vec<u32> = parsed.years.into_iter().collect();
            years.sort_unstable();
            Some(YearRange {
                min: min_year,
                max: max_year,
                years,
            })
        };
        let entry_types = if parsed.entry_types.is_empty() {
            None
        } else {
            Some(
                parsed
                    .entry_types
                    .into_iter()
                    .map(|(key, count)| (key, count as usize))
                    .collect::<BTreeMap<_, _>>(),
            )
        };
        let bibtex_metadata = BibtexMetadata {
            entry_count: parsed.entries.len(),
            citation_keys,
            authors: authors_list.clone(),
            year_range,
            entry_types,
        };
        let mut additional: AHashMap<Cow<'static, str>, serde_json::Value> = AHashMap::new();
        let entries_metadata: Vec<serde_json::Value> = parsed
            .entries
            .iter()
            .map(|(key, fields)| {
                let mut entry_obj = serde_json::Map::new();
                entry_obj.insert("key".to_string(), serde_json::json!(key));
                for (k, v) in fields {
                    entry_obj.insert(k.clone(), v.clone());
                }
                serde_json::Value::Object(entry_obj)
            })
            .collect();
        additional.insert(Cow::Borrowed("entries"), serde_json::json!(entries_metadata));

        let meta_authors = if authors_list.is_empty() {
            None
        } else {
            Some(authors_list)
        };

        let mut doc = builder.build();
        doc.mime_type = mime_type.to_string();
        doc.metadata = Metadata {
            authors: meta_authors,
            format: Some(FormatMetadata::Bibtex(bibtex_metadata)),
            additional,
            ..Default::default()
        };

        Ok(doc)
    }

    fn supported_mime_types(&self) -> &[&str] {
        &["application/x-bibtex", "text/x-bibtex", "application/x-biblatex"]
    }

    fn priority(&self) -> i32 {
        50
    }
}

#[cfg(all(test, feature = "office"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_can_extract_bibtex_mime_types() {
        let extractor = BibtexExtractor::new();
        let supported = extractor.supported_mime_types();

        assert!(supported.contains(&"application/x-bibtex"));
        assert!(supported.contains(&"text/x-bibtex"));
        assert!(supported.contains(&"application/x-biblatex"));
        assert_eq!(supported.len(), 3);
    }

    #[tokio::test]
    async fn test_extract_simple_bibtex() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{key2023,
    title = {Sample Title},
    author = {John Doe},
    year = {2023}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract valid BibTeX entry");

        let metadata = &result.metadata;
        if let Some(FormatMetadata::Bibtex(bib)) = &metadata.format {
            assert_eq!(bib.entry_count, 1);
        } else {
            panic!("Expected FormatMetadata::Bibtex");
        }
    }

    #[tokio::test]
    async fn test_extract_multiple_entries() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{first2020,
    title = {First Paper},
    author = {Author One},
    year = {2020},
    journal = {Test Journal}
}

@book{second2021,
    title = {Test Book},
    author = {Author Two},
    year = {2021},
    publisher = {Test Publisher}
}

@inproceedings{third2022,
    title = {Conference Paper},
    author = {Author Three},
    year = {2022}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract valid BibTeX entries");

        let metadata = &result.metadata;

        if let Some(FormatMetadata::Bibtex(bib)) = &metadata.format {
            assert_eq!(bib.entry_count, 3);
            assert_eq!(bib.citation_keys.len(), 3);
            if let Some(types) = &bib.entry_types {
                assert!(types.contains_key("article"));
                assert!(types.contains_key("book"));
                assert!(types.contains_key("inproceedings"));
            }
        } else {
            panic!("Expected FormatMetadata::Bibtex");
        }
    }

    #[tokio::test]
    async fn test_extract_article_entry() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{einstein1905,
    author = {Albert Einstein},
    title = {On the Electrodynamics of Moving Bodies},
    journal = {Annalen der Physik},
    year = {1905},
    volume = {17},
    pages = {891-921}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract valid article entry");

        let metadata = &result.metadata;
        if let Some(authors) = &metadata.authors {
            assert!(!authors.is_empty());
            assert!(authors[0].contains("Einstein"));
        }
    }

    #[tokio::test]
    async fn test_extract_book_entry() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@book{knuth1984,
    author = {Donald E. Knuth},
    title = {The TeXbook},
    publisher = {Addison-Wesley},
    year = {1984}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract valid book entry");

        let metadata = &result.metadata;
        if let Some(FormatMetadata::Bibtex(bib)) = &metadata.format {
            assert_eq!(bib.entry_count, 1);
            if let Some(yr) = &bib.year_range {
                assert_eq!(yr.min, Some(1984));
                assert_eq!(yr.max, Some(1984));
            }
        } else {
            panic!("Expected FormatMetadata::Bibtex");
        }
    }

    #[tokio::test]
    async fn test_extract_metadata() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{paper1,
    author = {Alice Smith and Bob Jones},
    title = {Title 1},
    year = {2020}
}

@article{paper2,
    author = {Charlie Brown},
    title = {Title 2},
    year = {2021}
}

@book{book1,
    author = {David Lee},
    title = {Book Title},
    year = {2019}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract valid metadata");
        let metadata = &result.metadata;

        if let Some(FormatMetadata::Bibtex(bib)) = &metadata.format {
            assert_eq!(bib.entry_count, 3);

            if let Some(authors) = &metadata.authors {
                assert!(authors.len() >= 4);
            }

            if let Some(yr) = &bib.year_range {
                assert_eq!(yr.min, Some(2019));
                assert_eq!(yr.max, Some(2021));
            }

            if let Some(types) = &bib.entry_types {
                assert_eq!(types.get("article"), Some(&2));
                assert_eq!(types.get("book"), Some(&1));
            }
        } else {
            panic!("Expected FormatMetadata::Bibtex");
        }
    }

    #[tokio::test]
    async fn test_empty_bibliography() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = b"";

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract empty bibliography");
        let metadata = &result.metadata;

        if let Some(FormatMetadata::Bibtex(bib)) = &metadata.format {
            assert_eq!(bib.entry_count, 0);
        } else {
            panic!("Expected FormatMetadata::Bibtex");
        }
    }

    #[tokio::test]
    async fn test_malformed_entry() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{incomplete
    title = {Missing fields}

Some random text that's not valid BibTeX"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_authors_extraction() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{collab2022,
    author = {First Author and Second Author and Third Author},
    title = {Collaborative Work},
    year = {2022}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await;

        assert!(result.is_ok());
        let result = result.expect("Should extract multiple authors");
        let metadata = &result.metadata;

        if let Some(authors) = &metadata.authors {
            assert!(authors.len() >= 3);
        }
    }

    #[tokio::test]
    async fn test_bibtex_extractor_plugin_interface() {
        let extractor = BibtexExtractor::new();
        assert_eq!(extractor.name(), "bibtex-extractor");
        assert_eq!(extractor.version(), env!("CARGO_PKG_VERSION"));
        assert_eq!(extractor.priority(), 50);
        assert!(!extractor.supported_mime_types().is_empty());
    }

    #[test]
    fn test_bibtex_extractor_default() {
        let extractor = BibtexExtractor;
        assert_eq!(extractor.name(), "bibtex-extractor");
    }

    #[tokio::test]
    async fn test_bibtex_extractor_initialize_shutdown() {
        let extractor = BibtexExtractor::new();
        assert!(extractor.initialize().is_ok());
        assert!(extractor.shutdown().is_ok());
    }

    #[tokio::test]
    async fn test_bibtex_entry_fields_extraction() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@article{einstein1905,
    author = {Albert Einstein},
    title = {On the Electrodynamics of Moving Bodies},
    journal = {Annalen der Physik},
    year = {1905},
    volume = {17},
    pages = {891-921},
    doi = {10.1002/andp.19053220806},
    publisher = {Wiley}
}"#;

        let config = ExtractionConfig::default();
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await
            .expect("Should extract entry fields");

        let metadata = &result.metadata;

        let entries = metadata.additional.get(&Cow::Borrowed("entries"));
        assert!(entries.is_some(), "Should have entries metadata");
        let entries_array = entries
            .expect("entries key should be present")
            .as_array()
            .expect("entries should be an array");
        assert_eq!(entries_array.len(), 1);

        let entry = &entries_array[0];
        assert_eq!(
            entry
                .get("key")
                .expect("key field")
                .as_str()
                .expect("key should be string"),
            "einstein1905"
        );
        assert_eq!(
            entry
                .get("entry_type")
                .expect("entry_type field")
                .as_str()
                .expect("entry_type should be string"),
            "article"
        );
        assert!(entry.get("journal").is_some(), "Should have journal field");
        assert!(entry.get("volume").is_some(), "Should have volume field");
        assert!(entry.get("pages").is_some(), "Should have pages field");
        assert!(entry.get("doi").is_some(), "Should have doi field");
        assert!(entry.get("publisher").is_some(), "Should have publisher field");
    }

    #[tokio::test]
    async fn test_bibtex_document_structure_attributes() {
        let extractor = BibtexExtractor::new();
        let bibtex_content = br#"@book{knuth1984,
    author = {Donald E. Knuth},
    title = {The TeXbook},
    publisher = {Addison-Wesley},
    year = {1984},
    isbn = {0-201-13447-0}
}"#;

        let config = ExtractionConfig {
            include_document_structure: true,
            ..Default::default()
        };
        let result = extractor
            .extract_content(bibtex_content, "application/x-bibtex", &config)
            .await
            .expect("Should extract with document structure");

        assert!(!result.elements.is_empty(), "Document should have elements");
    }
}
