#![cfg(feature = "notebook")]

use xberg::{ExtractionConfig, NodeContent, UriKind};

mod helpers;
use helpers::extract_bytes_document;

fn structure_config() -> ExtractionConfig {
    ExtractionConfig {
        include_document_structure: true,
        ..Default::default()
    }
}

async fn extract(content: &[u8], mime_type: &str, config: &ExtractionConfig) -> xberg::ExtractedDocument {
    extract_bytes_document(content, mime_type, config)
        .await
        .expect("issue #1538 fixture should extract")
}

#[tokio::test]
async fn should_map_myst_directives_formulas_and_roles_without_leaking_syntax() {
    const MYST: &[u8] = br#"---
title: Relativity
---
# Findings

:::{note}
Mass and energy are related.
:::

```{math}
:label: eq-energy
E = mc^2
```

See {ref}`eq-energy` and {cite}`einstein1905`.
"#;

    let document = extract(MYST, "text/markdown", &ExtractionConfig::default()).await;
    let syntax_presence =
        [":::", ":label:", "```{math}", "{ref}", "{cite}"].map(|syntax| document.content.contains(syntax));
    assert_eq!(
        syntax_presence, [false; 5],
        "MyST control syntax must not leak into text"
    );
    let semantic_content = [
        "Findings",
        "Note",
        "Mass and energy are related.",
        "E = mc^2",
        "eq-energy",
        "einstein1905",
    ]
    .map(|text| document.content.contains(text));
    assert_eq!(semantic_content, [true; 6], "MyST constructs must retain their meaning");
    assert_eq!(document.metadata.title.as_deref(), Some("Relativity"));
}

#[tokio::test]
async fn should_parse_myst_text_notebook_frontmatter_and_code_cell() {
    const NOTEBOOK: &[u8] = br#"---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
title: Reproducible analysis
---
# Analysis

```{code-cell} python
:tags: [parameters]
answer = 6 * 7
```
"#;

    let document = extract(NOTEBOOK, "text/markdown", &ExtractionConfig::default()).await;
    let visible_content = ["Analysis", "answer = 6 * 7"].map(|text| document.content.contains(text));
    assert_eq!(visible_content, [true, true]);
    let leaked_syntax =
        ["```{code-cell}", ":tags:", "format_name: myst", "kernelspec:"].map(|text| document.content.contains(text));
    assert_eq!(leaked_syntax, [false; 4]);
    assert_eq!(document.metadata.title.as_deref(), Some("Reproducible analysis"));
    assert_eq!(document.metadata.additional["kernelspec"]["language"], "python");
    assert_eq!(document.metadata.additional["cells"][1]["cell_type"], "code");
    assert_eq!(
        document.metadata.additional["cells"][1]["tags"],
        serde_json::json!(["parameters"])
    );
}

#[tokio::test]
async fn should_preserve_braced_executable_and_unknown_code_fences() {
    const MARKDOWN: &[u8] = br#"```{python}
print("python")
```

```{r, echo=FALSE}
print("r")
```

```{mermaid}
graph TD
```
"#;
    let document = extract(MARKDOWN, "text/markdown", &structure_config()).await;
    let code = document
        .document
        .expect("document structure")
        .nodes
        .into_iter()
        .filter_map(|node| match node.content {
            NodeContent::Code { text, language } => Some((text, language)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        code,
        vec![
            ("print(\"python\")".into(), Some("python".into())),
            ("print(\"r\")".into(), Some("r".into())),
            ("graph TD".into(), Some("mermaid".into())),
        ]
    );
}

#[tokio::test]
async fn should_extract_unclosed_myst_code_cell_without_panicking() {
    const NOTEBOOK: &[u8] = br#"---
kernelspec:
  language: python
---
```{code-cell} python
print("still available")
"#;
    let document = extract(NOTEBOOK, "text/markdown", &structure_config()).await;
    assert!(document.content.contains("still available"));
    assert!(document.document.expect("document structure").nodes.iter().any(|node| {
        matches!(
            &node.content,
            NodeContent::Code { text, language }
                if text == "print(\"still available\")" && language.as_deref() == Some("python")
        )
    }));
}

#[tokio::test]
async fn should_map_nested_myst_directives_to_semantic_output() {
    const MYST: &[u8] = br#"::::{admonition} Custom guidance
:name: guidance
Outer body.
:::{tip}
Inner body.
:::
::::

```{figure} diagram.svg
:alt: System architecture
:name: fig-system
The system architecture caption.
```

```{code-block} rust
:caption: Example code
:name: code-example
fn main() {}
```

```{table} Result table
:name: table-results
| Name | Value |
| ---- | ----- |
| A    | 1     |
```

(eq-standalone)=
```{math}
:label: eq-energy
E = mc^2
```

See {ref}`fig-system`, {ref}`Architecture <fig-system>`, and {cite}`einstein1905`.
"#;
    let formula_document = extract(MYST, "text/markdown", &ExtractionConfig::default()).await;
    let document = extract(MYST, "text/markdown", &structure_config()).await;
    let structure = document.document.as_ref().expect("document structure");
    let admonitions = structure
        .nodes
        .iter()
        .filter_map(|node| match &node.content {
            NodeContent::Admonition { kind, title } => Some((kind.as_str(), title.as_deref())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        admonitions,
        vec![("note", Some("Custom guidance")), ("tip", Some("Tip"))]
    );
    assert!(structure.nodes.iter().any(|node| {
        matches!(
            &node.content,
            NodeContent::Code { text, language }
                if text == "fn main() {}" && language.as_deref() == Some("rust")
        )
    }));
    assert_eq!(
        document.tables[0].cells,
        vec![
            vec!["Name".to_string(), "Value".to_string()],
            vec!["A".to_string(), "1".to_string()],
        ]
    );
    assert_eq!(
        formula_document
            .formulas
            .iter()
            .map(|formula| formula.latex.as_str())
            .collect::<Vec<_>>(),
        ["E = mc^2"],
        "math directives must populate the public formula projection"
    );
    let uris = document.uris.as_ref().expect("semantic URIs");
    assert!(
        uris.iter()
            .any(|uri| uri.kind == UriKind::Image && uri.url == "diagram.svg")
    );
    assert!(
        uris.iter()
            .any(|uri| uri.kind == UriKind::Anchor && uri.url == "#fig-system")
    );
    assert!(
        uris.iter()
            .any(|uri| uri.kind == UriKind::Citation && uri.url == "cite:einstein1905")
    );
    for expected in [
        "Custom guidance",
        "Outer body.",
        "Inner body.",
        "System architecture",
        "The system architecture caption.",
        "Example code",
        "Result table",
        "Architecture",
    ] {
        assert!(
            document.content.contains(expected),
            "missing semantic content: {expected}"
        );
    }
    for syntax in [
        "::::",
        ":::",
        "```{figure}",
        "```{code-block}",
        "```{table}",
        ":name:",
        "{ref}",
        "{cite}",
    ] {
        assert!(!document.content.contains(syntax), "leaked MyST syntax: {syntax}");
    }
}

#[tokio::test]
async fn should_not_route_jupytext_markers_through_markdown_notebook_detection() {
    const MARKDOWN: &[u8] = b"# %% [markdown]\n# A literal marker\n# %%\nvalue = 1\n";
    let document = extract(MARKDOWN, "text/markdown", &ExtractionConfig::default()).await;
    assert!(document.metadata.additional.get("cells").is_none());
    assert!(document.content.contains("literal marker"));
    assert!(document.content.contains("value = 1"));
}

#[tokio::test]
async fn should_apply_text_notebook_tags_and_support_opt_out() {
    const NOTEBOOK: &[u8] = br#"---
kernelspec:
  language: python
---
```{code-cell} python
:tags: [hide-input]
hidden_source = 42
```
"#;
    let hidden = extract(NOTEBOOK, "text/markdown", &ExtractionConfig::default()).await;
    let visible = extract(
        NOTEBOOK,
        "text/markdown",
        &ExtractionConfig {
            apply_notebook_cell_tags: false,
            ..Default::default()
        },
    )
    .await;
    assert!(!hidden.content.contains("hidden_source = 42"));
    assert!(visible.content.contains("hidden_source = 42"));
    assert_eq!(
        hidden.metadata.additional["cells"][0]["tags"],
        serde_json::json!(["hide-input"])
    );
}

#[tokio::test]
async fn should_preprocess_myst_inside_ipynb_markdown_cells() {
    let notebook = serde_json::json!({
        "cells": [{
            "cell_type": "markdown",
            "metadata": {},
            "source": [":::{warning}\nSaved warning.\n:::\n\n```{math}\n:label: eq-saved\nx = 1\n```\nSee {ref}`eq-saved`.\n"]
        }],
        "metadata": {"kernelspec": {"language": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5
    });
    let notebook_bytes = notebook.to_string();
    let formula_document = extract(
        notebook_bytes.as_bytes(),
        "application/x-ipynb+json",
        &ExtractionConfig::default(),
    )
    .await;
    let document = extract(
        notebook_bytes.as_bytes(),
        "application/x-ipynb+json",
        &structure_config(),
    )
    .await;
    assert!(document.content.contains("Saved warning."));
    assert_eq!(
        formula_document
            .formulas
            .iter()
            .map(|formula| formula.latex.as_str())
            .collect::<Vec<_>>(),
        ["x = 1"]
    );
    assert!(
        document
            .document
            .expect("document structure")
            .nodes
            .iter()
            .any(|node| { matches!(&node.content, NodeContent::Admonition { kind, .. } if kind == "warning") })
    );
    for syntax in [":::", "```{math}", ":label:", "{ref}"] {
        assert!(!document.content.contains(syntax));
    }
}

#[tokio::test]
async fn should_extract_vendored_myst_notebook_fixture() {
    let bytes = std::fs::read("../../test_documents/vendored/executablebooks-myst-nb/basic_unrun.md")
        .expect("vendored MyST notebook fixture should be committed");
    let document = extract(&bytes, "text/markdown", &structure_config()).await;
    assert_eq!(document.metadata.title.as_deref(), Some("a title"));
    assert!(document.content.contains("created using"));
    assert!(document.content.contains("a = 1\nprint(a)"));
    assert_eq!(document.metadata.additional["kernelspec"]["language"], "python");
    assert!(document.document.expect("document structure").nodes.iter().any(|node| {
        matches!(&node.content, NodeContent::Code { language, .. } if language.as_deref() == Some("ipython3"))
    }));
    for syntax in ["```{code-cell}", "file_format: mystnb", "kernelspec:"] {
        assert!(!document.content.contains(syntax));
    }
}

#[tokio::test]
async fn should_apply_tags_from_vendored_ipynb_fixture() {
    let bytes = std::fs::read("../../test_documents/vendored/executablebooks-myst-nb/hide_cell_content.ipynb")
        .expect("vendored tagged notebook fixture should be committed");
    let hidden = extract(&bytes, "application/x-ipynb+json", &ExtractionConfig::default()).await;
    let visible = extract(
        &bytes,
        "application/x-ipynb+json",
        &ExtractionConfig {
            apply_notebook_cell_tags: false,
            ..Default::default()
        },
    )
    .await;
    assert!(!hidden.content.contains("print(\"hide-input\")"));
    assert!(hidden.content.contains("print(\"hide-output\")"));
    assert!(!hidden.content.contains("print(\"hide-cell\")"));
    assert!(!hidden.content.contains("hide-cell custom message"));
    assert!(visible.content.contains("print(\"hide-input\")"));
    assert!(visible.content.contains("hide-cell custom message"));
}

#[tokio::test]
async fn should_resolve_myst_eval_from_saved_user_expressions_without_execution() {
    const NOTEBOOK: &[u8] = br#"{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "summary",
      "metadata": {
        "user_expressions": [
          {
            "expression": "array.sum()",
            "result": {
              "data": {"text/plain": "6"},
              "metadata": {},
              "status": "ok"
            }
          }
        ]
      },
      "source": ["The total is {eval}`array.sum()`.\n"]
    }
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
  "nbformat": 4,
  "nbformat_minor": 5
}"#;

    let document = extract(NOTEBOOK, "application/x-ipynb+json", &ExtractionConfig::default()).await;
    assert!(document.content.contains("The total is 6."));
    assert!(!document.content.contains("{eval}"));
    assert!(!document.content.contains("array.sum()"));
    assert_eq!(
        document.metadata.additional["cells"][0]["user_expressions"][0]["expression"],
        "array.sum()"
    );
}

#[tokio::test]
async fn should_apply_remove_and_hide_tags_by_default() {
    let document = extract(
        tagged_notebook().as_bytes(),
        "application/x-ipynb+json",
        &ExtractionConfig::default(),
    )
    .await;

    let visibility = [
        ("remove_cell_source", false),
        ("remove_cell_output", false),
        ("hide_cell_source", false),
        ("hide_cell_output", false),
        ("remove_input_source", false),
        ("remove_input_output", true),
        ("hide_input_source", false),
        ("hide_input_output", true),
        ("remove_output_source", true),
        ("remove_output_output", false),
        ("hide_output_source", true),
        ("hide_output_output", false),
    ]
    .map(|(token, expected)| (document.content.contains(token), expected));
    assert_eq!(
        visibility.map(|(actual, _)| actual),
        visibility.map(|(_, expected)| expected)
    );

    let cells = document.metadata.additional["cells"]
        .as_array()
        .expect("cell metadata must remain available");
    assert_eq!(cells.len(), 6);
    assert_eq!(cells[0]["tags"], serde_json::json!(["remove-cell"]));
    assert_eq!(cells[5]["tags"], serde_json::json!(["hide-output"]));
}

#[tokio::test]
async fn should_restore_tagged_content_when_tag_handling_is_explicitly_disabled() {
    let config = ExtractionConfig {
        apply_notebook_cell_tags: false,
        ..Default::default()
    };
    let document = extract(tagged_notebook().as_bytes(), "application/x-ipynb+json", &config).await;
    let all_tokens = [
        "remove_cell_source",
        "remove_cell_output",
        "hide_cell_source",
        "hide_cell_output",
        "remove_input_source",
        "remove_input_output",
        "hide_input_source",
        "hide_input_output",
        "remove_output_source",
        "remove_output_output",
        "hide_output_source",
        "hide_output_output",
    ]
    .map(|token| document.content.contains(token));
    assert_eq!(all_tokens, [true; 12]);
}

fn tagged_notebook() -> String {
    let tagged_cells = [
        ("remove-cell", "remove_cell"),
        ("hide-cell", "hide_cell"),
        ("remove-input", "remove_input"),
        ("hide-input", "hide_input"),
        ("remove-output", "remove_output"),
        ("hide-output", "hide_output"),
    ]
    .map(|(tag, token)| {
        serde_json::json!({
            "cell_type": "code",
            "execution_count": 1,
            "id": token,
            "metadata": {"tags": [tag]},
            "outputs": [{"name": "stdout", "output_type": "stream", "text": [format!("{token}_output\n")] }],
            "source": [format!("{token}_source\n")]
        })
    });
    serde_json::json!({
        "cells": tagged_cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5
    })
    .to_string()
}
