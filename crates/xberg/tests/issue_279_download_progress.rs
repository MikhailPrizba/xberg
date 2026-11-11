//! Compatibility checks for the data-only model configuration options.

use xberg::{EmbeddingConfig, LateInteractionConfig, RerankerConfig, SparseEmbeddingConfig};

#[test]
fn embedding_config_carries_show_download_progress_through_a_config_file() {
    for requested in [true, false] {
        let parsed = reparse_with_progress(&EmbeddingConfig::default(), requested);
        assert_eq!(
            parsed.show_download_progress, requested,
            "EmbeddingConfig must deserialize show_download_progress={requested} verbatim"
        );
    }
}

#[test]
fn sparse_embedding_config_carries_show_download_progress_through_a_config_file() {
    for requested in [true, false] {
        let parsed = reparse_with_progress(&SparseEmbeddingConfig::default(), requested);
        assert_eq!(
            parsed.show_download_progress, requested,
            "SparseEmbeddingConfig must deserialize show_download_progress={requested} verbatim"
        );
    }
}

#[test]
fn reranker_config_carries_show_download_progress_through_a_config_file() {
    for requested in [true, false] {
        let parsed = reparse_with_progress(&RerankerConfig::default(), requested);
        assert_eq!(
            parsed.show_download_progress, requested,
            "RerankerConfig must deserialize show_download_progress={requested} verbatim"
        );
    }
}

#[test]
fn late_interaction_config_carries_show_download_progress_through_a_config_file() {
    for requested in [true, false] {
        let parsed = reparse_with_progress(&LateInteractionConfig::default(), requested);
        assert_eq!(
            parsed.show_download_progress, requested,
            "LateInteractionConfig must deserialize show_download_progress={requested} verbatim"
        );
    }
}

#[test]
fn all_four_configs_default_to_no_progress() {
    assert!(!EmbeddingConfig::default().show_download_progress);
    assert!(!SparseEmbeddingConfig::default().show_download_progress);
    assert!(!RerankerConfig::default().show_download_progress);
    assert!(!LateInteractionConfig::default().show_download_progress);
}

/// Serialize a config, flip `show_download_progress`, and read it back — the same path a
/// `xberg.{toml,yaml,json}` file takes. A `#[serde(skip)]`ed or renamed field would drop the
/// override and fail here.
fn reparse_with_progress<T>(config: &T, requested: bool) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut value = serde_json::to_value(config).expect("config must serialize");
    value
        .as_object_mut()
        .expect("config must serialize as a JSON object")
        .insert("show_download_progress".to_string(), serde_json::Value::Bool(requested));
    serde_json::from_value(value).expect("config must deserialize")
}
