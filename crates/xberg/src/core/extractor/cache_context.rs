//! Mutable model dependencies of OSD recognition and optional PDF recovery.

use crate::cancellation::CancellationToken;
use crate::core::config::ExtractionConfig;

#[derive(Clone, Default)]
pub(crate) struct CacheDependency {}

impl CacheDependency {
    pub(crate) fn for_config(config: &ExtractionConfig, mime: Option<&str>) -> Self {
        {
            let _ = (config, mime);
            Self::default()
        }
    }
}

pub(crate) fn apply_context(key: Option<String>, context: &Option<String>) -> Option<String> {
    let key = key?;
    let context = context.as_deref()?;
    if context.is_empty() {
        return Some(key);
    }
    let mut hash = blake3::Hasher::new();
    hash.update(b"runtime-cache-context-v1");
    hash.update(&(key.len() as u64).to_le_bytes());
    hash.update(key.as_bytes());
    hash.update(context.as_bytes());
    Some(hash.finalize().to_hex().to_string())
}

pub(crate) async fn fingerprint(
    dependencies: &[CacheDependency],
    cancel: Option<&CancellationToken>,
) -> crate::Result<Option<String>> {
    {
        check_cancelled(cancel)?;
        let _ = dependencies;
        Ok(Some(String::new()))
    }
}

fn check_cancelled(cancel: Option<&CancellationToken>) -> crate::Result<()> {
    if cancel.is_some_and(CancellationToken::is_cancelled) {
        Err(crate::XbergError::Cancelled)
    } else {
        Ok(())
    }
}
