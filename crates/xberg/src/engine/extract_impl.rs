//! Engine-internal extraction implementation.
//!
//! This module holds the extraction internals moved verbatim from
//! `core/extract/mod.rs`. The public free functions `crate::extract` /
//! `crate::extract_batch` delegate here via a process-global default
//! [`crate::engine::Engine`].

use crate::core::extractor::cache_context::{CacheDependency, apply_context, fingerprint};
use crate::core::extractor::cacheability;
use std::borrow::Cow;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::future::Future;
#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::sync::Arc;
#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::time::Instant;

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
type PendingBatchItem<I = ExtractInput> = (usize, I, String);

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn scope_batch_input_progress<F, T>(
    progress: Arc<dyn crate::engine::seams::ProgressSink>,
    index: usize,
    future: F,
) -> T
where
    F: Future<Output = T>,
{
    crate::engine::seams::scope_progress(progress, Some(index), future).await
}

use crate::core::config::{
    ExtractInput, ExtractInputKind, ExtractionConfig, ExtractionErrorItem, ExtractionResult, ExtractionSummary,
};
use crate::types::{ExtractedDocument, UriKind};
use crate::{Result, XbergError};

use crate::core::extractor::extract_file;
use crate::engine::seams::ProgressEvent;

mod input_ref;
pub(super) use input_ref::BatchInputRef;

const HTTP_SCHEME: &str = "http://";
const HTTPS_SCHEME: &str = "https://";
const FILE_SCHEME: &str = "file://";

/// Stable progress-sink stage labels for the single-input [`extract`] entry point.
///
/// Chosen at coarse, stable lifecycle points only (session start, cache hit,
/// completion, error) — never per-page or per-chunk — per the [`ProgressSink`]
/// "coarse progress" contract (`crate::engine::seams::ProgressSink`).
const PROGRESS_STAGE_START: &str = "extract_start";
const PROGRESS_STAGE_CACHE_HIT: &str = "extract_cache_hit";
const PROGRESS_STAGE_COMPLETE: &str = "extract_complete";
const PROGRESS_STAGE_ERROR: &str = "extract_error";
const BATCH_PROGRESS_STAGE_START: &str = "extract_batch_start";
const BATCH_PROGRESS_STAGE_CACHE_HIT: &str = "extract_batch_cache_hit";
const BATCH_PROGRESS_STAGE_COMPLETE: &str = "extract_batch_complete";
const BATCH_PROGRESS_STAGE_ERROR: &str = "extract_batch_error";

/// Namespace prefix mixed into the content-hash cache key so a future,
/// incompatible key derivation can never collide with entries this version wrote.
const CACHE_KEY_NAMESPACE: &[u8] = b"xberg-engine-extract-v2";
const BATCH_CACHE_KEY_NAMESPACE: &[u8] = b"xberg-engine-extract-batch-v2";

/// Extract content from a single bytes or URI input.
///
/// Honours the injected [`CacheBackend`](crate::engine::seams::CacheBackend) and
/// [`ProgressSink`](crate::engine::seams::ProgressSink) seams: a content-hash cache
/// hit (bytes inputs only, keyed on the raw bytes plus the resolved
/// [`ExtractionConfig`]) returns the cached [`ExtractionResult`] and skips
/// extraction entirely; a miss runs extraction as before and, on success,
/// populates the cache for next time. Both seams default to no-ops
/// ([`NoopCache`](crate::engine::seams::NoopCache),
/// [`NoopProgressSink`](crate::engine::seams::NoopProgressSink)), so callers who
/// inject nothing see byte-identical behavior.
pub(crate) async fn extract(
    inner: &super::EngineInner,
    input: ExtractInput,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    cacheability::scope(extract_scoped(inner, input, config)).await
}

async fn extract_scoped(
    inner: &super::EngineInner,
    input: ExtractInput,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    inner.progress.emit(ProgressEvent {
        stage: PROGRESS_STAGE_START.to_string(),
        message: None,
        fraction: Some(0.0),
    });

    if let Err(error) = config
        .validate()
        .and_then(|()| validate_input_overrides(std::slice::from_ref(&input)))
        .and_then(|()| ensure_not_cancelled(config))
    {
        inner.progress.emit(ProgressEvent {
            stage: PROGRESS_STAGE_ERROR.to_string(),
            message: Some(error.to_string()),
            fraction: None,
        });
        return Err(error);
    }

    let cache_key = inner.cache_enabled.then(|| content_cache_key(&input, config)).flatten();
    let dependencies = cache_key
        .as_ref()
        .map(|_| input_cache_dependency(&input, config))
        .into_iter()
        .collect::<Vec<_>>();
    let context = fingerprint(&dependencies, config.cancel_token.as_ref()).await?;
    let cache_key = apply_context(cache_key, &context);
    if let Some(key) = cache_key.as_deref()
        && let Some(cached_bytes) = inner.cache.get(key).await
        && let Ok(cached_result) = serde_json::from_slice::<ExtractionResult>(&cached_bytes)
        && cacheability::is_cacheable()
    {
        inner.progress.emit(ProgressEvent {
            stage: PROGRESS_STAGE_CACHE_HIT.to_string(),
            message: None,
            fraction: Some(1.0),
        });
        return Ok(cached_result);
    }

    // Type-erased so `extract`'s future does not nest `extract_uncached`'s whole
    // (already very deep) future type inside its own. Without this the compiler
    // exceeds the recursion limit proving `Send` for the combined future, and any
    // caller doing `tokio::spawn(xberg::extract(..))` — the canonical usage — fails
    // to compile with E0275. ~keep
    //
    // No `+ Send` on wasm32: extractor futures are `!Send` there (`async_trait(?Send)`,
    // see `plugins/extractor/trait.rs`), and the JS-backed futures pulled in through
    // crawlberg/reqwest's wasm backend (`JsFuture`, wasm-bindgen closures) are never
    // `Send` either. wasm32 has no OS threads and no `tokio::spawn`, so nothing needs
    // the bound there — mirrors the `extract_batch` split below. ~keep
    #[cfg(not(target_arch = "wasm32"))]
    let uncached: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + Send>> =
        Box::pin(extract_uncached(input, config));
    #[cfg(target_arch = "wasm32")]
    let uncached: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>>>> =
        Box::pin(extract_uncached(input, config));
    let result = uncached.await;

    match &result {
        Ok(output) => {
            if output.errors.is_empty()
                && cacheability::is_cacheable()
                && let Some(key) = cache_key
                && context == fingerprint(&dependencies, config.cancel_token.as_ref()).await?
                && cacheability::is_cacheable()
                && let Ok(serialized) = serde_json::to_vec(output)
            {
                inner.cache.put(&key, serialized, None).await;
            }
            inner.progress.emit(ProgressEvent {
                stage: PROGRESS_STAGE_COMPLETE.to_string(),
                message: None,
                fraction: Some(1.0),
            });
        }
        Err(error) => {
            inner.progress.emit(ProgressEvent {
                stage: PROGRESS_STAGE_ERROR.to_string(),
                message: Some(error.to_string()),
                fraction: None,
            });
        }
    }

    result
}

fn ensure_not_cancelled(config: &ExtractionConfig) -> Result<()> {
    if config.cancel_token.as_ref().is_some_and(|token| token.is_cancelled()) {
        return Err(XbergError::Cancelled);
    }
    Ok(())
}

/// The extraction path proper, unwrapped from cache/progress bookkeeping so
/// [`extract`] can wrap it uniformly for both the cache-hit and cache-miss cases.
async fn extract_uncached(input: ExtractInput, config: &ExtractionConfig) -> Result<ExtractionResult> {
    let mut seen = initial_seen_urls(std::slice::from_ref(&input));
    let seed_hosts = initial_seed_hosts(std::slice::from_ref(&input));
    let mut output = Box::pin(extract_one(input, config, 0)).await?;

    follow_recursive_document_urls(&mut output, config, &mut seen, &seed_hosts).await?;

    Ok(output)
}

fn input_cache_dependency<I: BatchInputRef>(input: &I, config: &ExtractionConfig) -> CacheDependency {
    CacheDependency::for_config(
        &effective_input_config(input, config),
        input.input().mime_type.as_deref(),
    )
}

/// Derive a content-hash cache key for `input`, or `None` when the input is not
/// eligible for caching.
///
/// Only `bytes` inputs are cached, per the cache-key contract (content-hash of
/// file bytes + config, never path-based): a `uri` input's content is not yet
/// known at this point without fetching it, which would defeat the purpose of a
/// pre-extraction cache check. The key mixes the byte length and content, the
/// MIME/filename hints, and the fully-resolved [`ExtractionConfig`] (base config
/// merged with any per-input override), so a config or override change is a
/// guaranteed cache miss. The shared extraction cache version applies to both
/// single and batch keys through this item key.
fn content_cache_key<I: BatchInputRef>(item: &I, base_config: &ExtractionConfig) -> Option<String> {
    let input = item.input();
    if !base_config.use_cache || input.kind != ExtractInputKind::Bytes {
        return None;
    }
    let bytes = input.bytes.as_deref()?;
    let resolved_config = effective_input_config(item, base_config);
    // A findings file is mutable external input whose contents are absent from the cache key. ~keep
    if resolved_config
        .redaction
        .as_ref()
        .is_some_and(|redaction| redaction.findings_path.is_some())
    {
        return None;
    }
    let config_json = serde_json::to_vec(&resolved_config).ok()?;

    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_KEY_NAMESPACE);
    hasher.update(crate::cache::version::cache_version_tag().as_bytes());
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.update(input.mime_type.as_deref().unwrap_or_default().as_bytes());
    hasher.update(b"\0");
    hasher.update(input.filename.as_deref().unwrap_or_default().as_bytes());
    hasher.update(b"\0");
    hasher.update(&config_json);
    Some(hasher.finalize().to_hex().to_string())
}

fn batch_content_cache_key<I: BatchInputRef>(inputs: &[I], base_config: &ExtractionConfig) -> Option<String> {
    if inputs.is_empty() {
        return None;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(BATCH_CACHE_KEY_NAMESPACE);
    hasher.update(&(inputs.len() as u64).to_le_bytes());
    for input in inputs {
        let item_key = content_cache_key(input, base_config)?;
        hasher.update(&(item_key.len() as u64).to_le_bytes());
        hasher.update(item_key.as_bytes());
    }
    Some(hasher.finalize().to_hex().to_string())
}

/// Extract content from multiple bytes or URI inputs.
pub(super) async fn extract_batch<I: BatchInputRef>(
    inner: &super::EngineInner,
    inputs: Vec<I>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    cacheability::scope(extract_batch_scoped(inner, inputs, config)).await
}

async fn extract_batch_scoped<I: BatchInputRef>(
    inner: &super::EngineInner,
    inputs: Vec<I>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    inner.progress.emit(ProgressEvent {
        stage: BATCH_PROGRESS_STAGE_START.to_string(),
        message: None,
        fraction: Some(0.0),
    });

    if let Err(error) = config
        .validate()
        .and_then(|()| validate_input_overrides(inputs.iter().map(BatchInputRef::input)))
        .and_then(|()| ensure_not_cancelled(config))
    {
        inner.progress.emit(ProgressEvent {
            stage: BATCH_PROGRESS_STAGE_ERROR.to_string(),
            message: Some(error.to_string()),
            fraction: None,
        });
        return Err(error);
    }

    let cache_key = inner
        .cache_enabled
        .then(|| batch_content_cache_key(&inputs, config))
        .flatten();
    let dependencies = if cache_key.is_some() {
        inputs
            .iter()
            .map(|input| input_cache_dependency(input, config))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let context = fingerprint(&dependencies, config.cancel_token.as_ref()).await?;
    let cache_key = apply_context(cache_key, &context);
    if let Some(key) = cache_key.as_deref()
        && let Some(cached_bytes) = inner.cache.get(key).await
        && let Ok(cached_result) = serde_json::from_slice::<ExtractionResult>(&cached_bytes)
        && cacheability::is_cacheable()
    {
        inner.progress.emit(ProgressEvent {
            stage: BATCH_PROGRESS_STAGE_CACHE_HIT.to_string(),
            message: None,
            fraction: Some(1.0),
        });
        return Ok(cached_result);
    }

    let result = extract_batch_uncached(inner, inputs, config).await;
    match &result {
        Ok(output) => {
            if output.errors.is_empty()
                && cacheability::is_cacheable()
                && let Some(key) = cache_key
                && context == fingerprint(&dependencies, config.cancel_token.as_ref()).await?
                && cacheability::is_cacheable()
                && let Ok(serialized) = serde_json::to_vec(output)
            {
                inner.cache.put(&key, serialized, None).await;
            }
            inner.progress.emit(ProgressEvent {
                stage: BATCH_PROGRESS_STAGE_COMPLETE.to_string(),
                message: None,
                fraction: Some(1.0),
            });
        }
        Err(error) => {
            inner.progress.emit(ProgressEvent {
                stage: BATCH_PROGRESS_STAGE_ERROR.to_string(),
                message: Some(error.to_string()),
                fraction: None,
            });
        }
    }
    result
}

async fn extract_batch_uncached<I: BatchInputRef>(
    inner: &super::EngineInner,
    inputs: Vec<I>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    // `Instant::now()` panics on wasm32 (no usable timer there, see the wasm32 note on
    // `extract_file_uncached`'s timeout handling), so the batch counter/histogram pair is
    // skipped on that target rather than risking a panic for a metrics-only side effect. ~keep

    // `extract_batch_concurrent` spawns tasks on `tokio::task::JoinSet`, which requires `Send`
    // futures; extractor futures are `!Send` on wasm32 (async_trait(?Send), see
    // plugins/extractor/trait.rs) and wasm32 has no OS threads to run them on regardless. Use
    // the sequential path there even though `tokio-runtime` is active (it's pulled in by
    // `chunking-tokenizers`/`static-embeddings`, not concurrency support). ~keep
    // Type-erased for the same reason as the single-extract path above: leaving the
    // inner future's concrete type visible here makes callers that `tokio::spawn` a
    // batch exceed the recursion limit proving `Send`. That bit the generated
    // xberg-node binding (`run_bounded_batch_tasks`), which cannot carry a
    // `#![recursion_limit]` of its own because it is regenerated from scratch. ~keep
    #[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
    let batch: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + Send>> =
        Box::pin(extract_batch_concurrent(inner, inputs, config));

    // No `+ Send` on wasm32: extractor futures are `!Send` there (async_trait(?Send)). ~keep
    #[cfg(any(not(feature = "tokio-runtime"), target_arch = "wasm32"))]
    let batch: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>>>> = {
        let _ = inner;
        Box::pin(extract_batch_sequential(inputs, config))
    };

    let result = batch.await;

    result
}

#[cfg(any(not(feature = "tokio-runtime"), target_arch = "wasm32"))]
async fn extract_batch_sequential<I: BatchInputRef>(
    inputs: Vec<I>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    let mut seen = initial_seen_urls(inputs.iter().map(BatchInputRef::input));
    let seed_hosts = initial_seed_hosts(inputs.iter().map(BatchInputRef::input));
    let mut output = ExtractionResult {
        summary: ExtractionSummary {
            inputs: inputs.len(),
            ..Default::default()
        },
        ..Default::default()
    };

    for (index, input) in inputs.into_iter().enumerate() {
        let source = input_source(input.input());
        let resolved_config = resolve_input_config(&input, config);
        let item = Box::pin(extract_one_resolved(input.into_input(), &resolved_config, index));

        let item_result = item.await;

        match item_result {
            Ok(item_output) => append_extraction_output(&mut output, item_output),
            Err(error) => output.errors.push(error_item(index, source, &error)),
        }
    }

    output.refresh_counts();
    follow_recursive_document_urls(&mut output, config, &mut seen, &seed_hosts).await?;
    Ok(output)
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn extract_batch_concurrent<I: BatchInputRef>(
    inner: &super::EngineInner,
    inputs: Vec<I>,
    config: &ExtractionConfig,
) -> Result<ExtractionResult> {
    let input_count = inputs.len();
    let mut seen = initial_seen_urls(inputs.iter().map(BatchInputRef::input));
    let seed_hosts = initial_seed_hosts(inputs.iter().map(BatchInputRef::input));
    let mut output = ExtractionResult {
        summary: ExtractionSummary {
            inputs: input_count,
            ..Default::default()
        },
        ..Default::default()
    };

    if input_count == 0 {
        return Ok(output);
    }

    // Batch workers each receive a divided per-document budget, but Rayon is global and
    // immutable after first initialization, so the pools take the whole batch budget here
    // rather than the smaller share the first worker to start would otherwise install.
    crate::core::config::concurrency::init_thread_pools(config.concurrency.as_ref());
    let base_config = Arc::new(config.clone());
    let mut pending: VecDeque<PendingBatchItem<I>> = VecDeque::with_capacity(input_count);

    let mut items: Vec<Option<BatchItemResult>> = Vec::with_capacity(input_count);
    items.resize_with(input_count, || None);

    for (index, input) in inputs.into_iter().enumerate() {
        let source = input_source(input.input());

        pending.push_back((index, input, source));
    }

    let execution_plan = resolve_pending_batch_execution_plan(config, &pending);
    let pending = if should_prioritize_pending_batch_items(pending.len(), execution_plan.workers) {
        prioritize_pending_batch_items(pending, config).await
    } else {
        pending
    };
    let task_config = resolve_batch_base_config(&base_config, execution_plan.thread_budget);
    let progress = Arc::clone(&inner.progress);
    let completed = run_bounded_batch_tasks(pending, execution_plan.workers, move |(index, input, source)| {
        let base_config = Arc::clone(&task_config);
        let progress = Arc::clone(&progress);
        async move {
            let resolved_config = resolve_batch_input_config(&input, &base_config, execution_plan.thread_budget);
            let timeout_secs = resolved_config.extraction_timeout_secs;
            let cancel_token = resolved_config.cancel_token.clone();
            let item = run_batch_item(index, source, timeout_secs, cancel_token, || async move {
                // Type-erased, not merely boxed. `run_bounded_batch_tasks` requires the
                // enclosing async block to be `Send`, and proving that walks the whole
                // nested future type -- which reaches h2/hyper/slab through reqwest and
                // overflows rustc's 128 auto-trait recursion limit (E0275) in the
                // alef-generated binding crates, which cannot carry `#![recursion_limit]`.
                // A bare `Box::pin` does not help: it yields `Pin<Box<ConcreteFuture>>`,
                // so the concrete type stays in the proof. Coercing to `dyn Future` cuts
                // the chain here. ~keep
                let extraction: std::pin::Pin<Box<dyn Future<Output = Result<ExtractionResult>> + Send + '_>> =
                    Box::pin(extract_one_resolved(input.into_input(), &resolved_config, index));
                scope_batch_input_progress(progress, index, extraction).await
            });

            item.await
        }
    })
    .await?;

    for item in completed {
        let index = item.index;
        if index < items.len() {
            items[index] = Some(item);
        } else {
            return Err(XbergError::Other(format!("batch task returned invalid index: {index}")));
        }
    }

    let _ = inner;

    for item in items.into_iter().flatten() {
        match item.result {
            Ok(item_output) => append_extraction_output(&mut output, item_output),
            Err(error) => output.errors.push(error_item(item.index, item.source, &error)),
        }
    }

    output.refresh_counts();
    follow_recursive_document_urls(&mut output, config, &mut seen, &seed_hosts).await?;
    Ok(output)
}

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_engine_batch_execution_plan<I: BatchInputRef>(
    config: &ExtractionConfig,
    inputs: &[I],
) -> crate::core::config::concurrency::BatchExecutionPlan {
    resolve_engine_batch_execution_plan_for(config, classify_layout_batch(config, inputs), inputs.len())
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_pending_batch_execution_plan<I: BatchInputRef>(
    config: &ExtractionConfig,
    pending: &VecDeque<PendingBatchItem<I>>,
) -> crate::core::config::concurrency::BatchExecutionPlan {
    resolve_engine_batch_execution_plan_for(
        config,
        classify_layout_batch(config, pending.iter().map(|(_, input, _)| input)),
        pending.len(),
    )
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn should_prioritize_pending_batch_items(pending_count: usize, workers: usize) -> bool {
    workers > 1 && pending_count > workers
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn prioritize_pending_batch_items<I: BatchInputRef>(
    pending: VecDeque<PendingBatchItem<I>>,
    base_config: &ExtractionConfig,
) -> VecDeque<PendingBatchItem<I>> {
    const MAX_CONCURRENT_SIZE_HINTS: usize = 16;
    const SIZE_HINT_BUDGET: std::time::Duration = std::time::Duration::from_millis(25);

    let mut costs = vec![None; pending.len()];
    let mut local_paths = Vec::new();

    for (position, (_, item, _)) in pending.iter().enumerate() {
        let input = item.input();
        match input.kind {
            ExtractInputKind::Bytes => {
                costs[position] = input.bytes.as_ref().map(|bytes| bytes.len() as u64);
            }
            ExtractInputKind::Uri => {
                let effective_config = effective_input_config(item, base_config);
                let cancelled = effective_config
                    .cancel_token
                    .as_ref()
                    .is_some_and(crate::cancellation::CancellationToken::is_cancelled);
                if !cancelled && let Some(path) = local_batch_path(input, &effective_config) {
                    local_paths.push((position, path));
                }
            }
        }
    }

    let collect_costs = async {
        let mut pending_paths: VecDeque<_> = local_paths.into();
        let mut probes = tokio::task::JoinSet::new();
        while probes.len() < MAX_CONCURRENT_SIZE_HINTS {
            let Some((position, path)) = pending_paths.pop_front() else {
                break;
            };
            probes.spawn(metadata_size_hint(position, path));
        }

        while let Some(result) = probes.join_next().await {
            match result {
                Ok((position, Some(cost))) => costs[position] = Some(cost),
                Ok((position, None)) => {
                    tracing::debug!(position, "batch input size hint unavailable; preserving FIFO slot");
                }
                Err(error) => {
                    tracing::debug!(%error, "batch input size hint task failed; preserving FIFO slot");
                }
            }

            if let Some((position, path)) = pending_paths.pop_front() {
                probes.spawn(metadata_size_hint(position, path));
            }
        }
    };
    if tokio::time::timeout(SIZE_HINT_BUDGET, collect_costs).await.is_err() {
        tracing::debug!("batch input size hint budget exhausted; preserving unfinished FIFO slots");
    }

    reorder_pending_batch_items(pending, &costs)
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn metadata_size_hint(position: usize, path: PathBuf) -> (usize, Option<u64>) {
    let cost = tokio::fs::metadata(path).await.ok().map(|metadata| metadata.len());
    (position, cost)
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn local_batch_path(input: &ExtractInput, config: &ExtractionConfig) -> Option<PathBuf> {
    let uri = input.uri.as_deref()?;
    if uri.starts_with(HTTP_SCHEME) || uri.starts_with(HTTPS_SCHEME) {
        return None;
    }
    if uri.starts_with(FILE_SCHEME) {
        return config.url.allow_file_uris.then(|| file_uri_to_path(uri).ok()).flatten();
    }
    if uri.contains("://") {
        return None;
    }
    config.url.allow_local_file_inputs.then(|| PathBuf::from(uri))
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn reorder_pending_batch_items<I>(
    pending: VecDeque<PendingBatchItem<I>>,
    costs: &[Option<u64>],
) -> VecDeque<PendingBatchItem<I>> {
    let mut prioritized = Vec::new();
    let mut prioritized_positions = Vec::new();
    let mut scheduled = Vec::with_capacity(pending.len());

    for (position, item) in pending.into_iter().enumerate() {
        if let Some(cost) = costs.get(position).copied().flatten() {
            prioritized_positions.push(position);
            prioritized.push((cost, position, item));
            scheduled.push(None);
        } else {
            scheduled.push(Some(item));
        }
    }

    prioritized.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    for (position, (_, _, item)) in prioritized_positions.into_iter().zip(prioritized) {
        scheduled[position] = Some(item);
    }

    scheduled.into_iter().flatten().collect()
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn classify_layout_batch<'a, I: BatchInputRef + 'a>(
    config: &ExtractionConfig,
    inputs: impl IntoIterator<Item = &'a I>,
) -> crate::core::config::concurrency::LayoutBatchWorkload {
    {
        let _ = (config, inputs);
        crate::core::config::concurrency::LayoutBatchWorkload::None
    }
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_engine_batch_execution_plan_for(
    config: &ExtractionConfig,
    layout_workload: crate::core::config::concurrency::LayoutBatchWorkload,
    input_count: usize,
) -> crate::core::config::concurrency::BatchExecutionPlan {
    crate::core::config::concurrency::resolve_batch_execution_plan(
        config.concurrency.as_ref(),
        layout_workload,
        input_count,
        config.max_concurrent_extractions,
    )
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn run_bounded_batch_tasks<T, F, Fut>(
    mut pending: VecDeque<T>,
    max_concurrent: usize,
    mut task: F,
) -> Result<Vec<BatchItemResult>>
where
    T: Send + 'static,
    F: FnMut(T) -> Fut,
    Fut: Future<Output = BatchItemResult> + Send + 'static,
{
    use tokio::task::JoinSet;

    let mut tasks = JoinSet::new();
    let max_concurrent = max_concurrent.max(1);
    while tasks.len() < max_concurrent {
        let Some(item) = pending.pop_front() else {
            break;
        };
        let future = task(item);
        #[cfg(all(test, feature = "pdf", feature = "ocr-pipeline"))]
        let future = crate::plugins::registry::inherit_test_ocr_backend(future);
        tasks.spawn(cacheability::inherit(future));
    }

    let mut completed = Vec::with_capacity(tasks.len() + pending.len());
    while let Some(task_result) = tasks.join_next().await {
        let item = task_result.map_err(|error| XbergError::Other(format!("batch task failed to join: {error}")))?;
        completed.push(item);
        if let Some(next) = pending.pop_front() {
            let future = task(next);
            #[cfg(all(test, feature = "pdf", feature = "ocr-pipeline"))]
            let future = crate::plugins::registry::inherit_test_ocr_backend(future);
            tasks.spawn(cacheability::inherit(future));
        }
    }
    Ok(completed)
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
struct BatchItemResult {
    index: usize,
    source: String,
    result: Result<ExtractionResult>,
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
async fn run_batch_item<F, Fut>(
    index: usize,
    source: String,
    timeout_secs: Option<u64>,
    cancel_token: Option<crate::cancellation::CancellationToken>,
    extract_fn: F,
) -> BatchItemResult
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<ExtractionResult>>,
{
    let start = Instant::now();
    let extraction_future = Box::pin(crate::core::batch_mode::with_batch_mode(Box::pin(extract_fn())));

    let mut result = match timeout_secs {
        Some(secs) => match tokio::time::timeout(std::time::Duration::from_secs(secs), extraction_future).await {
            Ok(inner) => inner,
            Err(_elapsed) => {
                if let Some(ref token) = cancel_token {
                    token.cancel();
                }
                Err(XbergError::Timeout {
                    elapsed_ms: start.elapsed().as_millis() as u64,
                    limit_ms: secs * 1000,
                })
            }
        },
        None => extraction_future.await,
    };

    if let Ok(ref mut item_output) = result {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        for extraction_result in &mut item_output.results {
            extraction_result.metadata.extraction_duration_ms = Some(elapsed_ms);
        }
    }

    BatchItemResult { index, source, result }
}

fn append_extraction_output(output: &mut ExtractionResult, mut item_output: ExtractionResult) {
    output.summary.remote_urls += item_output.summary.remote_urls;
    output.summary.pages_crawled += item_output.summary.pages_crawled;
    output.summary.documents_downloaded += item_output.summary.documents_downloaded;
    output.results.append(&mut item_output.results);
    output.errors.append(&mut item_output.errors);
    merge_crawl_summary(
        output,
        item_output.crawl_final_urls,
        item_output.crawl_redirect_count,
        item_output.crawl_unique_normalized_urls,
    );
}

async fn extract_one(input: ExtractInput, base_config: &ExtractionConfig, index: usize) -> Result<ExtractionResult> {
    let config = resolve_input_config(&input, base_config);
    extract_one_resolved(input, &config, index).await
}

/// Validate the page settings and layout model of each input that overrides them.
///
/// An override replaces the page settings of the base config as a whole, so its own values decide.
fn validate_input_overrides<'a>(inputs: impl IntoIterator<Item = &'a ExtractInput>) -> Result<()> {
    for input in inputs {
        if let Some(pages) = input.config.as_ref().and_then(|overrides| overrides.pages.as_ref()) {
            pages.validate()?;
        }
        #[cfg(feature = "layout-types")]
        if let Some(layout) = input.config.as_ref().and_then(|overrides| overrides.layout.as_ref()) {
            layout.validate_backend()?;
        }
    }
    Ok(())
}

fn resolve_input_config<I: BatchInputRef>(input: &I, base_config: &ExtractionConfig) -> ExtractionConfig {
    let mut resolved = effective_input_config(input, base_config).into_owned();
    // Install a token here, before this item's own timeout wrapper (`run_batch_item`,
    // `finalize_shared_item`) races against `extract_file`/`extract_bytes`'s inner
    // timeout — the outer wrapper starts first and has the same duration, so it always
    // wins that race, and its `token.cancel()` needs the SAME token this config's
    // extractor checkpoints observe. See `ExtractionConfig::ensure_cancel_token`.
    resolved.ensure_cancel_token();
    resolved
}

fn effective_input_config<'a, I: BatchInputRef>(
    input: &'a I,
    base_config: &'a ExtractionConfig,
) -> Cow<'a, ExtractionConfig> {
    let mut resolved = match input.input().config.as_ref() {
        Some(overrides) => Cow::Owned(base_config.with_file_overrides(overrides)),
        None => Cow::Borrowed(base_config),
    };
    if let Some(use_layout_for_markdown) = input.use_layout_for_markdown() {
        resolved.to_mut().use_layout_for_markdown = use_layout_for_markdown;
    }
    resolved
}

/// Resolve config for batch items, taking Arc<ExtractionConfig> to avoid unnecessary clones.
/// When there are no per-item overrides, this returns Arc::clone (cheap reference increment)
/// rather than cloning the inner ExtractionConfig.
#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_input_config_arc<I: BatchInputRef>(input: &I, base_config: &Arc<ExtractionConfig>) -> Arc<ExtractionConfig> {
    match effective_input_config(input, base_config) {
        Cow::Owned(resolved) => Arc::new(resolved),
        Cow::Borrowed(_) => Arc::clone(base_config),
    }
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_batch_input_config<I: BatchInputRef>(
    input: &I,
    base_config: &Arc<ExtractionConfig>,
    thread_budget: usize,
) -> Arc<ExtractionConfig> {
    let resolved = resolve_input_config_arc(input, base_config);
    let needs_thread_budget =
        crate::core::config::concurrency::resolve_thread_budget(resolved.concurrency.as_ref()) != thread_budget;
    // See `ExtractionConfig::ensure_cancel_token` and `resolve_input_config`: this
    // item's `run_batch_item` timeout wrapper needs the same token its extractor
    // checkpoints observe, installed before that wrapper races `extract_one_resolved`.
    let needs_cancel_token = resolved.extraction_timeout_secs.is_some() && resolved.cancel_token.is_none();
    if !needs_thread_budget && !needs_cancel_token {
        return resolved;
    }

    let mut resolved = Arc::unwrap_or_clone(resolved);
    if needs_thread_budget {
        // Only the thread budget is divided across batch workers. A caller's
        // `max_concurrent_ocr` is a memory bound on the host, not a per-document
        // share, so it survives the rewrite; rebuilding the struct from scratch
        // silently dropped it. ~keep
        resolved.concurrency = Some(crate::core::config::ConcurrencyConfig {
            max_threads: Some(thread_budget),
            ..resolved.concurrency.unwrap_or_default()
        });
    }
    resolved.ensure_cancel_token();
    Arc::new(resolved)
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
fn resolve_batch_base_config(base_config: &Arc<ExtractionConfig>, thread_budget: usize) -> Arc<ExtractionConfig> {
    if crate::core::config::concurrency::resolve_thread_budget(base_config.concurrency.as_ref()) == thread_budget {
        return Arc::clone(base_config);
    }

    let mut resolved = (**base_config).clone();
    // As in `resolve_batch_input_config`: divide the thread budget, carry the rest. ~keep
    resolved.concurrency = Some(crate::core::config::ConcurrencyConfig {
        max_threads: Some(thread_budget),
        ..resolved.concurrency.unwrap_or_default()
    });
    Arc::new(resolved)
}

async fn extract_one_resolved(
    input: ExtractInput,
    config: &ExtractionConfig,
    index: usize,
) -> Result<ExtractionResult> {
    // Type-erased per arm. An inline `.await` embeds BOTH arms' coroutines in this
    // function's state, and that state is inlined into `extract_one`, which is
    // alloca'd at four `Box::pin` sites and embedded again in `extract_uncached` --
    // so every byte here is paid many times over. Boxing leaves an 8-byte pointer and
    // makes the two arms mutually exclusive in memory as well as in control flow.
    //
    // No `+ Send` on wasm32; see the `extract` entry point above. ~keep
    #[cfg(not(target_arch = "wasm32"))]
    let resolved: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + Send + '_>> =
        match input.kind {
            ExtractInputKind::Bytes => Box::pin(extract_bytes_input(input, config, index)),
            ExtractInputKind::Uri => Box::pin(extract_uri_input(input, config, index)),
        };
    #[cfg(target_arch = "wasm32")]
    let resolved: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + '_>> = match input.kind
    {
        ExtractInputKind::Bytes => Box::pin(extract_bytes_input(input, config, index)),
        ExtractInputKind::Uri => Box::pin(extract_uri_input(input, config, index)),
    };
    resolved.await
}

async fn extract_bytes_input(input: ExtractInput, config: &ExtractionConfig, index: usize) -> Result<ExtractionResult> {
    let filename = input.filename;
    let mime_type = input.mime_type;
    let bytes = input
        .bytes
        .ok_or_else(|| XbergError::validation("extract input kind 'bytes' requires the 'bytes' field".to_string()))?;
    let (bytes, mime_type) = resolve_owned_bytes_mime(bytes, mime_type, filename.clone(), config).await?;
    let mut cfg = config.clone();
    cfg.source_name.clone_from(&filename);
    let mut result = Box::pin(crate::core::extractor::extract_bytes(&bytes, &mime_type, &cfg)).await?;

    annotate_source(
        &mut result,
        "bytes",
        filename.as_deref().unwrap_or("<bytes>"),
        filename.as_deref().unwrap_or("<bytes>"),
        index,
    );
    Ok(ExtractionResult::single(result))
}

async fn extract_uri_input(input: ExtractInput, config: &ExtractionConfig, index: usize) -> Result<ExtractionResult> {
    let uri = input
        .uri
        .ok_or_else(|| XbergError::validation("extract input kind 'uri' requires the 'uri' field".to_string()))?;

    if uri.starts_with(HTTP_SCHEME) || uri.starts_with(HTTPS_SCHEME) {
        // Type-erased, not merely boxed, and for a stack reason rather than a `Send`-proof one.
        // This branch is not taken for the overwhelmingly common local-path/`file://` input, yet
        // an inline `.await` folds the entire URL-ingestion subtree — `extract_remote_uri` (which
        // holds a `CrawlConfig` by value plus a crawlberg engine future), `output_from_scrape`,
        // `output_from_crawl`, `extract_downloaded_document` and `run_url_page_pipeline`, the last
        // two each embedding a full unboxed `extract_bytes` pipeline — into this coroutine's TYPE.
        // That type is paid unconditionally: it inflates `extract_one_resolved` -> `extract_one`,
        // and `extract_one` is materialised in an `alloca` at every `Box::pin(extract_one(..))`
        // call site (rust-lang/rust#54628), so the cost lands on the stack of every extraction,
        // remote or not. A bare `Box::pin` does NOT fix this: `Pin<Box<Concrete>>` keeps the
        // concrete coroutine as a type parameter. Coercing to `dyn Future` turns the whole subtree
        // into a 16-byte pointer here and moves its materialisation into a frame cost paid only
        // when an HTTP(S) URI is actually seen. ~keep
        //
        // No `+ Send` on wasm32, for the same reason as `extract`/`extract_batch` above: extractor
        // futures are `!Send` there (`async_trait(?Send)`), as are crawlberg/reqwest's JS-backed
        // wasm futures, and wasm32 has no `tokio::spawn` that would need the bound. ~keep
        #[cfg(not(target_arch = "wasm32"))]
        let remote: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + Send + '_>> =
            Box::pin(extract_remote_uri(&uri, config, index));
        #[cfg(target_arch = "wasm32")]
        let remote: std::pin::Pin<Box<dyn std::future::Future<Output = Result<ExtractionResult>> + '_>> =
            Box::pin(extract_remote_uri(&uri, config, index));
        return remote.await;
    }

    if uri.contains("://") && !uri.starts_with(FILE_SCHEME) {
        return Err(XbergError::UnsupportedFormat(format!(
            "unsupported URI scheme for extraction input: {uri}"
        )));
    }

    let path = if uri.starts_with(FILE_SCHEME) {
        if !config.url.allow_local_file_inputs || !config.url.allow_file_uris {
            return Err(XbergError::validation(
                "file:// URI inputs are disabled by configuration".to_string(),
            ));
        }
        file_uri_to_path(&uri)?
    } else {
        if !config.url.allow_local_file_inputs {
            return Err(XbergError::validation(
                "local filesystem path inputs are disabled by configuration".to_string(),
            ));
        }
        PathBuf::from(&uri)
    };

    let mut result = Box::pin(extract_file(&path, input.mime_type.as_deref(), config)).await?;
    annotate_source(&mut result, "uri", &uri, path.to_string_lossy().as_ref(), index);
    Ok(ExtractionResult::single(result))
}

async fn resolve_owned_bytes_mime(
    bytes: Vec<u8>,
    mime_type: Option<String>,
    filename: Option<String>,
    config: &ExtractionConfig,
) -> Result<(Vec<u8>, String)> {
    crate::core::mime::resolve_owned_bytes_mime(bytes, filename, mime_type, config.mime_detection_policy).await
}

#[cfg(all(test, feature = "tokio-runtime"))]
fn resolve_bytes_mime_type(
    mime_type: Option<&str>,
    filename: Option<&str>,
    bytes: &[u8],
    config: &ExtractionConfig,
) -> Result<String> {
    let explicit = mime_type.filter(|mime| *mime != "application/octet-stream");
    crate::core::mime::detect_or_validate_bytes(bytes, filename, explicit, config.mime_detection_policy)
}

fn file_uri_to_path(uri: &str) -> Result<PathBuf> {
    let parsed = url::Url::parse(uri)
        .map_err(|error| XbergError::validation(format!("invalid file URI for extraction input: {error}")))?;
    if parsed.scheme() != "file" {
        return Err(XbergError::UnsupportedFormat(format!(
            "unsupported URI scheme for local file extraction: {}",
            parsed.scheme()
        )));
    }

    if let Some(host) = parsed.host_str().filter(|host| !host.is_empty())
        && !host.eq_ignore_ascii_case("localhost")
    {
        return Err(XbergError::UnsupportedFormat(format!(
            "unsupported non-local file URI host: {host}"
        )));
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        parsed
            .to_file_path()
            .map_err(|()| XbergError::UnsupportedFormat(format!("unsupported file URI path: {uri}")))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = parsed;
        Err(XbergError::UnsupportedFormat(format!(
            "file URIs are not supported on this platform: {uri}"
        )))
    }
}

fn annotate_source(result: &mut ExtractedDocument, source_kind: &str, source_uri: &str, final_uri: &str, index: usize) {
    result
        .metadata
        .additional
        .insert("source_kind".into(), serde_json::json!(source_kind));
    result
        .metadata
        .additional
        .insert("source_uri".into(), serde_json::json!(source_uri));
    result
        .metadata
        .additional
        .insert("final_uri".into(), serde_json::json!(final_uri));
    result
        .metadata
        .additional
        .insert("source_index".into(), serde_json::json!(index));
}

fn input_source(input: &ExtractInput) -> String {
    match input.kind {
        ExtractInputKind::Bytes => input.filename.clone().unwrap_or_else(|| "<bytes>".to_string()),
        ExtractInputKind::Uri => input.uri.clone().unwrap_or_else(|| "<uri>".to_string()),
    }
}

async fn extract_remote_uri(uri: &str, _config: &ExtractionConfig, _index: usize) -> Result<ExtractionResult> {
    Err(XbergError::UnsupportedFormat(format!(
        "HTTP(S) URI extraction requires the 'url-ingestion' feature: {uri}"
    )))
}

fn merge_crawl_summary(
    output: &mut ExtractionResult,
    final_urls: Vec<String>,
    redirect_count: usize,
    unique_normalized_urls: Vec<String>,
) {
    if final_urls.is_empty() && redirect_count == 0 && unique_normalized_urls.is_empty() {
        return;
    }

    output.crawl_redirect_count += redirect_count;
    for url in final_urls {
        if !output.crawl_final_urls.contains(&url) {
            output.crawl_final_urls.push(url);
        }
    }
    for url in unique_normalized_urls {
        if !output.crawl_unique_normalized_urls.contains(&url) {
            output.crawl_unique_normalized_urls.push(url);
        }
    }
}

async fn follow_recursive_document_urls(
    output: &mut ExtractionResult,
    config: &ExtractionConfig,
    seen: &mut HashSet<String>,
    seed_hosts: &HashSet<String>,
) -> Result<()> {
    if !follow_document_urls(config) {
        return Ok(());
    }

    let max_depth = document_url_depth(config);
    if max_depth == 0 {
        return Ok(());
    }

    let pattern = config
        .url
        .document_url_pattern
        .as_deref()
        .map(regex::Regex::new)
        .transpose()
        .map_err(|error| XbergError::validation(format!("invalid document_url_pattern regex: {error}")))?;

    let mut queue = VecDeque::new();
    enqueue_discovered_urls(output, config, &pattern, seed_hosts, seen, &mut queue, 1);

    while let Some((uri, depth)) = queue.pop_front() {
        let index = output.summary.inputs;
        output.summary.inputs += 1;

        match Box::pin(extract_one(ExtractInput::from_uri(uri.clone()), config, index)).await {
            Ok(mut item_output) => {
                if depth < max_depth {
                    enqueue_discovered_urls(&item_output, config, &pattern, seed_hosts, seen, &mut queue, depth + 1);
                }
                output.summary.remote_urls += item_output.summary.remote_urls;
                output.summary.pages_crawled += item_output.summary.pages_crawled;
                output.summary.documents_downloaded += item_output.summary.documents_downloaded;
                output.results.append(&mut item_output.results);
                output.errors.append(&mut item_output.errors);
            }
            Err(error) => output.errors.push(error_item(index, uri, &error)),
        }
    }

    output.refresh_counts();
    Ok(())
}

fn enqueue_discovered_urls(
    output: &ExtractionResult,
    config: &ExtractionConfig,
    pattern: &Option<regex::Regex>,
    seed_hosts: &HashSet<String>,
    seen: &mut HashSet<String>,
    queue: &mut VecDeque<(String, u32)>,
    depth: u32,
) {
    let max_total = config.url.max_total_urls.unwrap_or(1_000) as usize;
    if seen.len() >= max_total {
        return;
    }

    for result in &output.results {
        let max_per_result = config.url.max_document_urls_per_result.unwrap_or(100) as usize;
        for url in urls_from_result(result).into_iter().take(max_per_result) {
            if seen.len() >= max_total {
                return;
            }
            if should_follow_discovered_url(&url, config, pattern, seed_hosts) && seen.insert(url.clone()) {
                queue.push_back((url, depth));
            }
        }
    }
}

fn urls_from_result(result: &ExtractedDocument) -> Vec<String> {
    let mut urls = Vec::new();
    if let Some(uris) = &result.uris {
        urls.extend(
            uris.iter()
                .filter(|uri| matches!(uri.kind, UriKind::Hyperlink | UriKind::Reference | UriKind::Citation))
                .map(|uri| uri.url.clone()),
        );
    }

    let text_regex = regex::Regex::new(r#"https?://[^\s<>"')\]]+"#).expect("static URL regex is valid");
    urls.extend(text_regex.find_iter(&result.content).map(|matched| {
        matched
            .as_str()
            .trim_end_matches(['.', ',', ';', ':', '!', '?'])
            .to_string()
    }));
    urls
}

fn should_follow_discovered_url(
    url: &str,
    config: &ExtractionConfig,
    pattern: &Option<regex::Regex>,
    seed_hosts: &HashSet<String>,
) -> bool {
    if !(url.starts_with(HTTP_SCHEME) || url.starts_with(HTTPS_SCHEME)) {
        return false;
    }
    if let Some(pattern) = pattern
        && !pattern.is_match(url)
    {
        return false;
    }
    if stay_on_domain(config)
        && !seed_hosts.is_empty()
        && let Some(host) = http_host(url)
    {
        return seed_hosts.contains(&host)
            || (allow_subdomains(config) && seed_hosts.iter().any(|seed| host.ends_with(&format!(".{seed}"))));
    }
    true
}

fn follow_document_urls(config: &ExtractionConfig) -> bool {
    {
        let _ = config;
        false
    }
}

fn document_url_depth(config: &ExtractionConfig) -> u32 {
    {
        let _ = config;
        0
    }
}

fn stay_on_domain(config: &ExtractionConfig) -> bool {
    {
        let _ = config;
        false
    }
}

fn allow_subdomains(config: &ExtractionConfig) -> bool {
    {
        let _ = config;
        false
    }
}

fn initial_seen_urls<'a>(inputs: impl IntoIterator<Item = &'a ExtractInput>) -> HashSet<String> {
    inputs
        .into_iter()
        .filter_map(|input| input.uri.clone())
        .filter(|uri| uri.starts_with(HTTP_SCHEME) || uri.starts_with(HTTPS_SCHEME))
        .collect()
}

fn initial_seed_hosts<'a>(inputs: impl IntoIterator<Item = &'a ExtractInput>) -> HashSet<String> {
    inputs
        .into_iter()
        .filter_map(|input| input.uri.as_deref().and_then(http_host))
        .collect()
}

fn http_host(uri: &str) -> Option<String> {
    let parsed = url::Url::parse(uri).ok()?;
    match parsed.scheme() {
        "http" | "https" => parsed.host_str().map(|host| host.to_ascii_lowercase()),
        _ => None,
    }
}

fn error_item(index: usize, source: String, error: &XbergError) -> ExtractionErrorItem {
    ExtractionErrorItem {
        index,
        code: error_code(error),
        error_type: error_type(error).to_string(),
        source,
        message: error.to_string(),
    }
}

fn error_code(error: &XbergError) -> u32 {
    error.extraction_error_code()
}

fn error_type(error: &XbergError) -> &'static str {
    error.extraction_error_type()
}

#[cfg(all(test, feature = "tokio-runtime"))]
mod tests;

#[cfg(test)]
mod input_options_tests;
