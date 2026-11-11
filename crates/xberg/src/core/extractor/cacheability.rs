//! Private cache retention state shared only within one extraction operation (ADR 0008).

use std::cell::RefCell;
#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::collections::HashMap;
use std::future::Future;
#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
const MAX_MODEL_ROOTS: usize = 64;

#[derive(Clone, Default)]
pub(crate) struct Cacheability(Arc<Operation>);

#[derive(Default)]
struct Operation {
    invalidated: AtomicBool,
    #[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
    revisions: Mutex<HashMap<PathBuf, blake3::Hash>>,
}

#[cfg(feature = "tokio-runtime")]
tokio::task_local! {
    static ASYNC_CONTEXT: Cacheability;
}

thread_local! {
    static BLOCKING_CONTEXT: RefCell<Option<Cacheability>> = const { RefCell::new(None) };
}

impl Cacheability {
    pub(crate) fn is_cacheable(&self) -> bool {
        !self.0.invalidated.load(Ordering::Acquire)
    }

    #[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
    pub(crate) fn invalidate(&self) {
        self.0.invalidated.store(true, Ordering::Release);
    }

    #[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
    fn observe(&self, root: PathBuf, revision: blake3::Hash) {
        let Ok(mut revisions) = self.0.revisions.lock() else {
            self.invalidate();
            tracing::warn!("Model observation state was poisoned; bypassing operation caches");
            return;
        };
        match revisions.get(&root) {
            Some(previous) if *previous != revision => self.invalidate(),
            Some(_) => {}
            None if revisions.len() >= MAX_MODEL_ROOTS => {
                self.invalidate();
                tracing::warn!("Model observation bound reached; bypassing operation caches");
            }
            None => {
                revisions.insert(root, revision);
            }
        }
    }
}

pub(crate) fn current() -> Option<Cacheability> {
    let blocking = BLOCKING_CONTEXT.with(|context| context.borrow().clone());
    #[cfg(feature = "tokio-runtime")]
    {
        blocking.or_else(|| ASYNC_CONTEXT.try_with(Clone::clone).ok())
    }
    #[cfg(not(feature = "tokio-runtime"))]
    {
        blocking
    }
}

pub(crate) fn is_cacheable() -> bool {
    current().is_none_or(|context| context.is_cacheable())
}

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
pub(crate) fn invalidate_current() {
    if let Some(context) = current() {
        context.invalidate();
    }
}

pub(crate) async fn scope<F: Future>(future: F) -> F::Output {
    #[cfg(feature = "tokio-runtime")]
    {
        ASYNC_CONTEXT.scope(current().unwrap_or_default(), future).await
    }
    #[cfg(not(feature = "tokio-runtime"))]
    {
        future.await
    }
}

#[cfg(all(feature = "tokio-runtime", not(target_arch = "wasm32")))]
pub(crate) fn inherit<F: Future>(future: F) -> impl Future<Output = F::Output> {
    use tracing::instrument::WithSubscriber;

    let context = current();
    // A spawned task runs on a worker with no thread-local dispatcher, so a caller scoped with
    // `with_default` / `set_default` would lose the spans of every spawned batch or OCR task.
    async move {
        match context {
            Some(context) => ASYNC_CONTEXT.scope(context, future).await,
            None => future.await,
        }
    }
    .with_current_subscriber()
}

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
struct BlockingScope(Option<Cacheability>);

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
impl Drop for BlockingScope {
    fn drop(&mut self) {
        BLOCKING_CONTEXT.with(|context| {
            context.replace(self.0.take());
        });
    }
}

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
pub(crate) fn blocking<F: FnOnce() -> T, T>(operation: F) -> impl FnOnce() -> T {
    let inherited = current();
    move || {
        let _restore = BlockingScope(BLOCKING_CONTEXT.with(|context| context.replace(inherited)));
        operation()
    }
}

#[cfg(all(test, feature = "tokio-runtime", not(target_arch = "wasm32")))]
#[path = "cacheability/tests.rs"]
mod tests;
