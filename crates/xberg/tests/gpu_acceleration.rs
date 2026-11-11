//! GPU acceleration integration tests for all ORT-backed subsystems.
//!
//! Covers every code path that uses AccelerationConfig → apply_execution_providers:
//! 1. PaddleOCR (det + cls + rec) — feature: paddle-ocr
//! 2. Layout detection (RT-DETR) — feature: layout-detection
//! 3. Embeddings (ONNX models) — feature: embeddings
//! 4. Document orientation (auto-rotate via paddle-ocr)
//! 5. End-to-end extraction with CUDA acceleration
//!
//! All tests are `#[ignore]` and require:
//! - NVIDIA GPU with CUDA
//! - ONNX Runtime built with CUDA EP
//! - Network access (models auto-downloaded from HuggingFace)
//!
//! Run all GPU tests:
//!   cargo test -p xberg --features "full,ort-dynamic,cuda" --test gpu_acceleration -- --ignored
//!
//! The `ort-dynamic` feature overrides the bundled CPU-only ORT so a
//! GPU-enabled ONNX Runtime (via `ORT_DYLIB_PATH`) is loaded at runtime. The `cuda`
//! feature compiles in `ort::ep::CUDA` (gated out of `full` by default — see
//! `ort_discovery.rs` — since the plain `download-binaries` prebuilt has no CUDA support).

#![allow(clippy::print_stdout, clippy::print_stderr, clippy::dbg_macro)] // ~keep: test/bench binaries print by design; org logging policy exempts tests
#![allow(dead_code)]

mod helpers;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn test_documents_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TEST_DOCUMENTS_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test_documents")
}

fn cuda_accel() -> xberg::AccelerationConfig {
    xberg::AccelerationConfig {
        provider: xberg::ExecutionProviderType::Cuda,
        device_id: 0,
    }
}

struct LogCapture {
    messages: Arc<Mutex<Vec<String>>>,
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for LogCapture {
    fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
        let mut visitor = MessageVisitor(String::new());
        event.record(&mut visitor);
        if let Ok(mut msgs) = self.messages.lock() {
            msgs.push(visitor.0);
        }
    }
}

struct MessageVisitor(String);

impl tracing::field::Visit for MessageVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0 = format!("{:?}", value);
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "message" {
            self.0 = value.to_string();
        }
    }
}

fn setup_log_capture() -> (Arc<Mutex<Vec<String>>>, tracing::subscriber::DefaultGuard) {
    use tracing_subscriber::layer::SubscriberExt;

    let captured = Arc::new(Mutex::new(Vec::<String>::new()));
    let layer = LogCapture {
        messages: Arc::clone(&captured),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let guard = tracing::subscriber::set_default(subscriber);
    (captured, guard)
}

fn assert_cuda_requested(captured: &Arc<Mutex<Vec<String>>>) {
    let logs = captured.lock().unwrap();
    let cuda_active = logs
        .iter()
        .any(|msg| msg.contains("CUDA execution provider available") || msg.contains("CUDA available, using GPU"));
    assert!(
        cuda_active,
        "CUDA EP was NOT activated — AccelerationConfig not propagated to ORT session. \
         Captured logs:\n{}",
        logs.iter()
            .filter(|m| !m.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
