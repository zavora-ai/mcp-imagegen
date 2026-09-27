//! Image backends behind one trait (design §4.1; R3).

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tokio_util::sync::CancellationToken;

use crate::error::ImageGenError;
use crate::registry::ModelSpec;
use crate::request::ResolvedRequest;

#[cfg(feature = "codex")]
pub mod codex;
#[cfg(feature = "mflux")]
pub mod mflux;
#[cfg(any(test, feature = "mock"))]
pub mod mock;
pub mod process;
#[cfg(feature = "sdcpp")]
pub mod sdcpp;

/// Where a backend does its work. Cloud backends skip the memory pre-flight and run in their own job lane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runs {
    #[default]
    Local,
    Cloud,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum Availability {
    Ready,
    Unavailable { reason: String, fix: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Queued {
        position: usize,
    },
    /// A subprocess backend is starting its tool (e.g. `codex`).
    Starting {
        backend: String,
    },
    Loading,
    Encoding,
    Sampling {
        step: u32,
        total: u32,
    },
    Decoding,
    /// Waiting on a backend that reports no step counts (e.g. a cloud tool).
    Generating,
    Saving,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Queued { position } => write!(f, "queued (position {position})"),
            Self::Starting { backend } => write!(f, "starting {backend}"),
            Self::Loading => f.write_str("loading model"),
            Self::Encoding => f.write_str("encoding prompt"),
            Self::Sampling { step, total } => write!(f, "sampling {step}/{total}"),
            Self::Decoding => f.write_str("decoding"),
            Self::Generating => f.write_str("generating"),
            Self::Saving => f.write_str("saving"),
        }
    }
}

/// Cheap clonable progress callback. Backends call it; the job manager records it.
#[derive(Clone)]
pub struct ProgressSink(Arc<dyn Fn(Phase) + Send + Sync>);

impl ProgressSink {
    pub fn new(f: impl Fn(Phase) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }

    pub fn noop() -> Self {
        Self::new(|_| {})
    }

    pub fn report(&self, phase: Phase) {
        (self.0)(phase)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Timings {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<u64>,
}

/// What a backend hands back. Backends never write files themselves.
#[derive(Debug, Clone, Default)]
pub struct GeneratedImage {
    pub png: Vec<u8>,
    /// Seed the backend actually used; `None` when the backend has no seeds (e.g. Codex).
    pub seed: Option<u64>,
    pub timings: Timings,
    /// Things the caller should know, appended to the result's warnings.
    pub warnings: Vec<String>,
    /// Backend-specific provenance recorded in the sidecar (`backend_details`).
    pub details: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    Cancelled,
    Unavailable {
        reason: String,
        fix: String,
    },
    Failed {
        message: String,
        stderr_tail: String,
    },
}

impl BackendError {
    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed {
            message: message.into(),
            stderr_tail: String::new(),
        }
    }

    pub fn into_error(self, backend: &str) -> ImageGenError {
        match self {
            Self::Cancelled => ImageGenError::Cancelled,
            Self::Unavailable { reason, fix } => ImageGenError::BackendUnavailable {
                backend: backend.to_string(),
                reason,
                fix,
            },
            Self::Failed {
                message,
                stderr_tail,
            } => ImageGenError::BackendFailed {
                backend: backend.to_string(),
                message,
                stderr_tail,
            },
        }
    }
}

#[async_trait::async_trait]
pub trait ImageBackend: Send + Sync {
    fn id(&self) -> &'static str;

    /// Cheap check: binary present / service reachable. Never loads weights.
    async fn availability(&self) -> Availability;

    /// Run one generation. Must honour `cancel` and report progress.
    async fn generate(
        &self,
        model: &ModelSpec,
        files: &BTreeMap<String, PathBuf>,
        request: &ResolvedRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedImage, BackendError>;

    /// Free memory now. Idempotent.
    async fn unload(&self);

    /// Id of the model currently resident, if any.
    fn loaded_model(&self) -> Option<String>;

    /// Local machine or a cloud service.
    fn runs(&self) -> Runs {
        Runs::Local
    }

    /// Who receives prompts and images, for cloud backends.
    fn provider(&self) -> Option<&'static str> {
        None
    }
}

/// Backends keyed by id.
#[derive(Clone, Default)]
pub struct BackendRegistry {
    backends: HashMap<String, Arc<dyn ImageBackend>>,
}

impl BackendRegistry {
    pub fn insert(&mut self, backend: Arc<dyn ImageBackend>) {
        self.backends.insert(backend.id().to_string(), backend);
    }

    pub fn get(&self, id: &str) -> Option<Arc<dyn ImageBackend>> {
        self.backends.get(id).cloned()
    }

    pub fn all(&self) -> impl Iterator<Item = &Arc<dyn ImageBackend>> {
        self.backends.values()
    }

    pub fn ids(&self) -> Vec<String> {
        let mut ids: Vec<_> = self.backends.keys().cloned().collect();
        ids.sort();
        ids
    }
}
