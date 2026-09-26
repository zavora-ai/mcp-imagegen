//! Error type (design §7). Every variant says what went wrong and what to do next (R7).

use std::fmt::Display;
use std::path::{Path, PathBuf};

use rmcp::ErrorData as McpError;

use crate::registry::MissingFile;

#[derive(Debug, thiserror::Error)]
pub enum ImageGenError {
    #[error("config error in {path}: {message}")]
    Config { path: PathBuf, message: String },

    #[error("invalid `{field}`: {message}")]
    InvalidRequest { field: String, message: String },

    #[error("unknown model `{id}`. Available: {}", available.join(", "))]
    UnknownModel { id: String, available: Vec<String> },

    #[error("model `{model}` is missing {} file(s). Fetch with: {}", files.len(), files.iter().map(|f| f.fetch_hint()).collect::<Vec<_>>().join(" && "))]
    MissingWeights {
        model: String,
        files: Vec<MissingFile>,
    },

    #[error("backend `{backend}` is unavailable: {reason}. Fix: {fix}")]
    BackendUnavailable {
        backend: String,
        reason: String,
        fix: String,
    },

    #[error(
        "not enough free memory to load `{model}`: needs ~{need_mb} MB plus headroom, {free_mb} MB free. {hint}"
    )]
    InsufficientMemory {
        model: String,
        need_mb: u64,
        free_mb: u64,
        hint: String,
    },

    #[error("output path {} is outside the allowed roots ({})", path.display(), roots.iter().map(|r| r.display().to_string()).collect::<Vec<_>>().join(", "))]
    OutputPathNotAllowed { path: PathBuf, roots: Vec<PathBuf> },

    #[error(
        "generation queue is full ({max} waiting). Retry later or cancel a job with cancel_job"
    )]
    QueueFull { max: usize },

    #[error("backend `{backend}` failed: {message}{}", if stderr_tail.is_empty() { String::new() } else { format!("\n--- backend log tail ---\n{stderr_tail}") })]
    BackendFailed {
        backend: String,
        message: String,
        stderr_tail: String,
    },

    #[error("job was cancelled")]
    Cancelled,

    #[error(
        "unknown job `{0}` (finished jobs expire after job_ttl_secs, and jobs don't survive a server restart)"
    )]
    UnknownJob(String),

    #[error("I/O error on {}: {message}", path.display())]
    Io { path: PathBuf, message: String },
}

impl ImageGenError {
    pub fn config(path: &Path, message: impl Display) -> Self {
        Self::Config {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }

    pub fn invalid(field: &str, message: impl Display) -> Self {
        Self::InvalidRequest {
            field: field.to_string(),
            message: message.to_string(),
        }
    }

    pub fn io(path: &Path, err: impl Display) -> Self {
        Self::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        }
    }

    /// Stable machine-readable code for structured results.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config { .. } => "config_error",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::UnknownModel { .. } => "unknown_model",
            Self::MissingWeights { .. } => "missing_weights",
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::InsufficientMemory { .. } => "insufficient_memory",
            Self::OutputPathNotAllowed { .. } => "output_path_not_allowed",
            Self::QueueFull { .. } => "queue_full",
            Self::BackendFailed { .. } => "backend_failed",
            Self::Cancelled => "cancelled",
            Self::UnknownJob(_) => "unknown_job",
            Self::Io { .. } => "io_error",
        }
    }

    /// Caller mistakes map to invalid_params; everything else is an internal error.
    pub fn is_caller_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidRequest { .. }
                | Self::UnknownModel { .. }
                | Self::OutputPathNotAllowed { .. }
                | Self::UnknownJob(_)
        )
    }
}

impl From<ImageGenError> for McpError {
    fn from(err: ImageGenError) -> Self {
        let data = Some(serde_json::json!({ "code": err.code() }));
        if err.is_caller_error() {
            McpError::invalid_params(err.to_string(), data)
        } else {
            McpError::internal_error(err.to_string(), data)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_weights_message_has_fetch_commands() {
        let err = ImageGenError::MissingWeights {
            model: "m".into(),
            files: vec![MissingFile {
                role: "vae".into(),
                hf_repo: Some("org/repo".into()),
                hf_file: Some("vae/ae.safetensors".into()),
                path: None,
                size_mb: Some(320),
            }],
        };
        let msg = err.to_string();
        assert!(
            msg.contains("hf download org/repo vae/ae.safetensors"),
            "{msg}"
        );
        assert_eq!(err.code(), "missing_weights");
    }

    #[test]
    fn caller_errors_map_to_invalid_params() {
        let mcp: McpError = ImageGenError::invalid("width", "must be a multiple of 16").into();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INVALID_PARAMS);
        assert!(mcp.message.contains("`width`"));
        let mcp: McpError = ImageGenError::QueueFull { max: 4 }.into();
        assert_eq!(mcp.code, rmcp::model::ErrorCode::INTERNAL_ERROR);
    }

    #[test]
    fn backend_failed_includes_log_tail_only_when_present() {
        let with = ImageGenError::BackendFailed {
            backend: "sdcpp".into(),
            message: "exited".into(),
            stderr_tail: "boom".into(),
        };
        assert!(with.to_string().contains("--- backend log tail ---\nboom"));
        let without = ImageGenError::BackendFailed {
            backend: "sdcpp".into(),
            message: "exited".into(),
            stderr_tail: String::new(),
        };
        assert!(!without.to_string().contains("log tail"));
    }
}
