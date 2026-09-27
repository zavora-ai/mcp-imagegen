//! MCP surface (design §5; R1, R2, R4, R5).

use std::sync::Arc;
use std::time::Duration;

use adk_mcp_sdk::{HealthCheck, HealthStatus};
use base64::Engine as _;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{
    ErrorData as McpError, handler::server::wrapper::Parameters, schemars, tool, tool_router,
};
use serde::Serialize;
use serde_json::{Value, json};

use crate::backend::{Availability, BackendRegistry, ImageBackend, Runs};
use crate::config::Config;
use crate::error::ImageGenError;
use crate::input::InputPolicy;
use crate::jobs::{JobConfig, JobManager, JobRequest, JobSnapshot, JobStatus};
use crate::memory::{self, MemoryProbe};
use crate::output::OutputPolicy;
use crate::registry::{MissingFile, ModelRegistry, ModelSpec};
use crate::request::{EditRequest, GenerateRequest, Mode, ResolvedRequest};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct JobIdInput {
    /// Job id returned by generate_image
    pub job_id: String,
}

#[derive(Debug, Default, serde::Deserialize, schemars::JsonSchema)]
pub struct ListModelsInput {
    /// Only list models for this backend (e.g. "sdcpp", "mflux")
    #[serde(default)]
    pub backend: Option<String>,
}

struct State {
    cfg: Config,
    registry: ModelRegistry,
    jobs: JobManager,
    output: OutputPolicy,
    inputs: InputPolicy,
    memory: Arc<dyn MemoryProbe>,
}

#[derive(Clone)]
pub struct ImageGenServer {
    state: Arc<State>,
}

/// Backends compiled into this build.
pub fn default_backends(cfg: &Config) -> BackendRegistry {
    #[allow(unused_mut)]
    let mut backends = BackendRegistry::default();
    #[cfg(feature = "sdcpp")]
    backends.insert(Arc::new(crate::backend::sdcpp::SdcppBackend::new(
        cfg.sdcpp.clone(),
    )));
    #[cfg(feature = "codex")]
    backends.insert(Arc::new(crate::backend::codex::CodexBackend::new(
        cfg.codex.clone(),
    )));
    #[cfg(feature = "mock")]
    {
        let step_ms = std::env::var("MCP_IMAGEGEN_MOCK_STEP_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        backends.insert(Arc::new(crate::backend::mock::MockBackend::new(
            Duration::from_millis(step_ms),
        )));
        backends.insert(Arc::new(crate::backend::mock::MockBackend::cloud(
            Duration::from_millis(step_ms),
        )));
    }
    let _ = cfg;
    backends
}

impl ImageGenServer {
    /// Must be called inside a tokio runtime (spawns the job worker).
    pub fn new(
        cfg: Config,
        registry: ModelRegistry,
        backends: BackendRegistry,
        memory: Arc<dyn MemoryProbe>,
    ) -> Self {
        let jobs = JobManager::new(
            backends,
            JobConfig {
                max_queue: cfg.max_queue,
                job_ttl: Duration::from_secs(cfg.job_ttl_secs),
                idle_unload: Duration::from_secs(cfg.idle_unload_secs),
            },
        );
        let output = OutputPolicy {
            default_dir: cfg.default_output_dir.clone(),
            allowed_roots: cfg.allowed_output_roots.clone(),
        };
        let inputs = InputPolicy {
            base_dir: cfg.default_output_dir.clone(),
            allowed_roots: cfg.allowed_input_roots.clone(),
            max_bytes: cfg.max_input_bytes,
            max_side: cfg.max_input_side,
        };
        Self {
            state: Arc::new(State {
                cfg,
                registry,
                jobs,
                output,
                inputs,
                memory,
            }),
        }
    }

    /// Load config and registry from the default location and build every compiled backend.
    pub fn from_default_config() -> Result<Self, ImageGenError> {
        let cfg = Config::load_or_init(&Config::default_dir())?;
        let registry = ModelRegistry::load(&cfg.models_path())?;
        let backends = default_backends(&cfg);
        Ok(Self::new(
            cfg,
            registry,
            backends,
            Arc::new(memory::SystemMemory::default()),
        ))
    }

    pub fn config(&self) -> &Config {
        &self.state.cfg
    }

    /// Stop every backend process, even mid-job. Used on SIGTERM/SIGINT and stdin close.
    pub async fn shutdown(&self) {
        for backend in self.state.jobs.backends().all() {
            backend.unload().await;
        }
    }

    /// Validate, plan output, check weights/backend/memory, and queue a generation.
    async fn submit(&self, input: &GenerateRequest) -> Result<JobSnapshot, ImageGenError> {
        let (request, model) = input.resolve(&self.state.registry)?;
        self.enqueue(request, model, input).await
    }

    /// Same pipeline for an edit: inputs are loaded and fingerprinted during resolution.
    async fn submit_edit(&self, input: &EditRequest) -> Result<JobSnapshot, ImageGenError> {
        let (request, model) = input.resolve(&self.state.registry, &self.state.inputs)?;
        self.enqueue(request, model, &input.params).await
    }

    async fn enqueue(
        &self,
        request: ResolvedRequest,
        model: &ModelSpec,
        opts: &GenerateRequest,
    ) -> Result<JobSnapshot, ImageGenError> {
        let st = &self.state;
        let tag = (request.mode == Mode::Edit).then_some("edit");
        let output = st.output.plan_tagged(
            opts.output_path.as_deref(),
            opts.output_dir.as_deref(),
            &request.prompt,
            tag,
            request.seed,
        )?;
        let missing = model.missing_for_generation(&st.cfg.hf_cache_dir);
        if !missing.is_empty() {
            return Err(ImageGenError::MissingWeights {
                model: model.id.clone(),
                files: missing,
            });
        }
        let mut files = model
            .resolve_files(&st.cfg.hf_cache_dir)
            .unwrap_or_default();
        let mut request = request;
        request.loras = model
            .resolve_loras(&st.cfg.hf_cache_dir)
            .unwrap_or_default();
        if request.mode == Mode::Edit {
            model
                .resolve_edit_files(&st.cfg.hf_cache_dir)
                .map_err(|files| ImageGenError::MissingWeights {
                    model: model.id.clone(),
                    files,
                })?;
        }
        // Load edit-only weights whenever present so one backend process serves both modes.
        files.extend(model.present_edit_files(&st.cfg.hf_cache_dir));
        let backend = st.jobs.backends().get(&model.backend).ok_or_else(|| {
            ImageGenError::BackendUnavailable {
                backend: model.backend.clone(),
                reason: "not compiled into this build".into(),
                fix: format!("rebuild with `--features {}`", model.backend),
            }
        })?;
        if let Availability::Unavailable { reason, fix } = backend.availability().await {
            return Err(ImageGenError::BackendUnavailable {
                backend: model.backend.clone(),
                reason,
                fix,
            });
        }
        // Cloud backends use no local memory (R15).
        if backend.runs() == Runs::Local {
            let already_loaded = backend.loaded_model().as_deref() == Some(model.id.as_str());
            let reclaimable_mb = st
                .jobs
                .backends()
                .all()
                .filter_map(|b| b.loaded_model())
                .filter(|id| *id != model.id)
                .filter_map(|id| st.registry.get(&id).ok().map(|m| m.est_memory_mb))
                .sum();
            memory::preflight(
                model,
                already_loaded,
                reclaimable_mb,
                st.cfg.memory_headroom_mb,
                st.memory.as_ref(),
            )?;
        }
        st.jobs.submit(JobRequest {
            model: model.clone(),
            files,
            request,
            output,
            want_preview: opts.return_preview,
        })
    }

    /// Shared tail of generate_image / edit_image: return a job id, or wait and build the result.
    async fn finish(&self, snap: JobSnapshot, wait: bool) -> Result<CallToolResult, McpError> {
        if !wait {
            return Ok(structured(json!({
                "job_id": snap.job_id,
                "status": snap.status,
                "progress": snap.progress,
                "hint": "poll get_job with this job_id; cancel_job stops it",
            })));
        }
        // If the call itself is cancelled (tasks/cancel, request cancellation), stop the job too.
        let guard = CancelOnDrop {
            jobs: self.state.jobs.clone(),
            job_id: Some(snap.job_id.clone()),
        };
        let done = self.wait_with_progress(&snap.job_id).await?;
        guard.disarm();

        match (&done.status, &done.result) {
            (JobStatus::Done, Some(result)) => {
                let mut value = serde_json::to_value(result).unwrap_or(Value::Null);
                value["status"] = json!("done");
                let mut out = CallToolResult::structured(value);
                if let Some(preview) = &result.preview_png {
                    out.content.push(ContentBlock::image(
                        base64::engine::general_purpose::STANDARD.encode(preview),
                        "image/png",
                    ));
                }
                Ok(out)
            }
            _ => Ok(job_failure(&done)),
        }
    }

    /// Wait for a job, forwarding progress to the MCP task status when running as a Task.
    async fn wait_with_progress(&self, job_id: &str) -> Result<JobSnapshot, ImageGenError> {
        let mut rx = self.state.jobs.subscribe(job_id)?;
        let mut last = String::new();
        loop {
            let snap = rx.borrow_and_update().clone();
            if let Some(p) = &snap.progress {
                let text = p.to_string();
                if text != last {
                    adk_mcp_sdk::set_current_task_status(text.clone());
                    last = text;
                }
            }
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(ImageGenError::UnknownJob(job_id.to_string()));
            }
        }
    }

    fn model_entry(
        &self,
        model: &ModelSpec,
        backend: Option<&dyn ImageBackend>,
        availability: &Availability,
        loaded: bool,
    ) -> Value {
        let missing = model.missing_for_generation(&self.state.cfg.hf_cache_dir);
        let status = if !missing.is_empty() {
            "missing_files"
        } else if matches!(availability, Availability::Unavailable { .. }) {
            "backend_unavailable"
        } else {
            "ready"
        };
        let runs = backend.map(|b| b.runs()).unwrap_or_default();
        let mut entry = json!({
            "id": model.id,
            "backend": model.backend,
            "runs": runs,
            "status": status,
            "loaded": loaded,
            "license": model.license,
            "capabilities": model.capabilities,
            "defaults": model.defaults,
            "size_multiple": model.size_multiple,
            "max_pixels": model.max_pixels,
            "est_memory_mb": model.est_memory_mb,
            "fixed_steps": model.sigma_schedule.as_ref().map(|s| s.steps()),
            "loras": model
                .loras
                .iter()
                .map(|l| l.file.hf_file.clone().or_else(|| l.file.path.clone()))
                .collect::<Vec<_>>(),
        });
        if runs == Runs::Local {
            entry["commercial_weights"] = json!(model.commercial_weights);
        }
        if let Some(outputs) = model.commercial_outputs {
            entry["commercial_outputs"] = json!(outputs);
        }
        if let Some(provider) = backend.and_then(|b| b.provider()) {
            entry["provider"] = json!(provider);
            entry["privacy"] = json!(format!("prompts and input images are sent to {provider}"));
        }
        if let Some(codex) = &model.codex {
            entry["agent_model"] = json!(codex.model);
        }
        if !missing.is_empty() {
            entry["missing_files"] = json!(missing.iter().map(missing_json).collect::<Vec<_>>());
            let total: u64 = missing.iter().filter_map(|m| m.size_mb).sum();
            if total > 0 {
                entry["download_mb"] = json!(total);
            }
        }
        if let Availability::Unavailable { reason, fix } = availability {
            entry["backend_problem"] = json!({ "reason": reason, "fix": fix });
        }
        entry["edit"] = if !model.supports(crate::registry::Capability::Edit) {
            json!({ "status": "unsupported" })
        } else {
            match model.resolve_edit_files(&self.state.cfg.hf_cache_dir) {
                Ok(_) => json!({ "status": "ready", "max_images": model.max_ref_images }),
                Err(files) => json!({
                    "status": "missing_files",
                    "max_images": model.max_ref_images,
                    "missing_files": files.iter().map(missing_json).collect::<Vec<_>>(),
                    "download_mb": files.iter().filter_map(|m| m.size_mb).sum::<u64>(),
                }),
            }
        };
        entry
    }
}

fn missing_json(m: &MissingFile) -> Value {
    json!({ "role": m.role, "size_mb": m.size_mb, "fetch": m.fetch_hint() })
}

fn structured(value: impl Serialize) -> CallToolResult {
    CallToolResult::structured(serde_json::to_value(value).unwrap_or(Value::Null))
}

/// Tool-level failure (the call worked, the generation didn't).
fn job_failure(snap: &JobSnapshot) -> CallToolResult {
    CallToolResult::structured_error(json!({
        "job_id": snap.job_id,
        "status": snap.status,
        "error": snap.error,
    }))
}

#[tool_router]
impl ImageGenServer {
    #[tool(
        description = "Generate an image from a text prompt and write it as PNG, with a local model or a cloud one (list_models shows `runs`; cloud models send the prompt to their provider). Returns the file path, seed (null for cloud models) and all parameters. Large local models can take minutes: pass wait=false to get a job_id and poll get_job."
    )]
    async fn generate_image(
        &self,
        Parameters(input): Parameters<GenerateRequest>,
    ) -> Result<CallToolResult, McpError> {
        let snap = self.submit(&input).await?;
        self.finish(snap, input.wait).await
    }

    #[tool(
        description = "Edit existing images from an instruction (e.g. 'paint the crate red, keep the iron brackets'). `images` are 1-N local paths in order (the first sets the default output size); `prompt` is the instruction; all generate_image options apply. Writes a new PNG and never modifies the inputs."
    )]
    async fn edit_image(
        &self,
        Parameters(input): Parameters<EditRequest>,
    ) -> Result<CallToolResult, McpError> {
        let snap = self.submit_edit(&input).await?;
        self.finish(snap, input.params.wait).await
    }

    #[tool(description = "Get status, progress and result of a generation job")]
    async fn get_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        let snap = self.state.jobs.get(&input.job_id)?;
        Ok(if snap.status == JobStatus::Failed {
            job_failure(&snap)
        } else {
            structured(&snap)
        })
    }

    #[tool(
        description = "Cancel a queued or running generation job. No partial files are left behind."
    )]
    async fn cancel_job(
        &self,
        Parameters(input): Parameters<JobIdInput>,
    ) -> Result<CallToolResult, McpError> {
        Ok(structured(self.state.jobs.cancel(&input.job_id)?))
    }

    #[tool(
        description = "List configured models with backend, where they run (local / cloud, with the provider for cloud), readiness (ready / missing_files with download commands / backend_unavailable), license and commercial-use flags. Never loads a model or spends cloud quota."
    )]
    async fn list_models(
        &self,
        Parameters(input): Parameters<ListModelsInput>,
    ) -> Result<CallToolResult, McpError> {
        let st = &self.state;
        let mut models = Vec::new();
        for model in st.registry.models() {
            if input.backend.as_deref().is_some_and(|b| b != model.backend) {
                continue;
            }
            let backend = st.jobs.backends().get(&model.backend);
            let (availability, loaded) = match &backend {
                Some(b) => (
                    b.availability().await,
                    b.loaded_model().as_deref() == Some(model.id.as_str()),
                ),
                None => (
                    Availability::Unavailable {
                        reason: "backend not compiled into this build".into(),
                        fix: format!("rebuild with `--features {}`", model.backend),
                    },
                    false,
                ),
            };
            models.push(self.model_entry(model, backend.as_deref(), &availability, loaded));
        }
        Ok(structured(json!({
            "models": models,
            "default_model": st.registry.default_model().map(|m| m.id.clone()),
            "available_memory_mb": st.memory.available_mb(),
            "default_output_dir": st.cfg.default_output_dir,
            "allowed_output_roots": st.cfg.allowed_output_roots,
            "config_dir": st.cfg.config_dir,
            "jobs_busy": st.jobs.busy(),
        })))
    }

    #[tool(
        description = "Unload all loaded local models and stop backend processes to free memory. Refused while local jobs are queued or running; cloud jobs don't block it."
    )]
    async fn unload_models(&self) -> Result<CallToolResult, McpError> {
        let unloaded = self.state.jobs.unload_all().await?;
        Ok(structured(json!({
            "unloaded": unloaded,
            "available_memory_mb": self.state.memory.available_mb(),
        })))
    }
}

struct CancelOnDrop {
    jobs: JobManager,
    job_id: Option<String>,
}

impl CancelOnDrop {
    fn disarm(mut self) {
        self.job_id = None;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(id) = self.job_id.take() {
            tracing::info!(job_id = %id, "request dropped before completion; cancelling job");
            let _ = self.jobs.cancel(&id);
        }
    }
}

adk_mcp_sdk::mcp_2026_server! {
    server: ImageGenServer,
    task_tools: ["generate_image", "edit_image"],
    task_ttl_overrides: [("generate_image", 3_600_000), ("edit_image", 3_600_000)],
    approval_tools: [],
    mutating_tools: ["generate_image", "edit_image", "cancel_job", "unload_models"],
    destructive_tools: [],
    idempotent_tools: ["cancel_job", "unload_models"],
    cache_ttl_ms: 60_000,
    instructions: "Image generation with local models and optional cloud models. Call list_models first to see which models are ready, \
where each runs (cloud models send prompts and input images to their provider) and what their licences allow. \
generate_image and edit_image can take minutes on large local models; pass wait=false and poll get_job if your client times out. \
One local and one cloud generation can run at a time; call unload_models to free memory for other apps.",
}

#[async_trait::async_trait]
impl HealthCheck for ImageGenServer {
    async fn check_health(&self) -> HealthStatus {
        let started = std::time::Instant::now();
        let mut parts = Vec::new();
        for id in self.state.jobs.backends().ids() {
            if let Some(b) = self.state.jobs.backends().get(&id) {
                let state = match b.availability().await {
                    Availability::Ready => "ready".to_string(),
                    Availability::Unavailable { reason, .. } => format!("unavailable ({reason})"),
                };
                parts.push(format!("{id}: {state}"));
            }
        }
        HealthStatus {
            // A missing engine is reported per model by list_models; the server itself is fine.
            healthy: true,
            message: Some(format!(
                "{} model(s); backends: {}",
                self.state.registry.models().len(),
                parts.join(", ")
            )),
            latency_ms: Some(started.elapsed().as_millis() as u64),
        }
    }
}
