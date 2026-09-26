//! Job manager (design §7 steps 4–6; R4, R5).
//!
//! One worker runs jobs strictly one at a time from a bounded FIFO queue. Every job has a
//! `watch` channel carrying its latest snapshot, so callers can wait, poll or stream progress.
//! Finished jobs are kept for `job_ttl`; backends are unloaded after `idle_unload` without work.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendError, BackendRegistry, Phase, ProgressSink, Timings};
use crate::error::ImageGenError;
use crate::output;
use crate::registry::ModelSpec;
use crate::request::ResolvedRequest;

#[derive(Debug, Clone)]
pub struct JobConfig {
    pub max_queue: usize,
    pub job_ttl: Duration,
    pub idle_unload: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Failed | Self::Cancelled)
    }
}

/// Parameters and provenance recorded next to every image (the sidecar JSON).
#[derive(Debug, Clone, Serialize)]
pub struct GenerationRecord {
    pub model: String,
    pub backend: String,
    pub license: String,
    pub commercial_weights: bool,
    pub prompt: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub negative_prompt: String,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    pub elapsed_ms: u64,
    pub timings: Timings,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_options: BTreeMap<String, serde_json::Value>,
    pub mode: crate::request::Mode,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<crate::input::Reference>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<crate::registry::LoraUse>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub custom_sigmas: Vec<f32>,
    pub generator: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobResult {
    pub job_id: String,
    pub path: PathBuf,
    pub sidecar_path: PathBuf,
    #[serde(flatten)]
    pub record: GenerationRecord,
    #[serde(skip)]
    pub preview_png: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct JobSnapshot {
    pub job_id: String,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress: Option<Phase>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JobError>,
}

/// A validated job ready to run.
#[derive(Debug, Clone)]
pub struct JobRequest {
    pub model: ModelSpec,
    pub files: BTreeMap<String, PathBuf>,
    pub request: ResolvedRequest,
    /// Desired image path (already jailed); collisions are resolved at write time.
    pub output: PathBuf,
    pub want_preview: bool,
}

struct JobEntry {
    tx: watch::Sender<JobSnapshot>,
    cancel: CancellationToken,
    request: Mutex<Option<JobRequest>>,
    finished_at: Mutex<Option<Instant>>,
}

impl JobEntry {
    fn snapshot(&self) -> JobSnapshot {
        self.tx.borrow().clone()
    }

    fn finish(&self, status: JobStatus, result: Option<JobResult>, error: Option<JobError>) {
        self.tx.send_modify(|s| {
            s.status = status;
            s.result = result;
            s.error = error;
            if status != JobStatus::Done {
                s.progress = None;
            }
        });
        *self.finished_at.lock().unwrap() = Some(Instant::now());
    }
}

struct Inner {
    jobs: Mutex<HashMap<String, Arc<JobEntry>>>,
    queue: Mutex<VecDeque<String>>,
    running: Mutex<Option<String>>,
    wake: Notify,
    backends: BackendRegistry,
    cfg: JobConfig,
}

#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Inner>,
}

impl JobManager {
    /// Create the manager and spawn its worker on the current tokio runtime.
    pub fn new(backends: BackendRegistry, cfg: JobConfig) -> Self {
        let inner = Arc::new(Inner {
            jobs: Mutex::new(HashMap::new()),
            queue: Mutex::new(VecDeque::new()),
            running: Mutex::new(None),
            wake: Notify::new(),
            backends,
            cfg,
        });
        tokio::spawn(worker(Arc::downgrade(&inner)));
        Self { inner }
    }

    pub fn backends(&self) -> &BackendRegistry {
        &self.inner.backends
    }

    /// Queue a job. Fails with `QueueFull` when `max_queue` jobs are already waiting.
    pub fn submit(&self, job: JobRequest) -> Result<JobSnapshot, ImageGenError> {
        self.prune();
        let id = new_job_id();
        let mut queue = self.inner.queue.lock().unwrap();
        if queue.len() >= self.inner.cfg.max_queue {
            return Err(ImageGenError::QueueFull {
                max: self.inner.cfg.max_queue,
            });
        }
        let snapshot = JobSnapshot {
            job_id: id.clone(),
            status: JobStatus::Queued,
            progress: Some(Phase::Queued {
                position: queue.len() + 1,
            }),
            result: None,
            error: None,
        };
        let (tx, _) = watch::channel(snapshot.clone());
        let entry = Arc::new(JobEntry {
            tx,
            cancel: CancellationToken::new(),
            request: Mutex::new(Some(job)),
            finished_at: Mutex::new(None),
        });
        self.inner.jobs.lock().unwrap().insert(id.clone(), entry);
        queue.push_back(id);
        drop(queue);
        self.inner.wake.notify_one();
        Ok(snapshot)
    }

    pub fn get(&self, id: &str) -> Result<JobSnapshot, ImageGenError> {
        self.prune();
        Ok(self.entry(id)?.snapshot())
    }

    pub fn subscribe(&self, id: &str) -> Result<watch::Receiver<JobSnapshot>, ImageGenError> {
        Ok(self.entry(id)?.tx.subscribe())
    }

    /// Wait until the job reaches a terminal state.
    pub async fn wait(&self, id: &str) -> Result<JobSnapshot, ImageGenError> {
        let mut rx = self.subscribe(id)?;
        loop {
            let snap = rx.borrow_and_update().clone();
            if snap.status.is_terminal() {
                return Ok(snap);
            }
            if rx.changed().await.is_err() {
                return Err(ImageGenError::UnknownJob(id.to_string()));
            }
        }
    }

    /// Cancel a queued job immediately, or signal a running one. Idempotent on finished jobs.
    pub fn cancel(&self, id: &str) -> Result<JobSnapshot, ImageGenError> {
        let entry = self.entry(id)?;
        let removed = {
            let mut queue = self.inner.queue.lock().unwrap();
            let before = queue.len();
            queue.retain(|q| q != id);
            before != queue.len()
        };
        entry.cancel.cancel();
        if removed {
            entry.request.lock().unwrap().take();
            entry.finish(JobStatus::Cancelled, None, None);
            self.inner.renumber_queue();
        }
        Ok(entry.snapshot())
    }

    /// Jobs queued plus the one running.
    pub fn busy(&self) -> usize {
        self.inner.queue.lock().unwrap().len()
            + usize::from(self.inner.running.lock().unwrap().is_some())
    }

    /// Unload every backend now. Refused while a job is running or queued.
    pub async fn unload_all(&self) -> Result<Vec<String>, ImageGenError> {
        if self.busy() > 0 {
            return Err(ImageGenError::invalid(
                "unload_models",
                format!(
                    "{} job(s) queued or running; cancel them first or wait",
                    self.busy()
                ),
            ));
        }
        Ok(self.inner.unload_all().await)
    }

    fn entry(&self, id: &str) -> Result<Arc<JobEntry>, ImageGenError> {
        self.inner
            .jobs
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| ImageGenError::UnknownJob(id.to_string()))
    }

    fn prune(&self) {
        let ttl = self.inner.cfg.job_ttl;
        self.inner.jobs.lock().unwrap().retain(|_, e| {
            e.finished_at
                .lock()
                .unwrap()
                .is_none_or(|t| t.elapsed() < ttl)
        });
    }
}

impl Inner {
    fn renumber_queue(&self) {
        let queue = self.queue.lock().unwrap();
        let jobs = self.jobs.lock().unwrap();
        for (i, id) in queue.iter().enumerate() {
            if let Some(e) = jobs.get(id) {
                e.tx.send_modify(|s| s.progress = Some(Phase::Queued { position: i + 1 }));
            }
        }
    }

    async fn unload_all(&self) -> Vec<String> {
        let mut unloaded = Vec::new();
        for backend in self.backends.all() {
            if let Some(model) = backend.loaded_model() {
                backend.unload().await;
                unloaded.push(model);
            }
        }
        unloaded
    }

    fn next_job(&self) -> Option<(String, Arc<JobEntry>)> {
        let id = self.queue.lock().unwrap().pop_front()?;
        let entry = self.jobs.lock().unwrap().get(&id).cloned()?;
        *self.running.lock().unwrap() = Some(id.clone());
        self.renumber_queue();
        Some((id, entry))
    }

    async fn run(&self, id: &str, entry: &JobEntry) {
        let Some(job) = entry.request.lock().unwrap().take() else {
            return;
        };
        if entry.cancel.is_cancelled() {
            entry.finish(JobStatus::Cancelled, None, None);
            return;
        }
        entry.tx.send_modify(|s| {
            s.status = JobStatus::Running;
            s.progress = Some(Phase::Loading);
        });

        let backend_id = job.model.backend.clone();
        let Some(backend) = self.backends.get(&backend_id) else {
            let err = ImageGenError::BackendUnavailable {
                backend: backend_id.clone(),
                reason: "backend is not compiled in or not configured".into(),
                fix: format!("build with `--features {backend_id}` or fix models.toml"),
            };
            entry.finish(JobStatus::Failed, None, Some(to_job_error(&err)));
            return;
        };

        // Only one model resident at a time across backends (R5).
        for other in self.backends.all() {
            if other.id() != backend.id() && other.loaded_model().is_some() {
                other.unload().await;
            }
        }

        let tx = entry.tx.clone();
        let progress = ProgressSink::new(move |phase| {
            tx.send_modify(|s| s.progress = Some(phase));
        });
        let started = Instant::now();
        let outcome = backend
            .generate(
                &job.model,
                &job.files,
                &job.request,
                progress,
                entry.cancel.clone(),
            )
            .await;

        let image = match outcome {
            _ if entry.cancel.is_cancelled() => {
                entry.finish(JobStatus::Cancelled, None, None);
                return;
            }
            Err(BackendError::Cancelled) => {
                entry.finish(JobStatus::Cancelled, None, None);
                return;
            }
            Err(e) => {
                let err = e.into_error(&backend_id);
                entry.finish(JobStatus::Failed, None, Some(to_job_error(&err)));
                return;
            }
            Ok(image) => image,
        };

        entry.tx.send_modify(|s| s.progress = Some(Phase::Saving));
        let req = &job.request;
        let record = GenerationRecord {
            model: job.model.id.clone(),
            backend: backend_id.clone(),
            license: job.model.license.clone(),
            commercial_weights: job.model.commercial_weights,
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            seed: image.seed,
            width: req.width,
            height: req.height,
            steps: req.steps,
            cfg_scale: req.cfg_scale,
            sampler: req.sampler.clone(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            timings: image.timings.clone(),
            warnings: req.warnings.clone(),
            backend_options: job.model.backend_options.clone(),
            mode: req.mode,
            references: req.references.clone(),
            loras: req.loras.clone(),
            custom_sigmas: req.custom_sigmas.clone(),
            generator: format!("mcp-imagegen {}", env!("CARGO_PKG_VERSION")),
        };
        match output::write_image(&job.output, &image.png, &record) {
            Ok(written) => {
                let preview_png = if job.want_preview {
                    output::make_preview(&image.png).ok()
                } else {
                    None
                };
                let result = JobResult {
                    job_id: id.to_string(),
                    path: written.path,
                    sidecar_path: written.sidecar_path,
                    record,
                    preview_png,
                };
                entry.finish(JobStatus::Done, Some(result), None);
            }
            Err(err) => entry.finish(JobStatus::Failed, None, Some(to_job_error(&err))),
        }
    }
}

async fn worker(inner: std::sync::Weak<Inner>) {
    let mut idle_armed = false;
    loop {
        let Some(this) = inner.upgrade() else { return };
        if let Some((id, entry)) = this.next_job() {
            this.run(&id, &entry).await;
            *this.running.lock().unwrap() = None;
            idle_armed = true;
            continue;
        }
        let idle = this.cfg.idle_unload;
        let wake = async {
            this.wake.notified().await;
        };
        if idle_armed {
            tokio::select! {
                _ = wake => {}
                _ = tokio::time::sleep(idle) => {
                    this.unload_all().await;
                    idle_armed = false;
                }
            }
        } else {
            wake.await;
        }
    }
}

fn to_job_error(err: &ImageGenError) -> JobError {
    JobError {
        code: err.code().to_string(),
        message: err.to_string(),
    }
}

fn new_job_id() -> String {
    format!("job_{:016x}", rand::random::<u64>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ImageBackend;
    use crate::backend::mock::MockBackend;
    use crate::registry::{Capability, ModelDefaults};
    use std::sync::atomic::Ordering;

    const STEP: Duration = Duration::from_millis(100);

    fn model() -> ModelSpec {
        ModelSpec {
            id: "mock-model".into(),
            backend: "mock".into(),
            capabilities: vec![Capability::Txt2img],
            license: "MIT".into(),
            commercial_weights: true,
            est_memory_mb: 1,
            max_pixels: 1 << 22,
            size_multiple: 8,
            defaults: ModelDefaults {
                width: 64,
                height: 64,
                steps: 4,
                cfg_scale: 1.0,
                sampler: None,
            },
            files: BTreeMap::new(),
            extra_args: vec![],
            mflux: None,
            backend_options: BTreeMap::new(),
            max_ref_images: 0,
            edit_files: BTreeMap::new(),
            loras: vec![],
            sigma_schedule: None,
        }
    }

    fn job(dir: &std::path::Path, name: &str, steps: u32) -> JobRequest {
        JobRequest {
            model: model(),
            files: BTreeMap::new(),
            request: ResolvedRequest {
                model: "mock-model".into(),
                prompt: name.into(),
                negative_prompt: String::new(),
                width: 64,
                height: 64,
                steps,
                cfg_scale: 1.0,
                seed: 7,
                sampler: None,
                warnings: vec![],
                mode: crate::request::Mode::Txt2img,
                references: vec![],
                custom_sigmas: vec![],
                loras: vec![],
            },
            output: dir.join(format!("{name}.png")),
            want_preview: false,
        }
    }

    fn manager(max_queue: usize) -> (JobManager, Arc<MockBackend>) {
        let mock = Arc::new(MockBackend::new(STEP));
        let mut backends = BackendRegistry::default();
        backends.insert(mock.clone());
        let mgr = JobManager::new(
            backends,
            JobConfig {
                max_queue,
                job_ttl: Duration::from_secs(10),
                idle_unload: Duration::from_secs(5),
            },
        );
        (mgr, mock)
    }

    async fn settle() {
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn runs_one_at_a_time_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 3)).unwrap();
        let b = mgr.submit(job(dir.path(), "b", 3)).unwrap();
        settle().await;
        assert_eq!(mgr.get(&a.job_id).unwrap().status, JobStatus::Running);
        let b_now = mgr.get(&b.job_id).unwrap();
        assert_eq!(b_now.status, JobStatus::Queued);
        assert_eq!(b_now.progress, Some(Phase::Queued { position: 1 }));

        let a_done = mgr.wait(&a.job_id).await.unwrap();
        assert_eq!(a_done.status, JobStatus::Done);
        let b_done = mgr.wait(&b.job_id).await.unwrap();
        let ra = a_done.result.unwrap();
        let rb = b_done.result.unwrap();
        assert!(ra.path.exists() && rb.path.exists());
        assert!(ra.sidecar_path.exists());
        assert_eq!(rb.record.seed, 7);
        assert_eq!(rb.record.license, "MIT");
    }

    #[tokio::test(start_paused = true)]
    async fn queue_full() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _) = manager(1);
        mgr.submit(job(dir.path(), "a", 5)).unwrap();
        settle().await; // a is now running, queue empty
        mgr.submit(job(dir.path(), "b", 5)).unwrap();
        let err = mgr.submit(job(dir.path(), "c", 5)).unwrap_err();
        assert_eq!(err.code(), "queue_full");
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_queued_and_running_leave_no_files() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 50)).unwrap();
        let b = mgr.submit(job(dir.path(), "b", 50)).unwrap();
        settle().await;

        let b_snap = mgr.cancel(&b.job_id).unwrap();
        assert_eq!(b_snap.status, JobStatus::Cancelled);

        tokio::time::sleep(STEP * 3).await;
        mgr.cancel(&a.job_id).unwrap();
        let a_done = mgr.wait(&a.job_id).await.unwrap();
        assert_eq!(a_done.status, JobStatus::Cancelled);

        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
        assert_eq!(mock.generations.load(Ordering::SeqCst), 0);
        // Cancelling again is harmless.
        assert_eq!(mgr.cancel(&a.job_id).unwrap().status, JobStatus::Cancelled);
    }

    #[tokio::test(start_paused = true)]
    async fn backend_failure_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock) = manager(4);
        mock.fail_next();
        let a = mgr.submit(job(dir.path(), "a", 2)).unwrap();
        let done = mgr.wait(&a.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Failed);
        let err = done.error.unwrap();
        assert_eq!(err.code, "backend_failed");
        assert!(err.message.contains("mock: boom"), "{}", err.message);
    }

    #[tokio::test(start_paused = true)]
    async fn finished_jobs_expire() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 1)).unwrap();
        mgr.wait(&a.job_id).await.unwrap();
        tokio::time::sleep(Duration::from_secs(9)).await;
        assert!(mgr.get(&a.job_id).is_ok());
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert_eq!(mgr.get(&a.job_id).unwrap_err().code(), "unknown_job");
    }

    #[tokio::test(start_paused = true)]
    async fn idle_unload_fires_once() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 1)).unwrap();
        mgr.wait(&a.job_id).await.unwrap();
        assert!(mock.loaded_model().is_some());
        tokio::time::sleep(Duration::from_secs(4)).await;
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 0);
        tokio::time::sleep(Duration::from_secs(2)).await;
        settle().await;
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 1);
        tokio::time::sleep(Duration::from_secs(60)).await;
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn unload_all_refused_while_busy() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 5)).unwrap();
        settle().await;
        assert!(mgr.unload_all().await.is_err());
        mgr.wait(&a.job_id).await.unwrap();
        assert_eq!(
            mgr.unload_all().await.unwrap(),
            vec!["mock-model".to_string()]
        );
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn preview_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _) = manager(4);
        let mut j = job(dir.path(), "a", 1);
        j.want_preview = true;
        let a = mgr.submit(j).unwrap();
        let done = mgr.wait(&a.job_id).await.unwrap();
        assert!(done.result.unwrap().preview_png.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn progress_is_streamed() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _) = manager(4);
        let a = mgr.submit(job(dir.path(), "a", 3)).unwrap();
        let mut rx = mgr.subscribe(&a.job_id).unwrap();
        let mut seen = Vec::new();
        while rx.changed().await.is_ok() {
            let snap = rx.borrow_and_update().clone();
            if let Some(p) = snap.progress.clone() {
                seen.push(p.to_string());
            }
            if snap.status.is_terminal() {
                break;
            }
        }
        // watch keeps only the latest value, so observers see a subset of steps.
        assert!(seen.iter().any(|s| s.starts_with("sampling ")), "{seen:?}");
    }
}
