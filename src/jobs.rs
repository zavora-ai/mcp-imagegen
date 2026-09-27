//! Job manager (design §7 steps 4–6, §13.4; R4, R5, R15).
//!
//! Two lanes, each with a bounded FIFO queue and one worker: local backends run strictly one job
//! at a time, and so do cloud backends, but a cloud job never waits behind a local render.
//! Every job has a `watch` channel carrying its latest snapshot, so callers can wait, poll or
//! stream progress. Finished jobs are kept for `job_ttl`; local backends are unloaded after
//! `idle_unload` without local work.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{Notify, watch};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::backend::{BackendError, BackendRegistry, Phase, ProgressSink, Runs, Timings};
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

/// Which queue a job waits in, from its backend's `runs()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Lane {
    Local,
    Cloud,
}

impl Lane {
    const ALL: [Lane; 2] = [Lane::Local, Lane::Cloud];

    fn index(self) -> usize {
        match self {
            Self::Local => 0,
            Self::Cloud => 1,
        }
    }
}

impl From<Runs> for Lane {
    fn from(runs: Runs) -> Self {
        match runs {
            Runs::Local => Self::Local,
            Runs::Cloud => Self::Cloud,
        }
    }
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
    pub runs: Runs,
    pub license: String,
    /// Local models only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commercial_weights: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commercial_outputs: Option<bool>,
    pub prompt: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub negative_prompt: String,
    /// `null` when the backend has no seeds.
    pub seed: Option<u64>,
    /// Actual pixels of the written image.
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
    /// Backend provenance (e.g. Codex version, thread id, source file).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_details: BTreeMap<String, serde_json::Value>,
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
    lane: Lane,
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

#[derive(Default)]
struct LaneState {
    queue: Mutex<VecDeque<String>>,
    running: Mutex<Option<String>>,
    wake: Notify,
}

struct Inner {
    jobs: Mutex<HashMap<String, Arc<JobEntry>>>,
    lanes: [LaneState; 2],
    backends: BackendRegistry,
    cfg: JobConfig,
}

#[derive(Clone)]
pub struct JobManager {
    inner: Arc<Inner>,
}

impl JobManager {
    /// Create the manager and spawn one worker per lane on the current tokio runtime.
    pub fn new(backends: BackendRegistry, cfg: JobConfig) -> Self {
        let inner = Arc::new(Inner {
            jobs: Mutex::new(HashMap::new()),
            lanes: Default::default(),
            backends,
            cfg,
        });
        for lane in Lane::ALL {
            tokio::spawn(worker(Arc::downgrade(&inner), lane));
        }
        Self { inner }
    }

    pub fn backends(&self) -> &BackendRegistry {
        &self.inner.backends
    }

    /// Queue a job in its backend's lane. Fails with `QueueFull` when `max_queue` jobs are
    /// already waiting in that lane.
    pub fn submit(&self, job: JobRequest) -> Result<JobSnapshot, ImageGenError> {
        self.prune();
        let id = new_job_id();
        let lane = self
            .inner
            .backends
            .get(&job.model.backend)
            .map(|b| Lane::from(b.runs()))
            .unwrap_or(Lane::Local);
        let mut queue = self.inner.lane(lane).queue.lock().unwrap();
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
            lane,
            cancel: CancellationToken::new(),
            request: Mutex::new(Some(job)),
            finished_at: Mutex::new(None),
        });
        self.inner.jobs.lock().unwrap().insert(id.clone(), entry);
        queue.push_back(id);
        drop(queue);
        self.inner.lane(lane).wake.notify_one();
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
            let mut queue = self.inner.lane(entry.lane).queue.lock().unwrap();
            let before = queue.len();
            queue.retain(|q| q != id);
            before != queue.len()
        };
        entry.cancel.cancel();
        if removed {
            entry.request.lock().unwrap().take();
            entry.finish(JobStatus::Cancelled, None, None);
            self.inner.renumber_queue(entry.lane);
        }
        Ok(entry.snapshot())
    }

    /// Jobs queued plus running, across both lanes.
    pub fn busy(&self) -> usize {
        Lane::ALL.iter().map(|l| self.busy_in(*l)).sum()
    }

    /// Jobs queued plus the one running in `lane`.
    pub fn busy_in(&self, lane: Lane) -> usize {
        let state = self.inner.lane(lane);
        state.queue.lock().unwrap().len() + usize::from(state.running.lock().unwrap().is_some())
    }

    /// Unload every backend now. Refused while a local job is running or queued; cloud jobs
    /// hold no local memory, so they don't block it.
    pub async fn unload_all(&self) -> Result<Vec<String>, ImageGenError> {
        let local = self.busy_in(Lane::Local);
        if local > 0 {
            return Err(ImageGenError::invalid(
                "unload_models",
                format!("{local} local job(s) queued or running; cancel them first or wait"),
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
    fn lane(&self, lane: Lane) -> &LaneState {
        &self.lanes[lane.index()]
    }

    fn renumber_queue(&self, lane: Lane) {
        let queue = self.lane(lane).queue.lock().unwrap();
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

    fn next_job(&self, lane: Lane) -> Option<(String, Arc<JobEntry>)> {
        let state = self.lane(lane);
        let id = state.queue.lock().unwrap().pop_front()?;
        let entry = self.jobs.lock().unwrap().get(&id).cloned()?;
        *state.running.lock().unwrap() = Some(id.clone());
        self.renumber_queue(lane);
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

        // Only one local model resident at a time across backends (R5). Cloud jobs hold no local
        // memory, so they never evict anything.
        let runs = backend.runs();
        if runs == Runs::Local {
            for other in self.backends.all() {
                if other.id() != backend.id()
                    && other.runs() == Runs::Local
                    && other.loaded_model().is_some()
                {
                    other.unload().await;
                }
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
        let (width, height) =
            output::image_dimensions(&image.png).unwrap_or((req.width, req.height));
        let mut warnings = req.warnings.clone();
        warnings.extend(image.warnings.iter().cloned());
        if (width, height) != (req.width, req.height) {
            warnings.push(format!(
                "the backend returned {width}x{height} for the requested {}x{}",
                req.width, req.height
            ));
        }
        let record = GenerationRecord {
            model: job.model.id.clone(),
            backend: backend_id.clone(),
            runs,
            license: job.model.license.clone(),
            commercial_weights: (runs == Runs::Local).then_some(job.model.commercial_weights),
            commercial_outputs: job.model.commercial_outputs,
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone(),
            seed: image.seed,
            width,
            height,
            steps: req.steps,
            cfg_scale: req.cfg_scale,
            sampler: req.sampler.clone(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            timings: image.timings.clone(),
            warnings,
            backend_options: job.model.backend_options.clone(),
            mode: req.mode,
            references: req.references.clone(),
            loras: req.loras.clone(),
            custom_sigmas: req.custom_sigmas.clone(),
            backend_details: image.details.clone(),
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

async fn worker(inner: std::sync::Weak<Inner>, lane: Lane) {
    let mut idle_armed = false;
    loop {
        let Some(this) = inner.upgrade() else { return };
        if let Some((id, entry)) = this.next_job(lane) {
            this.run(&id, &entry).await;
            *this.lane(lane).running.lock().unwrap() = None;
            // Only local work leaves models resident, so only the local lane arms the idle unload.
            idle_armed = lane == Lane::Local;
            continue;
        }
        let idle = this.cfg.idle_unload;
        let wake = async {
            this.lane(lane).wake.notified().await;
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
            commercial_outputs: None,
            files: BTreeMap::new(),
            extra_args: vec![],
            mflux: None,
            codex: None,
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
        let (mgr, mock, _) = manager_with_cloud(max_queue);
        (mgr, mock)
    }

    /// A local mock (`mock`) and a cloud-flavoured one (`mock-cloud`).
    fn manager_with_cloud(max_queue: usize) -> (JobManager, Arc<MockBackend>, Arc<MockBackend>) {
        let mock = Arc::new(MockBackend::new(STEP));
        let cloud = Arc::new(MockBackend::cloud(STEP));
        let mut backends = BackendRegistry::default();
        backends.insert(mock.clone());
        backends.insert(cloud.clone());
        let mgr = JobManager::new(
            backends,
            JobConfig {
                max_queue,
                job_ttl: Duration::from_secs(10),
                idle_unload: Duration::from_secs(5),
            },
        );
        (mgr, mock, cloud)
    }

    fn cloud_job(dir: &std::path::Path, name: &str, steps: u32) -> JobRequest {
        let mut j = job(dir, name, steps);
        j.model.id = "cloud-model".into();
        j.model.backend = "mock-cloud".into();
        j.model.commercial_outputs = Some(true);
        j
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
        assert_eq!(rb.record.seed, Some(7));
        assert_eq!(rb.record.runs, Runs::Local);
        assert_eq!(rb.record.commercial_weights, Some(true));
        assert_eq!((rb.record.width, rb.record.height), (64, 64));
        assert!(rb.record.warnings.is_empty(), "{:?}", rb.record.warnings);
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

    #[tokio::test(start_paused = true)]
    async fn cloud_job_runs_while_a_local_job_is_busy() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _, _) = manager_with_cloud(4);
        let local = mgr.submit(job(dir.path(), "local", 50)).unwrap();
        let cloud = mgr.submit(cloud_job(dir.path(), "cloud", 2)).unwrap();
        settle().await;
        // Both lanes start at once: the cloud job isn't queued behind the local one.
        assert_eq!(mgr.get(&cloud.job_id).unwrap().status, JobStatus::Running);
        let done = mgr.wait(&cloud.job_id).await.unwrap();
        assert_eq!(done.status, JobStatus::Done);
        assert_eq!(mgr.get(&local.job_id).unwrap().status, JobStatus::Running);
        assert_eq!(mgr.busy_in(Lane::Local), 1);
        assert_eq!(mgr.busy_in(Lane::Cloud), 0);

        let r = done.result.unwrap().record;
        assert_eq!(r.runs, Runs::Cloud);
        assert_eq!(r.seed, None);
        assert_eq!(r.commercial_weights, None);
        assert_eq!(r.commercial_outputs, Some(true));
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.path().join("cloud.json")).unwrap()).unwrap();
        assert!(sidecar["seed"].is_null());
        assert_eq!(sidecar["runs"], "cloud");
        assert!(sidecar.get("commercial_weights").is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_jobs_queue_in_their_own_lane() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, _, _) = manager_with_cloud(1);
        mgr.submit(job(dir.path(), "l1", 20)).unwrap();
        mgr.submit(cloud_job(dir.path(), "c1", 20)).unwrap();
        settle().await;
        // One waiting job per lane is allowed with max_queue = 1.
        mgr.submit(job(dir.path(), "l2", 1)).unwrap();
        let c2 = mgr.submit(cloud_job(dir.path(), "c2", 1)).unwrap();
        assert_eq!(c2.progress, Some(Phase::Queued { position: 1 }));
        assert_eq!(
            mgr.submit(cloud_job(dir.path(), "c3", 1))
                .unwrap_err()
                .code(),
            "queue_full"
        );
        assert_eq!(mgr.busy(), 4);
    }

    #[tokio::test(start_paused = true)]
    async fn cloud_jobs_never_unload_local_models() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock, _) = manager_with_cloud(4);
        let a = mgr.submit(job(dir.path(), "a", 1)).unwrap();
        mgr.wait(&a.job_id).await.unwrap();
        assert!(mock.loaded_model().is_some());
        let c = mgr.submit(cloud_job(dir.path(), "c", 1)).unwrap();
        mgr.wait(&c.job_id).await.unwrap();
        assert!(mock.loaded_model().is_some());
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn unload_allowed_while_only_cloud_jobs_run() {
        let dir = tempfile::tempdir().unwrap();
        let (mgr, mock, _) = manager_with_cloud(4);
        let a = mgr.submit(job(dir.path(), "a", 1)).unwrap();
        mgr.wait(&a.job_id).await.unwrap();
        let c = mgr.submit(cloud_job(dir.path(), "c", 30)).unwrap();
        settle().await;
        assert_eq!(mgr.get(&c.job_id).unwrap().status, JobStatus::Running);
        assert_eq!(
            mgr.unload_all().await.unwrap(),
            vec!["mock-model".to_string()]
        );
        assert_eq!(mock.unloads.load(Ordering::SeqCst), 1);
        assert_eq!(mgr.wait(&c.job_id).await.unwrap().status, JobStatus::Done);
    }

    #[tokio::test(start_paused = true)]
    async fn size_mismatch_is_recorded_as_actual_with_a_warning() {
        use crate::backend::{GeneratedImage, ImageBackend};
        struct Fixed;
        #[async_trait::async_trait]
        impl ImageBackend for Fixed {
            fn id(&self) -> &'static str {
                "fixed"
            }
            async fn availability(&self) -> crate::backend::Availability {
                crate::backend::Availability::Ready
            }
            async fn generate(
                &self,
                _: &ModelSpec,
                _: &BTreeMap<String, PathBuf>,
                _: &ResolvedRequest,
                _: ProgressSink,
                _: CancellationToken,
            ) -> Result<GeneratedImage, BackendError> {
                let mut details = BTreeMap::new();
                details.insert("thread_id".to_string(), serde_json::json!("t-1"));
                Ok(GeneratedImage {
                    png: MockBackend::render_sized(1, 48, 32),
                    warnings: vec!["from the backend".into()],
                    details,
                    ..Default::default()
                })
            }
            async fn unload(&self) {}
            fn loaded_model(&self) -> Option<String> {
                None
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let mut backends = BackendRegistry::default();
        backends.insert(Arc::new(Fixed));
        let mgr = JobManager::new(
            backends,
            JobConfig {
                max_queue: 2,
                job_ttl: Duration::from_secs(10),
                idle_unload: Duration::from_secs(5),
            },
        );
        let mut j = job(dir.path(), "f", 1);
        j.model.backend = "fixed".into();
        let done = mgr.wait(&mgr.submit(j).unwrap().job_id).await.unwrap();
        let r = done.result.unwrap().record;
        assert_eq!((r.width, r.height), (48, 32));
        assert_eq!(r.warnings[0], "from the backend");
        assert!(
            r.warnings[1].contains("returned 48x32 for the requested 64x64"),
            "{:?}",
            r.warnings
        );
        assert_eq!(r.backend_details["thread_id"], "t-1");
    }
}
