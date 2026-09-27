//! stable-diffusion.cpp backend (design §4.2; R3, R4, R7).
//!
//! Owns one `sd-server` child process holding one model. Switching models restarts it.
//! Jobs go through the native async API (`/sdcpp/v1/img_gen`, `/sdcpp/v1/jobs/{id}`).
//! Per-step progress and timings are parsed from the child's output, because the job
//! status carries none. A running job can't be cancelled through the API
//! (`cancel_generating: false`), so cancelling one kills the child.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::Duration;

use base64::Engine as _;
use regex::Regex;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

use super::process::find_executable;
use super::{
    Availability, BackendError, GeneratedImage, ImageBackend, Phase, ProgressSink, Timings,
};
use crate::config::SdcppConfig;
use crate::registry::ModelSpec;
use crate::request::ResolvedRequest;

const LOG_TAIL_LINES: usize = 200;
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const READY_POLL: Duration = Duration::from_millis(250);

/// Shared between the child's output readers and the running job.
#[derive(Default)]
struct Monitor {
    tail: VecDeque<String>,
    sink: Option<ProgressSink>,
    timings: Timings,
}

type SharedMonitor = Arc<Mutex<Monitor>>;

struct Running {
    child: Child,
    model_id: String,
    /// Files the child was launched with; a job needing a different set (e.g. vision weights
    /// that appeared since) triggers a respawn.
    files: BTreeMap<String, PathBuf>,
    base_url: String,
}

pub struct SdcppBackend {
    cfg: SdcppConfig,
    http: reqwest::Client,
    running: tokio::sync::Mutex<Option<Running>>,
    loaded: Mutex<Option<String>>,
    monitor: SharedMonitor,
}

impl SdcppBackend {
    pub fn new(cfg: SdcppConfig) -> Self {
        // A previous instance that was SIGKILL'd can't run kill_on_drop; reap its child now.
        if let Some(pid) = kill_stale_child(&pid_file(&cfg.work_dir), "sd-server") {
            tracing::warn!(pid, "killed orphaned sd-server from a previous run");
        }
        Self {
            cfg,
            http: reqwest::Client::builder()
                .no_proxy()
                .build()
                .expect("reqwest client with default TLS config"),
            running: tokio::sync::Mutex::new(None),
            loaded: Mutex::new(None),
            monitor: SharedMonitor::default(),
        }
    }

    fn tail(&self) -> String {
        let m = self.monitor.lock().unwrap();
        let skip = m.tail.len().saturating_sub(20);
        m.tail
            .iter()
            .skip(skip)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn failed(&self, message: impl Into<String>) -> BackendError {
        BackendError::Failed {
            message: message.into(),
            stderr_tail: self.tail(),
        }
    }

    async fn kill(&self, running: &mut Option<Running>) {
        if let Some(mut r) = running.take() {
            let _ = r.child.kill().await;
            let _ = std::fs::remove_file(pid_file(&self.cfg.work_dir));
        }
        *self.loaded.lock().unwrap() = None;
    }

    /// Make sure a child holding `model` is up, and return its base URL.
    async fn ensure(
        &self,
        running: &mut Option<Running>,
        model: &ModelSpec,
        files: &BTreeMap<String, PathBuf>,
        cancel: &CancellationToken,
    ) -> Result<String, BackendError> {
        if let Some(r) = running.as_mut() {
            let alive = matches!(r.child.try_wait(), Ok(None));
            if alive && r.model_id == model.id && r.files == *files {
                return Ok(r.base_url.clone());
            }
        }
        self.kill(running).await;

        let port = free_port().map_err(|e| self.failed(format!("no free port: {e}")))?;
        for dir in [&self.cfg.lora_dir, &self.cfg.work_dir] {
            std::fs::create_dir_all(dir)
                .map_err(|e| self.failed(format!("cannot create {}: {e}", dir.display())))?;
        }
        let mut args = launch_args(port, files, &self.cfg.extra_args, &model.extra_args);
        if !args.iter().any(|a| a == "--lora-model-dir") {
            args.push("--lora-model-dir".into());
            args.push(self.cfg.lora_dir.to_string_lossy().into_owned());
        }
        tracing::info!(model = %model.id, port, bin = %self.cfg.server_bin.display(), "starting sd-server");
        let program =
            find_executable(&self.cfg.server_bin).unwrap_or_else(|| self.cfg.server_bin.clone());
        let mut child = Command::new(&program)
            .args(&args)
            .current_dir(&self.cfg.work_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| BackendError::Unavailable {
                reason: format!("cannot start {}: {e}", self.cfg.server_bin.display()),
                fix: "build stable-diffusion.cpp and set backends.sdcpp.server_bin in config.toml"
                    .into(),
            })?;
        {
            let mut m = self.monitor.lock().unwrap();
            m.tail.clear();
        }
        if let Some(out) = child.stdout.take() {
            tokio::spawn(read_output(out, self.monitor.clone()));
        }
        if let Some(err) = child.stderr.take() {
            tokio::spawn(read_output(err, self.monitor.clone()));
        }

        if let Some(pid) = child.id() {
            let _ = std::fs::write(pid_file(&self.cfg.work_dir), pid.to_string());
        }
        let base_url = format!("http://127.0.0.1:{port}");
        *running = Some(Running {
            child,
            model_id: model.id.clone(),
            files: files.clone(),
            base_url: base_url.clone(),
        });

        let deadline =
            tokio::time::Instant::now() + Duration::from_secs(self.cfg.startup_timeout_secs);
        loop {
            if cancel.is_cancelled() {
                self.kill(running).await;
                return Err(BackendError::Cancelled);
            }
            if let Some(r) = running.as_mut()
                && let Ok(Some(status)) = r.child.try_wait()
            {
                let err = self.failed(format!("sd-server exited during startup ({status})"));
                self.kill(running).await;
                return Err(err);
            }
            let ready = self
                .http
                .get(format!("{base_url}/sdcpp/v1/capabilities"))
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .is_ok_and(|r| r.status().is_success());
            if ready {
                *self.loaded.lock().unwrap() = Some(model.id.clone());
                return Ok(base_url);
            }
            if tokio::time::Instant::now() >= deadline {
                let err = self.failed(format!(
                    "sd-server did not become ready within {}s",
                    self.cfg.startup_timeout_secs
                ));
                self.kill(running).await;
                return Err(err);
            }
            tokio::time::sleep(READY_POLL).await;
        }
    }
}

#[async_trait::async_trait]
impl ImageBackend for SdcppBackend {
    fn id(&self) -> &'static str {
        "sdcpp"
    }

    async fn availability(&self) -> Availability {
        if find_executable(&self.cfg.server_bin).is_some() {
            Availability::Ready
        } else {
            Availability::Unavailable {
                reason: format!("sd-server not found at `{}`", self.cfg.server_bin.display()),
                fix: "build stable-diffusion.cpp (cmake -DSD_METAL=ON) and set backends.sdcpp.server_bin in config.toml".into(),
            }
        }
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        files: &BTreeMap<String, PathBuf>,
        request: &ResolvedRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedImage, BackendError> {
        let mut running = self.running.lock().await;
        {
            let mut m = self.monitor.lock().unwrap();
            m.sink = Some(progress.clone());
            m.timings = Timings::default();
        }
        progress.report(Phase::Loading);
        let load_started = tokio::time::Instant::now();
        let base_url = self.ensure(&mut running, model, files, &cancel).await?;
        let load_ms = load_started.elapsed().as_millis() as u64;
        progress.report(Phase::Encoding);

        let loras = link_loras(&self.cfg.lora_dir, &model.id, &request.loras).map_err(|e| {
            self.failed(format!(
                "cannot place LoRA in {}: {e}",
                self.cfg.lora_dir.display()
            ))
        })?;
        let client = SdClient {
            http: self.http.clone(),
            base_url,
        };
        let outcome = client
            .run(
                &request_body(request, &model.backend_options, &loras),
                &cancel,
                || {
                    running
                        .as_mut()
                        .is_some_and(|r| matches!(r.child.try_wait(), Ok(Some(_))))
                },
            )
            .await;

        let result = match outcome {
            Ok(png) => {
                let mut timings = self.monitor.lock().unwrap().timings.clone();
                timings.load_ms.get_or_insert(load_ms);
                Ok(GeneratedImage {
                    png,
                    seed: Some(request.seed),
                    timings,
                    ..Default::default()
                })
            }
            Err(RunError::Cancelled { needs_kill }) => {
                if needs_kill {
                    self.kill(&mut running).await;
                }
                Err(BackendError::Cancelled)
            }
            Err(RunError::ChildExited) => {
                let err = self.failed("sd-server exited during generation");
                self.kill(&mut running).await;
                Err(err)
            }
            Err(RunError::Failed(message)) => Err(self.failed(message)),
        };
        self.monitor.lock().unwrap().sink = None;
        result
    }

    async fn unload(&self) {
        let mut running = self.running.lock().await;
        self.kill(&mut running).await;
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}

/// HTTP client for one sd-server instance.
pub struct SdClient {
    pub http: reqwest::Client,
    pub base_url: String,
}

#[derive(Debug, PartialEq)]
pub enum RunError {
    Cancelled { needs_kill: bool },
    ChildExited,
    Failed(String),
}

impl SdClient {
    /// Submit a job, poll until it finishes, and return the first image's PNG bytes.
    pub async fn run(
        &self,
        body: &Value,
        cancel: &CancellationToken,
        mut child_exited: impl FnMut() -> bool,
    ) -> Result<Vec<u8>, RunError> {
        let resp = self
            .http
            .post(format!("{}/sdcpp/v1/img_gen", self.base_url))
            .json(body)
            .send()
            .await
            .map_err(|e| RunError::Failed(format!("submit failed: {e}")))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(RunError::Failed(format!(
                "submit returned {status}: {text}"
            )));
        }
        let job: Value = serde_json::from_str(&text)
            .map_err(|e| RunError::Failed(format!("bad submit response: {e}: {text}")))?;
        let id = job["id"]
            .as_str()
            .ok_or_else(|| RunError::Failed(format!("submit response has no id: {text}")))?
            .to_string();

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    let needs_kill = !self.cancel(&id).await;
                    return Err(RunError::Cancelled { needs_kill });
                }
                _ = tokio::time::sleep(POLL_INTERVAL) => {}
            }
            if child_exited() {
                return Err(RunError::ChildExited);
            }
            let job = match self
                .http
                .get(format!("{}/sdcpp/v1/jobs/{id}", self.base_url))
                .send()
                .await
            {
                Ok(r) => r
                    .json::<Value>()
                    .await
                    .map_err(|e| RunError::Failed(format!("bad job status: {e}")))?,
                // The child may be busy or restarting; the exit check above catches real deaths.
                Err(_) => continue,
            };
            match job["status"].as_str().unwrap_or_default() {
                "completed" => return decode_first_image(&job).map_err(RunError::Failed),
                "failed" => {
                    return Err(RunError::Failed(format!(
                        "generation failed: {}",
                        job["error"]["message"].as_str().unwrap_or("unknown error")
                    )));
                }
                "cancelled" => return Err(RunError::Cancelled { needs_kill: false }),
                _ => {}
            }
        }
    }

    /// True if the server accepted the cancel (job was still queued).
    async fn cancel(&self, id: &str) -> bool {
        self.http
            .post(format!("{}/sdcpp/v1/jobs/{id}/cancel", self.base_url))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }
}

fn decode_first_image(job: &Value) -> Result<Vec<u8>, String> {
    let b64 = job["result"]["images"][0]["b64_json"]
        .as_str()
        .ok_or("completed job has no result.images[0].b64_json")?;
    base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("invalid base64 image: {e}"))
}

/// Native sdcpp API request body. Omitted fields keep sd-server's defaults; per-model
/// `backend_options` fill in extra fields but never override the request's core fields.
pub fn request_body(
    req: &ResolvedRequest,
    options: &BTreeMap<String, Value>,
    loras: &[(String, f32)],
) -> Value {
    let mut sample = json!({
        "sample_steps": req.steps,
        "guidance": { "txt_cfg": req.cfg_scale },
    });
    if let Some(sampler) = &req.sampler {
        sample["sample_method"] = json!(sampler);
    }
    if !req.custom_sigmas.is_empty() {
        sample["custom_sigmas"] = json!(req.custom_sigmas);
    }
    let mut body = json!({
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "width": req.width,
        "height": req.height,
        "seed": req.seed,
        "batch_count": 1,
        "output_format": "png",
        "embed_image_metadata": true,
        "sample_params": sample,
    });
    if !req.references.is_empty() {
        body["ref_images"] = json!(
            req.references
                .iter()
                .map(|r| base64::engine::general_purpose::STANDARD.encode(&r.data))
                .collect::<Vec<_>>()
        );
    }
    if !loras.is_empty() {
        body["lora"] = json!(
            loras
                .iter()
                .map(|(path, m)| json!({ "path": path, "multiplier": m }))
                .collect::<Vec<_>>()
        );
    }
    let obj = body.as_object_mut().expect("body is an object");
    for (key, value) in options {
        obj.entry(key.clone()).or_insert_with(|| value.clone());
    }
    body
}

/// sd-server command line for a model's files.
pub fn launch_args(
    port: u16,
    files: &BTreeMap<String, PathBuf>,
    global_extra: &[String],
    model_extra: &[String],
) -> Vec<String> {
    let mut args = vec![
        "--listen-ip".to_string(),
        "127.0.0.1".to_string(),
        "--listen-port".to_string(),
        port.to_string(),
    ];
    for (role, path) in files {
        args.push(role_flag(role));
        args.push(path.to_string_lossy().into_owned());
    }
    args.extend(global_extra.iter().cloned());
    args.extend(model_extra.iter().cloned());
    args
}

/// Registry file role → sd-server flag.
fn role_flag(role: &str) -> String {
    match role {
        "model" => "--model".into(),
        // These flags keep underscores upstream.
        "clip_l" | "clip_g" | "clip_vision" | "t5xxl" | "llm_vision" => format!("--{role}"),
        other => format!("--{}", other.replace('_', "-")),
    }
}

/// sd-server only accepts LoRA paths relative to its `--lora-model-dir`, looked up in its own
/// scan of that folder. Place each resolved LoRA at `<lora_dir>/<model-id>/<file>` and return
/// the relative paths with multipliers.
pub fn link_loras(
    lora_dir: &Path,
    model_id: &str,
    loras: &[crate::registry::LoraUse],
) -> std::io::Result<Vec<(String, f32)>> {
    let mut out = Vec::new();
    for lora in loras {
        let name = lora
            .path
            .file_name()
            .ok_or_else(|| std::io::Error::other("LoRA path has no file name"))?;
        let dir = lora_dir.join(model_id);
        std::fs::create_dir_all(&dir)?;
        place_file(&lora.path, &dir.join(name))?;
        out.push((
            format!("{model_id}/{}", name.to_string_lossy()),
            lora.multiplier,
        ));
    }
    Ok(out)
}

/// Make `dest` refer to `src` as cheaply as the platform allows: a symlink, else a hard link
/// (same volume), else a copy. Windows needs Developer Mode or admin rights for symlinks, and the
/// Hugging Face cache may live on another drive, so every step has a fallback. Up-to-date targets
/// are left alone.
pub fn place_file(src: &Path, dest: &Path) -> std::io::Result<()> {
    let src_len = std::fs::metadata(src)?.len();
    if std::fs::read_link(dest).ok().as_deref() == Some(src) {
        return Ok(());
    }
    if let Ok(meta) = std::fs::metadata(dest)
        && meta.len() == src_len
        && std::fs::symlink_metadata(dest).is_ok_and(|m| !m.file_type().is_symlink())
    {
        return Ok(()); // an earlier hard link or copy of the same file
    }
    let _ = std::fs::remove_file(dest);
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(src, dest).is_ok();
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(src, dest).is_ok();
    #[cfg(not(any(unix, windows)))]
    let linked = false;
    if linked || std::fs::hard_link(src, dest).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dest).map(|_| ())
}

fn pid_file(work_dir: &Path) -> PathBuf {
    work_dir.join("sd-server.pid")
}

/// Kill the process recorded in `pid_file` if it is still alive and its name contains `expected_name`
/// (so a recycled PID belonging to something else is left alone). Returns the killed PID.
pub fn kill_stale_child(pid_file: &Path, expected_name: &str) -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(pid_file)
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let _ = std::fs::remove_file(pid_file);
    let spid = sysinfo::Pid::from_u32(pid);
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[spid]), true);
    let process = sys.process(spid)?;
    if !process.name().to_string_lossy().contains(expected_name) {
        return None;
    }
    process.kill().then_some(pid)
}

fn free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?.port())
}

static SAMPLING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+)/(\d+) - [\d.]+(?:s/it|it/s)").unwrap());
static TIMING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(get_learned_condition|sampling|decode_first_stage) completed, taking ([\d.]+)s")
        .unwrap()
});

/// What a single output line tells us.
#[derive(Debug, Clone, PartialEq)]
pub enum LineEvent {
    Progress(Phase),
    Timing { stage: String, ms: u64 },
    None,
}

pub fn parse_line(line: &str) -> LineEvent {
    if let Some(c) = SAMPLING_RE.captures(line) {
        let step = c[1].parse().unwrap_or(0);
        let total = c[2].parse().unwrap_or(0);
        return LineEvent::Progress(Phase::Sampling { step, total });
    }
    if let Some(c) = TIMING_RE.captures(line) {
        let secs: f64 = c[2].parse().unwrap_or(0.0);
        return LineEvent::Timing {
            stage: c[1].to_string(),
            ms: (secs * 1000.0).round() as u64,
        };
    }
    if line.contains("sampling using") {
        return LineEvent::Progress(Phase::Sampling { step: 0, total: 0 });
    }
    if line.contains("decode_first_stage") || line.contains("decoding") {
        return LineEvent::Progress(Phase::Decoding);
    }
    LineEvent::None
}

/// Read a child stream, split on `\r` / `\n` (progress bars use `\r`), and feed the monitor.
async fn read_output(mut stream: impl AsyncRead + Unpin, monitor: SharedMonitor) {
    let mut buf = vec![0u8; 8192];
    let mut pending = String::new();
    loop {
        let n = match stream.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        pending.push_str(&String::from_utf8_lossy(&buf[..n]));
        while let Some(pos) = pending.find(['\r', '\n']) {
            let line: String = pending.drain(..=pos).collect();
            let line = strip_ansi(line.trim_end_matches(['\r', '\n']));
            if !line.trim().is_empty() {
                handle_line(&monitor, &line);
            }
        }
    }
}

fn handle_line(monitor: &SharedMonitor, line: &str) {
    let mut m = monitor.lock().unwrap();
    match parse_line(line) {
        LineEvent::Progress(phase) => {
            // Skip the placeholder emitted on "sampling using …" once real steps arrive.
            if let Some(sink) = &m.sink
                && phase != (Phase::Sampling { step: 0, total: 0 })
            {
                sink.report(phase);
            }
            return; // progress bars are noise in the log tail
        }
        LineEvent::Timing { stage, ms } => match stage.as_str() {
            "sampling" => m.timings.sample_ms = Some(ms),
            "decode_first_stage" => m.timings.decode_ms = Some(ms),
            _ => {}
        },
        LineEvent::None => {}
    }
    if m.tail.len() >= LOG_TAIL_LINES {
        m.tail.pop_front();
    }
    m.tail.push_back(line.to_string());
}

fn strip_ansi(s: &str) -> String {
    static ANSI_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\x1b\[[0-9;]*[A-Za-z]").unwrap());
    ANSI_RE.replace_all(s, "").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    fn req() -> ResolvedRequest {
        ResolvedRequest {
            model: "qwen-image-2.1".into(),
            prompt: "a crate".into(),
            negative_prompt: "blurry".into(),
            width: 1024,
            height: 768,
            steps: 20,
            cfg_scale: 6.0,
            seed: 42,
            sampler: Some("euler".into()),
            warnings: vec![],
            mode: crate::request::Mode::Txt2img,
            references: vec![],
            custom_sigmas: vec![],
            loras: vec![],
        }
    }

    #[test]
    fn body_maps_resolved_request() {
        let body = request_body(&req(), &BTreeMap::new(), &[]);
        assert_eq!(body["prompt"], "a crate");
        assert_eq!(body["negative_prompt"], "blurry");
        assert_eq!(body["width"], 1024);
        assert_eq!(body["seed"], 42);
        assert_eq!(body["batch_count"], 1);
        assert_eq!(body["output_format"], "png");
        assert_eq!(body["sample_params"]["sample_steps"], 20);
        assert_eq!(body["sample_params"]["sample_method"], "euler");
        assert_eq!(body["sample_params"]["guidance"]["txt_cfg"], 6.0);
        let mut no_sampler = req();
        no_sampler.sampler = None;
        assert!(
            request_body(&no_sampler, &BTreeMap::new(), &[])["sample_params"]
                .get("sample_method")
                .is_none()
        );
    }

    #[test]
    fn backend_options_add_fields_but_never_override_core_ones() {
        let opts = BTreeMap::from([
            ("cache_mode".to_string(), json!("easycache")),
            ("seed".to_string(), json!(7)),
        ]);
        let body = request_body(&req(), &opts, &[]);
        assert_eq!(body["cache_mode"], "easycache");
        assert_eq!(body["seed"], 42);
    }

    #[test]
    fn references_become_ref_images() {
        let mut r = req();
        assert!(
            request_body(&r, &BTreeMap::new(), &[])
                .get("ref_images")
                .is_none()
        );
        r.mode = crate::request::Mode::Edit;
        r.references = vec![crate::input::Reference {
            path: "/a.png".into(),
            width: 1,
            height: 1,
            format: "png".into(),
            sha256: "00".into(),
            data: vec![1u8, 2, 3].into(),
        }];
        let body = request_body(&r, &BTreeMap::new(), &[]);
        assert_eq!(body["ref_images"], json!(["AQID"]));
    }

    #[test]
    fn sigmas_and_loras_go_into_the_body() {
        let mut r = req();
        r.custom_sigmas = vec![1.0, 0.5, 0.0];
        let body = request_body(&r, &BTreeMap::new(), &[("m/turbo.safetensors".into(), 1.0)]);
        assert_eq!(
            body["sample_params"]["custom_sigmas"],
            json!([1.0, 0.5, 0.0])
        );
        assert_eq!(
            body["lora"],
            json!([{"path": "m/turbo.safetensors", "multiplier": 1.0}])
        );
        let plain = request_body(&req(), &BTreeMap::new(), &[]);
        assert!(plain.get("lora").is_none());
        assert!(plain["sample_params"].get("custom_sigmas").is_none());
    }

    #[test]
    fn loras_are_linked_relative_to_the_lora_dir() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("cache/turbo.safetensors");
        std::fs::create_dir_all(src.parent().unwrap()).unwrap();
        std::fs::write(&src, b"w").unwrap();
        let loras = vec![crate::registry::LoraUse {
            path: src.clone(),
            multiplier: 0.8,
        }];
        let lora_dir = dir.path().join("loras");
        let rel = link_loras(&lora_dir, "qwen-turbo", &loras).unwrap();
        assert_eq!(rel, vec![("qwen-turbo/turbo.safetensors".to_string(), 0.8)]);
        let placed = lora_dir.join("qwen-turbo/turbo.safetensors");
        assert_eq!(std::fs::read(&placed).unwrap(), b"w");
        // Idempotent.
        assert_eq!(link_loras(&lora_dir, "qwen-turbo", &loras).unwrap(), rel);
    }

    #[test]
    fn launch_args_map_roles_to_flags() {
        let files = BTreeMap::from([
            ("diffusion_model".to_string(), PathBuf::from("/w/d.gguf")),
            ("llm".to_string(), PathBuf::from("/w/l.gguf")),
            ("vae".to_string(), PathBuf::from("/w/v.safetensors")),
            ("llm_vision".to_string(), PathBuf::from("/w/mm.gguf")),
        ]);
        let args = launch_args(9000, &files, &["--diffusion-fa".into()], &["--fa".into()]);
        assert_eq!(
            args,
            vec![
                "--listen-ip",
                "127.0.0.1",
                "--listen-port",
                "9000",
                "--diffusion-model",
                "/w/d.gguf",
                "--llm",
                "/w/l.gguf",
                "--llm_vision",
                "/w/mm.gguf",
                "--vae",
                "/w/v.safetensors",
                "--diffusion-fa",
                "--fa",
            ]
        );
    }

    #[test]
    fn parses_progress_and_timings() {
        // Captured from sd-cli / sd-server output.
        let bar = "  |==================>                               | 3/8 - 12.34s/it";
        assert_eq!(
            parse_line(bar),
            LineEvent::Progress(Phase::Sampling { step: 3, total: 8 })
        );
        assert_eq!(
            parse_line("|=====| 20/20 - 1.52it/s"),
            LineEvent::Progress(Phase::Sampling {
                step: 20,
                total: 20
            })
        );
        // Tensor-loading bars report MB/s and must not count as sampling.
        assert_eq!(
            parse_line("|####      | 120/450 - 850.12MB/s"),
            LineEvent::None
        );
        assert_eq!(
            parse_line("[INFO ] stable-diffusion.cpp:4389 - sampling completed, taking 384.91s"),
            LineEvent::Timing {
                stage: "sampling".into(),
                ms: 384_910
            }
        );
        assert_eq!(
            parse_line(
                "[INFO ] stable-diffusion.cpp:4129 - decode_first_stage completed, taking 22.17s"
            ),
            LineEvent::Timing {
                stage: "decode_first_stage".into(),
                ms: 22_170
            }
        );
        assert_eq!(
            parse_line("[INFO ] ggml_metal_init: found device: Apple M4"),
            LineEvent::None
        );
    }

    #[cfg(unix)]
    #[test]
    fn stale_child_is_killed_only_if_name_matches() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sd-server.pid");
        let mut child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .unwrap();
        std::fs::write(&file, child.id().to_string()).unwrap();

        // Wrong name: left alone, but the stale file is cleared.
        assert_eq!(kill_stale_child(&file, "sd-server"), None);
        assert!(!file.exists());
        assert!(child.try_wait().unwrap().is_none());

        std::fs::write(&file, child.id().to_string()).unwrap();
        assert_eq!(kill_stale_child(&file, "sleep"), Some(child.id()));
        assert!(child.wait().is_ok());

        // Dead PID or missing file: nothing to do.
        std::fs::write(&file, child.id().to_string()).unwrap();
        assert_eq!(kill_stale_child(&file, "sleep"), None);
        assert_eq!(kill_stale_child(&file, "sleep"), None);
    }

    #[test]
    fn strips_ansi() {
        assert_eq!(
            strip_ansi("\x1b[K| 1/8 - 2.00s/it\x1b[0m"),
            "| 1/8 - 2.00s/it"
        );
    }

    #[test]
    fn monitor_reports_progress_and_keeps_log_tail() {
        let monitor = SharedMonitor::default();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        monitor.lock().unwrap().sink = Some(ProgressSink::new(move |p| s.lock().unwrap().push(p)));
        handle_line(&monitor, "| 2/8 - 1.00s/it");
        handle_line(&monitor, "[ERROR] something broke");
        handle_line(&monitor, "sampling completed, taking 1.5s");
        assert_eq!(
            *seen.lock().unwrap(),
            vec![Phase::Sampling { step: 2, total: 8 }]
        );
        let m = monitor.lock().unwrap();
        assert_eq!(m.timings.sample_ms, Some(1500));
        assert!(m.tail.iter().any(|l| l.contains("something broke")));
        assert!(!m.tail.iter().any(|l| l.contains("2/8")));
    }

    // ---- stub sd-server: canned HTTP responses keyed by "METHOD path" ----

    type Routes = Arc<Mutex<Vec<(String, u16, String)>>>;

    async fn stub(routes: Routes) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let routes = routes.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..n]).to_string();
                    let mut parts = head.split_whitespace();
                    let key = format!(
                        "{} {}",
                        parts.next().unwrap_or(""),
                        parts.next().unwrap_or("")
                    );
                    let (status, body) = {
                        let mut r = routes.lock().unwrap();
                        match r.iter().position(|(k, _, _)| *k == key) {
                            // Routes are consumed in order; the last one for a key repeats.
                            Some(i) if r.iter().filter(|(k, _, _)| *k == key).count() > 1 => {
                                let (_, s, b) = r.remove(i);
                                (s, b)
                            }
                            Some(i) => (r[i].1, r[i].2.clone()),
                            None => (404, "{}".to_string()),
                        }
                    };
                    let resp = format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });
        format!("http://{addr}")
    }

    fn client(base_url: String) -> SdClient {
        SdClient {
            http: reqwest::Client::builder().no_proxy().build().unwrap(),
            base_url,
        }
    }

    fn route(key: &str, status: u16, body: Value) -> (String, u16, String) {
        (key.to_string(), status, body.to_string())
    }

    #[tokio::test]
    async fn submit_poll_and_decode() {
        let png = crate::backend::mock::MockBackend::render(42);
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        let routes: Routes = Arc::new(Mutex::new(vec![
            route(
                "POST /sdcpp/v1/img_gen",
                202,
                json!({"id": "job_1", "status": "queued"}),
            ),
            route(
                "GET /sdcpp/v1/jobs/job_1",
                200,
                json!({"id": "job_1", "status": "generating"}),
            ),
            route(
                "GET /sdcpp/v1/jobs/job_1",
                200,
                json!({"id": "job_1", "status": "completed", "result": {"output_format": "png", "images": [{"index": 0, "b64_json": b64}]}}),
            ),
        ]));
        let base = stub(routes).await;
        let got = client(base)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &CancellationToken::new(),
                || false,
            )
            .await
            .unwrap();
        assert_eq!(got, png);
    }

    #[tokio::test]
    async fn failed_job_reports_message() {
        let routes: Routes = Arc::new(Mutex::new(vec![
            route("POST /sdcpp/v1/img_gen", 202, json!({"id": "j"})),
            route(
                "GET /sdcpp/v1/jobs/j",
                200,
                json!({"status": "failed", "error": {"code": "generation_failed", "message": "out of memory"}}),
            ),
        ]));
        let err = client(stub(routes).await)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &CancellationToken::new(),
                || false,
            )
            .await
            .unwrap_err();
        assert_eq!(
            err,
            RunError::Failed("generation failed: out of memory".into())
        );
    }

    #[tokio::test]
    async fn cancel_while_generating_needs_kill() {
        let routes: Routes = Arc::new(Mutex::new(vec![
            route("POST /sdcpp/v1/img_gen", 202, json!({"id": "j"})),
            route("GET /sdcpp/v1/jobs/j", 200, json!({"status": "generating"})),
            route(
                "POST /sdcpp/v1/jobs/j/cancel",
                409,
                json!({"error": "job is generating"}),
            ),
        ]));
        let cancel = CancellationToken::new();
        let c = cancel.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(700)).await;
            c.cancel();
        });
        let err = client(stub(routes).await)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &cancel,
                || false,
            )
            .await
            .unwrap_err();
        assert_eq!(err, RunError::Cancelled { needs_kill: true });
    }

    #[tokio::test]
    async fn cancel_while_queued_does_not_kill() {
        let routes: Routes = Arc::new(Mutex::new(vec![
            route("POST /sdcpp/v1/img_gen", 202, json!({"id": "j"})),
            route("GET /sdcpp/v1/jobs/j", 200, json!({"status": "queued"})),
            route(
                "POST /sdcpp/v1/jobs/j/cancel",
                200,
                json!({"status": "cancelled"}),
            ),
        ]));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let err = client(stub(routes).await)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &cancel,
                || false,
            )
            .await
            .unwrap_err();
        assert_eq!(err, RunError::Cancelled { needs_kill: false });
    }

    #[tokio::test]
    async fn child_death_is_detected() {
        let routes: Routes = Arc::new(Mutex::new(vec![
            route("POST /sdcpp/v1/img_gen", 202, json!({"id": "j"})),
            route("GET /sdcpp/v1/jobs/j", 200, json!({"status": "generating"})),
        ]));
        let err = client(stub(routes).await)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &CancellationToken::new(),
                || true,
            )
            .await
            .unwrap_err();
        assert_eq!(err, RunError::ChildExited);
    }

    #[tokio::test]
    async fn submit_error_is_reported() {
        let routes: Routes = Arc::new(Mutex::new(vec![route(
            "POST /sdcpp/v1/img_gen",
            400,
            json!({"error": "width must be a multiple of 32"}),
        )]));
        let err = client(stub(routes).await)
            .run(
                &request_body(&req(), &BTreeMap::new(), &[]),
                &CancellationToken::new(),
                || false,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, RunError::Failed(m) if m.contains("multiple of 32")));
    }
}
