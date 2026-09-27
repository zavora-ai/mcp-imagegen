//! Codex CLI backend (design §13; R12–R16).
//!
//! Each job runs `codex exec --json` in a read-only sandbox with the instruction on stdin (no
//! argument escaping, so `.cmd` shims on Windows and long or multi-line prompts are safe).
//! Codex's built-in image tool saves the picture under `$CODEX_HOME/generated_images/<thread_id>/`,
//! and the thread id is the first JSONL event, so the output is found without trusting the
//! agent's reply. Prompts and reference images are sent to OpenAI.

use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::process::{find_executable, kill_tree, own_process_group};
use super::{Availability, BackendError, GeneratedImage, ImageBackend, Phase, ProgressSink, Runs};
use crate::config::CodexConfig;
use crate::input::Reference;
use crate::registry::{CodexModel, ModelSpec};
use crate::request::{Mode, ResolvedRequest};

/// How long a login check stays valid, so `list_models` doesn't spawn Codex every call.
const AVAILABILITY_TTL: Duration = Duration::from_secs(60);
/// Limit for `codex login status` / `codex --version`.
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
/// After stdout closes, how long Codex gets to exit before it's killed.
const EXIT_GRACE: Duration = Duration::from_secs(10);
const STDERR_TAIL_LINES: usize = 20;
const IMAGE_EXTENSIONS: [&str; 4] = ["png", "jpg", "jpeg", "webp"];
const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

pub const PROVIDER: &str = "OpenAI, through Codex CLI";

pub struct CodexBackend {
    cfg: CodexConfig,
    availability: Mutex<Option<(Instant, Availability)>>,
    version: tokio::sync::OnceCell<String>,
}

impl CodexBackend {
    pub fn new(cfg: CodexConfig) -> Self {
        Self {
            cfg,
            availability: Mutex::new(None),
            version: tokio::sync::OnceCell::new(),
        }
    }

    /// (reason, fix) when the binary can't be found.
    fn not_found_parts(&self) -> (String, String) {
        (
            format!("Codex CLI not found at `{}`", self.cfg.bin.display()),
            "install the Codex CLI (https://github.com/openai/codex) or set backends.codex.bin in config.toml".into(),
        )
    }

    fn not_found(&self) -> BackendError {
        let (reason, fix) = self.not_found_parts();
        BackendError::Unavailable { reason, fix }
    }

    /// Run a short Codex command; returns (success, stdout + stderr).
    async fn quick(&self, bin: &Path, args: &[&str]) -> Result<(bool, String), String> {
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .env("CODEX_HOME", &self.cfg.codex_home)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let out = tokio::time::timeout(PROBE_TIMEOUT, cmd.output())
            .await
            .map_err(|_| format!("timed out after {} s", PROBE_TIMEOUT.as_secs()))?
            .map_err(|e| e.to_string())?;
        let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
        text.push_str(&String::from_utf8_lossy(&out.stderr));
        Ok((out.status.success(), text.trim().to_string()))
    }

    async fn probe(&self) -> Availability {
        let Some(bin) = find_executable(&self.cfg.bin) else {
            let (reason, fix) = self.not_found_parts();
            return Availability::Unavailable { reason, fix };
        };
        match self.quick(&bin, &["login", "status"]).await {
            Ok((true, out)) if out.contains("Logged in") => Availability::Ready,
            Ok((_, out)) => Availability::Unavailable {
                reason: format!(
                    "Codex is not signed in ({})",
                    out.lines().next().unwrap_or("no output")
                ),
                fix: "run `codex login`".into(),
            },
            Err(e) => Availability::Unavailable {
                reason: format!("`codex login status` failed: {e}"),
                fix: "check the Codex install with `codex doctor`".into(),
            },
        }
    }

    async fn version(&self, bin: &Path) -> String {
        self.version
            .get_or_init(|| async {
                match self.quick(bin, &["--version"]).await {
                    Ok((true, out)) if !out.is_empty() => out,
                    _ => "unknown".to_string(),
                }
            })
            .await
            .clone()
    }

    fn forget_availability(&self) {
        *self.availability.lock().unwrap() = None;
    }
}

#[async_trait::async_trait]
impl ImageBackend for CodexBackend {
    fn id(&self) -> &'static str {
        "codex"
    }

    async fn availability(&self) -> Availability {
        if let Some((at, cached)) = self.availability.lock().unwrap().clone()
            && at.elapsed() < AVAILABILITY_TTL
        {
            return cached;
        }
        let fresh = self.probe().await;
        *self.availability.lock().unwrap() = Some((Instant::now(), fresh.clone()));
        fresh
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        _files: &BTreeMap<String, PathBuf>,
        request: &ResolvedRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedImage, BackendError> {
        let codex = model.codex.as_ref().ok_or_else(|| {
            BackendError::failed(format!(
                "model `{}` has no `codex` section in models.toml",
                model.id
            ))
        })?;
        let bin = find_executable(&self.cfg.bin).ok_or_else(|| self.not_found())?;
        progress.report(Phase::Starting {
            backend: "codex".into(),
        });
        let version = self.version(&bin).await;

        let job_dir = JobDir::create(&self.cfg.work_dir)
            .map_err(|e| BackendError::failed(format!("cannot create a Codex work folder: {e}")))?;
        let references = job_dir
            .write_references(&request.references)
            .map_err(|e| BackendError::failed(format!("cannot stage reference images: {e}")))?;

        let mut cmd = Command::new(&bin);
        cmd.args(exec_args(&self.cfg, codex, job_dir.path(), &references))
            .env("CODEX_HOME", &self.cfg.codex_home)
            .current_dir(job_dir.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        own_process_group(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| BackendError::Unavailable {
            reason: format!("cannot start {}: {e}", bin.display()),
            fix: "check the Codex install with `codex doctor`, or set backends.codex.bin".into(),
        })?;

        if let Some(mut stdin) = child.stdin.take() {
            let text = instruction(request);
            tokio::spawn(async move {
                let _ = stdin.write_all(text.as_bytes()).await;
                let _ = stdin.shutdown().await;
            });
        }
        let tail: Arc<Mutex<VecDeque<String>>> = Arc::default();
        if let Some(stderr) = child.stderr.take() {
            let tail = tail.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut t = tail.lock().unwrap();
                    if t.len() >= STDERR_TAIL_LINES {
                        t.pop_front();
                    }
                    t.push_back(line);
                }
            });
        }
        let tail_text = || {
            tail.lock()
                .unwrap()
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        };
        let failed = |message: String| BackendError::Failed {
            message,
            stderr_tail: tail_text(),
        };

        let mut events = RunEvents::default();
        let Some(stdout) = child.stdout.take() else {
            kill_tree(&mut child).await;
            return Err(failed("Codex started without an output stream".into()));
        };
        let mut lines = BufReader::new(stdout).lines();
        let deadline = tokio::time::sleep(Duration::from_secs(self.cfg.timeout_secs));
        tokio::pin!(deadline);
        let stop = loop {
            tokio::select! {
                _ = cancel.cancelled() => break Some(Stop::Cancelled),
                _ = &mut deadline => break Some(Stop::TimedOut),
                line = lines.next_line() => match line {
                    Ok(Some(line)) => {
                        if events.push_line(&line) {
                            progress.report(Phase::Generating);
                        }
                    }
                    Ok(None) => break None,
                    Err(e) => break Some(Stop::Read(e.to_string())),
                },
            }
        };
        match stop {
            Some(Stop::Cancelled) => {
                kill_tree(&mut child).await;
                return Err(BackendError::Cancelled);
            }
            Some(Stop::TimedOut) => {
                kill_tree(&mut child).await;
                return Err(failed(format!(
                    "Codex didn't finish within {} s (backends.codex.timeout_secs)",
                    self.cfg.timeout_secs
                )));
            }
            Some(Stop::Read(e)) => {
                kill_tree(&mut child).await;
                return Err(failed(format!("cannot read Codex output: {e}")));
            }
            None => {}
        }
        let exit = match tokio::time::timeout(EXIT_GRACE, child.wait()).await {
            Ok(Ok(status)) => status.to_string(),
            _ => {
                kill_tree(&mut child).await;
                "killed after closing its output".to_string()
            }
        };

        let Some(thread_id) = events.thread_id.clone() else {
            if let Some(err) = events.errors.last() {
                return Err(self.classified(err, codex, tail_text()));
            }
            return Err(failed(format!(
                "Codex ({version}) exited ({exit}) without starting a thread; its --json output may have changed"
            )));
        };
        let thread_dir = self
            .cfg
            .codex_home
            .join("generated_images")
            .join(&thread_id);
        let images = list_images(&thread_dir);
        let Some(chosen) = images.first() else {
            if let Some(err) = events.errors.last() {
                return Err(self.classified(err, codex, tail_text()));
            }
            return Err(failed(format!(
                "Codex produced no image. Its last reply: {}",
                events.last_message.as_deref().unwrap_or("(none)")
            )));
        };
        let bytes = std::fs::read(chosen)
            .map_err(|e| failed(format!("cannot read {}: {e}", chosen.display())))?;
        let png = to_png(bytes).map_err(|e| {
            failed(format!(
                "Codex's image {} isn't readable: {e}",
                chosen.display()
            ))
        })?;

        let mut warnings = Vec::new();
        if images.len() > 1 {
            warnings.push(format!(
                "Codex made {} images; kept the newest ({})",
                images.len(),
                chosen.display()
            ));
        }
        if let Some(err) = events.errors.last() {
            warnings.push(format!(
                "Codex reported an error after saving the image: {}",
                err.message
            ));
        }
        let mut details = BTreeMap::new();
        details.insert("codex_version".into(), json!(version));
        details.insert("agent_model".into(), json!(codex.model));
        if let Some(effort) = &codex.reasoning_effort {
            details.insert("reasoning_effort".into(), json!(effort));
        }
        details.insert("thread_id".into(), json!(thread_id));
        details.insert("source_path".into(), json!(chosen));
        if let Some(usage) = &events.usage {
            details.insert("usage".into(), usage.clone());
        }
        if images.len() > 1 {
            details.insert("all_images".into(), json!(images));
        }
        Ok(GeneratedImage {
            png,
            seed: None,
            warnings,
            details,
            ..Default::default()
        })
    }

    async fn unload(&self) {}

    fn loaded_model(&self) -> Option<String> {
        None
    }

    fn runs(&self) -> Runs {
        Runs::Cloud
    }

    fn provider(&self) -> Option<&'static str> {
        Some(PROVIDER)
    }
}

impl CodexBackend {
    fn classified(&self, err: &CodexError, codex: &CodexModel, tail: String) -> BackendError {
        let e = classify(err, &codex.model, tail);
        if matches!(e, BackendError::Unavailable { .. }) {
            self.forget_availability();
        }
        e
    }
}

enum Stop {
    Cancelled,
    TimedOut,
    Read(String),
}

/// Per-job folder under the work dir holding copies of the reference images. Removed on drop.
struct JobDir(PathBuf);

impl JobDir {
    fn create(work_dir: &Path) -> std::io::Result<Self> {
        let dir = work_dir.join(format!("job-{:016x}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir)?;
        Ok(Self(dir))
    }

    fn path(&self) -> &Path {
        &self.0
    }

    /// Write the bytes captured at submit, so later changes to the originals can't affect the job.
    fn write_references(&self, refs: &[Reference]) -> std::io::Result<Vec<PathBuf>> {
        refs.iter()
            .enumerate()
            .map(|(i, r)| {
                let ext = match r.format.as_str() {
                    "jpeg" | "jpg" => "jpg",
                    "webp" => "webp",
                    _ => "png",
                };
                let path = self.0.join(format!("reference-{}.{ext}", i + 1));
                std::fs::write(&path, &r.data)?;
                Ok(path)
            })
            .collect()
    }
}

impl Drop for JobDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Arguments for `codex exec`. The instruction goes on stdin, so no prompt argument follows.
pub fn exec_args(
    cfg: &CodexConfig,
    codex: &CodexModel,
    work_dir: &Path,
    references: &[PathBuf],
) -> Vec<String> {
    let mut args: Vec<String> = [
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "-C",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    args.push(work_dir.to_string_lossy().into_owned());
    args.push("-m".into());
    args.push(codex.model.clone());
    if let Some(effort) = &codex.reasoning_effort {
        args.push("-c".into());
        args.push(format!("model_reasoning_effort={effort}"));
    }
    args.extend(cfg.extra_args.iter().cloned());
    args.extend(codex.extra_args.iter().cloned());
    // `--image` takes several values, so the `=` form keeps each path to exactly one flag.
    args.extend(
        references
            .iter()
            .map(|r| format!("--image={}", r.display())),
    );
    args
}

/// The shape and size to ask Codex for (design A13): landscape 3:2, portrait 2:3 or square.
pub fn orientation(width: u32, height: u32) -> (&'static str, &'static str) {
    let ratio = f64::from(width.max(1)) / f64::from(height.max(1));
    if ratio > 1.2 {
        ("landscape", "1536x1024")
    } else if ratio < 1.0 / 1.2 {
        ("portrait", "1024x1536")
    } else {
        ("square", "1024x1024")
    }
}

/// The fixed wrapper around the user's prompt (design A14).
pub fn instruction(request: &ResolvedRequest) -> String {
    let (shape, size) = orientation(request.width, request.height);
    let rules = "Do not run shell commands, read or write files, browse, or use any other tool. \
When the image is done, reply with the single word DONE.";
    let mut text = match request.mode {
        Mode::Txt2img => format!(
            "Use your image generation tool exactly once to create the image described below as a \
{shape} image ({size}). {rules}\n\nImage description:\n{}",
            request.prompt
        ),
        Mode::Edit => format!(
            "The attached image(s) are references, in order. Use your image generation tool exactly once \
to edit them as instructed below, producing a {shape} image ({size}). Preserve everything the \
instruction doesn't ask to change. {rules}\n\nInstruction:\n{}",
            request.prompt
        ),
    };
    let avoid = request.negative_prompt.trim();
    if !avoid.is_empty() {
        text.push_str(&format!("\n\nAvoid: {avoid}"));
    }
    text
}

/// An error Codex reported in its event stream.
#[derive(Debug, Clone, PartialEq)]
pub struct CodexError {
    pub status: Option<u64>,
    pub message: String,
}

impl CodexError {
    /// Codex nests the API error as JSON inside the message string; unwrap it when it does.
    fn decode(raw: &str) -> Self {
        if let Ok(v) = serde_json::from_str::<Value>(raw) {
            let message = v["error"]["message"]
                .as_str()
                .or_else(|| v["message"].as_str())
                .unwrap_or(raw);
            return Self {
                status: v["status"].as_u64(),
                message: message.to_string(),
            };
        }
        Self {
            status: None,
            message: raw.to_string(),
        }
    }
}

/// What the JSONL stream said.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct RunEvents {
    pub thread_id: Option<String>,
    pub last_message: Option<String>,
    pub errors: Vec<CodexError>,
    pub usage: Option<Value>,
    pub completed: bool,
}

impl RunEvents {
    /// Record one line. Returns true when it started the thread (the moment generation begins).
    pub fn push_line(&mut self, line: &str) -> bool {
        let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
            return false;
        };
        match v["type"].as_str() {
            Some("thread.started") => {
                self.thread_id = v["thread_id"].as_str().map(str::to_string);
                return self.thread_id.is_some();
            }
            Some("item.completed") if v["item"]["type"] == "agent_message" => {
                self.last_message = v["item"]["text"].as_str().map(str::to_string);
            }
            Some("error") => {
                if let Some(m) = v["message"].as_str() {
                    self.add_error(CodexError::decode(m));
                }
            }
            Some("turn.failed") => {
                if let Some(m) = v["error"]["message"].as_str() {
                    self.add_error(CodexError::decode(m));
                }
            }
            Some("turn.completed") => {
                self.completed = true;
                if v["usage"].is_object() {
                    self.usage = Some(v["usage"].clone());
                }
            }
            _ => {}
        }
        false
    }

    /// `error` and `turn.failed` usually repeat the same message.
    fn add_error(&mut self, err: CodexError) {
        if self.errors.last() != Some(&err) {
            self.errors.push(err);
        }
    }
}

/// Map a Codex error to what the user should do (R15).
pub fn classify(err: &CodexError, agent_model: &str, stderr_tail: String) -> BackendError {
    let m = err.message.to_lowercase();
    if m.contains("not supported when using codex with a chatgpt account")
        || (err.status == Some(400) && m.contains("model") && m.contains("not supported"))
    {
        BackendError::Failed {
            message: format!(
                "Codex rejected agent model `{agent_model}`: {} Set a different `codex.model` for this model in models.toml \
(with a ChatGPT sign-in, gpt-6-astra and the gpt-5.6 models worked on Codex 0.153)",
                err.message
            ),
            stderr_tail,
        }
    } else if err.status == Some(401)
        || m.contains("not logged in")
        || m.contains("unauthorized")
        || m.contains("please log in")
    {
        BackendError::Unavailable {
            reason: format!("Codex is not signed in: {}", err.message),
            fix: "run `codex login`".into(),
        }
    } else if err.status == Some(429) || m.contains("usage limit") || m.contains("rate limit") {
        BackendError::Failed {
            message: format!("Codex usage limit reached: {}", err.message),
            stderr_tail,
        }
    } else {
        BackendError::Failed {
            message: format!("Codex failed: {}", err.message),
            stderr_tail,
        }
    }
}

/// Image files in `dir`, newest first (by modification time, then name).
pub fn list_images(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<(SystemTime, PathBuf)> = entries
        .filter_map(|e| {
            let e = e.ok()?;
            let path = e.path();
            let ext = path.extension()?.to_str()?.to_ascii_lowercase();
            if !IMAGE_EXTENSIONS.contains(&ext.as_str()) || !path.is_file() {
                return None;
            }
            let modified = e.metadata().and_then(|m| m.modified()).ok()?;
            Some((modified, path))
        })
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    found.into_iter().map(|(_, p)| p).collect()
}

/// PNG bytes as they are; anything else the image crate reads is re-encoded as PNG.
pub fn to_png(bytes: Vec<u8>) -> Result<Vec<u8>, String> {
    if bytes.starts_with(PNG_SIGNATURE) {
        return Ok(bytes);
    }
    let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> CodexConfig {
        CodexConfig {
            bin: PathBuf::from("codex"),
            codex_home: PathBuf::from("/home/u/.codex"),
            timeout_secs: 600,
            extra_args: vec!["--disable".into(), "browser_use".into()],
            work_dir: PathBuf::from("/cfg/.codex-work"),
        }
    }

    fn model(effort: Option<&str>) -> CodexModel {
        CodexModel {
            model: "gpt-5.6-terra".into(),
            reasoning_effort: effort.map(str::to_string),
            extra_args: vec!["--enable".into(), "x".into()],
        }
    }

    fn req(mode: Mode, w: u32, h: u32) -> ResolvedRequest {
        ResolvedRequest {
            model: "codex-image".into(),
            prompt: "a \"quoted\" crate\nsecond line".into(),
            negative_prompt: String::new(),
            width: w,
            height: h,
            steps: 1,
            cfg_scale: 1.0,
            seed: 5,
            sampler: None,
            warnings: vec![],
            mode,
            references: vec![],
            custom_sigmas: vec![],
            loras: vec![],
        }
    }

    #[test]
    fn args_in_order_with_images_last() {
        let refs = vec![
            PathBuf::from("/w/reference-1.png"),
            PathBuf::from("/w/my ref.jpg"),
        ];
        let args = exec_args(&cfg(), &model(Some("low")), Path::new("/w"), &refs);
        assert_eq!(
            args,
            vec![
                "exec",
                "--json",
                "--color",
                "never",
                "--skip-git-repo-check",
                "-s",
                "read-only",
                "-C",
                "/w",
                "-m",
                "gpt-5.6-terra",
                "-c",
                "model_reasoning_effort=low",
                "--disable",
                "browser_use",
                "--enable",
                "x",
                "--image=/w/reference-1.png",
                "--image=/w/my ref.jpg",
            ]
        );
        let no_effort = exec_args(&cfg(), &model(None), Path::new("/w"), &[]);
        assert!(
            !no_effort
                .iter()
                .any(|a| a.starts_with("model_reasoning_effort"))
        );
        assert!(!no_effort.iter().any(|a| a.starts_with("--image")));
    }

    #[test]
    fn orientation_thresholds() {
        assert_eq!(orientation(1024, 1024), ("square", "1024x1024"));
        assert_eq!(orientation(1100, 1000), ("square", "1024x1024"));
        assert_eq!(orientation(1536, 1024), ("landscape", "1536x1024"));
        assert_eq!(orientation(1344, 768), ("landscape", "1536x1024"));
        assert_eq!(orientation(832, 1216), ("portrait", "1024x1536"));
        assert_eq!(orientation(0, 0), ("square", "1024x1024"));
    }

    #[test]
    fn instruction_wraps_the_prompt_verbatim() {
        let t = instruction(&req(Mode::Txt2img, 832, 1216));
        assert!(t.contains("exactly once"));
        assert!(t.contains("portrait image (1024x1536)"));
        assert!(t.ends_with("Image description:\na \"quoted\" crate\nsecond line"));
        assert!(!t.contains("Avoid:"));

        let mut e = req(Mode::Edit, 1536, 1024);
        e.negative_prompt = "  text, watermark ".into();
        let t = instruction(&e);
        assert!(t.starts_with("The attached image(s) are references, in order."));
        assert!(t.contains("landscape image (1536x1024)"));
        assert!(t.contains("Instruction:\na \"quoted\" crate"));
        assert!(t.ends_with("\n\nAvoid: text, watermark"));
    }

    const OK_STREAM: &str = r#"{"type":"thread.started","thread_id":"01a0-thread"}
{"type":"turn.started"}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"DONE"}}
{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":5}}"#;

    #[test]
    fn parses_a_successful_stream() {
        let mut ev = RunEvents::default();
        let starts: Vec<bool> = OK_STREAM.lines().map(|l| ev.push_line(l)).collect();
        assert_eq!(starts, vec![true, false, false, false]);
        assert_eq!(ev.thread_id.as_deref(), Some("01a0-thread"));
        assert_eq!(ev.last_message.as_deref(), Some("DONE"));
        assert!(ev.completed);
        assert_eq!(ev.usage.unwrap()["input_tokens"], 100);
        assert!(ev.errors.is_empty());
        // Garbage and unknown events are ignored.
        let mut ev = RunEvents::default();
        assert!(!ev.push_line("not json"));
        assert!(!ev.push_line(r#"{"type":"item.started","item":{"type":"mcp_tool_call"}}"#));
        assert_eq!(ev, RunEvents::default());
    }

    #[test]
    fn nested_errors_are_decoded_and_deduplicated() {
        let inner = r#"{\"type\":\"error\",\"status\":400,\"error\":{\"type\":\"invalid_request_error\",\"message\":\"The 'gpt-6-luna' model is not supported when using Codex with a ChatGPT account.\"}}"#;
        let stream = format!(
            "{{\"type\":\"thread.started\",\"thread_id\":\"t\"}}\n{{\"type\":\"error\",\"message\":\"{inner}\"}}\n{{\"type\":\"turn.failed\",\"error\":{{\"message\":\"{inner}\"}}}}"
        );
        let mut ev = RunEvents::default();
        for l in stream.lines() {
            ev.push_line(l);
        }
        assert_eq!(ev.errors.len(), 1);
        assert_eq!(ev.errors[0].status, Some(400));
        assert!(ev.errors[0].message.starts_with("The 'gpt-6-luna' model"));
        assert_eq!(
            CodexError::decode("plain text"),
            CodexError {
                status: None,
                message: "plain text".into()
            }
        );
    }

    #[test]
    fn classification_tells_the_user_what_to_do() {
        let e = |status, msg: &str| CodexError {
            status,
            message: msg.into(),
        };
        let unsupported = classify(
            &e(
                Some(400),
                "The 'gpt-6-luna' model is not supported when using Codex with a ChatGPT account.",
            ),
            "gpt-6-luna",
            String::new(),
        );
        match unsupported {
            BackendError::Failed { message, .. } => {
                assert!(
                    message.contains("`gpt-6-luna`") && message.contains("codex.model"),
                    "{message}"
                )
            }
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            classify(&e(Some(401), "Unauthorized"), "m", String::new()),
            BackendError::Unavailable { fix, .. } if fix.contains("codex login")
        ));
        assert!(matches!(
            classify(&e(Some(429), "slow down"), "m", String::new()),
            BackendError::Failed { message, .. } if message.contains("usage limit")
        ));
        assert!(matches!(
            classify(&e(None, "You've hit your usage limit"), "m", String::new()),
            BackendError::Failed { message, .. } if message.contains("usage limit reached")
        ));
        assert!(matches!(
            classify(&e(None, "boom"), "m", "tail".into()),
            BackendError::Failed { message, stderr_tail } if message == "Codex failed: boom" && stderr_tail == "tail"
        ));
    }

    #[test]
    fn newest_image_first_and_non_images_ignored() {
        let dir = tempfile::tempdir().unwrap();
        assert!(list_images(&dir.path().join("missing")).is_empty());
        let a = dir.path().join("exec-a.png");
        std::fs::write(&a, b"a").unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"x").unwrap();
        std::thread::sleep(Duration::from_millis(30));
        let b = dir.path().join("exec-b.WEBP");
        std::fs::write(&b, b"b").unwrap();
        assert_eq!(list_images(dir.path()), vec![b, a]);
    }

    #[test]
    fn non_png_is_reencoded() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 3));
        let mut jpeg = Vec::new();
        img.write_to(&mut Cursor::new(&mut jpeg), image::ImageFormat::Jpeg)
            .unwrap();
        let png = to_png(jpeg).unwrap();
        assert!(png.starts_with(PNG_SIGNATURE));
        assert_eq!(crate::output::image_dimensions(&png), Some((4, 3)));
        let same = to_png(png.clone()).unwrap();
        assert_eq!(same, png);
        assert!(to_png(b"nope".to_vec()).is_err());
    }

    #[test]
    fn references_are_staged_from_captured_bytes_and_cleaned_up() {
        let work = tempfile::tempdir().unwrap();
        let refs = vec![
            Reference {
                path: PathBuf::from("/gone/original.png"),
                width: 1,
                height: 1,
                format: "png".into(),
                sha256: "00".into(),
                data: Arc::from(&b"png-bytes"[..]),
            },
            Reference {
                path: PathBuf::from("/gone/photo.jpg"),
                width: 1,
                height: 1,
                format: "jpeg".into(),
                sha256: "11".into(),
                data: Arc::from(&b"jpg-bytes"[..]),
            },
        ];
        let staged_dir;
        {
            let job = JobDir::create(work.path()).unwrap();
            staged_dir = job.path().to_path_buf();
            let staged = job.write_references(&refs).unwrap();
            assert_eq!(staged[0].file_name().unwrap(), "reference-1.png");
            assert_eq!(staged[1].file_name().unwrap(), "reference-2.jpg");
            assert_eq!(std::fs::read(&staged[1]).unwrap(), b"jpg-bytes");
        }
        assert!(!staged_dir.exists());
    }
}
