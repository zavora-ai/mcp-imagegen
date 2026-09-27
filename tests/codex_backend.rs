//! Codex backend against the fake Codex binary (design §13.6; R12, R14–R16).
#![cfg(all(feature = "mock", feature = "codex"))]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mcp_imagegen::backend::codex::{CodexBackend, PROVIDER};
use mcp_imagegen::backend::{
    Availability, BackendError, GeneratedImage, ImageBackend, ProgressSink, Runs,
};
use mcp_imagegen::config::CodexConfig;
use mcp_imagegen::input::Reference;
use mcp_imagegen::output::image_dimensions;
use mcp_imagegen::registry::{ModelRegistry, ModelSpec};
use mcp_imagegen::request::{Mode, ResolvedRequest};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

const REGISTRY: &str = r#"
[[models]]
id = "codex-image"
backend = "codex"
capabilities = ["txt2img", "edit"]
max_ref_images = 3
license = "OpenAI terms of use"
commercial_outputs = true
max_pixels = 4194304
size_multiple = 16
defaults = { width = 1024, height = 1024, steps = 1, cfg_scale = 1.0 }
codex = { model = "gpt-5.6-terra", reasoning_effort = "low" }
"#;

struct Setup {
    dir: tempfile::TempDir,
    backend: CodexBackend,
}

impl Setup {
    fn new(timeout_secs: u64) -> Self {
        Self::with_bin(
            PathBuf::from(env!("CARGO_BIN_EXE_fake-codex")),
            timeout_secs,
        )
    }

    fn with_bin(bin: PathBuf, timeout_secs: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let backend = CodexBackend::new(CodexConfig {
            bin,
            codex_home: dir.path().join("codex home"),
            timeout_secs,
            extra_args: vec![],
            work_dir: dir.path().join("work"),
        });
        Self { dir, backend }
    }

    fn home(&self) -> PathBuf {
        self.dir.path().join("codex home")
    }

    fn last_call(&self) -> Value {
        serde_json::from_slice(&std::fs::read(self.home().join("fake_last_call.json")).unwrap())
            .unwrap()
    }

    async fn run(&self, request: &ResolvedRequest) -> Result<GeneratedImage, BackendError> {
        self.backend
            .generate(
                &model(),
                &BTreeMap::new(),
                request,
                ProgressSink::noop(),
                CancellationToken::new(),
            )
            .await
    }
}

fn model() -> ModelSpec {
    ModelRegistry::from_toml(REGISTRY)
        .unwrap()
        .get("codex-image")
        .unwrap()
        .clone()
}

fn req(prompt: &str, width: u32, height: u32) -> ResolvedRequest {
    ResolvedRequest {
        model: "codex-image".into(),
        prompt: prompt.into(),
        negative_prompt: String::new(),
        width,
        height,
        steps: 1,
        cfg_scale: 1.0,
        seed: 1,
        sampler: None,
        warnings: vec![],
        mode: Mode::Txt2img,
        references: vec![],
        custom_sigmas: vec![],
        loras: vec![],
    }
}

fn failure(result: Result<GeneratedImage, BackendError>) -> String {
    match result {
        Err(BackendError::Failed { message, .. }) => message,
        other => panic!("expected a failure, got {other:?}"),
    }
}

fn job_dirs_left(work: &Path) -> usize {
    std::fs::read_dir(work).map(|d| d.count()).unwrap_or(0)
}

#[tokio::test]
async fn generates_and_records_provenance() {
    let s = Setup::new(60);
    let prompt = "a \"quoted\" crate\nwith a second line, ünïcode too";
    let image = s.run(&req(prompt, 832, 1216)).await.unwrap();

    assert_eq!(image_dimensions(&image.png), Some((32, 48)));
    assert_eq!(image.seed, None);
    assert!(image.warnings.is_empty(), "{:?}", image.warnings);
    let d = &image.details;
    assert!(d["thread_id"].as_str().unwrap().starts_with("fake-"));
    assert_eq!(d["codex_version"], "codex-cli 0.0.0-fake");
    assert_eq!(d["agent_model"], "gpt-5.6-terra");
    assert_eq!(d["reasoning_effort"], "low");
    assert_eq!(d["usage"]["input_tokens"], 1234);
    assert!(Path::new(d["source_path"].as_str().unwrap()).is_file());

    let call = s.last_call();
    let args: Vec<&str> = call["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    for expected in [
        "--json",
        "read-only",
        "gpt-5.6-terra",
        "model_reasoning_effort=low",
    ] {
        assert!(args.contains(&expected), "{args:?}");
    }
    // The prompt travels on stdin, unchanged, never as an argument.
    let sent = call["prompt"].as_str().unwrap();
    assert!(sent.contains("portrait image (1024x1536)"));
    assert!(sent.ends_with(prompt), "{sent}");
    assert!(!args.iter().any(|a| a.contains("quoted")));
    assert_eq!(job_dirs_left(&s.dir.path().join("work")), 0);
}

#[tokio::test]
async fn reports_progress_phases() {
    let s = Setup::new(60);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = {
        let seen = seen.clone();
        ProgressSink::new(move |p| seen.lock().unwrap().push(p.to_string()))
    };
    s.backend
        .generate(
            &model(),
            &BTreeMap::new(),
            &req("a crate", 1024, 1024),
            sink,
            CancellationToken::new(),
        )
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec!["starting codex", "generating"]);
}

#[tokio::test]
async fn edit_attaches_staged_references_in_order() {
    let s = Setup::new(60);
    let reference = |name: &str, format: &str| Reference {
        path: PathBuf::from(format!("/originals/{name}")),
        width: 8,
        height: 8,
        format: format.into(),
        sha256: "ab".repeat(32),
        data: Arc::from(&b"bytes"[..]),
    };
    let mut r = req("make it red", 1536, 1024);
    r.mode = Mode::Edit;
    r.references = vec![reference("a.png", "png"), reference("b.jpg", "jpeg")];
    let image = s.run(&r).await.unwrap();
    assert_eq!(image_dimensions(&image.png), Some((48, 32)));

    let call = s.last_call();
    let images: Vec<&str> = call["images"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_str().unwrap())
        .collect();
    assert_eq!(images.len(), 2);
    assert!(images[0].ends_with("reference-1.png"), "{images:?}");
    assert!(images[1].ends_with("reference-2.jpg"), "{images:?}");
    assert_eq!(call["images_exist"], true);
    assert!(
        call["prompt"]
            .as_str()
            .unwrap()
            .starts_with("The attached image(s) are references, in order.")
    );
    // Staged copies are removed afterwards.
    assert!(!Path::new(images[0]).exists());
}

#[tokio::test]
async fn keeps_the_newest_of_several_images() {
    let s = Setup::new(60);
    let image = s.run(&req("[fake:two] crates", 1024, 1024)).await.unwrap();
    assert_eq!(image_dimensions(&image.png), Some((44, 44)));
    assert!(
        image.warnings[0].contains("made 2 images"),
        "{:?}",
        image.warnings
    );
    assert_eq!(image.details["all_images"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn no_image_reports_codexs_reply() {
    let s = Setup::new(60);
    let msg = failure(s.run(&req("[fake:none] nope", 1024, 1024)).await);
    assert!(
        msg.contains("produced no image") && msg.contains("I can't create that image."),
        "{msg}"
    );
}

#[tokio::test]
async fn unsupported_agent_model_points_at_models_toml() {
    let s = Setup::new(60);
    let msg = failure(s.run(&req("[fake:unsupported] x", 1024, 1024)).await);
    assert!(
        msg.contains("`gpt-5.6-terra`") && msg.contains("codex.model"),
        "{msg}"
    );
}

#[tokio::test]
async fn usage_limit_is_named() {
    let s = Setup::new(60);
    let msg = failure(s.run(&req("[fake:usage] x", 1024, 1024)).await);
    assert!(msg.contains("usage limit reached"), "{msg}");
}

#[tokio::test]
async fn missing_thread_mentions_the_codex_version() {
    let s = Setup::new(60);
    let msg = failure(s.run(&req("[fake:no-thread] x", 1024, 1024)).await);
    assert!(
        msg.contains("0.0.0-fake") && msg.contains("without starting a thread"),
        "{msg}"
    );
}

#[tokio::test]
async fn image_saved_before_a_failed_turn_still_counts() {
    let s = Setup::new(60);
    let image = s
        .run(&req("[fake:fail-after-image] x", 1024, 1024))
        .await
        .unwrap();
    assert_eq!(image_dimensions(&image.png), Some((40, 40)));
    assert!(
        image
            .warnings
            .iter()
            .any(|w| w.contains("after saving the image")),
        "{:?}",
        image.warnings
    );
}

#[tokio::test]
async fn cancel_kills_codex_promptly() {
    let s = Setup::new(120);
    let cancel = CancellationToken::new();
    let started = Instant::now();
    let (m, files, r) = (
        model(),
        BTreeMap::new(),
        req("[fake:sleep] slow", 1024, 1024),
    );
    let run = s
        .backend
        .generate(&m, &files, &r, ProgressSink::noop(), cancel.clone());
    let trigger = async {
        tokio::time::sleep(Duration::from_millis(700)).await;
        cancel.cancel();
    };
    let (result, ()) = tokio::join!(run, trigger);
    assert!(matches!(result, Err(BackendError::Cancelled)), "{result:?}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(job_dirs_left(&s.dir.path().join("work")), 0);
}

#[tokio::test]
async fn timeout_kills_codex_and_says_so() {
    let s = Setup::new(1);
    let started = Instant::now();
    let msg = failure(s.run(&req("[fake:sleep] slow", 1024, 1024)).await);
    assert!(msg.contains("didn't finish within 1 s"), "{msg}");
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn availability_follows_the_sign_in() {
    let s = Setup::new(60);
    assert_eq!(s.backend.availability().await, Availability::Ready);
    assert_eq!(s.backend.runs(), Runs::Cloud);
    assert_eq!(s.backend.provider(), Some(PROVIDER));
    assert_eq!(s.backend.loaded_model(), None);

    let out = Setup::new(60);
    std::fs::create_dir_all(out.home()).unwrap();
    std::fs::write(out.home().join("fake_logged_out"), b"").unwrap();
    match out.backend.availability().await {
        Availability::Unavailable { reason, fix } => {
            assert!(reason.contains("not signed in"), "{reason}");
            assert!(fix.contains("codex login"), "{fix}");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn missing_binary_is_unavailable() {
    let s = Setup::with_bin(PathBuf::from("no-such-codex-binary-xyz"), 60);
    assert!(matches!(
        s.backend.availability().await,
        Availability::Unavailable { reason, .. } if reason.contains("not found")
    ));
    assert!(matches!(
        s.run(&req("x", 1024, 1024)).await,
        Err(BackendError::Unavailable { .. })
    ));
}
