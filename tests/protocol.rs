//! Protocol tests (design §8): spawn the real binary over stdio with the mock backend and
//! drive it as MCP 2025-11-25 (legacy) and 2026-07-28 (Tasks) clients.
#![cfg(feature = "mock")]

use std::path::{Path, PathBuf};
use std::time::Duration;

use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ClientCapabilities, ClientInfo,
    GetTaskParams, Implementation, ProtocolVersion, TaskPayload,
};
use rmcp::service::{RoleClient, RunningService};
use rmcp::transport::TokioChildProcess;
use rmcp::{ClientHandler, ServiceExt};
use serde_json::{Value, json};

#[derive(Clone)]
struct LegacyClient;

impl ClientHandler for LegacyClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::default(),
            Implementation::new("legacy-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2025_11_25;
        info
    }
}

#[derive(Clone)]
struct TaskClient;

impl ClientHandler for TaskClient {
    fn get_info(&self) -> ClientInfo {
        let mut info = ClientInfo::new(
            ClientCapabilities::builder().enable_tasks().build(),
            Implementation::new("task-test", "1"),
        );
        info.protocol_version = ProtocolVersion::V_2026_07_28;
        info
    }
}

struct Env {
    _dir: tempfile::TempDir,
    cfg: PathBuf,
    out: PathBuf,
}

/// Temp config with a mock model and an sd.cpp model whose weights don't exist.
fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("cfg");
    let out = dir.path().join("out");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        cfg.join("config.toml"),
        format!(
            "default_output_dir = {out:?}\nallowed_output_roots = [{out:?}]\nmemory_headroom_mb = 0\nhf_cache_dir = {cache:?}\n\n[backends.codex]\nbin = {codex:?}\ncodex_home = {codex_home:?}\n",
            out = out.to_str().unwrap(),
            cache = dir.path().join("hf").to_str().unwrap(),
            codex = env!("CARGO_BIN_EXE_fake-codex"),
            codex_home = dir.path().join("codex-home").to_str().unwrap(),
        ),
    )
    .unwrap();
    std::fs::write(
        cfg.join("models.toml"),
        r#"
[[models]]
id = "mock-model"
backend = "mock"
capabilities = ["txt2img"]
license = "MIT"
commercial_weights = true
est_memory_mb = 1
max_pixels = 4194304
size_multiple = 8
defaults = { width = 64, height = 64, steps = 4, cfg_scale = 1.0 }

[[models]]
id = "mock-editor"
backend = "mock"
capabilities = ["txt2img", "edit"]
max_ref_images = 2
license = "MIT"
commercial_weights = true
est_memory_mb = 1
max_pixels = 4194304
size_multiple = 8
defaults = { width = 64, height = 64, steps = 3, cfg_scale = 1.0 }

[[models]]
id = "needs-weights"
backend = "sdcpp"
capabilities = ["txt2img"]
license = "Apache-2.0"
commercial_weights = true
est_memory_mb = 1
max_pixels = 4194304
size_multiple = 16
defaults = { width = 512, height = 512, steps = 8, cfg_scale = 1.0 }
[models.files]
diffusion_model = { hf_repo = "org/model-GGUF", hf_file = "model-Q8_0.gguf", size_mb = 7000 }

[[models]]
id = "cloud-fake"
backend = "codex"
capabilities = ["txt2img", "edit"]
max_ref_images = 3
license = "OpenAI terms of use"
commercial_outputs = true
max_pixels = 4194304
size_multiple = 16
defaults = { width = 1024, height = 1024, steps = 1, cfg_scale = 1.0 }
codex = { model = "gpt-5.6-terra", reasoning_effort = "low" }
"#,
    )
    .unwrap();
    Env {
        _dir: dir,
        cfg,
        out,
    }
}

async fn spawn<C: ClientHandler>(client: C, env: &Env) -> RunningService<RoleClient, C> {
    let mut cmd = tokio::process::Command::new(env!("CARGO_BIN_EXE_mcp-imagegen"));
    cmd.env("MCP_IMAGEGEN_CONFIG_DIR", &env.cfg)
        .env("MCP_IMAGEGEN_MOCK_STEP_MS", "40")
        .env("RUST_LOG", "warn")
        .current_dir(std::env::temp_dir());
    let transport = TokioChildProcess::new(cmd).unwrap();
    client.serve(transport).await.unwrap()
}

fn args(v: Value) -> CallToolRequestParams {
    CallToolRequestParams::new("generate_image").with_arguments(v.as_object().unwrap().clone())
}

fn call(name: &'static str, v: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(name).with_arguments(v.as_object().unwrap().clone())
}

fn sc(result: &CallToolResult) -> &Value {
    result
        .structured_content
        .as_ref()
        .expect("structured content")
}

fn pngs(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "png")
        })
        .count()
}

#[tokio::test]
async fn legacy_client_lists_tools_and_generates() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let manifest = adk_mcp_sdk::ServerManifest::from_toml(mcp_imagegen::MANIFEST_TOML).unwrap();
    let mut want: Vec<_> = manifest.tools.iter().map(|t| t.name.clone()).collect();
    want.sort();
    let mut got: Vec<_> = client
        .list_tools(None)
        .await
        .unwrap()
        .tools
        .iter()
        .map(|t| t.name.to_string())
        .collect();
    got.sort();
    assert_eq!(got, want);

    let result = client
        .call_tool(args(json!({
            "prompt": "a wooden crate",
            "model": "mock-model",
            "seed": 99,
            "return_preview": true
        })))
        .await
        .unwrap();
    let v = sc(&result);
    assert_eq!(v["status"], "done");
    assert_eq!(v["seed"], 99);
    assert_eq!(v["license"], "MIT");
    let path = PathBuf::from(v["path"].as_str().unwrap());
    assert!(path.starts_with(&env.out) && path.exists());
    assert!(PathBuf::from(v["sidecar_path"].as_str().unwrap()).exists());
    assert!(
        result.content.iter().any(
            |c| matches!(c, rmcp::model::ContentBlock::Image(i) if i.mime_type == "image/png")
        )
    );

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn task_client_gets_a_task_with_status_updates() {
    let env = env();
    let client = spawn(TaskClient, &env).await;

    let response = client
        .call_tool_once(args(
            json!({"prompt": "task crate", "model": "mock-model", "steps": 25}),
        ))
        .await
        .unwrap();
    let created = match response {
        CallToolResponse::Task(created) => created,
        other => panic!("expected a task, got {other:?}"),
    };
    assert_eq!(created.task.ttl_ms, Some(3_600_000));

    let mut messages = Vec::new();
    let result = loop {
        let task = client
            .peer()
            .get_task(GetTaskParams::new(created.task.task_id.clone()))
            .await
            .unwrap()
            .task;
        if let Some(m) = &task.task.status_message {
            messages.push(m.clone());
        }
        if task.status().is_terminal() {
            match task.payload {
                TaskPayload::Completed { result } => break result,
                other => panic!("unexpected terminal payload: {other:?}"),
            }
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    };
    assert_eq!(result["structuredContent"]["status"], "done");
    assert!(
        messages.iter().any(|m| m.starts_with("sampling ")),
        "status messages seen: {messages:?}"
    );
    assert_eq!(pngs(&env.out), 1);

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn no_wait_then_poll_get_job() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let queued = client
        .call_tool(args(
            json!({"prompt": "later", "model": "mock-model", "wait": false}),
        ))
        .await
        .unwrap();
    let job_id = sc(&queued)["job_id"].as_str().unwrap().to_string();

    let done = loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        let status = sc(&r)["status"].as_str().unwrap().to_string();
        if status == "done" {
            break r;
        }
        assert!(status == "queued" || status == "running", "{status}");
        tokio::time::sleep(Duration::from_millis(50)).await;
    };
    assert!(
        sc(&done)["result"]["path"]
            .as_str()
            .unwrap()
            .ends_with(".png")
    );

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn cancel_job_mid_run_leaves_no_file() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let queued = client
        .call_tool(args(
            json!({"prompt": "slow", "model": "mock-model", "steps": 100, "wait": false}),
        ))
        .await
        .unwrap();
    let job_id = sc(&queued)["job_id"].as_str().unwrap().to_string();
    tokio::time::sleep(Duration::from_millis(200)).await;
    client
        .call_tool(call("cancel_job", json!({"job_id": job_id})))
        .await
        .unwrap();

    let status = loop {
        let r = client
            .call_tool(call("get_job", json!({"job_id": job_id})))
            .await
            .unwrap();
        let s = sc(&r)["status"].as_str().unwrap().to_string();
        if s != "running" {
            break s;
        }
        tokio::time::sleep(Duration::from_millis(30)).await;
    };
    assert_eq!(status, "cancelled");
    assert_eq!(pngs(&env.out), 0);

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn list_models_reports_missing_weights_without_loading() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let r = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    let models = sc(&r)["models"].as_array().unwrap().clone();
    let needs = models.iter().find(|m| m["id"] == "needs-weights").unwrap();
    assert_eq!(needs["status"], "missing_files");
    assert_eq!(needs["loaded"], false);
    assert_eq!(needs["download_mb"], 7000);
    assert_eq!(
        needs["missing_files"][0]["fetch"],
        "hf download org/model-GGUF model-Q8_0.gguf"
    );
    let mock = models.iter().find(|m| m["id"] == "mock-model").unwrap();
    assert_eq!(mock["status"], "ready");
    assert_eq!(mock["loaded"], false);

    // Generating with missing weights fails with the fetch command, not a crash.
    let err = client
        .call_tool(args(json!({"prompt": "x", "model": "needs-weights"})))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("hf download org/model-GGUF"),
        "{err}"
    );

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn edit_image_round_trip_and_rejections() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    // A source image to edit (made by the mock model itself).
    let src = client
        .call_tool(args(
            json!({"prompt": "plain crate", "model": "mock-model", "seed": 1}),
        ))
        .await
        .unwrap();
    let src_path = sc(&src)["path"].as_str().unwrap().to_string();
    let src_bytes = std::fs::read(&src_path).unwrap();

    let edited = client
        .call_tool(call(
            "edit_image",
            json!({"images": [src_path], "prompt": "paint it red", "model": "mock-editor", "seed": 2}),
        ))
        .await
        .unwrap();
    let v = sc(&edited);
    assert_eq!(v["status"], "done");
    assert_eq!(v["mode"], "edit");
    assert!(v["path"].as_str().unwrap().contains("paint-it-red-edit-2"));
    // Output size defaults to the source's (the mock renders at the requested 64x64).
    assert_eq!(
        (v["width"].as_u64(), v["height"].as_u64()),
        (Some(64), Some(64))
    );
    let refs = v["references"].as_array().unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0]["sha256"].as_str().unwrap().len(), 64);
    let sidecar: Value =
        serde_json::from_slice(&std::fs::read(v["sidecar_path"].as_str().unwrap()).unwrap())
            .unwrap();
    assert_eq!(sidecar["references"][0]["sha256"], refs[0]["sha256"]);
    // The input is untouched.
    assert_eq!(std::fs::read(&src_path).unwrap(), src_bytes);

    // Too many images.
    let err = client
        .call_tool(call(
            "edit_image",
            json!({"images": [src_path, src_path, src_path], "prompt": "x", "model": "mock-editor"}),
        ))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("takes 1-2 images, got 3"), "{err}");

    // Model without edit support.
    let err = client
        .call_tool(call(
            "edit_image",
            json!({"images": [src_path], "prompt": "x", "model": "mock-model"}),
        ))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("does not support editing"),
        "{err}"
    );

    // Input outside the allowed roots.
    let outside = tempfile::tempdir().unwrap();
    let foreign = outside.path().join("foreign.png");
    std::fs::copy(&src_path, &foreign).unwrap();
    let err = client
        .call_tool(call(
            "edit_image",
            json!({"images": [foreign.to_str().unwrap()], "prompt": "x", "model": "mock-editor"}),
        ))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("outside the allowed input roots"),
        "{err}"
    );

    // list_models reports edit readiness.
    let r = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    let models = sc(&r)["models"].as_array().unwrap().clone();
    let editor = models.iter().find(|m| m["id"] == "mock-editor").unwrap();
    assert_eq!(editor["edit"]["status"], "ready");
    assert_eq!(editor["edit"]["max_images"], 2);
    let plain = models.iter().find(|m| m["id"] == "mock-model").unwrap();
    assert_eq!(plain["edit"]["status"], "unsupported");

    client.cancel().await.unwrap();
}

#[tokio::test]
async fn codex_model_is_listed_as_cloud_and_generates() {
    let env = env();
    let client = spawn(LegacyClient, &env).await;

    let listed = client
        .call_tool(call("list_models", json!({})))
        .await
        .unwrap();
    let models = sc(&listed)["models"].as_array().unwrap().clone();
    let cloud = models.iter().find(|m| m["id"] == "cloud-fake").unwrap();
    assert_eq!(cloud["runs"], "cloud");
    assert_eq!(cloud["status"], "ready");
    assert_eq!(cloud["provider"], "OpenAI, through Codex CLI");
    assert_eq!(cloud["commercial_outputs"], true);
    assert_eq!(cloud["agent_model"], "gpt-5.6-terra");
    assert!(cloud.get("commercial_weights").is_none());
    assert_eq!(cloud["edit"]["status"], "ready");
    let local = models.iter().find(|m| m["id"] == "mock-model").unwrap();
    assert_eq!(local["runs"], "local");
    assert_eq!(local["commercial_weights"], true);

    let done = client
        .call_tool(args(json!({
            "prompt": "a crate",
            "model": "cloud-fake",
            "width": 1536,
            "height": 1024,
            "seed": 3
        })))
        .await
        .unwrap();
    let v = sc(&done);
    assert_eq!(v["status"], "done", "{v}");
    assert_eq!(v["runs"], "cloud");
    assert!(v["seed"].is_null());
    assert_eq!(
        (v["width"].as_u64(), v["height"].as_u64()),
        (Some(48), Some(32))
    );
    let warnings = v["warnings"].as_array().unwrap();
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().unwrap().contains("`seed` is ignored"))
    );
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().unwrap().contains("returned 48x32"))
    );
    assert!(
        v["backend_details"]["thread_id"]
            .as_str()
            .unwrap()
            .starts_with("fake-")
    );
    assert!(std::path::Path::new(v["path"].as_str().unwrap()).exists());

    client.cancel().await.unwrap();
}
