//! Live test against the real Codex CLI (design §13.6). Spends ChatGPT/OpenAI quota, so it's ignored by default.
//!
//! Run by hand, signed in to Codex:
//!   cargo test --test codex_live -- --ignored --nocapture
//! Optional: `MCP_IMAGEGEN_CODEX_MODEL` (default gpt-5.6-terra), `MCP_IMAGEGEN_CODEX_ARGS` (extra args, space-separated).
#![cfg(feature = "codex")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use mcp_imagegen::backend::codex::CodexBackend;
use mcp_imagegen::backend::{Availability, ImageBackend, ProgressSink};
use mcp_imagegen::config::{CodexConfig, default_codex_home};
use mcp_imagegen::output::image_dimensions;
use mcp_imagegen::registry::ModelRegistry;
use mcp_imagegen::request::{Mode, ResolvedRequest};
use tokio_util::sync::CancellationToken;

#[tokio::test]
#[ignore = "spends Codex quota; run by hand"]
async fn real_codex_generates_one_image() {
    let agent =
        std::env::var("MCP_IMAGEGEN_CODEX_MODEL").unwrap_or_else(|_| "gpt-5.6-terra".into());
    let extra: Vec<String> = std::env::var("MCP_IMAGEGEN_CODEX_ARGS")
        .map(|a| a.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    let work = tempfile::tempdir().unwrap();
    let backend = CodexBackend::new(CodexConfig {
        bin: PathBuf::from("codex"),
        codex_home: default_codex_home(),
        timeout_secs: 600,
        extra_args: extra,
        work_dir: work.path().to_path_buf(),
    });
    assert_eq!(
        backend.availability().await,
        Availability::Ready,
        "sign in with `codex login` first"
    );

    let model = ModelRegistry::from_toml(&format!(
        r#"
[[models]]
id = "codex-live"
backend = "codex"
capabilities = ["txt2img"]
license = "OpenAI terms of use"
commercial_outputs = true
max_pixels = 4194304
size_multiple = 16
defaults = {{ width = 1024, height = 1024, steps = 1, cfg_scale = 1.0 }}
codex = {{ model = "{agent}", reasoning_effort = "low" }}
"#
    ))
    .unwrap()
    .get("codex-live")
    .unwrap()
    .clone();
    let request = ResolvedRequest {
        model: model.id.clone(),
        prompt: "A single weathered wooden supply crate with iron corner brackets, three-quarter view, plain light grey background, game concept art".into(),
        negative_prompt: String::new(),
        width: 1536,
        height: 1024,
        steps: 1,
        cfg_scale: 1.0,
        seed: 1,
        sampler: None,
        warnings: vec![],
        mode: Mode::Txt2img,
        references: vec![],
        custom_sigmas: vec![],
        loras: vec![],
    };
    let started = Instant::now();
    let image = backend
        .generate(
            &model,
            &BTreeMap::new(),
            &request,
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .expect("Codex generation");
    let (w, h) = image_dimensions(&image.png).expect("a readable PNG");
    let out = std::env::temp_dir().join("mcp-imagegen-codex-live.png");
    std::fs::write(&out, &image.png).unwrap();
    println!(
        "{w}x{h} in {:.1} s -> {}\nwarnings: {:?}\ndetails: {}",
        started.elapsed().as_secs_f64(),
        out.display(),
        image.warnings,
        serde_json::to_string_pretty(&image.details).unwrap()
    );
    assert!(w > h, "asked for landscape, got {w}x{h}");
    assert!(image.details["thread_id"].is_string());
    assert!(image.seed.is_none());
}

#[tokio::test]
#[ignore = "spends Codex quota; run by hand"]
async fn real_codex_edits_an_image() {
    use std::sync::Arc;
    // A plain source drawn here: a red square on white.
    let mut img = image::RgbImage::from_pixel(512, 512, image::Rgb([255, 255, 255]));
    for y in 156..356 {
        for x in 156..356 {
            img.put_pixel(x, y, image::Rgb([220, 30, 30]));
        }
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();

    let work = tempfile::tempdir().unwrap();
    let backend = CodexBackend::new(CodexConfig {
        bin: PathBuf::from("codex"),
        codex_home: default_codex_home(),
        timeout_secs: 600,
        extra_args: vec![],
        work_dir: work.path().to_path_buf(),
    });
    let model = ModelRegistry::from_toml(
        r#"
[[models]]
id = "codex-live"
backend = "codex"
capabilities = ["txt2img", "edit"]
max_ref_images = 3
license = "OpenAI terms of use"
max_pixels = 4194304
size_multiple = 16
defaults = { width = 1024, height = 1024, steps = 1, cfg_scale = 1.0 }
codex = { model = "gpt-5.6-terra", reasoning_effort = "low" }
"#,
    )
    .unwrap()
    .get("codex-live")
    .unwrap()
    .clone();
    let request = ResolvedRequest {
        model: model.id.clone(),
        prompt: "Change the red square to a blue circle of the same size. Keep the plain white background.".into(),
        negative_prompt: String::new(),
        width: 512,
        height: 512,
        steps: 1,
        cfg_scale: 1.0,
        seed: 1,
        sampler: None,
        warnings: vec![],
        mode: Mode::Edit,
        references: vec![mcp_imagegen::input::Reference {
            path: PathBuf::from("drawn-in-test.png"),
            width: 512,
            height: 512,
            format: "png".into(),
            sha256: "00".repeat(32),
            data: Arc::from(png.as_slice()),
        }],
        custom_sigmas: vec![],
        loras: vec![],
    };
    let started = Instant::now();
    let image = backend
        .generate(
            &model,
            &BTreeMap::new(),
            &request,
            ProgressSink::noop(),
            CancellationToken::new(),
        )
        .await
        .expect("Codex edit");
    let out = std::env::temp_dir().join("mcp-imagegen-codex-live-edit.png");
    std::fs::write(&out, &image.png).unwrap();
    let decoded = image::load_from_memory(&image.png).unwrap().to_rgb8();
    let centre = decoded.get_pixel(decoded.width() / 2, decoded.height() / 2);
    println!(
        "{}x{} in {:.1} s -> {} (centre pixel {centre:?})\ndetails: {}",
        decoded.width(),
        decoded.height(),
        started.elapsed().as_secs_f64(),
        out.display(),
        serde_json::to_string_pretty(&image.details).unwrap()
    );
    // The centre should now be blue-dominant.
    assert!(centre[2] > centre[0], "centre pixel {centre:?} isn't blue");
}
