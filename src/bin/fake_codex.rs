//! Stand-in for the Codex CLI in tests (built only with the `mock` feature, never installed).
//!
//! Understands `--version`, `login status` and `exec`. `exec` reads the instruction from stdin,
//! records its arguments to `$CODEX_HOME/fake_last_call.json`, prints a JSONL stream shaped like
//! Codex 0.153's, and writes a small PNG into `$CODEX_HOME/generated_images/<thread>/`.
//!
//! Behaviour is picked by a marker in the prompt, so parallel tests can't affect each other:
//! `[fake:two]`, `[fake:none]`, `[fake:unsupported]`, `[fake:usage]`, `[fake:sleep]`,
//! `[fake:no-thread]`, `[fake:fail-after-image]`. A file named `fake_logged_out` in `CODEX_HOME`
//! makes `login status` report no sign-in.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--version") => println!("codex-cli 0.0.0-fake"),
        Some("login") if args.get(1).map(String::as_str) == Some("status") => {
            if codex_home().join("fake_logged_out").exists() {
                eprintln!("Not logged in");
                std::process::exit(1);
            }
            println!("Logged in using ChatGPT");
        }
        Some("exec") => exec(&args[1..]),
        _ => {
            eprintln!("fake-codex: unsupported arguments {args:?}");
            std::process::exit(2);
        }
    }
}

fn codex_home() -> PathBuf {
    PathBuf::from(std::env::var_os("CODEX_HOME").expect("CODEX_HOME must be set"))
}

fn value_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn emit(value: serde_json::Value) {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}").unwrap();
    out.flush().unwrap();
}

/// Scaled-down stand-ins for Codex's real sizes, so tests see "actual differs from requested".
fn size_for(prompt: &str) -> (u32, u32) {
    if prompt.contains("1536x1024") {
        (48, 32)
    } else if prompt.contains("1024x1536") {
        (32, 48)
    } else {
        (40, 40)
    }
}

fn write_png(path: &Path, (w, h): (u32, u32), shade: u8) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([shade, 0, 0])))
        .save_with_format(path, image::ImageFormat::Png)
        .unwrap();
}

fn api_error(status: u16, message: &str) -> String {
    json!({"type": "error", "status": status, "error": {"type": "invalid_request_error", "message": message}})
        .to_string()
}

fn fail(message: String) -> ! {
    emit(json!({"type": "error", "message": message}));
    emit(json!({"type": "turn.failed", "error": {"message": message}}));
    std::process::exit(1);
}

fn exec(args: &[String]) {
    let home = codex_home();
    let mut prompt = String::new();
    std::io::stdin().read_to_string(&mut prompt).unwrap();
    let images: Vec<&str> = args
        .iter()
        .filter_map(|a| a.strip_prefix("--image="))
        .collect();
    let model = value_after(args, "-m").unwrap_or("default");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::write(
        home.join("fake_last_call.json"),
        serde_json::to_vec_pretty(&json!({
            "args": args,
            "prompt": prompt,
            "images": images,
            "images_exist": images.iter().all(|p| Path::new(p).is_file()),
            "cwd": std::env::current_dir().ok(),
        }))
        .unwrap(),
    )
    .unwrap();

    let mode = [
        "two",
        "none",
        "unsupported",
        "usage",
        "sleep",
        "no-thread",
        "fail-after-image",
    ]
    .into_iter()
    .find(|m| prompt.contains(&format!("[fake:{m}]")))
    .unwrap_or("ok");
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let thread = format!("fake-{}-{nanos}", std::process::id());
    let dir = home.join("generated_images").join(&thread);

    if mode != "no-thread" {
        emit(json!({"type": "thread.started", "thread_id": thread}));
    }
    emit(json!({"type": "turn.started"}));
    match mode {
        "unsupported" => fail(api_error(
            400,
            &format!(
                "The '{model}' model is not supported when using Codex with a ChatGPT account."
            ),
        )),
        "usage" => fail(api_error(429, "You've hit your usage limit.")),
        "none" | "no-thread" => {
            emit(
                json!({"type": "item.completed", "item": {"id": "item_0", "type": "agent_message", "text": "I can't create that image."}}),
            );
            emit(
                json!({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 3}}),
            );
            return;
        }
        "sleep" => std::thread::sleep(Duration::from_secs(30)),
        _ => {}
    }
    write_png(&dir.join("exec-1.png"), size_for(&prompt), 200);
    if mode == "two" {
        std::thread::sleep(Duration::from_millis(50));
        write_png(&dir.join("exec-2.png"), (44, 44), 100);
    }
    if mode == "fail-after-image" {
        fail("stream disconnected before completion".into());
    }
    emit(
        json!({"type": "item.completed", "item": {"id": "item_1", "type": "agent_message", "text": "DONE"}}),
    );
    emit(
        json!({"type": "turn.completed", "usage": {"input_tokens": 1234, "cached_input_tokens": 1000, "output_tokens": 5}}),
    );
}
