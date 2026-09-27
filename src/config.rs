//! Server configuration (design §3): paths, output roots, limits and backend settings.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::ImageGenError;

pub const CONFIG_EXAMPLE: &str = include_str!("../config.example.toml");
pub const MODELS_EXAMPLE: &str = include_str!("../models.example.toml");

/// Environment variable overriding the config directory.
pub const CONFIG_DIR_ENV: &str = "MCP_IMAGEGEN_CONFIG_DIR";

#[derive(Debug, Clone)]
pub struct Config {
    pub config_dir: PathBuf,
    pub default_output_dir: PathBuf,
    pub allowed_output_roots: Vec<PathBuf>,
    pub idle_unload_secs: u64,
    pub max_queue: usize,
    pub job_ttl_secs: u64,
    pub memory_headroom_mb: u64,
    /// Edit inputs may only be read from here (defaults to the output roots).
    pub allowed_input_roots: Vec<PathBuf>,
    pub max_input_bytes: u64,
    pub max_input_side: u32,
    pub hf_cache_dir: PathBuf,
    pub sdcpp: SdcppConfig,
    pub mflux: MfluxConfig,
    pub codex: CodexConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SdcppConfig {
    pub server_bin: PathBuf,
    pub extra_args: Vec<String>,
    pub startup_timeout_secs: u64,
    /// Passed as `--lora-model-dir`. sd-server defaults to `.` and rescans it recursively on every
    /// capabilities call, which walks the whole disk when launched from `/`.
    pub lora_dir: PathBuf,
    /// Working directory for the child, so nothing depends on the MCP client's cwd.
    pub work_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MfluxConfig {
    /// Directory with mflux-generate-* entry points; `None` searches PATH.
    pub bin_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CodexConfig {
    /// The Codex CLI; a bare name is looked up on PATH (with PATHEXT on Windows).
    pub bin: PathBuf,
    /// Where Codex keeps its state and saves images (`generated_images/<thread>/`).
    pub codex_home: PathBuf,
    /// A run longer than this is killed and reported as failed.
    pub timeout_secs: u64,
    /// Added to every `codex exec` call, before per-model arguments.
    pub extra_args: Vec<String>,
    /// Working directory for `codex exec -C` (read-only sandbox, so nothing is written there).
    pub work_dir: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    default_output_dir: Option<String>,
    allowed_output_roots: Option<Vec<String>>,
    idle_unload_secs: Option<u64>,
    max_queue: Option<usize>,
    job_ttl_secs: Option<u64>,
    memory_headroom_mb: Option<u64>,
    allowed_input_roots: Option<Vec<String>>,
    max_input_bytes: Option<u64>,
    max_input_side: Option<u32>,
    hf_cache_dir: Option<String>,
    #[serde(default)]
    backends: RawBackends,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBackends {
    sdcpp: Option<RawSdcpp>,
    mflux: Option<RawMflux>,
    codex: Option<RawCodex>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCodex {
    bin: Option<String>,
    codex_home: Option<String>,
    timeout_secs: Option<u64>,
    #[serde(default)]
    extra_args: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSdcpp {
    server_bin: Option<String>,
    #[serde(default)]
    extra_args: Vec<String>,
    startup_timeout_secs: Option<u64>,
    lora_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMflux {
    bin_dir: Option<String>,
}

impl Config {
    /// Resolve the config directory: `$MCP_IMAGEGEN_CONFIG_DIR`, else the platform config dir:
    /// `%APPDATA%\mcp-imagegen` on Windows, `$XDG_CONFIG_HOME/mcp-imagegen` or `~/.config/mcp-imagegen` elsewhere.
    pub fn default_dir() -> PathBuf {
        if let Some(dir) = non_empty_env(CONFIG_DIR_ENV) {
            return expand_tilde(&dir);
        }
        if cfg!(windows)
            && let Some(appdata) = non_empty_env("APPDATA")
        {
            return PathBuf::from(appdata).join("mcp-imagegen");
        }
        match non_empty_env("XDG_CONFIG_HOME") {
            Some(xdg) => expand_tilde(&xdg).join("mcp-imagegen"),
            None => home_dir().join(".config").join("mcp-imagegen"),
        }
    }

    /// Load `config.toml` from `dir`, writing the example config and model registry first if absent.
    pub fn load_or_init(dir: &Path) -> Result<Self, ImageGenError> {
        std::fs::create_dir_all(dir).map_err(|e| ImageGenError::config(dir, e))?;
        let config_path = dir.join("config.toml");
        if !config_path.exists() {
            std::fs::write(&config_path, CONFIG_EXAMPLE)
                .map_err(|e| ImageGenError::config(&config_path, e))?;
        }
        let models_path = dir.join("models.toml");
        if !models_path.exists() {
            std::fs::write(&models_path, MODELS_EXAMPLE)
                .map_err(|e| ImageGenError::config(&models_path, e))?;
        }
        let text = std::fs::read_to_string(&config_path)
            .map_err(|e| ImageGenError::config(&config_path, e))?;
        Self::from_toml(dir, &text).map_err(|msg| ImageGenError::config(&config_path, msg))
    }

    /// Parse config text; `dir` is recorded as the config directory.
    pub fn from_toml(dir: &Path, text: &str) -> Result<Self, String> {
        let raw: RawConfig = toml::from_str(text).map_err(|e| e.to_string())?;

        let default_output_dir = expand_tilde(
            raw.default_output_dir
                .as_deref()
                .unwrap_or("~/Pictures/mcp-imagegen"),
        );
        let allowed_output_roots: Vec<PathBuf> = match raw.allowed_output_roots {
            Some(roots) if !roots.is_empty() => roots.iter().map(|r| expand_tilde(r)).collect(),
            _ => vec![default_output_dir.clone()],
        };
        if !allowed_output_roots
            .iter()
            .any(|root| default_output_dir.starts_with(root))
        {
            return Err(format!(
                "default_output_dir {} is not inside any allowed_output_roots entry",
                default_output_dir.display()
            ));
        }

        let allowed_input_roots = match raw.allowed_input_roots {
            Some(roots) if !roots.is_empty() => roots.iter().map(|r| expand_tilde(r)).collect(),
            _ => allowed_output_roots.clone(),
        };

        let max_queue = raw.max_queue.unwrap_or(4);
        if max_queue == 0 {
            return Err("max_queue must be at least 1".into());
        }

        let raw_sdcpp = raw.backends.sdcpp.unwrap_or(RawSdcpp {
            server_bin: None,
            extra_args: Vec::new(),
            startup_timeout_secs: None,
            lora_dir: None,
        });
        let sdcpp = SdcppConfig {
            server_bin: expand_tilde(raw_sdcpp.server_bin.as_deref().unwrap_or("sd-server")),
            extra_args: raw_sdcpp.extra_args,
            startup_timeout_secs: raw_sdcpp.startup_timeout_secs.unwrap_or(180),
            lora_dir: raw_sdcpp
                .lora_dir
                .map(|d| expand_tilde(&d))
                .unwrap_or_else(|| dir.join("loras")),
            work_dir: dir.join(".sdcpp"),
        };
        let raw_codex = raw.backends.codex.unwrap_or_default();
        let codex_timeout = raw_codex.timeout_secs.unwrap_or(600);
        if codex_timeout == 0 {
            return Err("backends.codex.timeout_secs must be at least 1".into());
        }
        let codex = CodexConfig {
            bin: expand_tilde(
                raw_codex
                    .bin
                    .as_deref()
                    .filter(|b| !b.trim().is_empty())
                    .unwrap_or("codex"),
            ),
            codex_home: raw_codex
                .codex_home
                .filter(|d| !d.trim().is_empty())
                .map(|d| expand_tilde(&d))
                .unwrap_or_else(default_codex_home),
            timeout_secs: codex_timeout,
            extra_args: raw_codex.extra_args,
            work_dir: dir.join(".codex-work"),
        };
        let mflux = MfluxConfig {
            bin_dir: raw
                .backends
                .mflux
                .and_then(|m| m.bin_dir)
                .filter(|d| !d.is_empty())
                .map(|d| expand_tilde(&d)),
        };

        Ok(Self {
            config_dir: dir.to_path_buf(),
            default_output_dir,
            allowed_output_roots,
            idle_unload_secs: raw.idle_unload_secs.unwrap_or(300),
            max_queue,
            job_ttl_secs: raw.job_ttl_secs.unwrap_or(3600),
            memory_headroom_mb: raw.memory_headroom_mb.unwrap_or(2048),
            allowed_input_roots,
            max_input_bytes: raw.max_input_bytes.unwrap_or(32 * 1024 * 1024),
            max_input_side: raw.max_input_side.unwrap_or(4096),
            hf_cache_dir: raw
                .hf_cache_dir
                .map(|d| expand_tilde(&d))
                .unwrap_or_else(default_hf_cache),
            sdcpp,
            mflux,
            codex,
        })
    }

    pub fn models_path(&self) -> PathBuf {
        self.config_dir.join("models.toml")
    }
}

/// `$HF_HUB_CACHE`, then `$HF_HOME/hub`, then `~/.cache/huggingface/hub`.
pub fn default_hf_cache() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|d| !d.is_empty()) {
        return expand_tilde(&dir.to_string_lossy());
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|d| !d.is_empty()) {
        return expand_tilde(&home.to_string_lossy()).join("hub");
    }
    home_dir().join(".cache").join("huggingface").join("hub")
}

/// `$CODEX_HOME`, else `~/.codex` (what the Codex CLI itself uses).
pub fn default_codex_home() -> PathBuf {
    match non_empty_env("CODEX_HOME") {
        Some(dir) => expand_tilde(&dir),
        None => home_dir().join(".codex"),
    }
}

fn non_empty_env(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.trim().is_empty())
}

/// The user's home directory: `$HOME`, else `%USERPROFILE%`, else `%HOMEDRIVE%%HOMEPATH%`.
pub fn home_dir() -> PathBuf {
    if let Some(home) = non_empty_env("HOME").or_else(|| non_empty_env("USERPROFILE")) {
        return PathBuf::from(home);
    }
    match (non_empty_env("HOMEDRIVE"), non_empty_env("HOMEPATH")) {
        (Some(drive), Some(path)) => PathBuf::from(format!("{drive}{path}")),
        _ => std::env::temp_dir(),
    }
}

/// Expand a leading `~`, `~/` or `~\\` to the home directory.
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" {
        home_dir()
    } else if let Some(rest) = path.strip_prefix("~/").or_else(|| path.strip_prefix("~\\")) {
        home_dir().join(rest)
    } else {
        PathBuf::from(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses() {
        let cfg = Config::from_toml(Path::new("/cfg"), CONFIG_EXAMPLE).unwrap();
        assert_eq!(cfg.max_queue, 4);
        assert_eq!(cfg.idle_unload_secs, 300);
        assert_eq!(cfg.sdcpp.server_bin, PathBuf::from("sd-server"));
        assert!(cfg.default_output_dir.ends_with("Pictures/mcp-imagegen"));
        assert_eq!(cfg.models_path(), PathBuf::from("/cfg/models.toml"));
        assert_eq!(cfg.sdcpp.lora_dir, PathBuf::from("/cfg/loras"));
        assert_eq!(cfg.sdcpp.work_dir, PathBuf::from("/cfg/.sdcpp"));
    }

    #[test]
    fn empty_config_uses_defaults() {
        let cfg = Config::from_toml(Path::new("/cfg"), "").unwrap();
        assert_eq!(
            cfg.allowed_output_roots,
            vec![cfg.default_output_dir.clone()]
        );
        assert_eq!(cfg.job_ttl_secs, 3600);
        assert_eq!(cfg.mflux.bin_dir, None);
        assert_eq!(cfg.allowed_input_roots, cfg.allowed_output_roots);
        assert_eq!(cfg.max_input_side, 4096);
    }

    #[test]
    fn default_output_dir_must_be_inside_a_root() {
        let err = Config::from_toml(
            Path::new("/cfg"),
            "default_output_dir = \"/a/out\"\nallowed_output_roots = [\"/b\"]\n",
        )
        .unwrap_err();
        assert!(err.contains("not inside any allowed_output_roots"), "{err}");
    }

    #[test]
    fn unknown_field_is_an_error_with_location() {
        let err = Config::from_toml(Path::new("/cfg"), "max_qeue = 3\n").unwrap_err();
        assert!(err.contains("max_qeue"), "{err}");
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn codex_section_defaults_and_overrides() {
        let cfg = Config::from_toml(Path::new("/cfg"), "").unwrap();
        assert_eq!(cfg.codex.bin, PathBuf::from("codex"));
        assert_eq!(cfg.codex.timeout_secs, 600);
        assert_eq!(cfg.codex.codex_home, default_codex_home());
        assert_eq!(cfg.codex.work_dir, PathBuf::from("/cfg/.codex-work"));

        let cfg = Config::from_toml(
            Path::new("/cfg"),
            "[backends.codex]\nbin = \"/opt/codex\"\ncodex_home = \"/data/codex\"\ntimeout_secs = 90\nextra_args = [\"--disable\", \"browser_use\"]\n",
        )
        .unwrap();
        assert_eq!(cfg.codex.bin, PathBuf::from("/opt/codex"));
        assert_eq!(cfg.codex.codex_home, PathBuf::from("/data/codex"));
        assert_eq!(cfg.codex.timeout_secs, 90);
        assert_eq!(cfg.codex.extra_args, vec!["--disable", "browser_use"]);
        assert!(
            Config::from_toml(Path::new("/cfg"), "[backends.codex]\ntimeout_secs = 0\n").is_err()
        );
    }

    #[test]
    fn zero_queue_rejected() {
        assert!(Config::from_toml(Path::new("/cfg"), "max_queue = 0").is_err());
    }

    #[test]
    fn tilde_expansion() {
        let home = home_dir();
        assert_eq!(expand_tilde("~"), home);
        assert_eq!(expand_tilde("~/x/y"), home.join("x/y"));
        assert_eq!(expand_tilde("/abs"), PathBuf::from("/abs"));
        assert_eq!(expand_tilde("~other"), PathBuf::from("~other"));
    }

    #[test]
    fn load_or_init_writes_examples() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config::load_or_init(dir.path()).unwrap();
        assert!(dir.path().join("config.toml").exists());
        assert!(cfg.models_path().exists());
        // Second load reads the existing files.
        std::fs::write(dir.path().join("config.toml"), "max_queue = 2\n").unwrap();
        assert_eq!(Config::load_or_init(dir.path()).unwrap().max_queue, 2);
    }
}
