//! Model registry loaded from `models.toml` (design §3; R2, R3, R8).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::config::expand_tilde;
use crate::error::ImageGenError;
use crate::schedule::SigmaSchedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Txt2img,
    Img2img,
    Edit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefaults {
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
}

/// A weight file: an explicit `path`, or `hf_repo` + `hf_file` in the HF cache.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRef {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hf_file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size_mb: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MfluxModel {
    pub command: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantize: Option<u8>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

/// Settings for a model served through the Codex CLI (`backend = "codex"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodexModel {
    /// Agent model passed to `codex exec -m` (the image itself comes from Codex's built-in tool).
    pub model: String,
    /// Passed as `-c model_reasoning_effort=<value>` (e.g. "low"); Codex's default when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    pub backend: String,
    pub capabilities: Vec<Capability>,
    pub license: String,
    /// Whether local weights may be used commercially. Not meaningful for cloud models.
    #[serde(default)]
    pub commercial_weights: bool,
    /// Whether generated images may be used commercially, when the licence speaks about outputs
    /// (e.g. a cloud service's terms). Omitted when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commercial_outputs: Option<bool>,
    /// Estimated resident memory; 0 for cloud models.
    #[serde(default)]
    pub est_memory_mb: u64,
    pub max_pixels: u64,
    pub size_multiple: u32,
    pub defaults: ModelDefaults,
    /// Role (e.g. `diffusion_model`, `llm`, `vae`) → file.
    #[serde(default)]
    pub files: BTreeMap<String, FileRef>,
    /// Backend-specific launch arguments for this model.
    #[serde(default)]
    pub extra_args: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mflux: Option<MfluxModel>,
    /// Codex settings; required when `backend = "codex"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codex: Option<CodexModel>,
    /// Extra backend request fields for this model, e.g. `{ cache_mode = "easycache" }` for sd.cpp.
    /// Core fields (prompt, size, seed, steps, cfg) always come from the request.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub backend_options: BTreeMap<String, serde_json::Value>,
    /// Most reference images an edit may pass (0 = editing unsupported).
    #[serde(default)]
    pub max_ref_images: u32,
    /// Files only editing needs (e.g. `llm_vision`). Loaded whenever present.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub edit_files: BTreeMap<String, FileRef>,
    /// LoRAs applied to every job on this model.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraRef>,
    /// Fixed sigma schedule (e.g. a distilled turbo model); overrides steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma_schedule: Option<SigmaSchedule>,
}

/// A LoRA applied on top of the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraRef {
    pub file: FileRef,
    #[serde(default = "one")]
    pub multiplier: f32,
}

fn one() -> f32 {
    1.0
}

/// A resolved LoRA, ready for the backend.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LoraUse {
    pub path: PathBuf,
    pub multiplier: f32,
}

/// A weight file that isn't on disk, with how to fetch it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MissingFile {
    pub role: String,
    pub hf_repo: Option<String>,
    pub hf_file: Option<String>,
    pub path: Option<PathBuf>,
    pub size_mb: Option<u64>,
}

impl MissingFile {
    pub fn fetch_hint(&self) -> String {
        match (&self.hf_repo, &self.hf_file) {
            (Some(repo), Some(file)) => format!("hf download {repo} {file}"),
            _ => match &self.path {
                Some(p) => format!("place the file at {}", p.display()),
                None => format!("configure a path for `{}`", self.role),
            },
        }
    }
}

impl ModelSpec {
    pub fn supports(&self, cap: Capability) -> bool {
        self.capabilities.contains(&cap)
    }

    /// Resolve every base file to a local path, or list what's missing.
    pub fn resolve_files(
        &self,
        hf_cache: &Path,
    ) -> Result<BTreeMap<String, PathBuf>, Vec<MissingFile>> {
        resolve_all(&self.files, hf_cache)
    }

    /// Resolve the edit-only files, or list what's missing.
    pub fn resolve_edit_files(
        &self,
        hf_cache: &Path,
    ) -> Result<BTreeMap<String, PathBuf>, Vec<MissingFile>> {
        resolve_all(&self.edit_files, hf_cache)
    }

    /// Resolve the model's LoRAs, or list what's missing.
    pub fn resolve_loras(&self, hf_cache: &Path) -> Result<Vec<LoraUse>, Vec<MissingFile>> {
        let files: BTreeMap<String, FileRef> = self
            .loras
            .iter()
            .enumerate()
            .map(|(i, l)| (format!("lora{i}"), l.file.clone()))
            .collect();
        let found = resolve_all(&files, hf_cache)?;
        Ok(self
            .loras
            .iter()
            .enumerate()
            .map(|(i, l)| LoraUse {
                path: found[&format!("lora{i}")].clone(),
                multiplier: l.multiplier,
            })
            .collect())
    }

    /// Everything a job needs, missing files from base weights and LoRAs together.
    pub fn missing_for_generation(&self, hf_cache: &Path) -> Vec<MissingFile> {
        let mut missing = self.resolve_files(hf_cache).err().unwrap_or_default();
        missing.extend(self.resolve_loras(hf_cache).err().unwrap_or_default());
        missing
    }

    /// Edit-only files that are present right now (loaded opportunistically).
    pub fn present_edit_files(&self, hf_cache: &Path) -> BTreeMap<String, PathBuf> {
        self.edit_files
            .iter()
            .filter_map(|(role, f)| resolve_file(f, hf_cache).map(|p| (role.clone(), p)))
            .collect()
    }

    fn validate(&self) -> Result<(), String> {
        if self.id.trim().is_empty() {
            return Err("model id is empty".into());
        }
        if self.size_multiple == 0 {
            return Err(format!("model `{}`: size_multiple must be > 0", self.id));
        }
        if self.capabilities.is_empty() {
            return Err(format!("model `{}`: capabilities is empty", self.id));
        }
        if self.supports(Capability::Edit) != (self.max_ref_images > 0) {
            return Err(format!(
                "model `{}`: the `edit` capability and max_ref_images >= 1 go together",
                self.id
            ));
        }
        if (self.backend == "codex") != self.codex.is_some() {
            return Err(format!(
                "model `{}`: `backend = \"codex\"` and a `codex` section go together",
                self.id
            ));
        }
        if let Some(codex) = &self.codex
            && codex.model.trim().is_empty()
        {
            return Err(format!("model `{}`: codex.model is empty", self.id));
        }
        if let Some(schedule) = &self.sigma_schedule {
            schedule
                .validate()
                .map_err(|e| format!("model `{}`: {e}", self.id))?;
        }
        let lora_files: Vec<(String, &FileRef)> = self
            .loras
            .iter()
            .enumerate()
            .map(|(i, l)| (format!("loras[{i}]"), &l.file))
            .collect();
        for (role, f) in self
            .files
            .iter()
            .chain(&self.edit_files)
            .map(|(r, f)| (r.clone(), f))
            .chain(lora_files)
        {
            let has_path = f.path.is_some();
            let has_hf = f.hf_repo.is_some() && f.hf_file.is_some();
            if has_path == has_hf {
                return Err(format!(
                    "model `{}` file `{role}`: set either `path` or both `hf_repo` and `hf_file`",
                    self.id
                ));
            }
        }
        Ok(())
    }
}

fn resolve_all(
    files: &BTreeMap<String, FileRef>,
    hf_cache: &Path,
) -> Result<BTreeMap<String, PathBuf>, Vec<MissingFile>> {
    let mut found = BTreeMap::new();
    let mut missing = Vec::new();
    for (role, file) in files {
        match resolve_file(file, hf_cache) {
            Some(path) => {
                found.insert(role.clone(), path);
            }
            None => missing.push(MissingFile {
                role: role.clone(),
                hf_repo: file.hf_repo.clone(),
                hf_file: file.hf_file.clone(),
                path: file.path.as_deref().map(expand_tilde),
                size_mb: file.size_mb,
            }),
        }
    }
    if missing.is_empty() {
        Ok(found)
    } else {
        Err(missing)
    }
}

/// Explicit path if it exists; otherwise the newest HF snapshot containing the file.
pub fn resolve_file(file: &FileRef, hf_cache: &Path) -> Option<PathBuf> {
    if let Some(p) = &file.path {
        let p = expand_tilde(p);
        return p.is_file().then_some(p);
    }
    let (repo, name) = (file.hf_repo.as_deref()?, file.hf_file.as_deref()?);
    let snapshots = hf_cache
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path().join(name);
            // is_file follows the snapshot symlink into blobs/.
            if !path.is_file() {
                return None;
            }
            let modified = entry.metadata().and_then(|m| m.modified()).ok()?;
            Some((modified, path))
        })
        .collect();
    candidates.sort_by_key(|c| std::cmp::Reverse(c.0));
    candidates.into_iter().next().map(|(_, p)| p)
}

#[derive(Debug, Clone, Default)]
pub struct ModelRegistry {
    models: Vec<ModelSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RegistryFile {
    #[serde(default)]
    models: Vec<ModelSpec>,
}

impl ModelRegistry {
    pub fn load(path: &Path) -> Result<Self, ImageGenError> {
        let text = std::fs::read_to_string(path).map_err(|e| ImageGenError::config(path, e))?;
        Self::from_toml(&text).map_err(|msg| ImageGenError::config(path, msg))
    }

    pub fn from_toml(text: &str) -> Result<Self, String> {
        let file: RegistryFile = toml::from_str(text).map_err(|e| e.to_string())?;
        let mut seen = HashSet::new();
        for model in &file.models {
            model.validate()?;
            if !seen.insert(model.id.clone()) {
                return Err(format!("duplicate model id `{}`", model.id));
            }
        }
        Ok(Self {
            models: file.models,
        })
    }

    pub fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    pub fn get(&self, id: &str) -> Result<&ModelSpec, ImageGenError> {
        self.models
            .iter()
            .find(|m| m.id == id)
            .ok_or_else(|| ImageGenError::UnknownModel {
                id: id.to_string(),
                available: self.ids(),
            })
    }

    pub fn ids(&self) -> Vec<String> {
        self.models.iter().map(|m| m.id.clone()).collect()
    }

    /// The model used when a request names none: the first one in the file.
    pub fn default_model(&self) -> Option<&ModelSpec> {
        self.models.first()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    fn snapshot(cache: &Path, repo: &str, rev: &str, file: &str) -> PathBuf {
        let p = cache
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(rev)
            .join(file);
        touch(&p);
        p
    }

    #[test]
    fn example_registry_parses() {
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        assert_eq!(
            reg.ids(),
            vec![
                "qwen-image-2.1-turbo",
                "qwen-image-2.1",
                "qwen-image-2.1-hq",
                "codex-image",
                "codex-image-astra"
            ]
        );
        let q = reg.get("qwen-image-2.1").unwrap();
        assert!(!q.commercial_weights);
        assert_eq!(q.size_multiple, 32);
        assert_eq!(q.extra_args, vec!["--fa"]);
        assert_eq!(q.files.len(), 3);
        assert_eq!(reg.default_model().unwrap().id, "qwen-image-2.1-turbo");
        assert_eq!(q.backend_options["cache_mode"], "easycache");
        assert!(
            reg.get("qwen-image-2.1-hq")
                .unwrap()
                .backend_options
                .is_empty()
        );
    }

    #[test]
    fn resolves_from_hf_cache_and_reports_missing() {
        let cache = tempfile::tempdir().unwrap();
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let q = reg.get("qwen-image-2.1").unwrap();

        snapshot(
            cache.path(),
            "leejet/Qwen-Image-2.1-GGUF",
            "abc",
            "qwen_image_2.1-Q8_0.gguf",
        );
        let missing = q.resolve_files(cache.path()).unwrap_err();
        let roles: Vec<_> = missing.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["llm", "vae"]);
        assert_eq!(
            missing[1].fetch_hint(),
            "hf download Comfy-Org/Qwen-Image-2.1 vae/qwen_image_2.1_vae_bf16.safetensors"
        );

        snapshot(
            cache.path(),
            "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            "r1",
            "Qwen3VL-8B-Instruct-Q4_K_M.gguf",
        );
        snapshot(
            cache.path(),
            "Comfy-Org/Qwen-Image-2.1",
            "r1",
            "vae/qwen_image_2.1_vae_bf16.safetensors",
        );
        let found = q.resolve_files(cache.path()).unwrap();
        assert!(found["vae"].ends_with("vae/qwen_image_2.1_vae_bf16.safetensors"));
    }

    #[test]
    fn newest_snapshot_wins() {
        let cache = tempfile::tempdir().unwrap();
        let old = snapshot(cache.path(), "o/r", "old", "f.gguf");
        std::thread::sleep(std::time::Duration::from_millis(20));
        let new = snapshot(cache.path(), "o/r", "new", "f.gguf");
        let file = FileRef {
            path: None,
            hf_repo: Some("o/r".into()),
            hf_file: Some("f.gguf".into()),
            size_mb: None,
        };
        let got = resolve_file(&file, cache.path()).unwrap();
        assert_eq!(got, new);
        assert_ne!(got, old);
    }

    #[test]
    fn explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("w.gguf");
        let file = FileRef {
            path: Some(p.to_string_lossy().into()),
            hf_repo: None,
            hf_file: None,
            size_mb: None,
        };
        assert_eq!(resolve_file(&file, dir.path()), None);
        touch(&p);
        assert_eq!(resolve_file(&file, dir.path()), Some(p));
    }

    #[test]
    fn malformed_registry_names_the_field() {
        let err = ModelRegistry::from_toml("[[models]]\nid = \"x\"\nbackend = 3\n").unwrap_err();
        assert!(err.contains("backend"), "{err}");
        assert!(err.contains("line 3"), "{err}");
    }

    #[test]
    fn duplicate_ids_and_bad_file_refs_rejected() {
        let one = r#"
[[models]]
id = "a"
backend = "sdcpp"
capabilities = ["txt2img"]
license = "MIT"
commercial_weights = true
est_memory_mb = 1
max_pixels = 1
size_multiple = 8
defaults = { width = 8, height = 8, steps = 1, cfg_scale = 1.0 }
"#;
        let dup = format!("{one}{one}");
        assert!(
            ModelRegistry::from_toml(&dup)
                .unwrap_err()
                .contains("duplicate model id")
        );
        let bad = format!("{one}files.vae = {{ hf_repo = \"o/r\" }}\n");
        assert!(
            ModelRegistry::from_toml(&bad)
                .unwrap_err()
                .contains("either `path`")
        );
    }

    #[test]
    fn edit_fields_parse_and_resolve() {
        let cache = tempfile::tempdir().unwrap();
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let q = reg.get("qwen-image-2.1").unwrap();
        assert!(q.supports(Capability::Edit));
        assert_eq!(q.max_ref_images, 3);
        let missing = q.resolve_edit_files(cache.path()).unwrap_err();
        assert_eq!(missing[0].role, "llm_vision");
        assert!(
            missing[0]
                .fetch_hint()
                .contains("mmproj-Qwen3VL-8B-Instruct-F16.gguf")
        );
        assert!(q.present_edit_files(cache.path()).is_empty());
        snapshot(
            cache.path(),
            "Qwen/Qwen3-VL-8B-Instruct-GGUF",
            "r",
            "mmproj-Qwen3VL-8B-Instruct-F16.gguf",
        );
        assert_eq!(q.present_edit_files(cache.path()).len(), 1);
        assert!(q.resolve_edit_files(cache.path()).is_ok());
    }

    #[test]
    fn edit_capability_requires_ref_limit() {
        let base = r#"
[[models]]
id = "a"
backend = "sdcpp"
license = "MIT"
commercial_weights = true
est_memory_mb = 1
max_pixels = 1
size_multiple = 8
defaults = { width = 8, height = 8, steps = 1, cfg_scale = 1.0 }
"#;
        let no_limit = format!("{base}capabilities = [\"txt2img\", \"edit\"]\n");
        assert!(
            ModelRegistry::from_toml(&no_limit)
                .unwrap_err()
                .contains("go together")
        );
        let no_cap = format!("{base}capabilities = [\"txt2img\"]\nmax_ref_images = 2\n");
        assert!(
            ModelRegistry::from_toml(&no_cap)
                .unwrap_err()
                .contains("go together")
        );
        let ok = format!("{base}capabilities = [\"txt2img\", \"edit\"]\nmax_ref_images = 2\n");
        assert!(ModelRegistry::from_toml(&ok).is_ok());
    }

    #[test]
    fn turbo_loras_and_schedule() {
        let cache = tempfile::tempdir().unwrap();
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let t = reg.get("qwen-image-2.1-turbo").unwrap();
        assert_eq!(t.sigma_schedule.as_ref().unwrap().steps(), 6);
        assert_eq!(t.loras.len(), 1);
        let missing = t.resolve_loras(cache.path()).unwrap_err();
        assert!(
            missing[0]
                .fetch_hint()
                .contains("viggle-turbo-v0.2.1-6step-lora-r128")
        );
        assert!(
            t.missing_for_generation(cache.path())
                .iter()
                .any(|m| m.role == "lora0")
        );
        snapshot(
            cache.path(),
            "Viggle/Qwen-Image-2.1-viggle-turbo",
            "r",
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors",
        );
        let loras = t.resolve_loras(cache.path()).unwrap();
        assert_eq!(loras[0].multiplier, 1.0);
        assert!(
            reg.get("qwen-image-2.1")
                .unwrap()
                .resolve_loras(cache.path())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn bad_schedule_rejected() {
        let bad = MODELS_EXAMPLE.replace("raw = [1.0, 0.9375", "raw = [0.5, 0.9375");
        assert!(
            ModelRegistry::from_toml(&bad)
                .unwrap_err()
                .contains("strictly decreasing")
        );
    }

    #[test]
    fn codex_models_parse_and_validate() {
        let base = r#"
[[models]]
id = "cloud"
backend = "codex"
capabilities = ["txt2img", "edit"]
license = "OpenAI terms"
commercial_outputs = true
max_pixels = 4194304
size_multiple = 16
max_ref_images = 3
defaults = { width = 1024, height = 1024, steps = 1, cfg_scale = 1.0 }
"#;
        let ok =
            format!("{base}codex = {{ model = \"gpt-5.6-terra\", reasoning_effort = \"low\" }}\n");
        let reg = ModelRegistry::from_toml(&ok).unwrap();
        let m = reg.get("cloud").unwrap();
        assert_eq!(m.codex.as_ref().unwrap().model, "gpt-5.6-terra");
        assert_eq!(m.commercial_outputs, Some(true));
        assert!(!m.commercial_weights);
        assert_eq!(m.est_memory_mb, 0);

        let missing = ModelRegistry::from_toml(base).unwrap_err();
        assert!(missing.contains("go together"), "{missing}");
        let empty = format!("{base}codex = {{ model = \" \" }}\n");
        assert!(
            ModelRegistry::from_toml(&empty)
                .unwrap_err()
                .contains("codex.model is empty")
        );
        let wrong_backend = ok.replace("backend = \"codex\"", "backend = \"sdcpp\"");
        assert!(
            ModelRegistry::from_toml(&wrong_backend)
                .unwrap_err()
                .contains("go together")
        );
    }

    #[test]
    fn unknown_model_lists_available() {
        let reg = ModelRegistry::from_toml(MODELS_EXAMPLE).unwrap();
        let msg = reg.get("nope").unwrap_err().to_string();
        assert!(
            msg.contains("qwen-image-2.1-turbo, qwen-image-2.1, qwen-image-2.1-hq"),
            "{msg}"
        );
    }
}
