//! `generate_image` input, validation and default resolution (design §7 step 1; R1, R7).

use rmcp::schemars;
use serde::{Deserialize, Serialize};

use crate::error::ImageGenError;
use crate::input::{InputPolicy, Reference};
use crate::registry::{Capability, LoraUse, ModelRegistry, ModelSpec};

/// Prompts longer than this get a warning: most text encoders truncate around here.
pub const LONG_PROMPT_CHARS: usize = 1500;
pub const MAX_STEPS: u32 = 100;
pub const MAX_CFG: f32 = 30.0;

#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
pub struct GenerateRequest {
    /// What to draw.
    pub prompt: String,
    /// What to avoid (ignored by models that run without CFG).
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// Model id from list_models. Defaults to the first model in models.toml.
    #[serde(default)]
    pub model: Option<String>,
    /// Width in pixels; must be a multiple of the model's size_multiple.
    #[serde(default)]
    pub width: Option<u32>,
    /// Height in pixels; must be a multiple of the model's size_multiple.
    #[serde(default)]
    pub height: Option<u32>,
    /// Sampling steps (1-100). Defaults per model.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Classifier-free guidance scale (0-30). Defaults per model.
    #[serde(default)]
    pub cfg_scale: Option<f32>,
    /// Seed for reproducibility. Random when omitted; always reported back.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Sampler name understood by the backend (e.g. "euler"). Defaults per model.
    #[serde(default)]
    pub sampler: Option<String>,
    /// Exact output file (.png). Must be inside an allowed output root.
    #[serde(default)]
    pub output_path: Option<String>,
    /// Output directory; the file name is derived from the prompt and seed.
    #[serde(default)]
    pub output_dir: Option<String>,
    /// Include a small inline preview image in the result.
    #[serde(default)]
    pub return_preview: bool,
    /// Wait for the image (default). false returns a job_id to poll with get_job.
    #[serde(default = "default_true")]
    pub wait: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Deserialize, schemars::JsonSchema)]
pub struct EditRequest {
    /// Images to edit, in order (local paths inside the allowed input roots). The first sets the default output size.
    pub images: Vec<String>,
    /// The edit instruction goes in `prompt`; all generate_image options apply.
    #[serde(flatten)]
    pub params: GenerateRequest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Txt2img,
    Edit,
}

/// Fully resolved parameters handed to a backend. Everything is concrete.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ResolvedRequest {
    pub model: String,
    pub prompt: String,
    pub negative_prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg_scale: f32,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampler: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    pub mode: Mode,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub references: Vec<Reference>,
    /// Shifted sigma schedule (terminal 0 included) when the model fixes one.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub custom_sigmas: Vec<f32>,
    /// LoRAs applied to this job (filled by the server from the registry).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub loras: Vec<LoraUse>,
}

impl GenerateRequest {
    /// Validate against the registry and fill in model defaults.
    pub fn resolve<'r>(
        &self,
        registry: &'r ModelRegistry,
    ) -> Result<(ResolvedRequest, &'r ModelSpec), ImageGenError> {
        let model = self.pick_model(registry, |m| m.supports(Capability::Txt2img))?;
        if !model.supports(Capability::Txt2img) {
            return Err(ImageGenError::invalid(
                "model",
                format!("`{}` does not support text-to-image", model.id),
            ));
        }
        let size = (model.defaults.width, model.defaults.height);
        let resolved = self.resolve_core(model, size, Mode::Txt2img, Vec::new())?;
        Ok((resolved, model))
    }

    /// Named model, else the first model matching `default_if`.
    fn pick_model<'r>(
        &self,
        registry: &'r ModelRegistry,
        default_if: impl Fn(&ModelSpec) -> bool,
    ) -> Result<&'r ModelSpec, ImageGenError> {
        match self
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            Some(id) => registry.get(id),
            None => registry
                .models()
                .iter()
                .find(|m| default_if(m))
                .ok_or_else(|| {
                    ImageGenError::invalid(
                        "model",
                        "no suitable model is configured in models.toml",
                    )
                }),
        }
    }

    fn resolve_core(
        &self,
        model: &ModelSpec,
        default_size: (u32, u32),
        mode: Mode,
        references: Vec<Reference>,
    ) -> Result<ResolvedRequest, ImageGenError> {
        let prompt = self.prompt.trim();
        if prompt.is_empty() {
            return Err(ImageGenError::invalid("prompt", "must not be empty"));
        }
        let d = &model.defaults;
        let width = self.width.unwrap_or(default_size.0);
        let height = self.height.unwrap_or(default_size.1);
        check_dimension("width", width, model)?;
        check_dimension("height", height, model)?;
        let pixels = u64::from(width) * u64::from(height);
        if pixels > model.max_pixels {
            return Err(ImageGenError::invalid(
                "width",
                format!(
                    "{width}x{height} = {pixels} pixels exceeds `{}` max of {}",
                    model.id, model.max_pixels
                ),
            ));
        }

        let steps = match &model.sigma_schedule {
            Some(schedule) => {
                let fixed = schedule.steps();
                if self.steps.is_some_and(|s| s != fixed) {
                    return Err(ImageGenError::invalid(
                        "steps",
                        format!(
                            "`{}` uses a fixed {fixed}-step sigma schedule; omit steps or pass {fixed}",
                            model.id
                        ),
                    ));
                }
                fixed
            }
            None => self.steps.unwrap_or(d.steps),
        };
        if !(1..=MAX_STEPS).contains(&steps) {
            return Err(ImageGenError::invalid(
                "steps",
                format!("must be 1-{MAX_STEPS}, got {steps}"),
            ));
        }

        let cfg_scale = self.cfg_scale.unwrap_or(d.cfg_scale);
        if !cfg_scale.is_finite() || !(0.0..=MAX_CFG).contains(&cfg_scale) {
            return Err(ImageGenError::invalid(
                "cfg_scale",
                format!("must be 0-{MAX_CFG}, got {cfg_scale}"),
            ));
        }

        let seed = match self.seed {
            Some(s) if s > i64::MAX as u64 => {
                return Err(ImageGenError::invalid(
                    "seed",
                    format!("must be <= {}", i64::MAX),
                ));
            }
            Some(s) => s,
            None => u64::from(rand::random::<u32>()),
        };

        let mut warnings = Vec::new();
        if model.codex.is_some() {
            let given = [
                ("seed", self.seed.is_some()),
                ("steps", self.steps.is_some()),
                ("cfg_scale", self.cfg_scale.is_some()),
                (
                    "sampler",
                    self.sampler
                        .as_deref()
                        .is_some_and(|s| !s.trim().is_empty()),
                ),
            ];
            for (field, _) in given.iter().filter(|(_, g)| *g) {
                warnings.push(format!(
                    "`{field}` is ignored: `{}` runs through Codex, which has no {field} control",
                    model.id
                ));
            }
        }
        if prompt.chars().count() > LONG_PROMPT_CHARS {
            warnings.push(format!(
                "prompt is over {LONG_PROMPT_CHARS} characters; the text encoder may truncate it"
            ));
        }

        let sampler = self
            .sampler
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| d.sampler.clone());

        Ok(ResolvedRequest {
            model: model.id.clone(),
            prompt: prompt.to_string(),
            negative_prompt: self.negative_prompt.clone().unwrap_or_default(),
            width,
            height,
            steps,
            cfg_scale,
            seed,
            sampler,
            warnings,
            mode,
            references,
            custom_sigmas: model
                .sigma_schedule
                .as_ref()
                .map(|sch| sch.sigmas_for(width, height))
                .unwrap_or_default(),
            loras: Vec::new(),
        })
    }
}

impl EditRequest {
    /// Validate, load and fingerprint the inputs, and fill in defaults (output size from the first image).
    pub fn resolve<'r>(
        &self,
        registry: &'r ModelRegistry,
        inputs: &InputPolicy,
    ) -> Result<(ResolvedRequest, &'r ModelSpec), ImageGenError> {
        let model = self
            .params
            .pick_model(registry, |m| m.supports(Capability::Edit))?;
        if !model.supports(Capability::Edit) {
            let editable: Vec<_> = registry
                .models()
                .iter()
                .filter(|m| m.supports(Capability::Edit))
                .map(|m| m.id.clone())
                .collect();
            return Err(ImageGenError::invalid(
                "model",
                format!(
                    "`{}` does not support editing. Models that do: {}",
                    model.id,
                    if editable.is_empty() {
                        "none".into()
                    } else {
                        editable.join(", ")
                    }
                ),
            ));
        }
        let max = model.max_ref_images as usize;
        if self.images.is_empty() || self.images.len() > max {
            return Err(ImageGenError::invalid(
                "images",
                format!(
                    "`{}` takes 1-{max} images, got {}",
                    model.id,
                    self.images.len()
                ),
            ));
        }
        let references = inputs.load(&self.images)?;
        let first = &references[0];
        let size = fit_size(first.width, first.height, model);
        let resolved = self
            .params
            .resolve_core(model, size, Mode::Edit, references)?;
        Ok((resolved, model))
    }
}

/// Keep `w:h`, scale down to fit `max_pixels`, and round each side down to `size_multiple`.
pub fn fit_size(w: u32, h: u32, model: &ModelSpec) -> (u32, u32) {
    let m = model.size_multiple.max(1);
    let (mut fw, mut fh) = (f64::from(w.max(1)), f64::from(h.max(1)));
    let pixels = fw * fh;
    if pixels > model.max_pixels as f64 {
        let s = (model.max_pixels as f64 / pixels).sqrt();
        fw *= s;
        fh *= s;
    }
    let snap = |v: f64| ((v as u32) / m * m).max(m);
    (snap(fw), snap(fh))
}

fn check_dimension(field: &str, value: u32, model: &ModelSpec) -> Result<(), ImageGenError> {
    if value == 0 || !value.is_multiple_of(model.size_multiple) {
        return Err(ImageGenError::invalid(
            field,
            format!(
                "must be a positive multiple of {} for `{}`, got {value}",
                model.size_multiple, model.id
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;

    /// The example registry plus a generate-only model, so "can't edit" stays testable.
    fn registry() -> ModelRegistry {
        let extra = r#"
[[models]]
id = "txt2img-only"
backend = "sdcpp"
capabilities = ["txt2img"]
license = "MIT"
commercial_weights = true
est_memory_mb = 1
max_pixels = 1048576
size_multiple = 16
defaults = { width = 512, height = 512, steps = 8, cfg_scale = 1.0 }
"#;
        ModelRegistry::from_toml(&format!("{MODELS_EXAMPLE}\n{extra}")).unwrap()
    }

    fn req(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: prompt.into(),
            wait: true,
            ..Default::default()
        }
    }

    fn field_of(err: ImageGenError) -> String {
        match err {
            ImageGenError::InvalidRequest { field, .. } => field,
            other => panic!("expected InvalidRequest, got {other:?}"),
        }
    }

    #[test]
    fn defaults_come_from_the_model() {
        let reg = registry();
        let mut r = req("  a red crate  ");
        r.model = Some("qwen-image-2.1".into());
        let (res, model) = r.resolve(&reg).unwrap();
        assert_eq!(model.id, "qwen-image-2.1");
        assert_eq!(res.prompt, "a red crate");
        assert_eq!((res.width, res.height, res.steps), (1024, 1024, 20));
        assert_eq!(res.cfg_scale, 1.0);
        assert_eq!(res.sampler.as_deref(), Some("euler"));
        assert!(res.warnings.is_empty());
    }

    #[test]
    fn default_model_is_first_and_seed_is_concrete() {
        let (res, _) = req("x").resolve(&registry()).unwrap();
        assert_eq!(res.model, "qwen-image-2.1-turbo");
        assert!(res.seed <= u64::from(u32::MAX));
        let mut r = req("x");
        r.seed = Some(42);
        assert_eq!(r.resolve(&registry()).unwrap().0.seed, 42);
    }

    #[test]
    fn overrides_win() {
        let mut r = req("x");
        r.model = Some("qwen-image-2.1".into());
        r.width = Some(512);
        r.height = Some(768);
        r.steps = Some(4);
        r.cfg_scale = Some(2.5);
        r.sampler = Some("dpm++2m".into());
        let (res, _) = r.resolve(&registry()).unwrap();
        assert_eq!((res.width, res.height, res.steps), (512, 768, 4));
        assert_eq!(res.cfg_scale, 2.5);
        assert_eq!(res.sampler.as_deref(), Some("dpm++2m"));
    }

    #[test]
    fn validation_rules() {
        let reg = registry();
        assert_eq!(field_of(req("   ").resolve(&reg).unwrap_err()), "prompt");

        let mut r = req("x");
        r.width = Some(1000); // qwen multiple is 32
        assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "width");

        let mut r = req("x");
        r.model = Some("qwen-image-2.1".into());
        r.height = Some(1040); // multiple of 16 but not 32
        assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "height");

        let mut r = req("x");
        r.width = Some(2080);
        r.height = Some(2048); // over qwen max_pixels (2048x2048)
        let err = r.resolve(&reg).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");

        for steps in [0, 101] {
            let mut r = req("x");
            r.model = Some("qwen-image-2.1".into());
            r.steps = Some(steps);
            assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "steps");
        }
        for cfg in [-1.0, 31.0, f32::NAN] {
            let mut r = req("x");
            r.cfg_scale = Some(cfg);
            assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "cfg_scale");
        }
        let mut r = req("x");
        r.seed = Some(u64::MAX);
        assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "seed");
    }

    #[test]
    fn unknown_model_is_reported() {
        let mut r = req("x");
        r.model = Some("sdxl".into());
        assert!(matches!(
            r.resolve(&registry()).unwrap_err(),
            ImageGenError::UnknownModel { .. }
        ));
    }

    #[test]
    fn long_prompt_warns_not_errors() {
        let (res, _) = req(&"a".repeat(LONG_PROMPT_CHARS + 1))
            .resolve(&registry())
            .unwrap();
        assert_eq!(res.warnings.len(), 1);
    }

    fn qwen() -> ModelSpec {
        registry().get("qwen-image-2.1").unwrap().clone()
    }

    #[test]
    fn fit_size_keeps_aspect_snaps_and_caps() {
        let q = qwen(); // multiple 32, max 2048x2048
        assert_eq!(fit_size(1024, 1024, &q), (1024, 1024));
        assert_eq!(fit_size(1000, 750, &q), (992, 736));
        assert_eq!(fit_size(4000, 3000, &q), (2336, 1760));
        assert!(
            fit_size(4000, 3000, &q).0 as u64 * fit_size(4000, 3000, &q).1 as u64 <= q.max_pixels
        );
        assert_eq!(fit_size(10, 10, &q), (32, 32));
    }

    fn inputs(root: &std::path::Path) -> InputPolicy {
        InputPolicy {
            base_dir: root.to_path_buf(),
            allowed_roots: vec![root.to_path_buf()],
            max_bytes: 10_000_000,
            max_side: 4096,
        }
    }

    fn png(path: &std::path::Path, w: u32, h: u32) -> String {
        image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h))
            .save_with_format(path, image::ImageFormat::Png)
            .unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn edit_defaults_size_from_first_image() {
        let dir = tempfile::tempdir().unwrap();
        let a = png(&dir.path().join("a.png"), 1000, 750);
        let b = png(&dir.path().join("b.png"), 64, 64);
        let req = EditRequest {
            images: vec![a, b],
            params: req("paint it red"),
        };
        let reg = registry();
        let (res, model) = req.resolve(&reg, &inputs(dir.path())).unwrap();
        assert_eq!(model.id, "qwen-image-2.1-turbo"); // first edit-capable model
        assert_eq!(res.mode, Mode::Edit);
        assert_eq!((res.width, res.height), (992, 736));
        assert_eq!(res.references.len(), 2);
        assert_eq!(res.references[0].width, 1000);

        let mut explicit = req.clone();
        explicit.params.width = Some(512);
        explicit.params.height = Some(512);
        let (res, _) = explicit.resolve(&registry(), &inputs(dir.path())).unwrap();
        assert_eq!((res.width, res.height), (512, 512));
    }

    #[test]
    fn edit_validation() {
        let dir = tempfile::tempdir().unwrap();
        let a = png(&dir.path().join("a.png"), 64, 64);
        let reg = registry();
        let pol = inputs(dir.path());

        let mut r = EditRequest {
            images: vec![a.clone()],
            params: req("x"),
        };
        r.params.model = Some("txt2img-only".into());
        let err = r.resolve(&reg, &pol).unwrap_err().to_string();
        assert!(
            err.contains("does not support editing") && err.contains("qwen-image-2.1"),
            "{err}"
        );

        let none = EditRequest {
            images: vec![],
            params: req("x"),
        };
        assert_eq!(field_of(none.resolve(&reg, &pol).unwrap_err()), "images");

        let many = EditRequest {
            images: vec![a.clone(); 4],
            params: req("x"),
        };
        let err = many.resolve(&reg, &pol).unwrap_err();
        assert!(err.to_string().contains("takes 1-3 images, got 4"), "{err}");

        let missing = EditRequest {
            images: vec!["nope.png".into()],
            params: req("x"),
        };
        assert_eq!(field_of(missing.resolve(&reg, &pol).unwrap_err()), "images");
    }

    #[test]
    fn generate_resolves_as_txt2img_without_references() {
        let (res, _) = req("x").resolve(&registry()).unwrap();
        assert_eq!(res.mode, Mode::Txt2img);
        assert!(res.references.is_empty());
    }

    #[test]
    fn edit_request_deserializes_flattened() {
        let r: EditRequest =
            serde_json::from_str(r#"{"images":["a.png"],"prompt":"red","seed":3}"#).unwrap();
        assert_eq!(r.images, vec!["a.png"]);
        assert_eq!(r.params.prompt, "red");
        assert_eq!(r.params.seed, Some(3));
        assert!(r.params.wait);
    }

    #[test]
    fn turbo_fixes_steps_and_computes_sigmas() {
        let reg = registry();
        let mut r = req("x");
        r.model = Some("qwen-image-2.1-turbo".into());
        let (res, _) = r.resolve(&reg).unwrap();
        assert_eq!(res.steps, 6);
        assert_eq!(res.custom_sigmas.len(), 7);
        assert!((res.custom_sigmas[1] - 0.967754).abs() < 2e-6);
        r.steps = Some(6);
        assert!(r.resolve(&reg).is_ok());
        r.steps = Some(20);
        assert_eq!(field_of(r.resolve(&reg).unwrap_err()), "steps");
        // Models without a schedule send none.
        let (plain, _) = {
            let mut p = req("x");
            p.model = Some("qwen-image-2.1".into());
            p.resolve(&reg).unwrap()
        };
        assert!(plain.custom_sigmas.is_empty());
    }

    #[test]
    fn wait_defaults_to_true_when_deserialized() {
        let r: GenerateRequest = serde_json::from_str(r#"{"prompt":"x"}"#).unwrap();
        assert!(r.wait);
        assert!(!r.return_preview);
    }
}
