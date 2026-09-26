//! Output files (design §7 steps 2 and 5; R1, R6).
//!
//! The only module that writes to disk. Paths are jailed under the configured roots,
//! existing files are never overwritten, and images land atomically via `.tmp` + rename.

use std::io::Cursor;
use std::path::{Component, Path, PathBuf};

use serde::Serialize;

use crate::config::expand_tilde;
use crate::error::ImageGenError;

/// Longest side of an inline preview.
pub const PREVIEW_MAX_SIDE: u32 = 512;
/// Upper bound on preview size so results stay well under MCP message limits.
pub const PREVIEW_MAX_BYTES: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq)]
pub struct OutputPolicy {
    pub default_dir: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WrittenImage {
    pub path: PathBuf,
    pub sidecar_path: PathBuf,
}

impl OutputPolicy {
    /// Where the image should go before collision handling. Rejects anything outside the roots.
    pub fn plan(
        &self,
        output_path: Option<&str>,
        output_dir: Option<&str>,
        prompt: &str,
        seed: u64,
    ) -> Result<PathBuf, ImageGenError> {
        self.plan_tagged(output_path, output_dir, prompt, None, seed)
    }

    /// Like `plan`, with a tag in the default file name: `<slug>-<tag>-<seed>.png`.
    pub fn plan_tagged(
        &self,
        output_path: Option<&str>,
        output_dir: Option<&str>,
        prompt: &str,
        tag: Option<&str>,
        seed: u64,
    ) -> Result<PathBuf, ImageGenError> {
        let file_name = match tag {
            Some(tag) => format!("{}-{tag}-{seed}.png", slug(prompt)),
            None => format!("{}-{seed}.png", slug(prompt)),
        };
        let target = match (output_path, output_dir) {
            (Some(p), _) if !p.trim().is_empty() => {
                let p = self.absolutize(p.trim());
                match p.extension().and_then(|e| e.to_str()) {
                    Some(ext) if ext.eq_ignore_ascii_case("png") => p,
                    _ => {
                        return Err(ImageGenError::invalid("output_path", "must end in .png"));
                    }
                }
            }
            (_, Some(d)) if !d.trim().is_empty() => self.absolutize(d.trim()).join(file_name),
            _ => self.default_dir.join(file_name),
        };
        self.jail(&target)?;
        Ok(target)
    }

    fn absolutize(&self, p: &str) -> PathBuf {
        let p = expand_tilde(p);
        if p.is_absolute() {
            p
        } else {
            self.default_dir.join(p)
        }
    }

    fn jail(&self, target: &Path) -> Result<(), ImageGenError> {
        let not_allowed = || ImageGenError::OutputPathNotAllowed {
            path: target.to_path_buf(),
            roots: self.allowed_roots.clone(),
        };
        // `..` resolves lexically on Windows but not on Unix (and can step around symlinks), so
        // reject it outright for identical, safe behaviour everywhere.
        if target.components().any(|c| c == Component::ParentDir) {
            return Err(not_allowed());
        }
        let inside = within_roots(target, &self.allowed_roots);
        if inside { Ok(()) } else { Err(not_allowed()) }
    }
}

/// Canonicalize the deepest existing ancestor (resolving symlinks) and append the rest.
/// `..` in the existing prefix is resolved by canonicalize; in the missing remainder it has no
/// file name to peel off, so the walk returns `None` and the path is rejected.
pub(crate) fn resolve_lenient(path: &Path) -> Option<PathBuf> {
    let mut existing = path.to_path_buf();
    let mut rest = Vec::new();
    while !existing.exists() {
        let name = existing.file_name()?.to_owned();
        rest.push(name);
        existing = existing.parent()?.to_path_buf();
    }
    let mut out = existing.canonicalize().ok()?;
    for part in rest.iter().rev() {
        match Path::new(part).components().next() {
            Some(Component::Normal(_)) => out.push(part),
            _ => return None,
        }
    }
    Some(out)
}

/// True if `path` resolves (symlinks followed) strictly inside one of `roots`.
pub(crate) fn within_roots(path: &Path, roots: &[PathBuf]) -> bool {
    let Some(resolved) = resolve_lenient(path) else {
        return false;
    };
    roots
        .iter()
        .filter_map(|root| resolve_lenient(root))
        .any(|root| resolved.starts_with(&root) && resolved != root)
}

/// First free `name.png` / `name-1.png` / … whose sidecar `.json` is also free.
pub fn unique_path(desired: &Path) -> PathBuf {
    let free = |p: &Path| !p.exists() && !p.with_extension("json").exists();
    if free(desired) {
        return desired.to_path_buf();
    }
    let stem = desired
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".into());
    let parent = desired.parent().unwrap_or_else(|| Path::new("."));
    (1..)
        .map(|n| parent.join(format!("{stem}-{n}.png")))
        .find(|p| free(p))
        .expect("unbounded search always finds a free name")
}

/// Write the PNG atomically plus its sidecar JSON. Never overwrites.
pub fn write_image(
    desired: &Path,
    png: &[u8],
    metadata: &impl Serialize,
) -> Result<WrittenImage, ImageGenError> {
    let parent = desired
        .parent()
        .ok_or_else(|| ImageGenError::io(desired, "no parent directory"))?;
    std::fs::create_dir_all(parent).map_err(|e| ImageGenError::io(parent, e))?;

    let path = unique_path(desired);
    let tmp = path.with_extension("png.tmp");
    std::fs::write(&tmp, png).map_err(|e| ImageGenError::io(&tmp, e))?;
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(ImageGenError::io(&path, e));
    }

    let sidecar_path = path.with_extension("json");
    let json =
        serde_json::to_vec_pretty(metadata).map_err(|e| ImageGenError::io(&sidecar_path, e))?;
    std::fs::write(&sidecar_path, json).map_err(|e| ImageGenError::io(&sidecar_path, e))?;

    Ok(WrittenImage { path, sidecar_path })
}

/// Downscaled PNG for inline display: longest side ≤ 512 px and under 1 MB.
pub fn make_preview(png: &[u8]) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory(png).map_err(|e| e.to_string())?;
    let mut side = PREVIEW_MAX_SIDE;
    loop {
        let thumb = if img.width().max(img.height()) > side {
            img.thumbnail(side, side)
        } else {
            img.clone()
        };
        let mut out = Vec::new();
        thumb
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        if out.len() <= PREVIEW_MAX_BYTES || side <= 64 {
            return Ok(out);
        }
        side /= 2;
    }
}

/// File-name-safe slug of the prompt: lowercase ASCII words joined by `-`, at most 48 chars.
pub fn slug(prompt: &str) -> String {
    let mut out = String::new();
    for ch in prompt.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 48 {
            break;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() { "image".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path) -> OutputPolicy {
        OutputPolicy {
            default_dir: root.join("generated"),
            allowed_roots: vec![root.to_path_buf()],
        }
    }

    fn tiny_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 255) as u8, (y % 255) as u8, 7])
        });
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    #[test]
    fn default_name_from_prompt_and_seed() {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        let got = p.plan(None, None, "A red Crate, weathered!", 42).unwrap();
        assert_eq!(
            got,
            dir.path().join("generated/a-red-crate-weathered-42.png")
        );
    }

    #[test]
    fn tagged_default_name() {
        let dir = tempfile::tempdir().unwrap();
        let got = policy(dir.path())
            .plan_tagged(None, None, "paint it red", Some("edit"), 5)
            .unwrap();
        assert!(got.ends_with("generated/paint-it-red-edit-5.png"));
    }

    #[test]
    fn output_dir_and_relative_paths() {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        let abs = dir.path().join("textures");
        let got = p.plan(None, Some(abs.to_str().unwrap()), "x", 1).unwrap();
        assert_eq!(got, abs.join("x-1.png"));
        let rel = p.plan(Some("props/crate.png"), None, "x", 1).unwrap();
        assert_eq!(rel, dir.path().join("generated/props/crate.png"));
    }

    #[test]
    fn rejects_escapes() {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        let outside = tempfile::tempdir().unwrap();
        let foreign = outside.path().join("x.png");
        let foreign = foreign.to_str().unwrap();
        for bad in [foreign, "../../x.png", "../escape/x.png", "sub/../../x.png"] {
            let err = p.plan(Some(bad), None, "x", 1).unwrap_err();
            assert_eq!(err.code(), "output_path_not_allowed", "{bad}: {err}");
        }
        assert_eq!(
            p.plan(Some("x.jpg"), None, "x", 1).unwrap_err().code(),
            "invalid_request"
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_escape() {
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).unwrap();
        let p = policy(root.path());
        let target = root.path().join("link/x.png");
        let err = p
            .plan(Some(target.to_str().unwrap()), None, "x", 1)
            .unwrap_err();
        assert_eq!(err.code(), "output_path_not_allowed");
    }

    #[test]
    fn root_itself_is_not_a_file_target() {
        let dir = tempfile::tempdir().unwrap();
        let p = OutputPolicy {
            default_dir: dir.path().to_path_buf(),
            allowed_roots: vec![dir.path().join("root.png")],
        };
        let target = dir.path().join("root.png");
        assert!(
            p.plan(Some(target.to_str().unwrap()), None, "x", 1)
                .is_err()
        );
    }

    #[test]
    fn collisions_get_suffixes_and_nothing_is_overwritten() {
        let dir = tempfile::tempdir().unwrap();
        let desired = dir.path().join("out/crate.png");
        let meta = serde_json::json!({"seed": 1});
        let a = write_image(&desired, b"first", &meta).unwrap();
        let b = write_image(&desired, b"second", &meta).unwrap();
        let c = write_image(&desired, b"third", &meta).unwrap();
        assert_eq!(a.path, desired);
        assert_eq!(b.path, dir.path().join("out/crate-1.png"));
        assert_eq!(c.path, dir.path().join("out/crate-2.png"));
        assert_eq!(std::fs::read(&a.path).unwrap(), b"first");
        assert!(!dir.path().join("out/crate.png.tmp").exists());
        let sidecar: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&b.sidecar_path).unwrap()).unwrap();
        assert_eq!(sidecar["seed"], 1);
        assert_eq!(b.sidecar_path, dir.path().join("out/crate-1.json"));
    }

    #[test]
    fn stray_sidecar_blocks_the_name() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.json"), b"{}").unwrap();
        assert_eq!(
            unique_path(&dir.path().join("a.png")),
            dir.path().join("a-1.png")
        );
    }

    #[test]
    fn preview_is_downscaled() {
        let png = tiny_png(1024, 768);
        let preview = make_preview(&png).unwrap();
        let img = image::load_from_memory(&preview).unwrap();
        assert_eq!((img.width(), img.height()), (512, 384));
        assert!(preview.len() <= PREVIEW_MAX_BYTES);
        let small = make_preview(&tiny_png(100, 50)).unwrap();
        assert_eq!(image::load_from_memory(&small).unwrap().width(), 100);
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("  Hello,   World  "), "hello-world");
        assert_eq!(slug("日本語"), "image");
        assert!(slug(&"word ".repeat(40)).len() <= 48);
    }
}
