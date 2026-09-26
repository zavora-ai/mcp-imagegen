//! Edit inputs (design §11.4; R11): jailed, format-sniffed, size-limited, fingerprinted.
//!
//! Inputs are read once at submit, so a file changed afterwards can't alter a queued job.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::config::expand_tilde;
use crate::error::ImageGenError;
use crate::output::within_roots;

#[derive(Debug, Clone)]
pub struct InputPolicy {
    /// Relative input paths resolve against this (the default output dir).
    pub base_dir: PathBuf,
    pub allowed_roots: Vec<PathBuf>,
    pub max_bytes: u64,
    pub max_side: u32,
}

/// A validated reference image.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Reference {
    pub path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub format: String,
    pub sha256: String,
    #[serde(skip)]
    pub data: Arc<[u8]>,
}

impl InputPolicy {
    pub fn load(&self, paths: &[String]) -> Result<Vec<Reference>, ImageGenError> {
        paths.iter().map(|p| self.load_one(p)).collect()
    }

    fn load_one(&self, raw: &str) -> Result<Reference, ImageGenError> {
        let bad = |msg: String| ImageGenError::invalid("images", msg);
        let raw = raw.trim();
        if raw.is_empty() {
            return Err(bad("empty image path".into()));
        }
        let path = expand_tilde(raw);
        let path = if path.is_absolute() {
            path
        } else {
            self.base_dir.join(path)
        };
        let path = path
            .canonicalize()
            .map_err(|e| bad(format!("{}: {e}", path.display())))?;
        if !path.is_file() {
            return Err(bad(format!("{} is not a regular file", path.display())));
        }
        if !within_roots(&path, &self.allowed_roots) {
            return Err(bad(format!(
                "{} is outside the allowed input roots ({})",
                path.display(),
                self.allowed_roots
                    .iter()
                    .map(|r| r.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let len = std::fs::metadata(&path)
            .map_err(|e| bad(format!("{}: {e}", path.display())))?
            .len();
        if len > self.max_bytes {
            return Err(bad(format!(
                "{} is {len} bytes; the limit is {} (max_input_bytes)",
                path.display(),
                self.max_bytes
            )));
        }
        let data = std::fs::read(&path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let format = match image::guess_format(&data) {
            Ok(image::ImageFormat::Png) => "png",
            Ok(image::ImageFormat::Jpeg) => "jpeg",
            Ok(image::ImageFormat::WebP) => "webp",
            _ => {
                return Err(bad(format!(
                    "{} is not a PNG, JPEG or WebP image",
                    path.display()
                )));
            }
        };
        let (width, height) = image::ImageReader::new(Cursor::new(&data))
            .with_guessed_format()
            .map_err(|e| bad(format!("{}: {e}", path.display())))?
            .into_dimensions()
            .map_err(|e| bad(format!("{}: unreadable image: {e}", path.display())))?;
        if width.max(height) > self.max_side {
            return Err(bad(format!(
                "{} is {width}x{height}; the longest side may be at most {} px (max_input_side)",
                path.display(),
                self.max_side
            )));
        }
        Ok(Reference {
            sha256: format!("{:x}", Sha256::digest(&data)),
            path,
            width,
            height,
            format: format.into(),
            data: data.into(),
        })
    }
}

/// Convenience for tests and callers that already hold a path.
pub fn is_supported_image(path: &Path) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|d| image::guess_format(&d).ok())
        .is_some_and(|f| {
            matches!(
                f,
                image::ImageFormat::Png | image::ImageFormat::Jpeg | image::ImageFormat::WebP
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(root: &Path) -> InputPolicy {
        InputPolicy {
            base_dir: root.join("generated"),
            allowed_roots: vec![root.to_path_buf()],
            max_bytes: 1_000_000,
            max_side: 256,
        }
    }

    fn write_image(path: &Path, w: u32, h: u32, format: image::ImageFormat) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 30, 30]));
        image::DynamicImage::ImageRgb8(img)
            .save_with_format(path, format)
            .unwrap();
    }

    #[test]
    fn accepts_png_jpeg_webp_and_fingerprints() {
        let dir = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        write_image(&dir.path().join("a.png"), 64, 32, image::ImageFormat::Png);
        write_image(&dir.path().join("b.jpg"), 40, 40, image::ImageFormat::Jpeg);
        write_image(&dir.path().join("c.webp"), 16, 16, image::ImageFormat::WebP);
        let refs = p
            .load(&[
                dir.path().join("a.png").to_string_lossy().into(),
                dir.path().join("b.jpg").to_string_lossy().into(),
                dir.path().join("c.webp").to_string_lossy().into(),
            ])
            .unwrap();
        assert_eq!(
            refs.iter().map(|r| r.format.as_str()).collect::<Vec<_>>(),
            ["png", "jpeg", "webp"]
        );
        assert_eq!((refs[0].width, refs[0].height), (64, 32));
        let bytes = std::fs::read(dir.path().join("a.png")).unwrap();
        assert_eq!(refs[0].sha256, format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(refs[0].sha256.len(), 64);
        assert_eq!(&*refs[0].data, bytes.as_slice());
    }

    #[test]
    fn relative_paths_resolve_against_the_output_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_image(
            &dir.path().join("generated/prev.png"),
            8,
            8,
            image::ImageFormat::Png,
        );
        let refs = policy(dir.path()).load(&["prev.png".into()]).unwrap();
        assert!(refs[0].path.ends_with("generated/prev.png"));
    }

    #[test]
    fn rejects_bad_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let p = policy(dir.path());
        let msg = |path: &Path| {
            p.load(&[path.to_string_lossy().into()])
                .unwrap_err()
                .to_string()
        };

        std::fs::write(dir.path().join("fake.png"), b"just some text").unwrap();
        assert!(msg(&dir.path().join("fake.png")).contains("not a PNG, JPEG or WebP"));

        assert!(msg(&dir.path().join("missing.png")).contains("missing.png"));

        write_image(&outside.path().join("x.png"), 8, 8, image::ImageFormat::Png);
        assert!(msg(&outside.path().join("x.png")).contains("outside the allowed input roots"));

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.path(), dir.path().join("link")).unwrap();
            assert!(
                msg(&dir.path().join("link/x.png")).contains("outside the allowed input roots")
            );
        }

        write_image(
            &dir.path().join("big.png"),
            300,
            10,
            image::ImageFormat::Png,
        );
        assert!(msg(&dir.path().join("big.png")).contains("max_input_side"));

        std::fs::create_dir_all(dir.path().join("adir")).unwrap();
        assert!(msg(&dir.path().join("adir")).contains("not a regular file"));

        let mut tight = policy(dir.path());
        tight.max_bytes = 10;
        write_image(&dir.path().join("ok.png"), 8, 8, image::ImageFormat::Png);
        let err = tight
            .load(&[dir.path().join("ok.png").to_string_lossy().into()])
            .unwrap_err();
        assert!(err.to_string().contains("max_input_bytes"));
        assert_eq!(err.code(), "invalid_request");
    }

    #[test]
    fn sniffing_ignores_extension() {
        let dir = tempfile::tempdir().unwrap();
        write_image(
            &dir.path().join("really-png.jpg"),
            8,
            8,
            image::ImageFormat::Png,
        );
        assert!(is_supported_image(&dir.path().join("really-png.jpg")));
        let refs = policy(dir.path())
            .load(&[dir.path().join("really-png.jpg").to_string_lossy().into()])
            .unwrap();
        assert_eq!(refs[0].format, "png");
    }
}
