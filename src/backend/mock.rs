//! Deterministic fake backend for tests and demos (feature `mock`).
//!
//! Produces a small solid-colour PNG derived from the seed, sleeps `step_delay` per step,
//! honours cancellation and can be told to fail the next job.

use std::collections::BTreeMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use tokio_util::sync::CancellationToken;

use super::{
    Availability, BackendError, GeneratedImage, ImageBackend, Phase, ProgressSink, Timings,
};
use crate::registry::ModelSpec;
use crate::request::ResolvedRequest;

pub struct MockBackend {
    step_delay: Duration,
    loaded: Mutex<Option<String>>,
    fail_next: AtomicBool,
    pub generations: AtomicUsize,
    pub unloads: AtomicUsize,
}

impl MockBackend {
    pub fn new(step_delay: Duration) -> Self {
        Self {
            step_delay,
            loaded: Mutex::new(None),
            fail_next: AtomicBool::new(false),
            generations: AtomicUsize::new(0),
            unloads: AtomicUsize::new(0),
        }
    }

    pub fn fail_next(&self) {
        self.fail_next.store(true, Ordering::SeqCst);
    }

    /// 32x32 PNG whose colour is a pure function of the seed.
    pub fn render(seed: u64) -> Vec<u8> {
        let rgb = [
            (seed & 0xff) as u8,
            ((seed >> 8) & 0xff) as u8,
            ((seed >> 16) & 0xff) as u8,
        ];
        let img = image::RgbImage::from_pixel(32, 32, image::Rgb(rgb));
        let mut out = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encoding a PNG in memory cannot fail");
        out
    }
}

#[async_trait::async_trait]
impl ImageBackend for MockBackend {
    fn id(&self) -> &'static str {
        "mock"
    }

    async fn availability(&self) -> Availability {
        Availability::Ready
    }

    async fn generate(
        &self,
        model: &ModelSpec,
        _files: &BTreeMap<String, PathBuf>,
        request: &ResolvedRequest,
        progress: ProgressSink,
        cancel: CancellationToken,
    ) -> Result<GeneratedImage, BackendError> {
        progress.report(Phase::Loading);
        *self.loaded.lock().unwrap() = Some(model.id.clone());
        if self.fail_next.swap(false, Ordering::SeqCst) {
            return Err(BackendError::Failed {
                message: "injected failure".into(),
                stderr_tail: "mock: boom".into(),
            });
        }
        for step in 1..=request.steps {
            tokio::select! {
                _ = cancel.cancelled() => return Err(BackendError::Cancelled),
                _ = tokio::time::sleep(self.step_delay) => {}
            }
            progress.report(Phase::Sampling {
                step,
                total: request.steps,
            });
        }
        progress.report(Phase::Decoding);
        self.generations.fetch_add(1, Ordering::SeqCst);
        // Edits mix the first reference's fingerprint into the colour, still deterministic.
        let colour_seed = match request.references.first() {
            Some(r) => {
                request.seed
                    ^ u64::from_str_radix(&r.sha256[..r.sha256.len().min(12)], 16).unwrap_or(0)
            }
            None => request.seed,
        };
        Ok(GeneratedImage {
            png: Self::render(colour_seed),
            seed: request.seed,
            timings: Timings {
                load_ms: Some(0),
                sample_ms: Some(self.step_delay.as_millis() as u64 * u64::from(request.steps)),
                decode_ms: Some(0),
            },
        })
    }

    async fn unload(&self) {
        if self.loaded.lock().unwrap().take().is_some() {
            self.unloads.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn loaded_model(&self) -> Option<String> {
        self.loaded.lock().unwrap().clone()
    }
}
