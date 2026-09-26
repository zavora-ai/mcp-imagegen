//! Custom sigma schedules (design §12): raw sigma nodes plus a resolution-dependent shift.
//!
//! stable-diffusion.cpp uses `custom_sigmas` verbatim, so the shift the model's own scheduler
//! would apply has to happen here.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SigmaSchedule {
    /// Raw sigma nodes, highest noise first, each in (0, 1]. One step per node.
    pub raw: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shift: Option<SigmaShift>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SigmaShift {
    /// Flux-style dynamic shift: `mu` grows linearly with the image token count.
    /// Qwen-Image-2.1: base_seq 256, max_seq 8192, base_shift 0.5, max_shift 0.9, patch 16.
    FluxDynamic {
        base_seq: u32,
        max_seq: u32,
        base_shift: f32,
        max_shift: f32,
        /// Pixels per token side (VAE factor × patch size).
        patch: u32,
    },
}

impl SigmaSchedule {
    pub fn steps(&self) -> u32 {
        self.raw.len() as u32
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.raw.is_empty() {
            return Err("sigma_schedule.raw is empty".into());
        }
        if self.raw.iter().any(|s| !(*s > 0.0 && *s <= 1.0)) {
            return Err("sigma_schedule.raw values must be in (0, 1]".into());
        }
        if self.raw.windows(2).any(|w| w[1] >= w[0]) {
            return Err("sigma_schedule.raw must be strictly decreasing".into());
        }
        if let Some(SigmaShift::FluxDynamic {
            base_seq,
            max_seq,
            patch,
            ..
        }) = &self.shift
            && (max_seq <= base_seq || *patch == 0)
        {
            return Err("sigma_schedule.shift needs max_seq > base_seq and patch > 0".into());
        }
        Ok(())
    }

    /// Shifted sigmas for a `width`×`height` image, with the terminal 0 appended.
    pub fn sigmas_for(&self, width: u32, height: u32) -> Vec<f32> {
        let mut out: Vec<f32> = match &self.shift {
            None => self.raw.clone(),
            Some(SigmaShift::FluxDynamic {
                base_seq,
                max_seq,
                base_shift,
                max_shift,
                patch,
            }) => {
                let seq = f64::from((width / patch) * (height / patch));
                let m = f64::from(max_shift - base_shift) / f64::from(max_seq - base_seq);
                let b = f64::from(*base_shift) - m * f64::from(*base_seq);
                let e = (seq * m + b).exp();
                self.raw
                    .iter()
                    .map(|&s| {
                        let s = f64::from(s);
                        (e / (e + (1.0 / s - 1.0))) as f32
                    })
                    .collect()
            }
        };
        out.push(0.0);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viggle() -> SigmaSchedule {
        SigmaSchedule {
            raw: vec![1.0, 0.9375, 0.875, 0.75, 0.5, 0.25],
            shift: Some(SigmaShift::FluxDynamic {
                base_seq: 256,
                max_seq: 8192,
                base_shift: 0.5,
                max_shift: 0.9,
                patch: 16,
            }),
        }
    }

    #[test]
    fn matches_measured_1024_schedule() {
        // sd.cpp logged seq 4096 / mu 0.694 at 1024²; these are the sigmas used in the live test.
        let want = [1.0, 0.967754, 0.933358, 0.857192, 0.666756, 0.400096, 0.0];
        let got = viggle().sigmas_for(1024, 1024);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 2e-6, "{got:?}");
        }
    }

    #[test]
    fn smaller_images_shift_less() {
        let s512 = viggle().sigmas_for(512, 512);
        let s1024 = viggle().sigmas_for(1024, 1024);
        assert_eq!(s512[0], 1.0);
        assert!(s512[4] < s1024[4]);
        assert!(s512.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn unshifted_passes_through() {
        let s = SigmaSchedule {
            raw: vec![1.0, 0.5],
            shift: None,
        };
        assert_eq!(s.sigmas_for(64, 64), vec![1.0, 0.5, 0.0]);
        assert_eq!(s.steps(), 2);
    }

    #[test]
    fn validation() {
        assert!(viggle().validate().is_ok());
        for raw in [vec![], vec![1.0, 1.0], vec![0.5, 0.9], vec![1.5], vec![0.0]] {
            let s = SigmaSchedule { raw, shift: None };
            assert!(s.validate().is_err(), "{s:?}");
        }
    }
}
