//! Memory pre-flight (design §7 step 3; R5): refuse to load a model that would push the
//! machine into swap, instead of starving Blender or Unreal.

use std::sync::Mutex;

use crate::error::ImageGenError;
use crate::registry::ModelSpec;

pub trait MemoryProbe: Send + Sync {
    /// Memory the OS could hand out now, in MB (free + reclaimable caches).
    fn available_mb(&self) -> u64;
}

/// Real probe backed by `sysinfo`.
pub struct SystemMemory {
    sys: Mutex<sysinfo::System>,
}

impl Default for SystemMemory {
    fn default() -> Self {
        Self {
            sys: Mutex::new(sysinfo::System::new()),
        }
    }
}

impl MemoryProbe for SystemMemory {
    fn available_mb(&self) -> u64 {
        let mut sys = self.sys.lock().unwrap();
        sys.refresh_memory();
        // sysinfo's available_memory() reads 0 on macOS under compression; total - used
        // matches vm_stat's free + inactive + purgeable + speculative.
        let bytes = if cfg!(target_os = "macos") || sys.available_memory() == 0 {
            sys.total_memory().saturating_sub(sys.used_memory())
        } else {
            sys.available_memory()
        };
        bytes / (1024 * 1024)
    }
}

/// Fixed value, for tests.
pub struct FixedMemory(pub u64);

impl MemoryProbe for FixedMemory {
    fn available_mb(&self) -> u64 {
        self.0
    }
}

/// Check that `model` fits.
///
/// * `already_loaded` — the model is resident, nothing new will be allocated.
/// * `reclaimable_mb` — memory held by other loaded models that will be unloaded first.
pub fn preflight(
    model: &ModelSpec,
    already_loaded: bool,
    reclaimable_mb: u64,
    headroom_mb: u64,
    probe: &dyn MemoryProbe,
) -> Result<(), ImageGenError> {
    if already_loaded {
        return Ok(());
    }
    let free_mb = probe.available_mb() + reclaimable_mb;
    if free_mb >= model.est_memory_mb + headroom_mb {
        return Ok(());
    }
    Err(ImageGenError::InsufficientMemory {
        model: model.id.clone(),
        need_mb: model.est_memory_mb,
        free_mb,
        hint: format!(
            "Keep {headroom_mb} MB headroom: close Unreal/Blender, call unload_models, pick a smaller quantization, or lower memory_headroom_mb."
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MODELS_EXAMPLE;
    use crate::registry::ModelRegistry;

    fn qwen() -> ModelSpec {
        ModelRegistry::from_toml(MODELS_EXAMPLE)
            .unwrap()
            .get("qwen-image-2.1")
            .unwrap()
            .clone()
    }

    #[test]
    fn passes_with_enough_memory() {
        // 13500 needed + 2048 headroom
        assert!(preflight(&qwen(), false, 0, 2048, &FixedMemory(15_548)).is_ok());
    }

    #[test]
    fn fails_with_actionable_message() {
        let err = preflight(&qwen(), false, 0, 2048, &FixedMemory(10_000)).unwrap_err();
        assert_eq!(err.code(), "insufficient_memory");
        let msg = err.to_string();
        assert!(msg.contains("~13500 MB"), "{msg}");
        assert!(msg.contains("10000 MB free"), "{msg}");
        assert!(msg.contains("unload_models"), "{msg}");
    }

    #[test]
    fn already_loaded_skips_the_check() {
        assert!(preflight(&qwen(), true, 0, 2048, &FixedMemory(0)).is_ok());
    }

    #[test]
    fn memory_of_models_being_unloaded_counts() {
        assert!(preflight(&qwen(), false, 0, 2048, &FixedMemory(6_000)).is_err());
        assert!(preflight(&qwen(), false, 10_000, 2048, &FixedMemory(6_000)).is_ok());
    }

    #[test]
    fn system_probe_reports_something() {
        assert!(SystemMemory::default().available_mb() > 0);
    }
}
