//! Process helpers shared by subprocess backends: executable lookup and killing a whole process tree.

use std::path::{Path, PathBuf};

use tokio::process::{Child, Command};

/// Resolve a binary name against PATH, or check an explicit path. On Windows, `PATHEXT`
/// extensions (`.exe`, …) are tried too, so `sd-server` finds `sd-server.exe`.
pub fn find_executable(bin: &Path) -> Option<PathBuf> {
    let candidates = |p: &Path| -> Vec<PathBuf> {
        let mut out = vec![p.to_path_buf()];
        if cfg!(windows) && p.extension().is_none() {
            let exts = std::env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.CMD;.BAT".into());
            out.extend(
                exts.split(';')
                    .filter(|e| !e.is_empty())
                    .map(|e| p.with_extension(e.trim_start_matches('.').to_ascii_lowercase())),
            );
        }
        out
    };
    if bin.components().count() > 1 || bin.is_absolute() {
        return candidates(bin).into_iter().find(|p| p.is_file());
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .flat_map(|dir| candidates(&dir.join(bin)))
        .find(|p| p.is_file())
}

/// Make the child lead its own process group (Unix), so `kill_tree` also reaches what it starts.
/// On Windows `kill_tree` walks the tree with `taskkill /T`, so nothing is needed at spawn time.
pub fn own_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    cmd.process_group(0);
    #[cfg(not(unix))]
    let _ = cmd;
}

/// Kill a child and everything it started, then reap it.
/// Unix: SIGKILL to the child's process group (see `own_process_group`). Windows: `taskkill /T /F`.
pub async fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        #[cfg(unix)]
        if let Ok(pgid) = i32::try_from(pid) {
            // SAFETY: plain syscall with no memory arguments. A negative pid addresses the process group
            // the child leads; if it already exited, the call fails harmlessly with ESRCH.
            unsafe {
                libc::kill(-pgid, libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await;
        }
    }
    let _ = child.kill().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_executable_checks_paths() {
        let here = std::env::current_exe().unwrap();
        assert_eq!(find_executable(&here), Some(here.clone()));
        let dir = here.parent().unwrap();
        let name = here.file_stem().unwrap();
        // Bare name on a PATH that contains the directory (PATHEXT handles .exe on Windows).
        let old = std::env::var_os("PATH");
        // SAFETY: tests in this module don't read PATH concurrently.
        unsafe { std::env::set_var("PATH", dir) };
        let found = find_executable(Path::new(name));
        match old {
            Some(p) => unsafe { std::env::set_var("PATH", p) },
            None => unsafe { std::env::remove_var("PATH") },
        }
        assert!(found.is_some());
        assert!(find_executable(Path::new("/definitely/not/here/sd-server")).is_none());
        assert!(find_executable(Path::new("no-such-binary-xyz")).is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn kill_tree_takes_grandchildren_down() {
        // sh starts a background sleep (a grandchild) and prints its pid, then waits.
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "sleep 30 & echo $!; wait"])
            .stdout(std::process::Stdio::piped())
            .kill_on_drop(true);
        own_process_group(&mut cmd);
        let mut child = cmd.spawn().unwrap();
        let mut out = child.stdout.take().unwrap();
        let mut buf = [0u8; 32];
        let n = tokio::io::AsyncReadExt::read(&mut out, &mut buf)
            .await
            .unwrap();
        let grandchild: i32 = std::str::from_utf8(&buf[..n])
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        kill_tree(&mut child).await;
        // Give the kernel a moment to deliver the signal to the group.
        for _ in 0..50 {
            // SAFETY: signal 0 only checks existence.
            if unsafe { libc::kill(grandchild, 0) } != 0 {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        panic!("grandchild {grandchild} survived kill_tree");
    }
}
