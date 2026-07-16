//! Stop a spawned inference server *and all of its children*, freeing GPU memory.
//!
//! vLLM and SGLang don't serve from a single process: the vLLM V1 `EngineCore`
//! (and, under tensor parallelism, one worker per GPU) run as *child* processes,
//! and those children are what hold the CUDA context and the VRAM. A plain
//! [`Child::kill`](tokio::process::Child::kill) sends a single SIGKILL to the
//! top-level launcher only — because SIGKILL can't be caught the launcher never
//! runs its own teardown, and because we only target one PID the workers are
//! reparented to init and keep the VRAM. The `/dash` Stop button then looks like
//! it worked (the card disappears with the runtime) while `nvidia-smi` still
//! shows the model resident. Signalling the whole process group (Unix) / tree
//! (Windows) is what actually reclaims the memory.
//!
//! llama.cpp serves from a single `llama-server` process, which is why only the
//! multi-process backends (vLLM, SGLang) need this.

use tokio::process::{Child, Command};

/// Launch a server as the leader of a fresh process group so its whole tree can
/// be signalled at once. Unix only; on Windows this is a no-op because
/// `taskkill /T` walks the OS parent-PID table without it.
///
/// Call on the [`Command`] just before `spawn()`.
pub fn spawn_in_own_group(cmd: &mut Command) {
    #[cfg(unix)]
    cmd.process_group(0);
    #[cfg(not(unix))]
    let _ = cmd;
}

/// Terminate a managed server and every descendant it spawned, then reap it.
///
/// Unix: SIGTERM the process group first so vLLM can release the GPU cleanly,
/// then SIGKILL the group after a short grace period for anything still alive.
/// Windows: `taskkill /F /T` kills the whole process tree. The child must have
/// been spawned via [`spawn_in_own_group`] for the Unix path to reach the
/// workers.
pub async fn kill_tree(child: &mut Child) {
    let Some(pid) = child.id() else {
        return; // already exited and reaped
    };

    #[cfg(unix)]
    {
        // A negative pid targets the whole group (the child is its leader).
        let group = -(pid as i32);
        // SAFETY: `kill(2)` with a process-group id and a signal number touches
        // no memory and has no preconditions beyond a valid signal.
        unsafe { libc::kill(group, libc::SIGTERM) };
        // Let vLLM tear its workers down and free VRAM before we get forceful.
        for _ in 0..40 {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        unsafe { libc::kill(group, libc::SIGKILL) };
        let _ = child.wait().await;
    }

    #[cfg(windows)]
    {
        // `/T` kills the tree, `/F` forces it. Best-effort: if taskkill isn't
        // found we still fall through to reaping the handle below.
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &pid.to_string()])
            .output()
            .await;
        let _ = child.wait().await;
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        let _ = child.kill().await;
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A launcher that spawns a long-lived grandchild mirrors how vLLM's
    /// top-level `serve` process spawns the EngineCore/worker that actually holds
    /// the GPU. Killing only the launcher must not leave the grandchild alive.
    #[tokio::test]
    async fn kill_tree_reaps_grandchildren() {
        let dir = tempfile::tempdir().unwrap();
        let pidfile = dir.path().join("gc.pid");
        // Background a `sleep` (the "worker"), record its pid, then block. With
        // job control off (`sh -c`) the background job stays in the shell's
        // process group, exactly like vLLM's workers.
        let script = format!("sleep 300 & echo $! > {}; wait", pidfile.display());
        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&script);
        spawn_in_own_group(&mut cmd);
        let mut child = cmd.spawn().expect("spawn launcher");

        let gc_pid: i32 = loop {
            if let Ok(s) = std::fs::read_to_string(&pidfile) {
                if let Ok(p) = s.trim().parse() {
                    break p;
                }
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        };
        // Grandchild is alive (signal 0 is an existence check).
        assert_eq!(unsafe { libc::kill(gc_pid, 0) }, 0, "worker should be running");

        kill_tree(&mut child).await;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;

        // The whole group is gone — the orphaned-worker VRAM leak can't happen.
        assert_eq!(
            unsafe { libc::kill(gc_pid, 0) },
            -1,
            "worker must die with the group"
        );
    }
}
