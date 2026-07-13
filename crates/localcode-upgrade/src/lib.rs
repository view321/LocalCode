//! Update checking and self-update for LocalCode.
//!
//! LocalCode is installed from source: the install scripts clone the git repo
//! into an install dir (default `~/.local/share/localcode`), build with cargo,
//! and copy the binary to `~/.local/bin`. Updating therefore means: fetch the
//! branch the installer tracks, rebuild, and swap the running binary.
//!
//! Two safety properties this module maintains:
//! - The build never writes to the running executable. Self-update builds
//!   into a dedicated `target/self-update` dir, so cancelling mid-build (Esc
//!   in the TUI aborts the task) leaves the installed binary untouched.
//! - The final swap is rename+rename (metadata ops), so the window in which
//!   no binary exists is microseconds, and a stale `.old` sibling is cleaned
//!   up on next start.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tracing::{debug, warn};

/// The version this binary was built as (workspace version).
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A newer version is available on the tracked branch.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
}

/// Outcome of a successful self-update.
#[derive(Debug, Clone)]
pub struct UpdateReport {
    pub version: String,
    pub binary_path: PathBuf,
}

// ---------------------------------------------------------------------------
// Version check
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct UpdateChecker {
    http: reqwest::Client,
    repo_url: String,
    branch: String,
}

impl UpdateChecker {
    pub fn new(repo_url: &str, branch: &str) -> Result<Self, LocalCodeError> {
        let http = reqwest::Client::builder()
            .user_agent(format!("LocalCode/{CURRENT_VERSION}"))
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .map_err(|e| LocalCodeError::new(ErrorCode::Internal, e.to_string()))?;
        Ok(Self {
            http,
            repo_url: repo_url.to_string(),
            branch: branch.to_string(),
        })
    }

    /// URL of the raw workspace manifest on the tracked branch. Only GitHub
    /// repos are supported for checking — the same source the installer uses.
    fn raw_manifest_url(&self) -> Result<String, LocalCodeError> {
        let slug = github_slug(&self.repo_url).ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::UpdateCheckFailed,
                format!("Update checks need a github.com repo url, got {}", self.repo_url),
            )
            .with_hint("Set updates.repo_url to a github.com repository")
            .retryable(false)
        })?;
        Ok(format!(
            "https://raw.githubusercontent.com/{slug}/{}/Cargo.toml",
            self.branch
        ))
    }

    /// Fetch the latest version on the tracked branch. `Ok(None)` means this
    /// build is already up to date (or ahead, e.g. a dev build).
    pub async fn check(&self) -> Result<Option<UpdateInfo>, LocalCodeError> {
        let url = self.raw_manifest_url()?;
        debug!(%url, "checking for updates");
        let resp = self.http.get(&url).send().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::UpdateCheckFailed, e.to_string())
                .with_cause("Cannot reach GitHub to check for updates")
                .with_hint("Check your network; LocalCode works fine without updating")
                .retryable(true)
        })?;
        if !resp.status().is_success() {
            return Err(LocalCodeError::new(
                ErrorCode::UpdateCheckFailed,
                format!("Update check returned {} for {url}", resp.status()),
            )
            .with_hint("Verify updates.repo_url and updates.branch in config")
            .retryable(true));
        }
        let manifest = resp.text().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::UpdateCheckFailed, e.to_string()).retryable(true)
        })?;
        let latest = parse_workspace_version(&manifest).ok_or_else(|| {
            LocalCodeError::new(
                ErrorCode::UpdateCheckFailed,
                "Could not find a workspace version in the remote Cargo.toml",
            )
            .retryable(false)
        })?;

        if is_newer(&latest, CURRENT_VERSION) {
            Ok(Some(UpdateInfo {
                current: CURRENT_VERSION.to_string(),
                latest,
            }))
        } else {
            Ok(None)
        }
    }
}

/// `owner/repo` from a github.com URL (https or ssh, optional `.git`).
fn github_slug(repo_url: &str) -> Option<String> {
    let trimmed = repo_url.trim().trim_end_matches('/');
    let trimmed = trimmed.strip_suffix(".git").unwrap_or(trimmed);
    let rest = trimmed
        .strip_prefix("https://github.com/")
        .or_else(|| trimmed.strip_prefix("http://github.com/"))
        .or_else(|| trimmed.strip_prefix("git@github.com:"))
        .or_else(|| trimmed.strip_prefix("ssh://git@github.com/"))?;
    let mut parts = rest.split('/');
    let owner = parts.next().filter(|s| !s.is_empty())?;
    let repo = parts.next().filter(|s| !s.is_empty())?;
    Some(format!("{owner}/{repo}"))
}

/// `version = "x.y.z"` from `[workspace.package]` (falls back to `[package]`
/// so a non-workspace layout still works).
fn parse_workspace_version(manifest: &str) -> Option<String> {
    let doc: toml::Value = toml::from_str(manifest).ok()?;
    let ws = doc
        .get("workspace")
        .and_then(|w| w.get("package"))
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str());
    let pkg = doc
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(|v| v.as_str());
    ws.or(pkg).map(|s| s.to_string())
}

/// Numeric triple from "x.y.z" (pre-release/build suffixes ignored).
fn parse_semver(s: &str) -> Option<(u64, u64, u64)> {
    let core = s.split(['-', '+']).next()?;
    let mut it = core.split('.');
    let maj = it.next()?.trim().parse().ok()?;
    let min = it.next()?.trim().parse().ok()?;
    let pat = it.next().unwrap_or("0").trim().parse().ok()?;
    Some((maj, min, pat))
}

/// Strictly newer. Unparsable versions never report an update — better to
/// stay quiet than nag on a version string we don't understand.
fn is_newer(latest: &str, current: &str) -> bool {
    match (parse_semver(latest), parse_semver(current)) {
        (Some(l), Some(c)) => l > c,
        _ => {
            warn!(latest, current, "unparsable version; skipping update notice");
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Self-update
// ---------------------------------------------------------------------------

pub struct SelfUpdater {
    pub install_dir: PathBuf,
    repo_url: String,
    branch: String,
}

impl SelfUpdater {
    /// Locate the source checkout. Precedence: `LOCALCODE_INSTALL_DIR` env
    /// (resolved at use time, mirroring the install scripts) → config value →
    /// the installer default `~/.local/share/localcode`.
    pub fn resolve(
        config_install_dir: Option<&str>,
        repo_url: &str,
        branch: &str,
    ) -> Result<Self, LocalCodeError> {
        let dir = std::env::var("LOCALCODE_INSTALL_DIR")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(|| config_install_dir.map(PathBuf::from))
            .or_else(default_install_dir)
            .ok_or_else(|| {
                LocalCodeError::new(ErrorCode::UpdateFailed, "Cannot determine the install dir")
                    .with_hint("Set updates.install_dir in config or LOCALCODE_INSTALL_DIR")
            })?;

        if !dir.join(".git").exists() || !dir.join("Cargo.toml").exists() {
            return Err(LocalCodeError::new(
                ErrorCode::UpdateFailed,
                format!("No LocalCode source checkout at {}", dir.display()),
            )
            .with_cause("Self-update rebuilds from the git checkout the installer created")
            .with_hint("Re-run the install script from the README, or set updates.install_dir")
            .retryable(false));
        }

        Ok(Self {
            install_dir: dir,
            repo_url: repo_url.to_string(),
            branch: branch.to_string(),
        })
    }

    /// Fetch, rebuild, and swap the running binary. Progress lines go to
    /// `progress` (best-effort). Steps before the final swap are safe to
    /// cancel at any point.
    pub async fn run(
        &self,
        progress: mpsc::UnboundedSender<String>,
    ) -> Result<UpdateReport, LocalCodeError> {
        let say = |msg: &str| {
            let _ = progress.send(msg.to_string());
        };

        say("Fetching latest code…");
        // Fetch the configured URL directly (not the checkout's `origin`) so
        // updates.repo_url in config is always honored.
        self.git(&["fetch", "--depth", "1", &self.repo_url, &self.branch])
            .await?;
        self.git(&["checkout", "-B", &self.branch, "FETCH_HEAD"]).await?;

        let manifest = tokio::fs::read_to_string(self.install_dir.join("Cargo.toml"))
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
                    .with_cause("Cannot read the checkout's Cargo.toml after update")
            })?;
        let version = parse_workspace_version(&manifest).unwrap_or_else(|| "unknown".into());

        say(&format!("Building v{version} — this can take a few minutes…"));
        self.cargo_build(&progress).await?;

        let built = self.built_binary_path()?;
        say("Installing new binary…");
        let installed = swap_binary(&built)?;

        Ok(UpdateReport {
            version,
            binary_path: installed,
        })
    }

    async fn git(&self, args: &[&str]) -> Result<(), LocalCodeError> {
        let out = Command::new("git")
            .args(args)
            .current_dir(&self.install_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::UpdateFailed, format!("git failed to start: {e}"))
                    .with_hint("Install git and ensure it is on PATH")
            })?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(LocalCodeError::new(
                ErrorCode::UpdateFailed,
                format!("git {} failed: {}", args.first().unwrap_or(&""), stderr.trim()),
            )
            .with_hint(format!("Check the checkout at {}", self.install_dir.display()))
            .retryable(true));
        }
        Ok(())
    }

    /// `cargo build --release` into a dedicated target dir so the running
    /// executable is never a build output. Cargo progress (stderr lines) is
    /// forwarded so the UI can show live "Compiling …" status.
    async fn cargo_build(
        &self,
        progress: &mpsc::UnboundedSender<String>,
    ) -> Result<(), LocalCodeError> {
        let mut child = Command::new("cargo")
            .args(["build", "--release", "-p", "localcode-cli"])
            .arg("--target-dir")
            .arg(self.self_update_target_dir())
            .current_dir(&self.install_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                LocalCodeError::new(ErrorCode::UpdateFailed, format!("cargo failed to start: {e}"))
                    .with_hint("Install Rust via rustup and ensure cargo is on PATH")
            })?;

        // Both pipes must be drained or the child can deadlock on a full pipe.
        let stdout = child.stdout.take();
        let drain = tokio::spawn(async move {
            if let Some(out) = stdout {
                let mut lines = BufReader::new(out).lines();
                while let Ok(Some(_)) = lines.next_line().await {}
            }
        });

        let mut tail: Vec<String> = Vec::new();
        if let Some(err) = child.stderr.take() {
            let mut lines = BufReader::new(err).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let trimmed = line.trim().to_string();
                if trimmed.is_empty() {
                    continue;
                }
                tail.push(trimmed.clone());
                if tail.len() > 30 {
                    tail.remove(0);
                }
                if trimmed.starts_with("Compiling")
                    || trimmed.starts_with("Finished")
                    || trimmed.starts_with("Downloading")
                    || trimmed.starts_with("error")
                {
                    let _ = progress.send(format!("cargo: {trimmed}"));
                }
            }
        }

        let status = child.wait().await.map_err(|e| {
            LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
        })?;
        let _ = drain.await;

        if !status.success() {
            return Err(LocalCodeError::new(
                ErrorCode::UpdateFailed,
                "cargo build failed during self-update",
            )
            .with_cause(tail.join("\n"))
            .with_hint("Run `localcode update` in a terminal to see the full build log")
            .retryable(true));
        }
        Ok(())
    }

    fn self_update_target_dir(&self) -> PathBuf {
        self.install_dir.join("target").join("self-update")
    }

    fn built_binary_path(&self) -> Result<PathBuf, LocalCodeError> {
        let name = if cfg!(windows) { "localcode.exe" } else { "localcode" };
        let path = self.self_update_target_dir().join("release").join(name);
        if !path.exists() {
            return Err(LocalCodeError::new(
                ErrorCode::UpdateFailed,
                format!("Build succeeded but binary not found at {}", path.display()),
            ));
        }
        Ok(path)
    }
}

fn default_install_dir() -> Option<PathBuf> {
    directories::BaseDirs::new()
        .map(|b| b.home_dir().join(".local").join("share").join("localcode"))
}

/// Sibling path with a suffix appended to the file name (keeps `.exe` intact:
/// `localcode.exe` → `localcode.exe.old`).
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// Replace the running executable with `built`.
///
/// A running exe cannot be overwritten (Windows) or copied over (Linux
/// ETXTBSY), but it *can* be renamed. Copy the new binary next to the old one
/// first, then two renames — the no-binary window is microseconds, and a
/// leftover `.old` is removed on next start by [`cleanup_stale_artifacts`].
fn swap_binary(built: &Path) -> Result<PathBuf, LocalCodeError> {
    let current = std::env::current_exe().map_err(|e| {
        LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
            .with_cause("Cannot locate the running executable")
    })?;
    if current == *built {
        // Shouldn't happen with the dedicated target dir, but harmless.
        return Ok(current);
    }

    let staged = sibling(&current, ".new");
    let backup = sibling(&current, ".old");

    std::fs::copy(built, &staged).map_err(|e| {
        LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
            .with_cause(format!("Cannot stage new binary at {}", staged.display()))
            .with_hint("Check write permission on the install location")
    })?;
    let _ = std::fs::remove_file(&backup);
    std::fs::rename(&current, &backup).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
            .with_cause("Cannot move the current binary aside")
    })?;
    if let Err(e) = std::fs::rename(&staged, &current) {
        // Restore the old binary so the user is never left without one.
        let _ = std::fs::rename(&backup, &current);
        return Err(LocalCodeError::new(ErrorCode::UpdateFailed, e.to_string())
            .with_cause("Cannot move the new binary into place (old binary restored)"));
    }
    Ok(current)
}

/// Best-effort removal of `.old`/`.new` siblings left by a previous update.
/// Call on startup; failures are ignored (Windows keeps `.old` locked until
/// the last process using it exits).
pub fn cleanup_stale_artifacts() {
    let Ok(current) = std::env::current_exe() else {
        return;
    };
    for suffix in [".old", ".new"] {
        let stale = sibling(&current, suffix);
        if stale.exists() {
            let _ = std::fs::remove_file(&stale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slug_from_url_variants() {
        for url in [
            "https://github.com/view321/LocalCode.git",
            "https://github.com/view321/LocalCode",
            "https://github.com/view321/LocalCode/",
            "git@github.com:view321/LocalCode.git",
        ] {
            assert_eq!(github_slug(url).as_deref(), Some("view321/LocalCode"), "{url}");
        }
        assert_eq!(github_slug("https://gitlab.com/a/b"), None);
        assert_eq!(github_slug("https://github.com/onlyowner"), None);
    }

    #[test]
    fn workspace_version_parse() {
        let manifest = "[workspace]\nmembers=[]\n[workspace.package]\nversion = \"1.2.3\"\n";
        assert_eq!(parse_workspace_version(manifest).as_deref(), Some("1.2.3"));
        let pkg = "[package]\nname=\"x\"\nversion = \"0.9.0\"\n";
        assert_eq!(parse_workspace_version(pkg).as_deref(), Some("0.9.0"));
        assert_eq!(parse_workspace_version("not toml ["), None);
    }

    #[test]
    fn version_ordering() {
        assert!(is_newer("0.2.0", "0.1.0"));
        assert!(is_newer("1.0.0", "0.9.9"));
        assert!(is_newer("0.1.10", "0.1.9"));
        assert!(!is_newer("0.1.0", "0.1.0"));
        assert!(!is_newer("0.1.0", "0.2.0")); // never suggest downgrades
        assert!(!is_newer("garbage", "0.1.0")); // unparsable stays quiet
        assert!(is_newer("0.2.0-rc.1", "0.1.0")); // suffix ignored
    }

    #[test]
    fn sibling_keeps_full_name() {
        let p = Path::new("/x/bin/localcode.exe");
        assert!(sibling(p, ".old").ends_with("localcode.exe.old"));
    }

    #[test]
    fn resolve_rejects_non_checkout() {
        let dir = tempfile::tempdir().unwrap();
        std::env::remove_var("LOCALCODE_INSTALL_DIR");
        let err = SelfUpdater::resolve(Some(dir.path().to_str().unwrap()), "u", "main")
            .err()
            .expect("must reject a dir without .git");
        assert_eq!(err.code, ErrorCode::UpdateFailed);
    }
}
