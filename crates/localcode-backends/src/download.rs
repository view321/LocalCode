//! Resumable, out-of-process model-weight downloads.
//!
//! Large GGUF / colibrì containers are gigabytes over a possibly flaky link, so
//! two properties matter beyond a plain HTTP GET:
//!
//! * **Resume** — a dropped connection must not restart from byte 0. Each file
//!   streams into `{name}.part`; on retry we ask the server for the remaining
//!   bytes with a `Range:` header and *append*, and a partial `.part` is kept
//!   (never deleted) so the next attempt — even in a brand-new process — picks
//!   up where it left off.
//! * **Survives app exit** — the download runs in a *detached* `localcode
//!   download` worker (see [`spawn_detached_worker`]). The worker publishes its
//!   progress to a small JSON [`DownloadState`] file in the model directory; the
//!   TUI renders that file and, on relaunch, re-attaches to (or resumes) whatever
//!   it finds. The on-disk state file is the single source of truth, so tracking
//!   is stateless across restarts.
//!
//! The same [`run_download`] engine backs both the detached worker and the
//! inline `localcode deploy` path, so resume behaviour is identical everywhere.

use crate::models_store::sanitize_model_dir;
use crate::BackendKind;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Progress/state file dropped in a model's cache directory by the download
/// worker. Read by the TUI to render progress and to re-attach on relaunch.
pub const DOWNLOAD_STATE_FILE: &str = ".localcode_download.json";

/// A worker is considered live while its state file was touched within this
/// window. The worker refreshes it every second or two, so a staler file means
/// the process died (crash, reboot, laptop sleep) and the download is resumable.
pub const STALE_AFTER_SECS: u64 = 30;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Max gap between streamed chunks before a transfer is considered stalled.
const READ_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    /// A worker is actively fetching (or was, until it went stale).
    Downloading,
    /// All files are on disk; weights are ready to deploy.
    Completed,
    /// The worker gave up (all mirrors failed, disk full, …). Resumable.
    Failed,
}

/// The persisted state of one model download. Written atomically by the worker,
/// read by the TUI. Serialised to [`DOWNLOAD_STATE_FILE`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadState {
    pub model_id: String,
    pub quantization: Option<String>,
    pub backend: BackendKind,
    /// Weight files being fetched (filenames only).
    pub files: Vec<String>,
    /// One download URL per file (primary host; mirrors expanded at fetch time).
    pub download_urls: Vec<String>,
    /// Total expected bytes across all files (0 when unknown).
    pub total_bytes: u64,
    /// Bytes on disk so far (finished files + the current `.part`).
    pub downloaded_bytes: u64,
    pub status: DownloadStatus,
    /// Human-readable last line (mirror in use, error text, …).
    pub message: String,
    /// Resolved primary weight path once [`DownloadStatus::Completed`].
    pub primary_file: Option<String>,
    /// PID of the worker that last wrote this file (0 if unknown).
    pub pid: u32,
    /// Unix seconds of the last write — drives the staleness / liveness check.
    pub updated_unix: u64,
}

impl DownloadState {
    /// Fraction complete in `0.0..=1.0` (0 when the total isn't known yet).
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.downloaded_bytes as f64 / self.total_bytes as f64).clamp(0.0, 1.0)
    }

    /// Percent complete, `0..=100`.
    pub fn percent(&self) -> u8 {
        (self.fraction() * 100.0).round() as u8
    }

    /// True when the state file has not been refreshed within
    /// [`STALE_AFTER_SECS`] — i.e. no worker is currently driving it, so the
    /// download is interrupted and can be resumed.
    pub fn is_stale(&self, now_unix: u64) -> bool {
        now_unix.saturating_sub(self.updated_unix) >= STALE_AFTER_SECS
    }

    /// A download the user can act on: failed, or downloading-but-stale (the
    /// worker died). Completed and actively-downloading states are left alone.
    pub fn is_resumable(&self, now_unix: u64) -> bool {
        match self.status {
            DownloadStatus::Completed => false,
            DownloadStatus::Failed => true,
            DownloadStatus::Downloading => self.is_stale(now_unix),
        }
    }
}

fn pct(done: u64, total: u64) -> u8 {
    if total == 0 {
        return 0;
    }
    ((done as f64 / total as f64).clamp(0.0, 1.0) * 100.0).round() as u8
}

/// Current Unix time in seconds (0 before the epoch, which never happens).
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The cache directory a model's weights live in (deterministic from its id).
pub fn model_dir(models_cache: &Path, model_id: &str) -> PathBuf {
    models_cache.join(sanitize_model_dir(model_id))
}

/// Read the download state file from a model directory, if present and valid.
pub fn read_state(dir: &Path) -> Option<DownloadState> {
    let raw = std::fs::read_to_string(dir.join(DOWNLOAD_STATE_FILE)).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Write the state file atomically (temp + rename) so a reader never sees a
/// half-written JSON document. Best-effort: a failed write is not fatal to the
/// download itself.
pub fn write_state(dir: &Path, state: &DownloadState) {
    let Ok(json) = serde_json::to_string_pretty(state) else {
        return;
    };
    let final_path = dir.join(DOWNLOAD_STATE_FILE);
    let tmp = dir.join(format!("{DOWNLOAD_STATE_FILE}.tmp"));
    if std::fs::write(&tmp, json).is_ok() {
        let _ = std::fs::rename(&tmp, &final_path);
    }
}

/// Remove the state file (called once the weights are consumed / deployed).
pub fn clear_state(dir: &Path) {
    let _ = std::fs::remove_file(dir.join(DOWNLOAD_STATE_FILE));
}

/// Every model directory under `models_cache` that has a download state file,
/// newest activity first. Backs the TUI's "downloads in progress" view and its
/// resume-on-relaunch scan.
pub fn scan_active(models_cache: &Path) -> Vec<DownloadState> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(models_cache) else {
        return out;
    };
    for entry in rd.flatten() {
        if entry.path().is_dir() {
            if let Some(s) = read_state(&entry.path()) {
                out.push(s);
            }
        }
    }
    out.sort_by_key(|s| std::cmp::Reverse(s.updated_unix));
    out
}

/// A single model download to run. Files and URLs are paired by index.
#[derive(Debug, Clone)]
pub struct DownloadSpec {
    pub model_id: String,
    pub quantization: Option<String>,
    pub backend: BackendKind,
    /// Destination directory (usually `model_dir(models_cache, model_id)`).
    pub dir: PathBuf,
    pub files: Vec<String>,
    pub download_urls: Vec<String>,
    pub total_bytes: u64,
}

/// What to do with an existing `.part` given the server's response to our
/// `Range:` request. Pulled out as a pure function so the resume decision — the
/// bit most likely to harbour an off-by-one — is unit-tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResumeAction {
    /// 206 Partial Content: append the streamed bytes after the existing part.
    Append,
    /// 200 OK (fresh, or the server ignored our Range): (re)write from zero.
    Restart,
    /// 416 and the part already spans the whole resource: it's complete.
    AlreadyComplete,
}

fn resume_action(status: u16, part_len: u64, content_range_total: Option<u64>) -> ResumeAction {
    match status {
        206 => ResumeAction::Append,
        416 => match content_range_total {
            // Our offset was exactly the size → the part is the whole file.
            Some(total) if part_len >= total && total > 0 => ResumeAction::AlreadyComplete,
            _ => ResumeAction::Restart,
        },
        // 200 and anything else we let through (4xx/5xx are filtered earlier).
        _ => ResumeAction::Restart,
    }
}

/// Parse the total size out of a `Content-Range: bytes */12345` (or
/// `bytes 200-1000/12345`) header, used to decide whether a 416 means "done".
fn parse_content_range_total(headers: &reqwest::header::HeaderMap) -> Option<u64> {
    let v = headers.get(reqwest::header::CONTENT_RANGE)?.to_str().ok()?;
    v.rsplit('/').next()?.trim().parse().ok()
}

/// Keep only the file name of an untrusted filename/URL so no `..` or absolute
/// component can escape the model directory.
fn safe_filename(hint: Option<&String>, url: &str) -> String {
    let raw = hint
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| url.rsplit('/').next().unwrap_or("weights.bin"));
    Path::new(raw)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "weights.bin".into())
}

/// Expand a resolved download URL into the same path on every mirror host
/// (primary first, in `hosts` order); an off-host URL is its own sole candidate.
/// Mirrors `localcode_hf::UrlBuilder::mirror_candidates` without the dependency.
pub fn expand_mirror_candidates(url: &str, hosts: &[String]) -> Vec<String> {
    let trimmed: Vec<String> = hosts
        .iter()
        .map(|h| h.trim_end_matches('/').to_string())
        .collect();
    for h in &trimmed {
        if let Some(rest) = url.strip_prefix(h.as_str()) {
            let mut out: Vec<String> = trimmed.iter().map(|host| format!("{host}{rest}")).collect();
            out.dedup();
            return out;
        }
    }
    vec![url.to_string()]
}

fn download_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        // No overall timeout: a multi-GB file on a slow link must not be cut off
        // mid-stream. The connect timeout still fails fast on an unreachable host.
        .build()
        .unwrap_or_default()
}

/// Download every file in `spec` into `spec.dir` with resume + mirror fallback,
/// returning the primary weight path (first GGUF, else first file). `on_progress`
/// is called with the cumulative bytes-on-disk across all files, throttled by the
/// caller as it sees fit.
///
/// Partial `.part` files are preserved on any network failure, so a later call
/// (this process or a fresh worker) resumes instead of restarting.
pub async fn run_download<F: FnMut(u64, &str)>(
    spec: &DownloadSpec,
    token: Option<&str>,
    mirror_hosts: &[String],
    mut on_progress: F,
) -> Result<PathBuf, LocalCodeError> {
    if spec.download_urls.is_empty() {
        return Err(LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            "No downloadable weight files for this model/quantization",
        ));
    }
    std::fs::create_dir_all(&spec.dir).map_err(LocalCodeError::from)?;
    crate::models_store::write_marker(&spec.dir, &spec.model_id);

    let client = download_client();
    let mut base: u64 = 0;
    let mut paths: Vec<PathBuf> = Vec::new();

    for (i, url) in spec.download_urls.iter().enumerate() {
        let filename = safe_filename(spec.files.get(i), url);
        let dest = spec.dir.join(&filename);

        // Already fully downloaded (finished in a prior run): count and skip.
        if let Ok(meta) = std::fs::metadata(&dest) {
            if meta.len() > 0 {
                base += meta.len();
                on_progress(base, &format!("Cached {filename}"));
                paths.push(dest);
                continue;
            }
        }

        let candidates = expand_mirror_candidates(url, mirror_hosts);
        let part = spec.dir.join(format!("{filename}.part"));
        let file_base = base;
        let bytes = fetch_file_resumable(
            &client,
            &candidates,
            &part,
            &dest,
            token,
            &filename,
            |file_bytes, msg| on_progress(file_base + file_bytes, msg),
        )
        .await?;
        base += bytes;
        paths.push(dest);
    }

    paths.sort();
    let primary = paths
        .iter()
        .find(|p| {
            p.extension()
                .map(|e| e.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        })
        .or_else(|| paths.first())
        .cloned()
        .ok_or_else(|| {
            LocalCodeError::new(ErrorCode::DeployDownloadFailed, "No files downloaded")
        })?;
    Ok(primary)
}

/// Drive one download to completion as the detached worker does: publish a
/// `Downloading` state file (heartbeated ~1×/s from streamed progress), then run
/// the resume engine, and finally record `Completed` (with the primary weight
/// path) or `Failed` (with how far it got). The `Result` is returned too so a
/// foreground/CLI caller can surface the error.
pub async fn run_worker(
    spec: &DownloadSpec,
    token: Option<&str>,
    mirror_hosts: &[String],
) -> Result<PathBuf, LocalCodeError> {
    // Seed the state with whatever is already on disk so a resume doesn't flash
    // back to 0%.
    let start_bytes = on_disk_bytes(spec);
    write_progress_state(spec, start_bytes, "Starting download", DownloadStatus::Downloading, None);

    let mut last_write = std::time::Instant::now();
    let result = run_download(spec, token, mirror_hosts, |downloaded, msg| {
        // Throttle disk writes to ~1/s; the engine calls back far more often.
        if last_write.elapsed() >= std::time::Duration::from_millis(1000) {
            last_write = std::time::Instant::now();
            write_progress_state(spec, downloaded, msg, DownloadStatus::Downloading, None);
            // Visible when run in a foreground terminal; discarded (stderr=null)
            // for the TUI's detached worker.
            eprintln!("[{}%] {msg}", pct(downloaded, spec.total_bytes));
        }
    })
    .await;

    match &result {
        Ok(primary) => write_progress_state(
            spec,
            spec.total_bytes.max(on_disk_bytes(spec)),
            "Download complete",
            DownloadStatus::Completed,
            Some(primary.display().to_string()),
        ),
        Err(e) => write_progress_state(
            spec,
            on_disk_bytes(spec),
            &e.message,
            DownloadStatus::Failed,
            None,
        ),
    }
    result
}

/// Bytes currently on disk for this spec: each finished file's size, else its
/// `.part`'s size. Used to seed/repair the reported progress accurately.
fn on_disk_bytes(spec: &DownloadSpec) -> u64 {
    let mut total = 0u64;
    for (i, url) in spec.download_urls.iter().enumerate() {
        let filename = safe_filename(spec.files.get(i), url);
        let dest = spec.dir.join(&filename);
        if let Ok(m) = std::fs::metadata(&dest) {
            total += m.len();
            continue;
        }
        if let Ok(m) = std::fs::metadata(spec.dir.join(format!("{filename}.part"))) {
            total += m.len();
        }
    }
    total
}

/// Build and persist a [`DownloadState`] for `spec` (stamped with this process's
/// pid and the current time).
pub fn write_progress_state(
    spec: &DownloadSpec,
    downloaded: u64,
    message: &str,
    status: DownloadStatus,
    primary_file: Option<String>,
) {
    let state = DownloadState {
        model_id: spec.model_id.clone(),
        quantization: spec.quantization.clone(),
        backend: spec.backend,
        files: spec.files.clone(),
        download_urls: spec.download_urls.clone(),
        total_bytes: spec.total_bytes,
        downloaded_bytes: downloaded,
        status,
        message: message.to_string(),
        primary_file,
        pid: std::process::id(),
        updated_unix: now_unix(),
    };
    write_state(&spec.dir, &state);
}

/// Fetch one file into `part` with `Range` resume, trying each candidate URL
/// (primary then mirrors) in turn, then rename to `dest`. Returns the file's
/// total byte length. A transport/stream error falls through to the next mirror
/// and **keeps** the partial file; only a local disk-write error is fatal.
#[allow(clippy::too_many_arguments)]
async fn fetch_file_resumable<F: FnMut(u64, &str)>(
    client: &reqwest::Client,
    candidates: &[String],
    part: &Path,
    dest: &Path,
    token: Option<&str>,
    filename: &str,
    mut on_progress: F,
) -> Result<u64, LocalCodeError> {
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let n = candidates.len();
    let mut last_err: Option<LocalCodeError> = None;

    for (attempt, url) in candidates.iter().enumerate() {
        let part_len = tokio::fs::metadata(part).await.map(|m| m.len()).unwrap_or(0);
        let via = if attempt == 0 {
            if part_len > 0 {
                format!("Resuming {filename} at {:.2} GiB", part_len as f64 / GIB)
            } else {
                format!("Downloading {filename}")
            }
        } else {
            format!("Downloading {filename} (mirror {}/{})", attempt + 1, n)
        };
        on_progress(part_len, &via);

        let mut req = client.get(url);
        if let Some(t) = token {
            req = req.bearer_auth(t);
        }
        if part_len > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={part_len}-"));
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(
                    LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                        .with_cause(format!("Network error downloading {filename}"))
                        .with_hint("Check connectivity or configure registry.mirrors")
                        .retryable(true),
                );
                continue;
            }
        };

        let status = resp.status();
        if !status.is_success() && status.as_u16() != 416 {
            last_err = Some(
                LocalCodeError::new(
                    ErrorCode::DeployDownloadFailed,
                    format!("Download failed ({status}): {filename}"),
                )
                .with_hint("Gated model? Set HF_TOKEN")
                .retryable(true),
            );
            continue;
        }

        let cr_total = parse_content_range_total(resp.headers());
        let action = resume_action(status.as_u16(), part_len, cr_total);

        // The part already covers the whole file (server said 416 at EOF).
        if action == ResumeAction::AlreadyComplete {
            rename_into_place(part, dest).await?;
            on_progress(part_len, &via);
            return Ok(part_len);
        }

        let resume = action == ResumeAction::Append;
        let mut file = match tokio::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(resume)
            .truncate(!resume)
            .open(part)
            .await
        {
            Ok(f) => f,
            Err(e) => {
                return Err(LocalCodeError::new(ErrorCode::IoError, e.to_string())
                    .with_hint("Check free disk space and cache-dir permissions"));
            }
        };

        let mut written = if resume { part_len } else { 0 };
        let mut stream = resp.bytes_stream();
        let mut stream_err = None;
        let mut last_emit = std::time::Instant::now();
        loop {
            // Bound a mid-stream stall: with no overall timeout (multi-GB files),
            // a silently dropped connection would otherwise hang the worker
            // forever. A gap this long is treated as a failure — keep the .part
            // and fall through to the next mirror / a later resume.
            let next = match tokio::time::timeout(READ_TIMEOUT, stream.next()).await {
                Ok(n) => n,
                Err(_) => {
                    stream_err = Some(
                        LocalCodeError::new(
                            ErrorCode::DeployDownloadFailed,
                            format!("Stalled: no data for {}s downloading {filename}", READ_TIMEOUT.as_secs()),
                        )
                        .retryable(true),
                    );
                    break;
                }
            };
            let Some(chunk) = next else { break };
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    stream_err = Some(
                        LocalCodeError::new(ErrorCode::DeployDownloadFailed, e.to_string())
                            .retryable(true),
                    );
                    break;
                }
            };
            if let Err(e) = file.write_all(&chunk).await {
                // A disk error won't be fixed by another mirror — fail, but keep
                // the part so a later retry (after freeing space) can resume.
                let _ = file.flush().await;
                return Err(LocalCodeError::new(ErrorCode::IoError, e.to_string())
                    .with_hint("Check free disk space"));
            }
            written += chunk.len() as u64;
            if last_emit.elapsed() > std::time::Duration::from_millis(500) {
                last_emit = std::time::Instant::now();
                on_progress(
                    written,
                    &format!("{via} — {:.2} GiB", written as f64 / GIB),
                );
            }
        }
        let _ = file.flush().await;
        drop(file);

        if let Some(e) = stream_err {
            // Keep the .part for resume; try the next mirror.
            last_err = Some(e);
            continue;
        }
        rename_into_place(part, dest).await?;
        on_progress(written, &via);
        return Ok(written);
    }

    Err(last_err.unwrap_or_else(|| {
        LocalCodeError::new(
            ErrorCode::DeployDownloadFailed,
            format!("No reachable source for {filename}"),
        )
        .retryable(true)
    }))
}

async fn rename_into_place(part: &Path, dest: &Path) -> Result<(), LocalCodeError> {
    tokio::fs::rename(part, dest).await.map_err(|e| {
        LocalCodeError::new(ErrorCode::IoError, e.to_string())
            .with_cause("Failed to finalize a downloaded file")
    })
}

/// Spawn a detached `localcode download` worker for `model_id`. The child
/// **outlives this process**: on Windows it gets its own process group and no
/// console window so a `taskkill /T` on our tree can't reach it; on Unix it
/// leads its own group and is reparented to init. Returns the worker PID.
///
/// `exe` is the path to the current `localcode` binary
/// (`std::env::current_exe()`); the worker resolves the rest of the task from the
/// [`DownloadState`] the caller already wrote into the model directory.
pub fn spawn_detached_worker(
    exe: &Path,
    model_id: &str,
    quant: Option<&str>,
    backend: BackendKind,
) -> Result<u32, LocalCodeError> {
    // Deliberately std (not tokio) Command: we never wait on it, and dropping a
    // std Child does NOT kill the process (tokio's kill_on_drop would).
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("download")
        .arg(model_id)
        .arg("--backend")
        .arg(backend.as_str());
    if let Some(q) = quant {
        cmd.arg("--quant").arg(q);
    }
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: no
        // inherited console, own group → survives our exit and the tree-kill.
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        // Lead a new process group so a terminal-close SIGHUP to our group and
        // our own group-targeted kills don't reach the worker.
        cmd.process_group(0);
    }

    let child = cmd.spawn().map_err(|e| {
        LocalCodeError::new(ErrorCode::Internal, e.to_string())
            .with_cause("Failed to launch the background download worker")
            .with_hint("Deploy still works in the foreground; report this if it persists")
    })?;
    Ok(child.id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_action_maps_status_codes() {
        // 206 → append after the existing bytes.
        assert_eq!(resume_action(206, 100, None), ResumeAction::Append);
        // 200 → server ignored range (or fresh); rewrite from zero.
        assert_eq!(resume_action(200, 100, None), ResumeAction::Restart);
        assert_eq!(resume_action(200, 0, None), ResumeAction::Restart);
        // 416 with matching total → the part is the whole file.
        assert_eq!(resume_action(416, 500, Some(500)), ResumeAction::AlreadyComplete);
        assert_eq!(resume_action(416, 600, Some(500)), ResumeAction::AlreadyComplete);
        // 416 with a smaller/absent total → start over (corrupt/oversized part).
        assert_eq!(resume_action(416, 100, Some(500)), ResumeAction::Restart);
        assert_eq!(resume_action(416, 100, None), ResumeAction::Restart);
    }

    #[test]
    fn content_range_total_parsed() {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes */12345".parse().unwrap(),
        );
        assert_eq!(parse_content_range_total(&h), Some(12345));
        h.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 200-1000/2048".parse().unwrap(),
        );
        assert_eq!(parse_content_range_total(&h), Some(2048));
        // Unknown total ("*") is not a number.
        h.insert(
            reqwest::header::CONTENT_RANGE,
            "bytes 0-1/*".parse().unwrap(),
        );
        assert_eq!(parse_content_range_total(&h), None);
    }

    #[test]
    fn safe_filename_strips_path_components() {
        assert_eq!(
            safe_filename(Some(&"../../etc/passwd".to_string()), "http://x/y"),
            "passwd"
        );
        assert_eq!(safe_filename(None, "https://h/org/model/resolve/main/m.gguf"), "m.gguf");
        assert_eq!(safe_filename(Some(&String::new()), "https://h/a/b.bin"), "b.bin");
    }

    #[test]
    fn mirror_candidates_swap_and_order() {
        let hosts = vec![
            "https://huggingface.co".to_string(),
            "https://hf-mirror.com".to_string(),
        ];
        let got =
            expand_mirror_candidates("https://huggingface.co/org/model/resolve/main/m.gguf", &hosts);
        assert_eq!(
            got,
            vec![
                "https://huggingface.co/org/model/resolve/main/m.gguf".to_string(),
                "https://hf-mirror.com/org/model/resolve/main/m.gguf".to_string(),
            ]
        );
        // An off-host URL is returned as its own only candidate.
        assert_eq!(
            expand_mirror_candidates("https://other.example/a/b.gguf", &hosts),
            vec!["https://other.example/a/b.gguf".to_string()]
        );
    }

    #[test]
    fn staleness_and_resumability() {
        let base = DownloadState {
            model_id: "org/m".into(),
            quantization: Some("Q4_K_M".into()),
            backend: BackendKind::LlamaCpp,
            files: vec!["m.gguf".into()],
            download_urls: vec!["https://h/m.gguf".into()],
            total_bytes: 100,
            downloaded_bytes: 40,
            status: DownloadStatus::Downloading,
            message: String::new(),
            primary_file: None,
            pid: 1234,
            updated_unix: 1_000,
        };
        // Fresh worker (just wrote): not stale, not resumable.
        assert!(!base.is_stale(1_000 + 5));
        assert!(!base.is_resumable(1_000 + 5));
        // Worker went quiet past the window: stale and resumable.
        assert!(base.is_stale(1_000 + STALE_AFTER_SECS));
        assert!(base.is_resumable(1_000 + STALE_AFTER_SECS));
        // Completed is never resumable, even if old.
        let done = DownloadState { status: DownloadStatus::Completed, ..base.clone() };
        assert!(!done.is_resumable(1_000 + 10_000));
        // Failed is resumable immediately.
        let failed = DownloadState { status: DownloadStatus::Failed, ..base.clone() };
        assert!(failed.is_resumable(1_000 + 1));
        assert_eq!(base.percent(), 40);
    }

    /// A minimal HTTP/1.1 server that honours `Range: bytes=N-` and, on the very
    /// first connection, sends only half the requested bytes before hanging up —
    /// exactly the mid-stream drop an unstable link produces. The engine must
    /// keep the `.part` and resume from it on the second request.
    async fn spawn_range_server(body: Vec<u8>) -> std::net::SocketAddr {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let conns = Arc::new(AtomicUsize::new(0));
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let body = body.clone();
                let n = conns.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    // Read the request head.
                    let mut buf = vec![0u8; 2048];
                    let read = sock.read(&mut buf).await.unwrap_or(0);
                    let head = String::from_utf8_lossy(&buf[..read]).to_lowercase();
                    let start: u64 = head
                        .split("range: bytes=")
                        .nth(1)
                        .and_then(|s| s.split('-').next())
                        .and_then(|s| s.trim().parse().ok())
                        .unwrap_or(0);
                    let avail = &body[start as usize..];
                    let total = body.len() as u64;
                    let header = if start > 0 {
                        format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nAccept-Ranges: bytes\r\n\r\n",
                            avail.len(), start, total - 1, total
                        )
                    } else {
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\n\r\n",
                            avail.len()
                        )
                    };
                    let _ = sock.write_all(header.as_bytes()).await;
                    if n == 0 {
                        // First connection: send half, then drop (short read).
                        let half = avail.len() / 2;
                        let _ = sock.write_all(&avail[..half]).await;
                        let _ = sock.flush().await;
                        // Close without sending the rest.
                    } else {
                        let _ = sock.write_all(avail).await;
                        let _ = sock.flush().await;
                    }
                });
            }
        });
        addr
    }

    #[tokio::test]
    async fn download_resumes_after_a_midstream_drop() {
        let dir = tempfile::tempdir().unwrap();
        let body: Vec<u8> = (0..20_000u32).map(|i| (i % 251) as u8).collect();
        let addr = spawn_range_server(body.clone()).await;
        let url = format!("http://{addr}/org/model/resolve/main/model.gguf");
        let spec = DownloadSpec {
            model_id: "org/model".into(),
            quantization: Some("Q4_K_M".into()),
            backend: BackendKind::LlamaCpp,
            dir: dir.path().to_path_buf(),
            files: vec!["model.gguf".into()],
            download_urls: vec![url.clone()],
            total_bytes: body.len() as u64,
        };

        // First attempt drops mid-stream → keeps the .part, errors out.
        let first = run_download(&spec, None, &[], |_, _| {}).await;
        assert!(first.is_err(), "the mid-stream drop should surface as an error");
        let part = dir.path().join("model.gguf.part");
        let part_len = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        assert!(part_len > 0, "partial bytes must be kept for resume");
        assert!(part_len < body.len() as u64, "and it must be incomplete");
        assert!(!dir.path().join("model.gguf").exists());

        // Second attempt resumes via Range and completes to the exact bytes.
        let primary = run_download(&spec, None, &[], |_, _| {}).await.unwrap();
        assert_eq!(primary, dir.path().join("model.gguf"));
        let got = std::fs::read(dir.path().join("model.gguf")).unwrap();
        assert_eq!(got, body, "resumed download must reconstruct the file exactly");
        // The .part is consumed on success.
        assert!(!part.exists());
    }

    #[test]
    fn state_roundtrips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let st = DownloadState {
            model_id: "org/m".into(),
            quantization: None,
            backend: BackendKind::Colibri,
            files: vec!["out-0.safetensors".into()],
            download_urls: vec!["https://h/out-0.safetensors".into()],
            total_bytes: 10,
            downloaded_bytes: 3,
            status: DownloadStatus::Downloading,
            message: "hi".into(),
            primary_file: None,
            pid: 7,
            updated_unix: 42,
        };
        write_state(dir.path(), &st);
        let back = read_state(dir.path()).expect("state reads back");
        assert_eq!(back.model_id, "org/m");
        assert_eq!(back.downloaded_bytes, 3);
        assert_eq!(back.backend, BackendKind::Colibri);
        // scan_active finds it and clear_state removes it.
        let parent = dir.path().parent().unwrap();
        assert!(scan_active(parent).iter().any(|s| s.model_id == "org/m"));
        clear_state(dir.path());
        assert!(read_state(dir.path()).is_none());
    }
}
