//! Installing benchmark suites ("task images ship separately"): copy a local
//! suite directory, or download and unpack a `.tar.gz` bundle, into the
//! suites directory — then pre-pull the Docker images the suite needs.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::{Path, PathBuf};

use crate::fsops::{copy_tree, remove_tree_best_effort};
use crate::spec::{load_bench_suite, BenchSuite, SUITE_MANIFEST};

/// Install a suite from a local directory (validated, then copied under
/// `suites_root/<suite-id>`). Returns the installed path.
pub fn install_suite_from_dir(
    suites_root: &Path,
    source: &Path,
) -> Result<PathBuf, LocalCodeError> {
    let suite = load_bench_suite(source)?;
    let dest = suites_root.join(&suite.meta.id);
    let src_canon = std::fs::canonicalize(source).ok();
    let dest_canon = std::fs::canonicalize(&dest).ok();
    if src_canon.is_some() && src_canon == dest_canon {
        return Ok(dest); // already installed in place
    }
    remove_tree_best_effort(&dest);
    copy_tree(source, &dest)?;
    Ok(dest)
}

/// Download a `.tar.gz` suite bundle and install it. The archive must contain
/// `suite.toml` at its root or inside a single top-level directory.
pub async fn install_suite_from_url(
    suites_root: &Path,
    url: &str,
) -> Result<PathBuf, LocalCodeError> {
    let resp = reqwest::Client::new()
        .get(url)
        .send()
        .await
        .map_err(|e| {
            LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string())
                .with_cause(format!("Downloading {url}"))
                .retryable(true)
        })?;
    if !resp.status().is_success() {
        return Err(LocalCodeError::new(
            ErrorCode::HfUnreachable,
            format!("Suite download failed with HTTP {}", resp.status()),
        )
        .retryable(true));
    }
    let bytes = resp.bytes().await.map_err(|e| {
        LocalCodeError::new(ErrorCode::HfUnreachable, e.to_string()).retryable(true)
    })?;

    // Unpack into a scratch dir next to the destination (same volume).
    let scratch = suites_root.join(".pull-tmp");
    remove_tree_best_effort(&scratch);
    std::fs::create_dir_all(&scratch)
        .map_err(|e| LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string()))?;
    let gz = flate2::read::GzDecoder::new(&bytes[..]);
    // `unpack` refuses entries that escape the destination.
    tar::Archive::new(gz).unpack(&scratch).map_err(|e| {
        LocalCodeError::new(ErrorCode::ConfigParseFailed, e.to_string())
            .with_cause("Could not unpack the suite archive (expected .tar.gz)")
    })?;

    let root = find_suite_root(&scratch).ok_or_else(|| {
        LocalCodeError::new(
            ErrorCode::ConfigParseFailed,
            "Archive does not contain a suite (no suite.toml at its root)",
        )
    })?;
    let installed = install_suite_from_dir(suites_root, &root);
    remove_tree_best_effort(&scratch);
    installed
}

/// `suite.toml` at the top, or inside exactly one top-level directory.
fn find_suite_root(dir: &Path) -> Option<PathBuf> {
    if dir.join(SUITE_MANIFEST).is_file() {
        return Some(dir.to_path_buf());
    }
    let entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    match entries.as_slice() {
        [only] if only.join(SUITE_MANIFEST).is_file() => Some(only.clone()),
        _ => None,
    }
}

/// Pre-pull every image a suite needs. `progress` gets one line per image.
/// Returns the images that failed (empty = all present).
pub async fn pull_suite_images(
    suite: &BenchSuite,
    progress: impl Fn(String),
) -> Vec<(String, String)> {
    let mut failures = Vec::new();
    for image in suite.images() {
        if crate::docker::image_present(&image).await {
            progress(format!("{image} — already present"));
            continue;
        }
        progress(format!("{image} — pulling…"));
        match crate::docker::pull_image(&image).await {
            Ok(()) => progress(format!("{image} — done")),
            Err(e) => {
                progress(format!("{image} — FAILED: {}", e.message));
                failures.push((image, e.message));
            }
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_from_dir_copies_under_suite_id() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("anywhere");
        std::fs::create_dir_all(src.join("tasks/t1/workspace")).unwrap();
        std::fs::write(
            src.join("suite.toml"),
            "id = \"mysuite\"\ntitle = \"M\"\nversion = \"1\"\n",
        )
        .unwrap();
        std::fs::write(
            src.join("tasks/t1/task.toml"),
            "title = \"T\"\nprompt = \"p\"\nimage = \"i\"\n[[grader.checks]]\nname = \"c\"\ncmd = \"true\"\n",
        )
        .unwrap();
        std::fs::write(src.join("tasks/t1/workspace/f"), "x").unwrap();

        let suites = tmp.path().join("suites");
        std::fs::create_dir_all(&suites).unwrap();
        let dest = install_suite_from_dir(&suites, &src).unwrap();
        assert!(dest.ends_with("mysuite"));
        assert!(dest.join("tasks/t1/workspace/f").is_file());
        // Reinstall replaces cleanly.
        install_suite_from_dir(&suites, &src).unwrap();
    }

    #[test]
    fn suite_root_found_at_top_or_single_subdir() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(find_suite_root(tmp.path()).is_none());
        let sub = tmp.path().join("bundle");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("suite.toml"), "id=\"s\"").unwrap();
        assert_eq!(find_suite_root(tmp.path()).unwrap(), sub);
        // suite.toml directly at top wins.
        std::fs::write(tmp.path().join("suite.toml"), "id=\"t\"").unwrap();
        assert_eq!(find_suite_root(tmp.path()).unwrap(), tmp.path());
    }
}
