//! Directory-tree copy/overlay helpers for assembling task workspaces and
//! graded trees.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::Path;

/// Recursively copy `src` into `dst`, creating directories and overwriting
/// existing files. This doubles as the overlay operation: copying `hidden/`
/// over a graded tree replaces any same-named file the agent wrote (the
/// anti-tampering guarantee for test files).
pub fn copy_tree(src: &Path, dst: &Path) -> Result<(), LocalCodeError> {
    let io_err = |e: std::io::Error, what: &Path| {
        LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
            .with_cause(format!("Copying {}", what.display()))
    };
    std::fs::create_dir_all(dst).map_err(|e| io_err(e, dst))?;
    let rd = std::fs::read_dir(src).map_err(|e| io_err(e, src))?;
    for entry in rd {
        let entry = entry.map_err(|e| io_err(e, src))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let ty = entry.file_type().map_err(|e| io_err(e, &from))?;
        if ty.is_dir() {
            copy_tree(&from, &to)?;
        } else if ty.is_file() {
            std::fs::copy(&from, &to).map_err(|e| io_err(e, &from))?;
        }
        // Symlinks are skipped: task content is plain files, and a link could
        // point outside the tree being assembled.
    }
    Ok(())
}

/// Delete `path` if it exists (best-effort, logged on failure).
pub fn remove_tree_best_effort(path: &Path) {
    if path.exists() {
        if let Err(e) = std::fs::remove_dir_all(path) {
            tracing::warn!(path = %path.display(), error = %e, "could not remove tree");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_overwrites_and_adds() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        let over = dir.path().join("over");
        std::fs::create_dir_all(base.join("sub")).unwrap();
        std::fs::write(base.join("keep.txt"), "keep").unwrap();
        std::fs::write(base.join("sub/test.py"), "TAMPERED BY AGENT").unwrap();
        std::fs::create_dir_all(over.join("sub")).unwrap();
        std::fs::write(over.join("sub/test.py"), "real test").unwrap();
        std::fs::write(over.join("new.txt"), "added").unwrap();

        copy_tree(&over, &base).unwrap();

        assert_eq!(std::fs::read_to_string(base.join("keep.txt")).unwrap(), "keep");
        assert_eq!(
            std::fs::read_to_string(base.join("sub/test.py")).unwrap(),
            "real test",
            "overlay must stomp agent edits to hidden files"
        );
        assert_eq!(std::fs::read_to_string(base.join("new.txt")).unwrap(), "added");
    }

    #[test]
    fn copy_tree_roundtrip_nested() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        std::fs::create_dir_all(src.join("a/b/c")).unwrap();
        std::fs::write(src.join("a/b/c/deep.txt"), "deep").unwrap();
        let dst = dir.path().join("dst");
        copy_tree(&src, &dst).unwrap();
        assert_eq!(
            std::fs::read_to_string(dst.join("a/b/c/deep.txt")).unwrap(),
            "deep"
        );
    }
}
