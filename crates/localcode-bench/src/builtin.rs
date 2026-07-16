//! The built-in "smoke" suite, embedded at compile time from
//! `bench/suites/smoke/` in the repo and materialized into the user's suites
//! directory on demand.
//!
//! Smoke is managed by LocalCode: it is (re)written on every materialize so
//! upgrades propagate. To customize a task, copy the suite directory under a
//! new id and edit the copy.

use localcode_core::error::{ErrorCode, LocalCodeError};
use std::path::{Path, PathBuf};

pub const SMOKE_SUITE_ID: &str = "smoke";

macro_rules! smoke_file {
    ($rel:literal) => {
        ($rel, include_str!(concat!("../../../bench/suites/smoke/", $rel)))
    };
}

/// Every file of the smoke suite, workspace-relative → contents.
const SMOKE_FILES: &[(&str, &str)] = &[
    smoke_file!("suite.toml"),
    // Python — implement a .env parser.
    smoke_file!("tasks/py-dotenv/task.toml"),
    smoke_file!("tasks/py-dotenv/workspace/dotenv_parser.py"),
    smoke_file!("tasks/py-dotenv/hidden/test_dotenv.py"),
    smoke_file!("tasks/py-dotenv/solution/dotenv_parser.py"),
    // Python — fix an LRU cache.
    smoke_file!("tasks/py-lru/task.toml"),
    smoke_file!("tasks/py-lru/workspace/lru.py"),
    smoke_file!("tasks/py-lru/hidden/test_lru.py"),
    smoke_file!("tasks/py-lru/solution/lru.py"),
    // JavaScript — implement slugify.
    smoke_file!("tasks/js-slugify/task.toml"),
    smoke_file!("tasks/js-slugify/workspace/slugify.js"),
    smoke_file!("tasks/js-slugify/hidden/slugify.test.js"),
    smoke_file!("tasks/js-slugify/solution/slugify.js"),
    // JavaScript — fix debounce.
    smoke_file!("tasks/js-debounce/task.toml"),
    smoke_file!("tasks/js-debounce/workspace/debounce.js"),
    smoke_file!("tasks/js-debounce/hidden/debounce.test.js"),
    smoke_file!("tasks/js-debounce/solution/debounce.js"),
    // Rust — implement stats helpers.
    smoke_file!("tasks/rs-stats/task.toml"),
    smoke_file!("tasks/rs-stats/workspace/Cargo.toml"),
    smoke_file!("tasks/rs-stats/workspace/src/lib.rs"),
    smoke_file!("tasks/rs-stats/hidden/tests/stats_test.rs"),
    smoke_file!("tasks/rs-stats/solution/src/lib.rs"),
    // Rust — fix bracket matching.
    smoke_file!("tasks/rs-brackets/task.toml"),
    smoke_file!("tasks/rs-brackets/workspace/Cargo.toml"),
    smoke_file!("tasks/rs-brackets/workspace/src/lib.rs"),
    smoke_file!("tasks/rs-brackets/hidden/tests/brackets_test.rs"),
    smoke_file!("tasks/rs-brackets/solution/src/lib.rs"),
    // C — implement run-length encoding.
    smoke_file!("tasks/c-rle/task.toml"),
    smoke_file!("tasks/c-rle/workspace/rle.h"),
    smoke_file!("tasks/c-rle/workspace/rle.c"),
    smoke_file!("tasks/c-rle/hidden/test_rle.c"),
    smoke_file!("tasks/c-rle/solution/rle.c"),
    // C — fix a ring buffer.
    smoke_file!("tasks/c-ringbuf/task.toml"),
    smoke_file!("tasks/c-ringbuf/workspace/ringbuf.h"),
    smoke_file!("tasks/c-ringbuf/workspace/ringbuf.c"),
    smoke_file!("tasks/c-ringbuf/hidden/test_ringbuf.c"),
    smoke_file!("tasks/c-ringbuf/solution/ringbuf.c"),
];

/// Write (or refresh) the smoke suite under `suites_root` and return its path.
pub fn ensure_smoke_suite(suites_root: &Path) -> Result<PathBuf, LocalCodeError> {
    let root = suites_root.join(SMOKE_SUITE_ID);
    for (rel, content) in SMOKE_FILES {
        let path = root.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
                    .with_cause(format!("Creating {}", parent.display()))
            })?;
        }
        // Skip the write when identical — keeps mtimes stable.
        if std::fs::read_to_string(&path).map(|c| c == *content).unwrap_or(false) {
            continue;
        }
        std::fs::write(&path, content).map_err(|e| {
            LocalCodeError::new(ErrorCode::ConfigSaveFailed, e.to_string())
                .with_cause(format!("Writing {}", path.display()))
        })?;
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::load_bench_suite;

    /// The embedded suite must materialize into something the loader accepts —
    /// this guards every task manifest and file in `bench/suites/smoke/` at
    /// test time.
    #[test]
    fn smoke_materializes_and_loads() {
        let dir = tempfile::tempdir().unwrap();
        let root = ensure_smoke_suite(dir.path()).unwrap();
        let suite = load_bench_suite(&root).unwrap();
        assert_eq!(suite.meta.id, SMOKE_SUITE_ID);
        assert_eq!(suite.tasks.len(), 8);
        for task in &suite.tasks {
            assert!(task.has_solution(), "{} needs a solution/ for verify", task.spec.id);
            assert!(task.hidden_dir().is_dir(), "{} needs hidden/ tests", task.spec.id);
            assert_eq!(task.spec.network, "none", "{} must stay offline", task.spec.id);
        }
        // Materialize is idempotent.
        ensure_smoke_suite(dir.path()).unwrap();
        assert_eq!(load_bench_suite(&root).unwrap().tasks.len(), 8);
    }
}
