//! Skills discovery (SKILL.md folders with optional YAML frontmatter).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub body: String,
    pub enabled: bool,
}

pub struct SkillLoader {
    dir: PathBuf,
    skills: Vec<Skill>,
}

impl SkillLoader {
    pub fn new(dir: PathBuf) -> Self {
        let mut s = Self {
            dir,
            skills: vec![],
        };
        s.reload();
        s
    }

    pub fn reload(&mut self) {
        self.skills.clear();

        // Bundled sample skill (always present).
        self.skills.push(Skill {
            name: "localcode-doctor".into(),
            description: "Diagnose LocalCode backend and config issues".into(),
            path: PathBuf::from("bundled/localcode-doctor"),
            body: concat!(
                "# localcode-doctor\n\n",
                "When the user has backend/deploy issues:\n",
                "1. Check GPU discovery (`nvidia-smi` via bash if available).\n",
                "2. Verify the active backend is running and healthy.\n",
                "3. If a download failed, check ports, PATH, and network first. An HF token \
matters only for gated models (401/403); public models need none. If huggingface.co does \
not respond, use the mirror hf-mirror.com (HF_ENDPOINT=https://hf-mirror.com).\n",
                "4. Suggest `/doctor` and concrete config fixes.\n",
            )
            .into(),
            enabled: true,
        });

        if !self.dir.exists() {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                let skill_md = e.path().join("SKILL.md");
                if !skill_md.is_file() {
                    continue;
                }
                let Ok(raw) = std::fs::read_to_string(&skill_md) else {
                    continue;
                };
                let folder = e.file_name().to_string_lossy().to_string();
                let (meta_name, meta_desc, body) = parse_skill_md(&raw);
                let name = meta_name.unwrap_or(folder);
                let description = meta_desc.unwrap_or_else(|| {
                    body.lines()
                        .find(|l| l.starts_with("# "))
                        .map(|l| l.trim_start_matches('#').trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| "User skill".into())
                });
                debug!(%name, "loaded skill");
                // User skills override the bundled one with the same name.
                self.skills.retain(|s| s.name != name);
                self.skills.push(Skill {
                    name,
                    description,
                    path: e.path(),
                    body,
                    enabled: true,
                });
            }
        }
    }

    pub fn list(&self) -> &[Skill] {
        &self.skills
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name && s.enabled)
    }

    pub fn set_enabled(&mut self, name: &str, enabled: bool) {
        if let Some(s) = self.skills.iter_mut().find(|s| s.name == name) {
            s.enabled = enabled;
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Parse optional YAML-ish frontmatter between `---` fences.
/// Supports `name:` and `description:` keys.
fn parse_skill_md(raw: &str) -> (Option<String>, Option<String>, String) {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return (None, None, raw.to_string());
    }
    let rest = &trimmed[3..];
    let Some(end) = rest.find("\n---") else {
        return (None, None, raw.to_string());
    };
    let front = &rest[..end];
    let body = rest[end + 4..].trim_start_matches('\r').trim_start_matches('\n');
    let mut name = None;
    let mut description = None;
    for line in front.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = Some(unquote(v.trim()));
        } else if let Some(v) = line.strip_prefix("description:") {
            description = Some(unquote(v.trim()));
        }
    }
    (name, description, body.to_string())
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    // `len() >= 2` guards the single-character case: a lone `"` (or `'`) is
    // both `starts_with` and `ends_with` the same byte, so `s[1..len-1]` would
    // be `s[1..0]` and panic — a half-edited SKILL.md would then crash every
    // agent turn (SkillLoader runs inside every `CodingAgent::new`).
    if s.len() >= 2
        && ((s.starts_with('"') && s.ends_with('"'))
            || (s.starts_with('\'') && s.ends_with('\'')))
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn parses_frontmatter() {
        let raw = "---\nname: my-skill\ndescription: Does a thing\n---\n\n# Body\n\nHello.";
        let (n, d, b) = parse_skill_md(raw);
        assert_eq!(n.as_deref(), Some("my-skill"));
        assert_eq!(d.as_deref(), Some("Does a thing"));
        assert!(b.contains("Hello"));
    }

    #[test]
    fn unquote_handles_degenerate_and_quoted_values() {
        // A lone quote must not panic (half-edited frontmatter).
        assert_eq!(unquote("\""), "\"");
        assert_eq!(unquote("'"), "'");
        assert_eq!(unquote(""), "");
        // Proper quote pairs are stripped; multi-byte inner content is fine.
        assert_eq!(unquote("\"\""), "");
        assert_eq!(unquote("\"hi\""), "hi");
        assert_eq!(unquote("'日本'"), "日本");
        // Unbalanced or bare values pass through unchanged.
        assert_eq!(unquote("\"open"), "\"open");
        assert_eq!(unquote("bare"), "bare");
    }

    #[test]
    fn frontmatter_with_lone_quote_does_not_panic() {
        let (n, _d, _b) = parse_skill_md("---\nname: \"\ndescription: x\n---\nbody");
        assert_eq!(n.as_deref(), Some("\""));
    }

    #[test]
    fn loads_from_dir() {
        let dir = tempdir().unwrap();
        let skill_dir = dir.path().join("cool");
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: cool\ndescription: Cool skill\n---\nDo cool things.\n",
        )
        .unwrap();
        let loader = SkillLoader::new(dir.path().to_path_buf());
        let s = loader.get("cool").expect("cool skill");
        assert_eq!(s.description, "Cool skill");
        assert!(s.body.contains("Do cool things"));
        assert!(loader.get("localcode-doctor").is_some());
    }
}
