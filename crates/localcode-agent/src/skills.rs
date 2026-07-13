//! Skills discovery (SKILL.md folders).

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
        // Bundled sample skill
        self.skills.push(Skill {
            name: "localcode-doctor".into(),
            description: "Diagnose LocalCode backend and config issues".into(),
            path: PathBuf::from("bundled/localcode-doctor"),
            body: "When the user has backend/deploy issues, run through GPU, PATH, ports, and HF token checks.".into(),
            enabled: true,
        });

        if !self.dir.exists() {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                let skill_md = e.path().join("SKILL.md");
                if skill_md.is_file() {
                    if let Ok(body) = std::fs::read_to_string(&skill_md) {
                        let name = e.file_name().to_string_lossy().to_string();
                        let description = body
                            .lines()
                            .find(|l| l.starts_with("description:") || l.starts_with("# "))
                            .unwrap_or("User skill")
                            .trim_start_matches("description:")
                            .trim_start_matches('#')
                            .trim()
                            .to_string();
                        debug!(%name, "loaded skill");
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
        }
    }

    pub fn list(&self) -> &[Skill] {
        &self.skills
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
