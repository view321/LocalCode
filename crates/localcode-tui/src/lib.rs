//! LocalCode TUI.

mod app;
mod doctor;
mod theme;
mod ui;
mod widgets;

pub use app::{run_tui, App};
pub use doctor::run_doctor;
