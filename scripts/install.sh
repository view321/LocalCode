#!/usr/bin/env bash
# LocalCode installer (Linux / macOS)
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/view321/LocalCode/main/scripts/install.sh | bash
#   LOCALCODE_INSTALL_DIR=/opt/localcode bash install.sh
set -euo pipefail

REPO_URL="${LOCALCODE_REPO_URL:-https://github.com/view321/LocalCode.git}"
REPO_BRANCH="${LOCALCODE_BRANCH:-main}"
INSTALL_DIR="${LOCALCODE_INSTALL_DIR:-${HOME}/.local/share/localcode}"
BIN_DIR="${LOCALCODE_BIN_DIR:-${HOME}/.local/bin}"
BINARY_NAME="localcode"

info()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33mwarn\033[0m %s\n' "$*"; }
error() { printf '\033[1;31merror\033[0m %s\n' "$*" >&2; exit 1; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || error "Missing required command: $1"
}

ensure_rust() {
  if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    info "Found Rust $(rustc --version)"
    return
  fi

  info "Rust not found — installing via rustup (default toolchain)"
  need_cmd curl
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  if [[ -f "${HOME}/.cargo/env" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/.cargo/env"
  fi
  command -v cargo >/dev/null 2>&1 || error "cargo still not on PATH after rustup install"
  info "Installed $(rustc --version)"
}

clone_or_update() {
  need_cmd git
  mkdir -p "$(dirname "$INSTALL_DIR")"

  if [[ -d "${INSTALL_DIR}/.git" ]]; then
    info "Updating existing checkout at ${INSTALL_DIR}"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REPO_BRANCH"
    git -C "$INSTALL_DIR" checkout -B "$REPO_BRANCH" "origin/${REPO_BRANCH}"
  else
    # Never delete a non-empty directory we don't recognize — a mistyped
    # LOCALCODE_INSTALL_DIR must not wipe user data.
    if [[ -d "$INSTALL_DIR" ]] && [[ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]]; then
      error "${INSTALL_DIR} exists, is not empty, and is not a LocalCode checkout. Move it or set LOCALCODE_INSTALL_DIR elsewhere."
    fi
    info "Cloning LocalCode into ${INSTALL_DIR}"
    rm -rf "$INSTALL_DIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
  fi
}

build_and_install() {
  need_cmd cargo
  mkdir -p "$BIN_DIR"

  info "Building localcode (release) — this may take a few minutes"
  (
    cd "$INSTALL_DIR"
    cargo build --release -p localcode-cli
  )

  local built="${INSTALL_DIR}/target/release/${BINARY_NAME}"
  [[ -x "$built" ]] || error "Build succeeded but binary not found at ${built}"

  install -m 755 "$built" "${BIN_DIR}/${BINARY_NAME}"
  info "Installed ${BIN_DIR}/${BINARY_NAME}"
}

path_hint() {
  if echo ":${PATH}:" | grep -q ":${BIN_DIR}:"; then
    info "${BIN_DIR} is already on PATH"
    return
  fi

  warn "${BIN_DIR} is not on your PATH"
  cat <<EOF

Add this to your shell profile (~/.bashrc, ~/.zshrc, etc.):

  export PATH="${BIN_DIR}:\$PATH"

Then reload the shell or run: source ~/.bashrc
EOF
}

verify() {
  local bin="${BIN_DIR}/${BINARY_NAME}"
  if [[ -x "$bin" ]]; then
    info "Done. Run: ${BINARY_NAME}"
    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
      "$BINARY_NAME" --help 2>/dev/null | head -n 5 || true
    else
      "$bin" --help 2>/dev/null | head -n 5 || true
    fi
  else
    error "Install failed: ${bin} is not executable"
  fi
}

main() {
  info "LocalCode installer"
  ensure_rust
  clone_or_update
  build_and_install
  path_hint
  verify
}

main "$@"
