@AGENTS.md

## Claude Code Session Workflow

- Local read-only upstream checkouts live outside this repository at `/home/chewbaccaun/refs/` (`xcl`, `VIctory`, `Scarlight`). Treat them as strictly read-only and never commit or copy them wholesale into this repo.
- Read `IMPLEMENTATION_REUSE_PLAN.md` and `HANDOFF.md` before scaffolding a new module or porting code from `xcl-dev/xcl`, `Marvisak/VIctory`, `pablogonz12/Scarlight`, or `WinDurango`.
- Enforce the **Absolute Neutrality Rule**: never mention any commercial game title, franchise, publisher, studio, proprietary engine, or single-title archive fingerprint in code, docs, tests, or commits.
- Never test against or request commercial software files; write synthetic unit tests, hand-assembled RDNA2 microcode tests (`x2l-rdna2-spirv`), or test cases inside `crates/x2l-testtitle`.
- Before concluding any coding task:
  1. Run `cargo fmt --all -- --check` and `cargo clippy --workspace --all-targets -- -D warnings`.
  2. Run `cargo test --workspace` and `cargo run -p x2l-cli -- --help`.
  3. Run `cargo deny check licenses` when `Cargo.toml` or `Cargo.lock` changes.
  4. Confirm every adapted upstream file carries its GPL-3.0 Section 5 copyright + dated modification header and is recorded in `THIRD_PARTY.md` and `NOTICE`.
