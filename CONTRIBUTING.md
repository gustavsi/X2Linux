# Contributing to X2Linux

## Hard Clean-Room & Neutrality Rules
1. **Absolute Neutrality Rule:** Never mention **any** commercial game title, franchise, abbreviation, character/location name, publisher, studio, proprietary game engine, or single-title archive/shader fingerprint anywhere in the repository (code, comments, docs, tests, issues, PRs, or commit messages). Use strictly generic terms: *"unpacked GDK/Scarlett software"*, *"console titles"*, *"archive files"*, *"compressed audio containers"*, and *"console shader containers"*.
2. **No leaked or NDA SDK material:** Write all new signatures from public `docs.xbox.com`, `learn.microsoft.com`, MIT `DirectX-Headers`/`libHttpClient`, AMD public ISA specs, or sanitized `report.json` telemetry.
3. **No commercial software files, keys, firmware, shader dumps, or single-title hacks (`victory-game`):** Run `cargo xtask install-hooks` immediately after cloning.
4. **No DRM decryption or `XStore` license spoofing.**

## License Compliance (`GPL-3.0-or-later`)
- **Allowed:** `GPL-3.0-or-later`, `GPL-3.0-only`, `LGPL-3.0-*`, `MIT`, `BSD-2-Clause`, `BSD-3-Clause`, `Zlib`, `ISC`, `LGPL-2.1-*`, `Apache-2.0`, `Unicode-3.0`, `CC0-1.0`, `GPL-2.0-or-later`.
- **Allowed upstream code sources:** `xcl-dev/xcl` (`GPL-3.0-or-later`, post-relicense), `Marvisak/VIctory` (`GPL-3.0-or-later`, excluding `victory-game`/`victory-loader`), `pablogonz12/Scarlight` (`GPL-3.0-or-later`), and `WinDurango` (`LGPL-3.0`).
- **Forbidden for code copying:** `GPL-2.0-only` repositories (including `boykopovar/AnyPS5`) are legally incompatible with GPL-3.0 — NEVER copy or derive code from them.
- **GPL-3.0 Section 5 Attribution:** Every file adapted from `xcl-dev/xcl`, `Marvisak/VIctory`, `pablogonz12/Scarlight`, or `WinDurango` must retain original copyright headers, add a dated modification notice, and be recorded in `THIRD_PARTY.md` and `NOTICE`.

## Before Submitting a PR
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo run -p x2l-cli -- --help
cargo deny check licenses
```
