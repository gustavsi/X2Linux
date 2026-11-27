//! Integration test running `x2l-resolver` against the cross-compiled synthetic
//! `x2l-testtitle.exe` PE32+ console fixture.

use std::path::PathBuf;
use x2l_resolver::{ImportPolicy, ImportResolution, inspect_pe_file};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn find_built_testtitle() -> Option<PathBuf> {
    let base = workspace_root().join("crates/x2l-testtitle/target/x86_64-pc-windows-gnu");
    for profile in ["debug", "release"] {
        let candidate = base.join(profile).join("x2l-testtitle.exe");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

#[test]
fn inspects_cross_compiled_synthetic_testtitle() {
    let Some(exe_path) = find_built_testtitle() else {
        if std::env::var_os("X2L_REQUIRE_TESTTITLE").is_some() {
            panic!(
                "X2L_REQUIRE_TESTTITLE is set but x2l-testtitle.exe was not built; run ./scripts/build_testtitle.sh first"
            );
        }
        eprintln!(
            "[x2l-resolver] Skipping x2l-testtitle.exe fixture check (run ./scripts/build_testtitle.sh to enable)"
        );
        return;
    };

    let root = workspace_root();
    let policy = ImportPolicy::load_from_paths(
        Some(&root.join("data/aliases.toml")),
        Some(&root.join("data/ordinals")),
    )
    .expect("load repo policy files");

    let report = inspect_pe_file(&exe_path, &policy).expect("inspect x2l-testtitle.exe");

    assert_eq!(report.machine, "0x8664", "must be AMD64 PE32+");
    assert!(!report.is_dll, "x2l-testtitle.exe must be an EXE");
    assert!(
        report.console_dlls.contains(&"kernelx.dll".to_owned()),
        "expected kernelx.dll in console_dlls: {:?}",
        report.console_dlls
    );
    assert!(
        report.console_dlls.contains(&"xgameruntime.dll".to_owned()),
        "expected xgameruntime.dll in console_dlls: {:?}",
        report.console_dlls
    );
    assert!(
        report.console_dlls.contains(&"d3d12_xs.dll".to_owned()),
        "expected d3d12_xs.dll in console_dlls: {:?}",
        report.console_dlls
    );

    assert!(
        report
            .imports
            .iter()
            .any(|i| i.dll.eq_ignore_ascii_case("kernelx.dll") && i.symbol == "XMemAlloc"),
        "expected kernelx.dll!XMemAlloc import"
    );
    assert!(
        report.imports.iter().any(|i| {
            i.dll.eq_ignore_ascii_case("xgameruntime.dll")
                && i.symbol == "XGameRuntimeInitialize"
                && matches!(
                    &i.resolution,
                    ImportResolution::Stub { default_return, .. } if default_return == "E_NOTIMPL"
                )
        }),
        "expected xgameruntime.dll!XGameRuntimeInitialize stub with E_NOTIMPL"
    );
    assert!(
        report.imports.iter().any(|i| {
            i.dll.eq_ignore_ascii_case("d3d12_xs.dll")
                && i.symbol == "D3D12CreateDevice"
                && matches!(
                    &i.resolution,
                    ImportResolution::Stub { default_return, .. } if default_return == "E_NOTIMPL"
                )
        }),
        "expected d3d12_xs.dll!D3D12CreateDevice stub with E_NOTIMPL"
    );
    assert!(
        report.imports.iter().any(|i| {
            i.dll.eq_ignore_ascii_case("kernelx.dll")
                && i.symbol == "#999"
                && matches!(
                    &i.resolution,
                    ImportResolution::UnknownOrdinal { ordinal: 999, .. }
                )
        }),
        "expected kernelx.dll ordinal #999 classified as UnknownOrdinal"
    );
    assert!(
        report.tls.is_some(),
        "expected static TLS directory in x2l-testtitle.exe"
    );
}
