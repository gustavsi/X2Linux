//! PE32+ read-only inspector, API-set alias resolver, and ordinal policy engine for X2Linux.

pub mod inspect;
pub mod pe;
pub mod policy;

pub use inspect::{
    ImportCounts, InspectError, InspectReport, InspectedImport, TlsSummary, inspect_pe_bytes,
    inspect_pe_file,
};
pub use pe::{
    DataDir, IMAGE_FILE_MACHINE_AMD64, ImportDll, ImportEntry, ImportSymbol, PE32_PLUS_MAGIC,
    PeError, PeImage, Section, TlsInfo,
};
pub use policy::{DllRoute, ImportPolicy, ImportResolution, OrdinalEntry, PolicyError};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_dir(label: &str) -> PathBuf {
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("x2l-resolver-{label}-{}-{id}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn rejects_malformed_truncated_and_non_pe_inputs_without_panicking() {
        let cases: &[&[u8]] = &[
            &[],
            b"MZ",
            b"not a pe file at all",
            b"XVC0encrypted-container-header-bytes-0123456789abcdef0123456789abcdef",
            &[0x4D, 0x5A, 0x90, 0x00],
        ];
        for bytes in cases {
            assert!(PeImage::parse(bytes).is_err());
        }

        // Valid MZ + e_lfanew pointing beyond EOF.
        let mut bad_lfanew = vec![0u8; 0x80];
        bad_lfanew[0..2].copy_from_slice(b"MZ");
        bad_lfanew[0x3C..0x40].copy_from_slice(&0xFFFF_FFF0u32.to_le_bytes());
        assert!(PeImage::parse(&bad_lfanew).is_err());

        // Valid MZ + PE signature + non-AMD64 machine (x86 0x014C).
        let mut x86_pe = vec![0u8; 0x200];
        x86_pe[0..2].copy_from_slice(b"MZ");
        x86_pe[0x3C..0x40].copy_from_slice(&0x40u32.to_le_bytes());
        x86_pe[0x40..0x44].copy_from_slice(b"PE\0\0");
        x86_pe[0x44..0x46].copy_from_slice(&0x014Cu16.to_le_bytes());
        x86_pe[0x54..0x56].copy_from_slice(&0xF0u16.to_le_bytes());
        x86_pe[0x58..0x5A].copy_from_slice(&0x010Bu16.to_le_bytes());
        assert_eq!(
            PeImage::parse(&x86_pe).unwrap_err(),
            PeError::NotAmd64(0x014C)
        );
    }

    #[test]
    fn resolves_aliases_ordinals_defaults_and_unknown_ordinals() {
        let mut policy = ImportPolicy::default();
        policy
            .load_aliases_str(
                r#"
                [exact]
                "kernelx.dll" = "hle:kernelx"
                "xgameruntime.dll" = "hle:xgameruntime"

                [prefix]
                "api-ms-win-core-memory-" = "hle:kernelx"
                "api-ms-win-crt-" = "system:ucrtbase"

                [module_defaults]
                "xgameruntime.dll" = "E_NOTIMPL"

                [implemented]
                kernelx = ["XMemAlloc"]
                "#,
                "test-aliases",
            )
            .unwrap();
        policy
            .load_ordinals_str(
                "kernelx.dll",
                r#"
                [ordinals]
                "1" = { name = "XMemAlloc" }
                "2" = { name = "XMemProbeUnimplemented", default_return = "0x80004001" }
                "#,
                "test-ordinals",
            )
            .unwrap();

        // Implemented via exact match.
        assert_eq!(
            policy.classify(
                "KERNELX.DLL",
                &ImportSymbol::Name {
                    hint: 0,
                    name: "XMemAlloc".into()
                },
                None
            ),
            ImportResolution::Implemented {
                module: "kernelx".into(),
                resolved_name: "XMemAlloc".into(),
            }
        );

        // Implemented via API-set prefix + ordinal lookup.
        assert_eq!(
            policy.classify("kernelx.dll", &ImportSymbol::Ordinal { ordinal: 1 }, None),
            ImportResolution::Implemented {
                module: "kernelx".into(),
                resolved_name: "XMemAlloc".into(),
            }
        );

        // Known ordinal but not implemented -> Stub with per-ordinal default return.
        assert_eq!(
            policy.classify("kernelx.dll", &ImportSymbol::Ordinal { ordinal: 2 }, None),
            ImportResolution::Stub {
                module: "kernelx".into(),
                resolved_name: "XMemProbeUnimplemented".into(),
                default_return: "0x80004001".into(),
            }
        );

        // Unknown ordinal -> UnknownOrdinal.
        assert_eq!(
            policy.classify("kernelx.dll", &ImportSymbol::Ordinal { ordinal: 999 }, None),
            ImportResolution::UnknownOrdinal {
                module: "kernelx".into(),
                ordinal: 999,
                default_return: "0x0".into(),
            }
        );

        // Module default E_NOTIMPL for HRESULT console DLL.
        assert_eq!(
            policy.classify(
                "xgameruntime.dll",
                &ImportSymbol::Name {
                    hint: 0,
                    name: "XGameRuntimeInitialize".into()
                },
                None
            ),
            ImportResolution::Stub {
                module: "xgameruntime".into(),
                resolved_name: "XGameRuntimeInitialize".into(),
                default_return: "E_NOTIMPL".into(),
            }
        );

        // Prefix rule -> SystemFallback.
        assert_eq!(
            policy.classify(
                "api-ms-win-crt-runtime-l1-1-0.dll",
                &ImportSymbol::Name {
                    hint: 0,
                    name: "_initterm".into()
                },
                None
            ),
            ImportResolution::SystemFallback {
                target: "ucrtbase".into(),
                resolved_name: "_initterm".into(),
            }
        );
    }

    #[test]
    fn detects_local_dll_and_parses_synthetic_pe32_plus_image() {
        let dir = temp_dir("local-dll");
        let local_dll = dir.join("TitleHelper.dll");
        fs::write(&local_dll, b"stub").unwrap();

        let policy = ImportPolicy::builtin_defaults();
        let res = policy.classify(
            "titlehelper.dll",
            &ImportSymbol::Name {
                hint: 0,
                name: "HelperInit".into(),
            },
            Some(&dir),
        );
        assert_eq!(
            res,
            ImportResolution::GameLocal {
                path: local_dll.display().to_string()
            }
        );

        // Construct a minimal synthetic PE32+ AMD64 image with regular + delay imports and TLS.
        let image = build_synthetic_pe64();
        let report = inspect_pe_bytes(&image, "synthetic.exe", Some(&dir), &policy)
            .expect("parse synthetic PE32+");

        assert_eq!(report.machine, "0x8664");
        assert!(!report.is_dll);
        assert!(report.has_relocations);
        assert_eq!(report.tls.as_ref().map(|t| t.template_bytes), Some(16));
        assert!(report.console_dlls.contains(&"kernelx.dll".to_owned()));
        assert!(report.console_dlls.contains(&"d3d12_xs.dll".to_owned()));
        assert_eq!(report.counts.unknown_ordinals, 1);
        assert!(
            report
                .imports
                .iter()
                .any(|i| i.delay_load && i.dll == "d3d12_xs.dll")
        );

        let json = report.to_json_pretty().expect("serialize InspectReport");
        let parsed: InspectReport = serde_json::from_str(&json).expect("deserialize InspectReport");
        assert_eq!(parsed, report);

        let telemetry = report.to_telemetry_report("synthetic");
        assert_eq!(telemetry.unknown_ordinals.len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }

    /// Build a minimal valid PE32+ AMD64 image in memory for deterministic unit testing.
    fn build_synthetic_pe64() -> Vec<u8> {
        // Layout:
        //   0x000..0x200: DOS + NT + Optional + 1 section header (.rdata at RVA 0x1000, raw 0x200)
        //   0x200..0x600: .rdata content (RVA 0x1000..0x1400)
        let mut buf = vec![0u8; 0x600];
        let write_u16 =
            |b: &mut [u8], o: usize, v: u16| b[o..o + 2].copy_from_slice(&v.to_le_bytes());
        let write_u32 =
            |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        let write_u64 =
            |b: &mut [u8], o: usize, v: u64| b[o..o + 8].copy_from_slice(&v.to_le_bytes());

        // DOS header
        buf[0..2].copy_from_slice(b"MZ");
        write_u32(&mut buf, 0x3C, 0x40);

        // NT signature + COFF header at 0x40
        buf[0x40..0x44].copy_from_slice(b"PE\0\0");
        write_u16(&mut buf, 0x44, IMAGE_FILE_MACHINE_AMD64);
        write_u16(&mut buf, 0x46, 1); // 1 section
        write_u16(&mut buf, 0x54, 0xF0); // OptionalHeader size
        write_u16(&mut buf, 0x56, 0x0022); // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE

        // Optional header at 0x58
        let opt = 0x58;
        write_u16(&mut buf, opt, PE32_PLUS_MAGIC);
        write_u32(&mut buf, opt + 16, 0x1000); // entry_rva
        write_u64(&mut buf, opt + 24, 0x0000_0001_4000_0000); // image_base
        write_u32(&mut buf, opt + 56, 0x2000); // size_of_image
        write_u32(&mut buf, opt + 60, 0x200); // size_of_headers
        write_u16(&mut buf, opt + 68, 3); // CUI subsystem
        write_u64(&mut buf, opt + 72, 0x10_0000); // stack_reserve
        write_u32(&mut buf, opt + 108, 16); // NumberOfRvaAndSizes

        let dirs = opt + 112;
        // DIR_IMPORT (index 1): RVA 0x1000, size 40
        write_u32(&mut buf, dirs + 8, 0x1000);
        write_u32(&mut buf, dirs + 12, 40);
        // DIR_BASERELOC (index 5): RVA 0x1380, size 8
        write_u32(&mut buf, dirs + 40, 0x1380);
        write_u32(&mut buf, dirs + 44, 8);
        // DIR_TLS (index 9): RVA 0x1300, size 40
        write_u32(&mut buf, dirs + 72, 0x1300);
        write_u32(&mut buf, dirs + 76, 40);
        // DIR_DELAY_IMPORT (index 13): RVA 0x1200, size 64
        write_u32(&mut buf, dirs + 104, 0x1200);
        write_u32(&mut buf, dirs + 108, 64);

        // Section header at 0x58 + 0xF0 = 0x148
        let sec = opt + 0xF0;
        buf[sec..sec + 6].copy_from_slice(b".rdata");
        write_u32(&mut buf, sec + 8, 0x400); // virtual_size
        write_u32(&mut buf, sec + 12, 0x1000); // virtual_address
        write_u32(&mut buf, sec + 16, 0x400); // raw_size
        write_u32(&mut buf, sec + 20, 0x200); // raw_offset

        let rdata = 0x200; // corresponds to RVA 0x1000
        // Regular import descriptor 0 (kernelx.dll): INT at 0x1080, name at 0x1100, IAT at 0x1080
        write_u32(&mut buf, rdata, 0x1080);
        write_u32(&mut buf, rdata + 12, 0x1100);
        write_u32(&mut buf, rdata + 16, 0x1080);
        // INT entries at RVA 0x1080 (file 0x280): named "XMemAlloc" + ordinal #777
        write_u64(&mut buf, rdata + 0x80, 0x1120);
        write_u64(&mut buf, rdata + 0x88, (1u64 << 63) | 777);
        // Strings
        buf[rdata + 0x100..rdata + 0x10C].copy_from_slice(b"kernelx.dll\0");
        write_u16(&mut buf, rdata + 0x120, 0);
        buf[rdata + 0x122..rdata + 0x12C].copy_from_slice(b"XMemAlloc\0");

        // Delay import descriptor 0 at RVA 0x1200 (file 0x400): attrs=1 (RVA-based), d3d12_xs.dll
        write_u32(&mut buf, rdata + 0x200, 1);
        write_u32(&mut buf, rdata + 0x204, 0x1260); // dll name
        write_u32(&mut buf, rdata + 0x20C, 0x1240); // IAT
        write_u32(&mut buf, rdata + 0x210, 0x1240); // INT
        write_u64(&mut buf, rdata + 0x240, 0x1280); // thunk -> D3D12CreateDevice
        buf[rdata + 0x260..rdata + 0x26D].copy_from_slice(b"d3d12_xs.dll\0");
        write_u16(&mut buf, rdata + 0x280, 0);
        buf[rdata + 0x282..rdata + 0x294].copy_from_slice(b"D3D12CreateDevice\0");

        // TLS directory at RVA 0x1300 (file 0x500): 16-byte raw template
        write_u64(&mut buf, rdata + 0x300, 0x0000_0001_4000_1340);
        write_u64(&mut buf, rdata + 0x308, 0x0000_0001_4000_1350);
        write_u32(&mut buf, rdata + 0x320, 8); // zero-fill

        buf
    }
}
