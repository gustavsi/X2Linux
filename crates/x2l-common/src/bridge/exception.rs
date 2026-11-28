/// Original exception snapshot prefix. Its layout is fixed for older shims and cores.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CrashSnapshot {
    pub code: u32,
    pub has_control: u32,
    pub has_integer: u32,
    pub bt_count: u32,
    pub address: u64,
    pub rva: u64,
    pub module_ptr: u64,
    pub module_len: u64,
    pub rip: u64,
    pub rsp: u64,
    pub rax: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub r8: u64,
    pub r9: u64,
    pub bt_rvas: [u64; 8],
}

/// Extended exception snapshot. The bridge's a5 argument carries its size.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CrashSnapshotV2 {
    pub base: CrashSnapshot,
    pub has_fault: u32,
    pub rip_bytes_len: u32,
    pub fault_access: u64,
    pub fault_address: u64,
    pub rip_bytes: [u8; 16],
}

// Check the layout in both the native and PE builds.
const _: () = {
    assert!(std::mem::size_of::<CrashSnapshot>() == 168);
    assert!(std::mem::offset_of!(CrashSnapshot, bt_rvas) == 104);
    assert!(std::mem::size_of::<CrashSnapshotV2>() == 208);
    assert!(std::mem::offset_of!(CrashSnapshotV2, base) == 0);
    assert!(std::mem::offset_of!(CrashSnapshotV2, has_fault) == 168);
    assert!(std::mem::offset_of!(CrashSnapshotV2, rip_bytes_len) == 172);
    assert!(std::mem::offset_of!(CrashSnapshotV2, fault_access) == 176);
    assert!(std::mem::offset_of!(CrashSnapshotV2, fault_address) == 184);
    assert!(std::mem::offset_of!(CrashSnapshotV2, rip_bytes) == 192);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_snapshot_starts_with_the_legacy_payload() {
        let snap = CrashSnapshotV2 {
            base: CrashSnapshot {
                code: 0xc000_0005,
                rip: 0x1234,
                ..Default::default()
            },
            has_fault: 1,
            ..Default::default()
        };
        assert_eq!(std::mem::size_of_val(&snap), 208);
        assert_eq!(std::mem::size_of_val(&snap.base), 168);
        assert_eq!(
            std::ptr::from_ref(&snap).cast(),
            std::ptr::from_ref(&snap.base)
        );
    }
}
