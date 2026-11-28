// SPDX-License-Identifier: GPL-3.0-or-later

//! Shared Win64 COM interface identifiers and QueryInterface handling.

use crate::com::{E_NOINTERFACE, E_POINTER, S_OK};
use crate::{log_d3d12_trace, trace_d3d12_enabled};
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicPtr, Ordering};

/// Receives the interface name and canonical GUID when a COM interface is rejected.
pub type RejectedQiCallback = fn(interface: &'static str, iid: &str);

static REJECTED_QI_CALLBACK: AtomicPtr<()> = AtomicPtr::new(ptr::null_mut());

/// Installs or clears the process-wide callback for rejected D3D12 interface IDs.
pub fn set_d3d12_rejected_qi_callback(cb: Option<RejectedQiCallback>) {
    let address = cb.map_or(ptr::null_mut(), |cb| cb as *mut ());
    REJECTED_QI_CALLBACK.store(address, Ordering::Release);
}

/// Formats a Windows GUID from its 16-byte in-memory representation.
pub fn format_guid(bytes: &[u8; 16]) -> String {
    let [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p] = *bytes;
    let data1 = u32::from_le_bytes([a, b, c, d]);
    let data2 = u16::from_le_bytes([e, f]);
    let data3 = u16::from_le_bytes([g, h]);
    let data4_head = u16::from_be_bytes([i, j]);
    let data4_tail = u64::from_be_bytes([0, 0, k, l, m, n, o, p]);
    format!("{{{data1:08x}-{data2:04x}-{data3:04x}-{data4_head:04x}-{data4_tail:012x}}}")
}

#[derive(Clone, Copy)]
enum Base {
    Unknown,
    Object,
    DeviceChild,
    Pageable,
}

pub(crate) struct Interface {
    name: &'static str,
    base: Base,
    iids: &'static [[u8; 16]],
}

impl Interface {
    pub(crate) fn supports(&self, iid: &[u8; 16]) -> bool {
        const BASE_IIDS: [[u8; 16]; 4] = [
            guid(0, 0, 0, 0xc000_0000_0000_0046), // IUnknown
            guid(0xc4fec28f, 0x7966, 0x4e95, 0x9f94_f431_cb56_c3b8), // ID3D12Object
            guid(0x905db94b, 0xa00c, 0x4140, 0x9df5_2b64_ca9e_a357), // ID3D12DeviceChild
            guid(0x63ee58fb, 0x1268, 0x4835, 0x86da_f008_ce62_f0d6), // ID3D12Pageable
        ];
        BASE_IIDS[..=self.base as usize].contains(iid) || self.iids.contains(iid)
    }
}

pub(crate) const fn guid(a: u32, b: u16, c: u16, tail: u64) -> [u8; 16] {
    let [g, h] = c.to_le_bytes();
    let [e, f] = b.to_le_bytes();
    let [a, b, c, d] = a.to_le_bytes();
    let [i, j, k, l, m, n, o, p] = tail.to_be_bytes();
    [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p]
}

// Standard IID values from the public Microsoft DirectX-Headers declarations.
pub(crate) const DEVICE: Interface = Interface {
    name: "ID3D12DeviceX",
    base: Base::Object,
    iids: &[
        guid(0x189819f1, 0x1db6, 0x4b57, 0xbe54_1821_339b_85f7), // ID3D12Device
        guid(0x77acce80, 0x638e, 0x4e65, 0x8895_c1f2_3386_863e), // ID3D12Device1
        guid(0x30baa41e, 0xb15b, 0x475c, 0xa0bb_1af5_c5b6_4328), // ID3D12Device2
        guid(0x81dadc15, 0x2bad, 0x4392, 0x93c5_1013_45c4_aa98), // ID3D12Device3
        guid(0xe865df17, 0xa9ee, 0x46f9, 0xa463_3098_315a_a2e5), // ID3D12Device4
        guid(0x8b4f173b, 0x2fea, 0x4b80, 0x8f58_4307_191a_b95d), // ID3D12Device5
        guid(0xc70b221b, 0x40e4, 0x4a17, 0x89af_025a_0727_a6dc), // ID3D12Device6
        guid(0x5c014b53, 0x68a1, 0x4b9b, 0x8bd1_dd60_46b9_358b), // ID3D12Device7
        guid(0x9218e6bb, 0xf944, 0x4f7e, 0xa75c_b1b2_c7b7_01f3), // ID3D12Device8
        guid(0x4c80e962, 0xf032, 0x4f60, 0xbc9e_ebc2_cfa1_d83c), // ID3D12Device9
        guid(0x517f8718, 0xaa66, 0x49f9, 0xb02b_a7ab_89c0_6031), // ID3D12Device10
        guid(0x5405c344, 0xd457, 0x444e, 0xb4dd_2366_e45a_ee39), // ID3D12Device11
        guid(0x5af5c532, 0x4c91, 0x4cd0, 0xb541_15a4_0539_5fc5), // ID3D12Device12
        guid(0x14eecffc, 0x4df8, 0x40f7, 0xa118_5c81_6f45_695e), // ID3D12Device13
        guid(0x5f6e592d, 0xd895, 0x44c2, 0x8e4a_88ad_4926_d323), // ID3D12Device14
    ],
};

pub(crate) const LIST: Interface = Interface {
    name: "ID3D12GraphicsCommandListX",
    base: Base::DeviceChild,
    iids: &[
        guid(0x7116d91c, 0xe7e4, 0x47ce, 0xb8c6_ec81_68f4_37e5), // ID3D12CommandList
        guid(0x5b160d0f, 0xac1b, 0x4185, 0x8ba8_b3ae_42a5_a455), // ID3D12GraphicsCommandList
        guid(0x553103fb, 0x1fe7, 0x4557, 0xbb38_946d_7d0e_7ca7), // ID3D12GraphicsCommandList1
        guid(0x38c3e585, 0xff17, 0x412c, 0x9150_4fc6_f9d7_2a28), // ID3D12GraphicsCommandList2
        guid(0x6fda83a7, 0xb84c, 0x4e38, 0x9ac8_c7bd_2201_6b3d), // ID3D12GraphicsCommandList3
        guid(0x8754318e, 0xd3a9, 0x4541, 0x98cf_645b_50dc_4874), // ID3D12GraphicsCommandList4
        guid(0x55050859, 0x4024, 0x474c, 0x87f5_6472_eaee_44ea), // ID3D12GraphicsCommandList5
        guid(0xc3827890, 0xe548, 0x4cfa, 0x96cf_5689_a937_0f80), // ID3D12GraphicsCommandList6
        guid(0xdd171223, 0x8b61, 0x4769, 0x90e3_160c_cde4_e2c1), // ID3D12GraphicsCommandList7
        guid(0xee936ef9, 0x599d, 0x4d28, 0x938e_23c4_ad05_ce51), // ID3D12GraphicsCommandList8
        guid(0x34ed2808, 0xffe6, 0x4c2b, 0xb11a_cabd_2b0c_59e1), // ID3D12GraphicsCommandList9
        guid(0x7013c015, 0xd161, 0x4b63, 0xa08c_2385_52dd_8acc), // ID3D12GraphicsCommandList10
    ],
};

pub(crate) const QUEUE: Interface = Interface {
    name: "ID3D12CommandQueueX",
    base: Base::Pageable,
    iids: &[
        guid(0x0ec870a6, 0x5d7e, 0x4c22, 0x8cfc_5baa_e076_16ed), // ID3D12CommandQueue
    ],
};

pub(crate) const ALLOCATOR: Interface = Interface {
    name: "ID3D12CommandAllocator",
    base: Base::Pageable,
    iids: &[guid(0x6102dee4, 0xaf59, 0x4b09, 0xb999_b44d_73f0_9b24)],
};

pub(crate) const FENCE: Interface = Interface {
    name: "ID3D12Fence",
    base: Base::Pageable,
    iids: &[
        guid(0x0a753dcf, 0xc4d8, 0x4b91, 0xadf6_be5a_60d9_5a76),
        guid(0x433685fe, 0xe22b, 0x4ca0, 0xa8db_b5b4_f4dd_0e4a), // ID3D12Fence1
    ],
};

pub(crate) const RESOURCE: Interface = Interface {
    name: "ID3D12Resource",
    base: Base::Pageable,
    iids: &[
        guid(0x696442be, 0xa72e, 0x4059, 0xbc79_5b5c_9804_0fad),
        guid(0x9d5e227a, 0x4430, 0x4161, 0x88b3_3eca_6bb1_6e19), // ID3D12Resource1
        guid(0xbe36ec3b, 0xea85, 0x4aeb, 0xa45a_e9d7_6404_a495), // ID3D12Resource2
    ],
};

pub(crate) const HEAP: Interface = Interface {
    name: "ID3D12Heap",
    base: Base::Pageable,
    iids: &[guid(0x6b3b2502, 0x6e51, 0x45b3, 0x90ee_9884_265e_8df3)],
};

pub(crate) const DESCRIPTOR_HEAP: Interface = Interface {
    name: "ID3D12DescriptorHeap",
    base: Base::Pageable,
    iids: &[guid(0x8efb471d, 0x616c, 0x4f49, 0x90f7_127b_b763_fa51)],
};

pub(crate) const ROOT_SIGNATURE: Interface = Interface {
    name: "ID3D12RootSignature",
    base: Base::DeviceChild,
    iids: &[guid(0xc54a6b66, 0x72df, 0x4ee8, 0x8be5_a946_a142_9214)],
};

pub(crate) const PIPELINE_STATE: Interface = Interface {
    name: "ID3D12PipelineState",
    base: Base::Pageable,
    iids: &[guid(0x765a30f3, 0xf624, 0x4c6f, 0xa828_ace9_4862_2445)],
};

pub(crate) const QUERY_HEAP: Interface = Interface {
    name: "ID3D12QueryHeap",
    base: Base::Pageable,
    iids: &[guid(0x0d9658ae, 0xed45, 0x469e, 0xa61d_970e_c583_cab4)],
};

pub(crate) const COMMAND_SIGNATURE: Interface = Interface {
    name: "ID3D12CommandSignature",
    base: Base::Pageable,
    iids: &[guid(0xc36a797c, 0xec80, 0x4f0a, 0x8985_a7b2_4750_82d1)],
};

pub(crate) const BLOB: Interface = Interface {
    name: "ID3DBlob",
    base: Base::Unknown,
    iids: &[guid(0x8ba5fb08, 0x5195, 0x40e2, 0xac58_0d98_9c3a_0102)],
};

pub(crate) fn notify_rejected(interface: &'static str, iid: &str) {
    let address = REJECTED_QI_CALLBACK.load(Ordering::Acquire);
    if !address.is_null() {
        // SAFETY: Every non-null address was stored from a RejectedQiCallback function pointer.
        let callback: RejectedQiCallback = unsafe { std::mem::transmute(address) };
        callback(interface, iid);
    }
}

/// # Safety
/// A non-null IID must point to 16 readable bytes.
pub(crate) unsafe fn check_creation_iid(interface: &Interface, riid: *const c_void) -> bool {
    if riid.is_null() {
        return true;
    }
    // SAFETY: The caller supplies a readable GUID, and the byte array has alignment 1.
    let bytes = unsafe { riid.cast::<[u8; 16]>().read() };
    if interface.supports(&bytes) {
        return true;
    }
    notify_rejected(interface.name, &format_guid(&bytes));
    false
}

/// # Safety
/// Non-null arguments must refer to a live COM object, a readable GUID and a writable out pointer.
pub(crate) unsafe fn query_interface<T>(
    interface: &Interface,
    this: *mut T,
    riid: *const c_void,
    out: *mut *mut c_void,
    add_ref: extern "win64" fn(*mut T) -> u32,
) -> i32 {
    // SAFETY: The caller supplies a readable GUID when riid is non-null.
    let iid = unsafe { riid.cast::<[u8; 16]>().as_ref() };
    let hr = if out.is_null() {
        E_POINTER
    } else {
        let (hr, result) = match iid {
            Some(iid) if !this.is_null() && interface.supports(iid) => {
                add_ref(this);
                (S_OK, this.cast())
            }
            _ if this.is_null() || riid.is_null() => (E_POINTER, ptr::null_mut()),
            _ => (E_NOINTERFACE, ptr::null_mut()),
        };
        // SAFETY: The caller supplies a writable COM output pointer.
        unsafe { out.write(result) };
        hr
    };

    let trace = trace_d3d12_enabled();
    if trace || hr == E_NOINTERFACE {
        let iid = iid.map(format_guid);
        let iid = iid.as_deref().unwrap_or("<null>");
        if trace {
            log_d3d12_trace(format_args!(
                "{}::QueryInterface({iid}) -> hr=0x{:08x}",
                interface.name, hr as u32
            ));
        }
        if hr == E_NOINTERFACE {
            notify_rejected(interface.name, iid);
        }
    }
    hr
}

/// # Safety
/// When tracing is enabled, a non-null IID must point to 16 readable bytes.
pub(crate) unsafe fn trace_create_iid(method: &str, riid: *const c_void, hr: i32) {
    if trace_d3d12_enabled() {
        // SAFETY: The caller supplies a readable GUID when riid is non-null.
        let iid = unsafe { riid.cast::<[u8; 16]>().as_ref() }.map(format_guid);
        let iid = iid.as_deref().unwrap_or("<null>");
        log_d3d12_trace(format_args!(
            "ID3D12DeviceX::{method}(riid={iid}) -> hr=0x{:08x}",
            hr as u32
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    thread_local! {
        static REJECTION: RefCell<Option<(&'static str, String)>> = const { RefCell::new(None) };
    }

    fn capture_rejection(interface: &'static str, iid: &str) {
        REJECTION.replace(Some((interface, iid.to_owned())));
    }

    #[test]
    fn null_device_query_interface_checks_iid_and_notifies_rejections() {
        extern "win64" fn probe(_slot: u32, _caller: u64) -> u64 {
            0
        }
        let vtable = crate::build_null_device_vtable(probe);
        let device = crate::create_null_device(vtable.expect("null device vtable"));
        set_d3d12_rejected_qi_callback(Some(capture_rejection));
        let unknown = guid(0x12345678, 0x9abc, 0xdef0, 0x0123_4567_89ab_cdef);
        let mut out = ptr::dangling_mut();
        let rejected = crate::null_query_interface(device, unknown.as_ptr().cast(), &mut out);
        set_d3d12_rejected_qi_callback(None);

        assert_eq!(rejected, E_NOINTERFACE);
        assert!(out.is_null());
        assert_eq!(
            REJECTION.take(),
            Some((
                "ID3D12DeviceX",
                "{12345678-9abc-def0-0123-456789abcdef}".to_owned()
            ))
        );
        assert_eq!(
            crate::null_query_interface(device, unknown.as_ptr().cast(), ptr::null_mut()),
            E_POINTER
        );
        assert_eq!(
            crate::null_query_interface(device, ptr::null(), &mut out),
            E_POINTER
        );
        assert!(out.is_null());

        let device14 = DEVICE.iids.last().unwrap();
        assert_eq!(
            crate::null_query_interface(device, device14.as_ptr().cast(), &mut out),
            S_OK
        );
        assert_eq!(out, device.cast());
        assert_eq!(crate::null_release(device), 1);
        assert_eq!(crate::null_release(device), 0);
    }
}
