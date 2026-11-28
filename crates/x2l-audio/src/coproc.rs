// Console audio co-processor shim interfaces (`AcpHal.dll`, `xapu.dll`, `chad.dll`) informed by
// Marvisak/VIctory (https://github.com/Marvisak/VIctory, commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/{audio,apu,dsp}.rs)
// and xcl-dev/xcl (commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/hle/acphal.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: heap/command-ring/stream state tracking for native Linux audio.

//! Audio co-processor HLE state (`AcpHal`, `xapu`, `chad`) tracking APU heaps, virtual-to-APU
//! address maps, XAPU command rings/results, and DSP stream activations.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

struct ApuAllocation {
    va: usize,
    apu_addr: u32,
    layout: Layout,
}

struct XApuConnection {
    next_ticket: u32,
    pending_results: VecDeque<u32>,
}

struct ChadDspConnection {
    max_streams: u32,
    active_streams: u32,
}

struct ChadStream {
    dsp_id: u64,
    next_ticket: u32,
}

struct CoprocInner {
    heap_cached_bytes: u32,
    heap_noncached_bytes: u32,
    next_apu_addr: u32,
    allocs_by_va: BTreeMap<usize, ApuAllocation>,
    va_by_apu: BTreeMap<u32, usize>,
    shape_contexts_active: bool,
    next_handle: u64,
    xapu_conns: BTreeMap<u64, XApuConnection>,
    chad_conns: BTreeMap<u64, ChadDspConnection>,
    chad_streams: BTreeMap<u64, ChadStream>,
}

impl Default for CoprocInner {
    fn default() -> Self {
        Self {
            heap_cached_bytes: 0,
            heap_noncached_bytes: 0,
            next_apu_addr: 0x0010_0000,
            allocs_by_va: BTreeMap::new(),
            va_by_apu: BTreeMap::new(),
            shape_contexts_active: false,
            next_handle: 0x100,
            xapu_conns: BTreeMap::new(),
            chad_conns: BTreeMap::new(),
            chad_streams: BTreeMap::new(),
        }
    }
}

impl Drop for CoprocInner {
    fn drop(&mut self) {
        let allocs = std::mem::take(&mut self.allocs_by_va);
        for (_, rec) in allocs {
            // SAFETY: Each APU allocation came from `alloc_zeroed(rec.layout)` and is freed once.
            unsafe {
                dealloc(rec.va as *mut u8, rec.layout);
            }
        }
    }
}

/// Thread-safe audio co-processor state shared by `AcpHal.dll`, `xapu.dll`, and `chad.dll`.
#[derive(Default)]
pub struct AudioCoprocState {
    inner: Mutex<CoprocInner>,
}

impl AudioCoprocState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apu_create_heap(&self, cached_bytes: u32, noncached_bytes: u32) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.heap_cached_bytes = cached_bytes;
        guard.heap_noncached_bytes = noncached_bytes;
        true
    }

    pub fn apu_alloc(&self, size: u32, alignment: u32) -> Option<(usize, u32)> {
        let alloc_size = (size as usize).max(16);
        let align = (alignment as usize).max(16).next_power_of_two().min(65536);
        let layout = Layout::from_size_align(alloc_size, align).ok()?;
        // SAFETY: `layout` has non-zero size and power-of-two alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        let va = ptr as usize;
        let Ok(mut guard) = self.inner.lock() else {
            // SAFETY: Roll back allocation on poisoned lock.
            unsafe { dealloc(ptr, layout) };
            return None;
        };
        let aligned_apu =
            (guard.next_apu_addr.saturating_add(align as u32 - 1)) & !(align as u32 - 1);
        let next = aligned_apu.checked_add(alloc_size as u32)?;
        guard.next_apu_addr = next;
        guard.allocs_by_va.insert(
            va,
            ApuAllocation {
                va,
                apu_addr: aligned_apu,
                layout,
            },
        );
        guard.va_by_apu.insert(aligned_apu, va);
        Some((va, aligned_apu))
    }

    pub fn apu_free(&self, va: usize) -> bool {
        if va == 0 {
            return true;
        }
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let Some(rec) = guard.allocs_by_va.remove(&va) else {
            return false;
        };
        guard.va_by_apu.remove(&rec.apu_addr);
        // SAFETY: `rec.va` came from `alloc_zeroed(rec.layout)` and was removed from the table.
        unsafe { dealloc(rec.va as *mut u8, rec.layout) };
        true
    }

    pub fn apu_map_apu_address(&self, apu_addr: u32) -> usize {
        let Ok(guard) = self.inner.lock() else {
            return 0;
        };
        if let Some((&base_apu, &base_va)) = guard.va_by_apu.range(..=apu_addr).next_back()
            && let Some(rec) = guard.allocs_by_va.get(&base_va)
        {
            let off = (apu_addr - base_apu) as usize;
            if off < rec.layout.size() {
                return base_va + off;
            }
        }
        0
    }

    pub fn apu_map_virtual_address(&self, va: usize) -> Option<u32> {
        let Ok(guard) = self.inner.lock() else {
            return None;
        };
        let (&base_va, rec) = guard.allocs_by_va.range(..=va).next_back()?;
        let off = va - base_va;
        if off < rec.layout.size() {
            Some(rec.apu_addr.saturating_add(off as u32))
        } else {
            None
        }
    }

    pub fn apu_is_virtual_address_valid(&self, va: usize, size: u32) -> bool {
        let Ok(guard) = self.inner.lock() else {
            return false;
        };
        let Some((&base_va, rec)) = guard.allocs_by_va.range(..=va).next_back() else {
            return false;
        };
        let off = va - base_va;
        off.saturating_add(size as usize) <= rec.layout.size()
    }

    pub fn allocate_shape_contexts(&self) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.shape_contexts_active = true;
        true
    }

    pub fn release_shape_contexts(&self) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.shape_contexts_active = false;
        true
    }

    // ---- xapu.dll ----

    pub fn xapu_connect(&self) -> Option<u64> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let id = guard.next_handle;
        guard.next_handle += 1;
        guard.xapu_conns.insert(
            id,
            XApuConnection {
                next_ticket: 1,
                pending_results: VecDeque::new(),
            },
        );
        Some(id)
    }

    pub fn xapu_disconnect(&self, conn_id: u64) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.xapu_conns.remove(&conn_id).is_some()
    }

    pub fn xapu_enqueue_command(&self, conn_id: u64) -> Option<u32> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let conn = guard.xapu_conns.get_mut(&conn_id)?;
        let ticket = conn.next_ticket;
        conn.next_ticket = conn.next_ticket.wrapping_add(1).max(1);
        conn.pending_results.push_back(ticket);
        Some(ticket)
    }

    pub fn xapu_dequeue_result(&self, conn_id: u64) -> Option<u32> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let conn = guard.xapu_conns.get_mut(&conn_id)?;
        conn.pending_results.pop_front()
    }

    pub fn xapu_result_count(&self, conn_id: u64) -> u32 {
        let Ok(guard) = self.inner.lock() else {
            return 0;
        };
        guard
            .xapu_conns
            .get(&conn_id)
            .map(|c| c.pending_results.len() as u32)
            .unwrap_or(0)
    }

    // ---- chad.dll ----

    pub fn chad_connect(&self, max_streams: u32) -> Option<u64> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let id = guard.next_handle;
        guard.next_handle += 1;
        guard.chad_conns.insert(
            id,
            ChadDspConnection {
                max_streams: max_streams.max(1),
                active_streams: 0,
            },
        );
        Some(id)
    }

    pub fn chad_disconnect(&self, dsp_id: u64) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.chad_streams.retain(|_, s| s.dsp_id != dsp_id);
        guard.chad_conns.remove(&dsp_id).is_some()
    }

    pub fn chad_activate(&self, dsp_id: u64) -> Option<u64> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let conn = guard.chad_conns.get_mut(&dsp_id)?;
        if conn.active_streams >= conn.max_streams {
            return None;
        }
        conn.active_streams += 1;
        let stream_id = guard.next_handle;
        guard.next_handle += 1;
        guard.chad_streams.insert(
            stream_id,
            ChadStream {
                dsp_id,
                next_ticket: 1,
            },
        );
        Some(stream_id)
    }

    pub fn chad_deactivate(&self, stream_id: u64) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        let Some(stream) = guard.chad_streams.remove(&stream_id) else {
            return false;
        };
        if let Some(conn) = guard.chad_conns.get_mut(&stream.dsp_id) {
            conn.active_streams = conn.active_streams.saturating_sub(1);
        }
        true
    }

    pub fn chad_submit_command(&self, stream_id: u64) -> Option<u32> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let stream = guard.chad_streams.get_mut(&stream_id)?;
        let ticket = stream.next_ticket;
        stream.next_ticket = stream.next_ticket.wrapping_add(1).max(1);
        Some(ticket)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apu_xapu_and_chad_lifecycle_roundtrip() {
        let state = AudioCoprocState::new();
        assert!(state.apu_create_heap(4 * 1024 * 1024, 2 * 1024 * 1024));
        assert!(state.allocate_shape_contexts());

        let (va, apu) = state.apu_alloc(1024, 64).expect("apu alloc");
        assert_eq!(va % 64, 0);
        assert_eq!(apu % 64, 0);
        assert_eq!(state.apu_map_apu_address(apu + 16), va + 16);
        assert_eq!(state.apu_map_virtual_address(va + 16), Some(apu + 16));
        assert!(state.apu_is_virtual_address_valid(va, 1024));
        assert!(!state.apu_is_virtual_address_valid(va, 2048));
        assert!(state.apu_free(va));
        assert!(state.release_shape_contexts());

        let conn = state.xapu_connect().expect("xapu connect");
        let ticket = state.xapu_enqueue_command(conn).expect("enqueue");
        assert_eq!(state.xapu_result_count(conn), 1);
        assert_eq!(state.xapu_dequeue_result(conn), Some(ticket));
        assert_eq!(state.xapu_result_count(conn), 0);
        assert!(state.xapu_disconnect(conn));

        let dsp = state.chad_connect(4).expect("chad connect");
        let stream = state.chad_activate(dsp).expect("chad activate");
        assert_eq!(state.chad_submit_command(stream), Some(1));
        assert!(state.chad_deactivate(stream));
        assert!(state.chad_disconnect(dsp));
    }
}
