// Console audio co-processor shim interfaces (`AcpHal.dll`, `xapu.dll`, `chad.dll`) informed by
// Marvisak/VIctory (https://github.com/Marvisak/VIctory, commit
// 63bcca43c6a00a758510398fa43096ce7e4f7df0, crates/victory-hle/src/modules/{audio,apu,dsp}.rs)
// and xcl-dev/xcl (commit 6089122a1c96dff16e93b6fdf04c8ed20cfb6102,
// crates/xcl/src/hle/acphal.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27: heap/command-ring/stream state tracking for native Linux audio.
// Modified 2026-11-28: bounded completions, PCM submission, and stable DSP stream status.

//! Audio co-processor HLE state (`AcpHal`, `xapu`, `chad`) tracking APU heaps, virtual-to-APU
//! address maps, XAPU command rings/results, and DSP stream activations.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use crate::AudioRingBuffer;

struct CompletionRing<T> {
    entries: VecDeque<T>,
    limit: usize,
    written: u32,
    read: u32,
}

impl<T> CompletionRing<T> {
    fn new(limit: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            limit,
            written: 0,
            read: 0,
        }
    }

    fn push(&mut self, entry: T) -> Option<u32> {
        if self.entries.len() == self.limit {
            return None;
        }
        self.entries.push_back(entry);
        self.written = self.written.wrapping_add(1);
        Some(self.written)
    }

    fn pop(&mut self) -> Option<T> {
        let entry = self.entries.pop_front()?;
        self.read = self.read.wrapping_add(1);
        Some(entry)
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct AcpMessage {
    kind: u32,
    dropped: u32,
    timestamp_us: u32,
    data: [u8; 16],
}

pub struct AcpCommandRing {
    connected: bool,
    mask: u32,
    dropped: u32,
    started: std::time::Instant,
    results: CompletionRing<AcpMessage>,
}

impl Default for AcpCommandRing {
    fn default() -> Self {
        Self {
            connected: false,
            mask: 0,
            dropped: 0,
            started: std::time::Instant::now(),
            results: CompletionRing::new(0),
        }
    }
}

impl AcpCommandRing {
    pub fn is_connected(&self) -> bool {
        self.connected
    }

    pub fn connect(&mut self, command_slots: u32, result_slots: u32) -> bool {
        if self.connected || command_slots == 0 || result_slots > 65_536 {
            return false;
        }
        *self = Self {
            connected: true,
            results: CompletionRing::new(result_slots as usize),
            ..Self::default()
        };
        true
    }

    pub fn disconnect(&mut self) -> bool {
        if !self.connected {
            return false;
        }
        self.post(0x80, [0; 16]);
        self.connected = false;
        true
    }

    pub fn submit(&mut self, kind: u32, id: u64, word: u32) -> bool {
        if !self.connected {
            return false;
        }
        match kind {
            1 => {
                let mut data = [0; 16];
                data[..4].copy_from_slice(&word.to_le_bytes());
                self.post(2, data);
            }
            2 => self.mask |= word,
            3 => self.mask &= !word,
            _ => {}
        }
        if id != 0 {
            let frame = (self.started.elapsed().as_micros() * 375 / 1_000_000) as u32;
            let mut data = [0; 16];
            data[..4].copy_from_slice(&kind.to_le_bytes());
            data[4..12].copy_from_slice(&id.to_le_bytes());
            data[12..].copy_from_slice(&frame.to_le_bytes());
            self.post(0x10, data);
        }
        true
    }

    fn post(&mut self, kind: u32, data: [u8; 16]) {
        if self.mask & kind == 0 {
            return;
        }
        let entry = AcpMessage {
            kind,
            dropped: self.dropped,
            timestamp_us: self.started.elapsed().as_micros() as u32,
            data,
        };
        if self.results.push(entry).is_some() {
            self.dropped = 0;
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    pub fn pop(&mut self) -> Option<AcpMessage> {
        self.results.pop()
    }

    pub fn count(&self) -> u32 {
        self.results.written.wrapping_sub(self.results.read)
    }
}

struct ApuAllocation {
    va: usize,
    apu_addr: u32,
    layout: Layout,
}

enum XApuConnection {
    Pcm(CompletionRing<u32>),
    Decoder(XApuDecoderConnection),
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct XApuCommandId {
    pub kind: u32,
    pub stream: u32,
    pub sequence: u32,
    pub options: u32,
    pub context: u64,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
pub struct XApuResult {
    pub id: XApuCommandId,
    pub state: u32,
    pub state_data: u64,
    pub usec: u64,
    pub error: i32,
    pub frames: u32,
    pub output_length: u32,
    pub output: usize,
    pub pitch: f32,
    pub input_length: u32,
    pub input: usize,
}

const _: () = assert!(std::mem::size_of::<XApuResult>() == 80);

pub const XAPU_QUEUE_FULL: i32 = 0x8ac8_1000u32 as i32;
pub const XAPU_PENDING_RESULTS: i32 = 0x8ac8_1001u32 as i32;
pub const XAPU_INVALID_COMMAND: i32 = 0x8ac8_1002u32 as i32;
const XAPU_INVALID_STREAM: i32 = 0x8ac8_1003u32 as i32;
const XAPU_ALREADY_ACTIVE: i32 = 0x8ac8_1005u32 as i32;
const XAPU_BAD_PACKET: i32 = 0x8ac8_1008u32 as i32;
const E_INVALIDARG: i32 = 0x8007_0057u32 as i32;
const E_NOTIMPL: i32 = 0x8000_4001u32 as i32;

pub enum XApuCommand<'a> {
    Activate(u32),
    Deactivate,
    Process {
        input_address: usize,
        input: &'a [u8],
        output: &'a mut [u8],
        first_frame: u32,
        frames: u32,
    },
    Notify,
    Reset,
}

struct XApuDecoderConnection {
    memory: usize,
    event: usize,
    commands_per_stream: u32,
    deinterleave: bool,
    streams: Vec<XApuDecoderStream>,
    results: CompletionRing<XApuResult>,
}

#[derive(Default)]
struct XApuDecoderStream {
    decoder: Option<OpusDecoder>,
    pending: u32,
}

type OpusCreate = unsafe extern "C" fn(i32, i32, *mut i32) -> *mut c_void;
type OpusDestroy = unsafe extern "C" fn(*mut c_void);
type OpusDecode = unsafe extern "C" fn(*mut c_void, *const u8, i32, *mut f32, i32, i32) -> i32;

struct OpusApi {
    create: OpusCreate,
    destroy: OpusDestroy,
    decode: OpusDecode,
}

// SAFETY: These are glibc's dynamic loader declarations for Linux x86-64.
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> i32;
}

impl OpusApi {
    fn load() -> Option<&'static Self> {
        static API: OnceLock<Option<OpusApi>> = OnceLock::new();
        API.get_or_init(|| {
            // SAFETY: Strings are NUL terminated. A successful handle stays open for decoder instances.
            unsafe {
                let handle = dlopen(c"libopus.so.0".as_ptr(), 2);
                if handle.is_null() {
                    return None;
                }
                let create = dlsym(handle, c"opus_decoder_create".as_ptr());
                let destroy = dlsym(handle, c"opus_decoder_destroy".as_ptr());
                let decode = dlsym(handle, c"opus_decode_float".as_ptr());
                if create.is_null() || destroy.is_null() || decode.is_null() {
                    dlclose(handle);
                    return None;
                }
                Some(OpusApi {
                    create: std::mem::transmute::<*mut c_void, OpusCreate>(create),
                    destroy: std::mem::transmute::<*mut c_void, OpusDestroy>(destroy),
                    decode: std::mem::transmute::<*mut c_void, OpusDecode>(decode),
                })
            }
        })
        .as_ref()
    }
}

struct OpusDecoder {
    api: &'static OpusApi,
    state: usize,
    channels: u32,
}

impl OpusDecoder {
    fn new(channels: u32) -> Result<Self, i32> {
        if !matches!(channels, 1 | 2) {
            return Err(E_INVALIDARG);
        }
        let api = OpusApi::load().ok_or(E_NOTIMPL)?;
        // SAFETY: Libopus supports these sample rate/channel values and reports failure as null.
        let state = unsafe { (api.create)(48_000, channels as i32, std::ptr::null_mut()) } as usize;
        if state == 0 {
            return Err(E_INVALIDARG);
        }
        Ok(Self {
            api,
            state,
            channels,
        })
    }

    fn decode(&mut self, input: &[u8], output: &mut [f32], max_frames: i32) -> Result<usize, i32> {
        let input_len = i32::try_from(input.len()).map_err(|_| E_INVALIDARG)?;
        // SAFETY: The decoder is exclusively borrowed and output holds max_frames * channels samples.
        let frames = unsafe {
            (self.api.decode)(
                self.state as *mut c_void,
                input.as_ptr(),
                input_len,
                output.as_mut_ptr(),
                max_frames,
                0,
            )
        };
        usize::try_from(frames).map_err(|_| XAPU_BAD_PACKET)
    }
}

impl Drop for OpusDecoder {
    fn drop(&mut self) {
        // SAFETY: The state is uniquely owned and was allocated by this library's create function.
        unsafe { (self.api.destroy)(self.state as *mut c_void) };
    }
}

struct ChadDspConnection {
    max_streams: u32,
    active_streams: u32,
}

struct ChadStream {
    dsp_id: u64,
    status: Arc<ChadStatus>,
    submission: Mutex<()>,
}

/// Shared Win64 status block. Completion is published after the result and output bytes.
#[repr(C)]
#[derive(Default)]
pub struct ChadStatus {
    submitted: AtomicU32,
    completed: AtomicU32,
    reserved: AtomicU32,
    result: AtomicI32,
}

impl ChadStatus {
    pub fn completion(&self) -> (u32, i32) {
        (
            self.completed.load(Ordering::Acquire),
            self.result.load(Ordering::Relaxed),
        )
    }
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
    chad_streams: BTreeMap<u64, Arc<ChadStream>>,
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
        let mut guard = self.inner.lock().ok()?;
        let aligned_apu = guard.next_apu_addr.checked_add(align as u32 - 1)? & !(align as u32 - 1);
        let next = aligned_apu.checked_add(alloc_size as u32)?;
        // SAFETY: The validated layout has nonzero size and power-of-two alignment.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        let va = ptr as usize;
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

    pub fn apu_notify(&self, address: u32) -> bool {
        let Ok(guard) = self.inner.lock() else {
            return false;
        };
        let Some((&base, &va)) = guard.va_by_apu.range(..=address).next_back() else {
            return false;
        };
        let offset = (address - base) as usize;
        if !address.is_multiple_of(4) || offset + 4 > guard.allocs_by_va[&va].layout.size() {
            return false;
        }
        // SAFETY: The aligned word is within a live allocation held by the lock.
        unsafe { AtomicU32::from_ptr((va + offset) as *mut u32) }.store(1, Ordering::Release);
        true
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
        guard
            .xapu_conns
            .insert(id, XApuConnection::Pcm(CompletionRing::new(256)));
        Some(id)
    }

    pub fn xapu_disconnect(&self, conn_id: u64) -> bool {
        self.xapu_disconnect_with_event(conn_id).is_some()
    }

    pub fn xapu_disconnect_with_event(&self, conn_id: u64) -> Option<usize> {
        let conn = self.inner.lock().ok()?.xapu_conns.remove(&conn_id)?;
        match conn {
            XApuConnection::Pcm(_) => Some(0),
            XApuConnection::Decoder(conn) => {
                self.apu_free(conn.memory);
                Some(conn.event)
            }
        }
    }

    pub fn xapu_enqueue_command(&self, conn_id: u64) -> Option<u32> {
        self.xapu_submit(conn_id, None)
    }

    pub fn xapu_submit_pcm(
        &self,
        conn_id: u64,
        output: &AudioRingBuffer,
        samples: &[f32],
    ) -> Option<u32> {
        self.xapu_submit(conn_id, Some((output, samples)))
    }

    fn xapu_submit(&self, conn_id: u64, pcm: Option<(&AudioRingBuffer, &[f32])>) -> Option<u32> {
        let mut guard = self.inner.lock().ok()?;
        let XApuConnection::Pcm(results) = guard.xapu_conns.get_mut(&conn_id)? else {
            return None;
        };
        if results.entries.len() == results.limit {
            return None;
        }
        if let Some((output, samples)) = pcm {
            push_stereo_pcm(output, samples)?;
        }
        results.push(results.written.wrapping_add(1))
    }

    pub fn acp_submit_pcm(&self, output: &AudioRingBuffer, samples: &[f32]) -> Option<()> {
        // The output ring has one producer. Serialize the AcpHal and XApu writers.
        let _guard = self.inner.lock().ok()?;
        push_stereo_pcm(output, samples)
    }

    pub fn xapu_dequeue_result(&self, conn_id: u64) -> Option<u32> {
        let Ok(mut guard) = self.inner.lock() else {
            return None;
        };
        let XApuConnection::Pcm(results) = guard.xapu_conns.get_mut(&conn_id)? else {
            return None;
        };
        results.pop()
    }

    pub fn xapu_result_count(&self, conn_id: u64) -> u32 {
        self.xapu_result_counts(conn_id).0
    }

    pub fn xapu_result_counts(&self, conn_id: u64) -> (u32, u32) {
        let Ok(guard) = self.inner.lock() else {
            return (0, 0);
        };
        match guard.xapu_conns.get(&conn_id) {
            Some(XApuConnection::Pcm(results)) => {
                let pending = results.written.wrapping_sub(results.read);
                (pending, pending)
            }
            Some(XApuConnection::Decoder(conn)) => (
                conn.results.written.wrapping_sub(conn.results.read),
                conn.results.read,
            ),
            None => (0, 0),
        }
    }

    pub fn xapu_connect_decoder(
        &self,
        streams: u32,
        commands_per_stream: u32,
        memory: u32,
        event: usize,
        deinterleave: bool,
    ) -> Option<(u64, usize)> {
        if !(1..=500).contains(&streams)
            || !(1..=16).contains(&commands_per_stream)
            || !(1..=64 * 1024 * 1024).contains(&memory)
        {
            return None;
        }
        let (memory, _) = self.apu_alloc(memory, 65_536)?;
        let mut guard = self.inner.lock().ok()?;
        let id = guard.next_handle;
        guard.next_handle += 1;
        guard.xapu_conns.insert(
            id,
            XApuConnection::Decoder(XApuDecoderConnection {
                memory,
                event,
                commands_per_stream,
                deinterleave,
                streams: (0..streams).map(|_| XApuDecoderStream::default()).collect(),
                results: CompletionRing::new((streams * commands_per_stream) as usize),
            }),
        );
        Some((id, memory))
    }

    pub fn xapu_is_decoder(&self, conn_id: u64) -> Option<bool> {
        let guard = self.inner.lock().ok()?;
        Some(matches!(
            guard.xapu_conns.get(&conn_id)?,
            XApuConnection::Decoder(_)
        ))
    }

    pub fn xapu_submit_decoder(
        &self,
        conn_id: u64,
        id: XApuCommandId,
        command: XApuCommand<'_>,
    ) -> Result<u32, i32> {
        let mut guard = self.inner.lock().map_err(|_| E_INVALIDARG)?;
        let Some(XApuConnection::Decoder(conn)) = guard.xapu_conns.get_mut(&conn_id) else {
            return Err(E_INVALIDARG);
        };
        let stream = conn
            .streams
            .get_mut(id.stream as usize)
            .ok_or(XAPU_INVALID_STREAM)?;
        if stream.pending == conn.commands_per_stream {
            return Err(XAPU_QUEUE_FULL);
        }
        let mut result = XApuResult {
            id,
            state: 0,
            state_data: 0,
            usec: 0,
            error: 0,
            frames: 0,
            output_length: 0,
            output: 0,
            pitch: 1.0,
            input_length: 0,
            input: 0,
        };
        match command {
            XApuCommand::Activate(channels) => {
                if stream.decoder.is_some() {
                    result.error = XAPU_ALREADY_ACTIVE;
                } else {
                    match OpusDecoder::new(channels) {
                        Ok(decoder) => stream.decoder = Some(decoder),
                        Err(error) => result.error = error,
                    }
                }
            }
            XApuCommand::Deactivate => stream.decoder = None,
            XApuCommand::Notify => {}
            XApuCommand::Reset => {
                if let Some(decoder) = &mut stream.decoder {
                    match OpusDecoder::new(decoder.channels) {
                        Ok(reset) => *decoder = reset,
                        Err(error) => result.error = error,
                    }
                }
            }
            XApuCommand::Process {
                input_address,
                input,
                output,
                first_frame,
                frames,
            } => {
                result.input = input_address;
                result.input_length = input.len() as u32;
                result.output = output.as_mut_ptr() as usize;
                if let Some(decoder) = &mut stream.decoder {
                    match process_opus(
                        decoder,
                        input,
                        output,
                        first_frame,
                        frames,
                        conn.deinterleave,
                    ) {
                        Ok((frames, length)) => {
                            result.frames = frames;
                            result.output_length = length;
                        }
                        Err(error) => result.error = error,
                    }
                } else {
                    result.error = XAPU_INVALID_STREAM;
                }
            }
        }
        let written = conn.results.push(result).ok_or(XAPU_QUEUE_FULL)?;
        stream.pending += 1;
        Ok(written)
    }

    pub fn xapu_signal_decoder(&self, conn_id: u64, signal: impl FnOnce(usize)) {
        if let Ok(guard) = self.inner.lock()
            && let Some(XApuConnection::Decoder(conn)) = guard.xapu_conns.get(&conn_id)
            && conn.event != 0
        {
            // Serialize signalling against returning the handle to the shim for closure.
            signal(conn.event);
        }
    }

    pub fn xapu_dequeue_decoder(&self, conn_id: u64) -> Option<XApuResult> {
        let mut guard = self.inner.lock().ok()?;
        let XApuConnection::Decoder(conn) = guard.xapu_conns.get_mut(&conn_id)? else {
            return None;
        };
        let result = conn.results.pop()?;
        conn.streams[result.id.stream as usize].pending -= 1;
        Some(result)
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
        self.chad_activate_with_status(dsp_id).map(|(id, _)| id)
    }

    pub fn chad_activate_with_status(&self, dsp_id: u64) -> Option<(u64, Arc<ChadStatus>)> {
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
        let status = Arc::new(ChadStatus::default());
        guard.chad_streams.insert(
            stream_id,
            Arc::new(ChadStream {
                dsp_id,
                status: Arc::clone(&status),
                submission: Mutex::new(()),
            }),
        );
        Some((stream_id, status))
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
        self.chad_execute(stream_id, || 0).map(|(ticket, _)| ticket)
    }

    pub fn chad_execute(&self, stream_id: u64, run: impl FnOnce() -> i32) -> Option<(u32, i32)> {
        let stream = Arc::clone(self.inner.lock().ok()?.chad_streams.get(&stream_id)?);
        let _submission = stream.submission.lock().ok()?;
        let status = &stream.status;
        let ticket = status
            .submitted
            .load(Ordering::Relaxed)
            .wrapping_add(1)
            .max(1);
        status.submitted.store(ticket, Ordering::Release);
        let result = run();
        status.result.store(result, Ordering::Relaxed);
        status.completed.store(ticket, Ordering::Release);
        Some((ticket, result))
    }
}

fn push_stereo_pcm(output: &AudioRingBuffer, samples: &[f32]) -> Option<()> {
    if samples.is_empty()
        || !samples.len().is_multiple_of(2)
        || samples.len() > output.available_write()
    {
        return None;
    }
    (output.push_f32(samples) == samples.len()).then_some(())
}

fn process_opus(
    decoder: &mut OpusDecoder,
    input: &[u8],
    output: &mut [u8],
    first_frame: u32,
    frames: u32,
    deinterleave: bool,
) -> Result<(u32, u32), i32> {
    const MAX_FRAMES: i32 = 5760;
    let channels = decoder.channels as usize;
    let mut pcm = vec![0.0f32; MAX_FRAMES as usize * channels];
    let decoded = decoder.decode(input, &mut pcm, MAX_FRAMES)?;
    let first = (first_frame as usize).min(decoded);
    let kept = (decoded - first)
        .min(frames as usize)
        .min(output.len() / (channels * 4));
    let pcm = &pcm[first * channels..(first + kept) * channels];
    let bytes = kept * channels * 4;
    let (out, _) = output[..bytes].as_chunks_mut::<4>();
    if deinterleave && channels == 2 {
        let (left, right) = out.split_at_mut(kept);
        for ((left, right), &[l, r]) in left.iter_mut().zip(right).zip(pcm.as_chunks::<2>().0) {
            *left = l.to_le_bytes();
            *right = r.to_le_bytes();
        }
    } else {
        for (dst, sample) in out.iter_mut().zip(pcm) {
            *dst = sample.to_le_bytes();
        }
    }
    Ok((kept as u32, bytes as u32))
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

    #[test]
    fn xapu_pcm_backpressure_preserves_ring_and_completion_order() {
        let state = AudioCoprocState::new();
        let conn = state.xapu_connect().unwrap();
        let output = AudioRingBuffer::new(4, 48_000);
        assert_eq!(state.xapu_submit_pcm(conn + 1, &output, &[0.1, 0.2]), None);
        assert_eq!(output.available_read(), 0);
        let samples = [0.1, 0.2, 0.3, 0.4].repeat(output.capacity() / 4);
        assert_eq!(state.xapu_submit_pcm(conn, &output, &samples), Some(1));
        assert_eq!(state.xapu_submit_pcm(conn, &output, &[0.5, 0.6]), None);
        assert_eq!(state.xapu_result_count(conn), 1);
        let mut frames = vec![0.0; samples.len()];
        assert_eq!(output.pop_f32(&mut frames), samples.len());
        assert_eq!(frames, samples);
        assert_eq!(state.xapu_submit_pcm(conn, &output, &[0.5, 0.6]), Some(2));
        assert_eq!(state.xapu_dequeue_result(conn), Some(1));
        assert_eq!(state.xapu_dequeue_result(conn), Some(2));
        assert_eq!(state.xapu_result_count(conn), 0);
    }

    #[test]
    fn acp_message_ring_advances_and_reports_overflow() {
        let mut ring = AcpCommandRing::default();
        assert!(!ring.submit(1, 0, 0));
        assert!(ring.connect(2, 1));
        assert!(ring.submit(2, 0, 0x90));
        assert!(ring.submit(1, 99, 7));
        assert!(ring.submit(1, 100, 8));
        assert_eq!(ring.count(), 1);
        let msg = ring.pop().unwrap();
        let (kind, dropped, data) = (msg.kind, msg.dropped, msg.data);
        assert_eq!((kind, dropped), (0x10, 0));
        assert_eq!(u64::from_le_bytes(data[4..12].try_into().unwrap()), 99);
        assert!(ring.submit(1, 101, 9));
        let dropped = ring.pop().unwrap().dropped;
        assert_eq!(dropped, 1);
        assert_eq!(ring.count(), 0);
        assert!(ring.disconnect());
        let kind = ring.pop().unwrap().kind;
        assert_eq!(kind, 0x80);
        assert!(!ring.is_connected());
    }

    #[test]
    fn xapu_opus_process_reports_decoded_frames() {
        let state = AudioCoprocState::new();
        let (conn, _) = state.xapu_connect_decoder(1, 2, 4096, 0, true).unwrap();
        let mut id = XApuCommandId {
            kind: 1,
            stream: 0,
            sequence: 0,
            options: 0,
            context: 0,
        };
        assert_eq!(
            state.xapu_submit_decoder(conn, id, XApuCommand::Activate(2)),
            Ok(1)
        );
        let activation = state.xapu_dequeue_decoder(conn).unwrap().error;
        if OpusApi::load().is_none() {
            assert_eq!(activation, E_NOTIMPL);
            return;
        }
        assert_eq!(activation, 0);
        id.kind = 3;
        let silence = [0xf8, 0xff, 0xfe];
        let mut output = [0xa5; 8 * 16];
        let process = XApuCommand::Process {
            input_address: silence.as_ptr() as usize,
            input: &silence,
            output: &mut output,
            first_frame: 2,
            frames: 32,
        };
        assert_eq!(state.xapu_submit_decoder(conn, id, process), Ok(2));
        let completion = state.xapu_dequeue_decoder(conn).unwrap();
        let (error, frames, bytes) = (
            completion.error,
            completion.frames,
            completion.output_length,
        );
        assert_eq!((error, frames, bytes), (0, 16, output.len() as u32));
        let (left, right) = output.split_at(output.len() / 2);
        assert_eq!(left, right);
        assert!(
            output
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| f32::from_le_bytes(*v).abs() < 1e-6)
        );
    }

    #[test]
    fn chad_status_advances_for_success_and_failure() {
        let state = AudioCoprocState::new();
        let dsp = state.chad_connect(1).unwrap();
        let (stream, status) = state.chad_activate_with_status(dsp).unwrap();
        assert!(state.chad_activate_with_status(dsp).is_none());
        assert_eq!(status.completion(), (0, 0));
        assert_eq!(
            state.chad_execute(stream, || {
                assert_eq!(status.submitted.load(Ordering::Acquire), 1);
                assert_eq!(status.completion(), (0, 0));
                -1
            }),
            Some((1, -1))
        );
        assert_eq!(status.completion(), (1, -1));
        assert_eq!(state.chad_submit_command(stream), Some(2));
        assert_eq!(status.completion(), (2, 0));
        assert!(state.chad_deactivate(stream));
        assert_eq!(state.chad_execute(stream, || unreachable!()), None);
    }
}
