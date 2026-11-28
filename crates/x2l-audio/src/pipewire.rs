//! Native Linux PipeWire PCM playback sink (`libpipewire-0.3.so.0` via runtime `dlopen`) with a
//! wait-free SPSC ring buffer and deterministic [`NullAudioSink`] fallback for headless CI.

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Once};
use x2l_formats::opus::PcmBuffer;

/// Default playback sample rate (48 kHz, matching standard console audio output).
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 48_000;
/// Default stereo channel count.
pub const DEFAULT_CHANNELS: u8 = 2;
/// Default ring buffer capacity in `f32` samples (16384 samples = ~170.7 ms at 48 kHz stereo).
pub const DEFAULT_RING_CAPACITY_SAMPLES: usize = 16_384;

const RTLD_LAZY: c_int = 0x0001;
const RTLD_LOCAL: c_int = 0x0000;

const PW_VERSION_STREAM_EVENTS: u32 = 2;
const PW_DIRECTION_OUTPUT: u32 = 1;
const PW_ID_ANY: u32 = 0xffff_ffff;
const PW_STREAM_FLAG_AUTOCONNECT: u32 = 1 << 0;
const PW_STREAM_FLAG_MAP_BUFFERS: u32 = 1 << 2;
const PW_STREAM_FLAG_RT_PROCESS: u32 = 1 << 4;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
}

/// Lock-free single-producer / single-consumer ring buffer for interleaved `f32` PCM samples.
///
/// Uses atomic `u32` bit storage (`f32::to_bits` / `f32::from_bits`) so guest audio submission
/// never blocks on PipeWire's real-time callback thread.
pub struct AudioRingBuffer {
    slots: Box<[AtomicU32]>,
    mask: usize,
    read_pos: AtomicUsize,
    write_pos: AtomicUsize,
    dropped_samples: AtomicU64,
    underrun_samples: AtomicU64,
    sample_rate_hz: u32,
}

impl AudioRingBuffer {
    /// Allocate a ring buffer whose capacity is rounded up to a power of two (minimum 64 samples).
    pub fn new(min_capacity_samples: usize, sample_rate_hz: u32) -> Self {
        let cap = min_capacity_samples.max(64).next_power_of_two();
        let slots = (0..cap).map(|_| AtomicU32::new(0)).collect::<Vec<_>>();
        Self {
            slots: slots.into_boxed_slice(),
            mask: cap - 1,
            read_pos: AtomicUsize::new(0),
            write_pos: AtomicUsize::new(0),
            dropped_samples: AtomicU64::new(0),
            underrun_samples: AtomicU64::new(0),
            sample_rate_hz: sample_rate_hz.max(8_000),
        }
    }

    /// Total capacity in interleaved `f32` samples.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Configured output sample rate in Hz.
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Number of samples currently queued for playback.
    pub fn available_read(&self) -> usize {
        let w = self.write_pos.load(Ordering::Acquire);
        let r = self.read_pos.load(Ordering::Acquire);
        w.wrapping_sub(r)
    }

    /// Free sample slots available before overflow.
    pub fn available_write(&self) -> usize {
        self.capacity().saturating_sub(self.available_read())
    }

    /// Cumulative samples dropped due to ring-buffer overflow.
    pub fn dropped_samples(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    /// Cumulative silence samples synthesized due to ring-buffer underrun.
    pub fn underrun_samples(&self) -> u64 {
        self.underrun_samples.load(Ordering::Relaxed)
    }

    /// Push interleaved stereo `f32` PCM without blocking. Returns the count accepted.
    pub fn push_f32(&self, interleaved: &[f32]) -> usize {
        let w = self.write_pos.load(Ordering::Relaxed);
        let r = self.read_pos.load(Ordering::Acquire);
        let free = self.capacity().saturating_sub(w.wrapping_sub(r));
        let count = interleaved.len().min(free);
        for (i, &sample) in interleaved.iter().take(count).enumerate() {
            let idx = (w.wrapping_add(i)) & self.mask;
            self.slots[idx].store(sample.clamp(-1.0, 1.0).to_bits(), Ordering::Relaxed);
        }
        self.write_pos
            .store(w.wrapping_add(count), Ordering::Release);
        if count < interleaved.len() {
            self.dropped_samples
                .fetch_add((interleaved.len() - count) as u64, Ordering::Relaxed);
        }
        count
    }

    /// Push a [`PcmBuffer`] (mono or stereo, resampling linearly if needed) into stereo `f32` output.
    pub fn push_pcm_buffer(&self, pcm: &PcmBuffer) -> usize {
        let stereo = upmix_and_resample_to_stereo(
            &pcm.samples,
            pcm.channels,
            pcm.sample_rate_hz,
            self.sample_rate_hz,
        );
        self.push_f32(&stereo)
    }

    /// Convert interleaved signed 16-bit PCM (e.g. XMA2 decoded frames) to stereo `f32` and push.
    pub fn push_i16(&self, pcm_i16: &[i16], channels: u8, sample_rate_hz: u32) -> usize {
        let f32_samples: Vec<f32> = pcm_i16.iter().map(|&s| f32::from(s) / 32768.0).collect();
        let stereo = upmix_and_resample_to_stereo(
            &f32_samples,
            channels,
            sample_rate_hz,
            self.sample_rate_hz,
        );
        self.push_f32(&stereo)
    }

    /// Pop interleaved `f32` samples into `out`, zero-filling any trailing underrun region.
    /// Returns the number of live samples read from the ring buffer.
    pub fn pop_f32(&self, out: &mut [f32]) -> usize {
        let r = self.read_pos.load(Ordering::Relaxed);
        let w = self.write_pos.load(Ordering::Acquire);
        let avail = w.wrapping_sub(r);
        let count = out.len().min(avail);
        for (i, dst) in out.iter_mut().take(count).enumerate() {
            let idx = (r.wrapping_add(i)) & self.mask;
            *dst = f32::from_bits(self.slots[idx].load(Ordering::Relaxed));
        }
        self.read_pos
            .store(r.wrapping_add(count), Ordering::Release);
        if count < out.len() {
            out[count..].fill(0.0);
            self.underrun_samples
                .fetch_add((out.len() - count) as u64, Ordering::Relaxed);
        }
        count
    }
}

/// Mix `src` additively into `dst` scaled by `gain`, clamping every sample to `[-1.0, 1.0]`.
pub fn mix_add_stereo(dst: &mut [f32], src: &[f32], gain: f32) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = (*d + s * gain).clamp(-1.0, 1.0);
    }
}

/// Upmix mono/stereo `f32` samples to stereo and linearly resample when `src_rate_hz != dst_rate_hz`.
pub fn upmix_and_resample_to_stereo(
    samples: &[f32],
    channels: u8,
    src_rate_hz: u32,
    dst_rate_hz: u32,
) -> Vec<f32> {
    if samples.is_empty() || src_rate_hz == 0 || dst_rate_hz == 0 {
        return Vec::new();
    }
    let stereo: Vec<f32> = match channels {
        1 => {
            let mut out = Vec::with_capacity(samples.len() * 2);
            for &s in samples {
                out.push(s);
                out.push(s);
            }
            out
        }
        2 => {
            let frames = samples.len() / 2;
            samples[..frames * 2].to_vec()
        }
        _ => {
            let ch = usize::from(channels.max(1));
            let frames = samples.len() / ch;
            let mut out = Vec::with_capacity(frames * 2);
            for f in 0..frames {
                out.push(samples[f * ch]);
                out.push(samples[f * ch + 1]);
            }
            out
        }
    };

    if src_rate_hz == dst_rate_hz {
        return stereo;
    }

    let src_frames = stereo.len() / 2;
    if src_frames == 0 {
        return Vec::new();
    }
    let dst_frames =
        (((src_frames as u64) * u64::from(dst_rate_hz)) / u64::from(src_rate_hz)).max(1) as usize;
    let mut out = Vec::with_capacity(dst_frames * 2);
    let ratio = f64::from(src_rate_hz) / f64::from(dst_rate_hz);
    for i in 0..dst_frames {
        let pos = (i as f64) * ratio;
        let idx0 = (pos.floor() as usize).min(src_frames - 1);
        let idx1 = (idx0 + 1).min(src_frames - 1);
        let frac = (pos - (idx0 as f64)) as f32;
        let l = stereo[idx0 * 2] + (stereo[idx1 * 2] - stereo[idx0 * 2]) * frac;
        let r = stereo[idx0 * 2 + 1] + (stereo[idx1 * 2 + 1] - stereo[idx0 * 2 + 1]) * frac;
        out.push(l);
        out.push(r);
    }
    out
}

/// Probe whether a live PipeWire daemon socket is accessible in the current Linux session.
pub fn probe_pipewire_socket() -> Option<PathBuf> {
    let remote = std::env::var("PIPEWIRE_REMOTE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "pipewire-0".to_owned());
    let runtime_dir = std::env::var("PIPEWIRE_RUNTIME_DIR")
        .or_else(|_| std::env::var("XDG_RUNTIME_DIR"))
        .ok()
        .filter(|s| !s.is_empty())?;
    let path = PathBuf::from(runtime_dir).join(remote);
    if path.exists() { Some(path) } else { None }
}

/// Active audio output sink name for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveAudioBackend {
    PipeWire,
    Null,
}

/// Deterministic headless audio sink draining submitted frames without touching host audio daemons.
pub struct NullAudioSink {
    ring: Arc<AudioRingBuffer>,
    drained_samples: AtomicU64,
}

impl NullAudioSink {
    pub fn new(sample_rate_hz: u32) -> Self {
        Self {
            ring: Arc::new(AudioRingBuffer::new(
                DEFAULT_RING_CAPACITY_SAMPLES,
                sample_rate_hz,
            )),
            drained_samples: AtomicU64::new(0),
        }
    }

    pub fn ring(&self) -> &Arc<AudioRingBuffer> {
        &self.ring
    }

    /// Drain up to `max_samples` from the ring buffer, returning how many live samples were drained.
    pub fn drain(&self, max_samples: usize) -> usize {
        let mut scratch = vec![0.0f32; max_samples.min(4096)];
        let n = self.ring.pop_f32(&mut scratch);
        self.drained_samples.fetch_add(n as u64, Ordering::Relaxed);
        n
    }

    pub fn drained_samples(&self) -> u64 {
        self.drained_samples.load(Ordering::Relaxed)
    }
}

// ============================================================================
// PipeWire 0.3 runtime `dlopen` FFI (verified against PipeWire 1.x ABI)
// ============================================================================

#[repr(C)]
struct SpaChunk {
    offset: u32,
    size: u32,
    stride: i32,
    flags: i32,
}

#[repr(C)]
struct SpaData {
    r#type: u32,
    flags: u32,
    fd: i64,
    mapoffset: u32,
    maxsize: u32,
    data: *mut c_void,
    chunk: *mut SpaChunk,
}

#[repr(C)]
struct SpaBuffer {
    n_metas: u32,
    n_datas: u32,
    metas: *mut c_void,
    datas: *mut SpaData,
}

#[repr(C)]
struct PwBuffer {
    buffer: *mut SpaBuffer,
    user_data: *mut c_void,
    size: u64,
    requested: u64,
    time: u64,
}

#[repr(C)]
struct PwStreamEvents {
    version: u32,
    _pad: u32,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    state_changed: Option<unsafe extern "C" fn(*mut c_void, c_int, c_int, *const c_char)>,
    control_info: Option<unsafe extern "C" fn(*mut c_void, u32, *const c_void)>,
    io_changed: Option<unsafe extern "C" fn(*mut c_void, u32, *mut c_void, u32)>,
    param_changed: Option<unsafe extern "C" fn(*mut c_void, u32, *const c_void)>,
    add_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut PwBuffer)>,
    remove_buffer: Option<unsafe extern "C" fn(*mut c_void, *mut PwBuffer)>,
    process: Option<unsafe extern "C" fn(*mut c_void)>,
    drained: Option<unsafe extern "C" fn(*mut c_void)>,
    command: Option<unsafe extern "C" fn(*mut c_void, *const c_void)>,
    trigger_done: Option<unsafe extern "C" fn(*mut c_void)>,
}

struct PwLibrary {
    handle: *mut c_void,
    pw_init: unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_char),
    pw_thread_loop_new: unsafe extern "C" fn(*const c_char, *const c_void) -> *mut c_void,
    pw_thread_loop_destroy: unsafe extern "C" fn(*mut c_void),
    pw_thread_loop_get_loop: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    pw_thread_loop_start: unsafe extern "C" fn(*mut c_void) -> c_int,
    pw_thread_loop_stop: unsafe extern "C" fn(*mut c_void),
    pw_thread_loop_lock: unsafe extern "C" fn(*mut c_void),
    pw_thread_loop_unlock: unsafe extern "C" fn(*mut c_void),
    pw_properties_new_string: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    pw_stream_new_simple: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        *mut c_void,
        *const PwStreamEvents,
        *mut c_void,
    ) -> *mut c_void,
    pw_stream_connect:
        unsafe extern "C" fn(*mut c_void, u32, u32, u32, *mut *const c_void, u32) -> c_int,
    pw_stream_dequeue_buffer: unsafe extern "C" fn(*mut c_void) -> *mut PwBuffer,
    pw_stream_queue_buffer: unsafe extern "C" fn(*mut c_void, *mut PwBuffer) -> c_int,
    pw_stream_destroy: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: Function pointers loaded from `libpipewire-0.3.so.0` are process-global immutable code
// addresses and the `dlopen` handle is kept alive until `PipeWireAudioSink::drop`.
unsafe impl Send for PwLibrary {}
// SAFETY: Function pointers are immutable and thread-safe to invoke through PipeWire's API.
unsafe impl Sync for PwLibrary {}

impl PwLibrary {
    fn load() -> Option<Self> {
        let lib_name = CString::new("libpipewire-0.3.so.0").ok()?;
        // SAFETY: Standard POSIX `dlopen` on a NUL-terminated library soname.
        let handle = unsafe { dlopen(lib_name.as_ptr(), RTLD_LAZY | RTLD_LOCAL) };
        if handle.is_null() {
            return None;
        }
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let cname = CString::new($name).ok()?;
                // SAFETY: `handle` is valid and `cname` is NUL-terminated.
                let ptr = unsafe { dlsym(handle, cname.as_ptr()) };
                if ptr.is_null() {
                    // SAFETY: Clean up `dlopen` handle on missing symbol.
                    unsafe { dlclose(handle) };
                    return None;
                }
                // SAFETY: Symbol signature verified against `/usr/include/pipewire-0.3/pipewire/`.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
            }};
        }

        Some(Self {
            handle,
            pw_init: sym!(
                "pw_init",
                unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_char)
            ),
            pw_thread_loop_new: sym!(
                "pw_thread_loop_new",
                unsafe extern "C" fn(*const c_char, *const c_void) -> *mut c_void
            ),
            pw_thread_loop_destroy: sym!(
                "pw_thread_loop_destroy",
                unsafe extern "C" fn(*mut c_void)
            ),
            pw_thread_loop_get_loop: sym!(
                "pw_thread_loop_get_loop",
                unsafe extern "C" fn(*mut c_void) -> *mut c_void
            ),
            pw_thread_loop_start: sym!(
                "pw_thread_loop_start",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            pw_thread_loop_stop: sym!("pw_thread_loop_stop", unsafe extern "C" fn(*mut c_void)),
            pw_thread_loop_lock: sym!("pw_thread_loop_lock", unsafe extern "C" fn(*mut c_void)),
            pw_thread_loop_unlock: sym!("pw_thread_loop_unlock", unsafe extern "C" fn(*mut c_void)),
            pw_properties_new_string: sym!(
                "pw_properties_new_string",
                unsafe extern "C" fn(*const c_char) -> *mut c_void
            ),
            pw_stream_new_simple: sym!(
                "pw_stream_new_simple",
                unsafe extern "C" fn(
                    *mut c_void,
                    *const c_char,
                    *mut c_void,
                    *const PwStreamEvents,
                    *mut c_void,
                ) -> *mut c_void
            ),
            pw_stream_connect: sym!(
                "pw_stream_connect",
                unsafe extern "C" fn(*mut c_void, u32, u32, u32, *mut *const c_void, u32) -> c_int
            ),
            pw_stream_dequeue_buffer: sym!(
                "pw_stream_dequeue_buffer",
                unsafe extern "C" fn(*mut c_void) -> *mut PwBuffer
            ),
            pw_stream_queue_buffer: sym!(
                "pw_stream_queue_buffer",
                unsafe extern "C" fn(*mut c_void, *mut PwBuffer) -> c_int
            ),
            pw_stream_destroy: sym!("pw_stream_destroy", unsafe extern "C" fn(*mut c_void)),
        })
    }
}

impl Drop for PwLibrary {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was returned by `dlopen` and is closed once on drop.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

struct PwCallbackCtx {
    ring: Arc<AudioRingBuffer>,
    stream: *mut c_void,
    dequeue_fn: unsafe extern "C" fn(*mut c_void) -> *mut PwBuffer,
    queue_fn: unsafe extern "C" fn(*mut c_void, *mut PwBuffer) -> c_int,
    active: AtomicBool,
}

unsafe extern "C" fn pw_on_process(userdata: *mut c_void) {
    if userdata.is_null() {
        return;
    }
    // SAFETY: `userdata` is the live `Box<PwCallbackCtx>` owned by `PipeWireAudioSink` while the
    // PipeWire thread loop is running (stopped before freeing in `Drop`).
    let ctx = unsafe { &*(userdata as *const PwCallbackCtx) };
    if !ctx.active.load(Ordering::Acquire) || ctx.stream.is_null() {
        return;
    }
    // SAFETY: Invoked on PipeWire's RT thread with the live `pw_stream*`.
    let pw_buf = unsafe { (ctx.dequeue_fn)(ctx.stream) };
    if pw_buf.is_null() {
        return;
    }
    // SAFETY: `pw_buf` was dequeued from PipeWire and owns a valid `spa_buffer` while dequeued.
    unsafe {
        let spa_buf = (*pw_buf).buffer;
        if !spa_buf.is_null() && (*spa_buf).n_datas > 0 && !(*spa_buf).datas.is_null() {
            let d = &mut *(*spa_buf).datas;
            if !d.data.is_null() && !d.chunk.is_null() && d.maxsize >= 8 {
                let stride = usize::from(DEFAULT_CHANNELS) * std::mem::size_of::<f32>();
                let max_frames = (d.maxsize as usize) / stride;
                let req_frames = (*pw_buf).requested as usize;
                let frames = if req_frames > 0 && req_frames <= max_frames {
                    req_frames
                } else {
                    max_frames.min(1024)
                };
                let sample_count = frames * usize::from(DEFAULT_CHANNELS);
                let out_slice = std::slice::from_raw_parts_mut(d.data as *mut f32, sample_count);
                let _ = ctx.ring.pop_f32(out_slice);
                (*d.chunk).offset = 0;
                (*d.chunk).stride = stride as i32;
                (*d.chunk).size = (frames * stride) as u32;
            }
        }
        let _ = (ctx.queue_fn)(ctx.stream, pw_buf);
    }
}

/// Build the 168-byte SPA POD (`SPA_PARAM_EnumFormat`) describing interleaved F32LE stereo PCM.
fn build_spa_audio_raw_f32_pod(sample_rate_hz: u32, channels: u32) -> [u32; 42] {
    [
        0x0000_00a0, // pod body size (160 B)
        0x0000_000f, // SPA_TYPE_Object
        0x0004_0003, // SPA_TYPE_OBJECT_Format
        0x0000_0003, // SPA_PARAM_EnumFormat
        // mediaType = SPA_MEDIA_TYPE_audio (1)
        0x0000_0001,
        0x0000_0000,
        0x0000_0004,
        0x0000_0003,
        0x0000_0001,
        0x0000_0000,
        // mediaSubtype = SPA_MEDIA_SUBTYPE_raw (1)
        0x0000_0002,
        0x0000_0000,
        0x0000_0004,
        0x0000_0003,
        0x0000_0001,
        0x0000_0000,
        // format = SPA_AUDIO_FORMAT_F32 (0x11b)
        0x0001_0001,
        0x0000_0000,
        0x0000_0004,
        0x0000_0003,
        0x0000_011b,
        0x0000_0000,
        // rate = sample_rate_hz
        0x0001_0003,
        0x0000_0000,
        0x0000_0004,
        0x0000_0004,
        sample_rate_hz,
        0x0000_0000,
        // channels = channels
        0x0001_0004,
        0x0000_0000,
        0x0000_0004,
        0x0000_0004,
        channels,
        0x0000_0000,
        // empty channel position array (FL/FR default)
        0x0001_0005,
        0x0000_0000,
        0x0000_0010,
        0x0000_000d,
        0x0000_0004,
        0x0000_0003,
        0x0000_0000,
        0x0000_0000,
    ]
}

static PW_INIT_ONCE: Once = Once::new();

/// Live PipeWire playback sink driven by a dedicated `pw_thread_loop`.
pub struct PipeWireAudioSink {
    lib: PwLibrary,
    thread_loop: *mut c_void,
    stream: *mut c_void,
    events: Box<PwStreamEvents>,
    cb_ctx: *mut PwCallbackCtx,
    ring: Arc<AudioRingBuffer>,
}

// SAFETY: `PipeWireAudioSink` coordinates all mutations through PipeWire's `pw_thread_loop_lock`
// and atomic ring-buffer indices; `Drop` stops the thread loop before freeing resources.
unsafe impl Send for PipeWireAudioSink {}
// SAFETY: Public methods only touch the lock-free `Arc<AudioRingBuffer>`.
unsafe impl Sync for PipeWireAudioSink {}

impl PipeWireAudioSink {
    /// Attempt to open a live PipeWire playback stream. Returns `None` if the daemon socket or
    /// `libpipewire-0.3.so.0` is unavailable.
    pub fn try_open(sample_rate_hz: u32) -> Option<Self> {
        let _socket = probe_pipewire_socket()?;
        let lib = PwLibrary::load()?;
        let rate = sample_rate_hz.max(8_000);
        let ring = Arc::new(AudioRingBuffer::new(DEFAULT_RING_CAPACITY_SAMPLES, rate));

        PW_INIT_ONCE.call_once(|| {
            // SAFETY: `pw_init(NULL, NULL)` is safe to invoke once per process.
            unsafe {
                (lib.pw_init)(ptr::null_mut(), ptr::null_mut());
            }
        });

        let loop_name = CString::new("x2linux-audio").ok()?;
        let stream_name = CString::new("X2Linux Audio").ok()?;
        let props_str = CString::new(
            "media.type=Audio media.category=Playback media.role=Game app.name=X2Linux",
        )
        .ok()?;

        // SAFETY: `loop_name` is a valid NUL-terminated C string.
        let thread_loop = unsafe { (lib.pw_thread_loop_new)(loop_name.as_ptr(), ptr::null()) };
        if thread_loop.is_null() {
            return None;
        }

        let cb_ctx = Box::into_raw(Box::new(PwCallbackCtx {
            ring: Arc::clone(&ring),
            stream: ptr::null_mut(),
            dequeue_fn: lib.pw_stream_dequeue_buffer,
            queue_fn: lib.pw_stream_queue_buffer,
            active: AtomicBool::new(false),
        }));

        let events = Box::new(PwStreamEvents {
            version: PW_VERSION_STREAM_EVENTS,
            _pad: 0,
            destroy: None,
            state_changed: None,
            control_info: None,
            io_changed: None,
            param_changed: None,
            add_buffer: None,
            remove_buffer: None,
            process: Some(pw_on_process),
            drained: None,
            command: None,
            trigger_done: None,
        });

        // SAFETY: PipeWire requires holding `pw_thread_loop_lock` around stream creation/connect.
        let stream = unsafe {
            (lib.pw_thread_loop_lock)(thread_loop);
            let pw_loop = (lib.pw_thread_loop_get_loop)(thread_loop);
            let props = (lib.pw_properties_new_string)(props_str.as_ptr());
            let stream = (lib.pw_stream_new_simple)(
                pw_loop,
                stream_name.as_ptr(),
                props,
                &*events,
                cb_ctx.cast::<c_void>(),
            );
            if stream.is_null() {
                (lib.pw_thread_loop_unlock)(thread_loop);
                (lib.pw_thread_loop_destroy)(thread_loop);
                drop(Box::from_raw(cb_ctx));
                return None;
            }
            (*cb_ctx).stream = stream;
            (*cb_ctx).active.store(true, Ordering::Release);

            let pod = build_spa_audio_raw_f32_pod(rate, u32::from(DEFAULT_CHANNELS));
            let mut param_ptr = pod.as_ptr().cast::<c_void>();
            let rc = (lib.pw_stream_connect)(
                stream,
                PW_DIRECTION_OUTPUT,
                PW_ID_ANY,
                PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS | PW_STREAM_FLAG_RT_PROCESS,
                &mut param_ptr,
                1,
            );
            (lib.pw_thread_loop_unlock)(thread_loop);
            if rc < 0 {
                (*cb_ctx).active.store(false, Ordering::Release);
                (lib.pw_stream_destroy)(stream);
                (lib.pw_thread_loop_destroy)(thread_loop);
                drop(Box::from_raw(cb_ctx));
                return None;
            }
            if (lib.pw_thread_loop_start)(thread_loop) < 0 {
                (*cb_ctx).active.store(false, Ordering::Release);
                (lib.pw_stream_destroy)(stream);
                (lib.pw_thread_loop_destroy)(thread_loop);
                drop(Box::from_raw(cb_ctx));
                return None;
            }
            stream
        };

        Some(Self {
            lib,
            thread_loop,
            stream,
            events,
            cb_ctx,
            ring,
        })
    }

    pub fn ring(&self) -> &Arc<AudioRingBuffer> {
        &self.ring
    }
}

impl Drop for PipeWireAudioSink {
    fn drop(&mut self) {
        // SAFETY: Stop the PipeWire thread loop before destroying the stream and freeing `cb_ctx`
        // so `pw_on_process` can never race with teardown.
        unsafe {
            if !self.cb_ctx.is_null() {
                (*self.cb_ctx).active.store(false, Ordering::Release);
            }
            if !self.thread_loop.is_null() {
                (self.lib.pw_thread_loop_stop)(self.thread_loop);
            }
            if !self.stream.is_null() {
                (self.lib.pw_stream_destroy)(self.stream);
                self.stream = ptr::null_mut();
            }
            if !self.thread_loop.is_null() {
                (self.lib.pw_thread_loop_destroy)(self.thread_loop);
                self.thread_loop = ptr::null_mut();
            }
            if !self.cb_ctx.is_null() {
                drop(Box::from_raw(self.cb_ctx));
                self.cb_ctx = ptr::null_mut();
            }
            let _ = &self.events;
        }
    }
}

/// Unified audio output selecting PipeWire when available and falling back to [`NullAudioSink`].
pub enum AudioOutput {
    PipeWire(PipeWireAudioSink),
    Null(NullAudioSink),
}

impl AudioOutput {
    /// Open PipeWire when `prefer_pipewire` is true and a live daemon is available; otherwise open
    /// [`NullAudioSink`].
    pub fn open(sample_rate_hz: u32, prefer_pipewire: bool) -> Self {
        if prefer_pipewire && let Some(pw) = PipeWireAudioSink::try_open(sample_rate_hz) {
            return Self::PipeWire(pw);
        }
        Self::Null(NullAudioSink::new(sample_rate_hz))
    }

    /// Explicitly create a headless [`NullAudioSink`].
    pub fn null(sample_rate_hz: u32) -> Self {
        Self::Null(NullAudioSink::new(sample_rate_hz))
    }

    pub fn backend(&self) -> ActiveAudioBackend {
        match self {
            Self::PipeWire(_) => ActiveAudioBackend::PipeWire,
            Self::Null(_) => ActiveAudioBackend::Null,
        }
    }

    pub fn ring(&self) -> &Arc<AudioRingBuffer> {
        match self {
            Self::PipeWire(pw) => pw.ring(),
            Self::Null(null) => null.ring(),
        }
    }

    /// Push interleaved stereo `f32` PCM into the active audio output without blocking.
    pub fn push_f32(&self, interleaved: &[f32]) -> usize {
        self.ring().push_f32(interleaved)
    }

    /// Push a decoded `x2l_formats::opus::PcmBuffer` into the active audio output.
    pub fn push_pcm_buffer(&self, pcm: &PcmBuffer) -> usize {
        self.ring().push_pcm_buffer(pcm)
    }

    /// Push decoded XMA2/16-bit PCM into the active audio output.
    pub fn push_i16(&self, pcm_i16: &[i16], channels: u8, sample_rate_hz: u32) -> usize {
        self.ring().push_i16(pcm_i16, channels, sample_rate_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_buffer_push_pop_wraparound_and_underrun() {
        let rb = AudioRingBuffer::new(64, DEFAULT_SAMPLE_RATE_HZ);
        assert_eq!(rb.capacity(), 64);

        let input: Vec<f32> = (0..48).map(|i| (i as f32) / 48.0).collect();
        assert_eq!(rb.push_f32(&input), 48);
        assert_eq!(rb.available_read(), 48);

        let mut out = vec![-1.0f32; 32];
        assert_eq!(rb.pop_f32(&mut out), 32);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[31] - (31.0 / 48.0)).abs() < 1e-6);

        // Push across wrap boundary and test overflow counter
        let burst = vec![0.5f32; 60];
        let accepted = rb.push_f32(&burst);
        assert_eq!(accepted, 48);
        assert_eq!(rb.dropped_samples(), 12);

        // Drain all 64 + trigger 8-sample silence underrun
        let mut drain = vec![1.0f32; 72];
        assert_eq!(rb.pop_f32(&mut drain), 64);
        assert_eq!(rb.underrun_samples(), 8);
        assert!(drain[64..72].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn xma_i16_opus_pcm_and_stereo_mixing_roundtrip() {
        let sink = NullAudioSink::new(48_000);

        // Mono 24 kHz Opus PcmBuffer -> stereo 48 kHz (4 frames -> 8 stereo frames = 16 samples)
        let opus_mono = PcmBuffer::from_i16(24_000, 1, &[0, 16384, -16384, 32767]).unwrap();
        let pushed_opus = sink.ring().push_pcm_buffer(&opus_mono);
        assert_eq!(pushed_opus, 16);

        // Stereo 48 kHz XMA2 i16 PCM
        let pushed_xma = sink
            .ring()
            .push_i16(&[16384, -16384, 8192, -8192], 2, 48_000);
        assert_eq!(pushed_xma, 4);

        let mut mixed = [0.25f32, -0.25, 0.9, -0.9];
        mix_add_stereo(&mut mixed, &[0.5, -0.5, 0.5, -0.5], 0.5);
        assert!((mixed[0] - 0.5).abs() < 1e-6);
        assert!((mixed[1] + 0.5).abs() < 1e-6);
        assert_eq!(mixed[2], 1.0);
        assert_eq!(mixed[3], -1.0);

        assert_eq!(sink.drain(64), 20);
        assert_eq!(sink.drained_samples(), 20);
    }

    #[test]
    fn null_fallback_and_live_pipewire_probe() {
        let null_out = AudioOutput::null(48_000);
        assert_eq!(null_out.backend(), ActiveAudioBackend::Null);
        assert_eq!(null_out.push_f32(&[0.1, -0.1, 0.2, -0.2]), 4);

        // Probe live PipeWire daemon on host; opens cleanly when present, falls back cleanly in CI.
        let socket = probe_pipewire_socket();
        let auto = AudioOutput::open(48_000, true);
        if socket.is_some() {
            assert_eq!(auto.backend(), ActiveAudioBackend::PipeWire);
            assert_eq!(auto.push_f32(&[0.0; 128]), 128);
        } else {
            assert_eq!(auto.backend(), ActiveAudioBackend::Null);
        }
    }
}
