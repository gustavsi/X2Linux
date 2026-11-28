// Clean-room Wine-thread-native `XTaskQueue` & `XAsync` implementation for `xgameruntime.dll`.
// Win64 ABI layouts (`XAsyncBlock` = 56 bytes, `XAsyncProviderData` = 32 bytes, dispatch modes,
// provider ops 0..4) informed by Marvisak/VIctory (commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/xasync.rs) and xcl-dev/xcl (commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/xgameruntime/{taskqueue,xasync}.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-28.

//! Functional Wine-managed `XTaskQueue` and `XAsync` runtime:
//! all guest callbacks (`XTaskQueueCallback`, `XAsyncCompletionRoutine`, `XAsyncProvider`) run
//! strictly on Wine-managed Win32 threads (never on Linux-side native worker threads).

use std::collections::{BTreeMap, VecDeque};
use std::ffi::{c_char, c_void};
use std::mem::{offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};
use x2l_shim_common::{
    HRESULT_E_ABORT, HRESULT_E_INVALIDARG, HRESULT_E_PENDING, HRESULT_E_POINTER, HRESULT_S_OK,
};

pub const HRESULT_E_BOUNDS: i32 = -2147483637; // 0x8000000B (insufficient buffer)

pub const PORT_WORK: u32 = 0;
pub const PORT_COMPLETION: u32 = 1;

pub const MODE_MANUAL: u32 = 0;
pub const MODE_THREAD_POOL: u32 = 1;
pub const MODE_SERIALIZED_THREAD_POOL: u32 = 2;
pub const MODE_IMMEDIATE: u32 = 3;

pub const XASYNC_OP_BEGIN: u32 = 0;
pub const XASYNC_OP_DO_WORK: u32 = 1;
pub const XASYNC_OP_GET_RESULT: u32 = 2;
pub const XASYNC_OP_CANCEL: u32 = 3;
pub const XASYNC_OP_CLEANUP: u32 = 4;

pub type XTaskQueueCallback = unsafe extern "win64" fn(context: *mut c_void, canceled: bool);
pub type XTaskQueueMonitorCallback =
    unsafe extern "win64" fn(context: *mut c_void, queue: *mut c_void, port: u32);
pub type XTaskQueueTerminatedCallback = unsafe extern "win64" fn(context: *mut c_void);
pub type XAsyncCompletionRoutine = unsafe extern "win64" fn(async_block: *mut XAsyncBlock);
pub type XAsyncProvider = unsafe extern "win64" fn(op: u32, data: *const XAsyncProviderData) -> i32;
pub type XAsyncWork = unsafe extern "win64" fn(async_block: *mut XAsyncBlock) -> i32;

/// Exact 56-byte Win64 `XAsyncBlock` layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XAsyncBlock {
    pub queue: *mut c_void,                        // @0
    pub context: *mut c_void,                      // @8
    pub callback: Option<XAsyncCompletionRoutine>, // @16
    pub internal: [u8; 32],                        // @24
}

/// Exact 32-byte Win64 `XAsyncProviderData` layout.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct XAsyncProviderData {
    pub async_block: *mut XAsyncBlock, // @0
    pub buffer_size: usize,            // @8
    pub buffer: *mut c_void,           // @16
    pub context: *mut c_void,          // @24
}

const _: () = assert!(size_of::<XAsyncBlock>() == 56);
const _: () = assert!(offset_of!(XAsyncBlock, queue) == 0);
const _: () = assert!(offset_of!(XAsyncBlock, context) == 8);
const _: () = assert!(offset_of!(XAsyncBlock, callback) == 16);
const _: () = assert!(offset_of!(XAsyncBlock, internal) == 24);
const _: () = assert!(size_of::<XAsyncProviderData>() == 32);
const _: () = assert!(offset_of!(XAsyncProviderData, async_block) == 0);
const _: () = assert!(offset_of!(XAsyncProviderData, buffer_size) == 8);
const _: () = assert!(offset_of!(XAsyncProviderData, buffer) == 16);
const _: () = assert!(offset_of!(XAsyncProviderData, context) == 24);

#[link(name = "kernel32")]
unsafe extern "system" {
    fn WaitForSingleObject(handle: *mut c_void, millis: u32) -> u32;
}

const WAIT_OBJECT_0: u32 = 0;

struct QueuedTask {
    due: Instant,
    context: usize,
    callback: XTaskQueueCallback,
}

struct PortInner {
    mode: u32,
    terminated: bool,
    items: VecDeque<QueuedTask>,
}

struct TaskQueuePort {
    inner: Mutex<PortInner>,
    cv: Condvar,
}

impl TaskQueuePort {
    fn new(mode: u32) -> Arc<Self> {
        let port = Arc::new(Self {
            inner: Mutex::new(PortInner {
                mode,
                terminated: false,
                items: VecDeque::new(),
            }),
            cv: Condvar::new(),
        });
        if mode == MODE_THREAD_POOL || mode == MODE_SERIALIZED_THREAD_POOL {
            let worker_port = Arc::clone(&port);
            let _ = std::thread::Builder::new()
                .name("x2l-xtaskqueue".into())
                .spawn(move || port_worker_loop(worker_port));
        }
        port
    }

    fn enqueue(
        &self,
        delay_ms: u32,
        context: usize,
        callback: XTaskQueueCallback,
    ) -> Result<(), i32> {
        let (mode, terminated) = {
            let guard = self.inner.lock().map_err(|_| HRESULT_E_ABORT)?;
            (guard.mode, guard.terminated)
        };
        if terminated {
            return Err(HRESULT_E_ABORT);
        }
        if mode == MODE_IMMEDIATE && delay_ms == 0 {
            // SAFETY: Immediate dispatch runs inline on the submitting Wine thread.
            unsafe {
                callback(context as *mut c_void, false);
            }
            return Ok(());
        }
        let due = Instant::now() + Duration::from_millis(u64::from(delay_ms));
        if let Ok(mut guard) = self.inner.lock() {
            if guard.terminated {
                return Err(HRESULT_E_ABORT);
            }
            guard.items.push_back(QueuedTask {
                due,
                context,
                callback,
            });
            self.cv.notify_one();
            Ok(())
        } else {
            Err(HRESULT_E_ABORT)
        }
    }

    fn dispatch_one(&self, timeout_ms: u32) -> bool {
        let deadline = if timeout_ms == u32::MAX {
            None
        } else {
            Some(Instant::now() + Duration::from_millis(u64::from(timeout_ms)))
        };

        let task: Option<(QueuedTask, bool)> = {
            let Ok(mut guard) = self.inner.lock() else {
                return false;
            };
            loop {
                let now = Instant::now();
                if let Some(pos) = guard
                    .items
                    .iter()
                    .position(|t| t.due <= now || guard.terminated)
                {
                    let item = guard.items.remove(pos).expect("index valid");
                    break Some((item, guard.terminated));
                }
                if guard.terminated {
                    break None;
                }
                let next_due = guard.items.iter().map(|t| t.due).min();
                let wait_dur = match (deadline, next_due) {
                    (Some(dl), Some(nd)) => {
                        let target = dl.min(nd);
                        if target <= now {
                            if dl <= now {
                                break None;
                            }
                            Duration::from_millis(0)
                        } else {
                            target - now
                        }
                    }
                    (Some(dl), None) => {
                        if dl <= now {
                            break None;
                        }
                        dl - now
                    }
                    (None, Some(nd)) => nd.saturating_duration_since(now),
                    (None, None) => {
                        guard = match self.cv.wait(guard) {
                            Ok(g) => g,
                            Err(_) => break None,
                        };
                        continue;
                    }
                };
                if wait_dur.is_zero() {
                    break None;
                }
                let (g, _) = match self.cv.wait_timeout(guard, wait_dur) {
                    Ok(v) => v,
                    Err(_) => break None,
                };
                guard = g;
            }
        };

        if let Some((item, canceled)) = task {
            // SAFETY: Guest callback executes on the dispatching Wine thread with lock released.
            unsafe {
                (item.callback)(item.context as *mut c_void, canceled);
            }
            true
        } else {
            false
        }
    }

    fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .map(|g| g.items.is_empty())
            .unwrap_or(true)
    }

    fn terminate(&self) {
        let canceled: Vec<QueuedTask> = {
            let Ok(mut guard) = self.inner.lock() else {
                return;
            };
            guard.terminated = true;
            self.cv.notify_all();
            if guard.mode == MODE_MANUAL {
                Vec::new()
            } else {
                guard.items.drain(..).collect()
            }
        };
        for item in canceled {
            // SAFETY: Deliver canceled notification on Wine thread outside port lock.
            unsafe {
                (item.callback)(item.context as *mut c_void, true);
            }
        }
    }
}

fn port_worker_loop(port: Arc<TaskQueuePort>) {
    loop {
        let task: Option<(QueuedTask, bool)> = {
            let Ok(mut guard) = port.inner.lock() else {
                return;
            };
            loop {
                let now = Instant::now();
                if let Some(pos) = guard.items.iter().position(|t| t.due <= now) {
                    let item = guard.items.remove(pos).expect("index valid");
                    break Some((item, guard.terminated));
                }
                if guard.terminated {
                    if let Some(item) = guard.items.pop_front() {
                        break Some((item, true));
                    }
                    return;
                }
                if let Some(next_due) = guard.items.iter().map(|t| t.due).min() {
                    let wait = next_due.saturating_duration_since(now);
                    let (g, _) = match port.cv.wait_timeout(guard, wait) {
                        Ok(v) => v,
                        Err(_) => return,
                    };
                    guard = g;
                } else {
                    guard = match port.cv.wait(guard) {
                        Ok(g) => g,
                        Err(_) => return,
                    };
                }
            }
        };
        if let Some((item, canceled)) = task {
            // SAFETY: Runs on Wine-managed `CreateThread` worker with port lock released.
            unsafe {
                (item.callback)(item.context as *mut c_void, canceled);
            }
        }
    }
}

struct MonitorEntry {
    context: usize,
    callback: XTaskQueueMonitorCallback,
}

struct WaiterEntry {
    stop: Arc<AtomicBool>,
}

struct TaskQueueObj {
    work: Arc<TaskQueuePort>,
    completion: Arc<TaskQueuePort>,
    monitors: Mutex<BTreeMap<u64, MonitorEntry>>,
    waiters: Mutex<BTreeMap<u64, WaiterEntry>>,
    terminated: AtomicBool,
}

struct QueueRegistry {
    next_id: u64,
    queues: BTreeMap<u64, Arc<TaskQueueObj>>,
    ports: BTreeMap<u64, Arc<TaskQueuePort>>,
    process_default: Option<u64>,
}

static QUEUES: OnceLock<Mutex<QueueRegistry>> = OnceLock::new();
static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn queue_reg() -> &'static Mutex<QueueRegistry> {
    QUEUES.get_or_init(|| {
        Mutex::new(QueueRegistry {
            next_id: 0x100,
            queues: BTreeMap::new(),
            ports: BTreeMap::new(),
            process_default: None,
        })
    })
}

fn alloc_queue_handle(obj: Arc<TaskQueueObj>) -> usize {
    let Ok(mut reg) = queue_reg().lock() else {
        return 0;
    };
    let id = reg.next_id;
    reg.next_id += 1;
    reg.queues.insert(id, obj);
    id as usize
}

fn lookup_queue(handle: *mut c_void) -> Option<Arc<TaskQueueObj>> {
    let id = handle as usize as u64;
    if id == 0 {
        return None;
    }
    let reg = queue_reg().lock().ok()?;
    reg.queues.get(&id).cloned()
}

pub fn get_or_create_process_queue_handle() -> *mut c_void {
    let Ok(mut reg) = queue_reg().lock() else {
        return ptr::null_mut();
    };
    if let Some(id) = reg.process_default
        && reg.queues.contains_key(&id)
    {
        return id as usize as *mut c_void;
    }
    let obj = Arc::new(TaskQueueObj {
        work: TaskQueuePort::new(MODE_THREAD_POOL),
        completion: TaskQueuePort::new(MODE_THREAD_POOL),
        monitors: Mutex::new(BTreeMap::new()),
        waiters: Mutex::new(BTreeMap::new()),
        terminated: AtomicBool::new(false),
    });
    let id = reg.next_id;
    reg.next_id += 1;
    reg.queues.insert(id, obj);
    reg.process_default = Some(id);
    id as usize as *mut c_void
}

pub fn is_valid_queue_handle(handle: *mut c_void) -> bool {
    let id = handle as usize as u64;
    queue_reg()
        .lock()
        .map(|r| r.queues.contains_key(&id) || r.ports.contains_key(&id))
        .unwrap_or(false)
}

fn notify_monitors(q: &TaskQueueObj, q_handle: *mut c_void, port: u32) {
    let snapshot: Vec<(usize, XTaskQueueMonitorCallback)> = q
        .monitors
        .lock()
        .map(|m| m.values().map(|e| (e.context, e.callback)).collect())
        .unwrap_or_default();
    for (ctx, cb) in snapshot {
        // SAFETY: Monitor callback executes outside queue lock on Wine thread.
        unsafe {
            cb(ctx as *mut c_void, q_handle, port);
        }
    }
}

// ============================================================================
// Exported XTaskQueue Win64 APIs
// ============================================================================

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueCreate(
    work_mode: u32,
    completion_mode: u32,
    out_queue: *mut *mut c_void,
) -> i32 {
    if out_queue.is_null() || work_mode > MODE_IMMEDIATE || completion_mode > MODE_IMMEDIATE {
        return HRESULT_E_INVALIDARG;
    }
    let obj = Arc::new(TaskQueueObj {
        work: TaskQueuePort::new(work_mode),
        completion: TaskQueuePort::new(completion_mode),
        monitors: Mutex::new(BTreeMap::new()),
        waiters: Mutex::new(BTreeMap::new()),
        terminated: AtomicBool::new(false),
    });
    let h = alloc_queue_handle(obj);
    // SAFETY: `out_queue` was verified non-null.
    unsafe {
        *out_queue = h as *mut c_void;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueCreateComposite(
    work_port: *mut c_void,
    completion_port: *mut c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if out_queue.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let (work, completion) = {
        let Ok(reg) = queue_reg().lock() else {
            return HRESULT_E_ABORT;
        };
        let Some(w) = reg.ports.get(&(work_port as usize as u64)).cloned() else {
            return HRESULT_E_INVALIDARG;
        };
        let Some(c) = reg.ports.get(&(completion_port as usize as u64)).cloned() else {
            return HRESULT_E_INVALIDARG;
        };
        (w, c)
    };
    let obj = Arc::new(TaskQueueObj {
        work,
        completion,
        monitors: Mutex::new(BTreeMap::new()),
        waiters: Mutex::new(BTreeMap::new()),
        terminated: AtomicBool::new(false),
    });
    let h = alloc_queue_handle(obj);
    // SAFETY: `out_queue` was verified non-null.
    unsafe {
        *out_queue = h as *mut c_void;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueDuplicateHandle(
    queue: *mut c_void,
    out_queue: *mut *mut c_void,
) -> i32 {
    if out_queue.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(obj) = lookup_queue(queue) else {
        return HRESULT_E_INVALIDARG;
    };
    let h = alloc_queue_handle(obj);
    // SAFETY: `out_queue` was verified non-null.
    unsafe {
        *out_queue = h as *mut c_void;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueCloseHandle(queue: *mut c_void) {
    let id = queue as usize as u64;
    if id == 0 {
        return;
    }
    if let Ok(mut reg) = queue_reg().lock() {
        if reg.process_default == Some(id) {
            return;
        }
        reg.queues.remove(&id);
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueGetCurrentProcessTaskQueue(out_queue: *mut *mut c_void) -> bool {
    if out_queue.is_null() {
        return false;
    }
    let h = get_or_create_process_queue_handle();
    if h.is_null() {
        return false;
    }
    // Return a duplicated handle per GDK contract so caller can safely call `CloseHandle`.
    let Some(obj) = lookup_queue(h) else {
        return false;
    };
    let dup = alloc_queue_handle(obj);
    // SAFETY: `out_queue` was verified non-null.
    unsafe {
        *out_queue = dup as *mut c_void;
    }
    dup != 0
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueSetCurrentProcessTaskQueue(queue: *mut c_void) {
    let id = queue as usize as u64;
    if let Ok(mut reg) = queue_reg().lock() {
        if id == 0 {
            reg.process_default = None;
        } else if reg.queues.contains_key(&id) {
            reg.process_default = Some(id);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueSubmitCallback(
    queue: *mut c_void,
    port: u32,
    context: *mut c_void,
    callback: Option<XTaskQueueCallback>,
) -> i32 {
    XTaskQueueSubmitDelayedCallback(queue, port, 0, context, callback)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueSubmitDelayedCallback(
    queue: *mut c_void,
    port: u32,
    delay_ms: u32,
    context: *mut c_void,
    callback: Option<XTaskQueueCallback>,
) -> i32 {
    let Some(cb) = callback else {
        return HRESULT_E_INVALIDARG;
    };
    let q_handle = if queue.is_null() {
        get_or_create_process_queue_handle()
    } else {
        queue
    };
    let Some(q) = lookup_queue(q_handle) else {
        return HRESULT_E_INVALIDARG;
    };
    if q.terminated.load(Ordering::Acquire) {
        return HRESULT_E_ABORT;
    }
    let target = match port {
        PORT_WORK => &q.work,
        PORT_COMPLETION => &q.completion,
        _ => return HRESULT_E_INVALIDARG,
    };
    match target.enqueue(delay_ms, context as usize, cb) {
        Ok(()) => {
            notify_monitors(&q, q_handle, port);
            HRESULT_S_OK
        }
        Err(hr) => hr,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueDispatch(queue: *mut c_void, port: u32, timeout_ms: u32) -> bool {
    let Some(q) = lookup_queue(queue) else {
        return false;
    };
    match port {
        PORT_WORK => q.work.dispatch_one(timeout_ms),
        PORT_COMPLETION => q.completion.dispatch_one(timeout_ms),
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueGetPort(
    queue: *mut c_void,
    port: u32,
    out_port: *mut *mut c_void,
) -> i32 {
    if out_port.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(q) = lookup_queue(queue) else {
        return HRESULT_E_INVALIDARG;
    };
    let p = match port {
        PORT_WORK => Arc::clone(&q.work),
        PORT_COMPLETION => Arc::clone(&q.completion),
        _ => return HRESULT_E_INVALIDARG,
    };
    let Ok(mut reg) = queue_reg().lock() else {
        return HRESULT_E_ABORT;
    };
    let id = reg.next_id;
    reg.next_id += 1;
    reg.ports.insert(id, p);
    // SAFETY: `out_port` was verified non-null.
    unsafe {
        *out_port = id as usize as *mut c_void;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueRegisterMonitor(
    queue: *mut c_void,
    callback_context: *mut c_void,
    callback: Option<XTaskQueueMonitorCallback>,
    out_token: *mut u64,
) -> i32 {
    let Some(cb) = callback else {
        return HRESULT_E_INVALIDARG;
    };
    if out_token.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let Some(q) = lookup_queue(queue) else {
        return HRESULT_E_INVALIDARG;
    };
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    if let Ok(mut guard) = q.monitors.lock() {
        guard.insert(
            token,
            MonitorEntry {
                context: callback_context as usize,
                callback: cb,
            },
        );
    }
    // SAFETY: `out_token` was verified non-null.
    unsafe {
        *out_token = token;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueUnregisterMonitor(queue: *mut c_void, token: u64) {
    if let Some(q) = lookup_queue(queue)
        && let Ok(mut guard) = q.monitors.lock()
    {
        guard.remove(&token);
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueRegisterWaiter(
    queue: *mut c_void,
    port: u32,
    wait_handle: *mut c_void,
    callback_context: *mut c_void,
    callback: Option<XTaskQueueCallback>,
    out_token: *mut u64,
) -> i32 {
    let Some(cb) = callback else {
        return HRESULT_E_INVALIDARG;
    };
    if wait_handle.is_null()
        || out_token.is_null()
        || (port != PORT_WORK && port != PORT_COMPLETION)
    {
        return HRESULT_E_INVALIDARG;
    }
    let Some(q) = lookup_queue(queue) else {
        return HRESULT_E_INVALIDARG;
    };
    let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed);
    let stop = Arc::new(AtomicBool::new(false));
    if let Ok(mut guard) = q.waiters.lock() {
        guard.insert(
            token,
            WaiterEntry {
                stop: Arc::clone(&stop),
            },
        );
    }
    let q_raw = queue as usize;
    let handle_raw = wait_handle as usize;
    let ctx_raw = callback_context as usize;
    let _ = std::thread::Builder::new()
        .name("x2l-xtask-waiter".into())
        .spawn(move || {
            while !stop.load(Ordering::Acquire) {
                // SAFETY: Polls Win32 event/waitable handle on a Wine-managed thread.
                let rc = unsafe { WaitForSingleObject(handle_raw as *mut c_void, 25) };
                if rc == WAIT_OBJECT_0 && !stop.load(Ordering::Acquire) {
                    let _ = XTaskQueueSubmitCallback(
                        q_raw as *mut c_void,
                        port,
                        ctx_raw as *mut c_void,
                        Some(cb),
                    );
                    break;
                }
            }
        });
    // SAFETY: `out_token` was verified non-null.
    unsafe {
        *out_token = token;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueUnregisterWaiter(queue: *mut c_void, token: u64) {
    if let Some(q) = lookup_queue(queue)
        && let Ok(mut guard) = q.waiters.lock()
        && let Some(entry) = guard.remove(&token)
    {
        entry.stop.store(true, Ordering::Release);
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueTerminate(
    queue: *mut c_void,
    _wait: bool,
    callback_context: *mut c_void,
    callback: Option<XTaskQueueTerminatedCallback>,
) -> i32 {
    let Some(q) = lookup_queue(queue) else {
        return HRESULT_E_INVALIDARG;
    };
    q.terminated.store(true, Ordering::Release);
    if let Ok(mut waiters) = q.waiters.lock() {
        for entry in waiters.values() {
            entry.stop.store(true, Ordering::Release);
        }
        waiters.clear();
    }
    q.work.terminate();
    q.completion.terminate();
    if let Some(cb) = callback {
        // SAFETY: Runs terminated notification on Wine thread outside queue locks.
        unsafe {
            cb(callback_context);
        }
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XTaskQueueIsEmpty(queue: *mut c_void, port: u32) -> bool {
    let Some(q) = lookup_queue(queue) else {
        return true;
    };
    match port {
        PORT_WORK => q.work.is_empty(),
        PORT_COMPLETION => q.completion.is_empty(),
        _ => true,
    }
}

// ============================================================================
// XAsync State Machine
// ============================================================================

struct AsyncRecord {
    queue: usize,
    provider: Option<XAsyncProvider>,
    provider_ctx: usize,
    identity: usize,
    status: i32,
    completed: bool,
    canceled: bool,
    result_size: usize,
    /// Optional inline payload for X2Linux-authored async APIs (`XUserAddAsync`, `XGameSave*`, etc.).
    inline_result: Option<Vec<u8>>,
    /// Optional `XAsyncRun` work callback.
    run_work: Option<XAsyncWork>,
}

static ASYNC_MAP: OnceLock<(Mutex<BTreeMap<usize, AsyncRecord>>, Condvar)> = OnceLock::new();

fn async_state() -> &'static (Mutex<BTreeMap<usize, AsyncRecord>>, Condvar) {
    ASYNC_MAP.get_or_init(|| (Mutex::new(BTreeMap::new()), Condvar::new()))
}

extern "win64" fn async_completion_trampoline(ctx: *mut c_void, _canceled: bool) {
    if ctx.is_null() {
        return;
    }
    let block = ctx.cast::<XAsyncBlock>();
    // SAFETY: Caller's `XAsyncBlock` remains live until its completion callback finishes.
    let cb = unsafe { (*block).callback };
    if let Some(func) = cb {
        // SAFETY: Invokes guest `XAsyncCompletionRoutine` on a Wine-managed thread.
        unsafe {
            func(block);
        }
    }
}

extern "win64" fn async_provider_work_trampoline(ctx: *mut c_void, canceled: bool) {
    if ctx.is_null() {
        return;
    }
    let key = ctx as usize;
    let (provider, provider_ctx, run_work) = {
        let Ok(guard) = async_state().0.lock() else {
            return;
        };
        let Some(rec) = guard.get(&key) else {
            return;
        };
        (rec.provider, rec.provider_ctx, rec.run_work)
    };
    if canceled {
        XAsyncComplete(ctx.cast::<XAsyncBlock>(), HRESULT_E_ABORT, 0);
        return;
    }
    if let Some(work_fn) = run_work {
        // SAFETY: Runs guest `XAsyncWork` on a Wine thread and publishes result.
        let hr = unsafe { work_fn(ctx.cast::<XAsyncBlock>()) };
        XAsyncComplete(ctx.cast::<XAsyncBlock>(), hr, 0);
        return;
    }
    if let Some(prov) = provider {
        let data = XAsyncProviderData {
            async_block: ctx.cast::<XAsyncBlock>(),
            buffer_size: 0,
            buffer: ptr::null_mut(),
            context: provider_ctx as *mut c_void,
        };
        // SAFETY: Runs guest `XAsyncProvider(DoWork)` on a Wine thread.
        let hr = unsafe { prov(XASYNC_OP_DO_WORK, &data) };
        if hr != HRESULT_E_PENDING {
            XAsyncComplete(ctx.cast::<XAsyncBlock>(), hr, 0);
        }
    }
}

/// Complete an X2Linux-implemented async GDK call cleanly with an optional inline result payload
/// and schedule `async_block.callback` on `async_block.queue` (or the process default queue).
pub fn complete_builtin_async(
    async_block: *mut XAsyncBlock,
    status: i32,
    payload: Option<Vec<u8>>,
) -> i32 {
    complete_builtin_async_with_identity(async_block, ptr::null(), status, payload)
}

pub(crate) fn complete_builtin_async_with_identity(
    async_block: *mut XAsyncBlock,
    identity: *const c_void,
    status: i32,
    payload: Option<Vec<u8>>,
) -> i32 {
    if async_block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    // SAFETY: Caller supplied a live `XAsyncBlock`.
    let q_handle = unsafe {
        if (*async_block).queue.is_null() {
            get_or_create_process_queue_handle()
        } else {
            (*async_block).queue
        }
    };
    let size = payload.as_ref().map(|v| v.len()).unwrap_or(0);
    let key = async_block as usize;
    {
        let (lock, cv) = async_state();
        if let Ok(mut guard) = lock.lock() {
            guard.insert(
                key,
                AsyncRecord {
                    queue: q_handle as usize,
                    provider: None,
                    provider_ctx: 0,
                    identity: identity as usize,
                    status,
                    completed: true,
                    canceled: false,
                    result_size: size,
                    inline_result: payload,
                    run_work: None,
                },
            );
            cv.notify_all();
        } else if !identity.is_null() {
            return HRESULT_E_ABORT;
        }
    }
    let submit_status = XTaskQueueSubmitCallback(
        q_handle,
        PORT_COMPLETION,
        async_block.cast::<c_void>(),
        Some(async_completion_trampoline),
    );
    if submit_status != HRESULT_S_OK && !identity.is_null() {
        if let Ok(mut records) = async_state().0.lock() {
            records.remove(&key);
        }
        return submit_status;
    }
    HRESULT_S_OK
}

/// Read the result of an X2Linux-implemented async call, optionally blocking if pending.
pub fn take_builtin_async_result(
    async_block: *mut XAsyncBlock,
) -> Result<(i32, Option<Vec<u8>>), i32> {
    if async_block.is_null() {
        return Err(HRESULT_E_INVALIDARG);
    }
    let status = XAsyncGetStatus(async_block, true);
    let key = async_block as usize;
    let (lock, _) = async_state();
    let Ok(mut guard) = lock.lock() else {
        return Err(HRESULT_E_ABORT);
    };
    let Some(rec) = guard.get_mut(&key) else {
        return Err(HRESULT_E_INVALIDARG);
    };
    if status < 0 {
        return Err(status);
    }
    Ok((rec.status, rec.inline_result.clone()))
}

pub(crate) fn read_builtin_async_payload<const N: usize>(
    block: *mut XAsyncBlock,
    identity: *const u8,
) -> Result<[u8; N], i32> {
    let records = async_state().0.lock().map_err(|_| HRESULT_E_ABORT)?;
    let record = records
        .get(&(block as usize))
        .filter(|record| record.identity == identity as usize && record.provider.is_none())
        .ok_or(HRESULT_E_INVALIDARG)?;
    if !record.completed {
        return Err(HRESULT_E_PENDING);
    }
    if record.status < 0 {
        return Err(record.status);
    }
    record
        .inline_result
        .as_deref()
        .and_then(|payload| payload.try_into().ok())
        .ok_or(HRESULT_E_INVALIDARG)
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncBegin(
    async_block: *mut XAsyncBlock,
    context: *mut c_void,
    identity: *const c_void,
    _identity_name: *const c_char,
    provider: Option<XAsyncProvider>,
) -> i32 {
    if async_block.is_null() || provider.is_none() {
        return HRESULT_E_INVALIDARG;
    }
    // SAFETY: Caller supplied a live `XAsyncBlock`.
    let q_handle = unsafe {
        if (*async_block).queue.is_null() {
            get_or_create_process_queue_handle()
        } else {
            (*async_block).queue
        }
    };
    let key = async_block as usize;
    {
        let (lock, _) = async_state();
        if let Ok(mut guard) = lock.lock() {
            guard.insert(
                key,
                AsyncRecord {
                    queue: q_handle as usize,
                    provider,
                    provider_ctx: context as usize,
                    identity: identity as usize,
                    status: HRESULT_E_PENDING,
                    completed: false,
                    canceled: false,
                    result_size: 0,
                    inline_result: None,
                    run_work: None,
                },
            );
        }
    }
    let data = XAsyncProviderData {
        async_block,
        buffer_size: 0,
        buffer: ptr::null_mut(),
        context,
    };
    // SAFETY: Calls provider `Begin` on the calling Wine thread outside `ASYNC_MAP` lock.
    let hr = unsafe { provider.unwrap()(XASYNC_OP_BEGIN, &data) };
    if hr < 0 && hr != HRESULT_E_PENDING {
        XAsyncComplete(async_block, hr, 0);
    }
    hr
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncSchedule(async_block: *mut XAsyncBlock, delay_ms: u32) -> i32 {
    if async_block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let key = async_block as usize;
    let q_handle = {
        let Ok(guard) = async_state().0.lock() else {
            return HRESULT_E_ABORT;
        };
        let Some(rec) = guard.get(&key) else {
            return HRESULT_E_INVALIDARG;
        };
        rec.queue as *mut c_void
    };
    XTaskQueueSubmitDelayedCallback(
        q_handle,
        PORT_WORK,
        delay_ms,
        async_block.cast::<c_void>(),
        Some(async_provider_work_trampoline),
    )
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncComplete(
    async_block: *mut XAsyncBlock,
    result: i32,
    required_buffer_size: usize,
) {
    if async_block.is_null() {
        return;
    }
    let key = async_block as usize;
    let q_handle = {
        let (lock, cv) = async_state();
        let Ok(mut guard) = lock.lock() else {
            return;
        };
        let Some(rec) = guard.get_mut(&key) else {
            return;
        };
        if rec.completed {
            return;
        }
        rec.status = result;
        rec.result_size = required_buffer_size;
        rec.completed = true;
        let q = rec.queue as *mut c_void;
        cv.notify_all();
        q
    };
    let _ = XTaskQueueSubmitCallback(
        q_handle,
        PORT_COMPLETION,
        async_block.cast::<c_void>(),
        Some(async_completion_trampoline),
    );
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncGetStatus(async_block: *mut XAsyncBlock, wait: bool) -> i32 {
    if async_block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let key = async_block as usize;
    let (lock, cv) = async_state();
    let Ok(mut guard) = lock.lock() else {
        return HRESULT_E_ABORT;
    };
    loop {
        let Some(rec) = guard.get(&key) else {
            return HRESULT_E_INVALIDARG;
        };
        if rec.completed {
            return rec.status;
        }
        if !wait {
            return HRESULT_E_PENDING;
        }
        guard = match cv.wait(guard) {
            Ok(g) => g,
            Err(_) => return HRESULT_E_ABORT,
        };
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncGetResultSize(
    async_block: *mut XAsyncBlock,
    out_size: *mut usize,
) -> i32 {
    if async_block.is_null() || out_size.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let key = async_block as usize;
    let Ok(guard) = async_state().0.lock() else {
        return HRESULT_E_ABORT;
    };
    let Some(rec) = guard.get(&key) else {
        return HRESULT_E_INVALIDARG;
    };
    if !rec.completed {
        return HRESULT_E_PENDING;
    }
    if rec.status < 0 {
        return rec.status;
    }
    // SAFETY: `out_size` was verified non-null.
    unsafe {
        *out_size = rec.result_size;
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncGetResult(
    async_block: *mut XAsyncBlock,
    identity: *const c_void,
    buffer_size: usize,
    buffer: *mut c_void,
    out_used: *mut usize,
) -> i32 {
    if async_block.is_null() {
        return HRESULT_E_INVALIDARG;
    }
    let key = async_block as usize;
    let (status, required, expected_id, provider, provider_ctx, inline) = {
        let Ok(guard) = async_state().0.lock() else {
            return HRESULT_E_ABORT;
        };
        let Some(rec) = guard.get(&key) else {
            return HRESULT_E_INVALIDARG;
        };
        if !rec.completed {
            return HRESULT_E_PENDING;
        }
        (
            rec.status,
            rec.result_size,
            rec.identity,
            rec.provider,
            rec.provider_ctx,
            rec.inline_result.clone(),
        )
    };
    if status < 0 {
        return status;
    }
    if expected_id != 0 && !identity.is_null() && (identity as usize) != expected_id {
        return HRESULT_E_INVALIDARG;
    }
    if required > 0 && (buffer.is_null() || buffer_size < required) {
        return HRESULT_E_BOUNDS;
    }
    if let Some(bytes) = inline {
        if !bytes.is_empty() && !buffer.is_null() {
            // SAFETY: `buffer_size >= required == bytes.len()`.
            unsafe {
                ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), bytes.len());
            }
        }
        if !out_used.is_null() {
            // SAFETY: Caller passed optional writable `out_used`.
            unsafe { *out_used = bytes.len() };
        }
        return HRESULT_S_OK;
    }
    if let Some(prov) = provider {
        let data = XAsyncProviderData {
            async_block,
            buffer_size,
            buffer,
            context: provider_ctx as *mut c_void,
        };
        // SAFETY: Runs provider `GetResult` on calling Wine thread.
        let hr = unsafe { prov(XASYNC_OP_GET_RESULT, &data) };
        if hr >= 0 && !out_used.is_null() {
            // SAFETY: Caller passed optional writable `out_used`.
            unsafe { *out_used = required };
        }
        return hr;
    }
    if !out_used.is_null() {
        // SAFETY: Caller passed optional writable `out_used`.
        unsafe { *out_used = 0 };
    }
    HRESULT_S_OK
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncCancel(async_block: *mut XAsyncBlock) {
    if async_block.is_null() {
        return;
    }
    let key = async_block as usize;
    let (provider, provider_ctx, already_done) = {
        let Ok(mut guard) = async_state().0.lock() else {
            return;
        };
        let Some(rec) = guard.get_mut(&key) else {
            return;
        };
        if rec.completed || rec.canceled {
            return;
        }
        rec.canceled = true;
        (rec.provider, rec.provider_ctx, rec.completed)
    };
    if !already_done {
        if let Some(prov) = provider {
            let data = XAsyncProviderData {
                async_block,
                buffer_size: 0,
                buffer: ptr::null_mut(),
                context: provider_ctx as *mut c_void,
            };
            // SAFETY: Notifies provider of cancellation on calling Wine thread.
            let _ = unsafe { prov(XASYNC_OP_CANCEL, &data) };
        }
        XAsyncComplete(async_block, HRESULT_E_ABORT, 0);
    }
}

#[unsafe(no_mangle)]
pub extern "win64" fn XAsyncRun(async_block: *mut XAsyncBlock, work: Option<XAsyncWork>) -> i32 {
    if async_block.is_null() || work.is_none() {
        return HRESULT_E_POINTER;
    }
    // SAFETY: Caller supplied a live `XAsyncBlock`.
    let q_handle = unsafe {
        if (*async_block).queue.is_null() {
            get_or_create_process_queue_handle()
        } else {
            (*async_block).queue
        }
    };
    let key = async_block as usize;
    {
        let (lock, _) = async_state();
        if let Ok(mut guard) = lock.lock() {
            guard.insert(
                key,
                AsyncRecord {
                    queue: q_handle as usize,
                    provider: None,
                    provider_ctx: 0,
                    identity: 0,
                    status: HRESULT_E_PENDING,
                    completed: false,
                    canceled: false,
                    result_size: 0,
                    inline_result: None,
                    run_work: work,
                },
            );
        }
    }
    XAsyncSchedule(async_block, 0)
}
