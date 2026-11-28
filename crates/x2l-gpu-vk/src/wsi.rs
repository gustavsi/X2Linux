// Clean-room SUD-free native Linux Vulkan WSI Swapchain (Wayland + XCB X11) via runtime `dlopen`.
// License: GNU General Public License v3.0 or later (GPL-3.0-or-later).

//! Native Linux Vulkan WSI swapchain presenter (`Wayland` via `libwayland-client.so.0` + `xdg-shell`,
//! `X11` fallback via `libxcb.so.1`) running on X2Linux's SUD-free native Linux GPU worker thread
//! with zero C build-time headers or extra Cargo dependencies.

use crate::GpuError;
use ash::vk;
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

const RTLD_NOW: c_int = 0x0002;
const RTLD_LOCAL: c_int = 0x0000;
const POLLIN: i16 = 0x0001;

const DEFAULT_WINDOW_WIDTH: u32 = 640;
const DEFAULT_WINDOW_HEIGHT: u32 = 480;

#[repr(C)]
struct PollFd {
    fd: c_int,
    events: i16,
    revents: i16,
}

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn free(ptr: *mut c_void);
    fn poll(fds: *mut PollFd, nfds: u64, timeout: c_int) -> c_int;
}

/// Active presentation surface reported in `report.json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresentSurfaceKind {
    Headless,
    Wayland,
    X11,
}

impl PresentSurfaceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Headless => "headless",
            Self::Wayland => "wayland",
            Self::X11 => "x11",
        }
    }
}

// ============================================================================
// 1. Wayland (`libwayland-client.so.0`) runtime `dlopen` + static `xdg-shell`
// ============================================================================

#[repr(C)]
struct WlMessage {
    name: *const c_char,
    signature: *const c_char,
    types: *const SyncIfacePtr,
}

#[repr(C)]
struct WlInterface {
    name: *const c_char,
    version: c_int,
    method_count: c_int,
    methods: *const WlMessage,
    event_count: c_int,
    events: *const WlMessage,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct SyncIfacePtr(*const WlInterface);

// SAFETY: Static Wayland protocol metadata tables are immutable global descriptors.
unsafe impl Sync for WlMessage {}
unsafe impl Sync for WlInterface {}
unsafe impl Sync for SyncIfacePtr {}

#[repr(C)]
union WlArgument {
    i: i32,
    u: u32,
    s: *const c_char,
    o: *mut c_void,
    n: u32,
}

const WL_MARSHAL_FLAG_DESTROY: u32 = 1 << 0;

static NULL_TYPES_4: [SyncIfacePtr; 4] = [SyncIfacePtr(ptr::null()); 4];
static XDG_WM_BASE_GET_SURFACE_TYPES: [SyncIfacePtr; 2] = [
    SyncIfacePtr(&XDG_SURFACE_INTERFACE as *const WlInterface),
    SyncIfacePtr(ptr::null()),
];
static XDG_SURFACE_GET_TOPLEVEL_TYPES: [SyncIfacePtr; 1] =
    [SyncIfacePtr(&XDG_TOPLEVEL_INTERFACE as *const WlInterface)];

static XDG_WM_BASE_REQUESTS: [WlMessage; 4] = [
    WlMessage {
        name: c"destroy".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"create_positioner".as_ptr(),
        signature: c"n".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"get_xdg_surface".as_ptr(),
        signature: c"no".as_ptr(),
        types: XDG_WM_BASE_GET_SURFACE_TYPES.as_ptr(),
    },
    WlMessage {
        name: c"pong".as_ptr(),
        signature: c"u".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
];

static XDG_WM_BASE_EVENTS: [WlMessage; 1] = [WlMessage {
    name: c"ping".as_ptr(),
    signature: c"u".as_ptr(),
    types: NULL_TYPES_4.as_ptr(),
}];

static XDG_WM_BASE_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_wm_base".as_ptr(),
    version: 1,
    method_count: 4,
    methods: XDG_WM_BASE_REQUESTS.as_ptr(),
    event_count: 1,
    events: XDG_WM_BASE_EVENTS.as_ptr(),
};

static XDG_SURFACE_REQUESTS: [WlMessage; 5] = [
    WlMessage {
        name: c"destroy".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"get_toplevel".as_ptr(),
        signature: c"n".as_ptr(),
        types: XDG_SURFACE_GET_TOPLEVEL_TYPES.as_ptr(),
    },
    WlMessage {
        name: c"get_popup".as_ptr(),
        signature: c"n?oo".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_window_geometry".as_ptr(),
        signature: c"iiii".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"ack_configure".as_ptr(),
        signature: c"u".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
];

static XDG_SURFACE_EVENTS: [WlMessage; 1] = [WlMessage {
    name: c"configure".as_ptr(),
    signature: c"u".as_ptr(),
    types: NULL_TYPES_4.as_ptr(),
}];

static XDG_SURFACE_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_surface".as_ptr(),
    version: 1,
    method_count: 5,
    methods: XDG_SURFACE_REQUESTS.as_ptr(),
    event_count: 1,
    events: XDG_SURFACE_EVENTS.as_ptr(),
};

static XDG_TOPLEVEL_REQUESTS: [WlMessage; 14] = [
    WlMessage {
        name: c"destroy".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_parent".as_ptr(),
        signature: c"?o".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_title".as_ptr(),
        signature: c"s".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_app_id".as_ptr(),
        signature: c"s".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"show_window_menu".as_ptr(),
        signature: c"ouii".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"move".as_ptr(),
        signature: c"ou".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"resize".as_ptr(),
        signature: c"ouu".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_max_size".as_ptr(),
        signature: c"ii".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_min_size".as_ptr(),
        signature: c"ii".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_maximized".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"unset_maximized".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_fullscreen".as_ptr(),
        signature: c"?o".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"unset_fullscreen".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"set_minimized".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
];

static XDG_TOPLEVEL_EVENTS: [WlMessage; 2] = [
    WlMessage {
        name: c"configure".as_ptr(),
        signature: c"iia".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
    WlMessage {
        name: c"close".as_ptr(),
        signature: c"".as_ptr(),
        types: NULL_TYPES_4.as_ptr(),
    },
];

static XDG_TOPLEVEL_INTERFACE: WlInterface = WlInterface {
    name: c"xdg_toplevel".as_ptr(),
    version: 1,
    method_count: 14,
    methods: XDG_TOPLEVEL_REQUESTS.as_ptr(),
    event_count: 2,
    events: XDG_TOPLEVEL_EVENTS.as_ptr(),
};

#[repr(C)]
struct WlRegistryListener {
    global: unsafe extern "C" fn(*mut c_void, *mut c_void, u32, *const c_char, u32),
    global_remove: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgWmBaseListener {
    ping: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgSurfaceListener {
    configure: unsafe extern "C" fn(*mut c_void, *mut c_void, u32),
}

#[repr(C)]
struct XdgToplevelListener {
    configure: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, *mut c_void),
    close: unsafe extern "C" fn(*mut c_void, *mut c_void),
}

struct WaylandLib {
    handle: *mut c_void,
    display_connect: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    display_disconnect: unsafe extern "C" fn(*mut c_void),
    display_roundtrip: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_dispatch_pending: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_flush: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_get_fd: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_prepare_read: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_read_events: unsafe extern "C" fn(*mut c_void) -> c_int,
    display_cancel_read: unsafe extern "C" fn(*mut c_void),
    proxy_marshal_array_flags: unsafe extern "C" fn(
        *mut c_void,
        u32,
        *const WlInterface,
        u32,
        u32,
        *mut WlArgument,
    ) -> *mut c_void,
    proxy_add_listener:
        unsafe extern "C" fn(*mut c_void, *const *const c_void, *mut c_void) -> c_int,
    proxy_destroy: unsafe extern "C" fn(*mut c_void),
    proxy_get_version: unsafe extern "C" fn(*mut c_void) -> u32,
    wl_registry_interface: *const WlInterface,
    wl_compositor_interface: *const WlInterface,
    wl_surface_interface: *const WlInterface,
}

impl WaylandLib {
    fn load() -> Result<Self, GpuError> {
        // SAFETY: Standard POSIX `dlopen` on immutable soname literal.
        let handle = unsafe { dlopen(c"libwayland-client.so.0".as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(GpuError::Vulkan(
                "libwayland-client.so.0 not available".into(),
            ));
        }
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let cname = CString::new($name).expect("static ASCII");
                // SAFETY: `handle` is live and `cname` is NUL-terminated.
                let p = unsafe { dlsym(handle, cname.as_ptr()) };
                if p.is_null() {
                    // SAFETY: Clean up handle on missing symbol.
                    unsafe { dlclose(handle) };
                    return Err(GpuError::Vulkan(format!(
                        "missing Wayland symbol {}",
                        $name
                    )));
                }
                // SAFETY: Symbol ABI verified against libwayland-client 1.22+.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        macro_rules! data_sym {
            ($name:literal) => {{
                let cname = CString::new($name).expect("static ASCII");
                // SAFETY: `handle` is live and `cname` is NUL-terminated.
                let p = unsafe { dlsym(handle, cname.as_ptr()) };
                if p.is_null() {
                    // SAFETY: Clean up handle on missing symbol.
                    unsafe { dlclose(handle) };
                    return Err(GpuError::Vulkan(format!(
                        "missing Wayland data symbol {}",
                        $name
                    )));
                }
                p as *const WlInterface
            }};
        }

        Ok(Self {
            handle,
            display_connect: sym!(
                "wl_display_connect",
                unsafe extern "C" fn(*const c_char) -> *mut c_void
            ),
            display_disconnect: sym!("wl_display_disconnect", unsafe extern "C" fn(*mut c_void)),
            display_roundtrip: sym!(
                "wl_display_roundtrip",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_dispatch_pending: sym!(
                "wl_display_dispatch_pending",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_flush: sym!(
                "wl_display_flush",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_get_fd: sym!(
                "wl_display_get_fd",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_prepare_read: sym!(
                "wl_display_prepare_read",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_read_events: sym!(
                "wl_display_read_events",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            display_cancel_read: sym!("wl_display_cancel_read", unsafe extern "C" fn(*mut c_void)),
            proxy_marshal_array_flags: sym!(
                "wl_proxy_marshal_array_flags",
                unsafe extern "C" fn(
                    *mut c_void,
                    u32,
                    *const WlInterface,
                    u32,
                    u32,
                    *mut WlArgument,
                ) -> *mut c_void
            ),
            proxy_add_listener: sym!(
                "wl_proxy_add_listener",
                unsafe extern "C" fn(*mut c_void, *const *const c_void, *mut c_void) -> c_int
            ),
            proxy_destroy: sym!("wl_proxy_destroy", unsafe extern "C" fn(*mut c_void)),
            proxy_get_version: sym!(
                "wl_proxy_get_version",
                unsafe extern "C" fn(*mut c_void) -> u32
            ),
            wl_registry_interface: data_sym!("wl_registry_interface"),
            wl_compositor_interface: data_sym!("wl_compositor_interface"),
            wl_surface_interface: data_sym!("wl_surface_interface"),
        })
    }
}

impl Drop for WaylandLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Closed once after all Wayland proxies and display have been destroyed.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

struct WaylandState {
    marshal_fn: unsafe extern "C" fn(
        *mut c_void,
        u32,
        *const WlInterface,
        u32,
        u32,
        *mut WlArgument,
    ) -> *mut c_void,
    get_version_fn: unsafe extern "C" fn(*mut c_void) -> u32,
    wl_compositor_interface: *const WlInterface,
    compositor: *mut c_void,
    xdg_wm_base: *mut c_void,
    configured: AtomicBool,
    closed: AtomicBool,
    resized: AtomicBool,
    staged_width: AtomicU32,
    staged_height: AtomicU32,
    width: AtomicU32,
    height: AtomicU32,
}

unsafe extern "C" fn wl_on_registry_global(
    data: *mut c_void,
    registry: *mut c_void,
    name: u32,
    interface: *const c_char,
    version: u32,
) {
    if data.is_null() || registry.is_null() || interface.is_null() {
        return;
    }
    // SAFETY: `data` is the pinned `Box<WaylandState>` alive for the full Wayland window lifetime.
    let st = unsafe { &mut *(data as *mut WaylandState) };
    // SAFETY: `interface` is a NUL-terminated ASCII interface name from libwayland.
    let iface_name = unsafe { CStr::from_ptr(interface) };
    if iface_name == c"wl_compositor" && st.compositor.is_null() {
        let bind_ver = version.min(4);
        let mut args = [
            WlArgument { u: name },
            WlArgument {
                s: c"wl_compositor".as_ptr(),
            },
            WlArgument { u: bind_ver },
            WlArgument { o: ptr::null_mut() },
        ];
        // SAFETY: `wl_registry.bind` (opcode 0) writes the new proxy into `args[3]`.
        st.compositor = unsafe {
            (st.marshal_fn)(
                registry,
                0,
                st.wl_compositor_interface,
                bind_ver,
                0,
                args.as_mut_ptr(),
            )
        };
    } else if iface_name == c"xdg_wm_base" && st.xdg_wm_base.is_null() {
        let bind_ver = version.min(1);
        let mut args = [
            WlArgument { u: name },
            WlArgument {
                s: c"xdg_wm_base".as_ptr(),
            },
            WlArgument { u: bind_ver },
            WlArgument { o: ptr::null_mut() },
        ];
        // SAFETY: `wl_registry.bind` (opcode 0) binds `xdg_wm_base` v1.
        st.xdg_wm_base = unsafe {
            (st.marshal_fn)(
                registry,
                0,
                &XDG_WM_BASE_INTERFACE,
                bind_ver,
                0,
                args.as_mut_ptr(),
            )
        };
    }
}

unsafe extern "C" fn wl_on_registry_remove(_data: *mut c_void, _registry: *mut c_void, _name: u32) {
}

unsafe extern "C" fn xdg_on_wm_base_ping(data: *mut c_void, wm_base: *mut c_void, serial: u32) {
    if data.is_null() || wm_base.is_null() {
        return;
    }
    // SAFETY: `data` is the live `WaylandState`; `xdg_wm_base.pong` is opcode 3.
    unsafe {
        let st = &*(data as *const WaylandState);
        let ver = (st.get_version_fn)(wm_base);
        let mut args = [WlArgument { u: serial }];
        let _ = (st.marshal_fn)(wm_base, 3, ptr::null(), ver, 0, args.as_mut_ptr());
    }
}

unsafe extern "C" fn xdg_on_surface_configure(
    data: *mut c_void,
    xdg_surface: *mut c_void,
    serial: u32,
) {
    if data.is_null() || xdg_surface.is_null() {
        return;
    }
    // SAFETY: `data` is the live `WaylandState`; `xdg_surface.ack_configure` is opcode 4.
    unsafe {
        let st = &*(data as *const WaylandState);
        let sw = st.staged_width.load(Ordering::Relaxed);
        let sh = st.staged_height.load(Ordering::Relaxed);
        if sw > 0 && sh > 0 {
            let prev_w = st.width.swap(sw, Ordering::Relaxed);
            let prev_h = st.height.swap(sh, Ordering::Relaxed);
            if prev_w != sw || prev_h != sh {
                st.resized.store(true, Ordering::Release);
            }
        }
        let ver = (st.get_version_fn)(xdg_surface);
        let mut args = [WlArgument { u: serial }];
        let _ = (st.marshal_fn)(xdg_surface, 4, ptr::null(), ver, 0, args.as_mut_ptr());
        st.configured.store(true, Ordering::Release);
    }
}

unsafe extern "C" fn xdg_on_toplevel_configure(
    data: *mut c_void,
    _toplevel: *mut c_void,
    width: i32,
    height: i32,
    _states: *mut c_void,
) {
    if data.is_null() {
        return;
    }
    // SAFETY: `data` is the live `WaylandState`.
    let st = unsafe { &*(data as *const WaylandState) };
    if width > 0 && height > 0 {
        st.staged_width.store(width as u32, Ordering::Relaxed);
        st.staged_height.store(height as u32, Ordering::Relaxed);
    }
}

unsafe extern "C" fn xdg_on_toplevel_close(data: *mut c_void, _toplevel: *mut c_void) {
    if !data.is_null() {
        // SAFETY: `data` is the live `WaylandState`.
        let st = unsafe { &*(data as *const WaylandState) };
        st.closed.store(true, Ordering::Release);
    }
}

static WL_REGISTRY_LISTENER: WlRegistryListener = WlRegistryListener {
    global: wl_on_registry_global,
    global_remove: wl_on_registry_remove,
};
static XDG_WM_BASE_LISTENER: XdgWmBaseListener = XdgWmBaseListener {
    ping: xdg_on_wm_base_ping,
};
static XDG_SURFACE_LISTENER: XdgSurfaceListener = XdgSurfaceListener {
    configure: xdg_on_surface_configure,
};
static XDG_TOPLEVEL_LISTENER: XdgToplevelListener = XdgToplevelListener {
    configure: xdg_on_toplevel_configure,
    close: xdg_on_toplevel_close,
};

pub(crate) struct WaylandWindow {
    lib: WaylandLib,
    display: *mut c_void,
    registry: *mut c_void,
    compositor: *mut c_void,
    xdg_wm_base: *mut c_void,
    wl_surface: *mut c_void,
    xdg_surface: *mut c_void,
    xdg_toplevel: *mut c_void,
    state: *mut WaylandState,
}

impl WaylandWindow {
    fn open(title: &str, width: u32, height: u32) -> Result<Self, GpuError> {
        if std::env::var("WAYLAND_DISPLAY")
            .ok()
            .filter(|v| !v.is_empty())
            .is_none()
        {
            return Err(GpuError::Vulkan("WAYLAND_DISPLAY is unset".into()));
        }
        let lib = WaylandLib::load()?;
        // SAFETY: Connects to default `WAYLAND_DISPLAY`.
        let display = unsafe { (lib.display_connect)(ptr::null()) };
        if display.is_null() {
            return Err(GpuError::Vulkan("wl_display_connect returned null".into()));
        }

        let state = Box::into_raw(Box::new(WaylandState {
            marshal_fn: lib.proxy_marshal_array_flags,
            get_version_fn: lib.proxy_get_version,
            wl_compositor_interface: lib.wl_compositor_interface,
            compositor: ptr::null_mut(),
            xdg_wm_base: ptr::null_mut(),
            configured: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            resized: AtomicBool::new(false),
            staged_width: AtomicU32::new(width),
            staged_height: AtomicU32::new(height),
            width: AtomicU32::new(width),
            height: AtomicU32::new(height),
        }));

        let mut win = Self {
            lib,
            display,
            registry: ptr::null_mut(),
            compositor: ptr::null_mut(),
            xdg_wm_base: ptr::null_mut(),
            wl_surface: ptr::null_mut(),
            xdg_surface: ptr::null_mut(),
            xdg_toplevel: ptr::null_mut(),
            state,
        };

        // SAFETY: All Wayland proxies are created and listened on this worker thread and torn
        // down cleanly in `Drop` even on early error return.
        unsafe {
            // wl_display.get_registry (opcode 1)
            let mut reg_args = [WlArgument { o: ptr::null_mut() }];
            win.registry = (win.lib.proxy_marshal_array_flags)(
                win.display,
                1,
                win.lib.wl_registry_interface,
                1,
                0,
                reg_args.as_mut_ptr(),
            );
            if win.registry.is_null() {
                return Err(GpuError::Vulkan("wl_display.get_registry failed".into()));
            }
            (win.lib.proxy_add_listener)(
                win.registry,
                (&WL_REGISTRY_LISTENER as *const WlRegistryListener).cast(),
                win.state.cast(),
            );
            if (win.lib.display_roundtrip)(win.display) < 0 {
                return Err(GpuError::Vulkan(
                    "wl_display_roundtrip(registry) failed".into(),
                ));
            }

            win.compositor = (*win.state).compositor;
            win.xdg_wm_base = (*win.state).xdg_wm_base;
            if win.compositor.is_null() || win.xdg_wm_base.is_null() {
                return Err(GpuError::Vulkan(
                    "Wayland compositor missing wl_compositor or xdg_wm_base".into(),
                ));
            }

            (win.lib.proxy_add_listener)(
                win.xdg_wm_base,
                (&XDG_WM_BASE_LISTENER as *const XdgWmBaseListener).cast(),
                win.state.cast(),
            );

            // wl_compositor.create_surface (opcode 0)
            let comp_ver = (win.lib.proxy_get_version)(win.compositor);
            let mut surf_args = [WlArgument { o: ptr::null_mut() }];
            win.wl_surface = (win.lib.proxy_marshal_array_flags)(
                win.compositor,
                0,
                win.lib.wl_surface_interface,
                comp_ver,
                0,
                surf_args.as_mut_ptr(),
            );
            if win.wl_surface.is_null() {
                return Err(GpuError::Vulkan(
                    "wl_compositor.create_surface failed".into(),
                ));
            }

            // xdg_wm_base.get_xdg_surface (opcode 2)
            let wm_ver = (win.lib.proxy_get_version)(win.xdg_wm_base);
            let mut xsurf_args = [
                WlArgument { o: ptr::null_mut() },
                WlArgument { o: win.wl_surface },
            ];
            win.xdg_surface = (win.lib.proxy_marshal_array_flags)(
                win.xdg_wm_base,
                2,
                &XDG_SURFACE_INTERFACE,
                wm_ver,
                0,
                xsurf_args.as_mut_ptr(),
            );
            if win.xdg_surface.is_null() {
                return Err(GpuError::Vulkan(
                    "xdg_wm_base.get_xdg_surface failed".into(),
                ));
            }
            (win.lib.proxy_add_listener)(
                win.xdg_surface,
                (&XDG_SURFACE_LISTENER as *const XdgSurfaceListener).cast(),
                win.state.cast(),
            );

            // xdg_surface.get_toplevel (opcode 1)
            let xs_ver = (win.lib.proxy_get_version)(win.xdg_surface);
            let mut top_args = [WlArgument { o: ptr::null_mut() }];
            win.xdg_toplevel = (win.lib.proxy_marshal_array_flags)(
                win.xdg_surface,
                1,
                &XDG_TOPLEVEL_INTERFACE,
                xs_ver,
                0,
                top_args.as_mut_ptr(),
            );
            if win.xdg_toplevel.is_null() {
                return Err(GpuError::Vulkan("xdg_surface.get_toplevel failed".into()));
            }
            (win.lib.proxy_add_listener)(
                win.xdg_toplevel,
                (&XDG_TOPLEVEL_LISTENER as *const XdgToplevelListener).cast(),
                win.state.cast(),
            );

            // xdg_toplevel.set_title (opcode 2) + set_app_id (opcode 3)
            let top_ver = (win.lib.proxy_get_version)(win.xdg_toplevel);
            let c_title = CString::new(title).unwrap_or_else(|_| CString::new("X2Linux").unwrap());
            let mut title_args = [WlArgument {
                s: c_title.as_ptr(),
            }];
            let _ = (win.lib.proxy_marshal_array_flags)(
                win.xdg_toplevel,
                2,
                ptr::null(),
                top_ver,
                0,
                title_args.as_mut_ptr(),
            );
            let mut app_args = [WlArgument {
                s: c"x2linux".as_ptr(),
            }];
            let _ = (win.lib.proxy_marshal_array_flags)(
                win.xdg_toplevel,
                3,
                ptr::null(),
                top_ver,
                0,
                app_args.as_mut_ptr(),
            );

            // Initial bufferless wl_surface.commit (opcode 6) so compositor sends xdg_surface.configure
            let surf_ver = (win.lib.proxy_get_version)(win.wl_surface);
            let mut empty_args: [WlArgument; 0] = [];
            let _ = (win.lib.proxy_marshal_array_flags)(
                win.wl_surface,
                6,
                ptr::null(),
                surf_ver,
                0,
                empty_args.as_mut_ptr(),
            );
            let _ = (win.lib.display_flush)(win.display);

            // Wait (bounded) for initial xdg_surface.configure + ack_configure flush
            let deadline = Instant::now() + Duration::from_millis(500);
            while !(*win.state).configured.load(Ordering::Acquire) && Instant::now() < deadline {
                if (win.lib.display_roundtrip)(win.display) < 0 {
                    break;
                }
            }
            let _ = (win.lib.display_flush)(win.display);
        }

        Ok(win)
    }

    fn pump_events(&mut self) {
        // SAFETY: Standard non-blocking Wayland read/dispatch cycle on owning worker thread.
        unsafe {
            let _ = (self.lib.display_dispatch_pending)(self.display);
            let _ = (self.lib.display_flush)(self.display);
            if (self.lib.display_prepare_read)(self.display) == 0 {
                let fd = (self.lib.display_get_fd)(self.display);
                let mut pfd = PollFd {
                    fd,
                    events: POLLIN,
                    revents: 0,
                };
                if poll(&mut pfd, 1, 0) > 0 && (pfd.revents & POLLIN) != 0 {
                    let _ = (self.lib.display_read_events)(self.display);
                } else {
                    (self.lib.display_cancel_read)(self.display);
                }
            }
            let _ = (self.lib.display_dispatch_pending)(self.display);
            let _ = (self.lib.display_flush)(self.display);
        }
    }

    fn extent(&self) -> vk::Extent2D {
        // SAFETY: `self.state` is owned and live until `Drop`.
        unsafe {
            vk::Extent2D {
                width: (*self.state).width.load(Ordering::Relaxed).max(1),
                height: (*self.state).height.load(Ordering::Relaxed).max(1),
            }
        }
    }

    fn is_closed(&self) -> bool {
        // SAFETY: `self.state` is live.
        unsafe { (*self.state).closed.load(Ordering::Acquire) }
    }

    fn take_resized(&self) -> bool {
        // SAFETY: `self.state` is live.
        unsafe { (*self.state).resized.swap(false, Ordering::AcqRel) }
    }
}

impl Drop for WaylandWindow {
    fn drop(&mut self) {
        // SAFETY: Strict Wayland protocol teardown order: xdg_toplevel -> xdg_surface ->
        // wl_surface -> xdg_wm_base -> wl_compositor -> wl_registry -> display -> state.
        unsafe {
            let mut empty: [WlArgument; 0] = [];
            if !self.xdg_toplevel.is_null() {
                let v = (self.lib.proxy_get_version)(self.xdg_toplevel);
                let _ = (self.lib.proxy_marshal_array_flags)(
                    self.xdg_toplevel,
                    0,
                    ptr::null(),
                    v,
                    WL_MARSHAL_FLAG_DESTROY,
                    empty.as_mut_ptr(),
                );
            }
            if !self.xdg_surface.is_null() {
                let v = (self.lib.proxy_get_version)(self.xdg_surface);
                let _ = (self.lib.proxy_marshal_array_flags)(
                    self.xdg_surface,
                    0,
                    ptr::null(),
                    v,
                    WL_MARSHAL_FLAG_DESTROY,
                    empty.as_mut_ptr(),
                );
            }
            if !self.wl_surface.is_null() {
                let v = (self.lib.proxy_get_version)(self.wl_surface);
                let _ = (self.lib.proxy_marshal_array_flags)(
                    self.wl_surface,
                    0,
                    ptr::null(),
                    v,
                    WL_MARSHAL_FLAG_DESTROY,
                    empty.as_mut_ptr(),
                );
            }
            if !self.xdg_wm_base.is_null() {
                let v = (self.lib.proxy_get_version)(self.xdg_wm_base);
                let _ = (self.lib.proxy_marshal_array_flags)(
                    self.xdg_wm_base,
                    0,
                    ptr::null(),
                    v,
                    WL_MARSHAL_FLAG_DESTROY,
                    empty.as_mut_ptr(),
                );
            }
            if !self.compositor.is_null() {
                (self.lib.proxy_destroy)(self.compositor);
            }
            if !self.registry.is_null() {
                (self.lib.proxy_destroy)(self.registry);
            }
            if !self.display.is_null() {
                let _ = (self.lib.display_flush)(self.display);
                (self.lib.display_disconnect)(self.display);
            }
            if !self.state.is_null() {
                drop(Box::from_raw(self.state));
            }
        }
    }
}

// ============================================================================
// 2. XCB X11 (`libxcb.so.1`) runtime `dlopen` fallback
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy)]
struct XcbVoidCookie {
    sequence: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct XcbInternAtomCookie {
    sequence: c_uint,
}

#[repr(C)]
struct XcbInternAtomReply {
    response_type: u8,
    pad0: u8,
    sequence: u16,
    length: u32,
    atom: u32,
}

#[repr(C)]
struct XcbScreen {
    root: u32,
    default_colormap: u32,
    white_pixel: u32,
    black_pixel: u32,
    current_input_masks: u32,
    width_in_pixels: u16,
    height_in_pixels: u16,
    width_in_millimeters: u16,
    height_in_millimeters: u16,
    min_installed_maps: u16,
    max_installed_maps: u16,
    root_visual: u32,
    backing_stores: u8,
    save_unders: u8,
    root_depth: u8,
    allowed_depths_len: u8,
}

#[repr(C)]
struct XcbScreenIterator {
    data: *mut XcbScreen,
    rem: c_int,
    index: c_int,
}

#[repr(C)]
struct XcbGenericEvent {
    response_type: u8,
    pad0: u8,
    sequence: u16,
    pad: [u32; 7],
    full_sequence: u32,
}

#[repr(C)]
struct XcbConfigureNotifyEvent {
    response_type: u8,
    pad0: u8,
    sequence: u16,
    event: u32,
    window: u32,
    above_sibling: u32,
    x: i16,
    y: i16,
    width: u16,
    height: u16,
    border_width: u16,
    override_redirect: u8,
    pad1: u8,
}

#[repr(C)]
struct XcbClientMessageEvent {
    response_type: u8,
    format: u8,
    sequence: u16,
    window: u32,
    r#type: u32,
    data32: [u32; 5],
}

const XCB_COPY_FROM_PARENT: u8 = 0;
const XCB_WINDOW_CLASS_INPUT_OUTPUT: u16 = 1;
const XCB_CW_EVENT_MASK: u32 = 1 << 11;
const XCB_EVENT_MASK_STRUCTURE_NOTIFY: u32 = 1 << 17;
const XCB_PROP_MODE_REPLACE: u8 = 0;
const XCB_ATOM_ATOM: u32 = 4;
const XCB_ATOM_STRING: u32 = 31;
const XCB_ATOM_WM_NAME: u32 = 39;

struct XcbLib {
    handle: *mut c_void,
    connect: unsafe extern "C" fn(*const c_char, *mut c_int) -> *mut c_void,
    connection_has_error: unsafe extern "C" fn(*mut c_void) -> c_int,
    disconnect: unsafe extern "C" fn(*mut c_void),
    get_setup: unsafe extern "C" fn(*mut c_void) -> *const c_void,
    setup_roots_iterator: unsafe extern "C" fn(*const c_void) -> XcbScreenIterator,
    screen_next: unsafe extern "C" fn(*mut XcbScreenIterator),
    generate_id: unsafe extern "C" fn(*mut c_void) -> u32,
    create_window_checked: unsafe extern "C" fn(
        *mut c_void,
        u8,
        u32,
        u32,
        i16,
        i16,
        u16,
        u16,
        u16,
        u16,
        u32,
        u32,
        *const c_void,
    ) -> XcbVoidCookie,
    map_window_checked: unsafe extern "C" fn(*mut c_void, u32) -> XcbVoidCookie,
    change_property_checked: unsafe extern "C" fn(
        *mut c_void,
        u8,
        u32,
        u32,
        u32,
        u8,
        u32,
        *const c_void,
    ) -> XcbVoidCookie,
    intern_atom: unsafe extern "C" fn(*mut c_void, u8, u16, *const c_char) -> XcbInternAtomCookie,
    intern_atom_reply: unsafe extern "C" fn(
        *mut c_void,
        XcbInternAtomCookie,
        *mut *mut c_void,
    ) -> *mut XcbInternAtomReply,
    request_check: unsafe extern "C" fn(*mut c_void, XcbVoidCookie) -> *mut c_void,
    flush: unsafe extern "C" fn(*mut c_void) -> c_int,
    poll_for_event: unsafe extern "C" fn(*mut c_void) -> *mut XcbGenericEvent,
    destroy_window: unsafe extern "C" fn(*mut c_void, u32) -> XcbVoidCookie,
}

impl XcbLib {
    fn load() -> Result<Self, GpuError> {
        // SAFETY: Standard POSIX `dlopen` on immutable soname literal.
        let handle = unsafe { dlopen(c"libxcb.so.1".as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if handle.is_null() {
            return Err(GpuError::Vulkan("libxcb.so.1 not available".into()));
        }
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let cname = CString::new($name).expect("static ASCII");
                // SAFETY: `handle` is live and `cname` is NUL-terminated.
                let p = unsafe { dlsym(handle, cname.as_ptr()) };
                if p.is_null() {
                    // SAFETY: Clean up handle on missing symbol.
                    unsafe { dlclose(handle) };
                    return Err(GpuError::Vulkan(format!("missing XCB symbol {}", $name)));
                }
                // SAFETY: Symbol ABI verified against libxcb 1.15+.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }

        Ok(Self {
            handle,
            connect: sym!(
                "xcb_connect",
                unsafe extern "C" fn(*const c_char, *mut c_int) -> *mut c_void
            ),
            connection_has_error: sym!(
                "xcb_connection_has_error",
                unsafe extern "C" fn(*mut c_void) -> c_int
            ),
            disconnect: sym!("xcb_disconnect", unsafe extern "C" fn(*mut c_void)),
            get_setup: sym!(
                "xcb_get_setup",
                unsafe extern "C" fn(*mut c_void) -> *const c_void
            ),
            setup_roots_iterator: sym!(
                "xcb_setup_roots_iterator",
                unsafe extern "C" fn(*const c_void) -> XcbScreenIterator
            ),
            screen_next: sym!(
                "xcb_screen_next",
                unsafe extern "C" fn(*mut XcbScreenIterator)
            ),
            generate_id: sym!("xcb_generate_id", unsafe extern "C" fn(*mut c_void) -> u32),
            create_window_checked: sym!(
                "xcb_create_window_checked",
                unsafe extern "C" fn(
                    *mut c_void,
                    u8,
                    u32,
                    u32,
                    i16,
                    i16,
                    u16,
                    u16,
                    u16,
                    u16,
                    u32,
                    u32,
                    *const c_void,
                ) -> XcbVoidCookie
            ),
            map_window_checked: sym!(
                "xcb_map_window_checked",
                unsafe extern "C" fn(*mut c_void, u32) -> XcbVoidCookie
            ),
            change_property_checked: sym!(
                "xcb_change_property_checked",
                unsafe extern "C" fn(
                    *mut c_void,
                    u8,
                    u32,
                    u32,
                    u32,
                    u8,
                    u32,
                    *const c_void,
                ) -> XcbVoidCookie
            ),
            intern_atom: sym!(
                "xcb_intern_atom",
                unsafe extern "C" fn(*mut c_void, u8, u16, *const c_char) -> XcbInternAtomCookie
            ),
            intern_atom_reply: sym!(
                "xcb_intern_atom_reply",
                unsafe extern "C" fn(
                    *mut c_void,
                    XcbInternAtomCookie,
                    *mut *mut c_void,
                ) -> *mut XcbInternAtomReply
            ),
            request_check: sym!(
                "xcb_request_check",
                unsafe extern "C" fn(*mut c_void, XcbVoidCookie) -> *mut c_void
            ),
            flush: sym!("xcb_flush", unsafe extern "C" fn(*mut c_void) -> c_int),
            poll_for_event: sym!(
                "xcb_poll_for_event",
                unsafe extern "C" fn(*mut c_void) -> *mut XcbGenericEvent
            ),
            destroy_window: sym!(
                "xcb_destroy_window",
                unsafe extern "C" fn(*mut c_void, u32) -> XcbVoidCookie
            ),
        })
    }
}

impl Drop for XcbLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Closed once after XCB connection is disconnected.
            unsafe {
                dlclose(self.handle);
            }
        }
    }
}

pub(crate) struct XcbWindow {
    lib: XcbLib,
    conn: *mut c_void,
    window: u32,
    wm_protocols: u32,
    wm_delete_window: u32,
    width: u32,
    height: u32,
    closed: bool,
    resized: bool,
}

impl XcbWindow {
    fn open(title: &str, width: u32, height: u32) -> Result<Self, GpuError> {
        if std::env::var("DISPLAY")
            .ok()
            .filter(|v| !v.is_empty())
            .is_none()
        {
            return Err(GpuError::Vulkan("DISPLAY is unset".into()));
        }
        let lib = XcbLib::load()?;
        let mut screen_num = 0;
        // SAFETY: Connects to default `DISPLAY`.
        let conn = unsafe { (lib.connect)(ptr::null(), &mut screen_num) };
        if conn.is_null() {
            return Err(GpuError::Vulkan("xcb_connect returned null".into()));
        }
        // SAFETY: `conn` is non-null.
        if unsafe { (lib.connection_has_error)(conn) } != 0 {
            // SAFETY: Disconnect failed connection.
            unsafe { (lib.disconnect)(conn) };
            return Err(GpuError::Vulkan("xcb_connect failed".into()));
        }

        let mut win = Self {
            lib,
            conn,
            window: 0,
            wm_protocols: 0,
            wm_delete_window: 0,
            width,
            height,
            closed: false,
            resized: false,
        };

        // SAFETY: Standard XCB window creation; any returned errors/replies are freed via `free`.
        unsafe {
            let setup = (win.lib.get_setup)(win.conn);
            if setup.is_null() {
                return Err(GpuError::Vulkan("xcb_get_setup returned null".into()));
            }
            let mut iter = (win.lib.setup_roots_iterator)(setup);
            for _ in 0..screen_num.max(0) {
                (win.lib.screen_next)(&mut iter);
            }
            if iter.data.is_null() {
                return Err(GpuError::Vulkan("XCB screen iterator empty".into()));
            }
            let screen = &*iter.data;
            let wid = (win.lib.generate_id)(win.conn);
            let event_mask: [u32; 1] = [XCB_EVENT_MASK_STRUCTURE_NOTIFY];
            let cookie = (win.lib.create_window_checked)(
                win.conn,
                XCB_COPY_FROM_PARENT,
                wid,
                screen.root,
                0,
                0,
                width.clamp(1, u16::MAX as u32) as u16,
                height.clamp(1, u16::MAX as u32) as u16,
                0,
                XCB_WINDOW_CLASS_INPUT_OUTPUT,
                screen.root_visual,
                XCB_CW_EVENT_MASK,
                event_mask.as_ptr().cast(),
            );
            let err = (win.lib.request_check)(win.conn, cookie);
            if !err.is_null() {
                free(err);
                return Err(GpuError::Vulkan("xcb_create_window_checked failed".into()));
            }
            win.window = wid;

            // Set WM_NAME title
            let title_bytes = title.as_bytes();
            let title_cookie = (win.lib.change_property_checked)(
                win.conn,
                XCB_PROP_MODE_REPLACE,
                wid,
                XCB_ATOM_WM_NAME,
                XCB_ATOM_STRING,
                8,
                title_bytes.len() as u32,
                title_bytes.as_ptr().cast(),
            );
            let err = (win.lib.request_check)(win.conn, title_cookie);
            if !err.is_null() {
                free(err);
            }

            // Register WM_DELETE_WINDOW protocol
            win.wm_protocols = intern_xcb_atom(&win.lib, win.conn, b"WM_PROTOCOLS");
            win.wm_delete_window = intern_xcb_atom(&win.lib, win.conn, b"WM_DELETE_WINDOW");
            if win.wm_protocols != 0 && win.wm_delete_window != 0 {
                let prop_cookie = (win.lib.change_property_checked)(
                    win.conn,
                    XCB_PROP_MODE_REPLACE,
                    wid,
                    win.wm_protocols,
                    XCB_ATOM_ATOM,
                    32,
                    1,
                    (&win.wm_delete_window as *const u32).cast(),
                );
                let err = (win.lib.request_check)(win.conn, prop_cookie);
                if !err.is_null() {
                    free(err);
                }
            }

            let map_cookie = (win.lib.map_window_checked)(win.conn, wid);
            let err = (win.lib.request_check)(win.conn, map_cookie);
            if !err.is_null() {
                free(err);
                return Err(GpuError::Vulkan("xcb_map_window_checked failed".into()));
            }
            let _ = (win.lib.flush)(win.conn);
        }

        Ok(win)
    }

    fn pump_events(&mut self) {
        // SAFETY: Drains all queued non-blocking XCB events and frees each event pointer.
        unsafe {
            loop {
                let ev = (self.lib.poll_for_event)(self.conn);
                if ev.is_null() {
                    break;
                }
                let code = (*ev).response_type & !0x80;
                match code {
                    17 => {
                        // XCB_DESTROY_NOTIFY
                        self.closed = true;
                    }
                    22 => {
                        // XCB_CONFIGURE_NOTIFY
                        let cfg = &*(ev as *const XcbConfigureNotifyEvent);
                        let w = u32::from(cfg.width).max(1);
                        let h = u32::from(cfg.height).max(1);
                        if w != self.width || h != self.height {
                            self.width = w;
                            self.height = h;
                            self.resized = true;
                        }
                    }
                    33 => {
                        // XCB_CLIENT_MESSAGE
                        let cm = &*(ev as *const XcbClientMessageEvent);
                        if cm.format == 32
                            && cm.r#type == self.wm_protocols
                            && cm.data32[0] == self.wm_delete_window
                        {
                            self.closed = true;
                        }
                    }
                    _ => {}
                }
                free(ev.cast());
            }
        }
    }

    fn extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.width.max(1),
            height: self.height.max(1),
        }
    }

    fn is_closed(&self) -> bool {
        self.closed
    }

    fn take_resized(&mut self) -> bool {
        std::mem::replace(&mut self.resized, false)
    }
}

impl Drop for XcbWindow {
    fn drop(&mut self) {
        // SAFETY: Destroy window before disconnecting XCB connection.
        unsafe {
            if self.window != 0 && !self.conn.is_null() {
                let _ = (self.lib.destroy_window)(self.conn, self.window);
                let _ = (self.lib.flush)(self.conn);
            }
            if !self.conn.is_null() {
                (self.lib.disconnect)(self.conn);
            }
        }
    }
}

unsafe fn intern_xcb_atom(lib: &XcbLib, conn: *mut c_void, name: &[u8]) -> u32 {
    // SAFETY: Caller passes live XCB connection and valid ASCII atom name slice.
    unsafe {
        let cookie = (lib.intern_atom)(conn, 0, name.len() as u16, name.as_ptr().cast());
        let mut err: *mut c_void = ptr::null_mut();
        let reply = (lib.intern_atom_reply)(conn, cookie, &mut err);
        if !err.is_null() {
            free(err);
        }
        if reply.is_null() {
            return 0;
        }
        let atom = (*reply).atom;
        free(reply.cast());
        atom
    }
}

// ============================================================================
// 3. Unified NativeWindow & Vulkan SwapchainPresenter
// ============================================================================

/// Native Linux window handle created on the SUD-free worker thread.
pub(crate) enum NativeWindow {
    Wayland(WaylandWindow),
    Xcb(XcbWindow),
}

// SAFETY: `NativeWindow` is created, pumped, and dropped strictly on the serialized native GPU
// worker thread (protected by `Mutex` inside `VulkanBackend`).
unsafe impl Send for NativeWindow {}

impl NativeWindow {
    /// Open a native window matching the requested presentation mode.
    pub fn open_for_mode(
        mode: x2l_common::PresentMode,
        title: &str,
    ) -> Result<Option<Self>, GpuError> {
        match mode {
            x2l_common::PresentMode::Headless => Ok(None),
            x2l_common::PresentMode::Wayland => Ok(Some(Self::Wayland(WaylandWindow::open(
                title,
                DEFAULT_WINDOW_WIDTH,
                DEFAULT_WINDOW_HEIGHT,
            )?))),
            x2l_common::PresentMode::X11 => Ok(Some(Self::Xcb(XcbWindow::open(
                title,
                DEFAULT_WINDOW_WIDTH,
                DEFAULT_WINDOW_HEIGHT,
            )?))),
            x2l_common::PresentMode::Window | x2l_common::PresentMode::Auto => {
                match WaylandWindow::open(title, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) {
                    Ok(w) => Ok(Some(Self::Wayland(w))),
                    Err(wl_err) => {
                        debug!(error = %wl_err, "Wayland WSI unavailable, trying XCB fallback");
                        match XcbWindow::open(title, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) {
                            Ok(x) => Ok(Some(Self::Xcb(x))),
                            Err(xcb_err) => {
                                warn!(
                                    wayland_err = %wl_err,
                                    xcb_err = %xcb_err,
                                    "no Wayland/X11 display available; falling back to headless presentation"
                                );
                                Ok(None)
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn kind(&self) -> PresentSurfaceKind {
        match self {
            Self::Wayland(_) => PresentSurfaceKind::Wayland,
            Self::Xcb(_) => PresentSurfaceKind::X11,
        }
    }

    pub fn required_instance_extension(&self) -> &'static CStr {
        match self {
            Self::Wayland(_) => ash::khr::wayland_surface::NAME,
            Self::Xcb(_) => ash::khr::xcb_surface::NAME,
        }
    }

    /// Create a `VkSurfaceKHR` bound to this native window.
    ///
    /// # Safety
    /// `entry` and `instance` must be live and have `VK_KHR_surface` + this window's platform
    /// surface extension enabled.
    pub unsafe fn create_vulkan_surface(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<vk::SurfaceKHR, GpuError> {
        match self {
            Self::Wayland(wl) => {
                let loader = ash::khr::wayland_surface::Instance::new(entry, instance);
                let ci = vk::WaylandSurfaceCreateInfoKHR::default()
                    .display(wl.display)
                    .surface(wl.wl_surface);
                // SAFETY: `wl.display` and `wl.wl_surface` are live Wayland pointers.
                unsafe { loader.create_wayland_surface(&ci, None) }.map_err(|e| {
                    GpuError::Vulkan(format!("vkCreateWaylandSurfaceKHR failed: {e:?}"))
                })
            }
            Self::Xcb(xcb) => {
                let loader = ash::khr::xcb_surface::Instance::new(entry, instance);
                let ci = vk::XcbSurfaceCreateInfoKHR::default()
                    .connection(xcb.conn)
                    .window(xcb.window);
                // SAFETY: `xcb.conn` and `xcb.window` are live XCB handles.
                unsafe { loader.create_xcb_surface(&ci, None) }
                    .map_err(|e| GpuError::Vulkan(format!("vkCreateXcbSurfaceKHR failed: {e:?}")))
            }
        }
    }

    pub fn pump_events(&mut self) {
        match self {
            Self::Wayland(w) => w.pump_events(),
            Self::Xcb(x) => x.pump_events(),
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        match self {
            Self::Wayland(w) => w.extent(),
            Self::Xcb(x) => x.extent(),
        }
    }

    pub fn is_closed(&self) -> bool {
        match self {
            Self::Wayland(w) => w.is_closed(),
            Self::Xcb(x) => x.is_closed(),
        }
    }

    pub fn take_resized(&mut self) -> bool {
        match self {
            Self::Wayland(w) => w.take_resized(),
            Self::Xcb(x) => x.take_resized(),
        }
    }
}

struct SwapchainGeneration {
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    render_finished: Vec<vk::Semaphore>,
    present_fences: Vec<vk::Fence>,
    present_fence_armed: Vec<bool>,
    extent: vk::Extent2D,
}

impl SwapchainGeneration {
    unsafe fn destroy(
        &mut self,
        device: &ash::Device,
        swapchain_loader: &ash::khr::swapchain::Device,
    ) {
        // SAFETY: Caller guarantees device is idle or all presents using this generation completed.
        unsafe {
            for &f in &self.present_fences {
                device.destroy_fence(f, None);
            }
            self.present_fences.clear();
            self.present_fence_armed.clear();
            for &sem in &self.render_finished {
                device.destroy_semaphore(sem, None);
            }
            self.render_finished.clear();
            if self.swapchain != vk::SwapchainKHR::null() {
                swapchain_loader.destroy_swapchain(self.swapchain, None);
                self.swapchain = vk::SwapchainKHR::null();
            }
        }
    }
}

/// Live Vulkan WSI swapchain presenter owned by [`crate::vulkan::VulkanBackend`].
pub(crate) struct SwapchainPresenter {
    surface_kind: PresentSurfaceKind,
    surface_loader: ash::khr::surface::Instance,
    swapchain_loader: ash::khr::swapchain::Device,
    surface: vk::SurfaceKHR,
    window: Option<NativeWindow>,
    physical_device: vk::PhysicalDevice,
    cmd_pool: vk::CommandPool,
    cmd_buf: vk::CommandBuffer,
    image_available: vk::Semaphore,
    submit_fence: vk::Fence,
    use_present_fences: bool,
    active: Option<SwapchainGeneration>,
    retired: Vec<SwapchainGeneration>,
    needs_recreate: bool,
    recreations: u64,
    shutdown_done: bool,
}

// SAFETY: `SwapchainPresenter` is serialized under `VulkanBackend`'s `Mutex` and executed on the
// SUD-free native GPU worker thread.
unsafe impl Send for SwapchainPresenter {}

impl SwapchainPresenter {
    /// Construct a new swapchain presenter bound to an already-created `NativeWindow` and
    /// `VkSurfaceKHR`.
    ///
    /// # Safety
    /// All Vulkan handles must be valid and `device` must have `VK_KHR_swapchain` enabled.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
        queue_family_index: u32,
        window: NativeWindow,
        surface: vk::SurfaceKHR,
        use_present_fences: bool,
    ) -> Result<Self, GpuError> {
        let surface_kind = window.kind();
        let surface_loader = ash::khr::surface::Instance::new(entry, instance);
        let swapchain_loader = ash::khr::swapchain::Device::new(instance, device);

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        // SAFETY: `device` is live.
        let cmd_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| GpuError::Vulkan(format!("vkCreateCommandPool(wsi) failed: {e:?}")))?;

        let cb_ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: `cmd_pool` is live.
        let cmd_buf = match unsafe { device.allocate_command_buffers(&cb_ai) } {
            Ok(v) => v[0],
            Err(e) => {
                // SAFETY: Clean up `cmd_pool` on error.
                unsafe { device.destroy_command_pool(cmd_pool, None) };
                return Err(GpuError::Vulkan(format!(
                    "vkAllocateCommandBuffers(wsi) failed: {e:?}"
                )));
            }
        };

        let sem_ci = vk::SemaphoreCreateInfo::default();
        // SAFETY: `device` is live.
        let image_available = match unsafe { device.create_semaphore(&sem_ci, None) } {
            Ok(s) => s,
            Err(e) => {
                // SAFETY: Clean up `cmd_pool` on error.
                unsafe { device.destroy_command_pool(cmd_pool, None) };
                return Err(GpuError::Vulkan(format!(
                    "vkCreateSemaphore(wsi acquire) failed: {e:?}"
                )));
            }
        };

        // Start signaled so the first frame's wait succeeds immediately.
        let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: `device` is live.
        let submit_fence = match unsafe { device.create_fence(&fence_ci, None) } {
            Ok(f) => f,
            Err(e) => {
                // SAFETY: Clean up semaphore and pool on error.
                unsafe {
                    device.destroy_semaphore(image_available, None);
                    device.destroy_command_pool(cmd_pool, None);
                }
                return Err(GpuError::Vulkan(format!(
                    "vkCreateFence(wsi submit) failed: {e:?}"
                )));
            }
        };

        let mut presenter = Self {
            surface_kind,
            surface_loader,
            swapchain_loader,
            surface,
            window: Some(window),
            physical_device,
            cmd_pool,
            cmd_buf,
            image_available,
            submit_fence,
            use_present_fences,
            active: None,
            retired: Vec::new(),
            needs_recreate: true,
            recreations: 0,
            shutdown_done: false,
        };

        if let Err(e) = presenter.recreate_swapchain(device) {
            presenter.shutdown(device);
            return Err(e);
        }

        info!(
            surface = surface_kind.as_str(),
            present_fences = use_present_fences,
            "initialized native Vulkan WSI swapchain presenter"
        );
        Ok(presenter)
    }

    pub fn surface_kind(&self) -> PresentSurfaceKind {
        self.surface_kind
    }

    #[allow(dead_code)]
    pub fn recreations(&self) -> u64 {
        self.recreations
    }

    fn recreate_swapchain(&mut self, device: &ash::Device) -> Result<(), GpuError> {
        let Some(ref win) = self.window else {
            return Err(GpuError::Vulkan("WSI window already destroyed".into()));
        };
        // SAFETY: `physical_device` and `surface` are live.
        let caps = unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
        }
        .map_err(|e| {
            GpuError::Vulkan(format!("vkGetPhysicalDeviceSurfaceCapabilitiesKHR: {e:?}"))
        })?;
        let formats = unsafe {
            self.surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
        }
        .map_err(|e| GpuError::Vulkan(format!("vkGetPhysicalDeviceSurfaceFormatsKHR: {e:?}")))?;
        let modes = unsafe {
            self.surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
        }
        .map_err(|e| {
            GpuError::Vulkan(format!("vkGetPhysicalDeviceSurfacePresentModesKHR: {e:?}"))
        })?;

        let surface_format = formats
            .iter()
            .copied()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| {
                formats.iter().copied().find(|f| {
                    f.format == vk::Format::R8G8B8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
            })
            .or_else(|| formats.first().copied())
            .ok_or_else(|| GpuError::Vulkan("surface advertised zero formats".into()))?;

        let present_mode = if modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let composite_alpha = if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        {
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };

        let win_ext = win.extent();
        let extent =
            if caps.current_extent.width != u32::MAX && caps.current_extent.height != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: win_ext.width.clamp(
                        caps.min_image_extent.width.max(1),
                        caps.max_image_extent.width.max(1),
                    ),
                    height: win_ext.height.clamp(
                        caps.min_image_extent.height.max(1),
                        caps.max_image_extent.height.max(1),
                    ),
                }
            };
        if extent.width == 0 || extent.height == 0 {
            return Ok(());
        }

        let mut min_image_count = caps.min_image_count.saturating_add(1);
        if caps.max_image_count > 0 && min_image_count > caps.max_image_count {
            min_image_count = caps.max_image_count;
        }

        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
        if !caps.supported_usage_flags.contains(usage) {
            return Err(GpuError::Vulkan(format!(
                "surface does not support COLOR_ATTACHMENT | TRANSFER_DST (supported=0x{:x})",
                caps.supported_usage_flags.as_raw()
            )));
        }

        let old_gen = self.active.take();
        let old_swapchain = old_gen
            .as_ref()
            .map(|g| g.swapchain)
            .unwrap_or(vk::SwapchainKHR::null());

        let ci = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite_alpha)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        // SAFETY: `self.surface` and `old_swapchain` are live.
        let new_sc_res = unsafe { self.swapchain_loader.create_swapchain(&ci, None) };
        if let Some(old) = old_gen {
            self.retired.push(old);
        }
        let swapchain = new_sc_res
            .map_err(|e| GpuError::Vulkan(format!("vkCreateSwapchainKHR failed: {e:?}")))?;

        // SAFETY: `swapchain` is live.
        let images = unsafe { self.swapchain_loader.get_swapchain_images(swapchain) }
            .map_err(|e| GpuError::Vulkan(format!("vkGetSwapchainImagesKHR failed: {e:?}")))?;

        let mut render_finished = Vec::with_capacity(images.len());
        let mut present_fences = Vec::with_capacity(images.len());
        let sem_ci = vk::SemaphoreCreateInfo::default();
        let fence_ci = vk::FenceCreateInfo::default();
        for _ in 0..images.len() {
            // SAFETY: `device` is live.
            let sem = unsafe { device.create_semaphore(&sem_ci, None) }.map_err(|e| {
                GpuError::Vulkan(format!("vkCreateSemaphore(render_finished) failed: {e:?}"))
            })?;
            render_finished.push(sem);
            if self.use_present_fences {
                // SAFETY: `device` is live.
                let f = unsafe { device.create_fence(&fence_ci, None) }.map_err(|e| {
                    GpuError::Vulkan(format!("vkCreateFence(present) failed: {e:?}"))
                })?;
                present_fences.push(f);
            }
        }

        let present_fence_armed = vec![false; present_fences.len()];
        self.active = Some(SwapchainGeneration {
            swapchain,
            images,
            render_finished,
            present_fences,
            present_fence_armed,
            extent,
        });
        self.needs_recreate = false;
        self.recreations = self.recreations.saturating_add(1);
        Ok(())
    }

    fn reap_retired_generations(&mut self, device: &ash::Device) {
        if self.retired.is_empty() || !self.use_present_fences {
            return;
        }
        let mut still_retired = Vec::new();
        for mut sc_gen in self.retired.drain(..) {
            let all_done = sc_gen
                .present_fences
                .iter()
                .zip(sc_gen.present_fence_armed.iter())
                .all(|(&f, &armed)| {
                    !armed || unsafe { device.get_fence_status(f) }.unwrap_or(false)
                });
            if all_done {
                // SAFETY: Every armed present fence in this retired generation is signaled.
                unsafe { sc_gen.destroy(device, &self.swapchain_loader) };
            } else {
                still_retired.push(sc_gen);
            }
        }
        self.retired = still_retired;
    }

    /// Blit a guest 2D render target (`src_image`) to the live Wayland/X11 swapchain and present.
    ///
    /// Returns `Ok(true)` if a frame was presented, or `Ok(false)` if the window was closed or
    /// minimized (zero extent).
    #[allow(clippy::too_many_arguments)]
    pub fn present_image(
        &mut self,
        device: &ash::Device,
        queue: vk::Queue,
        src_image: vk::Image,
        src_width: u32,
        src_height: u32,
        src_layout: vk::ImageLayout,
    ) -> Result<bool, GpuError> {
        if self.shutdown_done {
            return Ok(false);
        }
        // Wait for the previous presentation submission to finish before reusing `cmd_buf` &
        // `image_available`.
        // SAFETY: `submit_fence` was created signaled and is always submitted when reset.
        unsafe {
            device
                .wait_for_fences(&[self.submit_fence], true, u64::MAX)
                .map_err(|e| GpuError::Vulkan(format!("vkWaitForFences(wsi) failed: {e:?}")))?;
        }

        self.reap_retired_generations(device);

        if let Some(ref mut win) = self.window {
            win.pump_events();
            if win.is_closed() {
                return Ok(false);
            }
            if win.take_resized() {
                self.needs_recreate = true;
            }
        }

        if self.needs_recreate || self.active.is_none() {
            self.recreate_swapchain(device)?;
        }

        // Acquire swapchain image (retry once on OUT_OF_DATE).
        let (image_index, acquire_suboptimal) = {
            let mut attempt = 0;
            loop {
                let Some(ref sc_gen) = self.active else {
                    return Ok(false);
                };
                // SAFETY: `sc_gen.swapchain` and `self.image_available` are live and idle.
                match unsafe {
                    self.swapchain_loader.acquire_next_image(
                        sc_gen.swapchain,
                        100_000_000, // 100 ms finite timeout
                        self.image_available,
                        vk::Fence::null(),
                    )
                } {
                    Ok((idx, subopt)) => break (idx, subopt),
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) if attempt == 0 => {
                        attempt += 1;
                        self.recreate_swapchain(device)?;
                    }
                    Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                        return Ok(false);
                    }
                    Err(e) => {
                        return Err(GpuError::Vulkan(format!(
                            "vkAcquireNextImageKHR failed: {e:?}"
                        )));
                    }
                }
            }
        };

        let sc_gen = self
            .active
            .as_mut()
            .ok_or_else(|| GpuError::Vulkan("no active swapchain generation".into()))?;
        let dst_image = sc_gen.images[image_index as usize];
        let dst_extent = sc_gen.extent;
        let render_finished = sc_gen.render_finished[image_index as usize];

        if self.use_present_fences && sc_gen.present_fence_armed[image_index as usize] {
            let pf = sc_gen.present_fences[image_index as usize];
            // SAFETY: Wait until previous present on this swapchain image finished before reusing.
            unsafe {
                let _ = device.wait_for_fences(&[pf], true, u64::MAX);
                let _ = device.reset_fences(&[pf]);
            }
            sc_gen.present_fence_armed[image_index as usize] = false;
        }

        let cmd = self.cmd_buf;
        // SAFETY: `submit_fence` is signaled and we acquired a valid image; reset & record blit.
        unsafe {
            device
                .reset_fences(&[self.submit_fence])
                .map_err(|e| GpuError::Vulkan(format!("vkResetFences(wsi) failed: {e:?}")))?;
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .map_err(|e| {
                    GpuError::Vulkan(format!("vkResetCommandBuffer(wsi) failed: {e:?}"))
                })?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd, &begin).map_err(|e| {
                GpuError::Vulkan(format!("vkBeginCommandBuffer(wsi) failed: {e:?}"))
            })?;

            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            let (src_pre_stage, src_pre_access) = match src_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                    vk::PipelineStageFlags2::CLEAR | vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                _ => (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                ),
            };

            // 1. Transition guest RT -> TRANSFER_SRC_OPTIMAL and swapchain image -> TRANSFER_DST_OPTIMAL
            let pre_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(src_pre_stage)
                    .src_access_mask(src_pre_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(src_layout)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(src_image)
                    .subresource_range(subresource),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst_image)
                    .subresource_range(subresource),
            ];
            let pre_dep = vk::DependencyInfo::default().image_memory_barriers(&pre_barriers);
            device.cmd_pipeline_barrier2(cmd, &pre_dep);

            // 2. Blit guest RT -> swapchain image
            let layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            let blit_region = [vk::ImageBlit2::default()
                .src_subresource(layers)
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_width as i32,
                        y: src_height as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(layers)
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_extent.width as i32,
                        y: dst_extent.height as i32,
                        z: 1,
                    },
                ])];
            let blit_info = vk::BlitImageInfo2::default()
                .src_image(src_image)
                .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .dst_image(dst_image)
                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .regions(&blit_region)
                .filter(vk::Filter::NEAREST);
            device.cmd_blit_image2(cmd, &blit_info);

            // 3. Transition swapchain image -> PRESENT_SRC_KHR and restore guest RT -> src_layout
            let (restore_dst_stage, restore_dst_access) = match src_layout {
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => (
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                ),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL => (
                    vk::PipelineStageFlags2::CLEAR | vk::PipelineStageFlags2::COPY,
                    vk::AccessFlags2::TRANSFER_WRITE,
                ),
                _ => (
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                ),
            };

            let post_barriers = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(dst_image)
                    .subresource_range(subresource),
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .dst_stage_mask(restore_dst_stage)
                    .dst_access_mask(restore_dst_access)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(src_layout)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(src_image)
                    .subresource_range(subresource),
            ];
            let post_dep = vk::DependencyInfo::default().image_memory_barriers(&post_barriers);
            device.cmd_pipeline_barrier2(cmd, &post_dep);

            device
                .end_command_buffer(cmd)
                .map_err(|e| GpuError::Vulkan(format!("vkEndCommandBuffer(wsi) failed: {e:?}")))?;

            let wait_sems = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.image_available)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let sig_sems = [vk::SemaphoreSubmitInfo::default()
                .semaphore(render_finished)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let submits = [vk::SubmitInfo2::default()
                .wait_semaphore_infos(&wait_sems)
                .command_buffer_infos(&cb_infos)
                .signal_semaphore_infos(&sig_sems)];
            device
                .queue_submit2(queue, &submits, self.submit_fence)
                .map_err(|e| GpuError::Vulkan(format!("vkQueueSubmit2(wsi) failed: {e:?}")))?;

            let swapchains = [sc_gen.swapchain];
            let indices = [image_index];
            let present_waits = [render_finished];
            let present_fence_arr = [if self.use_present_fences {
                sc_gen.present_fences[image_index as usize]
            } else {
                vk::Fence::null()
            }];
            let mut present_fence_info =
                vk::SwapchainPresentFenceInfoEXT::default().fences(&present_fence_arr);
            let mut present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&present_waits)
                .swapchains(&swapchains)
                .image_indices(&indices);
            if self.use_present_fences {
                present_info = present_info.push_next(&mut present_fence_info);
                sc_gen.present_fence_armed[image_index as usize] = true;
            }

            match self.swapchain_loader.queue_present(queue, &present_info) {
                Ok(subopt) => {
                    if subopt || acquire_suboptimal {
                        self.needs_recreate = true;
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.needs_recreate = true;
                }
                Err(e) => {
                    return Err(GpuError::Vulkan(format!("vkQueuePresentKHR failed: {e:?}")));
                }
            }
        }

        if let Some(ref mut win) = self.window {
            win.pump_events();
        }

        Ok(true)
    }

    /// Idempotent, validation-clean teardown: drains device + present fences, destroys all
    /// swapchains/semaphores/fences/command pool, destroys `VkSurfaceKHR`, then drops native window.
    pub fn shutdown(&mut self, device: &ash::Device) {
        if self.shutdown_done {
            return;
        }
        self.shutdown_done = true;

        // SAFETY: Wait for device idle and any armed present fences before destroying WSI objects.
        unsafe {
            let _ = device.device_wait_idle();
            if self.use_present_fences {
                let mut armed_fences = Vec::new();
                if let Some(ref sc_gen) = self.active {
                    for (&f, &armed) in sc_gen
                        .present_fences
                        .iter()
                        .zip(sc_gen.present_fence_armed.iter())
                    {
                        if armed {
                            armed_fences.push(f);
                        }
                    }
                }
                for sc_gen in &self.retired {
                    for (&f, &armed) in sc_gen
                        .present_fences
                        .iter()
                        .zip(sc_gen.present_fence_armed.iter())
                    {
                        if armed {
                            armed_fences.push(f);
                        }
                    }
                }
                if !armed_fences.is_empty() {
                    let _ = device.wait_for_fences(&armed_fences, true, 500_000_000);
                }
            }

            if let Some(mut sc_gen) = self.active.take() {
                sc_gen.destroy(device, &self.swapchain_loader);
            }
            for mut sc_gen in self.retired.drain(..) {
                sc_gen.destroy(device, &self.swapchain_loader);
            }

            device.destroy_fence(self.submit_fence, None);
            device.destroy_semaphore(self.image_available, None);
            device.destroy_command_pool(self.cmd_pool, None);

            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
                self.surface = vk::SurfaceKHR::null();
            }
        }
        // Drop native Wayland/XCB window strictly AFTER `vkDestroySurfaceKHR`.
        let _ = self.window.take();
    }
}
