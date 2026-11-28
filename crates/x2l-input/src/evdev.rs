//! Pure-Rust Linux `/dev/input/event*` (`evdev`) gamepad enumeration and polling mapped directly
//! onto public GDK `GameInputGamepadState` (zero C build-header dependencies).

use std::ffi::c_int;
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

// Linux `input-event-codes.h` constants (x86-64 Linux ABI).
pub const EV_SYN: u16 = 0x00;
pub const EV_KEY: u16 = 0x01;
pub const EV_ABS: u16 = 0x03;

pub const BTN_SOUTH: u16 = 0x130; // A
pub const BTN_EAST: u16 = 0x131; // B
pub const BTN_NORTH: u16 = 0x133; // X (Linux BTN_X)
pub const BTN_WEST: u16 = 0x134; // Y (Linux BTN_Y)
pub const BTN_TL: u16 = 0x136; // Left shoulder
pub const BTN_TR: u16 = 0x137; // Right shoulder
pub const BTN_SELECT: u16 = 0x13a; // View
pub const BTN_START: u16 = 0x13b; // Menu
pub const BTN_THUMBL: u16 = 0x13d; // Left thumbstick click
pub const BTN_THUMBR: u16 = 0x13e; // Right thumbstick click

pub const BTN_DPAD_UP: u16 = 0x220;
pub const BTN_DPAD_DOWN: u16 = 0x221;
pub const BTN_DPAD_LEFT: u16 = 0x222;
pub const BTN_DPAD_RIGHT: u16 = 0x223;

pub const ABS_X: u16 = 0x00;
pub const ABS_Y: u16 = 0x01;
pub const ABS_Z: u16 = 0x02; // Left trigger
pub const ABS_RX: u16 = 0x03;
pub const ABS_RY: u16 = 0x04;
pub const ABS_RZ: u16 = 0x05; // Right trigger
pub const ABS_HAT0X: u16 = 0x10;
pub const ABS_HAT0Y: u16 = 0x11;

// Linux `EVIOC*` ioctl request codes (verified against `/usr/include/linux/input.h`).
const EVIOCGNAME_256: u64 = 0x8100_4506;
const EVIOCGBIT_EV_8: u64 = 0x8008_4520;
const EVIOCGBIT_KEY_96: u64 = 0x8060_4521;
const EVIOCGBIT_ABS_8: u64 = 0x8008_4523;
const EVIOCGABS_BASE: u64 = 0x8018_4540;

const O_NONBLOCK: c_int = 0x800;
const O_CLOEXEC: c_int = 0x80000;

unsafe extern "C" {
    fn ioctl(fd: c_int, request: u64, ...) -> c_int;
}

// Public Microsoft `GameInputGamepadButtons` bitflags (`GameInput.h`).
pub const GAMEINPUT_GAMEPAD_NONE: u32 = 0x0000_0000;
pub const GAMEINPUT_GAMEPAD_MENU: u32 = 0x0000_0001;
pub const GAMEINPUT_GAMEPAD_VIEW: u32 = 0x0000_0002;
pub const GAMEINPUT_GAMEPAD_A: u32 = 0x0000_0004;
pub const GAMEINPUT_GAMEPAD_B: u32 = 0x0000_0008;
pub const GAMEINPUT_GAMEPAD_X: u32 = 0x0000_0010;
pub const GAMEINPUT_GAMEPAD_Y: u32 = 0x0000_0020;
pub const GAMEINPUT_GAMEPAD_DPAD_UP: u32 = 0x0000_0040;
pub const GAMEINPUT_GAMEPAD_DPAD_DOWN: u32 = 0x0000_0080;
pub const GAMEINPUT_GAMEPAD_DPAD_LEFT: u32 = 0x0000_0100;
pub const GAMEINPUT_GAMEPAD_DPAD_RIGHT: u32 = 0x0000_0200;
pub const GAMEINPUT_GAMEPAD_LEFT_SHOULDER: u32 = 0x0000_0400;
pub const GAMEINPUT_GAMEPAD_RIGHT_SHOULDER: u32 = 0x0000_0800;
pub const GAMEINPUT_GAMEPAD_LEFT_THUMBSTICK: u32 = 0x0000_1000;
pub const GAMEINPUT_GAMEPAD_RIGHT_THUMBSTICK: u32 = 0x0000_2000;

/// Public GDK `GameInputGamepadState` Win64 ABI structure (28 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GameInputGamepadState {
    pub buttons: u32,
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_thumbstick_x: f32,
    pub left_thumbstick_y: f32,
    pub right_thumbstick_x: f32,
    pub right_thumbstick_y: f32,
}

/// Linux x86-64 `struct input_event` (24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputEvent {
    pub tv_sec: i64,
    pub tv_usec: i64,
    pub ev_type: u16,
    pub code: u16,
    pub value: i32,
}

/// Linux `struct input_absinfo` (24 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputAbsInfo {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub fuzz: i32,
    pub flat: i32,
    pub resolution: i32,
}

/// Axis calibration ranges for sticks and triggers.
#[derive(Debug, Clone, Copy)]
pub struct GamepadAxisCalibration {
    pub lx: InputAbsInfo,
    pub ly: InputAbsInfo,
    pub rx: InputAbsInfo,
    pub ry: InputAbsInfo,
    pub lt: InputAbsInfo,
    pub rt: InputAbsInfo,
    pub stick_deadzone: f32,
    pub trigger_deadzone: f32,
}

impl Default for GamepadAxisCalibration {
    fn default() -> Self {
        let stick = InputAbsInfo {
            minimum: -32768,
            maximum: 32767,
            ..Default::default()
        };
        let trigger = InputAbsInfo {
            minimum: 0,
            maximum: 1023,
            ..Default::default()
        };
        Self {
            lx: stick,
            ly: stick,
            rx: stick,
            ry: stick,
            lt: trigger,
            rt: trigger,
            stick_deadzone: 0.15,
            trigger_deadzone: 0.02,
        }
    }
}

/// Pure-Rust state machine translating Linux `evdev` [`InputEvent`] streams into
/// [`GameInputGamepadState`].
#[derive(Debug, Clone)]
pub struct EvdevGamepadParser {
    calibration: GamepadAxisCalibration,
    state: GameInputGamepadState,
}

impl Default for EvdevGamepadParser {
    fn default() -> Self {
        Self::new(GamepadAxisCalibration::default())
    }
}

impl EvdevGamepadParser {
    pub fn new(calibration: GamepadAxisCalibration) -> Self {
        Self {
            calibration,
            state: GameInputGamepadState::default(),
        }
    }

    pub fn state(&self) -> GameInputGamepadState {
        self.state
    }

    /// Apply a single Linux `evdev` event and update the normalized gamepad state.
    pub fn apply_event(&mut self, ev: &InputEvent) {
        match ev.ev_type {
            EV_KEY => self.apply_key(ev.code, ev.value != 0),
            EV_ABS => self.apply_abs(ev.code, ev.value),
            _ => {}
        }
    }

    fn apply_key(&mut self, code: u16, pressed: bool) {
        let bit = match code {
            BTN_SOUTH => GAMEINPUT_GAMEPAD_A,
            BTN_EAST => GAMEINPUT_GAMEPAD_B,
            BTN_NORTH => GAMEINPUT_GAMEPAD_X,
            BTN_WEST => GAMEINPUT_GAMEPAD_Y,
            BTN_TL => GAMEINPUT_GAMEPAD_LEFT_SHOULDER,
            BTN_TR => GAMEINPUT_GAMEPAD_RIGHT_SHOULDER,
            BTN_SELECT => GAMEINPUT_GAMEPAD_VIEW,
            BTN_START => GAMEINPUT_GAMEPAD_MENU,
            BTN_THUMBL => GAMEINPUT_GAMEPAD_LEFT_THUMBSTICK,
            BTN_THUMBR => GAMEINPUT_GAMEPAD_RIGHT_THUMBSTICK,
            BTN_DPAD_UP => GAMEINPUT_GAMEPAD_DPAD_UP,
            BTN_DPAD_DOWN => GAMEINPUT_GAMEPAD_DPAD_DOWN,
            BTN_DPAD_LEFT => GAMEINPUT_GAMEPAD_DPAD_LEFT,
            BTN_DPAD_RIGHT => GAMEINPUT_GAMEPAD_DPAD_RIGHT,
            _ => return,
        };
        if pressed {
            self.state.buttons |= bit;
        } else {
            self.state.buttons &= !bit;
        }
    }

    fn apply_abs(&mut self, code: u16, value: i32) {
        match code {
            ABS_X => {
                self.state.left_thumbstick_x =
                    normalize_stick(value, self.calibration.lx, self.calibration.stick_deadzone);
            }
            ABS_Y => {
                // Invert Y: Linux evdev has -Y up, GDK GameInputGamepadState has +Y up.
                let y =
                    normalize_stick(value, self.calibration.ly, self.calibration.stick_deadzone);
                self.state.left_thumbstick_y = if y == 0.0 { 0.0 } else { -y };
            }
            ABS_RX => {
                self.state.right_thumbstick_x =
                    normalize_stick(value, self.calibration.rx, self.calibration.stick_deadzone);
            }
            ABS_RY => {
                let y =
                    normalize_stick(value, self.calibration.ry, self.calibration.stick_deadzone);
                self.state.right_thumbstick_y = if y == 0.0 { 0.0 } else { -y };
            }
            ABS_Z => {
                self.state.left_trigger = normalize_trigger(
                    value,
                    self.calibration.lt,
                    self.calibration.trigger_deadzone,
                );
            }
            ABS_RZ => {
                self.state.right_trigger = normalize_trigger(
                    value,
                    self.calibration.rt,
                    self.calibration.trigger_deadzone,
                );
            }
            ABS_HAT0X => {
                self.state.buttons &= !(GAMEINPUT_GAMEPAD_DPAD_LEFT | GAMEINPUT_GAMEPAD_DPAD_RIGHT);
                if value < 0 {
                    self.state.buttons |= GAMEINPUT_GAMEPAD_DPAD_LEFT;
                } else if value > 0 {
                    self.state.buttons |= GAMEINPUT_GAMEPAD_DPAD_RIGHT;
                }
            }
            ABS_HAT0Y => {
                self.state.buttons &= !(GAMEINPUT_GAMEPAD_DPAD_UP | GAMEINPUT_GAMEPAD_DPAD_DOWN);
                if value < 0 {
                    self.state.buttons |= GAMEINPUT_GAMEPAD_DPAD_UP;
                } else if value > 0 {
                    self.state.buttons |= GAMEINPUT_GAMEPAD_DPAD_DOWN;
                }
            }
            _ => {}
        }
    }
}

fn normalize_stick(raw: i32, info: InputAbsInfo, deadzone: f32) -> f32 {
    let min = f64::from(info.minimum);
    let max = f64::from(info.maximum);
    let span = max - min;
    if span <= 0.0 {
        return 0.0;
    }
    let center = (min + max) * 0.5;
    let half = span * 0.5;
    let norm = ((f64::from(raw) - center) / half).clamp(-1.0, 1.0) as f32;
    let dz = deadzone.clamp(0.0, 0.95);
    if norm.abs() <= dz {
        0.0
    } else {
        let sign = norm.signum();
        (sign * ((norm.abs() - dz) / (1.0 - dz))).clamp(-1.0, 1.0)
    }
}

fn normalize_trigger(raw: i32, info: InputAbsInfo, deadzone: f32) -> f32 {
    let min = f64::from(info.minimum);
    let max = f64::from(info.maximum);
    let span = max - min;
    if span <= 0.0 {
        return 0.0;
    }
    let norm = ((f64::from(raw) - min) / span).clamp(0.0, 1.0) as f32;
    let dz = deadzone.clamp(0.0, 0.95);
    if norm <= dz {
        0.0
    } else {
        ((norm - dz) / (1.0 - dz)).clamp(0.0, 1.0)
    }
}

fn bit_is_set(bitmap: &[u8], bit: u16) -> bool {
    let idx = (bit as usize) / 8;
    let mask = 1u8 << ((bit as usize) % 8);
    bitmap.get(idx).is_some_and(|&b| (b & mask) != 0)
}

fn query_abs_info(fd: c_int, axis: u16) -> Option<InputAbsInfo> {
    let mut info = InputAbsInfo::default();
    // SAFETY: `EVIOCGABS(axis)` writes a 24-byte `struct input_absinfo` into `info`.
    let rc = unsafe {
        ioctl(
            fd,
            EVIOCGABS_BASE + u64::from(axis),
            &mut info as *mut InputAbsInfo,
        )
    };
    if rc >= 0 && info.maximum > info.minimum {
        Some(info)
    } else {
        None
    }
}

/// Open Linux `/dev/input/event*` gamepad handle.
pub struct EvdevGamepadDevice {
    path: PathBuf,
    name: String,
    file: File,
    parser: EvdevGamepadParser,
}

impl EvdevGamepadDevice {
    /// Open and classify an `/dev/input/event*` node. Returns `None` if unreadable or not a gamepad.
    pub fn try_open(path: PathBuf) -> Option<Self> {
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(O_NONBLOCK | O_CLOEXEC)
            .open(&path)
            .ok()?;
        let fd = file.as_raw_fd();

        let mut ev_bits = [0u8; 8];
        // SAFETY: `EVIOCGBIT(0, 8)` writes up to 8 bytes into `ev_bits`.
        if unsafe { ioctl(fd, EVIOCGBIT_EV_8, ev_bits.as_mut_ptr()) } < 0 {
            return None;
        }
        if !bit_is_set(&ev_bits, EV_KEY) || !bit_is_set(&ev_bits, EV_ABS) {
            return None;
        }

        let mut key_bits = [0u8; 96];
        // SAFETY: `EVIOCGBIT(EV_KEY, 96)` writes up to 96 bytes into `key_bits`.
        if unsafe { ioctl(fd, EVIOCGBIT_KEY_96, key_bits.as_mut_ptr()) } < 0 {
            return None;
        }
        if !bit_is_set(&key_bits, BTN_SOUTH) {
            return None;
        }

        let mut abs_bits = [0u8; 8];
        // SAFETY: `EVIOCGBIT(EV_ABS, 8)` writes up to 8 bytes into `abs_bits`.
        if unsafe { ioctl(fd, EVIOCGBIT_ABS_8, abs_bits.as_mut_ptr()) } < 0 {
            return None;
        }
        if !bit_is_set(&abs_bits, ABS_X) || !bit_is_set(&abs_bits, ABS_Y) {
            return None;
        }

        let mut name_buf = [0u8; 256];
        // SAFETY: `EVIOCGNAME(256)` writes a NUL-terminated UTF-8/ASCII name into `name_buf`.
        let name = if unsafe { ioctl(fd, EVIOCGNAME_256, name_buf.as_mut_ptr()) } >= 0 {
            let end = name_buf
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(name_buf.len());
            String::from_utf8_lossy(&name_buf[..end]).into_owned()
        } else {
            path.display().to_string()
        };

        let mut cal = GamepadAxisCalibration::default();
        if let Some(info) = query_abs_info(fd, ABS_X) {
            cal.lx = info;
        }
        if let Some(info) = query_abs_info(fd, ABS_Y) {
            cal.ly = info;
        }
        if let Some(info) = query_abs_info(fd, ABS_RX) {
            cal.rx = info;
        }
        if let Some(info) = query_abs_info(fd, ABS_RY) {
            cal.ry = info;
        }
        if let Some(info) = query_abs_info(fd, ABS_Z) {
            cal.lt = info;
        }
        if let Some(info) = query_abs_info(fd, ABS_RZ) {
            cal.rt = info;
        }

        Some(Self {
            path,
            name,
            file,
            parser: EvdevGamepadParser::new(cal),
        })
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn state(&self) -> GameInputGamepadState {
        self.parser.state()
    }

    /// Drain all currently pending non-blocking `evdev` events. Returns `false` if the device
    /// disconnected.
    pub fn poll(&mut self) -> bool {
        let mut buf = [0u8; std::mem::size_of::<InputEvent>() * 16];
        loop {
            match self.file.read(&mut buf) {
                Ok(0) => return false,
                Ok(n) => {
                    let stride = std::mem::size_of::<InputEvent>();
                    let count = n / stride;
                    for i in 0..count {
                        let off = i * stride;
                        // SAFETY: `InputEvent` is `#[repr(C)]` plain integer data (24 bytes),
                        // copied via unaligned read from a full 24-byte kernel event record.
                        let ev = unsafe {
                            std::ptr::read_unaligned(buf.as_ptr().add(off).cast::<InputEvent>())
                        };
                        self.parser.apply_event(&ev);
                    }
                    if n < buf.len() {
                        return true;
                    }
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => return true,
                Err(_) => return false,
            }
        }
    }
}

/// Manager scanning and polling connected Linux `/dev/input/event*` gamepads.
#[derive(Default)]
pub struct EvdevGamepadManager {
    devices: Vec<EvdevGamepadDevice>,
    scanned: bool,
}

impl EvdevGamepadManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Scan `/dev/input/event*` for readable Linux gamepads. Never errors in headless environments.
    pub fn scan(&mut self) {
        self.scanned = true;
        let Ok(entries) = fs::read_dir("/dev/input") else {
            return;
        };
        let mut paths: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("event"))
            })
            .collect();
        paths.sort();

        for path in paths {
            if self.devices.iter().any(|d| d.path() == &path) {
                continue;
            }
            if let Some(dev) = EvdevGamepadDevice::try_open(path) {
                self.devices.push(dev);
            }
        }
    }

    /// Number of currently open Linux `evdev` gamepads.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Poll all open gamepads and return the first connected gamepad's normalized state, if any.
    pub fn poll_primary(&mut self) -> Option<GameInputGamepadState> {
        if !self.scanned {
            self.scan();
        }
        self.devices.retain_mut(|dev| dev.poll());
        self.devices.first().map(EvdevGamepadDevice::state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(ev_type: u16, code: u16, value: i32) -> InputEvent {
        InputEvent {
            tv_sec: 0,
            tv_usec: 0,
            ev_type,
            code,
            value,
        }
    }

    #[test]
    fn synthetic_evdev_stream_maps_buttons_dpad_triggers_and_sticks() {
        let mut parser = EvdevGamepadParser::default();

        // Press A, B, X, Y, LB, RB, View, Menu, L3, R3
        for code in [
            BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST, BTN_TL, BTN_TR, BTN_SELECT, BTN_START,
            BTN_THUMBL, BTN_THUMBR,
        ] {
            parser.apply_event(&ev(EV_KEY, code, 1));
        }
        // D-pad via hat axes: left (-1) and up (-1)
        parser.apply_event(&ev(EV_ABS, ABS_HAT0X, -1));
        parser.apply_event(&ev(EV_ABS, ABS_HAT0Y, -1));

        // Triggers: full left, half right
        parser.apply_event(&ev(EV_ABS, ABS_Z, 1023));
        parser.apply_event(&ev(EV_ABS, ABS_RZ, 512));

        // Sticks: full right (+1.0), full up (-32768 -> +1.0 inverted), within deadzone -> 0.0
        parser.apply_event(&ev(EV_ABS, ABS_X, 32767));
        parser.apply_event(&ev(EV_ABS, ABS_Y, -32768));
        parser.apply_event(&ev(EV_ABS, ABS_RX, 1000)); // inside 15% deadzone
        parser.apply_event(&ev(EV_ABS, ABS_RY, 32767)); // down -> -1.0

        let st = parser.state();
        let expected_buttons = GAMEINPUT_GAMEPAD_A
            | GAMEINPUT_GAMEPAD_B
            | GAMEINPUT_GAMEPAD_X
            | GAMEINPUT_GAMEPAD_Y
            | GAMEINPUT_GAMEPAD_LEFT_SHOULDER
            | GAMEINPUT_GAMEPAD_RIGHT_SHOULDER
            | GAMEINPUT_GAMEPAD_VIEW
            | GAMEINPUT_GAMEPAD_MENU
            | GAMEINPUT_GAMEPAD_LEFT_THUMBSTICK
            | GAMEINPUT_GAMEPAD_RIGHT_THUMBSTICK
            | GAMEINPUT_GAMEPAD_DPAD_LEFT
            | GAMEINPUT_GAMEPAD_DPAD_UP;
        assert_eq!(st.buttons, expected_buttons);
        assert!((st.left_trigger - 1.0).abs() < 1e-4);
        assert!(st.right_trigger > 0.45 && st.right_trigger < 0.55);
        assert!((st.left_thumbstick_x - 1.0).abs() < 1e-4);
        assert!((st.left_thumbstick_y - 1.0).abs() < 1e-4);
        assert_eq!(st.right_thumbstick_x, 0.0);
        assert!((st.right_thumbstick_y + 1.0).abs() < 1e-4);

        // Release A and center hat
        parser.apply_event(&ev(EV_KEY, BTN_SOUTH, 0));
        parser.apply_event(&ev(EV_ABS, ABS_HAT0X, 0));
        parser.apply_event(&ev(EV_ABS, ABS_HAT0Y, 0));
        let st2 = parser.state();
        assert_eq!(st2.buttons & GAMEINPUT_GAMEPAD_A, 0);
        assert_eq!(
            st2.buttons & (GAMEINPUT_GAMEPAD_DPAD_LEFT | GAMEINPUT_GAMEPAD_DPAD_UP),
            0
        );
    }

    #[test]
    fn evdev_manager_scan_does_not_fail_without_controllers() {
        let mut mgr = EvdevGamepadManager::new();
        mgr.scan();
        let _ = mgr.poll_primary();
    }
}
