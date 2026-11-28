// Offline `GameInput` v0 state & callback manager informed by public Microsoft `GameInput.h`,
// Marvisak/VIctory (commit 63bcca43c6a00a758510398fa43096ce7e4f7df0,
// crates/victory-hle/src/modules/gameinput.rs), and xcl-dev/xcl (commit
// 6089122a1c96dff16e93b6fdf04c8ed20cfb6102, crates/xcl/src/hle/gameinput.rs).
// Original license: GNU General Public License v3.0 or later (GPL-3.0-or-later).
// Written for X2Linux on 2026-11-27.

//! Offline-ready `GameInput` state backing `GameInput.dll` ordinal `#1` (`GameInputCreate`).

use crate::evdev::{EvdevGamepadManager, GameInputGamepadState};
use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Public `GAMEINPUT_E_READING_NOT_FOUND` (`0x838A0007`) returned when no gamepad reading exists.
pub const GAMEINPUT_E_READING_NOT_FOUND: i32 = -2088108025;
/// Public `GAMEINPUT_E_DEVICE_NOT_FOUND` (`0x838A0001`).
pub const GAMEINPUT_E_DEVICE_NOT_FOUND: i32 = -2088108031;

/// Thread-safe `GameInput` v0 runtime state fed by Linux `/dev/input/event*` (`evdev`) polling.
pub struct GameInputState {
    epoch: Instant,
    next_token: AtomicU64,
    active_tokens: Mutex<BTreeSet<u64>>,
    focus_policy: AtomicU64,
    evdev: Mutex<EvdevGamepadManager>,
    synthetic_override: Mutex<Option<GameInputGamepadState>>,
}

impl Default for GameInputState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameInputState {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            next_token: AtomicU64::new(1),
            active_tokens: Mutex::new(BTreeSet::new()),
            focus_policy: AtomicU64::new(0),
            evdev: Mutex::new(EvdevGamepadManager::new()),
            synthetic_override: Mutex::new(None),
        }
    }

    /// Monotonic microsecond timestamp for `IGameInput::GetCurrentTimestamp`.
    pub fn current_timestamp_us(&self) -> u64 {
        self.epoch.elapsed().as_micros() as u64
    }

    /// Poll live Linux `evdev` gamepads (or a test-injected synthetic state) and return the
    /// current [`GameInputGamepadState`], or `None` when no controller is connected.
    pub fn current_gamepad_state(&self) -> Option<GameInputGamepadState> {
        if let Ok(guard) = self.synthetic_override.lock()
            && let Some(state) = *guard
        {
            return Some(state);
        }
        self.evdev.lock().ok()?.poll_primary()
    }

    /// Inject or clear a synthetic gamepad state (useful for deterministic headless testing).
    pub fn set_synthetic_gamepad_state(&self, state: Option<GameInputGamepadState>) {
        if let Ok(mut guard) = self.synthetic_override.lock() {
            *guard = state;
        }
    }

    /// Register a callback token for `RegisterReadingCallback` / `RegisterDeviceCallback` / etc.
    pub fn register_callback_token(&self) -> u64 {
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut set) = self.active_tokens.lock() {
            set.insert(token);
        }
        token
    }

    /// Stop or unregister an active callback token.
    pub fn unregister_callback_token(&self, token: u64) -> bool {
        if let Ok(mut set) = self.active_tokens.lock() {
            set.remove(&token)
        } else {
            false
        }
    }

    /// Store `IGameInput::SetFocusPolicy`.
    pub fn set_focus_policy(&self, policy: u32) {
        self.focus_policy
            .store(u64::from(policy), Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gameinput_tokens_and_timestamps_advance() {
        let gi = GameInputState::new();
        let t0 = gi.current_timestamp_us();
        let tok = gi.register_callback_token();
        assert_ne!(tok, 0);
        assert!(gi.unregister_callback_token(tok));
        assert!(!gi.unregister_callback_token(tok));
        assert!(gi.current_timestamp_us() >= t0);
    }
}
