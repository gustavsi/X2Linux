//! Native Linux gamepad input (`GameInput` / `XInput` → SDL3 Gamepad / `evdev`).

use thiserror::Error;

/// Errors raised while enumerating or polling Linux gamepad devices.
#[derive(Debug, Error)]
pub enum InputError {
    #[error("input backend error: {0}")]
    Backend(String),
}

/// Normalized gamepad snapshot bridged to console input APIs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GamepadState {
    pub buttons: u32,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub left_thumb_x: i16,
    pub left_thumb_y: i16,
    pub right_thumb_x: i16,
    pub right_thumb_y: i16,
}
