//! Native Linux gamepad input (`GameInput` / `XInput` → pure-Rust Linux `evdev`).

pub mod evdev;
pub mod gameinput;

pub use evdev::{
    EvdevGamepadDevice, EvdevGamepadManager, EvdevGamepadParser, GAMEINPUT_GAMEPAD_A,
    GAMEINPUT_GAMEPAD_B, GAMEINPUT_GAMEPAD_DPAD_DOWN, GAMEINPUT_GAMEPAD_DPAD_LEFT,
    GAMEINPUT_GAMEPAD_DPAD_RIGHT, GAMEINPUT_GAMEPAD_DPAD_UP, GAMEINPUT_GAMEPAD_LEFT_SHOULDER,
    GAMEINPUT_GAMEPAD_LEFT_THUMBSTICK, GAMEINPUT_GAMEPAD_MENU, GAMEINPUT_GAMEPAD_NONE,
    GAMEINPUT_GAMEPAD_RIGHT_SHOULDER, GAMEINPUT_GAMEPAD_RIGHT_THUMBSTICK, GAMEINPUT_GAMEPAD_VIEW,
    GAMEINPUT_GAMEPAD_X, GAMEINPUT_GAMEPAD_Y, GameInputGamepadState, GamepadAxisCalibration,
    InputAbsInfo, InputEvent,
};
pub use gameinput::{GAMEINPUT_E_DEVICE_NOT_FOUND, GAMEINPUT_E_READING_NOT_FOUND, GameInputState};

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
