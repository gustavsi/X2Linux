//! Console `kernelx` HLE: memory budgets, flag-stripped allocations, and thread naming/affinity.

use x2l_common::DeviceType;

/// Reported title memory budget in bytes for the selected console profile.
pub const fn title_memory_budget_bytes(device: DeviceType) -> u64 {
    match device {
        DeviceType::SeriesX => 13_500_000_000,
        DeviceType::SeriesS => 8_000_000_000,
    }
}
