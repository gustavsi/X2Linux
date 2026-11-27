//! PE32+ import resolver, API-set alias engine, and ordinal database for X2Linux.

use thiserror::Error;

/// Errors raised while inspecting PE32+ binaries or resolving console imports.
#[derive(Debug, Error)]
pub enum ResolverError {
    #[error("invalid PE32+ binary: {0}")]
    InvalidPe(#[from] goblin::error::Error),
}

/// Resolution target for an imported DLL name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportTarget {
    Hle(String),
    System(String),
    Unknown(String),
}
