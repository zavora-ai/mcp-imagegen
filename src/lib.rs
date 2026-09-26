//! mcp-imagegen — local image generation over MCP with pluggable backends.
//!
//! See `specs/image-gen-mcp/design.md` in the games workspace for the design.

pub mod backend;
pub mod config;
pub mod error;
pub mod input;
pub mod jobs;
pub mod memory;
pub mod output;
pub mod registry;
pub mod request;
pub mod schedule;
pub mod server;

/// The registry manifest, embedded so the binary works from any working directory.
pub const MANIFEST_TOML: &str = include_str!("../mcp-server.toml");
