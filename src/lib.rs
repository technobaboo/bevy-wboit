#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod histogram;
pub mod naive;
pub mod phase;
pub mod pipeline;
pub mod queue;
pub mod settings;
pub mod textures;

use bevy::prelude::*;

pub use histogram::HEWboitPlugin;
pub use naive::NaiveWboitPlugin;
pub use settings::{HEWboitSettings, WboitSettings};

/// Convenience plugin that enables naive WBOIT.
/// Add `WboitSettings` to a camera entity to opt in.
pub struct WboitPlugin;

impl Plugin for WboitPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(NaiveWboitPlugin);
    }
}
