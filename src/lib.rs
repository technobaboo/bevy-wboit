#![allow(clippy::too_many_arguments, clippy::type_complexity)]

pub mod histogram;
pub mod naive;
pub mod phase;
pub mod pipeline;
pub mod queue;
pub mod settings;
pub mod textures;

use bevy::prelude::*;

pub use naive::NaiveWboitPlugin;
pub use settings::{HistogramWboitSettings, WboitSettings};

/// Convenience plugin that enables naive WBOIT.
/// Add `WboitSettings` to a camera entity to opt in.
pub struct WboitPlugin;

impl Plugin for WboitPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(NaiveWboitPlugin);
    }
}

/// Placeholder for the histogram-equalized WBOIT plugin.
/// TODO: Phase 2 implementation.
pub struct HistogramWboitPlugin;

impl Plugin for HistogramWboitPlugin {
    fn build(&self, _app: &mut App) {
        // TODO: Phase 2 implementation
    }
}
