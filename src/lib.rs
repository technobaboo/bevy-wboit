pub mod settings;
pub mod phase;
pub mod pipeline;
pub mod queue;
pub mod textures;
pub mod naive;
pub mod histogram;

use bevy::prelude::*;

pub use settings::{WboitSettings, HistogramWboitSettings};
pub use naive::NaiveWboitPlugin;

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
