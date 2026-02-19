use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;

/// Enables naive WBOIT on this camera. Requires `Msaa::Off`.
///
/// Usage:
/// ```ignore
/// commands.spawn((Camera3d::default(), WboitSettings, Msaa::Off));
/// ```
#[derive(Component, Clone, Copy, Default, ExtractComponent, Reflect)]
#[reflect(Default)]
pub struct WboitSettings;

/// Enables histogram-equalized WBOIT on this camera. Requires `Msaa::Off`.
///
/// Usage:
/// ```ignore
/// commands.spawn((Camera3d::default(), HistogramWboitSettings::default(), Msaa::Off));
/// ```
#[derive(Component, Clone, Copy, ExtractComponent, Reflect)]
#[reflect(Default)]
pub struct HistogramWboitSettings {
    pub tile_size: u32,
    pub num_bins: u32,
}

impl Default for HistogramWboitSettings {
    fn default() -> Self {
        Self {
            tile_size: 32,
            num_bins: 64,
        }
    }
}
