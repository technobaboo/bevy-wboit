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
/// commands.spawn((Camera3d::default(), HEWboitSettings::default(), Msaa::Off));
/// ```
#[derive(Component, Clone, Copy, ExtractComponent, Reflect)]
#[reflect(Default)]
pub struct HEWboitSettings {
    pub tile_size: u32,
    pub num_bins: u32,
    /// Maximum scene depth (in world units) used to normalize linear depth into [0, 1]
    /// for histogram binning. Set this to approximately the farthest transparent object
    /// in your scene. Equivalent to the `far` plane in the reference implementation.
    pub max_depth: f32,
}

impl Default for HEWboitSettings {
    fn default() -> Self {
        Self {
            tile_size: 32,
            num_bins: 64,
            max_depth: 100.0,
        }
    }
}
