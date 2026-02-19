use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderDevice;
use bevy::render::texture::{CachedTexture, TextureCache};

use crate::settings::WboitSettings;

/// Per-camera WBOIT textures in the render world.
#[derive(Component)]
pub struct WboitTextures {
    /// Rgba16Float accumulation texture
    pub accum: CachedTexture,
    /// R8Unorm revealage textures, double-buffered for histogram variant
    pub revealage: [CachedTexture; 2],
    /// Toggles 0/1 each frame for double buffering
    pub frame_index: usize,
}

/// Prepare (create/resize) WBOIT textures for cameras with `WboitSettings`.
pub fn prepare_wboit_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera), With<WboitSettings>>,
    mut existing: Query<&mut WboitTextures>,
) {
    for (entity, camera) in &cameras {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        let width = size.x;
        let height = size.y;

        let accum = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("wboit_accum"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let revealage_a = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("wboit_revealage_a"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        let revealage_b = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("wboit_revealage_b"),
                size: Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Unorm,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Toggle frame index or initialize
        if let Ok(mut tex) = existing.get_mut(entity) {
            tex.accum = accum;
            tex.revealage = [revealage_a, revealage_b];
            tex.frame_index = 1 - tex.frame_index;
        } else {
            commands.entity(entity).insert(WboitTextures {
                accum,
                revealage: [revealage_a, revealage_b],
                frame_index: 0,
            });
        }
    }
}
