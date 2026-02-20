use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, Extent3d, Sampler,
    SamplerDescriptor, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureView, TextureViewDescriptor,
};
use bevy::render::render_resource::{FilterMode, MipmapFilterMode};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::TextureCache;

use crate::settings::HEWboitSettings;
use crate::textures::WboitTextures;

/// GPU-side histogram parameters (must match HistogramParams in WGSL shaders).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HistogramParams {
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub num_bins: u32,
    pub tile_size: u32,
    pub max_depth: f32,
    pub _padding: [u32; 3],
}

impl HistogramParams {
    fn as_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&self.tile_count_x.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.tile_count_y.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.num_bins.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.tile_size.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.max_depth.to_le_bytes());
        bytes
    }
}

/// Per-camera HE-WBOIT textures and buffers in the render world.
#[derive(Component)]
pub struct HistogramWboitTextures {
    /// Storage buffer for histogram data: tile_count_x * tile_count_y * num_bins u32 values.
    pub histogram_buffer: Buffer,
    /// 3D CDF texture (tile_count_x, tile_count_y, num_bins), Rgba16Float.
    pub cdf_texture: bevy::render::render_resource::Texture,
    /// Sampled view of cdf_texture (for fragment shader).
    pub cdf_view: TextureView,
    /// Sampler for CDF texture (filtering).
    pub cdf_sampler: Sampler,
    /// Uniform buffer for HistogramParams.
    pub histo_params_buffer: Buffer,
    pub tile_count_x: u32,
    pub tile_count_y: u32,
    pub num_bins: u32,
}

/// Prepare (create/resize) HE-WBOIT textures for cameras with `HEWboitSettings`.
///
/// Creates both `WboitTextures` (accum + revealage MRT) and `HistogramWboitTextures`
/// (histogram buffer, CDF texture, params buffer).
pub fn prepare_histogram_wboit_textures(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut texture_cache: ResMut<TextureCache>,
    cameras: Query<(Entity, &ExtractedCamera, &HEWboitSettings)>,
    mut existing_wboit: Query<&mut WboitTextures>,
    mut existing_histo: Query<&mut HistogramWboitTextures>,
) {
    for (entity, camera, he_settings) in &cameras {
        let Some(size) = camera.physical_viewport_size else {
            continue;
        };
        let width = size.x;
        let height = size.y;

        // --- WboitTextures (accum + double-buffered revealage) ---
        let accum = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("he_wboit_accum"),
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
                label: Some("he_wboit_revealage_a"),
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
                label: Some("he_wboit_revealage_b"),
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

        // Toggle frame_index or initialize
        let new_frame_index = if let Ok(mut tex) = existing_wboit.get_mut(entity) {
            let fi = 1 - tex.frame_index;
            tex.accum = accum;
            tex.revealage = [revealage_a, revealage_b];
            tex.frame_index = fi;
            fi
        } else {
            commands.entity(entity).insert(WboitTextures {
                accum,
                revealage: [revealage_a, revealage_b],
                frame_index: 0,
            });
            0
        };

        // --- HistogramWboitTextures ---
        let tile_size = he_settings.tile_size;
        let num_bins = he_settings.num_bins;
        let tile_count_x = width.div_ceil(tile_size);
        let tile_count_y = height.div_ceil(tile_size);

        let params = HistogramParams {
            tile_count_x,
            tile_count_y,
            num_bins,
            tile_size,
            max_depth: he_settings.max_depth,
            _padding: [0; 3],
        };

        // Check if we need to recreate (size or params changed)
        let needs_recreate = if let Ok(histo) = existing_histo.get(entity) {
            histo.tile_count_x != tile_count_x
                || histo.tile_count_y != tile_count_y
                || histo.num_bins != num_bins
        } else {
            true
        };

        if needs_recreate {
            // Histogram storage buffer: tile_count_x * tile_count_y * num_bins * 4 bytes (u32 per bin).
            // Initialized to zero; the CDF build shader clears it after each frame.
            let histogram_size = (tile_count_x * tile_count_y * num_bins * 4) as u64;
            let histogram_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("histo_histogram_buffer"),
                size: histogram_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            // Uniform buffer for HistogramParams
            let histo_params_buffer =
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("histo_params_buffer"),
                    contents: &params.as_bytes(),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });

            // CDF 3D texture: dims (tile_count_x, tile_count_y, num_bins), Rgba16Float.
            // Needs TEXTURE_BINDING (for fragment shader sampling) and STORAGE_BINDING (for compute write).
            let cdf_texture = render_device.create_texture(&TextureDescriptor {
                label: Some("histo_cdf_texture"),
                size: Extent3d {
                    width: tile_count_x,
                    height: tile_count_y,
                    depth_or_array_layers: num_bins,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D3,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });

            let cdf_view = cdf_texture.create_view(&TextureViewDescriptor::default());

            let cdf_sampler = render_device.create_sampler(&SamplerDescriptor {
                label: Some("histo_cdf_sampler"),
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: MipmapFilterMode::Nearest,
                ..default()
            });

            let new_histo = HistogramWboitTextures {
                histogram_buffer,
                cdf_texture,
                cdf_view,
                cdf_sampler,
                histo_params_buffer,
                tile_count_x,
                tile_count_y,
                num_bins,
            };

            if existing_histo.contains(entity) {
                if let Ok(mut histo) = existing_histo.get_mut(entity) {
                    *histo = new_histo;
                }
            } else {
                commands.entity(entity).insert(new_histo);
            }
        } else {
            // Same dimensions — just update the params buffer in case tile_size changed.
            if let Ok(histo) = existing_histo.get(entity) {
                render_queue.write_buffer(&histo.histo_params_buffer, 0, &params.as_bytes());
            }
        }

        let _ = new_frame_index; // used above
    }
}
