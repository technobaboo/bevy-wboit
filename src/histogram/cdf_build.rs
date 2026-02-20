use bevy::prelude::*;
use bevy::render::render_resource::{BindGroup, ComputePassDescriptor, PipelineCache};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::view::ExtractedView;

use super::pipeline::CdfBuildPipeline;
use super::textures::HistogramWboitTextures;

/// Per-camera bind group for the CDF build compute pass.
#[derive(Component)]
pub struct CdfBuildBindGroup(pub BindGroup);

/// Run the CDF build compute pass to convert the histogram into a per-tile CDF texture.
///
/// Dispatches (tile_count_x, tile_count_y, 1) workgroups, each with 64 threads (= num_bins).
/// The compute shader also clears the histogram buffer for the next frame.
pub fn histo_wboit_cdf_build(
    view: ViewQuery<(
        &ExtractedView,
        Option<&HistogramWboitTextures>,
        Option<&CdfBuildBindGroup>,
    )>,
    cdf_build_pipeline: Option<Res<CdfBuildPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (extracted_view, histo_textures_opt, cdf_bind_group_opt) = view.into_inner();

    let (Some(histo_textures), Some(cdf_bind_group)) =
        (histo_textures_opt, cdf_bind_group_opt)
    else {
        return;
    };

    let Some(cdf_build_pipeline) = cdf_build_pipeline else {
        return;
    };

    let Some(pipeline) =
        pipeline_cache.get_compute_pipeline(cdf_build_pipeline.pipeline_id)
    else {
        return;
    };

    let mut compute_pass =
        ctx.command_encoder()
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("histo_cdf_build_pass"),
                timestamp_writes: None,
            });

    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, &cdf_bind_group.0, &[]);
    compute_pass.dispatch_workgroups(
        histo_textures.tile_count_x,
        histo_textures.tile_count_y,
        1,
    );

    let _ = extracted_view;
}
