use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::{BindGroup, ComputePassDescriptor, PipelineCache};
use bevy::render::renderer::RenderContext;
use bevy::render::view::ExtractedView;

use super::pipeline::CdfBuildPipeline;
use super::textures::HistogramWboitTextures;

/// Per-camera bind group for the CDF build compute pass.
#[derive(Component)]
pub struct CdfBuildBindGroup(pub BindGroup);

/// Render graph label for the HE-WBOIT CDF build pass.
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct HistoCdfBuildPass;

/// Render graph node that runs the CDF build compute pass.
///
/// Dispatches (tile_count_x, tile_count_y, 1) workgroups, each with 64 threads (= num_bins).
/// The compute shader also clears the histogram buffer for the next frame.
#[derive(Default)]
pub struct HistoCdfBuildNode;

impl ViewNode for HistoCdfBuildNode {
    type ViewQuery = (
        &'static ExtractedView,
        Option<&'static HistogramWboitTextures>,
        Option<&'static CdfBuildBindGroup>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (_extracted_view, histo_textures_opt, cdf_bind_group_opt): QueryItem<Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(histo_textures), Some(cdf_bind_group)) =
            (histo_textures_opt, cdf_bind_group_opt)
        else {
            return Ok(());
        };

        let cdf_build_pipeline = match world.get_resource::<CdfBuildPipeline>() {
            Some(p) => p,
            None => return Ok(()),
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(cdf_build_pipeline.pipeline_id)
        else {
            return Ok(());
        };

        let mut compute_pass =
            render_context
                .command_encoder()
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

        Ok(())
    }
}
