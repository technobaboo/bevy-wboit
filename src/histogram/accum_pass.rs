use bevy::color::LinearRgba;
use bevy::ecs::query::QueryItem;
use bevy::pbr::{
    DrawMesh, MeshPipelineKey, RenderMeshInstances, SetMaterialBindGroup, SetMeshBindGroup,
    SetMeshViewBindGroup, ViewKeyCache,
};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_phase::{
    DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::{PipelineCache, SpecializedMeshPipelines};
use bevy::render::renderer::RenderContext;
use bevy::render::view::{ExtractedView, ViewDepthTexture};
use bevy::render::render_resource::{
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, StoreOp,
};
use bevy::core_pipeline::core_3d::Transparent3d;

use crate::phase::HistoAccum3d;
use crate::settings::HEWboitSettings;
use crate::textures::WboitTextures;
use super::composite::HistoAccumBindGroups;
use super::pipeline::HistogramWboitPipeline;

/// RenderCommand that sets the histogram data bind group (group 3) from `HistoAccumBindGroups`.
/// Selects the bind group matching the current `frame_index` from `WboitTextures`.
pub struct SetHistoAccumBindGroup<const I: usize>;

impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetHistoAccumBindGroup<I> {
    type Param = ();
    type ViewQuery = (&'static HistoAccumBindGroups, &'static WboitTextures);
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        (bind_groups, wboit_textures): (
            &'w HistoAccumBindGroups,
            &'w WboitTextures,
        ),
        _entity: Option<()>,
        _param: (),
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        pass.set_bind_group(I, &bind_groups.0[wboit_textures.frame_index], &[]);
        RenderCommandResult::Success
    }
}

/// Draw command type for HE-WBOIT transparent meshes.
/// Material at group 2 (pbr_bindings.wgsl hardcodes @group(2)).
/// Histo data at group 3 (histo_fragment.wgsl declares @group(3)).
pub type DrawHistoWboit = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetMaterialBindGroup<StandardMaterial, 2>,
    SetHistoAccumBindGroup<3>,
    DrawMesh,
);

/// Specialize and queue transparent meshes into `HistoAccum3d` for HE-WBOIT cameras.
///
/// Runs after `queue_material_meshes`, reads from `Transparent3d` to get the
/// already-filtered transparent entities, then re-specializes with the histo WBOIT pipeline.
pub fn queue_histo_wboit_meshes(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    histo_pipeline: Option<Res<HistogramWboitPipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<HistogramWboitPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<HistoAccum3d>>,
    mut histo_phases: ResMut<ViewSortedRenderPhases<HistoAccum3d>>,
    transparent_phases: Res<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView, With<HEWboitSettings>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let Some(histo_pipeline) = histo_pipeline else {
        return;
    };
    let draw_histo = draw_functions.read().id::<DrawHistoWboit>();

    for view in &views {
        let Some(histo_phase) = histo_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let Some(transparent_phase) = transparent_phases.get(&view.retained_view_entity) else {
            continue;
        };

        for item in &transparent_phase.items {
            let (render_entity, main_entity) = item.entity;

            let Some(mesh_instance) =
                render_mesh_instances.render_mesh_queue_data(main_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | MeshPipelineKey::BLEND_ALPHA;

            let pipeline_id = pipelines.specialize(
                &pipeline_cache,
                &histo_pipeline,
                mesh_key,
                &mesh.layout,
            );
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("HE-WBOIT pipeline specialization error: {err}");
                    continue;
                }
            };

            histo_phase.add(HistoAccum3d {
                distance: item.distance,
                pipeline: pipeline_id,
                entity: (render_entity, main_entity),
                draw_function: draw_histo,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: item.indexed,
            });
        }
    }
}

/// Drain `Transparent3d` phase items for HE-WBOIT cameras so the standard pass is a no-op.
pub fn drain_transparent_for_he_wboit(
    mut transparent_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView, With<HEWboitSettings>>,
) {
    for view in &views {
        if let Some(phase) = transparent_phases.get_mut(&view.retained_view_entity) {
            phase.items.clear();
        }
    }
}

/// Render graph label for the HE-WBOIT accumulation pass.
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct HistoWboitAccumPass;

/// Render graph node that renders the HE-WBOIT accumulation pass into MRT textures.
#[derive(Default)]
pub struct HistoWboitAccumNode;

impl ViewNode for HistoWboitAccumNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ExtractedView,
        &'static ViewDepthTexture,
        &'static WboitTextures,
    );

    fn run<'w>(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, extracted_view, depth, wboit_textures): QueryItem<Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let histo_phases = world.resource::<ViewSortedRenderPhases<HistoAccum3d>>();
        let Some(histo_phase) = histo_phases.get(&extracted_view.retained_view_entity) else {
            return Ok(());
        };

        if histo_phase.items.is_empty() {
            return Ok(());
        }

        let view_entity = graph.view_entity();
        let fi = wboit_textures.frame_index;

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("histo_wboit_accum_pass"),
            color_attachments: &[
                Some(RenderPassColorAttachment {
                    view: &wboit_textures.accum.default_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::new(0.0, 0.0, 0.0, 0.0).into()),
                        store: StoreOp::Store,
                    },
                }),
                Some(RenderPassColorAttachment {
                    view: &wboit_textures.revealage[fi].default_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(LinearRgba::new(1.0, 0.0, 0.0, 0.0).into()),
                        store: StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: depth.view(),
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if let Err(err) = histo_phase.render(&mut render_pass, world, view_entity) {
            error!("Error rendering HE-WBOIT accum phase: {err:?}");
        }

        Ok(())
    }
}
