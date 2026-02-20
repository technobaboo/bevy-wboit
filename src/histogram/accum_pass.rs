use bevy::color::LinearRgba;
use bevy::pbr::{
    DrawMesh, MeshPipelineKey, RenderMeshInstances, SetMaterialBindGroup, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup, ViewKeyCache,
    alpha_mode_pipeline_key, RenderMaterialInstances, PreparedMaterial,
};
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::erased_render_asset::ErasedRenderAssets;
use bevy::render::mesh::RenderMesh;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
    RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
};
use bevy::render::render_resource::{PipelineCache, SpecializedMeshPipelines};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::view::{ExtractedView, RenderVisibleEntities, ViewDepthTexture};
use bevy::render::render_resource::{
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, StoreOp,
};
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::material::RenderPhaseType;

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
/// Material is at group 4 (histo data occupies group 3).
pub type DrawHistoWboit = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetHistoAccumBindGroup<3>,
    SetMaterialBindGroup<4>,
    DrawMesh,
);

/// Specialize and queue transparent meshes into `HistoAccum3d` for HE-WBOIT cameras.
pub fn queue_histo_wboit_meshes(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances>,
    histo_pipeline: Option<Res<HistogramWboitPipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<HistogramWboitPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<HistoAccum3d>>,
    mut histo_phases: ResMut<ViewSortedRenderPhases<HistoAccum3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<HEWboitSettings>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let Some(histo_pipeline) = histo_pipeline else {
        return;
    };
    let draw_histo = draw_functions.read().id::<DrawHistoWboit>();

    for (view, visible_entities) in &views {
        let Some(histo_phase) = histo_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let rangefinder = view.rangefinder3d();

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            let Some(material_instance) =
                render_material_instances.instances.get(visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                continue;
            };

            // Only queue transparent materials.
            if !matches!(
                material.properties.render_phase_type,
                RenderPhaseType::Transparent
            ) {
                continue;
            }

            let Some(mesh_instance) =
                render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mut mesh_pipeline_key_bits: MeshPipelineKey =
                material.properties.mesh_pipeline_key_bits.downcast();
            mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
                material.properties.alpha_mode,
                &Msaa::Off,
            ));
            let mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | mesh_pipeline_key_bits;

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

            let distance =
                rangefinder.distance(&mesh_instance.center) + material.properties.depth_bias;

            histo_phase.add(HistoAccum3d {
                distance,
                pipeline: pipeline_id,
                entity: (*render_entity, *visible_entity),
                draw_function: draw_histo,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
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

/// Render the HE-WBOIT accumulation pass into MRT textures.
pub fn histo_wboit_accum_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &WboitTextures,
    )>,
    histo_phases: Res<ViewSortedRenderPhases<HistoAccum3d>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, depth, wboit_textures) = view.into_inner();

    let Some(histo_phase) = histo_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    if histo_phase.items.is_empty() {
        return;
    }

    let fi = wboit_textures.frame_index;

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("histo_wboit_accum_pass"),
        color_attachments: &[
            // Target 0: accumulation (Rgba16Float), clear to transparent
            Some(RenderPassColorAttachment {
                view: &wboit_textures.accum.default_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::new(0.0, 0.0, 0.0, 0.0).into()),
                    store: StoreOp::Store,
                },
            }),
            // Target 1: revealage (R8Unorm), clear to 1.0
            Some(RenderPassColorAttachment {
                view: &wboit_textures.revealage[fi].default_view,
                depth_slice: None,
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
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = histo_phase.render(&mut render_pass, world, view_entity) {
        error!("Error rendering HE-WBOIT accum phase: {err:?}");
    }
}
