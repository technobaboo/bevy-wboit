use bevy::prelude::*;
use bevy::pbr::{
    DrawMesh, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMaterialBindGroup,
    ViewKeyCache,
};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    DrawFunctions, PhaseItemExtraIndex, SetItemPipeline, ViewSortedRenderPhases,
};
use bevy::render::render_resource::{PipelineCache, SpecializedMeshPipelines};
use bevy::render::view::ExtractedView;
use bevy::render::mesh::RenderMesh;
use bevy::core_pipeline::core_3d::Transparent3d;

use crate::phase::WboitAccum3d;
use crate::pipeline::WboitPipeline;
use crate::settings::WboitSettings;

pub type DrawWboit = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    SetMaterialBindGroup<StandardMaterial, 2>,
    DrawMesh,
);

/// Specialize and queue transparent meshes into `WboitAccum3d` for WBOIT cameras.
///
/// Runs after `queue_material_meshes`, reads from `Transparent3d` to get the
/// already-filtered transparent entities, then re-specializes them with the WBOIT pipeline.
pub fn queue_wboit_meshes(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    wboit_pipeline: Option<Res<WboitPipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<WboitPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<WboitAccum3d>>,
    mut wboit_phases: ResMut<ViewSortedRenderPhases<WboitAccum3d>>,
    transparent_phases: Res<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView, With<WboitSettings>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let Some(wboit_pipeline) = wboit_pipeline else {
        return;
    };
    let draw_wboit = draw_functions.read().id::<DrawWboit>();

    for view in &views {
        let Some(wboit_phase) = wboit_phases.get_mut(&view.retained_view_entity) else {
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

            // Use BLEND_ALPHA as the default alpha mode key; WBOIT overrides the
            // fragment shader so this mainly affects vertex shader specialization.
            let mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | MeshPipelineKey::BLEND_ALPHA;

            let pipeline_id =
                pipelines.specialize(&pipeline_cache, &wboit_pipeline, mesh_key, &mesh.layout);
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("WBOIT pipeline specialization error: {err}");
                    continue;
                }
            };

            wboit_phase.add(WboitAccum3d {
                distance: item.distance,
                pipeline: pipeline_id,
                entity: (render_entity, main_entity),
                draw_function: draw_wboit,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: item.indexed,
            });
        }
    }
}

/// Drain transparent phase items for WBOIT cameras so the standard transparent pass is a no-op.
pub fn drain_transparent_for_wboit(
    mut transparent_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<&ExtractedView, With<WboitSettings>>,
) {
    for view in &views {
        if let Some(phase) = transparent_phases.get_mut(&view.retained_view_entity) {
            phase.items.clear();
        }
    }
}
