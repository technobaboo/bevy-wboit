use bevy::prelude::*;
use bevy::pbr::{
    DrawMesh, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
    SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup, SetMaterialBindGroup,
    ViewKeyCache, alpha_mode_pipeline_key,
    RenderMaterialInstances, PreparedMaterial,
};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_phase::{
    DrawFunctions, PhaseItemExtraIndex, SetItemPipeline, ViewSortedRenderPhases,
};
use bevy::render::render_resource::{PipelineCache, SpecializedMeshPipelines};
use bevy::render::view::{ExtractedView, RenderVisibleEntities};
use bevy::render::mesh::RenderMesh;
use bevy::core_pipeline::core_3d::Transparent3d;
use bevy::render::erased_render_asset::ErasedRenderAssets;
use bevy::material::RenderPhaseType;

use crate::phase::WboitAccum3d;
use crate::pipeline::WboitPipeline;
use crate::settings::WboitSettings;

pub type DrawWboit = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    SetMaterialBindGroup<3>,
    DrawMesh,
);

/// Specialize and queue transparent meshes into `WboitAccum3d` for WBOIT cameras.
pub fn queue_wboit_meshes(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_materials: Res<ErasedRenderAssets<PreparedMaterial>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances>,
    wboit_pipeline: Option<Res<WboitPipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<WboitPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<WboitAccum3d>>,
    mut wboit_phases: ResMut<ViewSortedRenderPhases<WboitAccum3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities), With<WboitSettings>>,
    view_key_cache: Res<ViewKeyCache>,
) {
    let Some(wboit_pipeline) = wboit_pipeline else {
        return;
    };
    let draw_wboit = draw_functions.read().id::<DrawWboit>();

    for (view, visible_entities) in &views {
        let Some(wboit_phase) = wboit_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let Some(view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let rangefinder = view.rangefinder3d();

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            // Get material
            let Some(material_instance) =
                render_material_instances.instances.get(visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(material_instance.asset_id) else {
                continue;
            };

            // Only queue transparent materials
            if !matches!(material.properties.render_phase_type, RenderPhaseType::Transparent) {
                continue;
            }

            // Get mesh
            let Some(mesh_instance) =
                render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            // Compute mesh pipeline key
            let mut mesh_pipeline_key_bits: MeshPipelineKey =
                material.properties.mesh_pipeline_key_bits.downcast();
            mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
                material.properties.alpha_mode,
                &Msaa::Off,
            ));
            let mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | mesh_pipeline_key_bits;

            // Specialize the WBOIT pipeline
            let pipeline_id =
                pipelines.specialize(&pipeline_cache, &wboit_pipeline, mesh_key, &mesh.layout);
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("WBOIT pipeline specialization error: {err}");
                    continue;
                }
            };

            let distance =
                rangefinder.distance(&mesh_instance.center) + material.properties.depth_bias;

            wboit_phase.add(WboitAccum3d {
                distance,
                pipeline: pipeline_id,
                entity: (*render_entity, *visible_entity),
                draw_function: draw_wboit,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: mesh.indexed(),
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
