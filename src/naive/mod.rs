pub mod accum_pass;
pub mod composite;

use bevy::asset::load_internal_asset;
use bevy::prelude::*;
use bevy::core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy::pbr::queue_material_meshes;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::pbr::MeshPipeline;
use bevy::render::render_graph::{RenderGraphApp, ViewNodeRunner};
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, SortedRenderPhasePlugin, ViewSortedRenderPhases,
    sort_phase_system,
};
use bevy::render::render_resource::{Shader, SpecializedMeshPipelines};
use bevy::render::view::RetainedViewEntity;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSet};
use std::collections::HashSet;

use crate::phase::WboitAccum3d;
use crate::pipeline::WboitPipeline;
use crate::queue::{DrawWboit, drain_transparent_for_wboit, queue_wboit_meshes};
use crate::textures::prepare_wboit_textures;

use self::accum_pass::{WboitAccumNode, WboitAccumPass};
use self::composite::{
    WboitCompositeNode, WboitCompositePass,
    WboitCompositePipeline, prepare_wboit_composite_bind_group,
    queue_wboit_composite_pipeline,
};

/// Populate `ViewSortedRenderPhases<WboitAccum3d>` with an entry for each active WBOIT camera.
///
/// Mirrors how `extract_core_3d_camera_phases` manages `Transparent3d`.
fn extract_wboit_camera_phases(
    mut wboit_phases: ResMut<ViewSortedRenderPhases<WboitAccum3d>>,
    cameras: Extract<Query<Entity, (With<Camera3d>, With<crate::settings::WboitSettings>)>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();
    for entity in &cameras {
        let retained = RetainedViewEntity::new(entity.into(), None, 0);
        wboit_phases.insert_or_clear(retained);
        live_entities.insert(retained);
    }
    wboit_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

/// Plugin that enables naive WBOIT (McGuire & Bavoil 2013) rendering.
///
/// Add `WboitSettings` to a camera entity to opt in.
pub struct NaiveWboitPlugin;

impl Plugin for NaiveWboitPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            crate::pipeline::WBOIT_FRAGMENT_SHADER_HANDLE,
            "../shaders/wboit_fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            composite::WBOIT_COMPOSITE_SHADER_HANDLE,
            "../shaders/wboit_composite.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            ExtractComponentPlugin::<crate::settings::WboitSettings>::default(),
            // Registers batch_and_prepare_sorted_render_phase + collect_buffers_for_phase for
            // WboitAccum3d, which populates phase_instance_buffers so SetMeshBindGroup<1>
            // can find the per-phase GPU buffer in GPU-preprocessing mode.
            SortedRenderPhasePlugin::<WboitAccum3d, MeshPipeline>::new(
                RenderDebugFlags::default(),
            ),
        ))
        .register_type::<crate::settings::WboitSettings>()
        .add_systems(Update, crate::pipeline::check_msaa_wboit)
        .add_systems(Last, crate::pipeline::configure_depth_texture_usages_wboit);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<WboitAccum3d>>()
            .init_resource::<SpecializedMeshPipelines<WboitPipeline>>()
            .add_render_command::<WboitAccum3d, DrawWboit>()
            .add_systems(ExtractSchedule, extract_wboit_camera_phases)
            .add_systems(
                Render,
                (
                    prepare_wboit_textures.in_set(RenderSet::PrepareResources),
                    queue_wboit_meshes
                        .in_set(RenderSet::QueueMeshes)
                        .after(queue_material_meshes::<StandardMaterial>),
                    drain_transparent_for_wboit
                        .in_set(RenderSet::QueueMeshes)
                        .after(queue_wboit_meshes),
                    sort_phase_system::<WboitAccum3d>.in_set(RenderSet::PhaseSort),
                    queue_wboit_composite_pipeline.in_set(RenderSet::Queue),
                    prepare_wboit_composite_bind_group
                        .in_set(RenderSet::PrepareBindGroups),
                ),
            )
            // Register render graph nodes: accum → composite, placed after MainTransparentPass
            .add_render_graph_node::<ViewNodeRunner<WboitAccumNode>>(Core3d, WboitAccumPass)
            .add_render_graph_node::<ViewNodeRunner<WboitCompositeNode>>(Core3d, WboitCompositePass)
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::MainTransparentPass,
                    WboitAccumPass,
                    WboitCompositePass,
                    Node3d::EndMainPass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<WboitPipeline>()
            .init_resource::<WboitCompositePipeline>();
    }
}
