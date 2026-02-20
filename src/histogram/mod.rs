pub mod accum_pass;
pub mod cdf_build;
pub mod composite;
pub mod pipeline;
pub mod textures;

use bevy::asset::load_internal_asset;
use bevy::prelude::*;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::pbr::queue_material_meshes;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::pbr::MeshPipeline;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, SortedRenderPhasePlugin, ViewSortedRenderPhases,
    sort_phase_system,
};
use bevy::render::render_resource::SpecializedMeshPipelines;
use bevy::render::view::RetainedViewEntity;
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSystems};
use bevy::shader::Shader;
use std::collections::HashSet;

use crate::phase::HistoAccum3d;
use crate::settings::HEWboitSettings;

use self::accum_pass::{
    DrawHistoWboit, drain_transparent_for_he_wboit, histo_wboit_accum_pass,
    queue_histo_wboit_meshes,
};
use self::cdf_build::histo_wboit_cdf_build;
use self::composite::{
    init_histo_composite_pipeline, prepare_histo_wboit_bind_groups,
    queue_histo_composite_pipeline, histo_wboit_composite_pass,
};
use self::pipeline::{
    HistogramWboitPipeline, check_msaa_he_wboit, configure_depth_texture_usages_he_wboit,
    init_histogram_wboit_pipeline,
};
use self::textures::prepare_histogram_wboit_textures;

/// Populate `ViewSortedRenderPhases<HistoAccum3d>` for each active HE-WBOIT camera.
fn extract_histo_wboit_camera_phases(
    mut histo_phases: ResMut<ViewSortedRenderPhases<HistoAccum3d>>,
    cameras: Extract<Query<Entity, (With<Camera3d>, With<HEWboitSettings>)>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
) {
    live_entities.clear();
    for entity in &cameras {
        let retained = RetainedViewEntity::new(entity.into(), None, 0);
        histo_phases.insert_or_clear(retained);
        live_entities.insert(retained);
    }
    histo_phases.retain(|view_entity, _| live_entities.contains(view_entity));
}

/// Plugin implementing histogram-equalized WBOIT (Phase 2).
///
/// Add `HEWboitSettings` to a camera entity to opt in.
pub struct HEWboitPlugin;

impl Plugin for HEWboitPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            pipeline::HISTO_FRAGMENT_SHADER_HANDLE,
            "../shaders/histo_fragment.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            pipeline::HISTO_CDF_BUILD_SHADER_HANDLE,
            "../shaders/histo_cdf_build.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            composite::HISTO_COMPOSITE_SHADER_HANDLE,
            "../shaders/histo_composite.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            ExtractComponentPlugin::<HEWboitSettings>::default(),
            SortedRenderPhasePlugin::<HistoAccum3d, MeshPipeline>::new(
                RenderDebugFlags::default(),
            ),
        ))
        .register_type::<HEWboitSettings>()
        .add_systems(Update, check_msaa_he_wboit)
        .add_systems(Last, configure_depth_texture_usages_he_wboit);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<HistoAccum3d>>()
            .init_resource::<SpecializedMeshPipelines<HistogramWboitPipeline>>()
            .add_render_command::<HistoAccum3d, DrawHistoWboit>()
            .add_systems(ExtractSchedule, extract_histo_wboit_camera_phases)
            .add_systems(
                Render,
                (
                    prepare_histogram_wboit_textures
                        .in_set(RenderSystems::PrepareResources),
                    queue_histo_wboit_meshes.in_set(RenderSystems::QueueMeshes),
                    drain_transparent_for_he_wboit
                        .in_set(RenderSystems::QueueMeshes)
                        .after(queue_material_meshes),
                    sort_phase_system::<HistoAccum3d>.in_set(RenderSystems::PhaseSort),
                    queue_histo_composite_pipeline.in_set(RenderSystems::Queue),
                    prepare_histo_wboit_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                (
                    histo_wboit_accum_pass
                        .after(main_transparent_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                    histo_wboit_cdf_build
                        .after(histo_wboit_accum_pass)
                        .in_set(Core3dSystems::MainPass),
                    histo_wboit_composite_pass
                        .after(histo_wboit_cdf_build)
                        .in_set(Core3dSystems::MainPass),
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.add_systems(
            bevy::render::RenderStartup,
            (init_histogram_wboit_pipeline, init_histo_composite_pipeline),
        );
    }
}
