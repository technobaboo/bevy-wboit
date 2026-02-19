pub mod accum_pass;
pub mod composite;

use bevy::asset::load_internal_asset;
use bevy::prelude::*;
use bevy::core_pipeline::{Core3d, Core3dSystems};
use bevy::core_pipeline::core_3d::main_transparent_pass_3d;
use bevy::pbr::queue_material_meshes;
use bevy::render::extract_component::ExtractComponentPlugin;
use bevy::render::render_phase::{
    AddRenderCommand, DrawFunctions, ViewSortedRenderPhases, sort_phase_system,
};
use bevy::render::render_resource::SpecializedMeshPipelines;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::Shader;

use crate::phase::WboitAccum3d;
use crate::pipeline::{
    WboitPipeline, check_msaa_wboit, configure_depth_texture_usages_wboit, init_wboit_pipeline,
};
use crate::queue::{DrawWboit, drain_transparent_for_wboit, queue_wboit_meshes};
use crate::textures::prepare_wboit_textures;

use self::composite::{
    init_wboit_composite_pipeline, prepare_wboit_composite_bind_group,
    queue_wboit_composite_pipeline,
};

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

        app.add_plugins(ExtractComponentPlugin::<crate::settings::WboitSettings>::default())
            .register_type::<crate::settings::WboitSettings>()
            .add_systems(Update, check_msaa_wboit)
            .add_systems(Last, configure_depth_texture_usages_wboit);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<DrawFunctions<WboitAccum3d>>()
            .init_resource::<ViewSortedRenderPhases<WboitAccum3d>>()
            .init_resource::<SpecializedMeshPipelines<WboitPipeline>>()
            .add_render_command::<WboitAccum3d, DrawWboit>()
            .add_systems(
                Render,
                (
                    prepare_wboit_textures.in_set(RenderSystems::PrepareResources),
                    queue_wboit_meshes.in_set(RenderSystems::QueueMeshes),
                    drain_transparent_for_wboit
                        .in_set(RenderSystems::QueueMeshes)
                        .after(queue_material_meshes),
                    sort_phase_system::<WboitAccum3d>.in_set(RenderSystems::PhaseSort),
                    queue_wboit_composite_pipeline.in_set(RenderSystems::Queue),
                    prepare_wboit_composite_bind_group
                        .in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                (
                    accum_pass::wboit_accum_pass
                        .after(main_transparent_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                    composite::wboit_composite_pass
                        .after(accum_pass::wboit_accum_pass)
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
            (init_wboit_pipeline, init_wboit_composite_pipeline),
        );
    }
}
