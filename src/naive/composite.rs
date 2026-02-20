use bevy::asset::{weak_handle, Handle};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingType,
    BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
    PipelineCache, RenderPassDescriptor, RenderPipelineDescriptor,
    Shader, ShaderStages, TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::ViewTarget;

use crate::settings::WboitSettings;
use crate::textures::WboitTextures;

pub const WBOIT_COMPOSITE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("5f2a9d1b-3c4e-4f7a-8b6c-1e2f3a4b5c6d");

/// Render graph label for the WBOIT composite pass.
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct WboitCompositePass;

/// Per-camera component storing the composite pipeline ID.
#[derive(Component)]
pub struct WboitCompositePipelineId(pub CachedRenderPipelineId);

/// Per-camera component storing the composite bind group.
#[derive(Component)]
pub struct WboitCompositeBindGroup(pub BindGroup);

/// Resource holding the composite pipeline layout.
#[derive(Resource)]
pub struct WboitCompositePipeline {
    pub bind_group_layout: BindGroupLayout,
    pub fragment_shader: Handle<Shader>,
}

impl FromWorld for WboitCompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let entries = vec![
            // Binding 0: accum texture
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // Binding 1: revealage texture
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];

        let bind_group_layout = render_device.create_bind_group_layout(
            "wboit_composite_bind_group_layout",
            &entries,
        );

        WboitCompositePipeline {
            bind_group_layout,
            fragment_shader: WBOIT_COMPOSITE_SHADER_HANDLE,
        }
    }
}

/// Queue the composite pipeline for each WBOIT camera.
pub fn queue_wboit_composite_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    composite_pipeline: Option<Res<WboitCompositePipeline>>,
    views: Query<(Entity, &ViewTarget), (With<WboitSettings>, Without<WboitCompositePipelineId>)>,
) {
    let Some(composite_pipeline) = composite_pipeline else {
        return;
    };
    for (entity, view_target) in &views {
        let format = if view_target.main_texture_format() == ViewTarget::TEXTURE_FORMAT_HDR {
            ViewTarget::TEXTURE_FORMAT_HDR
        } else {
            TextureFormat::bevy_default()
        };

        let pipeline_id = pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
            label: Some("wboit_composite_pipeline".into()),
            layout: vec![composite_pipeline.bind_group_layout.clone()],
            vertex: fullscreen_shader_vertex_state(),
            fragment: Some(FragmentState {
                shader: composite_pipeline.fragment_shader.clone(),
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: default(),
            depth_stencil: None,
            multisample: default(),
            zero_initialize_workgroup_memory: false,
            push_constant_ranges: vec![],
        });

        commands
            .entity(entity)
            .insert(WboitCompositePipelineId(pipeline_id));
    }
}

/// Prepare the composite bind group for each WBOIT camera.
pub fn prepare_wboit_composite_bind_group(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    composite_pipeline: Option<Res<WboitCompositePipeline>>,
    views: Query<(Entity, &WboitTextures), With<WboitSettings>>,
) {
    let Some(composite_pipeline) = composite_pipeline else {
        return;
    };
    for (entity, wboit_textures) in &views {
        let fi = wboit_textures.frame_index;
        let bind_group = render_device.create_bind_group(
            "wboit_composite_bind_group",
            &composite_pipeline.bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &wboit_textures.accum.default_view,
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: bevy::render::render_resource::BindingResource::TextureView(
                        &wboit_textures.revealage[fi].default_view,
                    ),
                },
            ],
        );

        commands
            .entity(entity)
            .insert(WboitCompositeBindGroup(bind_group));
    }
}

/// Render graph node that runs the WBOIT composite pass (fullscreen triangle).
#[derive(Default)]
pub struct WboitCompositeNode;

impl ViewNode for WboitCompositeNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        Option<&'static WboitCompositePipelineId>,
        Option<&'static WboitCompositeBindGroup>,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (camera, view_target, pipeline_id_opt, bind_group_opt): QueryItem<Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let (Some(pipeline_id), Some(bind_group)) = (pipeline_id_opt, bind_group_opt) else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
            return Ok(());
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("wboit_composite_pass"),
            color_attachments: &[Some(view_target.get_color_attachment())],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        render_pass.set_render_pipeline(pipeline);
        render_pass.set_bind_group(0, &bind_group.0, &[]);
        render_pass.draw(0..3, 0..1);

        Ok(())
    }
}
