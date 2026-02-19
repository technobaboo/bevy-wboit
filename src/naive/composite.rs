use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
    PipelineCache, RenderPassDescriptor, RenderPipelineDescriptor,
    ShaderStages, TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy::render::view::ViewTarget;
use bevy::core_pipeline::FullscreenShader;
use bevy::shader::Shader;

use crate::settings::WboitSettings;
use crate::textures::WboitTextures;

/// Per-camera component storing the composite pipeline ID.
#[derive(Component)]
pub struct WboitCompositePipelineId(pub CachedRenderPipelineId);

/// Per-camera component storing the composite bind group.
#[derive(Component)]
pub struct WboitCompositeBindGroup(pub BindGroup);

/// Resource holding the composite pipeline layout.
#[derive(Resource)]
pub struct WboitCompositePipeline {
    pub bind_group_layout_descriptor: BindGroupLayoutDescriptor,
    pub bind_group_layout: bevy::render::render_resource::BindGroupLayout,
    pub fragment_shader: Handle<Shader>,
}

/// Initialize the composite pipeline resource.
pub fn init_wboit_composite_pipeline(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    mut shaders: ResMut<Assets<Shader>>,
) {
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

    let bind_group_layout_descriptor = BindGroupLayoutDescriptor::new(
        "wboit_composite_bind_group_layout",
        &entries,
    );

    let bind_group_layout = render_device.create_bind_group_layout(
        "wboit_composite_bind_group_layout",
        &entries,
    );

    let fragment_shader = shaders.add(Shader::from_wgsl(
        include_str!("../shaders/wboit_composite.wgsl"),
        "wboit_composite.wgsl",
    ));

    commands.insert_resource(WboitCompositePipeline {
        bind_group_layout_descriptor,
        bind_group_layout,
        fragment_shader,
    });
}

/// Queue the composite pipeline for each WBOIT camera.
pub fn queue_wboit_composite_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    composite_pipeline: Option<Res<WboitCompositePipeline>>,
    fullscreen_shader: Res<FullscreenShader>,
    views: Query<(Entity, &ViewTarget), With<WboitSettings>>,
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
            layout: vec![composite_pipeline.bind_group_layout_descriptor.clone()],
            vertex: fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: composite_pipeline.fragment_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("fragment".into()),
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
            immediate_size: 0,
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

/// Render the WBOIT composite pass (fullscreen triangle).
pub fn wboit_composite_pass(
    view: ViewQuery<(
        &ExtractedCamera,
        &ViewTarget,
        &WboitCompositePipelineId,
        &WboitCompositeBindGroup,
    )>,
    pipeline_cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (camera, view_target, pipeline_id, bind_group) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("wboit_composite_pass"),
        color_attachments: &[Some(view_target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &bind_group.0, &[]);
    render_pass.draw(0..3, 0..1);
}
