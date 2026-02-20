use bevy::asset::{weak_handle, Handle};
use bevy::core_pipeline::fullscreen_vertex_shader::fullscreen_shader_vertex_state;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{NodeRunError, RenderGraphContext, RenderLabel, ViewNode};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource,
    BindingType, BlendState, CachedRenderPipelineId, ColorTargetState, ColorWrites, FragmentState,
    PipelineCache, RenderPassDescriptor, RenderPipelineDescriptor, Shader, ShaderStages,
    TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::ViewTarget;

use crate::settings::HEWboitSettings;
use crate::textures::WboitTextures;
use super::cdf_build::CdfBuildBindGroup;
use super::pipeline::{CdfBuildPipeline, HistogramWboitPipeline};
use super::textures::HistogramWboitTextures;

pub const HISTO_COMPOSITE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("c3d4e5f6-a7b8-9012-cdef-123456789012");

/// Render graph label for the HE-WBOIT composite pass.
#[derive(RenderLabel, Debug, Clone, Hash, PartialEq, Eq)]
pub struct HistoWboitCompositePass;

/// Per-camera component: two accum-pass bind groups (one per frame index).
///
/// `HistoAccumBindGroups.0[i]` binds `prev_revealage = revealage[1-i]`.
/// At render time, we select `bind_groups[frame_index]`.
#[derive(Component)]
pub struct HistoAccumBindGroups(pub [BindGroup; 2]);

/// Per-camera component storing the composite pipeline ID (queued once).
#[derive(Component)]
pub struct HistoCompositePipelineId(pub CachedRenderPipelineId);

/// Per-camera component storing the composite bind group.
#[derive(Component)]
pub struct HistoCompositeBindGroup(pub BindGroup);

/// Resource holding the composite pipeline layout.
#[derive(Resource)]
pub struct HistoCompositePipeline {
    pub bind_group_layout: BindGroupLayout,
    pub fragment_shader: Handle<Shader>,
}

impl FromWorld for HistoCompositePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let entries = vec![
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
            "histo_composite_bind_group_layout",
            &entries,
        );

        HistoCompositePipeline {
            bind_group_layout,
            fragment_shader: HISTO_COMPOSITE_SHADER_HANDLE,
        }
    }
}

/// Queue the composite pipeline once per HE-WBOIT camera.
pub fn queue_histo_composite_pipeline(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    composite_pipeline: Option<Res<HistoCompositePipeline>>,
    views: Query<
        (Entity, &ViewTarget),
        (With<HEWboitSettings>, Without<HistoCompositePipelineId>),
    >,
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
            label: Some("histo_composite_pipeline".into()),
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
            .insert(HistoCompositePipelineId(pipeline_id));
    }
}

/// Prepare bind groups for HE-WBOIT cameras every frame.
pub fn prepare_histo_wboit_bind_groups(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    histo_pipeline: Option<Res<HistogramWboitPipeline>>,
    composite_pipeline: Option<Res<HistoCompositePipeline>>,
    cdf_pipeline: Option<Res<CdfBuildPipeline>>,
    views: Query<(Entity, &WboitTextures, &HistogramWboitTextures), With<HEWboitSettings>>,
) {
    let (Some(histo_pipeline), Some(composite_pipeline), Some(cdf_pipeline)) =
        (histo_pipeline, composite_pipeline, cdf_pipeline)
    else {
        return;
    };

    for (entity, wboit_textures, histo_textures) in &views {
        let accum_bind_groups = [0usize, 1usize].map(|fi| {
            let prev_fi = 1 - fi;
            render_device.create_bind_group(
                "histo_accum_bind_group",
                &histo_pipeline.histo_data_layout_obj,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: histo_textures.histogram_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::TextureView(&histo_textures.cdf_view),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Sampler(&histo_textures.cdf_sampler),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: histo_textures.histo_params_buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(
                            &wboit_textures.revealage[prev_fi].default_view,
                        ),
                    },
                ],
            )
        });

        let cdf_bind_group = render_device.create_bind_group(
            "histo_cdf_build_bind_group",
            &cdf_pipeline.bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: histo_textures.histogram_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&histo_textures.cdf_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: histo_textures.histo_params_buffer.as_entire_binding(),
                },
            ],
        );

        let fi = wboit_textures.frame_index;
        let composite_bind_group = render_device.create_bind_group(
            "histo_composite_bind_group",
            &composite_pipeline.bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&wboit_textures.accum.default_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(
                        &wboit_textures.revealage[fi].default_view,
                    ),
                },
            ],
        );

        commands.entity(entity).insert((
            HistoAccumBindGroups(accum_bind_groups),
            CdfBuildBindGroup(cdf_bind_group),
            HistoCompositeBindGroup(composite_bind_group),
        ));
    }
}

/// Render graph node that renders the HE-WBOIT composite pass (fullscreen triangle).
#[derive(Default)]
pub struct HistoWboitCompositeNode;

impl ViewNode for HistoWboitCompositeNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        Option<&'static HistoCompositePipelineId>,
        Option<&'static HistoCompositeBindGroup>,
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
            label: Some("histo_composite_pass"),
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
