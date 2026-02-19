use bevy::asset::Handle;
use bevy::pbr::MeshPipeline;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    BlendComponent, BlendFactor, BlendOperation, BlendState, ColorTargetState, ColorWrites,
    RenderPipelineDescriptor, SpecializedMeshPipeline, SpecializedMeshPipelineError, TextureFormat,
};
use bevy::shader::Shader;
use bevy::{pbr::MeshPipelineKey, prelude::*};

/// The WBOIT accumulation pipeline.
///
/// Wraps `MeshPipeline` to reuse all standard bind group layouts and vertex processing,
/// but overrides the fragment shader and render targets for WBOIT MRT output.
#[derive(Resource, Clone)]
pub struct WboitPipeline {
    pub mesh_pipeline: MeshPipeline,
    pub fragment_shader: Handle<Shader>,
}

impl WboitPipeline {
    pub fn new(mesh_pipeline: MeshPipeline, fragment_shader: Handle<Shader>) -> Self {
        Self {
            mesh_pipeline,
            fragment_shader,
        }
    }
}

impl SpecializedMeshPipeline for WboitPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut desc = self.mesh_pipeline.specialize(key, layout)?;

        desc.label = Some("wboit_accum_pipeline".into());

        // Override fragment shader
        if let Some(ref mut fragment) = desc.fragment {
            fragment.shader = self.fragment_shader.clone();
        }

        // Override color targets for MRT:
        // Target 0: accum (Rgba16Float, additive blend)
        // Target 1: revealage (R8Unorm, multiplicative blend)
        if let Some(ref mut fragment) = desc.fragment {
            fragment.targets = vec![
                Some(ColorTargetState {
                    format: TextureFormat::Rgba16Float,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::One,
                            dst_factor: BlendFactor::One,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                }),
                Some(ColorTargetState {
                    format: TextureFormat::R8Unorm,
                    blend: Some(BlendState {
                        color: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::OneMinusSrc,
                            operation: BlendOperation::Add,
                        },
                        alpha: BlendComponent {
                            src_factor: BlendFactor::Zero,
                            dst_factor: BlendFactor::OneMinusSrc,
                            operation: BlendOperation::Add,
                        },
                    }),
                    write_mask: ColorWrites::ALL,
                }),
            ];
        }

        // Depth: test enabled, write disabled (preserve opaque depth)
        if let Some(ref mut ds) = desc.depth_stencil {
            ds.depth_write_enabled = false;
        }

        Ok(desc)
    }
}

/// Initialize the WBOIT pipeline resource.
pub fn init_wboit_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    mut shaders: ResMut<Assets<Shader>>,
) {
    let fragment_shader = shaders.add(Shader::from_wgsl(
        include_str!("shaders/wboit_fragment.wgsl"),
        "wboit_fragment.wgsl",
    ));
    commands.insert_resource(WboitPipeline::new(
        mesh_pipeline.clone(),
        fragment_shader,
    ));
}

/// Check that MSAA is off for cameras with WboitSettings.
pub fn check_msaa_wboit(
    cameras: Query<&Msaa, With<crate::settings::WboitSettings>>,
) {
    for msaa in &cameras {
        if *msaa != Msaa::Off {
            panic!("WBOIT requires Msaa::Off. Set Msaa::Off on cameras with WboitSettings.");
        }
    }
}

/// Ensure depth texture has TEXTURE_BINDING usage for WBOIT cameras.
pub fn configure_depth_texture_usages_wboit(
    mut cameras: Query<&mut Camera3d, With<crate::settings::WboitSettings>>,
) {
    use bevy::render::render_resource::TextureUsages;
    for mut camera_3d in &mut cameras {
        let required = TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let current = TextureUsages::from(camera_3d.depth_texture_usages);
        if !current.contains(required) {
            camera_3d.depth_texture_usages = required.into();
        }
    }
}
