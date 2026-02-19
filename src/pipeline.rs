use bevy::asset::{uuid_handle, Handle};
use bevy::pbr::{MeshPipeline, StandardMaterial};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, BindGroupLayoutDescriptor, BlendComponent, BlendFactor, BlendOperation,
    BlendState, ColorTargetState, ColorWrites, RenderPipelineDescriptor,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, TextureFormat,
};
use bevy::render::renderer::RenderDevice;
use bevy::shader::{Shader, ShaderDefVal};
use bevy::{pbr::MeshPipelineKey, prelude::*};

pub const WBOIT_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("3e4b7c2a-1f0d-4e8a-9b5c-2d6f7e8a9b0c");

/// The WBOIT accumulation pipeline.
///
/// Wraps `MeshPipeline` but adds the StandardMaterial bind group layout at index 3,
/// overrides the fragment shader for WBOIT MRT output.
#[derive(Resource, Clone)]
pub struct WboitPipeline {
    pub mesh_pipeline: MeshPipeline,
    /// StandardMaterial's bind group layout descriptor, inserted at index 3.
    pub material_layout: BindGroupLayoutDescriptor,
    pub fragment_shader: Handle<Shader>,
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

        // Add MATERIAL_BIND_GROUP shader def (index 3) so PBR imports resolve correctly.
        // MaterialPipelineSpecializer does the same (material.rs:466-475).
        desc.vertex.shader_defs.push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3));
        if let Some(ref mut fragment) = desc.fragment {
            fragment.shader_defs.push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3));
        }

        // Insert StandardMaterial bind group layout at index 3.
        // MeshPipeline::specialize() only produces layouts for groups 0-2;
        // without this the fragment shader's material bindings have no pipeline layout entry.
        desc.layout.insert(3, self.material_layout.clone());

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
    render_device: Res<RenderDevice>,
) {
    let material_layout = StandardMaterial::bind_group_layout_descriptor(&render_device);
    commands.insert_resource(WboitPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        material_layout,
        fragment_shader: WBOIT_FRAGMENT_SHADER_HANDLE,
    });
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
