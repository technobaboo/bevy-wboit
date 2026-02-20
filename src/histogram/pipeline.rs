use bevy::asset::{weak_handle, Handle};
use bevy::pbr::{material_uses_bindless_resources, MeshPipeline, StandardMaterial};
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, BindGroupLayout, BindGroupLayoutEntry, BindingType,
    BlendComponent, BlendFactor, BlendOperation, BlendState, BufferBindingType,
    CachedComputePipelineId, ColorTargetState, ColorWrites, ComputePipelineDescriptor,
    PipelineCache, RenderPipelineDescriptor, SamplerBindingType, Shader, ShaderDefVal,
    ShaderStages, SpecializedMeshPipeline, SpecializedMeshPipelineError, StorageTextureAccess,
    TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use bevy::{pbr::MeshPipelineKey, prelude::*};

pub const HISTO_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("a1b2c3d4-e5f6-7890-abcd-ef1234567890");

pub const HISTO_CDF_BUILD_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("b2c3d4e5-f6a7-8901-bcde-f12345678901");

/// The histogram-equalized WBOIT accumulation pipeline.
///
/// Group layout: 0=View, 1=Mesh, 2=HistogramData, 3=StandardMaterial
#[derive(Resource, Clone)]
pub struct HistogramWboitPipeline {
    pub mesh_pipeline: MeshPipeline,
    /// StandardMaterial bind group layout, inserted at group 3.
    pub material_layout: BindGroupLayout,
    /// Histogram data bind group layout (histogram buf, cdf tex, sampler, params, prev_revealage), group 2.
    pub histo_data_layout_obj: BindGroupLayout,
    pub fragment_shader: Handle<Shader>,
    /// Whether the device supports bindless resources for StandardMaterial.
    pub bindless: bool,
}

impl FromWorld for HistogramWboitPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let material_layout = StandardMaterial::bind_group_layout(render_device);
        let bindless = material_uses_bindless_resources::<StandardMaterial>(render_device);
        let mesh_pipeline = world.resource::<MeshPipeline>().clone();

        // Histogram data bind group layout (group 2 in fragment shader).
        let histo_data_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D3,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ];

        let histo_data_layout_obj = render_device.create_bind_group_layout(
            "histo_data_bind_group_layout",
            &histo_data_entries,
        );

        HistogramWboitPipeline {
            mesh_pipeline,
            material_layout,
            histo_data_layout_obj,
            fragment_shader: HISTO_FRAGMENT_SHADER_HANDLE,
            bindless,
        }
    }
}

impl SpecializedMeshPipeline for HistogramWboitPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut desc = self.mesh_pipeline.specialize(key, layout)?;

        desc.label = Some("histo_wboit_accum_pipeline".into());

        // Material is at group 2 (pbr_bindings.wgsl hardcodes @group(2)).
        // Histo data is at group 3 (histo_fragment.wgsl declares @group(3)).
        desc.vertex
            .shader_defs
            .push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 2));
        if let Some(ref mut fragment) = desc.fragment {
            fragment
                .shader_defs
                .push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 2));
        }

        if self.bindless {
            desc.vertex.shader_defs.push("BINDLESS".into());
            if let Some(ref mut fragment) = desc.fragment {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }

        // Material at group 2 (matches pbr_bindings.wgsl hardcoded @group(2)).
        // Histo data at group 3 (matches histo_fragment.wgsl @group(3) declarations).
        desc.layout.insert(2, self.material_layout.clone());
        desc.layout.push(self.histo_data_layout_obj.clone());

        // Override fragment shader.
        if let Some(ref mut fragment) = desc.fragment {
            fragment.shader = self.fragment_shader.clone();
        }

        // Same MRT targets as naive WBOIT.
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

        if let Some(ref mut ds) = desc.depth_stencil {
            ds.depth_write_enabled = false;
        }

        Ok(desc)
    }
}

/// Resource holding the CDF build compute pipeline.
#[derive(Resource)]
pub struct CdfBuildPipeline {
    pub pipeline_id: CachedComputePipelineId,
    pub bind_group_layout: BindGroupLayout,
}

impl FromWorld for CdfBuildPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        let cdf_build_entries = vec![
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba16Float,
                    view_dimension: TextureViewDimension::D3,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let cdf_build_layout = render_device.create_bind_group_layout(
            "cdf_build_bind_group_layout",
            &cdf_build_entries,
        );

        let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("histo_cdf_build_pipeline".into()),
            layout: vec![cdf_build_layout.clone()],
            shader: HISTO_CDF_BUILD_SHADER_HANDLE,
            shader_defs: vec![],
            entry_point: "main".into(),
            zero_initialize_workgroup_memory: false,
            push_constant_ranges: vec![],
        });

        CdfBuildPipeline {
            pipeline_id,
            bind_group_layout: cdf_build_layout,
        }
    }
}

/// Check that MSAA is off for cameras with HEWboitSettings.
pub fn check_msaa_he_wboit(cameras: Query<&Msaa, With<crate::settings::HEWboitSettings>>) {
    for msaa in &cameras {
        if *msaa != Msaa::Off {
            panic!("HE-WBOIT requires Msaa::Off. Set Msaa::Off on cameras with HEWboitSettings.");
        }
    }
}

/// Ensure depth texture has TEXTURE_BINDING usage for HE-WBOIT cameras.
pub fn configure_depth_texture_usages_he_wboit(
    mut cameras: Query<&mut Camera3d, With<crate::settings::HEWboitSettings>>,
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
