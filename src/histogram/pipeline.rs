use bevy::asset::{uuid_handle, Handle};
use bevy::pbr::{material_uses_bindless_resources, MeshPipeline, StandardMaterial};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BlendComponent, BlendFactor, BlendOperation, BlendState, BufferBindingType,
    CachedComputePipelineId, ColorTargetState, ColorWrites, ComputePipelineDescriptor,
    PipelineCache, RenderPipelineDescriptor, SamplerBindingType, ShaderStages,
    SpecializedMeshPipeline, SpecializedMeshPipelineError, StorageTextureAccess,
    TextureFormat, TextureSampleType, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use bevy::shader::{Shader, ShaderDefVal};
use bevy::{pbr::MeshPipelineKey, prelude::*};

pub const HISTO_FRAGMENT_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("a1b2c3d4-e5f6-7890-abcd-ef1234567890");

pub const HISTO_CDF_BUILD_SHADER_HANDLE: Handle<Shader> =
    uuid_handle!("b2c3d4e5-f6a7-8901-bcde-f12345678901");

/// The histogram-equalized WBOIT accumulation pipeline.
///
/// Group layout: 0=View, 1=ViewArray, 2=Mesh, 3=HistogramData, 4=StandardMaterial
#[derive(Resource, Clone)]
pub struct HistogramWboitPipeline {
    pub mesh_pipeline: MeshPipeline,
    /// StandardMaterial bind group layout descriptor, inserted at group 4.
    pub material_layout: BindGroupLayoutDescriptor,
    /// Histogram data bind group layout descriptor (histogram buf, cdf tex, sampler, params, prev_revealage), group 3.
    pub histo_data_layout: BindGroupLayoutDescriptor,
    /// Actual BindGroupLayout object for histo_data (for bind group creation).
    pub histo_data_layout_obj: BindGroupLayout,
    pub fragment_shader: Handle<Shader>,
    /// Whether the device supports bindless resources for StandardMaterial.
    pub bindless: bool,
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

        // Material is at group 4 (not 3 as in naive WBOIT).
        desc.vertex
            .shader_defs
            .push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 4));
        if let Some(ref mut fragment) = desc.fragment {
            fragment
                .shader_defs
                .push(ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 4));
        }

        if self.bindless {
            desc.vertex.shader_defs.push("BINDLESS".into());
            if let Some(ref mut fragment) = desc.fragment {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }

        // Insert histo_data layout at group 3, then material layout at group 4.
        // MeshPipeline::specialize() produces groups 0-2; we extend to 0-4.
        desc.layout.insert(3, self.histo_data_layout.clone());
        desc.layout.insert(4, self.material_layout.clone());

        // Override fragment shader.
        if let Some(ref mut fragment) = desc.fragment {
            fragment.shader = self.fragment_shader.clone();
        }

        // Same MRT targets as naive WBOIT:
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

        // Depth: test enabled, write disabled.
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

/// Initialize the histogram WBOIT pipeline and CDF build compute pipeline resources.
pub fn init_histogram_wboit_pipeline(
    mut commands: Commands,
    mesh_pipeline: Res<MeshPipeline>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    let material_layout = StandardMaterial::bind_group_layout_descriptor(&render_device);
    let bindless = material_uses_bindless_resources::<StandardMaterial>(&render_device);

    // Histogram data bind group layout (group 3 in fragment shader).
    // Matches histo_fragment.wgsl: @group(3) @binding(0..4)
    let histo_data_entries = vec![
        // binding 0: histogram storage buffer (written by fragment, read/cleared by compute)
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
        // binding 1: cdf_texture (texture_3d, filterable, from previous frame's build)
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
        // binding 2: cdf_sampler
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(SamplerBindingType::Filtering),
            count: None,
        },
        // binding 3: histo_params uniform
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
        // binding 4: prev_revealage_tex (non-filterable, from the other frame buffer)
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
    let histo_data_layout =
        BindGroupLayoutDescriptor::new("histo_data_bind_group_layout", &histo_data_entries);

    commands.insert_resource(HistogramWboitPipeline {
        mesh_pipeline: mesh_pipeline.clone(),
        material_layout,
        histo_data_layout,
        histo_data_layout_obj,
        fragment_shader: HISTO_FRAGMENT_SHADER_HANDLE,
        bindless,
    });

    // CDF build compute pipeline bind group layout (group 0 in compute shader).
    // Matches histo_cdf_build.wgsl: @group(0) @binding(0..2)
    let cdf_build_entries = vec![
        // binding 0: histogram storage buffer (read and cleared by compute)
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
        // binding 1: cdf_out storage texture 3d (written by compute)
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
        // binding 2: histo_params uniform
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

    let cdf_build_layout_desc =
        BindGroupLayoutDescriptor::new("cdf_build_bind_group_layout", &cdf_build_entries);

    let pipeline_id = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("histo_cdf_build_pipeline".into()),
        layout: vec![cdf_build_layout_desc],
        shader: HISTO_CDF_BUILD_SHADER_HANDLE,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        zero_initialize_workgroup_memory: false,
        immediate_size: 0,
    });

    commands.insert_resource(CdfBuildPipeline {
        pipeline_id,
        bind_group_layout: cdf_build_layout,
    });
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
