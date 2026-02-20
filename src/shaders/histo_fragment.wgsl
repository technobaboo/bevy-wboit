#import bevy_pbr::{
    pbr_types,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
    forward_io::VertexOutput,
}

const TILE_SIZE: u32 = 32u;
const OD_SCALE: f32 = 4096.0;

struct HistogramParams {
    tile_count_x: u32,
    tile_count_y: u32,
    num_bins: u32,
    tile_size: u32,
    max_depth: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(3) @binding(0) var<storage, read_write> histogram: array<atomic<u32>>;
@group(3) @binding(1) var cdf_texture: texture_3d<f32>;
@group(3) @binding(2) var cdf_sampler: sampler;
@group(3) @binding(3) var<uniform> histo_params: HistogramParams;
@group(3) @binding(4) var prev_revealage_tex: texture_2d<f32>;

struct WboitOutput {
    @location(0) accum: vec4<f32>,
    @location(1) revealage: f32,
}

@fragment
fn fragment(
    vertex_output: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> WboitOutput {
    var in = vertex_output;
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    var color: vec4<f32>;
    if (pbr_input.material.flags & STANDARD_MATERIAL_FLAGS_UNLIT_BIT) == 0u {
        color = apply_pbr_lighting(pbr_input);
    } else {
        color = pbr_input.material.base_color;
    }
    color = main_pass_post_lighting_processing(pbr_input, color);

    // Premultiply based on alpha mode
    let alpha_mode = pbr_input.material.flags
        & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;

    var premul: vec4<f32>;
    if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND {
        premul = vec4(color.rgb * color.a, color.a);
    } else {
        premul = color;
    }

    let alpha = premul.a;

    // Compute normalized depth [0,1] using linear eye-space depth.
    // in.position.w = 1/w_clip = 1/eye_z in WGSL fragment shaders, so 1/w gives linear depth.
    // We normalize by max_depth (analogous to the far plane in a finite perspective camera).
    let linear_depth = 1.0 / in.position.w;
    let normalized_z = clamp(linear_depth / histo_params.max_depth, 0.0, 1.0);

    // --- Histogram recording ---
    let nb = histo_params.num_bins;
    let bin = min(u32(normalized_z * f32(nb)), nb - 1u);

    let tile_x = u32(in.position.x) / TILE_SIZE;
    let tile_y = u32(in.position.y) / TILE_SIZE;
    let tile_idx = tile_y * histo_params.tile_count_x + tile_x;

    // Quantize optical depth and accumulate
    let optical_depth = -log(max(1.0 - alpha, 1e-6));
    let quantized_od = u32(clamp(optical_depth * OD_SCALE, 0.0, 65535.0));
    atomicAdd(&histogram[tile_idx * nb + bin], quantized_od);

    // --- CDF-based weight ---
    // Sample CDF from previous frame (trilinear interpolation)
    let u = in.position.x / f32(histo_params.tile_count_x * TILE_SIZE);
    let v = in.position.y / f32(histo_params.tile_count_y * TILE_SIZE);
    let w_coord = normalized_z;
    let equalized_z = textureSampleLevel(cdf_texture, cdf_sampler, vec3f(u, v, w_coord), 0.0).r;

    // Transmittance weight using previous frame's revealage
    let prev_R = textureLoad(prev_revealage_tex, vec2<i32>(in.position.xy), 0).r;
    let wt = pow(max(prev_R, 1e-4), equalized_z);

    var out: WboitOutput;
    out.accum = vec4(premul.rgb * wt, alpha * wt);
    out.revealage = alpha;
    return out;
}
