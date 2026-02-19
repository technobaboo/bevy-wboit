#import bevy_pbr::{
    pbr_types,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_types::STANDARD_MATERIAL_FLAGS_UNLIT_BIT,
    forward_io::VertexOutput,
}

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

    // Premultiply based on alpha mode (matching existing OIT behavior in pbr.wgsl)
    let alpha_mode = pbr_input.material.flags
        & pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_RESERVED_BITS;

    var premul: vec4<f32>;
    if alpha_mode == pbr_types::STANDARD_MATERIAL_FLAGS_ALPHA_MODE_BLEND {
        // Blend: manually premultiply
        premul = vec4(color.rgb * color.a, color.a);
    } else {
        // Premultiplied, Add: already premultiplied by post-processing
        premul = color;
    }

    // WBOIT weight function
    // Bevy uses reverse-Z: near=1, far=0, so convert to linear [0,1] where 0=near, 1=far
    let d = 1.0 - in.position.z;
    let alpha = premul.a;
    let w = alpha * clamp(exp2(13.0 - 26.0 * d), 1e-4, 8192.0);

    var out: WboitOutput;
    out.accum = vec4(premul.rgb * w, alpha * w);
    out.revealage = alpha;
    return out;
}
