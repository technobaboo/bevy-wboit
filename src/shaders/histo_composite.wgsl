#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var accum_tex: texture_2d<f32>;
@group(0) @binding(1) var revealage_tex: texture_2d<f32>;

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let coords = vec2<i32>(in.position.xy);
    let accum = textureLoad(accum_tex, coords, 0);
    let r = textureLoad(revealage_tex, coords, 0).r;

    // No transparent fragments at this pixel
    if accum.a < 1e-5 {
        discard;
    }

    // Recover average color from weighted sum
    let avg_color = accum.rgb / max(accum.a, 1e-5);

    // Alpha from revealage (product of (1 - alpha_i))
    let alpha = 1.0 - r;

    // Output premultiplied alpha for compositing onto opaque
    return vec4(avg_color * alpha, alpha);
}
