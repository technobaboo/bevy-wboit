use bevy::color::LinearRgba;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_phase::ViewSortedRenderPhases;
use bevy::render::render_resource::{
    LoadOp, Operations, RenderPassColorAttachment, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, StoreOp,
};
use bevy::render::renderer::{RenderContext, ViewQuery};
use bevy::render::view::{ExtractedView, ViewDepthTexture};

use crate::phase::WboitAccum3d;
use crate::textures::WboitTextures;

/// Render the WBOIT accumulation pass into MRT textures.
pub fn wboit_accum_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &WboitTextures,
    )>,
    wboit_phases: Res<ViewSortedRenderPhases<WboitAccum3d>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, depth, wboit_textures) = view.into_inner();

    let Some(wboit_phase) = wboit_phases.get(&extracted_view.retained_view_entity) else {
        return;
    };

    if wboit_phase.items.is_empty() {
        return;
    }

    let fi = wboit_textures.frame_index;

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("wboit_accum_pass"),
        color_attachments: &[
            // Target 0: accumulation (Rgba16Float), clear to transparent
            Some(RenderPassColorAttachment {
                view: &wboit_textures.accum.default_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::new(0.0, 0.0, 0.0, 0.0).into()),
                    store: StoreOp::Store,
                },
            }),
            // Target 1: revealage (R8Unorm), clear to 1.0
            Some(RenderPassColorAttachment {
                view: &wboit_textures.revealage[fi].default_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(LinearRgba::new(1.0, 0.0, 0.0, 0.0).into()),
                    store: StoreOp::Store,
                },
            }),
        ],
        // Use existing depth from opaque pass (load, don't clear)
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: depth.view(),
            depth_ops: Some(Operations {
                load: LoadOp::Load,
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = wboit_phase.render(&mut render_pass, world, view_entity) {
        error!("Error rendering WBOIT accum phase: {err:?}");
    }
}
