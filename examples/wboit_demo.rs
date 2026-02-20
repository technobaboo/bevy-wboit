use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy_wboit::{HEWboitPlugin, HEWboitSettings, WboitPlugin, WboitSettings};

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, WboitPlugin, HEWboitPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (toggle_mode, rotate_camera))
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera with WBOIT enabled by default (key 2 mode)
    commands.spawn((
        Camera3d::default(),
        Tonemapping::None,
        Transform::from_xyz(0., 2., 8.).looking_at(Vec3::ZERO, Vec3::Y),
        WboitSettings,
        Msaa::Off,
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.5, 0.5, 0.0)),
    ));

    // Opaque ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10.0, 10.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.3),
            ..default()
        })),
        Transform::from_xyz(0.0, -1.5, 0.0),
    ));

    // Overlapping transparent spheres
    let sphere = meshes.add(Sphere::new(1.0).mesh().ico(5).unwrap());

    let configs = [
        (Color::srgba(1.0, 0.0, 0.0, 0.5), Vec3::new(-1.0, 0.0, 0.0)),
        (Color::srgba(0.0, 1.0, 0.0, 0.4), Vec3::new(0.5, 0.0, -0.5)),
        (Color::srgba(0.0, 0.0, 1.0, 0.5), Vec3::new(0.0, 0.0, 1.0)),
        (Color::srgba(1.0, 1.0, 0.0, 0.3), Vec3::new(1.0, 0.5, 0.0)),
        (
            Color::srgba(1.0, 0.0, 1.0, 0.35),
            Vec3::new(-0.5, 0.5, 0.5),
        ),
    ];

    for (color, pos) in configs {
        commands.spawn((
            Mesh3d(sphere.clone()),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: color,
                alpha_mode: AlphaMode::Blend,
                cull_mode: None,
                ..default()
            })),
            Transform::from_translation(pos),
        ));
    }

    // Instructions
    commands.spawn((
        Text::new("1: No OIT  |  2: WBOIT  |  3: HE-WBOIT\nDrag mouse to rotate"),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
    ));
}

fn toggle_mode(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    camera: Query<Entity, With<Camera3d>>,
) {
    let Ok(camera_entity) = camera.single() else {
        return;
    };

    if keys.just_pressed(KeyCode::Digit1) {
        // No OIT
        commands
            .entity(camera_entity)
            .remove::<WboitSettings>()
            .remove::<HEWboitSettings>();
        info!("Switched to standard transparency (no OIT)");
    }

    if keys.just_pressed(KeyCode::Digit2) {
        // Naive WBOIT
        commands
            .entity(camera_entity)
            .remove::<HEWboitSettings>()
            .insert(WboitSettings);
        info!("Switched to naive WBOIT");
    }

    if keys.just_pressed(KeyCode::Digit3) {
        // Histogram-equalized WBOIT
        commands
            .entity(camera_entity)
            .remove::<WboitSettings>()
            .insert(HEWboitSettings::default());
        info!("Switched to HE-WBOIT");
    }
}

fn rotate_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mouse_motion: Res<bevy::input::mouse::AccumulatedMouseMotion>,
    mouse_scroll: Res<bevy::input::mouse::AccumulatedMouseScroll>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    let Ok(mut transform) = camera.single_mut() else {
        return;
    };

    // Keyboard arrow keys
    let mut yaw = 0.0f32;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw += 1.0;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw -= 1.0;
    }
    if yaw != 0.0 {
        let angle = yaw * time.delta_secs();
        transform.rotate_around(Vec3::ZERO, Quat::from_rotation_y(angle));
    }

    // Scroll to zoom (move along camera's forward axis)
    let scroll = mouse_scroll.delta.y;
    if scroll != 0.0 {
        let forward = transform.forward();
        transform.translation += *forward * scroll * 0.5;
    }

    // Mouse drag orbit
    if mouse_buttons.pressed(MouseButton::Left) {
        let delta = mouse_motion.delta;
        if delta != Vec2::ZERO {
            let sensitivity = 0.005;
            transform.rotate_around(
                Vec3::ZERO,
                Quat::from_rotation_y(-delta.x * sensitivity),
            );
            let right = transform.right();
            transform.rotate_around(
                Vec3::ZERO,
                Quat::from_axis_angle(*right, -delta.y * sensitivity),
            );
        }
    }
}
