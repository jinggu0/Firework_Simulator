import numpy as np

from simulator.camera import CameraConfig, FreeCamera
from simulator.terrain import TerrainSurface


def _flat_surface(height_m: float = 2.0, water: bool = False) -> TerrainSurface:
    return TerrainSurface(
        np.full((2, 2), height_m, dtype=np.float32),
        np.array([-10.0, -10.0, 10.0, 10.0], dtype=np.float32),
        np.full((2, 2), 255 if water else 0, dtype=np.uint8),
    )


def test_default_camera_faces_toward_negative_z() -> None:
    camera = FreeCamera()
    assert camera.forward[2] < -0.9
    assert camera.forward[1] > 0.0


def test_mouse_pitch_is_clamped() -> None:
    camera = FreeCamera()
    camera.look(0.0, -100_000.0)
    assert camera.pitch_deg == camera.config.maximum_pitch_deg
    camera.look(0.0, 100_000.0)
    assert camera.pitch_deg == camera.config.minimum_pitch_deg


def test_forward_movement_accelerates_smoothly() -> None:
    camera = FreeCamera(
        config=CameraConfig(movement_speed_mps=10.0, acceleration_response_hz=4.0)
    )
    start = camera.position_m.copy()
    camera.update(0.1, np.array([0.0, 0.0, 1.0]))
    assert 0.0 < np.linalg.norm(camera.velocity_mps) < 10.0
    assert camera.position_m[2] < start[2]


def test_diagonal_input_is_normalized() -> None:
    camera = FreeCamera()
    camera.update(10.0, np.array([1.0, 0.0, 1.0]))
    assert np.linalg.norm(camera.velocity_mps) <= (
        camera.config.movement_speed_mps + 1e-5
    )


def test_free_camera_cannot_put_the_lens_inside_the_rendered_ground() -> None:
    camera = FreeCamera(position_m=np.array([0.0, 2.05, 0.0], dtype=np.float32))
    surface = _flat_surface()
    camera.update(1.0, np.array([0.0, -1.0, 0.0]), terrain=surface)
    assert camera.position_m[1] == 2.0 + camera.config.camera_collision_radius_m
    assert camera.velocity_mps[1] >= 0.0


def test_walking_camera_tracks_ground_at_operator_height() -> None:
    surface = TerrainSurface(
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
    )
    camera = FreeCamera(
        config=CameraConfig(
            walking_speed_mps=1.0,
            acceleration_response_hz=1_000.0,
            maximum_step_height_m=2.0,
        ),
        position_m=np.array([0.0, 5.0, 5.0], dtype=np.float32),
        yaw_deg=90.0,
    )
    assert camera.set_walking(True, surface)
    camera.update(10.0, np.array([0.0, 1.0, 1.0]), terrain=surface)
    assert camera.position_m[0] == 10.0
    assert camera.position_m[1] == 1.0 + camera.config.standing_camera_height_m
    assert camera.velocity_mps[1] == 0.0


def test_walking_camera_does_not_enter_the_river() -> None:
    surface = TerrainSurface(
        np.zeros((2, 2), dtype=np.float32),
        np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
        np.array([[0, 255], [0, 255]], dtype=np.uint8),
    )
    camera = FreeCamera(
        config=CameraConfig(walking_speed_mps=1.0, acceleration_response_hz=1_000.0),
        position_m=np.array([0.0, 2.0, 5.0], dtype=np.float32),
        yaw_deg=90.0,
    )
    assert camera.set_walking(True, surface)
    camera.update(10.0, np.array([0.0, 0.0, 1.0]), terrain=surface)
    assert camera.position_m[0] == 0.0
    assert np.allclose(camera.velocity_mps[[0, 2]], 0.0)


def test_walk_mode_cannot_be_enabled_while_over_water() -> None:
    camera = FreeCamera(position_m=np.array([0.0, 5.0, 0.0], dtype=np.float32))
    assert not camera.set_walking(True, _flat_surface(water=True))
    assert not camera.walking
