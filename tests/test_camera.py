import numpy as np

from simulator.camera import CameraConfig, FreeCamera


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

