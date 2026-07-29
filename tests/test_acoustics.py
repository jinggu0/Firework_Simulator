import numpy as np

from simulator.acoustics import FireworkAcoustics, speed_of_sound_mps
from simulator.config import AcousticConfig, AtmosphereConfig


def test_sound_speed_responds_to_temperature_and_humidity() -> None:
    cold_dry = AtmosphereConfig(
        temperature_k=273.15, relative_humidity=0.0
    )
    warm_humid = AtmosphereConfig(
        temperature_k=303.15, relative_humidity=0.8
    )
    assert 330.0 < speed_of_sound_mps(cold_dry) < 333.0
    assert speed_of_sound_mps(warm_humid) > speed_of_sound_mps(cold_dry) + 17.0


def test_strong_shock_transitions_to_acoustic_propagation() -> None:
    atmosphere = AtmosphereConfig()
    energy_j = 255_000.0
    transition_s, transition_radius_m = (
        FireworkAcoustics.strong_shock_transition(energy_j, atmosphere)
    )
    early = FireworkAcoustics.shock_state(
        transition_s * 0.25, energy_j, atmosphere
    )
    late = FireworkAcoustics.shock_state(
        transition_s * 2.0, energy_j, atmosphere
    )
    assert early.strong_shock
    assert not late.strong_shock
    assert early.radius_m < transition_radius_m < late.radius_m


def test_downwind_sound_arrives_earlier_and_pressure_spreads_by_range() -> None:
    config = AcousticConfig(atmospheric_absorption_db_m=0.0)
    model = FireworkAcoustics(config, 5)
    source = np.array([0.0, 100.0, 0.0], dtype=np.float32)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    downwind_listener = np.array([200.0, 100.0, 0.0], dtype=np.float32)
    windy = AtmosphereConfig(
        wind_velocity_mps=(10.0, 0.0, 0.0),
        wind_velocity_100m_mps=(10.0, 0.0, 0.0),
    )
    calm = AtmosphereConfig(
        wind_velocity_mps=(0.0, 0.0, 0.0),
        wind_velocity_100m_mps=(0.0, 0.0, 0.0),
    )
    downwind = model.predict_arrival(
        source, 255_000.0, downwind_listener, right, windy
    )
    calm_near = model.predict_arrival(
        source, 255_000.0, downwind_listener, right, calm
    )
    calm_far = model.predict_arrival(
        source,
        255_000.0,
        np.array([400.0, 100.0, 0.0], dtype=np.float32),
        right,
        calm,
    )
    assert downwind.propagation_delay_s < calm_near.propagation_delay_s
    assert calm_near.rms_pressure_pa > calm_far.rms_pressure_pa * 2.0
    assert 0.5 < calm_near.propagation_delay_s < 0.7


def test_wavefront_is_emitted_only_after_propagation_delay() -> None:
    model = FireworkAcoustics(AcousticConfig(), 23)
    atmosphere = AtmosphereConfig(
        wind_velocity_mps=(0.0, 0.0, 0.0),
        wind_velocity_100m_mps=(0.0, 0.0, 0.0),
    )
    source = np.array([0.0, 100.0, 0.0], dtype=np.float32)
    listener = np.array([0.0, 100.0, 200.0], dtype=np.float32)
    right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    predicted = model.predict_arrival(
        source, 255_000.0, listener, right, atmosphere
    )
    model.emit(source, 255_000.0)
    assert model.update(
        predicted.propagation_delay_s - 0.001,
        listener,
        right,
        atmosphere,
    ) == []
    arrivals = model.update(0.002, listener, right, atmosphere)
    assert len(arrivals) == 1
    assert np.isclose(
        arrivals[0].propagation_delay_s,
        predicted.propagation_delay_s,
        rtol=1e-6,
    )


def test_pcm_synthesis_is_deterministic_stereo_and_bounded() -> None:
    config = AcousticConfig(sample_rate_hz=8_000, reverberation_time_s=0.25)
    model = FireworkAcoustics(config, 17)
    arrival = model.predict_arrival(
        np.array([0.0, 100.0, 0.0], dtype=np.float32),
        255_000.0,
        np.array([150.0, 20.0, 0.0], dtype=np.float32),
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        AtmosphereConfig(),
    )
    first = model.synthesize_pcm(arrival)
    second = model.synthesize_pcm(arrival)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2_000, 2)
    assert first.dtype == np.int16
    assert np.max(np.abs(first.astype(np.int32))) <= 32767
    assert not np.array_equal(first[:, 0], first[:, 1])
