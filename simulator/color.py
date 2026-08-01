from __future__ import annotations

import numpy as np

# CIE XYZ to linear sRGB (IEC 61966-2-1), D65 white point.
_XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570],
    ],
    dtype=np.float64,
)


def _piecewise_gaussian(
    wavelength_nm: float, peak_nm: float, sigma_low: float, sigma_high: float
) -> float:
    sigma = sigma_low if wavelength_nm < peak_nm else sigma_high
    t = (wavelength_nm - peak_nm) / sigma
    return float(np.exp(-0.5 * t * t))


def cie_xyz_at_wavelength(wavelength_nm: float) -> np.ndarray:
    """CIE 1931 colour matching functions at a single wavelength.

    Uses the multi-lobe Gaussian analytic fit of Wyman, Sloan and Shirley,
    "Simple Analytic Approximations to the CIE XYZ Color Matching Functions",
    Journal of Computer Graphics Techniques 2(2), 2013. The fit is accurate to
    well under a percent of peak across the visible band, which is far tighter
    than the uncertainty in any firework emission parameter this project holds.
    """

    x = (
        1.056 * _piecewise_gaussian(wavelength_nm, 599.8, 37.9, 31.0)
        + 0.362 * _piecewise_gaussian(wavelength_nm, 442.0, 16.0, 26.7)
        - 0.065 * _piecewise_gaussian(wavelength_nm, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _piecewise_gaussian(
        wavelength_nm, 568.8, 46.9, 40.5
    ) + 0.286 * _piecewise_gaussian(wavelength_nm, 530.9, 16.3, 31.1)
    z = 1.217 * _piecewise_gaussian(
        wavelength_nm, 437.0, 11.8, 36.0
    ) + 0.681 * _piecewise_gaussian(wavelength_nm, 459.0, 26.0, 13.8)
    return np.array([x, y, z], dtype=np.float64)


def wavelength_rgb(wavelength_nm: float) -> np.ndarray:
    """Linear RGB chromaticity of a narrow emission line.

    Coloured pyrotechnic stars radiate in narrow bands rather than as a black
    body, so a colour temperature cannot describe them. This maps a dominant
    emission wavelength — an abstract optical parameter — to linear RGB. It
    carries no information about how such an emission is produced.

    The result is normalised so the strongest channel is 1.0, matching the
    convention :func:`blackbody_rgb` already uses. Radiant power is a separate
    quantity, solved from the star's energy budget in
    :func:`simulator.lighting.combustion_peak_radiant_power_w`, so this vector
    carries hue only.

    Limitation: a three-channel representation cannot express that a deep blue
    line delivers far less luminance per watt than a green one. Correcting that
    requires a spectral renderer, not a different normalisation here.
    """

    xyz = cie_xyz_at_wavelength(float(wavelength_nm))
    linear = _XYZ_TO_LINEAR_SRGB @ xyz
    # Wavelengths outside the sRGB gamut produce negative components; clamping
    # projects them onto the nearest reproducible colour rather than inverting
    # the hue.
    linear = np.clip(linear, 0.0, None)
    peak = float(linear.max())
    if peak <= 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (linear / peak).astype(np.float32)


PLANCK_CONSTANT_J_S = 6.626_070_15e-34
BOLTZMANN_CONSTANT_J_K = 1.380_649e-23
SPEED_OF_LIGHT_M_S = 299_792_458.0

VISIBLE_BAND_NM = (360.0, 830.0)
"""Integration limits of the CIE 1931 observer."""


def planckian_spectral_radiance(
    wavelength_nm: float | np.ndarray, temperature_k: float
) -> np.ndarray:
    """Planck's law, in W/(m^3 sr).

    The absolute scale cancels in every use here — only the shape of the
    spectrum matters for a chromaticity — but it is left unnormalised so the
    function is the law rather than a rescaling of it.
    """

    metres = np.asarray(wavelength_nm, dtype=np.float64) * 1e-9
    exponent = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S / (
        metres * BOLTZMANN_CONSTANT_J_K * max(temperature_k, 1.0)
    )
    return (
        2.0 * PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_M_S**2
        / metres**5
        / np.expm1(exponent)
    )


def planckian_linear_srgb(
    temperature_k: float, step_nm: float = 1.0
) -> np.ndarray:
    """Linear sRGB of a Planckian radiator, normalised so the peak is 1.

    Planck's law integrated against the CIE 1931 colour matching functions and
    converted with the standard sRGB matrix. Both inputs are published
    standards, which is why this is preferred over :func:`blackbody_rgb` for
    anything a calibration depends on: that function is a convenient curve fit
    to the same locus and differs from this integration by a few percent.

    This is the reference the camera's white balance is defined against — the
    scene colour that a camera balanced at ``temperature_k`` renders neutral.
    """

    wavelengths = np.arange(
        VISIBLE_BAND_NM[0], VISIBLE_BAND_NM[1] + step_nm, step_nm
    )
    matching = np.array(
        [cie_xyz_at_wavelength(float(value)) for value in wavelengths]
    )
    radiance = planckian_spectral_radiance(wavelengths, temperature_k)
    xyz = (matching * radiance[:, None]).sum(axis=0)
    linear = _XYZ_TO_LINEAR_SRGB @ (xyz / max(xyz[1], 1e-30))
    # Very cool or very warm radiators fall outside the sRGB gamut; clamping
    # projects onto the nearest reproducible colour rather than inverting a
    # channel, matching what wavelength_rgb already does.
    linear = np.clip(linear, 0.0, None)
    return (linear / max(float(linear.max()), 1e-30)).astype(np.float64)


def color_temperature_from_bv(color_index_bv: float | np.ndarray) -> np.ndarray:
    """Stellar effective temperature from a Johnson B-V colour index.

    Uses the Ballesteros (2012) relation, EPL 97(3):

        T = 4600 K * (1 / (0.92 (B-V) + 1.7) + 1 / (0.92 (B-V) + 0.62))

    It reproduces the Sun (B-V = 0.65) at 5778 K and an A0V star (B-V = 0) near
    10 000 K, which is why a catalogue's colour index can drive the existing
    black-body path directly instead of an invented colour.
    """

    bv = np.clip(np.asarray(color_index_bv, dtype=np.float64), -0.4, 2.0)
    return 4600.0 * (
        1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62)
    )


def blackbody_rgb(temperature_k: float) -> np.ndarray:
    """Approximate a black-body chromaticity as a linear RGB triplet.

    This is used as a physically meaningful baseline. Spectral emission curves
    for individual pyrotechnic compositions can replace it without changing
    the renderer interface.
    """

    t = float(np.clip(temperature_k, 1_000.0, 40_000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0

    srgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 0.0, 1.0)
    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)

