// Atmospheric extinction along a path, shared by every stage that needs it.
//
// The renderer previously carried one unsourced scalar, air_extinction_per_m =
// 0.00012, which implies 32.6 km of visibility — roughly twice what the
// event's own weather record implies once the aerosol is grown by the observed
// humidity. These coefficients are instead the column optical depths of
// simulator/atmosphere.py divided by the scale height of the species that
// produced them, so the air that dims the stars is the same air that hazes the
// skyline. simulator.atmosphere.SurfaceExtinction.uniforms() names them and
// SurfaceExtinction.transmittance() is the CPU reference this must reproduce.
//
// Units: extinction per metre, heights and lengths in metres. Channels are the
// three wavelengths PhysicalCameraConfig defines its quantum efficiency at, so
// haze reddens distant lights by the amount the Angstrom exponent says it
// should rather than by a chosen tint.

uniform vec3 aerosol_extinction_per_m;
uniform vec3 molecular_extinction_per_m;
uniform float aerosol_scale_height_m;
uniform float molecular_scale_height_m;

// Mean of exp(-z/H) over a segment whose height runs linearly from start to
// end, which is the closed form of the integral divided by the path length.
// Heights are magnitudes so a mirrored reflection path, whose image point sits
// below the datum, integrates the air it actually crosses.
float height_profile_mean(float scale_height_m, float start_m, float end_m) {
    float low = abs(start_m);
    float high = abs(end_m);
    float rise = high - low;
    float at_start = exp(-low / scale_height_m);
    // The quotient is singular at zero rise; a level path is its own limit.
    if (abs(rise) < 1e-3) return at_start;
    return scale_height_m * (at_start - exp(-high / scale_height_m)) / rise;
}

vec3 air_optical_depth(
    float start_height_m, float end_height_m, float path_length_m
) {
    return max(path_length_m, 0.0) * (
        aerosol_extinction_per_m
            * height_profile_mean(
                aerosol_scale_height_m, start_height_m, end_height_m
            )
        + molecular_extinction_per_m
            * height_profile_mean(
                molecular_scale_height_m, start_height_m, end_height_m
            )
    );
}

vec3 air_transmittance(
    float start_height_m, float end_height_m, float path_length_m
) {
    return exp(
        -air_optical_depth(start_height_m, end_height_m, path_length_m)
    );
}
