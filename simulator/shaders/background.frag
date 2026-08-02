#version 330
in vec2 uv; out vec4 frag_color;
uniform vec3 camera_forward; uniform vec3 camera_right; uniform vec3 camera_up;
uniform float tan_half_fov; uniform float aspect;
uniform vec3 sun_direction; uniform vec3 moon_direction;
uniform float twilight_strength; uniform float moon_strength;
uniform float cloud_cover;
uniform float time_s;
uniform vec2 wind_xz;
uniform sampler2D cloud_noise;
uniform sampler2D star_catalogue;
uniform vec3 celestial_row_x;
uniform vec3 celestial_row_y;
uniform vec3 celestial_row_z;
uniform float star_optical_depth;
// 0 renders the sky as seen. 1 renders the airlight source field the haze pass
// reads: the same radiance model evaluated along the horizontal, because the
// air on a near-ground path is lit by the horizon sky. That is the radiance
// Koschmieder's 2% contrast threshold — and therefore the visibility this
// scene is driven by — is defined against, so using it here makes aerial
// perspective saturate to exactly the sky an infinitely distant object shows.
// Point sources are excluded: a star is direct beam, not diffuse airlight.
uniform float airlight_field;
const float PI = 3.14159265359;
void main() {
    vec2 screen = uv * 2.0 - 1.0;
    vec3 ray = normalize(camera_forward
                       + camera_right * screen.x * aspect * tan_half_fov
                       + camera_up * screen.y * tan_half_fov);
    // Projecting onto the horizontal preserves azimuth exactly, so the
    // western twilight and the moon still brighten the haze on their own side
    // of the scene. Straight up has no azimuth and no geometry along it, so
    // the degenerate case falls back to an arbitrary bearing.
    vec3 horizontal = length(ray.xz) > 1e-4
                    ? normalize(vec3(ray.x, 0.0, ray.z))
                    : vec3(1.0, 0.0, 0.0);
    ray = airlight_field > .5 ? horizontal : ray;
    float altitude = asin(clamp(ray.y, -1.0, 1.0));
    // The horizon fade darkens the lower screen before the land mesh covers
    // it; it is a compositing mask, not part of the radiance model, so the
    // airlight field is evaluated without it.
    float above_horizon = max(smoothstep(-.025, .06, ray.y), airlight_field);
    float zenith = smoothstep(-.02, .75, ray.y);
    // Calibrated against dark, smoke-free crops of three 2024-10-05 event
    // photographs.  Even the darkest reference retains blue urban airglow;
    // the former values rendered it almost neutral black through the camera.
    vec3 night = mix(vec3(.00090, .00120, .00220),
                     vec3(.000140, .000220, .000620), zenith);
    float sun_alignment = max(dot(ray, sun_direction), 0.0);
    float western_twilight = pow(sun_alignment, 24.0) * twilight_strength;
    vec3 twilight_color = mix(vec3(.0028, .00072, .00020),
                              vec3(.00042, .00075, .0015), zenith);
    float urban_horizon = exp(-max(altitude, 0.0) * 7.5)
                        * (.00032 + cloud_cover * .00070);
    float moon_disc = smoothstep(cos(radians(.31)), cos(radians(.24)),
                                 dot(ray, moon_direction)) * moon_strength;
    vec3 clear_sky = night
             + twilight_color * western_twilight * (1.0 - cloud_cover * .45)
             + vec3(.00045, .00036, .00024) * urban_horizon
             + vec3(.72, .80, 1.0) * moon_disc;
    float cloud_density = 0.0;
    if (ray.y > .018 && cloud_cover > .002) {
        vec2 advection = wind_xz * time_s * .00016;
        vec2 layer0 = ray.xz / max(ray.y, .065) * .19 + advection;
        float coverage_bias = mix(.82, .28, cloud_cover);
        float density_sample = texture(cloud_noise, layer0 * .085).r;
        float low = smoothstep(coverage_bias, 1.0, density_sample);
        cloud_density = clamp(low, 0.0, 1.0);
    }
    float cloud_optical_depth = cloud_density * mix(1.2, 7.0, cloud_cover);
    float cloud_transmission = exp(-cloud_optical_depth);
    float moon_forward = pow(max(dot(ray, moon_direction), 0.0), 12.0);
    vec3 cloud_scatter = (
        vec3(.0035, .0032, .0028) * (.26 + urban_horizon * 180.0)
        + vec3(.009, .010, .012) * moon_strength * (0.18 + moon_forward)
        + twilight_color * western_twilight * .35
    ) * (1.0 - cloud_transmission);
    vec3 sky = clear_sky * cloud_transmission + cloud_scatter;

    // The catalogue is stored in J2000 equatorial coordinates, so the sky
    // rotates by transforming the view ray rather than by re-rasterising the
    // texture. Rows are passed separately to keep the layout unambiguous.
    vec3 celestial = vec3(
        dot(celestial_row_x, ray),
        dot(celestial_row_y, ray),
        dot(celestial_row_z, ray)
    );
    // u wraps through the repeat mode, matching right ascension / 360.
    vec2 celestial_uv = vec2(
        atan(celestial.y, celestial.x) / (2.0 * PI),
        asin(clamp(celestial.z, -1.0, 1.0)) / PI + .5
    );
    vec3 catalogue_flux = texture(star_catalogue, celestial_uv).rgb;
    float magnitude_flux = max(
        max(catalogue_flux.r, catalogue_flux.g), catalogue_flux.b
    );
    float air_mass = 1.0 / max(ray.y + .025 * exp(-11.0 * ray.y), .035);
    // Zenith-relative extinction. The optical depth is the Rayleigh plus
    // aerosol depth computed on the CPU from the observed pressure, replacing
    // a hardcoded 0.12. The relative form is kept deliberately so the
    // calibrated star_radiance scale below retains its meaning; the absolute
    // zenith extinction is reported by the validation harness instead.
    float atmosphere_transmission =
        exp(-star_optical_depth * max(air_mass - 1.0, 0.0));
    // Flux is reconstructed over a finite raster footprint; this scale keeps
    // all but the brightest sources below Seoul's urban background.
    float star_radiance = .00035 * magnitude_flux
                        * atmosphere_transmission * cloud_transmission
                        * (1.0 - airlight_field);
    float background_radiance = max(
        dot(sky, vec3(.2126, .7152, .0722)), 1e-7
    );
    float contrast = star_radiance / background_radiance;
    float visibility = smoothstep(.10, .28, contrast);
    vec3 star_color = catalogue_flux / max(magnitude_flux, 1e-7);
    sky += star_color * star_radiance * visibility * above_horizon;
    vec3 below = vec3(.000025, .000035, .000055);
    frag_color = vec4(mix(below, sky, above_horizon), 1);
}
