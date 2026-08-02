#version 330
in vec3 world_position;
in vec3 world_normal;
in float surface;
in vec2 surface_uv;
in float facade_style;
uniform vec3 camera_position;
uniform float ambient_irradiance_w_m2;
uniform float window_radiance_w_m2_sr;
uniform float time_s;
uniform vec2 wind_xz;
uniform float wind_speed_mps;
uniform int static_light_count;
uniform vec3 static_light_position[4];
uniform vec3 static_light_color;
uniform float static_light_power_w;
uniform int dynamic_light_count;
uniform vec3 dynamic_light_position[8];
uniform vec3 dynamic_light_color[8];
uniform float dynamic_light_power_w[8];

// Material table, one row per surface code. See simulator/materials.py.
uniform vec3 material_base_primary[17];
uniform vec3 material_base_secondary[17];
uniform vec4 material_pattern[17];      // kind, scale u, scale v, mix
uniform vec4 material_reflectance[17];  // roughness, metallic, ao, transmission
uniform vec4 material_emissive[17];     // rgb, scale
uniform vec2 material_relief[17];       // normal strength, height scale

out vec4 frag_color;
#include "air_extinction.glsl"
const float PI = 3.14159265359;

const int PATTERN_UNIFORM = 0;
const int PATTERN_HASH_CELL = 1;
const int PATTERN_HASH_UV = 2;
const int PATTERN_HASH_VOLUME = 3;
const int PATTERN_GRID = 4;
const int PATTERN_PANELS = 5;
const int PATTERN_STRIPE = 6;
const int PATTERN_JOINTS = 7;
const int PATTERN_LANE = 8;
const int PATTERN_FRESNEL = 9;
const int PATTERN_FACADE = 10;
// Normal-incidence reflectance of a dielectric near an index of 1.5.
const float DIELECTRIC_NORMAL_REFLECTANCE = 0.04;
// Radiance a fully transmissive surface adds when backlit, in W/(m2 sr).
// Calibrated so grass blades match their previous appearance at the 0.35
// transmission the material table gives them.
const float TRANSMISSION_RADIANCE = 0.00018 / 0.35;
// Converts a per-pixel pattern gradient into a normal tilt. Purely a scaling
// choice: without the surface footprint in metres the height channel cannot be
// converted to a slope, so this sets how pronounced a seam reads at all.
const float RELIEF_GRADIENT_TO_NORMAL = 64.0;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float interval(float value, float lower, float upper, float antialias) {
    return smoothstep(lower - antialias, lower + antialias, value)
         * (1.0 - smoothstep(upper - antialias, upper + antialias, value));
}
// Trowbridge-Reitz GGX with the Smith height-correlated visibility and a
// Schlick Fresnel, matching the formulation the water pass already uses so the
// river and the city respond to a burst the same way.
vec3 specular_brdf(
    vec3 n, vec3 view_direction, vec3 light_direction,
    float roughness, vec3 normal_reflectance
) {
    float n_dot_l = max(dot(n, light_direction), 0.0);
    float n_dot_v = max(dot(n, view_direction), 1e-4);
    if (n_dot_l <= 0.0) return vec3(0.0);
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    vec3 half_vector = normalize(view_direction + light_direction);
    float n_dot_h = max(dot(n, half_vector), 0.0);
    float v_dot_h = max(dot(view_direction, half_vector), 0.0);
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    float distribution = alpha_squared
                       / max(PI * denominator * denominator, 1e-5);
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float geometry_v = n_dot_v / mix(n_dot_v, 1.0, k);
    float geometry_l = n_dot_l / mix(n_dot_l, 1.0, k);
    vec3 fresnel = normal_reflectance
        + (vec3(1.0) - normal_reflectance) * pow(1.0 - v_dot_h, 5.0);
    return fresnel * distribution * geometry_v * geometry_l
         / max(4.0 * n_dot_v * n_dot_l, 1e-5);
}

// reflectance is (roughness, metallic, ambient occlusion, transmission).
vec3 reflected_radiance(vec3 n, vec3 albedo, vec4 reflectance) {
    float roughness = clamp(reflectance.x, .03, 1.0);
    float metallic = reflectance.y;
    // A conductor has no diffuse lobe and tints its specular reflection with
    // its own colour; a dielectric reflects a few percent achromatically.
    vec3 normal_reflectance = mix(
        vec3(DIELECTRIC_NORMAL_REFLECTANCE), albedo, metallic
    );
    vec3 diffuse_albedo = albedo * (1.0 - metallic);
    vec3 view_direction = normalize(camera_position - world_position);

    // Ambient sky light is the term occlusion applies to: it arrives from the
    // whole hemisphere, so a recessed surface receives less of it.
    vec3 result = diffuse_albedo * ambient_irradiance_w_m2
                * reflectance.z / PI;
    for (int i = 0; i < 4; ++i) {
        if (i >= static_light_count) break;
        vec3 displacement = static_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), .5);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float down_lobe = pow(max(light_direction.y, 0.0), 3.0);
        vec3 irradiance = static_light_power_w * 2.0 * down_lobe
                        * air_transmittance(
                              static_light_position[i].y, world_position.y,
                              distance_m
                          )
                        / (PI * distance_squared);
        float n_dot_l = max(dot(n, light_direction), 0.0);
        vec3 brdf = diffuse_albedo / PI
            + specular_brdf(
                n, view_direction, light_direction,
                roughness, normal_reflectance
            );
        result += static_light_color * irradiance * brdf * n_dot_l;
    }
    for (int i = 0; i < 8; ++i) {
        if (i >= dynamic_light_count) break;
        vec3 displacement = dynamic_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), 1.0);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        vec3 irradiance = dynamic_light_power_w[i]
                        * air_transmittance(
                              dynamic_light_position[i].y, world_position.y,
                              distance_m
                          )
                        / (4.0 * PI * distance_squared);
        float n_dot_l = max(dot(n, light_direction), 0.0);
        vec3 brdf = diffuse_albedo / PI
            + specular_brdf(
                n, view_direction, light_direction,
                roughness, normal_reflectance
            );
        result += dynamic_light_color[i] * irradiance * brdf * n_dot_l;
    }
    return result;
}

// Blend factor between a material's two base colours. Each form is the one the
// surface previously carried inline; naming them turns a branch into a row.
float surface_pattern(int kind, vec2 scale, vec3 n) {
    if (kind == PATTERN_HASH_CELL) {
        return hash21(floor(world_position.xz * scale));
    }
    if (kind == PATTERN_HASH_UV) {
        return hash21(floor(surface_uv * scale));
    }
    if (kind == PATTERN_HASH_VOLUME) {
        return hash21(floor(world_position.xz * scale.x + world_position.y));
    }
    if (kind == PATTERN_GRID) {
        return step(.08, abs(fract(surface_uv.x * scale.x) - .5))
             * step(.08, abs(fract(surface_uv.y * scale.y) - .5));
    }
    if (kind == PATTERN_PANELS) {
        return step(.035, min(
            abs(fract(surface_uv.x * scale.x) - .5),
            abs(fract(surface_uv.y * scale.y) - .5)
        ));
    }
    if (kind == PATTERN_STRIPE) {
        return 1.0 - smoothstep(
            .025, .055, abs(fract(surface_uv.x * scale.x) - .5)
        );
    }
    if (kind == PATTERN_JOINTS) {
        vec2 joints = abs(fract(surface_uv * scale) - .5);
        return 1.0 - smoothstep(.025, .055, min(joints.x, joints.y));
    }
    if (kind == PATTERN_LANE) {
        return smoothstep(.46, .50, abs(fract(
            world_position.x * scale.x + world_position.z * scale.y
        ) - .5));
    }
    if (kind == PATTERN_FRESNEL) {
        return pow(
            1.0 - max(dot(n, normalize(camera_position - world_position)), 0.0),
            3.0
        );
    }
    return 0.0;
}

void main() {
    vec3 n = normalize(world_normal);
    int material = int(clamp(surface + .5, 0.0, 16.0));
    vec4 pattern = material_pattern[material];
    int kind = int(pattern.x + .5);
    vec4 reflectance = material_reflectance[material];
    vec4 emissive = material_emissive[material];
    vec2 relief = material_relief[material];

    if (kind != PATTERN_FACADE) {
        // Grass sward carries a wind-driven travelling wave in its normal;
        // the material's normal strength scales it.
        if (relief.x > 0.0) {
            vec2 wind_direction = length(wind_xz) > .01
                                ? normalize(wind_xz) : vec2(1.0, 0.0);
            float ripple = sin(
                dot(world_position.xz, wind_direction) * 6.5
                - time_s * (2.2 + wind_speed_mps * .35)
            );
            vec2 cross_wind = vec2(-wind_direction.y, wind_direction.x);
            ripple += .42 * sin(
                dot(world_position.xz, cross_wind) * 9.7 - time_s * 3.1
            );
            n = normalize(
                n + vec3(wind_direction.x, 0.0, wind_direction.y)
                  * ripple * min(.055 + wind_speed_mps * .006, .10)
                  * relief.x
            );
        }
        float pattern_value = surface_pattern(kind, pattern.yz, n);
        // Height reads the pattern as relief. The gradient is taken in screen
        // space, so this is a bump normal rather than true displacement: it
        // has no parallax and its strength varies with viewing distance. A
        // real height channel needs the surface footprint per pixel, which the
        // vertex stage does not currently carry.
        if (relief.y > 0.0) {
            n = normalize(
                n - vec3(dFdx(pattern_value), 0.0, dFdy(pattern_value))
                  * relief.y * RELIEF_GRADIENT_TO_NORMAL
            );
        }
        vec3 albedo = mix(
            material_base_primary[material],
            material_base_secondary[material],
            pattern_value * pattern.w
        );
        vec3 radiance = reflected_radiance(n, albedo, reflectance);
        if (material == 2) {
            // OSM supplies the bridge decks but not a reliable 2024 fixture
            // inventory.  The upload pass converts their UVs into distance
            // along deck and signed edge position.  A low continuous edge
            // glow plus 32 m LED points reproduces the line and elongated
            // river reflections visible in event photographs without adding
            // hundreds of point lights or draw calls.
            float edge = smoothstep(.72, .94, abs(surface_uv.y));
            float period = 32.0;
            float phase = abs(fract(surface_uv.x / period + .5) - .5);
            float footprint = max(fwidth(surface_uv.x / period), .002);
            float fixture = 1.0 - smoothstep(
                .018 + footprint, .050 + footprint * 2.0, phase
            );
            vec3 led_tint = vec3(1.0, .50, .16);
            radiance += led_tint * window_radiance_w_m2_sr * edge
                      * (.025 + fixture * .42);
        }
        // Thin surfaces lit from behind. The wind direction stands in for the
        // dominant backlight until a directional sky model supplies one.
        if (reflectance.w > 0.0) {
            float translucency = pow(
                max(dot(normalize(wind_xz.x == 0.0 && wind_xz.y == 0.0
                    ? vec3(1.0, .2, 0.0)
                    : vec3(wind_xz.x, .2, wind_xz.y)),
                    -n), 0.0), 2.0
            ) * min(wind_speed_mps / 5.0, 1.0);
            radiance += albedo * translucency
                      * reflectance.w * TRANSMISSION_RADIANCE;
        }
        radiance += emissive.rgb * window_radiance_w_m2_sr * emissive.w;
        frag_color = vec4(radiance, 1.0);
        return;
    }

    if (surface > .5) {
        vec3 roof = material_base_primary[material];
        if (facade_style > 1.5 && facade_style < 2.5) {
            roof = vec3(.30, .18, .055);
        } else if (facade_style > 4.5 && facade_style < 5.5) {
            roof = vec3(.31, .035, .025);
        }
        float roof_variation = hash21(floor(surface_uv * .08)) * .025;
        frag_color = vec4(
            reflected_radiance(n, roof + roof_variation, reflectance), 1.0
        );
        return;
    }

    float bay_width = 4.2;
    float floor_height = 3.25;
    vec4 pane_bounds = vec4(.13, .82, .18, .72);
    vec3 wall = vec3(.18, .20, .23);
    vec3 glass = vec3(.025, .042, .075);
    float glass_amount = .18;
    float occupancy_threshold = .68;
    if (facade_style > .5 && facade_style < 1.5) {
        bay_width = 3.6; floor_height = 4.57;
        pane_bounds = vec4(.05, .95, .08, .92);
        wall = vec3(.055, .09, .14);
        glass = vec3(.02, .055, .12);
        glass_amount = .88; occupancy_threshold = .72;
    } else if (facade_style > 1.5 && facade_style < 2.5) {
        bay_width = 3.25; floor_height = 4.0;
        pane_bounds = vec4(.04, .96, .06, .93);
        wall = vec3(.32, .21, .06);
        glass = vec3(.16, .085, .018);
        glass_amount = .93; occupancy_threshold = .75;
    } else if (facade_style > 2.5 && facade_style < 3.5) {
        bay_width = 3.4; floor_height = 3.05;
        pane_bounds = vec4(.18, .78, .24, .68);
        wall = vec3(.25, .22, .18);
        glass = vec3(.025, .04, .06);
        glass_amount = .28; occupancy_threshold = .55;
    } else if (facade_style > 3.5 && facade_style < 4.5) {
        bay_width = 5.4; floor_height = 3.8;
        pane_bounds = vec4(.22, .76, .28, .69);
        wall = vec3(.34, .31, .25);
        glass_amount = .12; occupancy_threshold = .82;
    } else if (facade_style > 4.5 && facade_style < 5.5) {
        bay_width = 3.8; floor_height = 4.65;
        pane_bounds = vec4(.05, .95, .08, .92);
        wall = vec3(.045, .07, .105);
        glass = vec3(.018, .035, .075);
        glass_amount = .90; occupancy_threshold = .72;
    } else if (facade_style > 5.5) {
        bay_width = 3.2; floor_height = 3.55;
        pane_bounds = vec4(.12, .86, .16, .78);
        wall = vec3(.19, .18, .17);
        glass = vec3(.025, .038, .055);
        glass_amount = .55; occupancy_threshold = .38;
    }

    vec2 grid = surface_uv / vec2(bay_width, floor_height);
    vec2 within = fract(grid);
    vec2 aa = min(fwidth(grid) * 1.25, vec2(.12));
    float pane = interval(within.x, pane_bounds.x, pane_bounds.y, aa.x)
               * interval(within.y, pane_bounds.z, pane_bounds.w, aa.y);
    vec2 cell = floor(grid);
    float occupancy_sample = hash21(cell + facade_style * 31.7);
    float occupied = step(occupancy_threshold, occupancy_sample);
    // Offices and apartments do not present identical luminous rectangles.
    // Keeping the hash beyond 1.6 km removes the uniform white grid that the
    // previous .38 fallback produced; the floor/cell intensity variation also
    // survives in the planar reflection.
    float room_dimmer = mix(
        .24, 1.0, hash21(cell * vec2(1.31, 2.17) + facade_style * 9.1)
    );
    float floor_blackout = step(
        .12, hash21(vec2(floor(cell.y / 3.0), facade_style * 13.7))
    );
    occupied *= room_dimmer * floor_blackout;
    float temperature = hash21(cell + facade_style * 17.0 + 11.3);
    vec3 window_color = mix(
        vec3(1.0, .42, .12), vec3(.55, .72, 1.0), temperature
    );
    vec3 view_direction = normalize(camera_position - world_position);
    float fresnel = pow(1.0 - max(dot(n, view_direction), 0.0), 4.0);
    vec3 facade = mix(wall, glass, pane * glass_amount);
    facade += glass * fresnel * glass_amount * .35;
    float mullion = 1.0 - smoothstep(
        .018, .055, min(abs(within.x-pane_bounds.x),
                        abs(within.x-pane_bounds.y))
    );
    mullion += 1.0 - smoothstep(
        .018, .055, min(abs(within.y-pane_bounds.z),
                        abs(within.y-pane_bounds.w))
    );
    facade = mix(facade, vec3(.035, .040, .045), clamp(mullion, 0.0, 1.0));
    float weathering = hash21(
        floor(surface_uv / vec2(18.0, 26.0)) + facade_style * 7.3
    );
    facade *= mix(.94, 1.035, weathering);
    vec3 emission = window_color * pane * occupied
                  * window_radiance_w_m2_sr * .72;

    if (facade_style > 2.5 && facade_style < 3.5) {
        float balcony = 1.0 - smoothstep(
            .035, .09, min(within.y, 1.0 - within.y)
        );
        facade += balcony * vec3(.10, .095, .086);
    }
    if (facade_style > 4.5 && facade_style < 5.5) {
        float column_mod = mod(surface_uv.x, 18.0);
        float column_distance = min(column_mod, 18.0 - column_mod);
        float red_column = 1.0 - smoothstep(.42, .86, column_distance);
        float beam_mod = mod(surface_uv.y, 32.0);
        float beam_distance = min(beam_mod, 32.0 - beam_mod);
        float red_beam = 1.0 - smoothstep(.30, .72, beam_distance);
        facade = mix(
            facade,
            vec3(.38, .018, .009),
            max(red_column, red_beam) * .92
        );
    }
    frag_color = vec4(
        reflected_radiance(n, facade, reflectance) + emission, 1.0
    );
}
