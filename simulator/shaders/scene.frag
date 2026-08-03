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
uniform vec3 material_base_primary[19];
uniform vec3 material_base_secondary[19];
uniform vec4 material_pattern[19];      // kind, scale u, scale v, mix
uniform vec4 material_reflectance[19];  // roughness, metallic, ao, transmission
uniform vec4 material_emissive[19];     // rgb, scale
uniform vec2 material_relief[19];       // normal strength, height scale
// Four CC0 photo-scanned PBR sets, packed as diffuse/normal/ARM layers.
uniform sampler2DArray scanned_material_texture;
uniform vec3 scanned_diffuse_mean[4];
uniform float scanned_texture_width_m[4];

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
// Current grade-D road-paint contract. The V coordinates are normalized road
// width, so the audit converts these values back to metres per rendered quad.
const float ROAD_MARKING_ENCODING_BASE = 1024.0;
const float ROAD_MARKING_LANE_STRIDE = 64.0;
const float ROAD_LANE_LINE_WIDTH_M = .15;
const float ROAD_LANE_DASH_PAINT_M = 3.0;
const float ROAD_LANE_DASH_GAP_M = 3.0;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float value_noise(vec2 p) {
    vec2 cell = floor(p);
    vec2 local = fract(p);
    local = local * local * (3.0 - 2.0 * local);
    float a = hash21(cell);
    float b = hash21(cell + vec2(1.0, 0.0));
    float c = hash21(cell + vec2(0.0, 1.0));
    float d = hash21(cell + vec2(1.0, 1.0));
    return mix(mix(a, b, local.x), mix(c, d, local.x), local.y);
}
float fbm2(vec2 p) {
    float value = value_noise(p) * .68;
    value += value_noise(p * 2.07 + 19.3) * .32;
    return value;
}
float interval(float value, float lower, float upper, float antialias) {
    return smoothstep(lower - antialias, lower + antialias, value)
         * (1.0 - smoothstep(upper - antialias, upper + antialias, value));
}

// Derivative cotangent frame: works on roads, horizontal park surfaces and
// vertical facility walls without storing tangent vectors in the city mesh.
mat3 cotangent_frame(vec3 n, vec3 position, vec2 texture_uv) {
    vec3 dp1 = dFdx(position);
    vec3 dp2 = dFdy(position);
    vec2 duv1 = dFdx(texture_uv);
    vec2 duv2 = dFdy(texture_uv);
    vec3 dp2_perpendicular = cross(dp2, n);
    vec3 dp1_perpendicular = cross(n, dp1);
    vec3 tangent = dp2_perpendicular * duv1.x
                 + dp1_perpendicular * duv2.x;
    vec3 bitangent = dp2_perpendicular * duv1.y
                   + dp1_perpendicular * duv2.y;
    float inverse_scale = inversesqrt(max(
        max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-8
    ));
    return mat3(
        tangent * inverse_scale, bitangent * inverse_scale, n
    );
}

vec2 metric_material_uv(vec3 n) {
    vec3 axis = abs(n);
    if (axis.y >= max(axis.x, axis.z)) return world_position.xz;
    if (axis.x >= axis.z) return world_position.zy;
    return world_position.xy;
}

// Break exact tile alignment without changing the authored metre scale. The
// two cross-axis sine offsets are at most 0.16 m and their Jacobian terms are
// bounded by 0.0114 and 0.0091, so local scale changes stay below 1.2%. A
// separate non-periodic macro field varies energy rather than rescaling UVs.
vec2 scanned_antitile_metric_uv(int layer, vec2 metric_uv) {
    float phase = float(layer) * 2.39996323;
    vec2 warp_m = vec2(
        sin(metric_uv.y * .071 + phase),
        sin(metric_uv.x * .057 - phase * 1.37)
    ) * .16;
    return metric_uv + warp_m;
}

float scanned_macro_variation(int layer, vec2 metric_uv) {
    vec2 phase = vec2(float(layer) * 17.17, float(layer) * -11.31);
    return fbm2(metric_uv * vec2(.047, .039) + phase);
}

int scanned_layer_for_material(int material) {
    if (material == 3) return 0;                 // asphalt
    if (material == 5) return 1;                 // concrete pavers
    if (material == 4 || material == 16) return 2; // grass
    if (material == 2 || material == 12 || material == 17) return 3;
    return -1;
}

void apply_scanned_material(
    int layer, float detail, float colour_strength,
    inout vec3 n, inout vec3 albedo, inout vec4 reflectance
) {
    if (layer < 0 || detail <= .001) return;
    vec2 metric_uv = metric_material_uv(n);
    vec2 map_uv = scanned_antitile_metric_uv(layer, metric_uv)
                / scanned_texture_width_m[layer];
    float macro_variation = scanned_macro_variation(layer, metric_uv);
    float base_layer = float(layer * 3);
    vec3 scanned_albedo = pow(texture(
        scanned_material_texture, vec3(map_uv, base_layer)
    ).rgb, vec3(2.2));
    // Preserve the event-photo colour calibration; only the scan's relative
    // surface variation is transferred to this site.
    vec3 relative_albedo = clamp(
        scanned_albedo / max(scanned_diffuse_mean[layer], vec3(.015)),
        vec3(.38), vec3(2.25)
    );
    relative_albedo *= mix(.94, 1.06, macro_variation);
    albedo *= mix(
        vec3(1.0), relative_albedo,
        min(detail * colour_strength, 1.0)
    );

    vec3 tangent_normal = texture(
        scanned_material_texture, vec3(map_uv, base_layer + 1.0)
    ).xyz * 2.0 - 1.0;
    tangent_normal.xy *= mix(.48, .92, colour_strength);
    vec3 mapped_normal = normalize(
        cotangent_frame(n, world_position, map_uv) * tangent_normal
    );
    n = normalize(mix(n, mapped_normal, detail));

    // Poly Haven ARM maps are ambient occlusion, roughness, metallic.
    vec3 arm = texture(
        scanned_material_texture, vec3(map_uv, base_layer + 2.0)
    ).rgb;
    float varied_roughness = clamp(
        arm.g + (macro_variation - .5) * .055, .08, 1.0
    );
    reflectance.x = mix(
        reflectance.x, varied_roughness, detail * .78
    );
    reflectance.z *= mix(1.0, arm.r, detail * .42);
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
    // A low-frequency environment-specular term keeps smooth metal and glass
    // from becoming black cardboard between point lights. It is bounded by
    // the same measured/modelled ambient irradiance as the diffuse sky term.
    float n_dot_v = max(dot(n, view_direction), 0.0);
    vec3 grazing_fresnel = normal_reflectance
        + (vec3(1.0) - normal_reflectance) * pow(1.0 - n_dot_v, 5.0);
    float sky_visibility = mix(.18, 1.0, clamp(n.y * .5 + .5, 0.0, 1.0));
    result += grazing_fresnel * ambient_irradiance_w_m2 / PI
            * sky_visibility * (1.0 - roughness * .72);
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
    int material = int(clamp(surface + .5, 0.0, 18.0));
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
        // Metric procedural detail replaces flat colour at close range while
        // naturally averaging away in the distance. It is deterministic in
        // world space, so moving the camera never makes the surface swim.
        float camera_distance_m = length(camera_position - world_position);
        float micro_detail = 1.0 - smoothstep(90.0, 420.0, camera_distance_m);
        int scanned_layer = scanned_layer_for_material(material);
        float scanned_colour_strength = material == 4 || material == 16
                                      ? .56 : .74;
        apply_scanned_material(
            scanned_layer, micro_detail, scanned_colour_strength,
            n, albedo, reflectance
        );
        if (material == 3) {
            // Asphalt aggregate and evidence-gated metric lane dividers. The
            // upload pass supplies u=metres along the road and v=-1..1 across
            // it, plus packed width/lane count only for explicit one-way OSM
            // ways. Untagged and bidirectional asphalt receives no paint.
            float repair_field = .5;
            if (micro_detail > .001) {
                repair_field = value_noise(world_position.xz * .045);
            }
            albedo *= mix(.88, 1.06, smoothstep(
                .38, .68, mix(.5, repair_field, micro_detail)
            ));
            float marking = 0.0;
            if (facade_style >= ROAD_MARKING_ENCODING_BASE) {
                float encoded_semantics = facade_style
                                      - ROAD_MARKING_ENCODING_BASE;
                float lane_count = floor(
                    encoded_semantics / ROAD_MARKING_LANE_STRIDE
                );
                float road_width_m = encoded_semantics
                                   - lane_count * ROAD_MARKING_LANE_STRIDE;
                float lateral_m = (surface_uv.y * .5 + .5) * road_width_m;
                float lane_coordinate = lateral_m * lane_count / road_width_m;
                float nearest_boundary = floor(lane_coordinate + .5);
                float internal_boundary = step(.5, nearest_boundary)
                    * step(nearest_boundary, lane_count - .5);
                float boundary_distance_m = abs(
                    lane_coordinate - nearest_boundary
                ) * road_width_m / lane_count;
                float paint_aa_m = max(fwidth(lateral_m), .006);
                float divider = 1.0 - smoothstep(
                    ROAD_LANE_LINE_WIDTH_M * .5 - paint_aa_m,
                    ROAD_LANE_LINE_WIDTH_M * .5 + paint_aa_m,
                    boundary_distance_m
                );
                float dash_period_m = ROAD_LANE_DASH_PAINT_M
                                    + ROAD_LANE_DASH_GAP_M;
                float dash_position_m = mod(surface_uv.x, dash_period_m);
                float dash_signed_distance_m = abs(
                    dash_position_m - ROAD_LANE_DASH_PAINT_M * .5
                ) - ROAD_LANE_DASH_PAINT_M * .5;
                float dash_aa_m = max(fwidth(surface_uv.x), .012);
                float dash = 1.0 - smoothstep(
                    -dash_aa_m, dash_aa_m, dash_signed_distance_m
                );
                marking = divider * internal_boundary * dash;
            }
            float paint_wear = value_noise(world_position.xz * 1.7 + 17.0);
            vec3 aged_paint = vec3(.52, .52, .46)
                             * mix(.72, 1.0, paint_wear);
            albedo = mix(albedo, aged_paint, marking * .82);
        } else if (material == 4 || material == 16) {
            float clumps = .5;
            float dry_tips = 0.0;
            if (micro_detail > .001) {
                clumps = fbm2(world_position.xz * .72);
                dry_tips = value_noise(world_position.xz * 3.8 + 41.0);
            }
            albedo *= mix(.72, 1.28, mix(.5, clumps, micro_detail));
            albedo = mix(albedo, vec3(.16, .13, .035),
                         smoothstep(.80, .96, dry_tips) * .24 * micro_detail);
        } else if (material == 5 || material == 12) {
            float pores = .5;
            float stains = .5;
            if (micro_detail > .001) {
                pores = fbm2(world_position.xz * 3.1);
                stains = value_noise(world_position.xz * .11 + 8.0);
            }
            albedo *= mix(.82, 1.14, pores * micro_detail + .5 * (1.0-micro_detail));
            albedo *= mix(.78, 1.03, smoothstep(
                .28, .72, mix(.5, stains, micro_detail)
            ));
        } else if (material == 14 || material == 15) {
            // Exposed soil and compacted trail: broad moisture variation with
            // fine gravel/leaf litter, rather than a uniformly coloured mat.
            float moisture = fbm2(world_position.xz * .075);
            float grit = micro_detail > .001
                       ? value_noise(world_position.xz * 5.7) : 0.0;
            vec3 soil = mix(vec3(.055, .033, .018),
                            vec3(.19, .12, .055), moisture);
            albedo = mix(albedo, soil, material == 14 ? .72 : .46);
            albedo += vec3(.08, .055, .018)
                    * smoothstep(.91, .98, grit) * micro_detail;
        } else if (material == 2) {
            float runoff = micro_detail > .001 ? value_noise(
                vec2(surface_uv.x * .028, world_position.y * .31)
            ) : .5;
            albedo *= mix(.70, 1.08, runoff);
        }
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
            roof = vec3(.040, .050, .060);
        } else if (facade_style > 6.5 && facade_style < 7.5) {
            roof = vec3(.018, .035, .052);
        } else if (facade_style > 11.5 && facade_style < 12.5) {
            // Weathered copper cladding on the National Assembly dome.
            roof = vec3(.055, .185, .145);
        }
        float roof_variation = hash21(floor(surface_uv * .08)) * .025;
        frag_color = vec4(
            reflected_radiance(n, roof + roof_variation, reflectance), 1.0
        );
        return;
    }

    if (facade_style > 8.5 && facade_style < 9.5) {
        // Parc.1's external red columns are geometry, not painted window cells.
        reflectance.x = .27;
        reflectance.y = .08;
        frag_color = vec4(
            reflected_radiance(n, vec3(.34, .012, .006), reflectance), 1.0
        );
        return;
    }
    if (facade_style > 9.5 && facade_style < 10.5) {
        // Twenty-four pale octagonal stone columns surround the Assembly.
        reflectance.x = .68;
        frag_color = vec4(
            reflected_radiance(n, vec3(.34, .325, .29), reflectance), 1.0
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
        glass_amount = .88; occupancy_threshold = .84;
    } else if (facade_style > 1.5 && facade_style < 2.5) {
        bay_width = 3.25; floor_height = 4.0;
        pane_bounds = vec4(.04, .96, .06, .93);
        wall = vec3(.32, .21, .06);
        glass = vec3(.16, .085, .018);
        glass_amount = .93; occupancy_threshold = .90;
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
        glass_amount = .90; occupancy_threshold = .88;
    } else if (facade_style > 5.5 && facade_style < 6.5) {
        bay_width = 3.2; floor_height = 3.55;
        pane_bounds = vec4(.12, .86, .16, .78);
        wall = vec3(.19, .18, .17);
        glass = vec3(.025, .038, .055);
        glass_amount = .55; occupancy_threshold = .38;
    } else if (facade_style > 6.5 && facade_style < 7.5) {
        bay_width = 3.35; floor_height = 4.8;
        pane_bounds = vec4(.04, .96, .05, .88);
        wall = vec3(.025, .055, .072);
        glass = vec3(.018, .050, .082);
        glass_amount = .82; occupancy_threshold = .87;
    } else if (facade_style > 7.5 && facade_style < 8.5) {
        bay_width = 5.8; floor_height = 3.6;
        pane_bounds = vec4(.24, .76, .26, .67);
        wall = vec3(.34, .325, .29);
        glass = vec3(.022, .035, .045);
        glass_amount = .07; occupancy_threshold = 1.1;
    }

    // FKI's architect documents 30-degree photovoltaic spandrels over
    // 15-degree downward-tilted vision panes. The alternating normal field is
    // resolved per floor and reproduces the tower's purposeful pleated skin.
    float fki_panel = 0.0;
    if (facade_style > 6.5 && facade_style < 7.5) {
        float panel_phase = fract(surface_uv.y / floor_height);
        fki_panel = interval(panel_phase, .73, .98, .018);
        float tilt = mix(-tan(radians(15.0)), tan(radians(30.0)), fki_panel);
        float fold_detail = 1.0 - smoothstep(
            260.0, 1100.0, length(camera_position - world_position)
        );
        n = normalize(n + vec3(0.0, tilt * fold_detail, 0.0));
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
    vec3 facade_tangent = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    float view_facing = max(dot(n, view_direction), .18);
    vec2 room_parallax = vec2(
        dot(view_direction, facade_tangent), view_direction.y
    ) / view_facing * vec2(.045, .032);
    vec2 room_within = fract(grid + room_parallax);
    // Venetian blinds, curtains and a darker reveal give the luminous plane
    // depth. The parallax is bounded to a few centimetres of normalized room
    // space, so it reads as an interior behind glass rather than sliding UVs.
    float blind_phase = fract(room_within.y * 11.0 + temperature * .37);
    float blinds = mix(.52, 1.0, smoothstep(.16, .31, blind_phase));
    float curtain_split = abs(room_within.x - .5);
    float curtains = mix(.58, 1.0, smoothstep(.16, .28, curtain_split));
    float fresnel = pow(1.0 - max(dot(n, view_direction), 0.0), 4.0);
    vec3 facade = mix(wall, glass, pane * glass_amount);
    if (facade_style > 6.5 && facade_style < 7.5) {
        facade = mix(facade, vec3(.012, .027, .038), fki_panel * .92);
    }
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
    float pane_edge_distance = min(
        min(abs(within.x - pane_bounds.x), abs(within.x - pane_bounds.y)),
        min(abs(within.y - pane_bounds.z), abs(within.y - pane_bounds.w))
    );
    float reveal_shadow = (1.0 - smoothstep(.018, .085, pane_edge_distance))
                        * pane;
    facade *= 1.0 - reveal_shadow * .34;
    float weathering = hash21(
        floor(surface_uv / vec2(18.0, 26.0)) + facade_style * 7.3
    );
    facade *= mix(.94, 1.035, weathering);
    vec3 emission = window_color * pane * occupied * blinds * curtains
                  * window_radiance_w_m2_sr * .72;

    if (facade_style > 2.5 && facade_style < 3.5) {
        float balcony = 1.0 - smoothstep(
            .035, .09, min(within.y, 1.0 - within.y)
        );
        facade += balcony * vec3(.10, .095, .086);
    }
    // Parc.1's red frame is authored as projecting geometry. Painting a
    // second grid into the glass made the columns read as a flat texture.
    float facade_detail = 1.0 - smoothstep(
        120.0, 520.0, length(camera_position - world_position)
    );
    float glass_coverage = clamp(pane * glass_amount, 0.0, 1.0);
    if (facade_detail > .001) {
        const int WALL_SCAN = 3;
        vec2 wall_uv = surface_uv / scanned_texture_width_m[WALL_SCAN];
        float wall_base_layer = float(WALL_SCAN * 3);
        vec3 wall_scan = pow(texture(
            scanned_material_texture, vec3(wall_uv, wall_base_layer)
        ).rgb, vec3(2.2));
        vec3 wall_variation = clamp(
            wall_scan / max(scanned_diffuse_mean[WALL_SCAN], vec3(.015)),
            vec3(.45), vec3(1.9)
        );
        facade *= mix(
            vec3(1.0), wall_variation,
            facade_detail * (1.0 - glass_coverage) * .48
        );
        vec3 wall_tangent_normal = texture(
            scanned_material_texture,
            vec3(wall_uv, wall_base_layer + 1.0)
        ).xyz * 2.0 - 1.0;
        wall_tangent_normal.xy *= .52;
        vec3 wall_normal = normalize(
            cotangent_frame(n, world_position, wall_uv) * wall_tangent_normal
        );
        n = normalize(mix(
            n, wall_normal,
            facade_detail * (1.0 - glass_coverage)
        ));
    }
    // Glass and cladding have different microfacet distributions. Keeping one
    // roughness for the whole facade made every elevation look printed flat.
    reflectance.x = mix(
        max(reflectance.x, .42), .105, glass_coverage
    );
    reflectance.z *= mix(.92, 1.0, glass_coverage);
    float landmark_emission_scale = 1.0;
    if (facade_style > 1.5 && facade_style < 2.5) {
        landmark_emission_scale = .48;
    } else if (facade_style > 4.5 && facade_style < 5.5) {
        landmark_emission_scale = .58;
    } else if (facade_style > 6.5 && facade_style < 7.5) {
        landmark_emission_scale = .62;
    } else if (facade_style > 7.5 && facade_style < 8.5) {
        landmark_emission_scale = .08;
    }
    frag_color = vec4(
        reflected_radiance(n, facade, reflectance)
        + emission * landmark_emission_scale, 1.0
    );
}
