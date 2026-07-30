#version 330
in vec3 world_position;
in vec3 world_normal;
in float surface;
in vec2 surface_uv;
in float facade_style;
uniform vec3 camera_position;
uniform float ambient_irradiance_w_m2;
uniform float window_radiance_w_m2_sr;
uniform float air_extinction_per_m;
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
out vec4 frag_color;
const float PI = 3.14159265359;
float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
float interval(float value, float lower, float upper, float antialias) {
    return smoothstep(lower - antialias, lower + antialias, value)
         * (1.0 - smoothstep(upper - antialias, upper + antialias, value));
}
vec3 reflected_radiance(vec3 n, vec3 albedo) {
    vec3 result = albedo * ambient_irradiance_w_m2 / PI;
    for (int i = 0; i < 4; ++i) {
        if (i >= static_light_count) break;
        vec3 displacement = static_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), .5);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float down_lobe = pow(max(light_direction.y, 0.0), 3.0);
        float irradiance = static_light_power_w * 2.0 * down_lobe
                         * exp(-air_extinction_per_m * distance_m)
                         / (PI * distance_squared);
        result += albedo * static_light_color * irradiance
                * max(dot(n, light_direction), 0.0) / PI;
    }
    for (int i = 0; i < 8; ++i) {
        if (i >= dynamic_light_count) break;
        vec3 displacement = dynamic_light_position[i] - world_position;
        float distance_squared = max(dot(displacement, displacement), 1.0);
        float distance_m = sqrt(distance_squared);
        vec3 light_direction = displacement / distance_m;
        float irradiance = dynamic_light_power_w[i]
                         * exp(-air_extinction_per_m * distance_m)
                         / (4.0 * PI * distance_squared);
        result += albedo * dynamic_light_color[i] * irradiance
                * max(dot(n, light_direction), 0.0) / PI;
    }
    return result;
}
void main() {
    vec3 n = normalize(world_normal);
    if (surface > 15.5) {
        float blade = hash21(floor(world_position.xz * 9.0));
        vec3 grass = mix(vec3(.028, .105, .018),
                         vec3(.11, .24, .045), blade);
        float translucency = pow(
            max(dot(normalize(wind_xz.x == 0.0 && wind_xz.y == 0.0
                ? vec3(1.0, .2, 0.0)
                : vec3(wind_xz.x, .2, wind_xz.y)),
                -n), 0.0), 2.0
        ) * min(wind_speed_mps / 5.0, 1.0);
        frag_color = vec4(
            reflected_radiance(n, grass) + grass * translucency * .00018,
            1.0
        );
        return;
    }
    if (surface > 14.5) {
        float aggregate = hash21(floor(world_position.xz * 1.35));
        vec3 trail = mix(vec3(.16, .13, .095), vec3(.26, .22, .16), aggregate);
        frag_color = vec4(reflected_radiance(n, trail), 1.0);
        return;
    }
    if (surface > 13.5) {
        float bed = hash21(floor(world_position.xz * .32));
        vec3 garden = mix(
            vec3(.035, .11, .025), vec3(.17, .055, .025), bed * .34
        );
        frag_color = vec4(reflected_radiance(n, garden), 1.0);
        return;
    }
    if (surface > 12.5) {
        float tiles = step(.08, abs(fract(surface_uv.x * .25) - .5))
                    * step(.08, abs(fract(surface_uv.y * .25) - .5));
        vec3 rubber = mix(vec3(.12, .035, .025), vec3(.035, .10, .12), tiles);
        frag_color = vec4(reflected_radiance(n, rubber), 1.0);
        return;
    }
    if (surface > 11.5) {
        float panels = step(.035, min(
            abs(fract(surface_uv.x * .22) - .5),
            abs(fract(surface_uv.y * .22) - .5)
        ));
        vec3 concrete = mix(vec3(.21), vec3(.15), panels * .12);
        frag_color = vec4(reflected_radiance(n, concrete), 1.0);
        return;
    }
    if (surface > 10.5) {
        float leaf = hash21(floor(world_position.xz * .65 + world_position.y));
        vec3 foliage = mix(vec3(.025, .095, .018),
                           vec3(.075, .19, .038), leaf);
        frag_color = vec4(reflected_radiance(n, foliage), 1.0);
        return;
    }
    if (surface > 9.5) {
        vec3 lamp = vec3(1.0, .48, .13) * window_radiance_w_m2_sr * 1.8;
        frag_color = vec4(lamp + reflected_radiance(n, vec3(.22)), 1.0);
        return;
    }
    if (surface > 8.5) {
        float grain = hash21(floor(surface_uv * vec2(1.8, .18)));
        vec3 wood = mix(vec3(.12, .050, .018), vec3(.28, .13, .045), grain);
        frag_color = vec4(reflected_radiance(n, wood), 1.0);
        return;
    }
    if (surface > 7.5) {
        float fresnel = pow(
            1.0 - max(dot(n, normalize(camera_position-world_position)), 0.0),
            3.0
        );
        vec3 metal = mix(vec3(.18), vec3(.42), fresnel);
        frag_color = vec4(reflected_radiance(n, metal), 1.0);
        return;
    }
    if (surface > 6.5) {
        float marking = 1.0 - smoothstep(
            .025, .055, abs(fract(surface_uv.x * .05) - .5)
        );
        vec3 sport = mix(vec3(.035, .13, .075), vec3(.75), marking);
        frag_color = vec4(reflected_radiance(n, sport), 1.0);
        return;
    }
    if (surface > 5.5) {
        float stripe = 1.0 - smoothstep(
            .025, .055, abs(fract(surface_uv.x * .11) - .5)
        );
        vec3 cycleway = mix(vec3(.19, .045, .032), vec3(.72), stripe * .6);
        frag_color = vec4(reflected_radiance(n, cycleway), 1.0);
        return;
    }
    if (surface > 4.5) {
        vec2 joints = abs(fract(surface_uv * vec2(.42, .28)) - .5);
        float seam = 1.0 - smoothstep(.025, .055, min(joints.x, joints.y));
        vec3 paving = mix(vec3(.24, .23, .21), vec3(.09), seam);
        frag_color = vec4(reflected_radiance(n, paving), 1.0);
        return;
    }
    if (surface > 3.5) {
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
        );
        float green_variation = hash21(floor(world_position.xz * .15));
        vec3 green = mix(vec3(.055, .16, .045),
                         vec3(.12, .25, .075), green_variation);
        frag_color = vec4(reflected_radiance(n, green), 1.0);
        return;
    }
    if (surface > 2.5) {
        float lane_hint = smoothstep(.46, .50,
            abs(fract(world_position.x * .12 + world_position.z * .08) - .5));
        vec3 asphalt = mix(
            vec3(.035, .038, .042), vec3(.32), lane_hint * .22
        );
        frag_color = vec4(reflected_radiance(n, asphalt), 1.0);
        return;
    }
    if (surface > 1.5) {
        frag_color = vec4(
            reflected_radiance(n, vec3(.22, .24, .26)), 1.0
        );
        return;
    }
    if (surface > .5) {
        vec3 roof = vec3(.12, .13, .14);
        if (facade_style > 1.5 && facade_style < 2.5) {
            roof = vec3(.30, .18, .055);
        } else if (facade_style > 4.5 && facade_style < 5.5) {
            roof = vec3(.31, .035, .025);
        }
        float roof_variation = hash21(floor(surface_uv * .08)) * .025;
        frag_color = vec4(
            reflected_radiance(n, roof + roof_variation), 1.0
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
    float distance_m = length(camera_position - world_position);
    float occupied = distance_m < 1600.0
        ? step(occupancy_threshold, hash21(cell + facade_style * 31.7))
        : .38;
    float temperature = distance_m < 1600.0
        ? hash21(cell + facade_style * 17.0 + 11.3)
        : .55;
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
                  * window_radiance_w_m2_sr;

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
    frag_color = vec4(reflected_radiance(n, facade) + emission, 1.0);
}
