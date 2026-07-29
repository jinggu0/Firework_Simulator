from __future__ import annotations

import math

import moderngl
import numpy as np

from .config import AtmosphereConfig, SmokeConfig
from .fluid import AIR_HEAT_CAPACITY_J_KG_K, GRAVITY_MPS2


COMMON_GLSL = """
uniform sampler3D velocity_u;
uniform sampler3D velocity_v;
uniform sampler3D velocity_w;
uniform vec3 domain_min;
uniform vec3 cell_size;
uniform ivec3 cell_count;

vec3 field_uv(vec3 world, vec3 offset, ivec3 size) {
    vec3 index = (world - domain_min) / cell_size - offset;
    return (index + 0.5) / vec3(size);
}
float sample_u(vec3 world) {
    return texture(
        velocity_u,
        field_uv(world, vec3(0.0, 0.5, 0.5),
                 cell_count + ivec3(1, 0, 0))
    ).r;
}
float sample_v(vec3 world) {
    return texture(
        velocity_v,
        field_uv(world, vec3(0.5, 0.0, 0.5),
                 cell_count + ivec3(0, 1, 0))
    ).r;
}
float sample_w(vec3 world) {
    return texture(
        velocity_w,
        field_uv(world, vec3(0.5, 0.5, 0.0),
                 cell_count + ivec3(0, 0, 1))
    ).r;
}
vec3 sample_velocity(vec3 world) {
    return vec3(sample_u(world), sample_v(world), sample_w(world));
}
vec3 midpoint_backtrace(vec3 world, float dt_s) {
    vec3 initial_velocity = sample_velocity(world);
    vec3 midpoint = world - 0.5 * dt_s * initial_velocity;
    return world - dt_s * sample_velocity(midpoint);
}
"""


STATE_ADVECTION = (
    """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(rg32f, binding=0) writeonly uniform image3D output_state;
uniform sampler3D source_state;
uniform sampler3D source_increment;
uniform float dt_s;
uniform float max_temperature_excess_k;
"""
    + COMMON_GLSL
    + """
void main() {
    ivec3 cell = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(cell, cell_count))) return;
    vec3 world = domain_min + (vec3(cell) + 0.5) * cell_size;
    vec3 previous = midpoint_backtrace(world, dt_s);
    vec2 advected = texture(
        source_state,
        field_uv(previous, vec3(0.5), cell_count)
    ).rg;
    vec2 increment = texelFetch(source_increment, cell, 0).rg;
    vec2 result = max(advected + increment, 0.0);
    result.g = min(result.g, max_temperature_excess_k);
    imageStore(output_state, cell, vec4(result, 0, 0));
}
"""
)


def _velocity_advection(component: int) -> str:
    offsets = (
        "vec3(0.0, 0.5, 0.5)",
        "vec3(0.5, 0.0, 0.5)",
        "vec3(0.5, 0.5, 0.0)",
    )
    size_offsets = (
        "ivec3(1, 0, 0)",
        "ivec3(0, 1, 0)",
        "ivec3(0, 0, 1)",
    )
    return (
        """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(r32f, binding=0) writeonly uniform image3D output_field;
uniform sampler3D source_field;
uniform float dt_s;
"""
        + COMMON_GLSL
        + f"""
void main() {{
    ivec3 field_size = cell_count + {size_offsets[component]};
    ivec3 face = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(face, field_size))) return;
    vec3 offset = {offsets[component]};
    vec3 world = domain_min + (vec3(face) + offset) * cell_size;
    vec3 previous = midpoint_backtrace(world, dt_s);
    float value = texture(
        source_field, field_uv(previous, offset, field_size)
    ).r;
    imageStore(output_field, face, vec4(value, 0, 0, 0));
}}
"""
    )


STATE_EVOLUTION = """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(rg32f, binding=0) writeonly uniform image3D output_state;
uniform sampler3D source_state;
uniform ivec3 cell_count;
uniform vec3 cell_size;
uniform vec2 diffusivity;
uniform vec2 half_life_s;
uniform float dt_s;
vec2 state_at(ivec3 cell) {
    return texelFetch(source_state, clamp(cell, ivec3(0), cell_count - 1), 0).rg;
}
void main() {
    ivec3 cell = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(cell, cell_count))) return;
    vec2 centre = state_at(cell);
    vec2 laplacian =
          (state_at(cell - ivec3(1,0,0)) - 2.0*centre
           + state_at(cell + ivec3(1,0,0))) / (cell_size.x*cell_size.x)
        + (state_at(cell - ivec3(0,1,0)) - 2.0*centre
           + state_at(cell + ivec3(0,1,0))) / (cell_size.y*cell_size.y)
        + (state_at(cell - ivec3(0,0,1)) - 2.0*centre
           + state_at(cell + ivec3(0,0,1))) / (cell_size.z*cell_size.z);
    vec2 decay = exp(-log(2.0) * dt_s / half_life_s);
    vec2 result = max((centre + diffusivity*dt_s*laplacian)*decay, 0.0);
    imageStore(output_state, cell, vec4(result, 0, 0));
}
"""


VORTICITY = (
    """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(rgba32f, binding=0) writeonly uniform image3D output_vorticity;
"""
    + COMMON_GLSL
    + """
void main() {
    ivec3 cell = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(cell, cell_count))) return;
    vec3 p = domain_min + (vec3(cell) + 0.5) * cell_size;
    float dwdy = (sample_w(p + vec3(0,.5*cell_size.y,0))
                - sample_w(p - vec3(0,.5*cell_size.y,0))) / cell_size.y;
    float dvdz = (sample_v(p + vec3(0,0,.5*cell_size.z))
                - sample_v(p - vec3(0,0,.5*cell_size.z))) / cell_size.z;
    float dudz = (sample_u(p + vec3(0,0,.5*cell_size.z))
                - sample_u(p - vec3(0,0,.5*cell_size.z))) / cell_size.z;
    float dwdx = (sample_w(p + vec3(.5*cell_size.x,0,0))
                - sample_w(p - vec3(.5*cell_size.x,0,0))) / cell_size.x;
    float dvdx = (sample_v(p + vec3(.5*cell_size.x,0,0))
                - sample_v(p - vec3(.5*cell_size.x,0,0))) / cell_size.x;
    float dudy = (sample_u(p + vec3(0,.5*cell_size.y,0))
                - sample_u(p - vec3(0,.5*cell_size.y,0))) / cell_size.y;
    vec3 omega = vec3(dwdy-dvdz, dudz-dwdx, dvdx-dudy);
    imageStore(output_vorticity, cell, vec4(omega, length(omega)));
}
"""
)


FORCE_COMMON = (
    """
uniform sampler3D source_field;
uniform sampler3D smoke_state;
uniform sampler3D vorticity_field;
uniform float dt_s;
uniform float viscosity;
uniform float vorticity_confinement;
uniform float ambient_temperature_k;
uniform float air_density_kg_m3;
uniform vec2 background_wind_xz;
"""
    + COMMON_GLSL
    + """
vec4 sample_vorticity(vec3 world) {
    return texture(
        vorticity_field,
        field_uv(world, vec3(0.5), cell_count)
    );
}
vec3 confinement_force(vec3 world) {
    vec3 gradient = vec3(
        sample_vorticity(world + vec3(cell_size.x,0,0)).a
          - sample_vorticity(world - vec3(cell_size.x,0,0)).a,
        sample_vorticity(world + vec3(0,cell_size.y,0)).a
          - sample_vorticity(world - vec3(0,cell_size.y,0)).a,
        sample_vorticity(world + vec3(0,0,cell_size.z)).a
          - sample_vorticity(world - vec3(0,0,cell_size.z)).a
    ) / (2.0 * cell_size);
    float gradient_length = length(gradient);
    if (gradient_length < 1e-7) return vec3(0.0);
    vec3 normal = gradient / gradient_length;
    vec3 omega = sample_vorticity(world).xyz;
    return vorticity_confinement * min(min(cell_size.x, cell_size.y), cell_size.z)
         * cross(normal, omega);
}
float field_at(ivec3 p, ivec3 size) {
    return texelFetch(source_field, clamp(p, ivec3(0), size-1), 0).r;
}
float field_laplacian(ivec3 p, ivec3 size) {
    float c = field_at(p, size);
    return
          (field_at(p-ivec3(1,0,0),size)-2.0*c
           +field_at(p+ivec3(1,0,0),size))/(cell_size.x*cell_size.x)
        + (field_at(p-ivec3(0,1,0),size)-2.0*c
           +field_at(p+ivec3(0,1,0),size))/(cell_size.y*cell_size.y)
        + (field_at(p-ivec3(0,0,1),size)-2.0*c
           +field_at(p+ivec3(0,0,1),size))/(cell_size.z*cell_size.z);
}
"""
)


def _force_shader(component: int) -> str:
    offsets = (
        "vec3(0.0,0.5,0.5)",
        "vec3(0.5,0.0,0.5)",
        "vec3(0.5,0.5,0.0)",
    )
    additions = (
        "ivec3(1,0,0)", "ivec3(0,1,0)", "ivec3(0,0,1)"
    )
    boundary = (
        """
    if (face.x == 0) { result = background_wind_xz.x; }
    else if (face.x == cell_count.x) {
        result = field_at(face-ivec3(1,0,0), field_size);
    }
""",
        """
    if (face.y == 0) { result = 0.0; }
    else if (face.y == cell_count.y) {
        result = field_at(face-ivec3(0,1,0), field_size);
    }
""",
        """
    if (face.z == 0) { result = background_wind_xz.y; }
    else if (face.z == cell_count.z) {
        result = field_at(face-ivec3(0,0,1), field_size);
    }
""",
    )
    buoyancy = (
        "",
        """
        vec3 state_uv = field_uv(world, vec3(0.5), cell_count);
        vec2 state = texture(smoke_state, state_uv).rg;
        force += 9.80665 * (
            state.g/ambient_temperature_k - state.r/air_density_kg_m3
        );
""",
        "",
    )
    return (
        """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(r32f, binding=0) writeonly uniform image3D output_field;
"""
        + FORCE_COMMON
        + f"""
void main() {{
    ivec3 field_size = cell_count + {additions[component]};
    ivec3 face = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(face, field_size))) return;
    vec3 world = domain_min + (vec3(face)+{offsets[component]})*cell_size;
    float result = field_at(face, field_size);
    bool boundary = false;
{boundary[component]}
    else {{
        float force = confinement_force(world)[{component}];
{buoyancy[component]}
        result += dt_s * (force + viscosity*field_laplacian(face,field_size));
    }}
    imageStore(output_field, face, vec4(result,0,0,0));
}}
"""
    )


DIVERGENCE = """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(r32f, binding=0) writeonly uniform image3D output_divergence;
uniform sampler3D velocity_u;
uniform sampler3D velocity_v;
uniform sampler3D velocity_w;
uniform ivec3 cell_count;
uniform vec3 cell_size;
void main() {
    ivec3 c = ivec3(gl_GlobalInvocationID);
    if (any(greaterThanEqual(c, cell_count))) return;
    float divergence =
          (texelFetch(velocity_u,c+ivec3(1,0,0),0).r
           - texelFetch(velocity_u,c,0).r)/cell_size.x
        + (texelFetch(velocity_v,c+ivec3(0,1,0),0).r
           - texelFetch(velocity_v,c,0).r)/cell_size.y
        + (texelFetch(velocity_w,c+ivec3(0,0,1),0).r
           - texelFetch(velocity_w,c,0).r)/cell_size.z;
    imageStore(output_divergence,c,vec4(divergence,0,0,0));
}
"""


PRESSURE = """
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(r32f, binding=0) writeonly uniform image3D output_pressure;
uniform sampler3D source_pressure;
uniform sampler3D divergence;
uniform ivec3 cell_count;
uniform vec3 cell_size;
uniform float air_density_kg_m3;
uniform float dt_s;
float pressure_at(ivec3 p) {
    return texelFetch(source_pressure,clamp(p,ivec3(0),cell_count-1),0).r;
}
float rhs_at(ivec3 p) {
    p=clamp(p,ivec3(0),cell_count-1);
    return air_density_kg_m3*texelFetch(divergence,p,0).r/dt_s;
}
float jacobi_once(ivec3 p) {
    vec3 h2=cell_size*cell_size;
    float numerator =
        (pressure_at(p-ivec3(1,0,0))+pressure_at(p+ivec3(1,0,0)))*h2.y*h2.z+
        (pressure_at(p-ivec3(0,1,0))+pressure_at(p+ivec3(0,1,0)))*h2.x*h2.z+
        (pressure_at(p-ivec3(0,0,1))+pressure_at(p+ivec3(0,0,1)))*h2.x*h2.y-
        rhs_at(p)*h2.x*h2.y*h2.z;
    return numerator/(2.0*(h2.y*h2.z+h2.x*h2.z+h2.x*h2.y));
}
void main() {
    ivec3 p=ivec3(gl_GlobalInvocationID);
    if(any(greaterThanEqual(p,cell_count))) return;
    vec3 h2=cell_size*cell_size;
    float numerator =
        (jacobi_once(p-ivec3(1,0,0))+jacobi_once(p+ivec3(1,0,0)))*h2.y*h2.z+
        (jacobi_once(p-ivec3(0,1,0))+jacobi_once(p+ivec3(0,1,0)))*h2.x*h2.z+
        (jacobi_once(p-ivec3(0,0,1))+jacobi_once(p+ivec3(0,0,1)))*h2.x*h2.y-
        rhs_at(p)*h2.x*h2.y*h2.z;
    float result=numerator/(2.0*(h2.y*h2.z+h2.x*h2.z+h2.x*h2.y));
    // Pin the arbitrary pressure gauge to prevent constant-mode drift.
    if (all(equal(p, ivec3(0)))) result = 0.0;
    imageStore(output_pressure,p,vec4(result,0,0,0));
}
"""


def _project_shader(component: int) -> str:
    additions = (
        "ivec3(1,0,0)", "ivec3(0,1,0)", "ivec3(0,0,1)"
    )
    axis = "xyz"[component]
    bg = ("background_wind_xz.x", "0.0", "background_wind_xz.y")
    return f"""
#version 430
layout(local_size_x=8, local_size_y=4, local_size_z=4) in;
layout(r32f, binding=0) writeonly uniform image3D output_velocity;
uniform sampler3D source_velocity;
uniform sampler3D pressure;
uniform ivec3 cell_count;
uniform vec3 cell_size;
uniform float dt_s;
uniform float air_density_kg_m3;
uniform vec2 background_wind_xz;
void main() {{
    ivec3 size=cell_count+{additions[component]};
    ivec3 face=ivec3(gl_GlobalInvocationID);
    if(any(greaterThanEqual(face,size))) return;
    float result;
    if(face.{axis}==0) result={bg[component]};
    else if(face.{axis}==cell_count.{axis}) {{
        ivec3 previous=face; previous.{axis}-=1;
        result=texelFetch(source_velocity,previous,0).r;
    }} else {{
        ivec3 lower=face; lower.{axis}-=1;
        float gradient=(texelFetch(pressure,face,0).r
                       -texelFetch(pressure,lower,0).r)/cell_size.{axis};
        result=texelFetch(source_velocity,face,0).r
              -dt_s*gradient/air_density_kg_m3;
    }}
    imageStore(output_velocity,face,vec4(result,0,0,0));
}}
"""


class GpuSmokeFluid3D:
    """Depth-resolved OpenGL 4.3 MAC plume with conservative CPU sources."""

    backend_name = "gpu_compute_mac_3d"
    is_3d = True

    def __init__(
        self,
        ctx: moderngl.Context,
        config: SmokeConfig,
        atmosphere: AtmosphereConfig,
    ) -> None:
        if ctx.version_code < 430:
            raise moderngl.Error("OpenGL 4.3 compute shaders unavailable")
        self.ctx, self.config = ctx, config
        self.nx, self.ny, self.nz = config.gpu_3d_grid_size
        self.cell_count = (self.nx, self.ny, self.nz)
        (
            self.x_min, self.x_max, self.y_min, self.y_max,
            self.z_min, self.z_max,
        ) = config.gpu_3d_bounds_m
        self.dx = (self.x_max-self.x_min)/self.nx
        self.dy = (self.y_max-self.y_min)/self.ny
        self.dz = (self.z_max-self.z_min)/self.nz
        self.update_hz = config.gpu_3d_update_hz
        self.pressure_iterations_per_step = max(
            round(config.pressure_iterations*config.update_hz/self.update_hz), 1
        )
        self.ambient_temperature_k = atmosphere.temperature_k
        self.air_density_kg_m3 = atmosphere.air_density_kg_m3
        self.background_wind_xz = np.asarray(
            [atmosphere.wind_velocity_mps[0], atmosphere.wind_velocity_mps[2]],
            dtype=np.float32,
        )
        self.revision = 0
        self.total_smoke_mass_kg = 0.0
        self.source_state = np.zeros(
            (self.nz, self.ny, self.nx, 2), dtype=np.float32
        )
        self.active_minimum = np.array(
            [self.x_max,self.y_max,self.z_max],dtype=np.float32
        )
        self.active_maximum = np.array(
            [self.x_min,self.y_min,self.z_min],dtype=np.float32
        )
        self.state = self._pair(self.cell_count, 2)
        self.source_texture = self._texture(self.cell_count, 2)
        self.u = self._pair((self.nx+1,self.ny,self.nz), 1)
        self.v = self._pair((self.nx,self.ny+1,self.nz), 1)
        self.w = self._pair((self.nx,self.ny,self.nz+1), 1)
        self.pressure = self._pair(self.cell_count, 1)
        self.divergence_texture = self._texture(self.cell_count, 1)
        self.vorticity_texture = self._texture(self.cell_count, 4)
        self.velocity_index = 0
        self.last_pressure_index = 0
        self.diagnostic_velocity_dirty = False
        self.programs = {
            "state_advect": ctx.compute_shader(STATE_ADVECTION),
            "u_advect": ctx.compute_shader(_velocity_advection(0)),
            "v_advect": ctx.compute_shader(_velocity_advection(1)),
            "w_advect": ctx.compute_shader(_velocity_advection(2)),
            "state_evolve": ctx.compute_shader(STATE_EVOLUTION),
            "vorticity": ctx.compute_shader(VORTICITY),
            "u_force": ctx.compute_shader(_force_shader(0)),
            "v_force": ctx.compute_shader(_force_shader(1)),
            "w_force": ctx.compute_shader(_force_shader(2)),
            "divergence": ctx.compute_shader(DIVERGENCE),
            "pressure": ctx.compute_shader(PRESSURE),
            "project_u": ctx.compute_shader(_project_shader(0)),
            "project_v": ctx.compute_shader(_project_shader(1)),
            "project_w": ctx.compute_shader(_project_shader(2)),
        }
        self._configure()
        zero_state = np.zeros_like(self.source_state)
        for texture in self.state:
            texture.write(zero_state.tobytes())
        zero_u=np.zeros((self.nz,self.ny,self.nx+1),np.float32)
        zero_u[:,:,0]=self.background_wind_xz[0]
        zero_u[:,:,-1]=self.background_wind_xz[0]
        zero_v=np.zeros((self.nz,self.ny+1,self.nx),np.float32)
        zero_w=np.zeros((self.nz+1,self.ny,self.nx),np.float32)
        zero_w[0]=self.background_wind_xz[1]
        zero_w[-1]=self.background_wind_xz[1]
        for texture in self.u: texture.write(zero_u.tobytes())
        for texture in self.v: texture.write(zero_v.tobytes())
        for texture in self.w: texture.write(zero_w.tobytes())
        zero_scalar = np.zeros(
            (self.nz, self.ny, self.nx), dtype=np.float32
        )
        for texture in self.pressure:
            texture.write(zero_scalar.tobytes())

    @property
    def render_state_texture(self) -> moderngl.Texture3D:
        return self.state[0]

    def _texture(self,size:tuple[int,int,int],components:int):
        texture=self.ctx.texture3d(size,components=components,dtype="f4")
        texture.filter=moderngl.LINEAR,moderngl.LINEAR
        texture.repeat_x=texture.repeat_y=texture.repeat_z=False
        return texture

    def _pair(self,size,components):
        return [self._texture(size,components),self._texture(size,components)]

    def _configure(self) -> None:
        def uniform(program, name, value):
            if name in program:
                program[name].value = value

        common=("state_advect","u_advect","v_advect","w_advect",
                "vorticity","u_force","v_force","w_force")
        for name in common:
            p=self.programs[name]
            uniform(p,"velocity_u",1);uniform(p,"velocity_v",2)
            uniform(p,"velocity_w",3)
            uniform(p,"domain_min",(self.x_min,self.y_min,self.z_min))
            uniform(p,"cell_size",(self.dx,self.dy,self.dz))
            uniform(p,"cell_count",self.cell_count)
        uniform(self.programs["state_advect"],"source_state",0)
        uniform(self.programs["state_advect"],"source_increment",6)
        uniform(
            self.programs["state_advect"],
            "max_temperature_excess_k",
            self.config.max_temperature_excess_k,
        )
        for name in ("u_advect","v_advect","w_advect"):
            uniform(self.programs[name],"source_field",0)
        e=self.programs["state_evolve"];uniform(e,"source_state",0)
        uniform(e,"cell_count",self.cell_count)
        uniform(e,"cell_size",(self.dx,self.dy,self.dz))
        uniform(e,"diffusivity",(self.config.smoke_diffusivity_m2_s,self.config.thermal_diffusivity_m2_s))
        uniform(e,"half_life_s",(self.config.smoke_half_life_s,self.config.thermal_half_life_s))
        for name in ("u_force","v_force","w_force"):
            p=self.programs[name];uniform(p,"source_field",0)
            uniform(p,"smoke_state",4);uniform(p,"vorticity_field",7)
            uniform(p,"viscosity",self.config.kinematic_viscosity_m2_s)
            uniform(p,"vorticity_confinement",self.config.vorticity_confinement)
        d=self.programs["divergence"];uniform(d,"velocity_u",1)
        uniform(d,"velocity_v",2);uniform(d,"velocity_w",3)
        uniform(d,"cell_count",self.cell_count)
        uniform(d,"cell_size",(self.dx,self.dy,self.dz))
        p=self.programs["pressure"];uniform(p,"source_pressure",5)
        uniform(p,"divergence",6);uniform(p,"cell_count",self.cell_count)
        uniform(p,"cell_size",(self.dx,self.dy,self.dz))
        for name in ("project_u","project_v","project_w"):
            q=self.programs[name];uniform(q,"source_velocity",0)
            uniform(q,"pressure",5);uniform(q,"cell_count",self.cell_count)
            uniform(q,"cell_size",(self.dx,self.dy,self.dz))

    def _groups(self,size):
        return (math.ceil(size[0]/8),math.ceil(size[1]/4),math.ceil(size[2]/4))

    def _run(self,name,size,output):
        output.bind_to_image(0,read=False,write=True)
        self.programs[name].run(*self._groups(size))
        self.ctx.memory_barrier()

    def set_atmosphere(self, atmosphere: AtmosphereConfig) -> None:
        self.ambient_temperature_k=atmosphere.temperature_k
        self.air_density_kg_m3=atmosphere.air_density_kg_m3
        self.background_wind_xz[:]=(
            atmosphere.wind_velocity_mps[0],atmosphere.wind_velocity_mps[2]
        )

    def _extend_bounds(self,positions,margin):
        if not len(positions): return
        p=np.asarray(positions)
        self.active_minimum=np.maximum(
            np.minimum(self.active_minimum,p.min(axis=0)-margin),
            (self.x_min,self.y_min,self.z_min))
        self.active_maximum=np.minimum(
            np.maximum(self.active_maximum,p.max(axis=0)+margin),
            (self.x_max,self.y_max,self.z_max))

    def inject_burst(self,position_m,smoke_mass_kg,thermal_energy_j):
        x=self.x_min+(np.arange(self.nx)+.5)*self.dx
        y=self.y_min+(np.arange(self.ny)+.5)*self.dy
        z=self.z_min+(np.arange(self.nz)+.5)*self.dz
        zz,yy,xx=np.meshgrid(z,y,x,indexing="ij")
        r2=self.config.source_radius_m**2
        weights=np.exp(-.5*((xx-position_m[0])**2+(yy-position_m[1])**2+
                           (zz-position_m[2])**2)/r2).astype(np.float32)
        represented=float(weights.sum())*self.dx*self.dy*self.dz
        if represented<=1e-9:return
        self.source_state[:,:,:,0]+=weights*(smoke_mass_kg/represented)
        heated=max(self.air_density_kg_m3*represented,1e-6)
        delta_t=min(thermal_energy_j/(heated*AIR_HEAT_CAPACITY_J_KG_K),
                    self.config.max_temperature_excess_k)
        self.source_state[:,:,:,1]+=weights*delta_t
        self.total_smoke_mass_kg+=smoke_mass_kg
        self._extend_bounds(np.asarray(position_m).reshape(1,3),3*self.config.source_radius_m)
        self.revision+=1

    def inject_particles(self,positions_m,smoke_mass_kg,thermal_energy_j):
        if not len(positions_m):return 0.0,0.0
        p=np.asarray(positions_m)
        ix=np.floor((p[:,0]-self.x_min)/self.dx).astype(np.int32)
        iy=np.floor((p[:,1]-self.y_min)/self.dy).astype(np.int32)
        iz=np.floor((p[:,2]-self.z_min)/self.dz).astype(np.int32)
        inside=(ix>=0)&(ix<self.nx)&(iy>=0)&(iy<self.ny)&(iz>=0)&(iz<self.nz)
        if not np.any(inside):return 0.0,0.0
        indices=(iz[inside]*self.ny+iy[inside])*self.nx+ix[inside]
        volume=self.dx*self.dy*self.dz
        self.source_state[:,:,:,0].ravel()[:]+=(
            np.bincount(indices,weights=np.asarray(smoke_mass_kg)[inside],
                        minlength=self.nx*self.ny*self.nz)/volume).astype(np.float32)
        self.source_state[:,:,:,1].ravel()[:]+=(
            np.bincount(indices,weights=np.asarray(thermal_energy_j)[inside],
                        minlength=self.nx*self.ny*self.nz)/
            (self.air_density_kg_m3*volume*AIR_HEAT_CAPACITY_J_KG_K)).astype(np.float32)
        mass=float(np.asarray(smoke_mass_kg)[inside].sum(dtype=np.float64))
        energy=float(np.asarray(thermal_energy_j)[inside].sum(dtype=np.float64))
        self.total_smoke_mass_kg+=mass
        self._extend_bounds(p[inside],max(self.dx,self.dy,self.dz))
        self.revision+=1
        return mass,energy

    def has_visible_smoke(self): return self.total_smoke_mass_kg>1e-9

    def active_render_bounds(self,_depth=None):
        if not self.has_visible_smoke():return None
        return self.active_minimum.copy(),self.active_maximum.copy()

    def step(self,dt_s):
        if dt_s<=0:return
        if (
            not self.has_visible_smoke()
            and not np.any(self.source_state)
            and not self.diagnostic_velocity_dirty
        ):
            return
        for p in self.programs.values():
            if "dt_s" in p:p["dt_s"]=dt_s
            if "air_density_kg_m3" in p:p["air_density_kg_m3"]=self.air_density_kg_m3
            if "ambient_temperature_k" in p:p["ambient_temperature_k"]=self.ambient_temperature_k
            if "background_wind_xz" in p:p["background_wind_xz"].value=tuple(self.background_wind_xz)
        self.source_texture.write(self.source_state.tobytes());self.source_state.fill(0)
        c=self.velocity_index;o=1-c
        self.state[0].use(0);self.u[c].use(1);self.v[c].use(2);self.w[c].use(3);self.source_texture.use(6)
        self._run("state_advect",self.cell_count,self.state[1])
        for name,fields,size in (
            ("u_advect",self.u,(self.nx+1,self.ny,self.nz)),
            ("v_advect",self.v,(self.nx,self.ny+1,self.nz)),
            ("w_advect",self.w,(self.nx,self.ny,self.nz+1))):
            fields[c].use(0);self._run(name,size,fields[o])
        self.state[1].use(0);self._run("state_evolve",self.cell_count,self.state[0])
        self.u[o].use(1);self.v[o].use(2);self.w[o].use(3)
        self._run("vorticity",self.cell_count,self.vorticity_texture)
        self.state[0].use(4);self.vorticity_texture.use(7)
        for name,fields,size in (
            ("u_force",self.u,(self.nx+1,self.ny,self.nz)),
            ("v_force",self.v,(self.nx,self.ny+1,self.nz)),
            ("w_force",self.w,(self.nx,self.ny,self.nz+1))):
            fields[o].use(0);self._run(name,size,fields[c])
        self.u[c].use(1);self.v[c].use(2);self.w[c].use(3)
        self._run("divergence",self.cell_count,self.divergence_texture)
        pi=self.last_pressure_index;self.divergence_texture.use(6)
        for _ in range(math.ceil(self.pressure_iterations_per_step/2)):
            po=1-pi;self.pressure[pi].use(5);self._run("pressure",self.cell_count,self.pressure[po]);pi=po
        self.pressure[pi].use(5)
        for name,fields,size in (
            ("project_u",self.u,(self.nx+1,self.ny,self.nz)),
            ("project_v",self.v,(self.nx,self.ny+1,self.nz)),
            ("project_w",self.w,(self.nx,self.ny,self.nz+1))):
            fields[c].use(0);self._run(name,size,fields[o])
        self.velocity_index=o;self.last_pressure_index=pi
        self.diagnostic_velocity_dirty = False
        self.total_smoke_mass_kg*=math.exp(-math.log(2)*dt_s/self.config.smoke_half_life_s)
        if self.has_visible_smoke():
            growth=np.array([(abs(self.background_wind_xz[0])+8)*dt_s,12*dt_s,
                             (abs(self.background_wind_xz[1])+8)*dt_s])
            self.active_minimum=np.maximum(self.active_minimum-growth,(self.x_min,self.y_min,self.z_min))
            self.active_maximum=np.minimum(self.active_maximum+growth,(self.x_max,self.y_max,self.z_max))
        self.revision+=1

    def readback_state(self):
        return np.frombuffer(self.state[0].read(alignment=1),np.float32).reshape(
            self.nz,self.ny,self.nx,2)

    def upload_velocity_fields(
        self, u: np.ndarray, v: np.ndarray, w: np.ndarray
    ) -> None:
        for texture in self.u:
            texture.write(np.asarray(u, dtype=np.float32).tobytes())
        for texture in self.v:
            texture.write(np.asarray(v, dtype=np.float32).tobytes())
        for texture in self.w:
            texture.write(np.asarray(w, dtype=np.float32).tobytes())
        self.velocity_index = 0
        self.diagnostic_velocity_dirty = True

    def readback_velocity_fields(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = self.velocity_index
        u = np.frombuffer(self.u[index].read(alignment=1), np.float32).reshape(
            self.nz, self.ny, self.nx + 1
        )
        v = np.frombuffer(self.v[index].read(alignment=1), np.float32).reshape(
            self.nz, self.ny + 1, self.nx
        )
        w = np.frombuffer(self.w[index].read(alignment=1), np.float32).reshape(
            self.nz + 1, self.ny, self.nx
        )
        return u, v, w
