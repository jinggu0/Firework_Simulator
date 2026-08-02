"""Render passes, each owning its own GPU resources.

``Renderer`` was a single 463-line constructor that allocated every program,
buffer, texture, and framebuffer in the engine, followed by a 190-line draw
method. Splitting it per pass means adding a material system or a second
display transform touches one file rather than threading through all of them.

``Renderer`` remains the coordinator: pass ordering, the reflection pre-pass,
and the uniforms that several passes share (camera, ambient scale, clustered
lights) are genuinely cross-pass concerns and stay there.
"""

from .ambient_occlusion import AmbientOcclusionPass
from .haze import HazePass, set_air_extinction
from .land import LandPass, initialise_static_lights
from .particles import STRIDE, ParticlePass
from .post import PostProcessPass
from .scene import SceneData, ScenePass
from .sky import SkyPass
from .smoke import SmokePass
from .targets import RenderTargets
from .water import WaterPass, wind_from_bearing_deg

__all__ = [
    "HazePass",
    "AmbientOcclusionPass",
    "LandPass",
    "ParticlePass",
    "PostProcessPass",
    "RenderTargets",
    "STRIDE",
    "SceneData",
    "ScenePass",
    "SkyPass",
    "SmokePass",
    "WaterPass",
    "initialise_static_lights",
    "set_air_extinction",
    "wind_from_bearing_deg",
]
