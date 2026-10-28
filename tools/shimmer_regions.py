"""Private geometry replay and disjoint spatial masks for shimmer measurements."""
from __future__ import annotations

import re

import moderngl
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from simulator import shaders
from simulator.passes.scene import VERTEX_LAYOUT


ASSIGNMENT = re.compile(r"frag_color\s*=\s*[^;]+;")


def semantic_fragment(source: str, kind: str) -> str:
    """Retain vertex interpolation and discard tests; replace colour outputs."""
    matches = list(ASSIGNMENT.finditer(source))
    expected = {"scene": 5, "land": 1, "water": 1}
    if kind not in expected or len(matches) != expected[kind]:
        raise ValueError("semantic fragment output contract changed")
    if kind == "scene" and "emission * landmark_emission_scale" not in matches[-1].group():
        raise ValueError("final facade output contract changed")
    for index, match in reversed(list(enumerate(matches))):
        value = "vec4(0.0, 0.0, 0.0, 1.0)"
        if kind == "water":
            value = "vec4(0.0, 0.0, 1.0, 1.0)"
        elif kind == "scene" and index == len(matches) - 1:
            value = "vec4(1.0, clamp(pane, 0.0, 1.0), 0.0, 1.0)"
        source = source[:match.start()] + "frag_color = " + value + ";" + source[match.end():]
    return source


def copy_uniforms(original, replay):
    for name in replay:
        target = replay[name]
        if not isinstance(target, moderngl.Uniform):
            continue
        if name not in original or target.gl_type != original[name].gl_type:
            raise ValueError(f"semantic uniform contract changed: {name}")
        raw = original[name].read()
        size = len(target.read())
        if size > len(raw):
            raise ValueError(f"semantic uniform array grew: {name}")
        target.write(raw[:size])


def scene_layout(program):
    formats, names = [], []
    for fmt, name, size in zip(VERTEX_LAYOUT[0].split(), VERTEX_LAYOUT[1:], (12, 12, 4, 8, 4), strict=True):
        formats.append(fmt if name in program else f"{size}x")
        if name in program:
            names.append(name)
    return " ".join(formats), *names


class SemanticReplay:
    """Render facade/pane/water fractions to a private MSAA target after capture.

    Red: the final window-bearing facade path. Green: its analytic pane mask.
    Blue: direct water coverage, including where reflected light is received.
    Special facade/roof paths and all other geometry are zero but still occlude.
    """
    def __init__(self, renderer):
        self.renderer, self.ctx = renderer, renderer.ctx
        self.programs, self.vaos, self.resources = {}, {}, []
        self.sources = {}
        try:
            for kind in ("scene", "land", "water"):
                source = semantic_fragment(shaders.source(f"{kind}.frag"), kind)
                self.sources[kind] = source
                self.programs[kind] = self.ctx.program(vertex_shader=shaders.source(f"{kind}.vert"), fragment_shader=source)
            scene = renderer.scene
            for vao, _ in scene.vaos:
                self.vaos[vao.glo] = self.ctx.vertex_array(self.programs["scene"],
                    [(scene.vertex_inputs[vao.glo], *scene_layout(self.programs["scene"]))])
            for vao, _, _, _ in scene.grass_vaos:
                self.vaos[vao.glo] = self.ctx.vertex_array(self.programs["scene"],
                    [(scene.vertex_inputs[vao.glo], *scene_layout(self.programs["scene"]))])
            land = renderer.land
            self.land = self.ctx.vertex_array(self.programs["land"],
                [(land.vertex_buffer, "2f", "in_xz")], land.index_buffer, index_element_size=4)
            self.resources.append(self.land)
            self.water = []
            for vertex, index in zip(renderer.water.buffers[::2], renderer.water.buffers[1::2], strict=True):
                vao = self.ctx.vertex_array(self.programs["water"], [(vertex, "2f", "in_xz")], index, index_element_size=4)
                self.water.append(vao)
                self.resources.append(vao)
            size = renderer.hdr_texture.size
            samples = renderer.targets.samples
            colour = self.ctx.texture(size, 4, dtype="f2", samples=samples if samples > 1 else 0)
            self.resources.append(colour)
            depth = self.ctx.depth_texture(size, samples=samples if samples > 1 else 0)
            self.resources.append(depth)
            self.target = self.ctx.framebuffer([colour], depth)
            self.resources.append(self.target)
            resolved = self.ctx.texture(size, 4, dtype="f2")
            self.resources.append(resolved)
            self.resolved = self.ctx.framebuffer([resolved])
            self.resources.append(self.resolved)
            self.check("initialisation")
        except BaseException:
            self.release()
            raise

    def check(self, stage):
        error = self.ctx.error
        if error != "GL_NO_ERROR":
            raise RuntimeError(f"semantic replay {stage}: {error}")

    def capture(self):
        renderer, ctx = self.renderer, self.ctx
        self.check("before capture")
        for kind, program in self.programs.items():
            copy_uniforms(getattr(renderer, kind).program, program)
            self.check(f"{kind} uniforms")
        with ctx.scope(framebuffer=self.target, enable_only=moderngl.DEPTH_TEST):
            self.target.viewport = (0, 0, *renderer.hdr_texture.size)
            self.target.scissor = None
            self.target.depth_mask = True
            ctx.depth_func = "<"
            self.target.clear(0, 0, 0, 1, depth=1)
            renderer.water_mask_texture.use(1)
            renderer.terrain_texture.use(2)
            renderer.scene.scanned_materials.bind()
            self.land.render(moderngl.TRIANGLES)
            self.check("land draw")
            for vao, count in renderer.scene.vaos:
                self.vaos[vao.glo].render(moderngl.TRIANGLES, vertices=count)
            for vao, count, centre, radius in renderer.scene.grass_vaos:
                if np.linalg.norm(centre - renderer.scene.camera_position_xz) <= renderer.scene.lod.blade_cutoff_m + radius:
                    self.vaos[vao.glo].render(moderngl.TRIANGLES, vertices=count)
            self.check("scene draw")
            for vao in self.water:
                vao.render(moderngl.TRIANGLES)
            self.check("water draw")
            ctx.copy_framebuffer(self.resolved, self.target)
            self.check("resolve")
            width, height = renderer.hdr_texture.size
            raw = self.resolved.read(components=3, dtype="f2", alignment=1)
        self.check("readback and scope restore")
        return np.flipud(np.frombuffer(raw, np.float16).reshape(height, width, 3)).copy()

    def release(self):
        for vao in self.vaos.values():
            vao.release()
        for resource in reversed(self.resources):
            resource.release()
        for program in self.programs.values():
            program.release()


def region_masks(fractions: np.ndarray, edge_radius: int) -> dict[str, np.ndarray]:
    """Classify whole-path support, never just the first pose. Disjoint masks."""
    if fractions.ndim != 4 or fractions.shape[-1] != 3 or len(fractions) < 2:
        raise ValueError("expected frame x height x width x 3 fractions")
    if not isinstance(edge_radius, int) or edge_radius < 1:
        raise ValueError("edge_radius must be a positive integer")
    if not np.isfinite(fractions).all() or fractions.min() < -0.001 or fractions.max() > 1.001:
        raise ValueError("invalid semantic fractions")
    facade, pane, water = (fractions[..., index] for index in range(3))
    if np.any(pane > facade + 0.001) or np.any(facade + water > 1.001):
        raise ValueError("inconsistent semantic channels")
    boundary = np.zeros(facade.shape[1:], dtype=bool)
    for frame in facade:
        body = frame >= .5
        boundary |= binary_dilation(body) ^ binary_erosion(body, border_value=0)
        boundary |= (frame > .001) & (frame < .999)
    boundary = binary_dilation(boundary, iterations=edge_radius)
    interior = np.all(facade >= .999, axis=0) & ~boundary
    masks = {"facade_boundary_band": boundary,
             "pane_interior": interior & np.all(pane >= .999, axis=0),
             "wall_interior": interior & np.all(pane <= .001, axis=0),
             "pane_transition": interior & ~np.all(pane >= .999, axis=0) & ~np.all(pane <= .001, axis=0),
             "water_interior": np.all(water >= .999, axis=0) & ~boundary}
    union = np.logical_or.reduce(list(masks.values()))
    masks["other_or_moving_support"] = ~union
    return masks


def region_statistics(run, fractions, edge_radius):
    masks = region_masks(fractions, edge_radius)
    scored = run["scored"]
    display = np.maximum(run["display_score"] - 2.0, 0)
    hdr = np.maximum(run["hdr_score"] - .05, 0)
    totals = {"display": float(display[scored].sum()), "hdr": float(hdr[scored].sum())}
    regions = {}
    for name, mask in masks.items():
        selected = scored & mask
        pixels = int(selected.sum())
        regions[name] = {"pixels": pixels, "pixel_fraction": float(pixels / scored.sum()) if scored.any() else 0}
        for key, values in (("display", display), ("hdr", hdr)):
            mass = float(values[selected].sum())
            regions[name][f"{key}_excess_mean"] = mass / pixels if pixels else 0
            regions[name][f"{key}_excess_share"] = mass / totals[key] if totals[key] else 0
    return {"edge_radius_px": edge_radius, "scored_pixels": int(scored.sum()), "regions": regions}
