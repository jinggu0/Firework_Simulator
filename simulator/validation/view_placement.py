"""Where a regression camera stands, checked against the built scene.

The capture gate measured clearance to the terrain and nothing else. A camera
eighteen metres above the ground passes that gate whether it hovers over a
street or stands on the fourth floor of a closed building shell, and the second
case renders the inside of the shell: every capture and metric taken from it
then describes a room the scene does not contain. `facade_landmark` did exactly
that from the day it was defined until V2-3c.

Buildings in the scene are extruded footprints — walls and a roof, no floor —
so a camera is inside one exactly when a building face lies directly above it.
A lens can also stand outside a wall yet close enough for the near plane to cut
through it, so the nearest face is measured as well, to the true closest point
of each triangle rather than to its plane.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .views import VisualRegressionView, VisualViewError


#: Crossings nearer than this to the ray origin are ignored, so a camera resting
#: exactly on a roof is not "under" it, and crossings closer than this to each
#: other are one crossing: a ray through the edge two triangles share hits both.
RAY_EPSILON_M = 1e-6

_UP = np.array([0.0, 1.0, 0.0])


def _triangles(vertices: np.ndarray) -> np.ndarray:
    values = np.asarray(vertices, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or len(values) % 3:
        raise ValueError(
            "building vertices must be whole triangles of (x, y, z, ...) rows"
        )
    return values[:, :3].reshape(-1, 3, 3)


def ray_crossings_m(
    origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Distances along a ray at which it crosses the triangles, ascending.

    Möller–Trumbore, vectorised over the triangles. A ray lying in a triangle's
    plane crosses nothing, which is what a vertical ray beside a wall must see.
    """

    start = np.asarray(origin, dtype=np.float64)
    heading = np.asarray(direction, dtype=np.float64)
    heading = heading / np.linalg.norm(heading)
    corner = triangles[:, 0]
    first_edge = triangles[:, 1] - corner
    second_edge = triangles[:, 2] - corner
    perpendicular = np.cross(heading, second_edge)
    determinant = np.einsum("ij,ij->i", first_edge, perpendicular)
    crossing = np.abs(determinant) > 1e-12
    inverse = np.divide(
        1.0, determinant, out=np.zeros_like(determinant), where=crossing
    )
    offset = start - corner
    u = np.einsum("ij,ij->i", offset, perpendicular) * inverse
    q = np.cross(offset, first_edge)
    v = (q @ heading) * inverse
    distance = np.einsum("ij,ij->i", second_edge, q) * inverse
    hit = (
        crossing
        & (u >= 0.0)
        & (v >= 0.0)
        & (u + v <= 1.0)
        & (distance > RAY_EPSILON_M)
    )
    ordered = np.sort(distance[hit])
    if ordered.size < 2:
        return ordered
    distinct = np.concatenate(([True], np.diff(ordered) > RAY_EPSILON_M))
    return ordered[distinct]


def building_faces_overhead_m(
    position_eus_m: tuple[float, float, float], building_vertices: np.ndarray
) -> np.ndarray:
    """Heights above the camera at which a building face crosses its vertical."""

    return ray_crossings_m(
        np.asarray(position_eus_m, dtype=np.float64),
        _UP,
        _triangles(building_vertices),
    )


def _closest_points(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Closest point of each triangle to `point` (Ericson, RTCD 5.1.5).

    The regions are assigned from lowest to highest priority, so each later
    assignment overrides an earlier one exactly where the scalar algorithm
    would have returned first.
    """

    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b - a, c - a
    ap, bp, cp = point - a, point - b, point - c
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0.0,
        )

    total = va + vb + vc
    closest = a + ab * ratio(vb, total)[:, None] + ac * ratio(vc, total)[:, None]
    edge_bc = (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0)
    along = ratio(d4 - d3, (d4 - d3) + (d5 - d6))
    closest[edge_bc] = (b + (c - b) * along[:, None])[edge_bc]
    edge_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    along = ratio(d2, d2 - d6)
    closest[edge_ac] = (a + ac * along[:, None])[edge_ac]
    vertex_c = (d6 >= 0.0) & (d5 <= d6)
    closest[vertex_c] = c[vertex_c]
    edge_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    along = ratio(d1, d1 - d3)
    closest[edge_ab] = (a + ab * along[:, None])[edge_ab]
    vertex_b = (d3 >= 0.0) & (d4 <= d3)
    closest[vertex_b] = b[vertex_b]
    vertex_a = (d1 <= 0.0) & (d2 <= 0.0)
    closest[vertex_a] = a[vertex_a]
    return closest


def nearest_building_face_m(
    position_eus_m: tuple[float, float, float], building_vertices: np.ndarray
) -> float:
    """Distance from the camera to the nearest point on any building face."""

    triangles = _triangles(building_vertices)
    if not len(triangles):
        return float("inf")
    point = np.asarray(position_eus_m, dtype=np.float64)
    closest = _closest_points(point, triangles)
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


def validate_view_placement(
    view: VisualRegressionView,
    building_vertices: np.ndarray,
    lens_clearance_m: float,
) -> dict[str, Any]:
    """Refuse a camera inside a building or with its lens against a wall.

    The measurements come back for a camera that passes, so a capture can
    record how close it stood rather than only that it was allowed.
    """

    overhead = building_faces_overhead_m(view.position_eus_m, building_vertices)
    if len(overhead):
        raise VisualViewError(
            f"view {view.view_id!r} camera stands inside a building: a "
            f"building face lies {overhead[0]:.2f} m directly above it"
        )
    nearest = nearest_building_face_m(view.position_eus_m, building_vertices)
    if nearest < lens_clearance_m:
        raise VisualViewError(
            f"view {view.view_id!r} camera is {nearest:.3f} m from a building "
            f"face, inside its {lens_clearance_m:.3f} m lens clearance"
        )
    return {
        "building_faces_overhead_m": [],
        "nearest_building_face_m": nearest,
        "lens_clearance_m": float(lens_clearance_m),
    }


#: The view file already carries camera coordinates to 0.1 mm.
_COORDINATE_DECIMALS = 4


def advance_out_of_buildings(
    view: VisualRegressionView,
    building_vertices: np.ndarray,
    terrain: Any,
    lens_clearance_m: float,
    maximum_advance_m: int = 500,
) -> tuple[VisualRegressionView, int]:
    """Walk a camera toward its target, in whole metres, until it stands clear.

    The target, the bearing and the camera's height above the terrain are all
    held, so the view keeps what it was defined to look at and from how high;
    only the building around or in front of the lens is left behind. `terrain`
    needs only ``height_at(x_m, z_m)``. A camera that is already clear comes
    back untouched with an advance of zero.
    """

    def clear(candidate: VisualRegressionView) -> bool:
        try:
            validate_view_placement(
                candidate, building_vertices, lens_clearance_m
            )
        except VisualViewError:
            return False
        return True

    if clear(view):
        return view, 0
    x0_m, y0_m, z0_m = view.position_eus_m
    clearance_m = round(
        y0_m - float(terrain.height_at(x0_m, z0_m)), _COORDINATE_DECIMALS
    )
    heading = np.array(
        [view.target_eus_m[0] - x0_m, view.target_eus_m[2] - z0_m],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(heading))
    if length == 0.0:
        raise VisualViewError(
            f"view {view.view_id!r} has no horizontal bearing to advance along"
        )
    heading /= length
    for advance_m in range(1, maximum_advance_m + 1):
        x_m = float(x0_m + heading[0] * advance_m)
        z_m = float(z0_m + heading[1] * advance_m)
        y_m = float(terrain.height_at(x_m, z_m)) + clearance_m
        position = (
            round(x_m, _COORDINATE_DECIMALS),
            round(y_m, _COORDINATE_DECIMALS),
            round(z_m, _COORDINATE_DECIMALS),
        )
        candidate = replace(view, position_eus_m=position)
        if clear(candidate):
            return candidate, advance_m
    raise VisualViewError(
        f"view {view.view_id!r} has no clear position within "
        f"{maximum_advance_m} m toward its target"
    )
