from __future__ import annotations

import math

import numpy as np

from .config import AtmosphereConfig, SmokeConfig

GRAVITY_MPS2 = 9.80665
AIR_HEAT_CAPACITY_J_KG_K = 1005.0


class SmokeFluid2D:
    """A staggered-grid vertical slice of the post-blast, low-Mach plume.

    Density and temperature excess live at cell centres. Horizontal and
    vertical velocities live on their corresponding cell faces (a MAC grid).
    This intentionally starts after the compressible shock has expanded; a
    pressure projection is physically appropriate only for the slower plume.
    """

    def __init__(
        self, config: SmokeConfig, atmosphere: AtmosphereConfig
    ) -> None:
        self.config = config
        self.nx, self.ny = config.grid_size
        self.x_min, self.x_max, self.y_min, self.y_max = config.bounds_m
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.density_kg_m3 = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.temperature_excess_k = np.zeros_like(self.density_kg_m3)
        self.u_mps = np.zeros((self.ny, self.nx + 1), dtype=np.float32)
        self.v_mps = np.zeros((self.ny + 1, self.nx), dtype=np.float32)
        self.pressure_pa = np.zeros_like(self.density_kg_m3)
        self.ambient_temperature_k = atmosphere.temperature_k
        self.air_density_kg_m3 = atmosphere.air_density_kg_m3
        self.background_wind_mps = float(atmosphere.wind_velocity_mps[0])
        self.u_mps.fill(self.background_wind_mps)
        self._cell_x, self._cell_y = np.meshgrid(
            self.x_min + (np.arange(self.nx, dtype=np.float32) + 0.5) * self.dx,
            self.y_min + (np.arange(self.ny, dtype=np.float32) + 0.5) * self.dy,
        )
        self.revision = 0

    def set_atmosphere(self, atmosphere: AtmosphereConfig) -> None:
        self.ambient_temperature_k = atmosphere.temperature_k
        self.air_density_kg_m3 = atmosphere.air_density_kg_m3
        self.background_wind_mps = float(atmosphere.wind_velocity_mps[0])

    def inject_burst(
        self,
        position_m: np.ndarray,
        smoke_mass_kg: float,
        thermal_energy_j: float,
    ) -> None:
        radius = self.config.source_radius_m
        radius2 = radius * radius
        weights = np.exp(
            -0.5
            * (
                (self._cell_x - float(position_m[0])) ** 2
                + (self._cell_y - float(position_m[1])) ** 2
            )
            / radius2
        ).astype(np.float32)
        volume_weights = weights * self.dx * self.dy * self.config.plume_depth_m
        represented_volume = float(volume_weights.sum())
        if represented_volume <= 1e-9:
            return

        density_increment = weights * (smoke_mass_kg / represented_volume)
        heated_air_mass = max(
            self.air_density_kg_m3 * represented_volume, 1e-6
        )
        temperature_increment = min(
            thermal_energy_j / (heated_air_mass * AIR_HEAT_CAPACITY_J_KG_K),
            self.config.max_temperature_excess_k,
        )
        self.density_kg_m3 += density_increment
        self.temperature_excess_k += weights * temperature_increment
        self.revision += 1

    def inject_particles(
        self,
        positions_m: np.ndarray,
        smoke_mass_kg: np.ndarray,
        thermal_energy_j: np.ndarray,
    ) -> tuple[float, float]:
        """Conservative finite-volume deposit; returns accepted mass/energy."""

        if not len(positions_m):
            return 0.0, 0.0
        ix = np.floor(
            (positions_m[:, 0] - self.x_min) / self.dx
        ).astype(np.int32)
        iy = np.floor(
            (positions_m[:, 1] - self.y_min) / self.dy
        ).astype(np.int32)
        inside = (
            (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        )
        if not np.any(inside):
            return 0.0, 0.0
        cell_volume_m3 = self.dx * self.dy * self.config.plume_depth_m
        accepted_mass = float(smoke_mass_kg[inside].sum(dtype=np.float64))
        accepted_energy = float(thermal_energy_j[inside].sum(dtype=np.float64))
        indices = iy[inside] * self.nx + ix[inside]
        density_flat = self.density_kg_m3.ravel()
        temperature_flat = self.temperature_excess_k.ravel()
        density_flat += (
            np.bincount(
                indices,
                weights=smoke_mass_kg[inside],
                minlength=self.nx * self.ny,
            ) / cell_volume_m3
        ).astype(np.float32)
        temperature_flat += (
            np.bincount(
                indices,
                weights=thermal_energy_j[inside],
                minlength=self.nx * self.ny,
            )
            / (
                self.air_density_kg_m3
                * cell_volume_m3
                * AIR_HEAT_CAPACITY_J_KG_K
            )
        ).astype(np.float32)
        np.minimum(
            self.temperature_excess_k,
            self.config.max_temperature_excess_k,
            out=self.temperature_excess_k,
        )
        self.revision += 1
        return accepted_mass, accepted_energy

    def divergence(self) -> np.ndarray:
        return (
            (self.u_mps[:, 1:] - self.u_mps[:, :-1]) / self.dx
            + (self.v_mps[1:, :] - self.v_mps[:-1, :]) / self.dy
        )

    def project(self, dt_s: float) -> None:
        rhs = self.air_density_kg_m3 * self.divergence() / dt_s
        rhs -= rhs.mean(dtype=np.float64)
        pressure = self.pressure_pa
        pressure.fill(0.0)
        dx2, dy2 = self.dx * self.dx, self.dy * self.dy
        denominator = 2.0 * (dx2 + dy2)
        for _ in range(self.config.pressure_iterations):
            padded = np.pad(pressure, 1, mode="edge")
            pressure = (
                (padded[1:-1, :-2] + padded[1:-1, 2:]) * dy2
                + (padded[:-2, 1:-1] + padded[2:, 1:-1]) * dx2
                - rhs * dx2 * dy2
            ) / denominator
            pressure -= pressure.mean(dtype=np.float64)
        self.pressure_pa[:] = pressure
        self.u_mps[:, 1:-1] -= (
            dt_s
            * (pressure[:, 1:] - pressure[:, :-1])
            / (self.air_density_kg_m3 * self.dx)
        )
        self.v_mps[1:-1, :] -= (
            dt_s
            * (pressure[1:, :] - pressure[:-1, :])
            / (self.air_density_kg_m3 * self.dy)
        )
        self._boundaries()

    @staticmethod
    def _sample(
        field: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        origin_x: float,
        origin_y: float,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        gx = np.clip((x - origin_x) / dx, 0.0, field.shape[1] - 1.0001)
        gy = np.clip((y - origin_y) / dy, 0.0, field.shape[0] - 1.0001)
        x0, y0 = np.floor(gx).astype(np.int32), np.floor(gy).astype(np.int32)
        x1 = np.minimum(x0 + 1, field.shape[1] - 1)
        y1 = np.minimum(y0 + 1, field.shape[0] - 1)
        tx, ty = gx - x0, gy - y0
        return (
            field[y0, x0] * (1.0 - tx) * (1.0 - ty)
            + field[y0, x1] * tx * (1.0 - ty)
            + field[y1, x0] * (1.0 - tx) * ty
            + field[y1, x1] * tx * ty
        )

    def _centred_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            0.5 * (self.u_mps[:, :-1] + self.u_mps[:, 1:]),
            0.5 * (self.v_mps[:-1, :] + self.v_mps[1:, :]),
        )

    def _velocity_at(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        uc, vc = self._centred_velocity()
        origin_x = self.x_min + 0.5 * self.dx
        origin_y = self.y_min + 0.5 * self.dy
        return (
            self._sample(uc, x, y, origin_x, origin_y, self.dx, self.dy),
            self._sample(vc, x, y, origin_x, origin_y, self.dx, self.dy),
        )

    def _backtrace(
        self, x: np.ndarray, y: np.ndarray, dt_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        u0, v0 = self._velocity_at(x, y)
        mid_x, mid_y = x - 0.5 * dt_s * u0, y - 0.5 * dt_s * v0
        um, vm = self._velocity_at(mid_x, mid_y)
        return x - dt_s * um, y - dt_s * vm

    def _advect(self, dt_s: float) -> None:
        previous_density = self.density_kg_m3.copy()
        previous_temperature = self.temperature_excess_k.copy()
        previous_u, previous_v = self.u_mps.copy(), self.v_mps.copy()

        bx, by = self._backtrace(self._cell_x, self._cell_y, dt_s)
        cell_origin_x = self.x_min + 0.5 * self.dx
        cell_origin_y = self.y_min + 0.5 * self.dy
        self.density_kg_m3[:] = self._sample(
            previous_density, bx, by, cell_origin_x, cell_origin_y, self.dx, self.dy
        )
        self.temperature_excess_k[:] = self._sample(
            previous_temperature, bx, by,
            cell_origin_x, cell_origin_y, self.dx, self.dy
        )

        ux, uy = np.meshgrid(
            self.x_min + np.arange(self.nx + 1, dtype=np.float32) * self.dx,
            self.y_min + (np.arange(self.ny, dtype=np.float32) + 0.5) * self.dy,
        )
        bx, by = self._backtrace(ux, uy, dt_s)
        self.u_mps[:] = self._sample(
            previous_u, bx, by, self.x_min, cell_origin_y, self.dx, self.dy
        )
        vx, vy = np.meshgrid(
            self.x_min + (np.arange(self.nx, dtype=np.float32) + 0.5) * self.dx,
            self.y_min + np.arange(self.ny + 1, dtype=np.float32) * self.dy,
        )
        bx, by = self._backtrace(vx, vy, dt_s)
        self.v_mps[:] = self._sample(
            previous_v, bx, by, cell_origin_x, self.y_min, self.dx, self.dy
        )

    def _forces(self, dt_s: float) -> None:
        temperature_faces = 0.5 * (
            self.temperature_excess_k[:-1, :] + self.temperature_excess_k[1:, :]
        )
        density_faces = 0.5 * (
            self.density_kg_m3[:-1, :] + self.density_kg_m3[1:, :]
        )
        # Boussinesq approximation: hot displaced air rises, particulate
        # loading contributes a small opposing gravitational acceleration.
        buoyancy = GRAVITY_MPS2 * (
            temperature_faces / self.ambient_temperature_k
            - density_faces / self.air_density_kg_m3
        )
        self.v_mps[1:-1, :] += buoyancy * dt_s

        uc, vc = self._centred_velocity()
        dvdx = np.gradient(vc, self.dx, axis=1)
        dudy = np.gradient(uc, self.dy, axis=0)
        curl = dvdx - dudy
        grad_y, grad_x = np.gradient(np.abs(curl), self.dy, self.dx)
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-7
        force_x = self.config.vorticity_confinement * self.dx * grad_y / norm * curl
        force_y = -self.config.vorticity_confinement * self.dy * grad_x / norm * curl
        self.u_mps[:, 1:-1] += 0.5 * (
            force_x[:, :-1] + force_x[:, 1:]
        ) * dt_s
        self.v_mps[1:-1, :] += 0.5 * (
            force_y[:-1, :] + force_y[1:, :]
        ) * dt_s

    @staticmethod
    def _diffuse(field: np.ndarray, coefficient: float, dt_s: float,
                 dx: float, dy: float) -> None:
        if coefficient <= 0.0:
            return
        padded = np.pad(field, 1, mode="edge")
        laplacian = (
            (padded[1:-1, :-2] - 2.0 * field + padded[1:-1, 2:]) / (dx * dx)
            + (padded[:-2, 1:-1] - 2.0 * field + padded[2:, 1:-1]) / (dy * dy)
        )
        field += coefficient * dt_s * laplacian

    def _boundaries(self) -> None:
        self.u_mps[:, 0] = self.background_wind_mps
        self.u_mps[:, -1] = self.u_mps[:, -2]
        self.v_mps[0, :] = 0.0
        self.v_mps[-1, :] = self.v_mps[-2, :]

    def step(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        self._advect(dt_s)
        self._forces(dt_s)
        self._diffuse(
            self.density_kg_m3, self.config.smoke_diffusivity_m2_s,
            dt_s, self.dx, self.dy
        )
        self._diffuse(
            self.temperature_excess_k, self.config.thermal_diffusivity_m2_s,
            dt_s, self.dx, self.dy
        )
        self.density_kg_m3 *= math.exp(
            -math.log(2.0) * dt_s / self.config.smoke_half_life_s
        )
        self.temperature_excess_k *= math.exp(
            -math.log(2.0) * dt_s / self.config.thermal_half_life_s
        )
        self.project(dt_s)
        np.maximum(self.density_kg_m3, 0.0, out=self.density_kg_m3)
        np.maximum(self.temperature_excess_k, 0.0, out=self.temperature_excess_k)
        self.revision += 1
