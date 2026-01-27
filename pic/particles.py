from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid
import astropy.units as u
from scipy.constants import e
from pic.MCC import MCC, InterspeciesMCC
from pic.functions import (
    max_vect3D_multiple_T,
    particle_to_grid,
    popout,
    popout_weighted,
    max_vect,
    max_vect3D,
    numba_return_part_diag,
    numba_return_part_diag_weighted,
    numba_interp1D_normed,
    remove,
    remove_array,
    remove_array_weighted,
    remove_weighted,
    remove_random,
    remove_random_weighted,
    find_cells,
    random_round,
)


class ParticlesGroup:
    """a SOA containing the data of a specie of tracked particules"""

    def __init__(self, m: float, charge: float, start, symbol, plasma) -> None:
        self.m: float = m
        self.charge: float = charge
        self.Npart: int = 0
        self.plasma = plasma
        self.n = np.zeros((plasma.N_cells), dtype="float64")
        self.u = np.zeros((plasma.N_cells, 3), dtype="float64", order="F")
        self.cells = np.zeros((plasma.N_cells), dtype="int64")
        self.mccs: list[MCC] = []
        self.imccs: list[InterspeciesMCC] = []
        self.need_u: bool = False
        self.need_c: bool = False
        self.start = start
        self.symbol = symbol

    def init_part(self) -> None:
        match self.start:
            case (T, n):
                N = int(n.to_value(u.m**-3) * self.plasma.Lx / self.plasma.initial_qf)
                self.init_uniform(N, T.to_value(u.eV), self.plasma.initial_qf)
            case (x, n, T, v, T_inj):
                x = x.to_value(u.m)
                n = n.to_value(u.m**-3)
                T = T.to_value(u.eV)
                if v is not None:
                    v = v.to_value(u.m / u.s)
                n_l = np.trapz(n, x)
                N = int(n_l / self.plasma.initial_qf)
                self.init_with_profile(N, x, n, T, v)
                if T_inj is not None:
                    self.T = T_inj.to_value(u.eV)
        self.E_interp = np.zeros(N, dtype="float64", order="F")

    def init_uniform(self, N, T, qf):
        """Generate uniform particle, with maxwellian stuff"""

        self.x = np.linspace(0, self.plasma.Lx, N)

        assert all(self.x >= 0)
        assert all(self.x <= self.plasma.Lx)

        self.V = max_vect3D(N, T, self.m)

        self.c = np.zeros_like(self.V)
        self.Npart = N
        self.T = T
        self.w = np.ones(N, dtype="float64") * qf   #adding weight

    def init_with_profile(self, N, x, n, T, v, qf):
        cdf = cumulative_trapezoid(n, x, initial=0.0)
        cdf /= cdf[-1]
        self.x = np.interp(np.random.rand(N), cdf, x)
        self.V = max_vect3D_multiple_T(N, np.interp(self.x, x, T), self.m)
        if v is not None:
            self.V[:, 0] += np.interp(self.x, x, v)
        self.c = np.zeros_like(self.V)
        self.Npart = N
        self.T = np.mean(T)

        self.w = np.ones(N, dtype="float64") * qf   #adding weight

    def init_restart(self, N, x, V, w):
        self.x = x
        self.V = V
        self.w = w
        self.c = np.zeros_like(self.V)
        self.Npart = N
        self.E_interp = np.zeros(N, dtype="float64", order="F")
        match self.start:
            case (T, _):
                self.T = T
            case (_x, _n, _T, _v, T_inj):
                if T_inj is not None:
                    self.T = T_inj.to_value(u.eV)

    def add_particles(self, x: np.ndarray, V: np.ndarray, w: np.ndarray) -> None:
        N = x.shape[0]
        if N > 0:
            while self.x.shape[0] <= self.Npart + N:
                # new_size = int(self.x.shape[0] * 1.1)
                new_size = self.x.shape[0] + 1000
                new_x = np.zeros((new_size), dtype="float64")
                new_x[: self.Npart] = self.x[: self.Npart]
                self.x = new_x
                new_V = np.zeros((new_size, 3), dtype="float64", order="F")
                new_V[: self.Npart, :] = self.V[: self.Npart, :]
                self.V = new_V
                new_E_interp = np.zeros((new_size), dtype="float64")
                new_E_interp[: self.Npart] = self.E_interp[: self.Npart]
                self.E_interp = new_E_interp
                #adding weights
                new_w = np.zeros((new_size), dtype="float64")
                new_w[: self.Npart] = self.w[: self.Npart]
                self.w = new_w

            # Fill the last elements with the new ones
            Nmin = self.Npart
            Nmax = self.Npart + N

            self.x[Nmin:Nmax] = x

            self.V[Nmin:Nmax, :] = V

            self.w[Nmin:Nmax] = w

            self.Npart += N

    def update_density(self, tabx):
        """interpolate the density"""

        self.n.fill(0)
        partx = self.x[: self.Npart]
        weight = self.w[: self.Npart]
        #return numba_return_part_diag(
        #   self.Npart, partx, partx, tabx, self.n, self.plasma.dx, power=0
        #)

        return numba_return_part_diag_weighted(
           self.Npart, partx, partx, weight, tabx, self.n, self.plasma.dx, power=0
        )

    def update_u(self, tabx):
        self.u.fill(0)
        for k in range(3):
            numba_return_part_diag_weighted(
                self.Npart,
                self.x[: self.Npart],
                self.V[: self.Npart, k],
                self.w[: self.Npart],
                tabx,
                self.u[:, k],
                self.plasma.dx, 
                power=1
            )            
            # particle_to_grid(
            #     self.Npart,
            #     self.x[: self.Npart],
            #     self.V[: self.Npart, k],
            #     tabx,
            #     self.u[:, k],
            #     self.plasma.dx,
            # )
            np.divide(self.u[:, k], self.n, where=self.n != 0, out=self.u[:, k])

    def update_c(self):
        if self.c.shape[0] < self.Npart:
            self.c = np.zeros((self.Npart, 3), dtype="float64", order="F")
        self.c[: self.Npart] = self.V[: self.Npart]
        for k in range(3):
            self.c[: self.Npart, k] -= numba_interp1D_normed(
                self.x[: self.Npart] / self.plasma.dx, self.u[:, k]
            )

    def remove_parts(self, Lx, absorbtion, bounds=["w", "w"]):
        """remove the pariclues thar are outside of the systeme, right or left

        The boundary arguments will be used to differe between wall and center
        (mirror)
        """
        (left, right) = popout_weighted(
            self.x[: self.Npart], self.V[: self.Npart], self.w[: self.Npart], Lx, absorbtion
        )        
        # (left, right) = popout(
        #     self.x[: self.Npart], self.V[: self.Npart], Lx, absorbtion
        # )
        self.flux = (left, right)
        compt = left + right
        self.Npart -= compt

        return compt

    def remove_index(self, Idxs):
        if isinstance(Idxs, np.ndarray):
            # self.Npart = remove_array(Idxs, self.x, self.V, self.Npart)
            self.Npart = remove_array_weighted(Idxs, self.x, self.V, self.w, self.Npart)
        else:
            # self.Npart = remove(Idxs, self.x, self.V, self.Npart)
            self.Npart = remove_weighted(Idxs, self.x, self.V, self.w, self.Npart)

    def remove_random(self, N_random):
        # self.Npart = remove_random(self.x, self.V, self.Npart, N_random)
        self.Npart = remove_random_weighted(self.x, self.V, self.w, self.Npart, N_random)


    # MCC methods are considred not impacted by the addition of weights
    def add_mcc(self, mcc):
        self.mccs.append(mcc)

    def add_imcc(self, imcc):
        self.imccs.append(imcc)

    def apply_mcc(self, dT):
        for mcc in self.mccs:
            mcc.scatter(dT)
        for imcc in self.imccs:
            imcc.scatter(dT)

    def coarse_sort(self):
        """sort the particles by cell and updaite the indexes in self.cells"""
        #coarse_buble_sort(self.x, self.V, self.Npart, self.cells, self.plasma.dx)
        p = (self.x[: self.Npart] / self.plasma.dx).astype(int).argsort(kind="stable")
        self.x[: self.Npart] = self.x[p]
        self.V[: self.Npart] = self.V[p]
        self.w[: self.Npart] = self.w[p]
        find_cells(self.x[: self.Npart], self.cells, self.plasma.dx)

    def partners(self, x):
        """return the indexes bounds of the particles that are in the same cell as i"""
        x_cells = (x / self.plasma.dx).astype(int)
        index_min = self.cells[x_cells]
        index_max = self.cells[x_cells + 1]
        return (
            np.random.rand(x.shape[0]) * (index_max - index_min) + index_min
        ).astype(int)

    def inject_same_speed(self, n0, v0):
        dt = self.plasma.dT
        N = random_round(n0 * v0 * dt / self.plasma.qf)
        v_temp = np.zeros((N, 3), dtype="float64", order="F")
        R = np.random.rand(N)
        v_temp[:, 0] = (
            v0 + self.charge * (self.plasma.E[0, 0] / self.m) * (R - 0.5) * dt
        )

        x_temp = R * v0 * dt + self.charge * self.plasma.E[0, 0] * self.plasma.dT**2 / (
            2 * self.m
        )

        w_temp = np.ones(N, dtype="float64") * self.plasma.initial_qf

        self.add_particles(x_temp, v_temp, w_temp)

    def inject_maxwellian_flux(self, n0, T0):
        N = random_round(
            n0 * np.sqrt(T0 / (2 * np.pi * self.m)) * self.plasma.dT / self.plasma.qf
        )
        v_temp = np.zeros((N, 3), dtype="float64", order="F")
        v_temp[:, 1] = max_vect(N, T0, self.m)
        v_temp[:, 2] = max_vect(N, T0, self.m)
        v_temp[:, 0] = np.sqrt(-e * T0 * np.log(np.random.rand(N)) / self.m)
        x_temp = np.random.rand(N) * v_temp[:, 0] * self.plasma.dT
        w_temp = np.ones(N, dtype="float64") * self.plasma.initial_qf
        self.add_particles(x_temp, v_temp, w_temp)

    def boris_push(self, E, Bz, dT):
        factor = self.charge * dT / (2 * self.m)
        particles_E = numba_interp1D_normed(self.x[: self.Npart] / self.plasma.dx, E)
        v_minus = self.V[: self.Npart] + factor * particles_E
        t = factor * numba_interp1D_normed(self.x[: self.Npart] / self.plasma.dx, Bz)
        s = 2 * t / (1 + t**2)
        v_prime = v_minus + np.cross(v_minus, t)
        v_plus = v_minus + np.cross(v_prime, s)
        self.V[: self.Npart] = v_plus + factor * particles_E
