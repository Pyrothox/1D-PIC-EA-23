from __future__ import annotations

import numpy as np
from astropy.constants import e
from pic.functions import numba_return_part_diag
import astropy.units as u

from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from pic import Plasma
    from pic.diagnostics import MeasureLossesYZ
    from pic.particles import ParticlesGroup
e = e.si.value


class AbstractWall:
    def __init__(
        self, electrons: ParticlesGroup, ions: List[ParticlesGroup], plasma: Plasma
    ):
        self.ions = ions
        self.electrons = electrons
        self.plasma = plasma
        self.rng = np.random.default_rng()
        self.elec_diag: Union[None, MeasureLossesYZ] = None
        self.ion_diags: List[Union[None, MeasureLossesYZ]] = [None] * len(ions)
        self.args_list = [None] * len(ions)

    def ions_to_remove(self, ions: ParticlesGroup, *args):
        pass

    def prepare(self):
        pass

    def absorbtion(self):
        self.prepare()
        removed_ions = 0
        for ions, args, diag in zip(self.ions, self.args_list, self.ion_diags):
            Idxs = self.ions_to_remove(ions, *args)
            if len(Idxs) > 0:
                removed_ions += len(Idxs)
                if diag:
                    diag.accumulate(ions.x[Idxs], ions.w[Idxs])
                ions.remove_index(Idxs)
        if removed_ions:
            Idxs = np.argpartition(
                np.sum(np.square(self.electrons.V[: self.electrons.Npart]), axis=1),
                self.electrons.Npart - removed_ions,
            )[-removed_ions:]
            if self.elec_diag:
                self.elec_diag.accumulate(self.electrons.x[Idxs], self.electrons.w[Idxs])
            self.electrons.remove_index(Idxs)

    def update(self):
        pass

    def add_diagnostic(self, diag: MeasureLossesYZ, particles: ParticlesGroup):
        if particles is self.electrons:
            self.elec_diag = diag
            return
        try:
            self.ion_diags[self.ions.index(particles)] = diag
        except ValueError:
            raise ValueError(
                f"Particles {particles.symbol} not found in the ions list. Is it a negative ion?"
            )


class Wall(AbstractWall):
    """A class to model the particle loss on the y and z directions."""

    def __init__(
        self,
        ly: float,
        lz: float,
        h2d: float,
        electrons: ParticlesGroup,
        ions: List[ParticlesGroup],
        plasma: Plasma,
    ):
        super().__init__(electrons, ions, plasma)
        self.sqrt_Te = 0.0
        self.args_list = [
            (2 * h2d * (1 / ly + 1 / lz) / np.sqrt(ion.m),) for ion in ions
        ]
        print(f"Wall losses with {h2d=} initialized.")

    def prepare(self):
        v_squared_xz = np.sum(
            np.square(self.electrons.V[: self.electrons.Npart]), axis=1
        )
        electron_teperature = numba_return_part_diag(
            self.electrons.Npart,
            self.electrons.x,
            v_squared_xz,
            self.plasma.x_j,
            np.zeros_like(self.plasma.x_j),
            self.plasma.x_j[1],
            1,
        )
        electron_teperature = np.divide(
            electron_teperature,
            self.electrons.n,
            out=np.zeros_like(electron_teperature),
            where=self.electrons.n != 0,
        )
        electron_teperature *= self.electrons.m * 1 / 3
        self.sqrt_Te = np.sqrt(np.mean(electron_teperature.mean()))

    def ions_to_remove(self, ions: ParticlesGroup, factor):
        p = 1 - np.exp(-self.sqrt_Te * factor * self.plasma.dT)
        n_coll = p * ions.Npart
        part = n_coll - np.floor(n_coll)
        n_coll = int(n_coll) + (1 if np.random.rand() < part else 0)
        dxs = self.rng.choice(ions.Npart, size=n_coll, replace=False)
        return dxs


class WallProfile(AbstractWall):
    """A class to model the particle loss on the y and z directions with a given neutral profile."""

    def __init__(
        self,
        ly: float,
        lz: float,
        electrons: ParticlesGroup,
        ions: List[ParticlesGroup],
        plasma: Plasma,
        corr=1.0,
    ) -> None:
        self.ly = ly
        self.lz = lz
        self.corr = corr
        self.c_s = 2 * (1 / self.ly + 1 / self.lz)
        super().__init__(electrons, ions, plasma)

        def temp_init(start):
            match start:
                case (T, _):
                    T0 = np.full_like(self.plasma.x_j, T.to_value(u.J))
                case (x, _, T, _, _):
                    T0 = np.interp(self.plasma.x_j, x.to_value(u.m), T.to_value(u.J))
            return T0

        Te = temp_init(electrons.start)
        nus = (self.compute_nu(Te, temp_init(ion.start), ion) for ion in ions)
        self.args_list = [(nu, np.max(nu)) for nu in nus]
        print("Wall losses with profile initialized.")

    def h2d(self, electon_temperature, ion_temperature, mean_free_path):
        hy = 0.55 / np.sqrt(
            3
            + 0.5 * self.lz / mean_free_path
            + 0.2
            * (ion_temperature / electon_temperature)
            * (self.lz / mean_free_path) ** 2
        )
        hz = 0.55 / np.sqrt(
            3
            + 0.5 * self.ly / mean_free_path
            + 0.2
            * (ion_temperature / electon_temperature)
            * (self.ly / mean_free_path) ** 2
        )

        return self.corr * (self.ly * hy + self.lz * hz) / (self.ly + self.lz)

    def update(self):
        elec_temp = get_temperature(self.plasma.diag_values, self.electrons)

        nus = (
            self.compute_nu(
                elec_temp, get_temperature(self.plasma.diag_values, ion), ion
            )
            for ion in self.ions
        )
        self.args_list = [(nu, np.max(nu)) for nu in nus]

    def compute_nu(
        self, elec_temp: np.ndarray, ion_temp: np.ndarray, ion: ParticlesGroup
    ):
        # mu = self.m_n * self.m_i / (self.m_n + self.m_i)
        # Tr = (self.m_i * self.Tn + self.m_n * ion_temp) / (self.m_n + self.m_i)
        # mfp_profile = mean_free_path(self.nn, Tr, self.en_tot, self.cs_tot)

        inverse_mfp = sum(
            mcc.inverse_mean_free_path(ion_temp, self.plasma.x_j) for mcc in ion.mccs
        )
        mfp_profile = 1 / inverse_mfp
        h2 = self.h2d(elec_temp, ion_temp, mfp_profile)
        return np.sqrt(elec_temp) * self.c_s / np.sqrt(ion.m) * h2

    def ions_to_remove(self, ions: ParticlesGroup, nu, nu_max) -> int:
        p = 1 - np.exp(-nu_max * self.plasma.dT)
        # number of candidate particles
        n_coll = p * ions.Npart
        part = n_coll - np.floor(n_coll)
        n_coll = int(n_coll) + (1 if np.random.rand() < part else 0)

        # choice of candidates
        Idxs = self.rng.choice(ions.Npart, size=n_coll, replace=False)
        r = self.rng.random(Idxs.shape[0]) * nu_max
        will_collide = r < np.interp(ions.x[Idxs], self.plasma.x_j, nu)
        return Idxs[will_collide]


class WallProfileElectronegative(AbstractWall):
    """A class to model the particle loss on the y and z directions with a given neutral profile."""

    def __init__(
        self,
        ly: float,
        lz: float,
        electrons: ParticlesGroup,
        pos_ions: list[ParticlesGroup],
        neg_ions: ParticlesGroup,
        plasma: Plasma,
        corr=1.0,
    ) -> None:
        self.neg_ions = neg_ions

        super().__init__(electrons, pos_ions, plasma)
        self.ly = ly
        self.lz = lz
        self.corr = corr
        self.c_s = 2 * (1 / ly + 1 / lz)

        def n_T_init(start):
            match start:
                case (T, n):
                    T0 = np.full_like(self.plasma.x_j, T.to_value(u.J))
                    n0 = np.full_like(self.plasma.x_j, n.to_value(u.m**-3))
                case (x, n, T, _, _):
                    T0 = np.interp(self.plasma.x_j, x.to_value(u.m), T.to_value(u.J))
                    n0 = np.interp(
                        self.plasma.x_j, x.to_value(u.m), n.to_value(u.m**-3)
                    )
            return n0, T0

        ne, Te = n_T_init(electrons.start)
        nn, Tn = n_T_init(neg_ions.start)

        alpha = np.zeros_like(nn)
        np.divide(nn, ne, where=ne != 0, out=alpha)
        gamma = np.ones_like(Te)
        np.divide(Te, Tn, where=Tn != 0, out=gamma)

        nus = (
            self.compute_nu(Te, n_T_init(ion.start)[1], ion, alpha, gamma)
            for ion in pos_ions
        )
        self.args_list = [(nu, np.max(nu)) for nu in nus]
        print("Electronegative Wall losses with profile initialized.")

    def h2d(self, electron_temperature, ion_temperature, mean_free_path, alpha, gamma):
        temp_ratio = np.divide(
            ion_temperature,
            electron_temperature,
            out=np.zeros_like(ion_temperature),
            where=electron_temperature != 0,
        )
        lz_mfp = self.lz / mean_free_path
        hy = (
            0.55
            * np.sqrt((gamma - 1) / (gamma * np.square(1 + alpha)) + gamma**-1)
            / np.sqrt(
                3 + 0.5 * lz_mfp + 0.2 * np.sqrt(1 + alpha) * temp_ratio * lz_mfp**2
            )
        )
        ly_mfp = self.ly / mean_free_path

        hz = (
            0.55
            * np.sqrt((gamma - 1) / (gamma * np.square(1 + alpha)) + gamma**-1)
            / np.sqrt(
                3 + 0.5 * ly_mfp + 0.2 * np.sqrt(1 + alpha) * temp_ratio * ly_mfp**2
            )
        )

        return self.corr * (self.ly * hy + self.lz * hz) / (self.ly + self.lz)

    def update(self):
        elec_temp = get_temperature(self.plasma.diag_values, self.electrons)
        ne = self.plasma.diag_values[(self.electrons.symbol, "n")]
        n_neg = self.plasma.diag_values[(self.neg_ions.symbol, "n")]
        neg_temp = get_temperature(self.plasma.diag_values, self.neg_ions)
        alpha = np.divide(n_neg, ne, where=ne != 0)
        np.clip(alpha, 0, 1000, out=alpha)
        gamma = np.divide(
            elec_temp,
            neg_temp,
            where=neg_temp != 0,
            out=np.ones_like(elec_temp),
        )
        np.nan_to_num(alpha, copy=False)
        gamma[elec_temp == 0] = 1
        gamma[alpha <= 1e-3] = 1
        np.nan_to_num(gamma, copy=False)

        nus = (
            self.compute_nu(
                elec_temp,
                get_temperature(self.plasma.diag_values, ion),
                ion,
                alpha,
                gamma,
            )
            for ion in self.ions
        )
        self.args_list = [(nu, np.max(nu)) for nu in nus]

    def compute_nu(
        self,
        elec_temp: np.ndarray,
        ion_temp: np.ndarray,
        ions: ParticlesGroup,
        alpha,
        gamma,
    ) -> np.ndarray:
        mfp = 1 / sum(
            mcc.inverse_mean_free_path(ion_temp, self.plasma.x_j) for mcc in ions.mccs
        )

        h2 = self.h2d(elec_temp, ion_temp, mfp, alpha, gamma)

        return np.sqrt(elec_temp) * self.c_s * h2 / np.sqrt(ions.m)

    def ions_to_remove(self, ions: ParticlesGroup, nu, nu_max) -> int:
        p = 1 - np.exp(-nu_max * self.plasma.dT)
        # number of candidate particles
        n_coll = p * ions.Npart
        part = n_coll - np.floor(n_coll)
        n_coll = int(n_coll) + (1 if np.random.rand() < part else 0)

        # choice of candidates
        Idxs = self.rng.choice(ions.Npart, size=n_coll, replace=False)
        r = self.rng.random(Idxs.shape[0]) * nu_max
        will_collide = r < np.interp(ions.x[Idxs], self.plasma.x_j, nu)
        return Idxs[will_collide]


def get_temperature(diags, part):
    try:
        if (part.symbol, "T") in diags:
            return diags[(part.symbol, "T")]
        n = diags[(part.symbol, "n")]
        if (part.symbol, "P") in diags:
            p = diags[(part.symbol, "P")]
            return np.divide((p[:, 0] + p[:, 2] + p[:, 5]), (3 * n), where=n != 0)
        else:
            ec = diags[(part.symbol, "E")]
            v = diags[(part.symbol, "V")]
            return (
                np.divide(ec, n, where=n != 0, out=np.zeros_like(ec)) * 2 / 3
                - np.sum(np.square(v), axis=1) * part.m / 3
            )
    except KeyError as e:
        raise KeyError(
            f"Missing diagnostic {e} for {part.symbol} to calculate wall losses"
        )
