# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 18:24:03 2021

@author: Nicolas Lequette
"""

from __future__ import annotations

from typing import Union, Dict, TYPE_CHECKING
from typing import List as PyList

from scipy import optimize
import numpy as np
from scipy.constants import e
from plasmapy.particles import electron, ParticleLike
import astropy.units as u
from pic.diagnostics import CollisionRate

from pic.functions import (
    fill_max_vect,
    fill_max_vect_variable_T,
    max_vect3D,
    max_vect3D_multiple_T,
    particle_to_grid_order0,
)
import pic.collisions as coll
from numba.typed import List

if TYPE_CHECKING:
    from pic.particles import ParticlesGroup
    from pic.reactions import ReactionsSet


class ConstantGas:
    def __init__(self, T_n, n_n, m_n, gas: ParticleLike):
        self.T_n = T_n
        self.n_n = n_n
        self.m_n = m_n
        self.max_n = n_n
        self.gas = gas

    def get_T(self, Idxs, x):
        return self.T_n

    def get_n(self, Idxs, x):
        return self.n_n

    def fill_v(self, x, Idxs, v):
        fill_max_vect(v, Idxs, Idxs.shape[0], self.T_n, self.m_n)

    def generate_v(self, x):
        return max_vect3D(x.shape[0], self.T_n, self.m_n)

    def get_T_grid(self, x_j):
        return self.T_n

    def get_n_grid(self, x_j):
        return self.n_n


class ProfileGas:
    def __init__(self, x, n_n, T_n, m_n, gas: ParticleLike):
        if np.min(T_n) < 0:
            raise ValueError(f"Negative temperature for {gas.symbol}")
        self.x = x
        self.T = T_n
        self.n = n_n
        self.m_n = m_n
        self.max_n = np.max(self.n)
        self.gas = gas

    def get_T(self, Idxs, x):
        return np.interp(x[Idxs], self.x, self.T)

    def get_n(self, Idxs, x):
        return np.interp(x[Idxs], self.x, self.n)

    def fill_v(self, x, Idxs, v):
        fill_max_vect_variable_T(v, Idxs, Idxs.shape[0], self.get_T(Idxs, x), self.m_n)

    def generate_v(self, x):
        return max_vect3D_multiple_T(x.shape[0], np.interp(x, self.x, self.T), self.m_n)

    def get_T_grid(self, x_j):
        return np.interp(x_j, self.x, self.T)

    def get_n_grid(self, x_j):
        return np.interp(x_j, self.x, self.n)


class ProfileDriftGas(ProfileGas):
    def __init__(self, x, n_n, u_n, T_n, m_n, gas: ParticleLike):
        super().__init__(x, n_n, T_n, m_n, gas)
        self.u_n = u_n

    def fill_v(self, x, Idxs, v):
        super().fill_v(x, Idxs, v)
        v[Idxs, 0] += np.interp(x[Idxs], self.x, self.u_n)

    def generate_v(self, x):
        v = super().generate_v(x)
        v[:, 0] += np.interp(x, self.x, self.u_n)
        return v


class MCC:
    def __init__(
        self, particles: ParticlesGroup, neutral: Union[ConstantGas, ProfileGas]
    ) -> None:
        """
        Constructor of the Monte-Carlo Collisions class.

        Parameters
        ----------
        particles : particles
            Particles attached to the MCC.
        plasma : plasma
            Plasma containing the particles.

        Returns
        -------
        None.

        """
        self.particles = particles
        self.neutral: Union[ConstantGas, ProfileGas] = neutral
        self.mu = 1 / (1 / particles.m + 1 / neutral.m_n)
        self.collisions_types: PyList[coll.CollisionType] = []
        self.to_remove = List()
        self.rng = np.random.default_rng()
        self.diags: PyList[CollisionRate] = []
        self.max_nu: float = 0
        self.max_sigma_v_sum: float = 0

    def set_neutral_profile(self, neutral: Union[ConstantGas, ProfileGas]):
        self.neutral = neutral
        self.update_max_nu()

    def compute_max_nu(self) -> None:
        """
        Computes yhe maximum collision frequency for null collisions.

        Returns
        -------
        None.

        """
        # cumulated collision frequency
        self.sigma_v_sum = lambda v: -np.sum(
            [collision.sigma_v(v**2) for collision in self.collisions_types]
        )

        # computation of minimum over [0 ; 1000 eV]
        _, self.max_sigma_v_sum, _, _ = optimize.fminbound(
            self.sigma_v_sum, 0, np.sqrt(2000 * e / self.mu), full_output=1
        )
        self.max_sigma_v_sum *= -1
        self.max_nu = self.max_sigma_v_sum * self.neutral.max_n

    def update_max_nu(self):
        self.max_nu = self.max_sigma_v_sum * self.neutral.max_n

    def particle_list(self, dT: float) -> np.ndarray:
        """


        Parameters
        ----------
        dT : float
            Timestep over which collisions are simulated.

        Returns
        -------
        I : np.array od ints
            Indices of particles candidate for collisions.

        """
        # Null collision probability
        p = 1 - np.exp(-self.max_nu * dT)
        # number of candidate particles
        n_coll = p * self.particles.Npart
        part = n_coll - np.floor(n_coll)
        n_coll = int(n_coll) + (1 if self.rng.random() < part else 0)

        # choice of candidates
        Idxs = self.rng.choice(self.particles.Npart, size=n_coll, replace=False)

        return Idxs

    def v_square(self, Idxs):
        return np.sum(np.square(self.particles.V[Idxs]), axis=1)

    def apply_collisions(self, Idxs):
        n = self.neutral.get_n(Idxs, self.particles.x)
        r = self.rng.random(Idxs.shape[0]) * self.max_nu
        r /= n
        mask = np.zeros(Idxs.shape[0], dtype=bool)
        not_collided = np.ones_like(mask)
        v_squared = self.v_square(Idxs)
        for collision in self.collisions_types:
            # choice of collision type
            nu = collision.sigma_v(v_squared[not_collided])

            mask.fill(False)
            mask[not_collided] = r[not_collided] < nu
            r[not_collided] -= nu
            not_collided ^= mask
            # collision computation
            if mask.any():
                Idxs_coll = Idxs[mask]
                for di in self.diags:
                    di.before(Idxs_coll, self, collision)

                collision.scatter(Idxs_coll, v_squared[mask])
                for di in self.diags:
                    di.after(Idxs_coll)
        if self.to_remove:
            self.particles.remove_index(self.to_remove)
            self.to_remove.clear()

    def scatter(self, dT):
        """
        Applies the collisions

        Parameters
        ----------
        dT : float
            Timestep over which collisions are simulated.

        Returns
        -------
        None.

        """

        Idxs = self.particle_list(dT)
        self.apply_collisions(Idxs)

    def inverse_mean_free_path(self, temperature_impacter, x_j):
        temperature_neutral = self.neutral.get_T_grid(x_j) * e
        reduced_temperature = (
            temperature_impacter * self.neutral.m_n
            + temperature_neutral * self.particles.m
        ) / (self.neutral.m_n + self.particles.m)

        cross_sections_integrated = (
            np.trapz(
                2
                * np.sqrt(collison.energy_array / np.pi)[:, None]
                * np.power(reduced_temperature, -3 / 2)[None, :]
                * np.exp(-collison.energy_array[:, None] / reduced_temperature[None, :])
                * collison.cross_section_array[:, None],
                collison.energy_array,
                axis=0,
            )
            for collison in self.collisions_types
        )
        neutral_density = self.neutral.get_n_grid(x_j)
        return neutral_density * sum(cross_sections_integrated)

    def target_symbol(self):
        return self.neutral.gas.symbol


class HeavyMCC(MCC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.V_neutral = np.zeros((self.particles.Npart, 3), dtype=float, order="F")

    def set_neutral(self, Idxs):
        if self.particles.V.shape[0] > self.V_neutral.shape[0]:
            self.V_neutral = np.zeros_like(self.particles.V)
        self.neutral.fill_v(self.particles.x, Idxs, self.V_neutral)

    def v_square(self, Idxs):
        return np.sum(np.square(self.particles.V[Idxs] - self.V_neutral[Idxs]), axis=1)

    def scatter(self, dT):
        Idxs = self.particle_list(dT)
        self.set_neutral(Idxs)
        self.apply_collisions(Idxs)


class InterspeciesMCC:
    def __init__(self, impacter: ParticlesGroup, target: ParticlesGroup):
        self.impacter = impacter
        self.target = target
        self.collisions_types = []
        self.to_remove_impacter = List()
        self.to_remove_target = List()
        self.mu = 1 / (1 / self.impacter.m + 1 / self.target.m)
        self.rng = np.random.default_rng()
        self.n = np.zeros_like(self.target.n)
        self.diags: PyList[CollisionRate] = []

    def compute_max_cs(self):
        """
        Computes yhe maximum collision frequency for null collisions.

        Returns
        -------
        None.

        """
        # cumulated collision frequency
        self.cs_sum = lambda v: -np.sum(
            [collision.cs_by_v(v) for collision in self.collisions_types]
        )

        # computation of minimum over [0 ; 1000 eV]
        _, self.max_cs, _, _ = optimize.fminbound(
            self.cs_sum, 0, np.sqrt(2000 * e / self.mu), full_output=1
        )
        self.max_cs *= -1

    def particle_list(self, dT):
        """


        Parameters
        ----------
        dT : float
            Timestep over which collisions are simulated.

        Returns
        -------
        I : np.array od ints
            Indices of particles candidate for collisions.

        """
        self.n.fill(0)
        particle_to_grid_order0(
            self.target.Npart,
            self.target.x,
            self.target.plasma.x_j,
            self.n,
            self.target.plasma.dx,
        )
        self.n *= self.target.plasma.qf / self.target.plasma.dx
        self.max_n = np.max(self.n)
        # Null collision probability
        p = 1 - np.exp(-self.max_n * self.max_cs * dT)
        # number of candidate particles
        n_coll = p * self.impacter.Npart
        part = n_coll - np.floor(n_coll)
        n_coll = int(n_coll) + (1 if self.rng.random() < part else 0)
        if n_coll == 0:
            return np.zeros(0), np.zeros(0)

        # choice of candidates
        self.target.coarse_sort()
        I_arr = self.rng.choice(
            np.arange(self.impacter.Npart), size=n_coll, replace=False
        )
        J_arr = self.target.partners(self.impacter.x[I_arr])
        return I_arr, J_arr

    def apply_collisions(self, I_arr, J_arr):
        r = self.rng.random(I_arr.shape[0]) * self.max_cs * self.max_n
        nu_c = np.zeros(I_arr.shape[0], dtype=float)
        mask = np.zeros(I_arr.shape[0], dtype=bool)
        for collision in self.collisions_types:
            # choice of collision type
            nu = collision.nu(I_arr, J_arr)
            mask = r >= nu_c
            nu_c += nu
            mask *= r < nu_c
            # collision computation
            if mask.any():
                I_coll = I_arr[mask]
                for di in self.diags:
                    di.before(I_coll, self, collision)

                collision.scatter(I_arr[mask], J_arr[mask])
                for di in self.diags:
                    di.after(I_coll)

        if self.to_remove_impacter:
            self.impacter.remove_index(self.to_remove_impacter)
            self.to_remove_impacter.clear()

        if self.to_remove_target:
            self.target.remove_index(self.to_remove_target)
            self.to_remove_target.clear()

    def scatter(self, dT):
        """
        Applies the collisions

        Parameters
        ----------
        dT : float
            Timestep over which collisions are simulated.

        Returns
        -------
        None.

        """
        I_arr, J_arr = self.particle_list(dT)
        if I_arr.shape[0] > 0:
            self.apply_collisions(I_arr, J_arr)

    def target_symbol(self):
        return self.target.symbol


def mcc_factory(
    species: dict[ParticleLike, ParticlesGroup],
    neutrals,  #: dict[
    #     ParticleLike,
    #     (tuple[u.m**-3, u.eV]| np.ndarray),
    # ],
    reactions: list[ReactionsSet],
    isotripic: bool = True,
):
    for reaction_set in reactions:
        print(f"{reaction_set.specie.symbol} + {reaction_set.neutral.symbol}")
        match neutrals[reaction_set.neutral]:
            case (T_n, n_n):
                neutral_gas = ConstantGas(
                    T_n.to_value(u.eV),
                    n_n.to_value(u.m**-3),
                    reaction_set.neutral.mass.to_value(u.kg),
                    reaction_set.neutral,
                )
            case (x, n_n, T_n, None, _):
                neutral_gas = ProfileGas(
                    x.to_value(u.m),
                    n_n.to_value(u.m**-3),
                    T_n.to_value(u.eV),
                    reaction_set.neutral.mass.to_value(u.kg),
                    reaction_set.neutral,
                )
            case (x, n_n, T_n, u_n, _):
                neutral_gas = ProfileDriftGas(
                    x.to_value(u.m),
                    n_n.to_value(u.m**-3),
                    u_n.to_value(u.m / u.s),
                    T_n.to_value(u.eV),
                    reaction_set.neutral.mass.to_value(u.kg),
                    reaction_set.neutral,
                )
            case other:
                raise ValueError(f"Unknown neutral gas format: {other}")
        mcc_args = [species[reaction_set.specie], neutral_gas]
        if reaction_set.specie == electron:
            mcc = MCC(*mcc_args)
        else:
            mcc = HeavyMCC(*mcc_args)
        for reaction in reaction_set.reactions:
            coll_args = [mcc, reaction.cross_section_interpolator(), reaction.name()]
            if reaction.type in ("ELASTIC", "EFFECTIVE"):
                if isotripic:
                    mcc.collisions_types.append(coll.Elastic_e(*coll_args))
                else:
                    mcc.collisions_types.append(coll.Elastic_anisotropic_e(*coll_args))

            elif reaction.type in ("EXCITATION", "DISSOCIATION"):
                mcc.collisions_types.append(
                    coll.Excitation_e(
                        *coll_args,
                        reaction.threshold.to_value(u.J),
                    )
                )

            elif reaction.type == "IONIZATION":
                mcc.collisions_types.append(
                    coll.Ionization_e(
                        *coll_args,
                        reaction.threshold.to_value(u.J),
                        species[reaction.product],
                    )
                )
            elif reaction.type == "DISSOCIATIVE_IONIZATION":
                mcc.collisions_types.append(
                    coll.DissociativeIonization(
                        *coll_args,
                        reaction.threshold.to_value(u.J),
                        float(reaction.infos["DISSOCIATION_ENERGY"]),
                        species[reaction.product],
                    )
                )

            elif reaction.type == "DISSOCIATIVE_ATTACHMENT":
                mcc.collisions_types.append(
                    coll.DissociativeAttachment(
                        *coll_args,
                        reaction.threshold.to_value(u.J),
                        species[reaction.product],
                    )
                )
            elif reaction.type == "ISOTROPIC":
                if (
                    np.abs(reaction_set.specie.mass - reaction_set.neutral.mass)
                    / reaction_set.neutral.mass
                ) < 1e-3:
                    mcc.collisions_types.append(coll.Isotropic_i(*coll_args))
                else:
                    mcc.collisions_types.append(coll.Isotropic_i_mass_diff(*coll_args))

            elif reaction.type == "BACKSCAT":
                mcc.collisions_types.append(coll.Backward_i(*coll_args))

            elif reaction.type == "CHARGE_EXCHANGE":
                mcc.collisions_types.append(
                    coll.ChargeExchange(
                        *coll_args,
                        species[reaction.product],
                        reaction.energy_loss.to_value(u.J),
                    )
                )

        mcc.compute_max_nu()
        species[reaction_set.specie].add_mcc(mcc)


def inter_species_mcc_factory(
    species: Dict[ParticleLike, ParticlesGroup],
    reactions: list[ReactionsSet],
):
    for reaction_set in reactions:
        mcc = InterspeciesMCC(
            species[reaction_set.specie], species[reaction_set.neutral]
        )
        for reaction in reaction_set.reactions:
            if reaction.type == "RECOMBINATION":
                mcc.collisions_types.append(
                    coll.Recombination(
                        mcc, reaction.cross_section_interpolator(), reaction.name()
                    )
                )
            elif reaction.type == "DETACHMENT":
                mcc.collisions_types.append(
                    coll.Detachment(
                        mcc,
                        reaction.cross_section_interpolator(),
                        reaction.name(),
                        reaction.threshold.to_value(u.J),
                    )
                )

        mcc.compute_max_cs()
        species[reaction_set.specie].add_imcc(mcc)
