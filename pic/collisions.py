# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:11:43 2021

@author: Nicolas Lequette
"""

from __future__ import annotations
import numpy as np

from pic.functions import (
    isotropic_scatter,
    fixed_angle_isotropic_scatter,
    max_vect3D,
)
from astropy.constants import e
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from pic.MCC import MCC, HeavyMCC
    from pic.particles import ParticlesGroup
e = e.si.value
E0: float = 27.2114 * e


class CollisionType:
    def __init__(
        self,
        mcc: MCC | HeavyMCC,
        cross_section: Tuple[np.ndarray, np.ndarray],
        name: str,
    ):
        """
        Abstract class for collisions

        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.

        Returns
        -------
        None.

        """
        self.energy_array, self.cross_section_array = cross_section
        self.sigma_v_array = self.cross_section_array * np.sqrt(
            self.energy_array * 2 / mcc.mu
        )
        self.name = name
        self.mcc = mcc

    def sigma_v(self, v_squared):
        """
        Computes the collision frequency for a given relative speed

        Parameters
        ----------
        v : np.array(N,3)
            relative velocities.

        Returns
        -------
        np.array(N)
            Collision frequencies.

        """
        return np.interp(
            (v_squared * self.mcc.mu / 2),
            self.energy_array,
            self.sigma_v_array,
        )

    def nu(self, Idxs):
        """
        Computes the collision frequency of selected particles

        Parameters
        ----------
        I : np.array(N)
            Indices of the candidate particles.

        Returns
        -------
        np.array(N)
            Collision frequencies.

        """
        return self.sigma_v(np.sum(np.square(self.mcc.particles.V[Idxs]), axis=1))

    def scatter(self, Idxs, v_squared):
        """
        Apply this collision type to the selected particles. Must be overridden

        Parameters
        ----------
        I : np.array
            Indices of particle to be scattered.

        Returns
        -------
        None.

        """
        pass


class IonCollision(CollisionType):
    def nu(self, Idxs):
        """


        Parameters
        ----------
        I : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        # frequency computation
        return self.sigma_v(
            np.sum(
                np.square(self.mcc.particles.V[Idxs] - self.mcc.V_neutral[Idxs]), axis=1
            )
        )


class Elastic_e(CollisionType):
    def __init__(self, mcc, cross_section, name):
        """

        Elastic electron-neutral collision
        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.


        Returns
        -------
        None.

        """
        super().__init__(mcc, cross_section, name)
        self.energy_loss_factor = self.mcc.particles.m / self.mcc.neutral.m_n
        print("Elastic electron collisions initalized.")

    def scatter(self, Idxs, v_squared):
        """
        Apply this collision type to the selected particles.

        Parameters
        ----------
        I : np.array
            Indices of particle to be scattered.

        Returns
        -------
        None.

        """
        # normalisation of particle velocities
        norm_v = np.sqrt(v_squared)

        v = self.mcc.particles.V[Idxs] / norm_v[:, None]

        # scatered speed calculation
        v_scat, cos_khi = isotropic_scatter(v, self.mcc.rng)

        # new norm calculation
        # norm_v_scat = np.sqrt(v_squared * (1 - self.energy_loss_factor * (1 - cos_khi)))

        self.mcc.particles.V[Idxs] = (
            v_scat * ((1 - self.energy_loss_factor * (1 - cos_khi)) * norm_v)[:, None]
        )


class Elastic_anisotropic_e(CollisionType):
    def __init__(self, mcc, cross_section, name):
        """

        Okhrimovskyy Elastic electron-neutral collision
        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.


        Returns
        -------
        None.

        """
        super().__init__(mcc, cross_section, name)
        self.energy_loss_factor = 2 * self.mcc.particles.m / self.mcc.neutral.m_n
        print("Anisotropic elastic electron collisions initalized.")

    def scatter(self, Idxs, v_squared):
        """
        Apply this collision type to the selected particles.

        Parameters
        ----------
        I : np.array
            Indices of particle to be scattered.

        Returns
        -------
        None.

        """
        # normalisation of particle velocities
        v = self.mcc.particles.V[Idxs] / np.sqrt(v_squared[:, None])

        # scatered speed calculation
        r = self.mcc.rng.random(Idxs.shape[0])
        cos_khi = 1 - 2 * r / (1 + 4 * v_squared * self.mcc.mu * (1 - r) / E0)
        r1 = self.mcc.rng.random(Idxs.shape[0])
        v_scat = fixed_angle_isotropic_scatter(v, r1, cos_khi)

        # new norm calculation
        norm_v_scat = np.sqrt(v_squared * (1 - self.energy_loss_factor * (1 - cos_khi)))

        self.mcc.particles.V[Idxs] = v_scat * norm_v_scat[:, None]


class Excitation_e(CollisionType):
    def __init__(self, mcc, cross_section, name, de):
        """

        Inelastic electron-neutral collision with enegy level excitation

        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.
        de : float
            excitation energy.

        Returns
        -------
        None.

        """
        super().__init__(mcc, cross_section, name)
        self.de = de
        print(
            f"Inelastic electron collisions with excitation of level {de / e}eV initalized."
        )

    def scatter(self, Idxs, v_squared):
        # normalisation of particle velocities
        v = self.mcc.particles.V[Idxs] / np.sqrt(v_squared[:, None])
        # scatered velocity calculation
        v_scat, cos_khi = isotropic_scatter(v, self.mcc.rng)
        # new norm
        try:
            norm_v_scat = np.sqrt(v_squared - 2 * self.de / self.mcc.particles.m)
        except FloatingPointError:
            raise RuntimeError(
                f"Unsufficient electron energy for excitation {self.de / e}eV: {self.mcc.particles.m * v_squared / (2 * e)}eV"
            )

        self.mcc.particles.V[Idxs] = v_scat * norm_v_scat[:, None]


class Ionization_e(CollisionType):
    def __init__(self, mcc, cross_section, name, de, product_parts):
        """

        Inelastic electron-neutral collision with ionization and creation of new prticles

        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.
        de : float
            Ionization energy.

        Returns
        -------
        None.

        """
        super().__init__(mcc, cross_section, name)
        self.de = de
        self.product_parts = product_parts
        print(
            f"Ionizing electron collisions with excitation of level {de / e}eV initalized."
        )

    def scatter(self, Idxs, v_squared):
        # velocity normalization and electron final energy calculation
        energy = self.mcc.particles.m * v_squared / 2 - self.de
        norm_v_scat = np.sqrt(energy / self.mcc.particles.m)
        v = self.mcc.particles.V[Idxs] / np.sqrt(v_squared[:, None])

        # incident electron scattering
        v_inc, _ = isotropic_scatter(v, self.mcc.rng)
        v_inc *= norm_v_scat[:, None]

        # secondary electron initialization
        v_sec, _ = isotropic_scatter(v, self.mcc.rng)
        v_sec *= norm_v_scat[:, None]

        # incident electron
        self.mcc.particles.V[Idxs] = v_inc

        # secondary electron
        self.mcc.particles.add_particles(self.mcc.particles.x[Idxs], v_sec, self.mcc.particles.w[Idxs])

        # created ion
        x_ions = self.mcc.particles.x[Idxs]
        self.product_parts.add_particles(x_ions, self.mcc.neutral.generate_v(x_ions), self.mcc.particles.w[Idxs])


class DissociativeAttachment(CollisionType):
    def __init__(
        self, mcc, cross_section, name, de: float, product_parts: ParticlesGroup
    ):
        """

        Inelastic electron-neutral collision with ionization and creation of new prticles

        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.
        de : float
            Ionization energy.

        Returns
        -------
        None.

        """
        super().__init__(mcc, cross_section, name)
        self.product_parts = product_parts
        self.de = de
        print(
            f"Dissociative attachment collision with released energy of {de / e}eV initalized."
        )

    def scatter(self, Idxs, v_squared):
        # electron energy given to the ion
        energy = v_squared * self.mcc.particles.m / 2 + self.de

        x_ions = self.mcc.particles.x[Idxs]
        w_ions = self.mcc.particles.w[Idxs]

        v_ion = self.mcc.neutral.generate_v(x_ions)

        v_n = max_vect3D(Idxs.shape[0], 1, 1)
        v_n /= np.sqrt(np.sum(np.square(v_n), axis=1))[:, None]
        v_n *= np.sqrt(energy / self.product_parts.m)[:, None]

        v_ion += v_n

        # created ion
        self.product_parts.add_particles(x_ions, v_ion, self.mcc.particles.w[Idxs])
        self.mcc.particles.V[Idxs] = 0
        self.mcc.to_remove.extend(Idxs)


class DissociativeIonization(CollisionType):
    def __init__(self, mcc, cross_section, name, de, dissociation_de, product_parts):
        super().__init__(mcc, cross_section, name)
        self.product_parts = product_parts
        self.de = de
        self.dissociation_de = dissociation_de
        print(
            f"Dissociative Ionization collision with energy delta of {de / e}eV initalized."
        )

    def scatter(self, Idxs, v_squared):
        # velocity normalization and electron final energy calculation
        energy = self.mcc.particles.m * v_squared / 2 - self.de
        norm_v_scat = np.sqrt(energy / self.mcc.particles.m)
        # v = self.mcc.particles.V[Idxs] / np.sqrt(v_squared[:, None])

        # TODO: use better scattering

        # incident electron scattering
        v_inc = self.mcc.rng.random((3, Idxs.shape[0])).T
        v_inc *= (
            norm_v_scat[:, None] / np.sqrt(np.sum(np.square(v_inc), axis=1))[:, None]
        )

        # secondary electron initialization
        v_sec = self.mcc.rng.random((3, Idxs.shape[0])).T
        v_sec *= (
            norm_v_scat[:, None] / np.sqrt(np.sum(np.square(v_sec), axis=1))[:, None]
        )

        # incident electron
        self.mcc.particles.V[Idxs] = v_inc

        x_ions = self.mcc.particles.x[Idxs]
        w_ions = self.mcc.particles.w[Idxs]
        # secondary electron
        self.mcc.particles.add_particles(x_ions, v_sec, self.mcc.particles.w[Idxs])

        # created ion

        v_ion = self.mcc.neutral.generate_v(x_ions)

        v_n = max_vect3D(Idxs.shape[0], 1, 1)
        v_n *= np.sqrt(self.dissociation_de / (2 * np.sum(np.square(v_n), axis=1)))[
            :, None
        ]

        v_ion += v_n

        # created ion
        self.product_parts.add_particles(x_ions, v_ion, self.mcc.particles.w[Idxs])


class Isotropic_i(IonCollision):
    def __init__(self, mcc, cross_section, name):
        super().__init__(mcc, cross_section, name)
        print("Elastic ion collision initalized.")

    def scatter(self, Idxs, v_squared):
        # incident velocity in the relative frame
        V_i = self.mcc.particles.V[Idxs] - self.mcc.V_neutral[Idxs]
        norm_v = np.sqrt(v_squared)

        # normalization
        v = V_i / norm_v[:, None]

        # scattering
        cos_khi = np.sqrt(1 - self.mcc.rng.random(Idxs.shape[0]))
        v_scat = fixed_angle_isotropic_scatter(
            v, self.mcc.rng.random(Idxs.shape[0]), cos_khi
        )

        # new velocity
        new_norm = norm_v * np.abs(cos_khi)
        self.mcc.particles.V[Idxs] = (
            v_scat * new_norm[:, None] + self.mcc.V_neutral[Idxs]
        )


class Isotropic_i_mass_diff(IonCollision):
    def __init__(self, mcc, cross_section, name):
        super().__init__(mcc, cross_section, name)
        mi = self.mcc.particles.m
        mt = self.mcc.neutral.m_n
        self.enrgy_loss_factor = 2 * mi * mt / (mi + mt) ** 2
        self.m_ratio = mi / mt
        print("Elastic ion collision for different masses initalized.")

    def scatter(self, Idxs, v_squared):
        # incident velocity in the relative frame
        V_i = self.mcc.particles.V[Idxs] - self.mcc.V_neutral[Idxs]
        norm_v = np.sqrt(v_squared)

        # normalization
        v = V_i / norm_v[:, None]

        cos_theta = 1 - 2 * self.mcc.rng.random(Idxs.shape[0])

        al = self.enrgy_loss_factor * (1 - cos_theta)
        new_norm = norm_v * np.sqrt(1 - al)

        denom = cos_theta + self.m_ratio

        y = np.divide(np.sqrt(1 - cos_theta**2), denom, where=denom != 0)

        cos_khi = np.sign(denom) / np.sqrt(1 + y**2)

        # scattering
        v_scat = fixed_angle_isotropic_scatter(
            v, self.mcc.rng.random(Idxs.shape[0]), cos_khi
        )

        # new velocity
        self.mcc.particles.V[Idxs] = (
            v_scat * new_norm[:, None] + self.mcc.V_neutral[Idxs]
        )


class Backward_i(IonCollision):
    def __init__(self, mcc, cross_section, name):
        super().__init__(mcc, cross_section, name)
        print("Backward ion collision initalized.")

    def scatter(self, Idxs, v_squared):
        # velocity exchange
        self.mcc.particles.V[Idxs] = self.mcc.V_neutral[Idxs]


class ChargeExchange(IonCollision):
    def __init__(
        self, mcc, cross_section, name, product: ParticlesGroup, energy_loss: float
    ):
        super().__init__(mcc, cross_section, name)
        self.product = product
        self.energy_loss = energy_loss
        print("Charge exchange ion collision initalized.")

    def scatter(self, Idxs, v_squared):
        # code from oopd1

        mu = self.mcc.mu

        e_com = 0.5 * mu * v_squared
        new_e_com = e_com - self.energy_loss

        mi = self.mcc.particles.m
        mt = self.mcc.neutral.m_n
        v_rel = self.mcc.particles.V[Idxs] - self.mcc.V_neutral[Idxs]
        v_rel_norm = np.sqrt(v_squared)
        v_rel_unit = v_rel / v_rel_norm[:, None]
        v_com = (mi * self.mcc.particles.V[Idxs] + mt * self.mcc.V_neutral[Idxs]) / (
            mi + mt
        )

        momentum_change = np.sqrt(2 * mu * np.minimum(new_e_com, 0))

        v_new_i = v_com + momentum_change[:, None] * v_rel_unit / mt
        # v_new_t = v_com - momentum_change * v_rel_unit/mi

        self.product.add_particles(self.mcc.particles.x[Idxs], v_new_i, self.mcc.particles.w[Idxs])
        self.mcc.particles.V[Idxs] = 0
        self.mcc.to_remove.extend(Idxs)


class InterspeciesCollision:
    def __init__(
        self, mcc: MCC, cross_section: Tuple[np.ndarray, np.ndarray], name: str
    ):
        """
        Abstract class for collisions

        Parameters
        ----------
        mcc : MCC
            MCC containing this collision.
        cross_section : callable
            Interpolated cross section function with imput in eV and output in m^2.

        Returns
        -------
        None.

        """
        self.energy_array, self.cross_section_array = cross_section
        self.mcc = mcc
        self.name = name

    def cs_by_v(self, v):
        """
        Computes the collision frequency for a given relative speed

        Parameters
        ----------
        v : np.array(N,3)
            relative velocities.

        Returns
        -------
        np.array(N)
            Collision frequencies.

        """
        return (
            np.interp(
                v**2 * self.mcc.mu / 2,
                self.energy_array,
                self.cross_section_array,
            )
            * v
        )

    def scatter(self, _I, _J):
        """
        Apply this collision type to the selected particles. Must be overridden

        Parameters
        ----------
        I : np.array
            Indices of particle to be scattered.

        Returns
        -------
        None.

        """
        pass

    def nu(self, I_arr, J_arr):
        """


        Parameters
        ----------
        I : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        # frequency computation
        n_target = np.interp(
            self.mcc.impacter.x[I_arr], self.mcc.target.plasma.x_j, self.mcc.n
        )

        return n_target * self.cs_by_v(
            np.sqrt(
                np.sum(
                    np.square(self.mcc.impacter.V[I_arr] - self.mcc.target.V[J_arr]),
                    axis=1,
                )
            )
        )


class Recombination(InterspeciesCollision):
    def __init__(self, mcc, cross_section, name):
        super().__init__(mcc, cross_section, name)
        print("Recombination initalized.")

    def scatter(self, I_arr, J_arr):
        self.mcc.to_remove_impacter.extend(I_arr)
        self.mcc.to_remove_target.extend(J_arr)


class Detachment(InterspeciesCollision):
    def __init__(self, mcc, cross_section, name, de):
        super().__init__(mcc, cross_section, name)
        self.de = de
        print("Detachment initalized.")

    def scatter(self, I_arr, J_arr):
        self.mcc.impacter.V[I_arr] -= self.mcc.target.V[J_arr]
        v_squared = np.sum(np.square(self.mcc.impacter.V[I_arr]), 1)
        energy = self.mcc.impacter.m * v_squared / 2 - self.de
        norm_v_scat = np.sqrt(energy / self.mcc.impacter.m)
        # v = self.mcc.impacter.V[I_arr] / np.sqrt(v_squared[:, None])
        # TODO: use better scattering
        # incident electron scattering
        v_inc = self.mcc.rng.random((3, I_arr.shape[0])).T
        v_inc *= (
            norm_v_scat[:, None] / np.sqrt(np.sum(np.square(v_inc), axis=1))[:, None]
        )
        v_inc += self.mcc.target.V[J_arr]

        # secondary electron initialization
        v_sec = self.mcc.rng.random((3, I_arr.shape[0])).T
        v_sec *= (
            norm_v_scat[:, None] / np.sqrt(np.sum(np.square(v_sec), axis=1))[:, None]
        )
        v_sec += self.mcc.target.V[J_arr]

        # incident electron
        self.mcc.impacter.V[I_arr] = v_inc

        # secondary electron
        self.mcc.impacter.add_particles(self.mcc.impacter.x[I_arr], v_sec, self.mcc.impacter.w[I_arr])

        self.mcc.to_remove_target.extend(J_arr)
