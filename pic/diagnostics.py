#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:20:46 2021

@author: lequette
"""

from __future__ import annotations
from itertools import chain
import h5py
from os import path
import numpy as np
from pic.functions import (
    particle_to_grid,
    particle_to_grid_order0,
    histograms,
    histograms_v,
    histograms_weighted,
    histograms_v_weighted,
)
import astropy.units as u
from astropy.constants import eps0


from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from pic.plasma import Plasma
    from pic.MCC import MCC, InterspeciesMCC

eps0_v = eps0.si.value


class Diagnostics:
    def __init__(
        self,
        plasma: Plasma,
        particle_diags,
        diags,
        start,
        value_dict,
    ):
        self.start = start

        self.measures: list[Measure] = []
        for specie, list_of_diags in particle_diags.items():
            for quant in list_of_diags:
                match quant:
                    case "n":
                        self.measures.append(Measure_n(plasma, specie, value_dict))
                    case "V":
                        self.measures.append(Measure_V(plasma, specie, value_dict))
                    case "V2":
                        self.measures.append(Measure_V2(plasma, specie, value_dict))
                    case "P":
                        self.measures.append(Measure_P(plasma, specie, value_dict))
                    case "q":
                        self.measures.append(Measure_q(plasma, specie, value_dict))
                    case "E":
                        self.measures.append(Measure_E(plasma, specie, value_dict))
                    case "Power":
                        self.measures.append(
                            Measure_PowerDensity(plasma, specie, value_dict)
                        )
                    case "T":
                        self.measures.append(Measure_T(plasma, specie, value_dict))

                    case "J":
                        self.measures.append(Measure_J(plasma, specie, value_dict))
                    case ("EDF", number, resolution, energy_max, centered):
                        self.measures.append(
                            MeasureEnergyDistribution(
                                plasma,
                                specie,
                                number,
                                resolution,
                                energy_max.to_value(u.J),
                                centered,
                                value_dict,
                            )
                        )
                    case ("VDF", number, resolution, energy_max, centered):
                        self.measures.append(
                            MeasureVelocityDistribution(
                                plasma,
                                specie,
                                number,
                                resolution,
                                energy_max.to_value(u.J),
                                centered,
                                value_dict,
                            )
                        )
                    case ("EDFN", number, resolution, energy_max, centered):
                        self.measures.append(
                            MeasureEnergyDdistributionNormalized(
                                plasma,
                                specie,
                                number,
                                resolution,
                                energy_max.to_value(u.J),
                                centered,
                                value_dict,
                            )
                        )
                    case "Gamma":
                        self.measures.append(MeasureFlux(plasma, specie, value_dict))
                    case "Ed":
                        self.measures.append(Measure_Ed(plasma, specie, value_dict))
                    case "Losses_yz":
                        self.measures.append(
                            MeasureLossesYZ(plasma, specie, value_dict)
                        )
                    case "collisions_rates", coll_types:
                        self.measures.append(
                            CollisionRate(plasma, specie, coll_types, value_dict)
                        )
                    case "collisions_momentum_transfer", coll_types:
                        self.measures.append(
                            CollisionMomentumTransfer(
                                plasma, specie, coll_types, value_dict
                            )
                        )
                    case "collisions_energy_transfer", coll_types:
                        self.measures.append(
                            CollisionEnergyTransfer(
                                plasma, specie, coll_types, value_dict
                            )
                        )
                    case _:
                        print(
                            f"Warning: {quant} is not a valid diagnostic for {specie.symbol}"
                        )
        for quant in diags:
            if quant == "phi":
                self.measures.append(Measure_phi(plasma, value_dict))
            elif quant == "Pe":
                self.measures.append(Measure_Pe(plasma, value_dict))
            elif quant == "inj":
                self.measures.append(MeasureInj(plasma, value_dict))
            else:
                print(f"diag {quant} not implemented")

    def diags(self, nt):
        if nt >= self.start:
            for m in self.measures:
                m.record()

    def average_diags(self, average):
        for m in self.measures:
            m.average(average)

    def save_diags(self, nt, folder):
        with h5py.File(path.join(folder, f"{nt + 1}.h5"), "a") as f:
            #CHANGED HERE#
            pla = self.measures[0].plasma
            grp = f.create_group("time")
            grp.attrs["nt"] = nt
            grp.attrs["dt"] = pla.dT
            grp.attrs["instant_time"] = nt*pla.dT
            for m in self.measures:
                m.save(f)
                m.reset()


class Measure:
    def __init__(self, name, plasma: Plasma):
        self.name = name
        self.plasma = plasma

    def record(self):
        pass

    def average(self, N_average):
        pass

    def reset(self):
        self.values.fill(0)

    def save(self, f):
        f.create_dataset(self.name, data=self.values)


class ParticleMeasure(Measure):
    def __init__(self, name, plasma: Plasma, specie, value_dict):
        super().__init__(name, plasma)
        self.specie = specie
        self.particles = plasma.species[specie]
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict[(specie.symbol, name)] = self.values

    def save(self, f):
        f.create_dataset(f"{self.specie.symbol}/{self.name}", data=self.values)


class Measure_n(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("n", plasma, specie, value_dict)

    def record(self):
        self.values += self.particles.n

    def average(self, N_average):
        # values are sums of particle weights per cell collected over records
        self.values *= 1.0 / (self.plasma.dx * N_average)


class Measure_phi(Measure):
    def __init__(self, plasma, value_dict):
        super().__init__("phi", plasma)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict["phi"] = self.values

    def record(self):
        self.values += self.plasma.phi

    def average(self, N_average):
        self.values /= N_average


class Measure_Pe(Measure):
    def __init__(self, plasma, value_dict):
        super().__init__("Pe", plasma)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict["Pe"] = self.values
        self.cst = eps0_v * self.plasma.alpha / 2

    def record(self):
        self.values += np.sum(np.square(self.plasma.E), axis=1)

    def average(self, N_average):
        self.values *= self.cst / N_average


class Measure_V(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("V", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells, 3), dtype="float64", order="F")
        value_dict[(specie.symbol, self.name)] = self.values

        self.particles.need_u = True

    def record(self):
        self.values += self.particles.u

    def average(self, N_average):
        self.values /= N_average


class Measure_V2(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("V2", plasma, specie, value_dict)
        value_dict[(specie.symbol, self.name)] = self.values

        self.particles.need_u = True

    def record(self):
        self.values += np.sum(np.square(self.particles.u), axis=1)

    def average(self, N_average):
        self.values /= N_average


class Measure_J(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("J", plasma, specie, value_dict)
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        info = self.particles.V[: self.particles.Npart, 0] * self.particles.w[: self.particles.Npart]
        particle_to_grid(
            self.particles.Npart,
            self.particles.x[: self.particles.Npart],
            info,
            self.plasma.x_j,
            self.values,
            self.plasma.dx,
        )

    def average(self, N_average):
        # weights already include real particle number, multiply by charge and normalize by cell length/time
        self.values *= (self.particles.charge / (N_average * self.plasma.dx))


class Measure_P(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("P", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells, 6), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values
        self.particles.need_u = True
        self.particles.need_c = True

    def record(self):
        # loop over xx, yx, yy, zx, zy, zz
        for k, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
            info = (
                self.particles.c[: self.particles.Npart, i]
                * self.particles.c[: self.particles.Npart, j]
                * self.particles.w[: self.particles.Npart]
            )
            particle_to_grid(
                self.particles.Npart,
                self.particles.x[: self.particles.Npart],
                info,
                self.plasma.x_j,
                self.values[:, k],
                self.plasma.dx,
            )

    def average(self, N_average):
        # weights already include real particle number, multiply by mass and normalize by cell length
        self.values *= self.particles.m / (self.plasma.dx * N_average)


class Measure_q(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("q", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells, 3), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values
        self.particles.need_c = True

    def record(self):
        c_square = np.sum(np.square(self.particles.c[: self.particles.Npart]), axis=1)
        for k in range(3):
            info = (
                self.particles.c[: self.particles.Npart, k] * c_square * self.particles.w[: self.particles.Npart]
            )
            particle_to_grid(
                self.particles.Npart,
                self.particles.x[: self.particles.Npart],
                info,
                self.plasma.x_j,
                self.values[:, k],
                self.plasma.dx,
            )

    def average(self, N_average):
        self.values *= (
            self.particles.m / (2 * self.plasma.dx * N_average)
        )


class Measure_E(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("E", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        info = np.sum(np.square(self.particles.V[: self.particles.Npart]), axis=1) * self.particles.w[: self.particles.Npart]
        particle_to_grid(
            self.particles.Npart,
            self.particles.x[: self.particles.Npart],
            info,
            self.plasma.x_j,
            self.values,
            self.plasma.dx,
        )

    def average(self, N_average):
        self.values *= (
            self.particles.m / (2 * self.plasma.dx * N_average)
        )


class Measure_PowerDensity(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("Power", plasma, specie, value_dict)

    def record(self):
        info = (
            self.particles.V[: self.particles.Npart, 0]
            * self.particles.E_interp[: self.particles.Npart]
            * self.particles.w[: self.particles.Npart]
        )
        particle_to_grid(
            self.particles.Npart,
            self.particles.x[: self.particles.Npart],
            info,
            self.plasma.x_j,
            self.values,
            self.plasma.dx,
        )

    # self.values += J * self.plasma.E[:, 0]

    def average(self, N_average):
        self.values *= (
            self.particles.charge / (self.plasma.dx * N_average)
        )


class MeasureEnergyDistribution(ParticleMeasure):
    def __init__(
        self, plasma, specie, number, resolution, energy_max, centered, value_dict
    ):
        super().__init__("EDF", plasma, specie, value_dict)
        self.energy_max = energy_max
        self.bin_size = energy_max * 2 / (self.particles.m * resolution)
        self.probe_size = self.plasma.Lx / number
        self.resolution = resolution
        self.number = number
        self.centered = centered
        if centered:
            specie.need_c = True

        self.values = np.zeros((number, resolution), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        if self.centered:
            v_squared = np.sum(
                np.square(
                    self.particles.V[: self.particles.Npart]
                    - self.particles.c[: self.particles.Npart]
                ),
                axis=1,
            )
        else:
            v_squared = np.sum(
                np.square(self.particles.V[: self.particles.Npart]), axis=1
            )
        histograms_weighted(
            self.particles.x[: self.particles.Npart],
            v_squared,
            self.particles.w[: self.particles.Npart],
            self.probe_size,
            self.bin_size,
            self.values,
        )

    def average(self, N_average):
        # weights already reflect real particle counts
        self.values *= (
            self.resolution
            / (N_average * self.probe_size * self.energy_max)
        )

    # override
    def save(self, f):
        f.create_dataset(self.name, data=self.values) # regular save : average EDF

        #saving instantaneous EDF 
        self.values.fill(0)
        self.record()
        self.values *= (
            self.resolution
            / (self.probe_size * self.energy_max)
        )
        f.create_dataset(f"{self.name}_instant", data=self.values)





class MeasureVelocityDistribution(ParticleMeasure):
    def __init__(
        self, plasma, specie, number, resolution, energy_max, centered, value_dict
    ):
        super().__init__("VDF", plasma, specie, value_dict)
        self.energy_max = energy_max
        self.vmax = np.sqrt(energy_max * 2.0 / (self.particles.m))
        self.bin_size = self.vmax / resolution
        self.probe_size = self.plasma.Lx / number
        self.resolution = resolution
        self.number = number
        self.centered = centered
        if centered:
            specie.need_c = True

        self.values = np.zeros((number, 2 * resolution), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        if self.centered:
            v_x = (
                self.particles.V[: self.particles.Npart, 0]
                - self.particles.c[: self.particles.Npart, 0]
            )
        else:
            v_x = self.particles.V[: self.particles.Npart, 0]
        histograms_v_weighted(
            self.particles.x[: self.particles.Npart],
            v_x,
            self.particles.w[: self.particles.Npart],
            self.probe_size,
            self.bin_size,
            self.values,
        )

    def average(self, N_average):
        # weights already reflect real particle counts
        self.values *= (
            self.resolution / (N_average * self.probe_size * self.vmax)
        )


class MeasureFlux(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("Gamma", plasma, specie, value_dict)
        self.values = np.zeros((2), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        left, right = self.particles.flux
        self.values[0] += left
        self.values[1] += right

    def average(self, N_average):
        # self.values contains sum of removed weights (real particle numbers) per record
        self.values *= 1.0 / (self.plasma.dT * N_average)


class MeasureLossesYZ(ParticleMeasure):
    def __init__(self, plasma: Plasma, specie, value_dict):
        super().__init__("Losses_yz", plasma, specie, value_dict)
        if plasma.wall is None:
            raise ValueError("No wall defined")
        plasma.wall.add_diagnostic(self, self.particles)

    def accumulate(self, x: np.ndarray, w: np.ndarray):
        # accumulate weight into grid (w is per-particle weight)
        particle_to_grid(
            x.shape[0],
            x,
            w,
            self.plasma.x_j,
            self.values,
            self.plasma.dx,
        )

    def average(self, N_average):
        # values are sums of weights per cell per record
        self.values *= 1.0 / (self.plasma.dx * self.plasma.dT * N_average)


class Measure_Ed(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("Ed", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values

    def record(self):
        info = np.sum(np.square(self.particles.V[: self.particles.Npart]), axis=1) * self.particles.w[: self.particles.Npart]
        e = particle_to_grid(
            self.particles.Npart,
            self.particles.x[: self.particles.Npart],
            info,
            self.plasma.x_j,
            np.zeros_like(self.values),
            self.plasma.dx,
        )
        # e is sum(v^2 * w) per cell, self.particles.n is sum(w) per cell, so division yields weighted average v^2
        self.values += np.divide(e, self.particles.n, where=self.particles.n != 0.0)

    def average(self, N_average):
        self.values *= self.particles.m / (2 * N_average)


class Measure_T(ParticleMeasure):
    def __init__(self, plasma, specie, value_dict):
        super().__init__("T", plasma, specie, value_dict)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values
        self.particles.need_u = True

    def record(self):
        info = np.sum(np.square(self.particles.V[: self.particles.Npart]), axis=1) * self.particles.w[: self.particles.Npart]
        e = particle_to_grid(
            self.particles.Npart,
            self.particles.x[: self.particles.Npart],
            info,
            self.plasma.x_j,
            np.zeros_like(self.values),
            self.plasma.dx,
        )
        np.divide(e, self.particles.n, where=self.particles.n != 0.0, out=e)
        v2 = np.sum(np.square(self.particles.u), axis=1)
        diff = e - v2

        np.add(self.values, diff, where=diff >= 0.0, out=self.values)

    def average(self, N_average):
        self.values *= self.particles.m / (3 * N_average)


class MeasureEnergyDdistributionNormalized(ParticleMeasure):
    def __init__(
        self, plasma, specie, number, resolution, energy_max, centered, value_dict
    ):
        super().__init__("EDFN", plasma, specie, value_dict)
        self.energy_max = energy_max
        self.bin_size = energy_max * 2 / (self.particles.m * resolution)
        self.probe_size = self.plasma.Lx / number
        self.resolution = resolution
        self.number = number
        self.centered = centered
        if centered:
            specie.need_c = True

        self.values = np.zeros((number, resolution), dtype="float64")
        value_dict[(specie.symbol, self.name)] = self.values
        self.count = np.zeros(number, dtype=np.int64)

    def record(self):
        if self.centered:
            v_squared = np.sum(
                np.square(
                    self.particles.V[: self.particles.Npart]
                    - self.particles.c[: self.particles.Npart]
                ),
                axis=1,
            )
        else:
            v_squared = np.sum(
                np.square(self.particles.V[: self.particles.Npart]), axis=1
            )
        temp_hist = np.zeros_like(self.values)
        histograms_weighted(
            self.particles.x[: self.particles.Npart],
            v_squared,
            self.particles.w[: self.particles.Npart],
            self.probe_size,
            self.bin_size,
            temp_hist,
        )
        norm = np.trapz(temp_hist, axis=1)
        part_present = norm != 0.0
        np.divide(temp_hist, norm[:, None], where=part_present[:, None], out=temp_hist)
        self.values += temp_hist
        self.count += part_present

    def average(self, N_average):
        self.values *= self.resolution / (self.energy_max)
        np.divide(
            self.values,
            self.count[:, None],
            where=self.count[:, None] != 0.0,
            out=self.values,
        )
        self.count.fill(0)


class CollisionRate(ParticleMeasure):
    def __init__(
        self, plasma, specie, coll_types: List[str], value_dict, name="collisions_rates"
    ):
        super().__init__(name, plasma, specie, value_dict)
        self.values: Dict[str, Dict[str, np.ndarray]] = dict()
        value_dict[(specie.symbol, self.name)] = self.values

        for mcc in chain(self.particles.mccs, self.particles.imccs):
            target = mcc.target_symbol()
            d = {
                coll.name: np.zeros((self.plasma.N_cells), dtype=np.float64)
                for coll in mcc.collisions_types
                if any(coll.name.startswith(c) for c in coll_types)
            }
            if len(d) > 0:
                mcc.diags.append(self)
                self.values[target] = d

        self.current = None

    def before(self, I_coll: np.ndarray, mcc: MCC | InterspeciesMCC, collision):
        buffer = self.values[mcc.target_symbol()].get(collision.name)
        if buffer is not None:
            # accumulate removed weights per cell for collision diagnostics
            particle_to_grid(
                I_coll.shape[0],
                self.particles.x[I_coll],
                self.particles.w[I_coll],
                self.plasma.x_j,
                buffer,
                self.plasma.dx,
            )

    def after(self, I_coll):
        pass

    def record(self):
        pass

    def average(self, N_average):
        for d in self.values.values():
            for v in d.values():
                v *= 1.0 / (self.plasma.dx * self.plasma.dT * N_average)

    def save(self, f):
        for target, d in self.values.items():
            for name, v in d.items():
                f.create_dataset(
                    f"{self.specie.symbol}/{self.name}/{target}/{name}", data=v
                )

    def reset(self):
        for d in self.values.values():
            for v in d.values():
                v.fill(0)


class CollisionMomentumTransfer(CollisionRate):
    def __init__(self, plasma, specie, coll_types: List[str], value_dict):
        super().__init__(
            plasma, specie, coll_types, value_dict, name="collisions_momentum_transfer"
        )
        self.current: None | np.ndarray = None

    def before(self, I_coll: np.ndarray, mcc, collision):
        self.current = self.values[mcc.target_symbol()].get(collision.name)
        if self.current is not None:
            # store positions, velocities and weights for this collision batch
            self.before_values = (
                self.particles.x[I_coll],
                self.particles.V[I_coll, 0],
                self.particles.w[I_coll],
            )

    def after(self, I_coll):
        if self.current is None:
            return
        x, vi, w = self.before_values
        info = (vi - self.particles.V[I_coll, 0]) * w
        particle_to_grid(
            I_coll.shape[0],
            x,
            info,
            self.plasma.x_j,
            self.current,
            self.plasma.dx,
        )
        self.current = None

    def average(self, N_average):
        for d in self.values.values():
            for v in d.values():
                v *= (
                    self.particles.m
                    / (self.plasma.dx * self.plasma.dT * N_average)
                )


class CollisionEnergyTransfer(CollisionRate):
    def __init__(self, plasma, specie, coll_types: List[str], value_dict):
        super().__init__(
            plasma, specie, coll_types, value_dict, name="collisions_energy_transfer"
        )
        self.current: None | np.ndarray = None

    def before(self, I_coll: np.ndarray, mcc, collision):
        self.current = self.values[mcc.target_symbol()].get(collision.name)
        if self.current is not None:
            self.before_values = (
                self.particles.x[I_coll],
                np.sum(np.square(self.particles.V[I_coll]), axis=1),
                self.particles.w[I_coll],
            )

    def after(self, I_coll):
        if self.current is None:
            return
        x, vi2, w = self.before_values
        info = (vi2 - np.sum(np.square(self.particles.V[I_coll]), axis=1)) * w
        particle_to_grid(
            I_coll.shape[0],
            x,
            info,
            self.plasma.x_j,
            self.current,
            self.plasma.dx,
        )
        self.current = None

    def average(self, N_average):
        for d in self.values.values():
            for v in d.values():
                v *= (
                    self.particles.m
                    / (self.plasma.dx * self.plasma.dT * N_average * 2)
                )


class MeasureInj(Measure):
    def __init__(self, plasma: Plasma, value_dict):
        super().__init__("inj", plasma)
        self.values = np.zeros((plasma.N_cells), dtype="float64")
        value_dict["inj"] = self.values
        plasma.inj_diag = True

    def average(self, N_average):
        # injected weights are physical counts, normalize by volume/time and averaging window
        self.values *= 1.0 / (self.plasma.dx * self.plasma.dT * N_average)
