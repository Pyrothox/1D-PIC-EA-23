# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:29:34 2021

@author: Nicolas Lequette
"""

from itertools import chain
import os.path
from typing import Any, Dict, Optional
import pickle
import numpy as np
import h5py
import matplotlib.pyplot as plt
import plasmapy.particles as p
from plasmapy.particles import ParticleLike
import astropy.units as u
from pathlib import Path

u.add_enabled_equivalencies(u.temperature_energy())


class Hdf5Viewer:
    def __init__(self, folder, only=None):
        self.units: Dict[str, u.Unit] = {
            "phi": u.V,
            "P": u.J / u.m**3,
            "n": u.m**-3,
            "v": u.m / u.s,
            "rho": u.C * u.m**-3,
            "V": u.m / u.s,
            "V2": u.m**2 / u.s**2,
            "J": u.A / u.m**2,
            "q": u.W / u.m**2,
            "T": u.eV,
            "T_P": u.eV,
            "E": u.J / u.m**3,
            "Ed": u.J,
            "Gamma": u.m**-2 * u.s**-1,
            "Gamma_yz": u.m**-2 * u.s**-1,
            "Power": u.W / u.m**3,
            "inj": u.m**-3 * u.s**-1,
            "Pe": u.Pa,
            "Losses_yz": u.m**-3 * u.s**-1,
        }
        self.lenghts = {"Gamma": 2, "Gamma_yz": 1}
        self.data_size = {"V": 3, "P": 6, "q": 3}
        self.parameters: Dict[str, Any] = pickle.load(
            open(os.path.join(folder, "parameters.pkl"), "rb")
        )
        self.folder = Path(folder)
        self.dx = (self.parameters["Lx"] / self.parameters["Nx"]).si
        self.x: u.Quantity = np.arange(self.parameters["Nx"] + 1) * self.dx
        N_files = int(np.ceil(self.parameters["Nt"] / self.parameters["n_average"])) + 1
        self.t: u.Quantity = (
            np.arange(N_files) * self.parameters["n_average"] * self.parameters["dT"]
        )
        self.particle_data: Dict[ParticleLike, Dict[str, u.Quantity]] = {
            specie: {
                name: u.Quantity(
                    np.zeros(
                        (
                            self.t.shape[0],
                            (
                                self.lenghts[name]
                                if name in self.lenghts
                                else self.x.shape[0]
                            ),
                            self.data_size[name] if name in self.data_size else 1,
                        ),
                        dtype=np.float64,
                    ).squeeze(),
                    self.units[name],
                )
                for name in diags_list
                if type(name) is str
            }
            for specie, diags_list in self.parameters["particle_diags"].items()
        }
        self.edf = {}
        self.edfn = {}
        self.vdf = {}
        self.collisions_rates = {}
        self.collisions_momentum_transfer = {}
        self.collisions_energy_transfer = {}

        def collisions_init(sp, c_names, unit):
            species_dict = dict()
            for reac_set in filter(
                lambda rs: rs.specie == sp,
                chain(self.parameters["reactions"], self.parameters["inter_species"]),
            ):
                coll_rates = {
                    name: np.zeros((self.t.shape[0], self.x.shape[0])) * unit
                    for name in (
                        reac.name()
                        for reac in reac_set.reactions
                        if any(
                            reac.name().startswith(coll_type) for coll_type in c_names
                        )
                    )
                }
                if coll_rates:
                    species_dict[reac_set.neutral] = coll_rates
            return species_dict

        for sp, ds in self.parameters["particle_diags"].items():
            for d in ds:
                match d:
                    case ("EDF", num_probes, resolution, max_e, _):
                        self.edf[sp] = (
                            (np.arange(num_probes) + 0.5) * self.x[-1] / num_probes,
                            (np.arange(resolution) + 0.5) * max_e / resolution,
                            np.zeros((self.t.shape[0], num_probes, resolution))
                            * u.m**-3
                            / u.J,
                        )
                    case ("VDF", num_probes, resolution, max_e, _):
                        v_max = np.sqrt(max_e * 2 / (sp.mass))
                        self.vdf[sp] = (
                            np.linspace(0.0 * u.m, self.x[-1], num_probes),
                            np.linspace(-v_max, v_max, 2 * resolution),
                            np.zeros((self.t.shape[0], num_probes, 2 * resolution))
                            * u.m**-3
                            / (u.m / u.s),
                        )
                    case ("EDFN", num_probes, resolution, max_e, _):
                        self.edfn[sp] = (
                            (np.arange(num_probes) + 0.5) * self.x[-1] / num_probes,
                            (np.arange(resolution) + 0.5) * max_e / resolution,
                            np.zeros((self.t.shape[0], num_probes, resolution))
                            * u.J**-(1),
                        )
                    case ("collisions_rates", c_names):
                        if species_dict := collisions_init(
                            sp, c_names, u.m**-3 * u.s**-1
                        ):
                            self.collisions_rates[sp] = species_dict
                    case ("collisions_momentum_transfer", c_names):
                        if species_dict := collisions_init(
                            sp, c_names, u.kg * u.m / u.s * u.m**-3 * u.s**-1
                        ):
                            self.collisions_momentum_transfer[sp] = species_dict
                    case ("collisions_energy_transfer", c_names):
                        if species_dict := collisions_init(
                            sp, c_names, u.J * u.m**-3 * u.s**-1
                        ):
                            self.collisions_energy_transfer[sp] = species_dict

        self.data: Dict[str, u.Quantity] = {
            name: u.Quantity(
                np.zeros(
                    (
                        self.t.shape[0],
                        self.x.shape[0],
                        self.data_size[name] if name in self.data_size else 1,
                    ),
                    dtype=np.float64,
                ).squeeze(),
                self.units[name],
            )
            for name in self.parameters["diagnostics"]
        }

        def fill_collisions_diag(diag_dict, f, i, diag_order, unit):
            for specie, neutrals_dict in diag_dict.items():
                for neutral, react_dir in neutrals_dict.items():
                    for name, quant in react_dir.items():
                        quant[i, ...] = u.Quantity(
                            f[f"{specie.symbol}/{diag_order}/{neutral.symbol}/{name}"],
                            unit,
                        )

        self.n_files = 0
        file_range = range(1, N_files + 1) if only is None else only
        for i in file_range:
            nt = self.parameters["start_step"] + i * self.parameters["n_average"]
            file = os.path.join(folder, f"{nt}.h5")

            if os.path.isfile(file):
                self.n_files += 1
                with h5py.File(file, "r") as f:
                    for specie, values in self.particle_data.items():
                        for name, quant in values.items():
                            quant[i, ...] = u.Quantity(
                                f[f"{specie.symbol}/{name}"], self.units[name]
                            )

                    for name, quant in self.data.items():
                        quant[i, ...] = u.Quantity(f[name], self.units[name])

                    for specie, value in self.edf.items():
                        edf_key = f"{specie.symbol}/EDF"
                        if edf_key in f:
                            value[2][i, ...] = f[edf_key] * u.m**-3 / u.J
                    for specie, value in self.edfn.items():
                        edfn_key = f"{specie.symbol}/EDFN"
                        if edfn_key in f:
                            value[2][i, ...] = f[edfn_key] * u.J**-1
                    for specie, value in self.vdf.items():
                        vdf_key = f"{specie.symbol}/VDF"
                        if vdf_key in f:
                            value[2][i, ...] = (
                                f[vdf_key] * u.m**-3 / (u.m / u.s)
                            )

                    fill_collisions_diag(
                        self.collisions_rates,
                        f,
                        i,
                        "collisions_rates",
                        u.m**-3 * u.s**-1,
                    )
                    fill_collisions_diag(
                        self.collisions_momentum_transfer,
                        f,
                        i,
                        "collisions_momentum_transfer",
                        u.kg * u.m / u.s * u.m**-3 * u.s**-1,
                    )
                    fill_collisions_diag(
                        self.collisions_energy_transfer,
                        f,
                        i,
                        "collisions_energy_transfer",
                        u.J * u.m**-3 * u.s**-1,
                    )

        for specie, values in self.particle_data.items():
            if "P" in values:
                values["P"] = symetrize(values["P"])

        self.completion = self.n_files / (N_files - 1)

    def get_index(self, t: u.Quantity | str) -> int:
        return np.abs(self.t - u.Quantity(t, u.s)).argmin()

    def plot(
        self,
        t: u.Quantity | str,
        quantity: str,
        particle: Optional[ParticleLike] = None,
    ):
        i = self.get_index(t)
        if particle is None:
            data = self.data[quantity]
        else:
            part = p.Particle(particle)
            data = self.particle_data[part][quantity]

        plt.figure()
        plt.plot(self.x, data[i])

    def spacetime(
        self,
        quantity: str,
        particle: Optional[ParticleLike] = None,
        unit: Optional[u.Unit] = None,
    ):
        if particle is None:
            data = self.data[quantity]
        else:
            data = self.particle_data[particle][quantity]

        if unit is not None:
            data = data.to(unit)

        plt.figure()
        col = plt.pcolormesh(self.t.value, self.x.value, data.T.value, shading="auto")
        plt.ylabel(f"x({self.x.unit})")
        plt.xlabel(f"t({self.t.unit})")
        cbar = plt.colorbar(col)

        cbar.set_label(data.unit)
        plt.show()

    def densities(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        plt.figure()
        for k, v in self.particle_data.items():
            if "n" in v:
                val = np.mean(v["n"][i_g:i_d], axis=0)
                plt.plot(self.x, val, label=k.symbol)
        plt.ylim(0)
        plt.xlim(0, self.x[-1].value)

        plt.legend()
        plt.show()

    def Temperature(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        plt.figure()
        for k, v in self.particle_data.items():
            if "T" in v:
                val = np.mean(v["T"][i_g:i_d], axis=0)
                plt.plot(self.x, val, label=k.symbol)
        plt.legend()
        plt.show()

    def densities_print(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        for k, v in self.particle_data.items():
            if "n" in v:
                val = np.mean(v["n"][i_g:i_d], axis=0)
                print(self.x)
                print(val)

    def Temperature_print(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        print(self.x)
        for k, v in self.particle_data.items():
            if "T" in v:
                val = np.mean(v["T"][i_g:i_d], axis=0)
                print(val)

    def PowerDensity(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        plt.figure()
        for k, v in self.particle_data.items():
            if "Power" in v:
                val = np.mean(v["Power"][i_g:i_d], axis=0)
                plt.plot(self.x, val, label=k.symbol)
        plt.legend()
        plt.show()

    def PowerDensity_print(self, t: u.Quantity | str, average=1):
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        print(self.x)
        for k, v in self.particle_data.items():
            if "Power" in v:
                val = np.mean(v["Power"][i_g:i_d], axis=0)
                print(val)

    def edf_3d(self, t: u.Quantity | str, particle: ParticleLike):
        i = self.get_index(t)
        x, e, edf = self.edf[particle]
        plt.figure()
        ax = plt.axes(projection="3d")
        for k in range(len(x)):
            ax.plot(
                e.to_value(u.eV),
                np.full_like(e.value, x[k].to_value(u.m)),
                edf[i, k, :].to_value(u.m**-3 / u.eV),
            )
        # ax.plot_surface(x.value, e.value, edf[i,:,:].value, cmap="viridis")
        ax.set_xlabel("e(eV)")
        ax.set_ylabel("x(m)")
        ax.set_zlabel("edf(m⁻³/eV)")
        plt.show()

    def edf_print(self, t: u.Quantity | str, particle: ParticleLike):
        i = self.get_index(t)
        x, e, edf = self.edf[particle]
        print(e.to_value(u.eV))
        for k in range(len(x)):
            print(edf[i, k, :].to_value(u.m**-3 / u.eV))

    def vdf_3d(self, t: u.Quantity | str, particle: ParticleLike):
        i = self.get_index(t)
        x, v, vdf = self.vdf[particle]
        plt.figure()
        ax = plt.axes(projection="3d")
        for k in range(len(x)):
            ax.plot(
                v.to_value(u.m / u.s),
                np.full_like(v.value, x[k].to_value(u.m)),
                vdf[i, k, :].to_value(u.m**-3 / (u.m / u.s)),
            )
            ax.set_xlabel("e(m/s)")
            ax.set_ylabel("x(m)")
            ax.set_zlabel("vdf(m⁻³/(m/s))")
            plt.show()

    def save_profiles(self, t: u.Quantity | str, name, average=1):
        try:
            os.mkdir(
                name,
            )
        except FileExistsError:
            pass
        i = self.get_index(t)
        i_g = i - int(average / 2)
        i_d = i_g + average
        for k, v in self.particle_data.items():
            if "n" in v and "T" in v:
                val = [
                    self.x.to_value(u.m),
                    np.mean(v["n"][i_g:i_d].to_value(u.m**-3), axis=0),
                    np.mean(v["T"][i_g:i_d].to_value(u.eV), axis=0),
                ]
                if "V" in v:
                    val.append(
                        np.mean(v["V"][i_g:i_d, :, 0].to_value(u.m / u.s), axis=0)
                    )
                np.savetxt(os.path.join(name, k.symbol), val)

    def new_parameters(
        self, t_sim: u.Quantity, alpha=1, i=-1, part_per_cell: Optional[int] = None
    ) -> Dict[str, Any]:
        from plasmapy.formulary import Debye_length, plasma_frequency
        from copy import deepcopy

        i = i if i >= 0 else self.t.shape[0] + i
        if i > self.n_files:
            raise IndexError(
                f"Simulation has not reached index {i}. Current index is {self.n_files}."
            )
        ne = self.particle_data[p.electron]["n"][i]
        Te = self.particle_data[p.electron]["T"][i]
        dbl = Debye_length(
            Te.to(u.K, equivalencies=u.temperature_energy()), ne
        ) * np.sqrt(alpha)

        neg_ion = [d for k, d in self.particle_data.items() if k.charge < 0]
        if neg_ion:
            ni = neg_ion[0]["n"][i]
            Ti = neg_ion[0]["T"][i]
            alpha_neg = ni / ne
            gamma = np.divide(Te, Ti, where=Ti != 0, out=np.zeros_like(Te))
            dbl /= np.sqrt(1 + alpha_neg * gamma)

        dx = dbl / 2
        omega = 2 * np.pi * plasma_frequency(ne, particle="e-", to_hz=True)
        dt = 0.2 / omega * np.sqrt(alpha)
        dxmin = np.min(dx)
        dtmin = np.min(dt)
        nx = int(1 + (self.parameters["Lx"] / dxmin).si)
        if part_per_cell is None:
            part_per_cell = self.parameters["Npart"] // self.parameters["Nx"]
        Nt = int(1 + (t_sim / dtmin).si)
        new_params = deepcopy(self.parameters)
        new_params["dX"] = self.parameters["Lx"] / nx
        new_params["dT"] = dtmin.to(u.s)

        new_params["Nx"] = nx
        new_params["Nt"] = Nt
        new_params["Npart"] = nx * part_per_cell
        new_params["Tt"] = t_sim
        new_params["density"] = np.max(ne)
        new_params["alpha"] = alpha
        new_sp_dict = {
            sp: (
                self.x,
                self.particle_data[sp]["n"][i],
                self.particle_data[sp]["T"][i],
                (
                    self.particle_data[sp]["V"][i, :, 0]
                    if "V" in self.particle_data[sp]
                    else None
                ),
                None,
            )
            for sp in self.parameters["species"].keys()
        }
        new_params["species"] = new_sp_dict
        if "inductive_heating" in self.parameters:
            (*old_ind, _) = self.parameters["inductive_heating"]
            if (rst := (self.folder / "restart.h5")).exists():
                with h5py.File(rst, "r") as f:
                    E0 = f["induction"]["E0"][()] * u.V / u.m
                new_params["inductive_heating"] = (*old_ind, E0)

        return new_params

    def is_complete(self):
        return self.n_files == self.t.shape[0] - 1


def symetrize(P_old):
    P_new = u.Quantity(np.empty((P_old.shape[0], P_old.shape[1], 3, 3)), u.J / u.m**3)
    for k, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        P_new[:, :, i, j] = P_old[:, :, k]
        if i != j:
            P_new[:, :, j, i] = P_old[:, :, k]
    return P_new
