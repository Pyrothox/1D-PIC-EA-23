# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 13:20:13 2021

@author: Nicolas Lequette
"""

import configparser
import itertools
from pathlib import Path
import numpy as np
import astropy.units as u
import plasmapy.particles as particles

import pic.reactions as reac

import os.path
from itertools import product


def get_parameters(file, cs_folder=Path("cross_sections")):
    config = configparser.RawConfigParser()
    file = Path(file)
    file_dir = file.parent
    config.read(file)

    parameters = {}
    sp = config["Simulation parameters"]
    parameters["Lx"] = u.Quantity(sp["Lx"], unit=u.m)  # System length

    parameters["Nx"] = sp.getint("Nx")  # cell number
    if "density" in sp:
        parameters["density"] = u.Quantity(sp["density"], unit=u.m**-3)
    parameters["Npart"] = (
        sp.getint("Npart") * parameters["Nx"]
    )  # particles number, calculated via particle par cell
    parameters["n_average"] = sp.getint("n_average")
    parameters["start_step"] = sp.getint("start_step")
    parameters["dT"] = u.Quantity(sp["dT"], unit=u.s)  # time step

    parameters["Nt"] = sp.getint("Nt")
    parameters["Tt"] = parameters["Nt"] * parameters["dT"]

    parameters["diagnostics"] = []
    if "diagnostics" in sp:
        parameters["diagnostics"] = [
            diag.strip()
            for diag in config["Simulation parameters"]["diagnostics"].split(",")
        ]

    parameters["left_boundary"] = config["left boundary"]
    parameters["right_boundary"] = config["right boundary"]

    parameters["species"] = {}
    parameters["neutrals"] = {}
    parameters["particle_diags"] = {}

    # optionnals

    parameters["alpha"] = sp.getfloat("alpha") if "alpha" in sp else 1.0
    parameters["injection"] = {}
    if "electron scattering" in sp:
        parameters["isotropic"] = sp["electron scattering"] == "isotropic"
    # charged species and neutrals
    for section in config:
        category, *symbol = section.split(" ")
        symbol = " ".join(symbol)
        if category in ("specie", "neutral"):
            part = parse_molecule(symbol)

            if "path" in config[section]:
                path = Path(config[section]["path"])
                array = np.loadtxt(file_dir / path)
                x = array[0] * u.m
                n = array[1] * u.m**-3
                T = array[2] * u.eV
                if array.shape[0] == 4:
                    v = array[3] * u.m / u.s
                    arrays = (x, n, T, v)
                else:
                    arrays = (x, n, T, None)
                T_inj = None
                if "T" in config[section]:
                    T_inj = temp_parser(config[section]["T"])
                arrays = (*arrays, T_inj)
                parameters[f"{category}s"][part] = arrays

            elif "T" in config[section]:
                T = temp_parser(config[section]["T"])
                n = density_parser(config[section]["n"], T)
                parameters[f"{category}s"][part] = (T, n)
            else:
                raise ValueError(f"missing T or path in {section}")

            if category == "specie" and "diagnostics" in config[section]:
                parameters["particle_diags"][part] = [
                    diag.strip() for diag in config[section]["diagnostics"].split(",")
                ]

            if category == "specie" and "EDF" in config[section]:
                [number, resolution, energy_max, *centered] = config[section][
                    "EDF"
                ].split(",")

                parameters["particle_diags"][part].append(
                    (
                        "EDF",
                        int(number),
                        int(resolution),
                        u.Quantity(energy_max, unit=u.J),
                        bool(centered),  # centered = True if there
                    )
                )

            if category == "specie" and "VDF" in config[section]:
                [number, resolution, energy_max, *centered] = config[section][
                    "VDF"
                ].split(",")

                parameters["particle_diags"][part].append(
                    (
                        "VDF",
                        int(number),
                        int(resolution),
                        u.Quantity(energy_max, unit=u.J),
                        bool(centered),  # centered = True if there
                    )
                )

            if category == "specie" and "EDFN" in config[section]:
                [number, resolution, energy_max, *centered] = config[section][
                    "EDFN"
                ].split(",")

                parameters["particle_diags"][part].append(
                    (
                        "EDFN",
                        int(number),
                        int(resolution),
                        u.Quantity(energy_max, unit=u.J),
                        bool(centered),  # centered = True if there
                    )
                )

            if category == "specie" and "injection" in config[section]:
                parameters["injection"][part] = parse_injection(
                    config[section]["injection"]
                )
            if category == "specie" and "collisions_rates" in config[section]:
                parameters["particle_diags"][part].append(
                    (
                        "collisions_rates",
                        [
                            diag.strip()
                            for diag in config[section]["collisions_rates"].split(",")
                            if diag
                        ],
                    )
                )
            if (
                category == "specie"
                and "collisions_momentum_transfer" in config[section]
            ):
                parameters["particle_diags"][part].append(
                    (
                        "collisions_momentum_transfer",
                        [
                            diag.strip()
                            for diag in config[section][
                                "collisions_momentum_transfer"
                            ].split(",")
                        ],
                    )
                )
            if category == "specie" and "collisions_energy_transfer" in config[section]:
                parameters["particle_diags"][part].append(
                    (
                        "collisions_energy_transfer",
                        [
                            diag.strip()
                            for diag in config[section][
                                "collisions_energy_transfer"
                            ].split(",")
                        ],
                    )
                )

    # reactions
    if "cross_sections" in sp:
        cs_folder = file_dir / Path(sp["cross_sections"])
        print(f"using custom cross sections in {cs_folder}")
    parameters["reactions"] = reactions_parser(
        parameters["species"].keys(), parameters["neutrals"].keys(), cs_folder
    )
    parameters["inter_species"] = inter_reactions_parser(parameters["species"].keys())

    parameters["recombination"] = []
    for section in config:
        if section.startswith("recombination"):
            _, p1, p2 = section.split()
            parameters["recombination"].append(
                (parse_molecule(p1), parse_molecule(p2), float(config[section]["k"]))
            )

    if "inductive heating" in config:
        parameters["inductive_heating"] = (
            u.Quantity(config["inductive heating"]["Wabs"], u.W / u.m**3),
            config.getfloat("inductive heating", "alpha"),
            config.getfloat("inductive heating", "r"),
            u.Quantity(config["inductive heating"]["f"], u.Hz),
            u.Quantity(config["inductive heating"]["E0"], u.V / u.m),
        )
    else:
        parameters["inductive_heating"] = None

    if "fake yz" in config:
        ly = u.Quantity(config["fake yz"]["Ly"], u.m)
        lz = u.Quantity(config["fake yz"]["Lz"], u.m)
        if "h2" in config["fake yz"]:
            data = config.getfloat("fake yz", "h2")
            parameters["fake_yz"] = ("constant", ly, lz, data)
        else:
            parameters["fake_yz"] = (
                "profile",
                ly,
                lz,
                config.getfloat("fake yz", "factor", fallback=1.0),
            )
    if "Other" in config:
        parameters["Other"] = config["Other"]
    else:
        parameters["Other"] = {}

    return parameters


def temp_parser(string):
    temp = u.Quantity(string)
    if temp.unit == u.dimensionless_unscaled:
        return temp * u.eV
    return temp.to(u.eV, equivalencies=u.temperature_energy())


def density_parser(string, temp):
    n = u.Quantity(string)
    if n.unit == u.dimensionless_unscaled:
        return n * u.m**-3
    if n.unit.physical_type == "pressure":
        return (n / temp).to(u.m**-3)
    return n.to(u.m**-3)


def excitation_function(section):
    function = section["type"]
    if function == "constant":
        V = u.Quantity(section["V"], u.V).to_value(u.V)

        def f(t):
            return V

        return f
    if function == "sin":
        V = u.Quantity(section["V"], u.V).to_value(u.V)
        omega = 2 * np.pi * u.Quantity(section["f"], u.Hz).to_value(u.Hz)

        def f(t):
            return V * np.sin(omega * t)

        return f
    return lambda t: 0


def reactions_parser(species, neutrals, cs_folder=Path("cross_sections")):
    reactions = []
    for specie, neutral in product(species, neutrals):
        file = cs_folder / f"{specie.symbol}_{neutral.symbol}.txt"
        if os.path.exists(file):
            reactions_set = reac.ReactionsSet(specie, neutral)
            reactions_set.read_txt(file)
            reactions.append(reactions_set)
    return reactions


def inter_reactions_parser(species):
    reactions = []
    for impacter, target in itertools.combinations(species, 2):
        file = os.path.join("cross_sections", f"{impacter.symbol}_{target.symbol}.txt")
        if os.path.exists(file):
            reactions_set = reac.ReactionsSet(impacter, target)
            reactions_set.read_txt(file)
            reactions.append(reactions_set)
    for target, impacter in itertools.combinations(species, 2):
        file = os.path.join("cross_sections", f"{impacter.symbol}_{target.symbol}.txt")
        if os.path.exists(file):
            reactions_set = reac.ReactionsSet(impacter, target)
            reactions_set.read_txt(file)
            reactions.append(reactions_set)
    return reactions


def get_collisions_names(folder):
    config = configparser.RawConfigParser()
    config.read(os.path.join(folder, "cross_sections.cfg"))

    return [config.get(section, "name") for section in config.sections()]


def parse_molecule(symbol):
    if "_" in symbol:
        s, v, *_ = symbol.split("_")
        result = particles.molecule(s)
        try:
            return particles.CustomParticle(
                result.mass, result.charge, f"{result.symbol}_{v}"
            )
        except particles.exceptions.ChargeError:
            return particles.CustomParticle(
                result.mass, 0 * u.C, f"{result.symbol}_{v}"
            )

    else:
        return particles.molecule(symbol)


def parse_injection(s):
    inj_name, *args = s.split(",")
    if inj_name == "fake ionization":
        return "fake ionization"
    if inj_name == "single speed":
        return (
            "single speed",
            u.Quantity(args[0], u.m**-3),
            u.Quantity(args[1], u.m / u.s),
        )
    if inj_name == "maxwellian flux":
        return (
            "maxwellian flux",
            u.Quantity(args[0], u.m**-3),
            u.Quantity(args[1], u.eV, equivalencies=u.temperature_energy()),
        )
