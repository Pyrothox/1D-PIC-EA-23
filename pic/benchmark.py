#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 13:03:47 2021

@author: lequette
"""

import os.path
import numpy as np
from pic.Hdf5Viewer import Hdf5Viewer
import matplotlib.pyplot as plt
import csv
from plasmapy import particles
import astropy.units as u
import pandas as pd

ele = particles.electron
he = particles.Particle("He+")


class Benchmark_results:
    def __init__(self, case, results):
        self.simu = Hdf5Viewer(results)
        file = os.path.join("benchmark", f"case{case}.dat")
        with open(file, "r") as infile:
            reader = csv.reader(infile, delimiter=" ", skipinitialspace=True)
            self.x = []
            self.ne = []
            self.sigma_ne = []
            self.pop_sigma_ne = []
            self.ni = []
            self.sigma_ni = []
            self.pop_sigma_ni = []

            for row in reader:
                self.x.append(float(row[0]))
                self.ne.append(float(row[1]))
                self.sigma_ne.append(float(row[2]))
                self.pop_sigma_ne.append(float(row[3]))
                self.ni.append(float(row[4]))
                self.sigma_ni.append(float(row[5]))
                self.pop_sigma_ni.append(float(row[6]))

            self.x = np.array(self.x) * u.m
            self.ne = np.array(self.ne) * u.m**-3
            self.sigma_ne = np.array(self.sigma_ne) * u.m**-3
            self.pop_sigma_ne = np.array(self.pop_sigma_ne) * u.m**-3
            self.ni = np.array(self.ni) * u.m**-3
            self.sigma_ni = np.array(self.sigma_ni) * u.m**-3
            self.pop_sigma_ni = np.array(self.pop_sigma_ni) * u.m**-3

        eepfs = np.loadtxt(os.path.join("benchmark", f"eepf{case}.csv"), delimiter=";")

        self.eepf = eepfs.T[1] / u.eV ** (3 / 2)
        self.e_eepf = eepfs.T[0] * u.eV

        ds = pd.read_csv(os.path.join("benchmark", "Turner_ionization.csv"), sep=",")
        cx = ds[f"case{case}X"]
        c_ionization = ds[f"case{case}Y"]
        index = cx.last_valid_index() + 1
        self.x_ioniz = cx[:index].to_numpy() * self.simu.parameters["Lx"]
        self.ioniz = c_ionization[:index].to_numpy() * 1e20 * u.m**-3 / u.s

        ds = pd.read_csv(os.path.join("benchmark", "Turner_power.csv"), sep=",")
        cx = ds[f"{case}X"]
        c_pow = ds[f"{case}Y"]
        index = cx.last_valid_index() + 1
        self.x_pow = cx[:index].to_numpy() * self.simu.parameters["Lx"]
        self.power = c_pow[:index].to_numpy() * u.kW * u.m**-3

    def _compare(self, array, specie, ax, mean=1):
        res = np.zeros_like(self.ne)
        for i in range(mean):
            res += self.simu.particle_data[specie]["n"][-1 - i, :]
        res /= mean

        ax.plot(self.x, array, label=f"benchmark {specie.symbol}")
        ax.plot(self.simu.x, res, label=f"simulation {specie.symbol}")

    def compare_ne(self, ax=None, mean=1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        self._compare(self.ne, ele, ax, mean=mean)

    def compare_ni(self, ax=None, mean=1):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        self._compare(self.ni, he, ax, mean=mean)

    def compare_eepf(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        _, energy, data = self.simu.edf[ele]
        edf = data[-1, 0, :]
        eepf = edf / (np.sqrt(energy) * np.trapz(edf, energy))
        ax.semilogy(self.e_eepf, self.eepf * np.sqrt(2 * np.pi), label="benchmark ")
        ax.semilogy(energy.to(u.eV), eepf.to(u.eV ** (-3 / 2)), label="simulation ")
        ax.legend()
        ax.set_xlabel("energy [eV]")
        ax.set_ylabel("eepf [eV$^{-3/2}$]")

    def compare_eepfn(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        _, energy, data = self.simu.edfn[ele]
        edf = data[-1, 0, :]
        eepf = edf / (np.sqrt(energy))
        ax.semilogy(self.e_eepf, self.eepf * np.sqrt(2 * np.pi), label="benchmark ")
        ax.semilogy(energy.to(u.eV), eepf.to(u.eV ** (-3 / 2)), label="simulation ")
        ax.legend()
        ax.set_xlabel("energy [eV]")
        ax.set_ylabel("eepf [eV$^{-3/2}$]")

    def compare_ionization(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plt.plot(self.x_ioniz, self.ioniz, label="benchmark")
        plt.plot(
            self.simu.x,
            self.simu.collisions_rates[ele][particles.Particle("He")]["IONIZATION"][-1],
            label="simulation",
        )
        plt.legend()

    def compare_power(self, ax=None):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        plt.plot(self.x_pow, self.power, label="benchmark")
        plt.plot(
            self.simu.x,
            self.simu.particle_data[ele]["Power"][-1].to(u.kW * u.m**-3),
            label="simulation",
        )
        plt.legend()

    def is_passed(self):
        np.testing.assert_allclose(
            self.ne,
            self.simu.particle_data[ele]["n"][-1, :],
            rtol=0.1,
            err_msg="electron density error above 10%",
        )
        np.testing.assert_allclose(
            self.ni,
            self.simu.particle_data[he]["n"][-1, :],
            rtol=0.1,
            err_msg="ion density error above 10%",
        )

    def error(self, array, specie, mean=1):
        res = np.zeros_like(self.ne)
        for i in range(mean):
            res += self.simu.particle_data[specie]["n"][-1 - i, :]
        res /= mean
        error = (array - res) / array
        plt.figure()
        plt.plot(self.x, error)
        plt.title(specie.symbol)
        plt.legend()
        plt.show()

    def error_ne(self, mean=1):
        self.error(self.ne, ele, mean=mean)

    def error_ni(self, mean=1):
        self.error(self.ni, he, mean=mean)
