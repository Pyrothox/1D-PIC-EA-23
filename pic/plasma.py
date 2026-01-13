from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

import astropy.units as u
import plasmapy.particles as particles

from pic.particles import ParticlesGroup
from pic.functions import (
    max_vect3D,
    numba_interp1D_normed,
    numba_interp1D_normed_buff,
    numba_return_part_diag,
    particle_to_grid_order0,
)
from pic.parsing import excitation_function
from pic.MCC import MCC, ProfileGas, inter_species_mcc_factory, mcc_factory
from pic.diagnostics import Diagnostics
from astropy.constants import eps0, e
from pic.poisson_solver import Dirichlet_Dynamic
import h5py
from os import PathLike, path

import pic.walls as walls
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pic.reactions import ReactionsSet


eps0_v = eps0.si.value
e_v = e.si.value


class Plasma:
    """a class with fields, parts, and method to pass from one to the other"""

    def __init__(
        self,
        dT: u.s,
        Nx: int,
        Lx: u.m,
        Npart: int,
        species: Dict[
            particles.ParticleLike,
            tuple[u.Quantity[u.m**-3, u.eV]],
        ],
        neutrals: Dict[
            particles.ParticleLike,
            tuple[u.Quantity[u.m**-3, u.eV]],
        ],
        reactions,
        density: Optional[u.Quantity[u.m**-3]] = None,
        start_step=0,
        particle_diags: Dict[particles.ParticleLike, list[str]] = {},
        diagnostics: list[str] = [],
        left_boundary={"type": "constant", "V": 0 * u.V},
        right_boundary={"type": "constant", "V": 0 * u.V},
        inductive_heating: Optional[
            tuple[
                u.Quantity[u.W / u.m**-3],
                float,
                float,
                u.Quantity[u.Hz],
                u.Quantity[u.V / u.m],
            ]
        ] = None,
        alpha: float = 1.0,
        inter_species: List[ReactionsSet] = [],
        recombination: List[ReactionsSet] = [],
        injection={},
        fake_yz=None,
        isotropic=True,
        **kwargs,
    ):
        # Parameters
        self.Lx: float = Lx.to_value(u.m)
        self.Nx: int = Nx
        self.N_cells: int = self.Nx + 1
        self.dT: float = dT.to_value(u.s)
        if density is None:
            density = species[particles.electron][1]
            if density.shape != ():
                raise ValueError(
                    "Density must be given if the electron population is generated from a profile."
                )
        if density is None:
            raise ValueError("No density given nor electron population")
        self.qf: float = (density * Lx / Npart).si.value
        # particles
        self.species: Dict[particles.ParticleLike, ParticlesGroup] = {}
        self.part_out: Dict[particles.ParticleLike, int] = {}
        for part, start in species.items():
            part_group = ParticlesGroup(
                part.mass.to_value(u.kg),
                part.charge.to_value(u.C),
                start,
                part.symbol,
                self,
            )
            self.species[part] = part_group
            self.part_out[part] = 0

        # poisson Solver
        self.PS = Dirichlet_Dynamic(
            self.N_cells,
            excitation_function(left_boundary),
            excitation_function(right_boundary),
        )
        self.PS.init_thomas()

        self.E = np.zeros((self.N_cells, 3), order="F")
        self.phi = np.zeros(self.N_cells)
        self.rho = np.zeros((self.N_cells))

        self.x_j = np.arange(0, self.N_cells, dtype="float64") * self.Lx / (self.Nx)
        self.Lx = self.x_j[-1]
        self.dx: float = self.x_j[1]
        self.alpha = alpha
        self.norm_rho: float = alpha * e_v * self.dx**-3
        self.norm_phi: float = e_v / (self.dx * eps0_v)

        """Collisions"""
        mcc_factory(self.species, neutrals, reactions, isotropic)
        inter_species_mcc_factory(self.species, inter_species)
        self.recombination = recombination
        self.injection = injection

        if inductive_heating:
            self.induction = True
            self.wabs: float = inductive_heating[0].to_value(u.W / u.m**3)
            self.r = inductive_heating[2]
            self.E_range = np.maximum(0, 1 - self.x_j * inductive_heating[1] / self.Lx)
            self.nrf = int((inductive_heating[3].to_value(u.Hz) * self.dT) ** -1)
            self.E0: float = inductive_heating[4].to_value(u.V / u.m)
            self.Jz = np.zeros((self.N_cells))
        else:
            self.induction = False

        if fake_yz:
            electrons = self.species[particles.electron]
            ions = [sp for sp in self.species.values() if sp.charge > 0]
            match fake_yz:
                case "constant", ly, lz, h2:
                    print("Constant wall")
                    self.wall = walls.Wall(
                        ly.to_value(u.m), lz.to_value(u.m), h2, electrons, ions, self
                    )
                case "profile", ly, lz, factor:
                    neg_ions_list = [
                        ion
                        for sp, ion in self.species.items()
                        if ion.charge < 0 and sp != particles.electron
                    ]
                    if neg_ions_list:
                        neg_ions = neg_ions_list[0]
                        print(f"Electronegative wall with {neg_ions.symbol}")
                        self.wall = walls.WallProfileElectronegative(
                            ly.to_value(u.m),
                            lz.to_value(u.m),
                            electrons,
                            ions,
                            neg_ions,
                            self,
                            factor,
                        )
                    else:
                        print("Profile wall")
                        self.wall = walls.WallProfile(
                            ly.to_value(u.m),
                            lz.to_value(u.m),
                            electrons,
                            ions,
                            self,
                            factor,
                        )
                case _:
                    raise ValueError("Unknown wall type")

        else:
            self.wall = None

        """Distribution parameters"""
        self.D = 10 * self.dx

        self.init_poisson = False

        """Physical constantes"""

        self.diag_values = dict()
        self.diagnostics = Diagnostics(
            self, particle_diags, diagnostics, start_step, self.diag_values
        )
        try:
            match species[particles.electron]:
                case T, n:
                    self.wpe = np.sqrt(
                        n * e.si**2 / (eps0 * particles.electron.mass)
                    ).si
                    self.LDe = np.sqrt(eps0 * T / (e.si**2 * n)).si
                case _x, n, T, _v, _T_inj:
                    self.wpe = np.max(
                        np.sqrt(n * e.si**2 / (eps0 * particles.electron.mass)).si
                    )
                    self.LDe = np.min(np.sqrt(eps0 * T / (e.si**2 * n)).si)
                case _:
                    self.wpe = np.NaN * u.rad / u.s
                    self.LDe = np.NaN * u.m
        except FloatingPointError:
            self.wpe = np.NaN * u.rad / u.s
            self.LDe = np.NaN * u.m

    def print_init(self):
        """print some stuffs, upgrade would be a graphic interface"""
        print("~~~~~~ Initialisation of Plasma simulation ~~~~~~~~~~")
        print(f"wpe*dt = {(self.wpe * self.dT * u.s).si}")
        print(
            "mesh step dX= {:2.2f} mu m, LDe = {:2.2f}".format(
                self.dx * 1e6, self.LDe.to(u.um)
            )
        )
        print(" Let's go !!")

    def init_particles(self):
        for part in self.species.values():
            part.init_part()

    def save(self, folder: PathLike, nt: int) -> None:
        with h5py.File(path.join(folder, "restart.h5"), "w") as f:
            f.create_dataset("nt", data=nt)
            for specie, parts in self.species.items():
                grp = f.create_group(specie.symbol)
                grp.attrs["Npart"] = parts.Npart
                grp.create_dataset("x", data=parts.x[: parts.Npart])
                grp.create_dataset("V", data=parts.V[: parts.Npart, :])
            if self.induction:
                grp = f.create_group("induction")
                grp.create_dataset("E0", data=self.E0)
                grp.create_dataset("Jz", data=self.Jz)

    def load(self, file: PathLike) -> int:
        with h5py.File(file, "r") as f:
            for specie, parts in self.species.items():
                grp = f[specie.symbol]
                parts.init_restart(grp.attrs["Npart"], grp["x"][()], grp["V"][()])
            if self.induction:
                grp = f["induction"]
                self.E0 = grp["E0"][()]
                self.Jz = grp["Jz"][()]
            nt = f["nt"][()]
        return nt

    def pusher(self) -> None:
        """push the particles"""

        for part in self.species.values():
            # fast calculation

            partx = part.x[: part.Npart]
            vectE = numba_interp1D_normed_buff(
                partx / self.dx, self.E[:, 0], part.E_interp[: part.Npart]
            )
            part.V[: part.Npart, 0] += part.charge / part.m * self.dT * vectE

            if self.induction:
                part.V[: part.Npart, 2] += (
                    part.charge
                    / part.m
                    * self.dT
                    * numba_interp1D_normed(partx / self.dx, self.E[:, 2])
                )

            part.x[: part.Npart] += part.V[: part.Npart, 0] * self.dT

    def half_depush(self):
        for part in self.species.values():
            partx = part.x[: part.Npart]
            vectE = part.E_interp[: part.Npart]
            part.V[: part.Npart, 0] -= part.charge / part.m * self.dT / 2 * vectE

            if self.induction:
                part.V[: part.Npart, 2] -= (
                    part.charge
                    / part.m
                    * self.dT
                    / 2
                    * numba_interp1D_normed(partx / self.dx, self.E[:, 2])
                )

    def half_push(self):
        for part in self.species.values():
            partx = part.x[: part.Npart]
            vectE = numba_interp1D_normed_buff(
                partx / self.dx, self.E[:, 0], part.E_interp[: part.Npart]
            )

            part.V[: part.Npart, 0] += part.charge / part.m * self.dT / 2 * vectE

            if self.induction:
                part.V[: part.Npart, 2] += (
                    part.charge
                    / part.m
                    * self.dT
                    / 2
                    * numba_interp1D_normed(partx / self.dx, self.E[:, 2])
                )

    def boundary(self, absorbtion=True):
        """look at the postition of the particle,
        and remove them if they are outside"""

        for specie, part in self.species.items():
            self.part_out[specie] += part.remove_parts(self.Lx, absorbtion)

    def apply_mcc(self):
        for part in self.species.values():
            part.apply_mcc(self.dT)
        if self.wall is not None:
            self.wall.absorbtion()

    def recombine(self, n_steps):   #not used
        for s1, s2, k in self.recombination:
            p1 = self.species[s1]
            p2 = self.species[s2]
            N = k * self.qf * p1.Npart * p2.Npart * self.dT * n_steps / self.Lx
            part = N - np.floor(N)
            N = int(N) + (1 if np.random.rand() < part else 0)
            if N:
                p1.remove_random(N)
                p2.remove_random(N)

    def compute_rho(self):
        """Compute the plasma density via the invers aera method"""
        self.rho[:] = 0
        for part in self.species.values():
            self.rho += part.update_density(self.x_j) * part.charge

        self.rho *= self.qf / self.dx

    def solve_poisson(self, nt: int):
        """solve poisson via the Thomas method"""

        normed_rho = self.rho / self.norm_rho

        normed_phi = self.PS.thomas_solver(normed_rho, nt * self.dT, self.norm_phi)

        self.phi = normed_phi * self.norm_phi

        #        Poisson finished
        self.E[:, 0] = -np.gradient(self.phi, self.dx)

        if self.induction:
            phase = np.sin(2 * np.pi * nt / self.nrf)
            self.E[:, 2] = self.E0 * phase * self.E_range

    def diags(self, nt: int):
        self.half_push()

        if self.induction:
            for part in self.species.values():
                phase = np.sin(2 * np.pi * nt / self.nrf)
                numba_return_part_diag(
                    part.Npart,
                    part.x[: part.Npart],
                    part.V[: part.Npart, 2] * (phase * part.charge * self.qf),
                    self.x_j,
                    self.Jz,
                    self.dx,
                    power=1,
                )

            if nt > 0 and not nt % self.nrf:
                newE = self.nrf * self.Lx * self.wabs / (np.sum(self.Jz * self.E_range))

                self.E0 = self.r * self.E0 + (1 - self.r) * newE
                self.Jz[:] = 0

        for part in self.species.values():
            if part.need_u:
                part.update_u(self.x_j)
            if part.need_c:
                part.update_c()
        self.diagnostics.diags(nt)
        self.half_depush()

    def save_diagnostics(self, nt, folder):
        self.diagnostics.average_diags(self.diagnostics.average)
        self.diagnostics.save_diags(nt, folder)

    def inject(self):
        """inject particles"""
        for specie, injection_type in self.injection.items():
            match injection_type:
                case "fake ionization":
                    self.fake_ionization(specie)
                case ("single speed", n, v):
                    self.species[specie].inject_same_speed(
                        n.to_value(u.m**-3), v.to_value(u.m / u.s)
                    )
                case ("maxwellian flux", n, T):
                    self.species[specie].inject_maxwellian_flux(
                        n.to_value(u.m**-3), T.to_value(u.eV)
                    )

    def fake_ionization(self, specie: particles.ParticleLike):
        elecs = self.species[particles.electron]
        ions = self.species[specie]
        x = np.random.choice(elecs.x[: elecs.Npart], self.part_out[specie])
        v_elecs = max_vect3D(len(x), elecs.T, elecs.m)
        v_ions = max_vect3D(len(x), ions.T, ions.m)
        elecs.add_particles(x, v_elecs)
        ions.add_particles(x, v_ions)
        self.part_out[specie] = 0
        self.part_out[particles.electron] -= len(x)
        if self.inj_diag:
            particle_to_grid_order0(
                x.shape[0], x, self.x_j, self.diag_values["inj"], self.dx
            )

    def change_neutral_profile(
        self, gas: particles.ParticleLike, x: np.ndarray, n: np.ndarray, T: np.ndarray
    ):
        for part in self.species.values():
            for neutral in (mcc.neutral for mcc in part.mccs if isinstance(mcc, MCC)):
                if isinstance(neutral, ProfileGas) and neutral.gas == gas:
                    neutral.x = x
                    neutral.n = n
                    neutral.T = T
