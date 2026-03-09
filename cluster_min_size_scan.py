#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster_min_size Scan - Stationary half-history average

Compare l'effet de cluster_min_size sur :
- densité
- dépôt de puissance
- vitesse (composante choisie, signée)
- conservation des particules

Chaque run est moyenné sur la 2ème moitié des fichiers HDF5
(pour ne garder que le régime stationnaire).
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from pic.Hdf5Viewer import Hdf5Viewer
from plasmapy import particles
import astropy.units as u

ele = particles.electron
ion = particles.Particle("He+")
neutral = particles.Particle("He")


class ClusterMinSizeScan:
    def __init__(self, base_data_dir="data", cluster_values=None):
        if cluster_values is None:
            cluster_values = [2, 3, 4, 5, 10, 20]

        self.base_data_dir = Path(base_data_dir)
        self.cluster_values = cluster_values
        self.viewers = {}

        print("\n=== Loading simulations for cluster_min_size scan ===")
        for val in self.cluster_values:
            folder = self.base_data_dir / f"cluster_min_size_{val}"
            print(f"cluster_min_size = {val}  -->  {folder}")
            if not folder.exists():
                print("  ✗ Folder not found, skipping.")
                continue
            try:
                self.viewers[val] = Hdf5Viewer(str(folder))
                print("  ✓ Loaded")
            except Exception as e:
                print(f"  ✗ Error loading: {e}")

        if not self.viewers:
            raise RuntimeError("No valid cluster_min_size folders found.")

        ref_val = next(iter(self.viewers.keys()))
        self.ref_viewer = self.viewers[ref_val]
        self.x = self.ref_viewer.x
        self.t = self.ref_viewer.t
        self.dx = self.ref_viewer.dx

        print("\nSimulations loaded for cluster_min_size values:", list(self.viewers.keys()))

    # ---------- fenêtre temporelle stationnaire (2ème moitié) ----------

    def _get_stationary_window(self, viewer):
        """
        Retourne (i, i_start, i_end) pour un viewer donné :
        - i_start = n_files/2
        - i_end   = n_files
        - i       = n_files - 1
        """
        n_files = viewer.n_files
        i_start = int(0.5 * n_files)
        i_end = n_files
        i = n_files - 1
        return i, i_start, i_end

    # ---------- Profils de densité ----------

    def plot_density_profiles(self, species="electron", output=None):
        """
        Profils de densité (stationary average) pour différentes valeurs de cluster_min_size.
        species : 'electron' ou 'ion'
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for val, V in self.viewers.items():
            i, i_start, i_end = self._get_stationary_window(V)

            if species == "electron":
                sp = ele
            else:
                sp = ion

            if "n" not in V.particle_data.get(sp, {}):
                continue

            n_prof = np.mean(V.particle_data[sp]["n"][i_start:i_end], axis=0)
            ax.plot(
                self.x,
                n_prof,
                label=f"{species.capitalize()} (cms={val})",
            )

        ax.set_xlabel("Position [m]", fontsize=12)
        ax.set_ylabel("Density [m⁻³]", fontsize=12)
        ax.set_title(f"{species.capitalize()} density vs cluster_min_size (stationary avg)", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if output is not None:
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"✓ Saved density plot: {output}")

        return fig, ax

    # ---------- Profils de dépôt de puissance ----------

    def plot_power_profiles(self, species="electron", output=None):
        """
        Profils de dépôt de puissance (stationary average) pour différentes valeurs de cluster_min_size.
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for val, V in self.viewers.items():
            i, i_start, i_end = self._get_stationary_window(V)

            if species == "electron":
                sp = ele
            else:
                sp = ion

            if "Power" not in V.particle_data.get(sp, {}):
                continue

            P_prof = np.mean(V.particle_data[sp]["Power"][i_start:i_end], axis=0)
            ax.plot(
                self.x,
                P_prof.to(u.kW / u.m**3),
                label=f"{species.capitalize()} (cms={val})",
            )

        ax.set_xlabel("Position [m]", fontsize=12)
        ax.set_ylabel("Power deposition [kW/m³]", fontsize=12)
        ax.set_title(f"{species.capitalize()} power vs cluster_min_size (stationary avg)", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if output is not None:
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"✓ Saved power plot: {output}")

        return fig, ax

    # ---------- Profils de vitesse (composante, signée) ----------

    def plot_velocity_profiles(self, component=0, output=None):
        """
        Profils de vitesse (composante choisie) pour e- et ions,
        moyenne sur la 2ème moitié de l'historique.
        component : 0 = Vx, 1 = Vy, 2 = Vz
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        comp_labels = ["x", "y", "z"]

        for val, V in self.viewers.items():
            i, i_start, i_end = self._get_stationary_window(V)

            # Electrons
            if "V" in V.particle_data.get(ele, {}):
                Ve_comp = np.mean(
                    V.particle_data[ele]["V"][i_start:i_end, :, component],
                    axis=0,
                )
                ax.plot(
                    self.x,
                    Ve_comp,
                    linestyle="-",
                    linewidth=1.5,
                    label=f"Electron V{comp_labels[component]} (cms={val})",
                )

            # Ions
            if "V" in V.particle_data.get(ion, {}):
                Vi_comp = np.mean(
                    V.particle_data[ion]["V"][i_start:i_end, :, component],
                    axis=0,
                )
                ax.plot(
                    self.x,
                    Vi_comp,
                    linestyle="--",
                    linewidth=1.5,
                    label=f"Ion V{comp_labels[component]} (cms={val})",
                )

        ax.set_xlabel("Position [m]", fontsize=12)
        ax.set_ylabel(f"Velocity {comp_labels[component]} [m/s]", fontsize=12)
        ax.set_title(
            f"Velocity {comp_labels[component]} vs cluster_min_size (stationary avg)",
            fontsize=14,
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if output is not None:
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"✓ Saved velocity plot: {output}")

        return fig, ax

    # ---------- Conservation des particules (évolution complète) ----------

    def plot_particle_conservation(self, output=None):
        """
        Évolution temporelle du nombre total de particules (densité intégrée),
        pour chaque cluster_min_size (sur tout l'historique).
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for val, V in self.viewers.items():
            if "n" not in V.particle_data.get(ele, {}):
                continue

            ne = V.particle_data[ele]["n"]
            Ne_tot = np.sum(ne, axis=1) * self.dx

            ax.plot(
                V.t[: V.n_files],
                Ne_tot[: V.n_files],
                label=f"Electrons (cms={val})",
            )

            if "n" in V.particle_data.get(ion, {}):
                ni = V.particle_data[ion]["n"]
                Ni_tot = np.sum(ni, axis=1) * self.dx
                ax.plot(
                    V.t[: V.n_files],
                    Ni_tot[: V.n_files],
                    linestyle="--",
                    label=f"Ions (cms={val})",
                )

        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Total particles (integrated density)", fontsize=12)
        ax.set_title("Particle conservation vs cluster_min_size", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        if output is not None:
            fig.savefig(output, dpi=300, bbox_inches="tight")
            print(f"✓ Saved particle conservation plot: {output}")

        return fig, ax

    # ---------- Figure globale ----------

    def generate_full_scan_report(self, output_dir="cluster_min_size_results"):
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("CLUSTER_MIN_SIZE SCAN REPORT (stationary half-history average)")
        print("=" * 60)

        # 1) Densité électronique
        self.plot_density_profiles(
            species="electron",
            output=os.path.join(output_dir, "density_electrons.png"),
        )

        # 2) Puissance électronique
        self.plot_power_profiles(
            species="electron",
            output=os.path.join(output_dir, "power_electrons.png"),
        )

        # 3) Vitesse Vx
        self.plot_velocity_profiles(
            component=0,
            output=os.path.join(output_dir, "velocity_Vx_profiles.png"),
        )

        # 4) Conservation particules
        self.plot_particle_conservation(
            output=os.path.join(output_dir, "particle_conservation.png"),
        )

        plt.show()

        print("\n" + "=" * 60)
        print("SCAN COMPLETE")
        print("=" * 60)


def run_cluster_scan(base_data_dir="data", output_dir="cluster_min_size_results"):
    cluster_values = [2, 3, 4, 5, 10, 20]
    scan = ClusterMinSizeScan(base_data_dir=base_data_dir, cluster_values=cluster_values)
    scan.generate_full_scan_report(output_dir=output_dir)
    return scan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan over cluster_min_size and compare diagnostics (stationary half-history average)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Base data directory containing cluster_min_size_i folders.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="cluster_min_size_results",
        help="Output directory for plots.",
    )

    args = parser.parse_args()

    scan = ClusterMinSizeScan(base_data_dir=args.data_dir)
    scan.generate_full_scan_report(output_dir=args.output)
