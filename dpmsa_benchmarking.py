#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPMSA Benchmarking Script
Compares PIC simulations with and without Dynamic Particle Merging & Splitting Algorithm

Created: 2026-02-08
Author: Manef
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from pathlib import Path

from pic.Hdf5Viewer import Hdf5Viewer
from plasmapy import particles
import astropy.units as u

# Define particles
ele = particles.electron
ion = particles.Particle("He+")  # Adjust based on your simulation
neutral = particles.Particle("He")  # Adjust based on your simulation


class DPMSA_Benchmark:
    """
    Benchmark class to compare simulations with and without DPMSA
    """
    
    def __init__(self, sim_without_ms_path, sim_with_ms_path):
        """
        Initialize benchmark with paths to simulation data folders
        
        Parameters:
        -----------
        sim_without_ms_path : str
            Path to simulation data WITHOUT merging & splitting
        sim_with_ms_path : str
            Path to simulation data WITH merging & splitting
        """
        print(f"Loading simulation without M&S from: {sim_without_ms_path}")
        self.sim_no_ms = Hdf5Viewer(sim_without_ms_path)
        
        print(f"Loading simulation with M&S from: {sim_with_ms_path}")
        self.sim_with_ms = Hdf5Viewer(sim_with_ms_path)
        
        # Extract metadata
        self.x = self.sim_no_ms.x
        self.t = self.sim_no_ms.t
        
        # Load parameters
        self.params_no_ms = self.sim_no_ms.parameters
        self.params_with_ms = self.sim_with_ms.parameters
        
        print(f"\nSimulation details:")
        print(f"  Domain length: {self.params_no_ms['Lx']}")
        print(f"  Grid cells: {self.params_no_ms['Nx']}")
        print(f"  Time steps: {self.params_no_ms['Nt']}")
        print(f"  Initial particles: {self.params_no_ms['Npart']}")
        
    def compare_densities(self, time_index=-1, average=5, ax=None, species='both'):
        """
        Compare electron and ion density profiles
        
        Parameters:
        -----------
        time_index : int
            Time index to compare (-1 for final time)
        average : int
            Number of time steps to average
        ax : matplotlib axis
            Axis to plot on (creates new if None)
        species : str
            'electron', 'ion', or 'both'
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        i_start = max(0, i - average // 2)
        i_end = i_start + average
        
        if species in ['electron', 'both']:
            ne_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
            )
            ne_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
            )
            
            ax.plot(self.x, ne_no_ms, 'b-', linewidth=2, label='Electron (No M&S)')
            ax.plot(self.x, ne_with_ms, 'b--', linewidth=2, label='Electron (With M&S)')
        
        if species in ['ion', 'both']:
            ni_no_ms = np.mean(
                self.sim_no_ms.particle_data[ion]["n"][i_start:i_end], axis=0
            )
            ni_with_ms = np.mean(
                self.sim_with_ms.particle_data[ion]["n"][i_start:i_end], axis=0
            )
            
            ax.plot(self.x, ni_no_ms, 'r-', linewidth=2, label='Ion (No M&S)')
            ax.plot(self.x, ni_with_ms, 'r--', linewidth=2, label='Ion (With M&S)')
        
        ax.set_xlabel('Position [m]', fontsize=12)
        ax.set_ylabel('Density [m⁻³]', fontsize=12)
        ax.set_title(f'Density Comparison at t = {self.t[i]:.2e}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_temperature(self, time_index=-1, average=5, ax=None, species='both'):
        """
        Compare electron and ion temperature profiles
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        i_start = max(0, i - average // 2)
        i_end = i_start + average
        
        if species in ['electron', 'both'] and 'T' in self.sim_no_ms.particle_data[ele]:
            Te_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            Te_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            
            ax.plot(self.x, Te_no_ms, 'b-', linewidth=2, label='Electron (No M&S)')
            ax.plot(self.x, Te_with_ms, 'b--', linewidth=2, label='Electron (With M&S)')
        
        if species in ['ion', 'both'] and 'T' in self.sim_no_ms.particle_data[ion]:
            Ti_no_ms = np.mean(
                self.sim_no_ms.particle_data[ion]["T"][i_start:i_end], axis=0
            )
            Ti_with_ms = np.mean(
                self.sim_with_ms.particle_data[ion]["T"][i_start:i_end], axis=0
            )
            
            ax.plot(self.x, Ti_no_ms, 'r-', linewidth=2, label='Ion (No M&S)')
            ax.plot(self.x, Ti_with_ms, 'r--', linewidth=2, label='Ion (With M&S)')
        
        ax.set_xlabel('Position [m]', fontsize=12)
        ax.set_ylabel('Temperature [eV]', fontsize=12)
        ax.set_title(f'Temperature Comparison at t = {self.t[i]:.2e}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_eepf(self, time_index=-1, probe_index=0, ax=None):
        """
        Compare Electron Energy Probability Function (EEPF)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        
        if ele in self.sim_no_ms.edf:
            x_probes, energy_no_ms, edf_no_ms = self.sim_no_ms.edf[ele]
            _, energy_with_ms, edf_with_ms = self.sim_with_ms.edf[ele]
            
            # Convert EDF to EEPF (normalized by sqrt(energy))
            eepf_no_ms = edf_no_ms[i, probe_index, :] / np.sqrt(energy_no_ms)
            eepf_with_ms = edf_with_ms[i, probe_index, :] / np.sqrt(energy_with_ms)
            
            # Normalize by integral
            eepf_no_ms /= np.trapz(eepf_no_ms, energy_no_ms.to(u.eV).value)
            eepf_with_ms /= np.trapz(eepf_with_ms, energy_with_ms.to(u.eV).value)
            
            ax.semilogy(
                energy_no_ms.to(u.eV), 
                eepf_no_ms, 
                'b-', 
                linewidth=2, 
                label='No M&S'
            )
            ax.semilogy(
                energy_with_ms.to(u.eV), 
                eepf_with_ms, 
                'r--', 
                linewidth=2, 
                label='With M&S'
            )
            
            ax.set_xlabel('Energy [eV]', fontsize=12)
            ax.set_ylabel('EEPF [eV⁻³/²]', fontsize=12)
            ax.set_title(
                f'EEPF Comparison at x = {x_probes[probe_index]:.3f} m', 
                fontsize=14
            )
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'EDF data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def compare_power_deposition(self, time_index=-1, average=5, ax=None):
        """
        Compare power deposition profiles
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        i_start = max(0, i - average // 2)
        i_end = i_start + average
        
        if 'Power' in self.sim_no_ms.particle_data[ele]:
            power_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            )
            power_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            )
            
            ax.plot(
                self.x, 
                power_no_ms.to(u.kW / u.m**3), 
                'b-', 
                linewidth=2, 
                label='No M&S'
            )
            ax.plot(
                self.x, 
                power_with_ms.to(u.kW / u.m**3), 
                'r--', 
                linewidth=2, 
                label='With M&S'
            )
            
            ax.set_xlabel('Position [m]', fontsize=12)
            ax.set_ylabel('Power Deposition [kW/m³]', fontsize=12)
            ax.set_title(f'Power Deposition at t = {self.t[i]:.2e}', fontsize=14)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Power deposition data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def compare_ionization_rate(self, time_index=-1, ax=None):
        """
        Compare ionization rate spatial distribution
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        
        if (ele in self.sim_no_ms.collisions_rates and 
            neutral in self.sim_no_ms.collisions_rates[ele]):
            
            rates_no_ms = self.sim_no_ms.collisions_rates[ele][neutral]
            rates_with_ms = self.sim_with_ms.collisions_rates[ele][neutral]
            
            # Find ionization reaction
            ioniz_key = [k for k in rates_no_ms.keys() if 'IONIZATION' in k.upper()]
            
            if ioniz_key:
                ioniz_no_ms = rates_no_ms[ioniz_key[0]][i, :]
                ioniz_with_ms = rates_with_ms[ioniz_key[0]][i, :]
                
                ax.plot(
                    self.x, 
                    ioniz_no_ms.to(u.m**-3 / u.s), 
                    'b-', 
                    linewidth=2, 
                    label='No M&S'
                )
                ax.plot(
                    self.x, 
                    ioniz_with_ms.to(u.m**-3 / u.s), 
                    'r--', 
                    linewidth=2, 
                    label='With M&S'
                )
                
                ax.set_xlabel('Position [m]', fontsize=12)
                ax.set_ylabel('Ionization Rate [m⁻³/s]', fontsize=12)
                ax.set_title(f'Ionization Rate at t = {self.t[i]:.2e}', fontsize=14)
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Ionization data not found', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Collision rate data not available', 
                   ha='center', va='center', transform=ax.transAxes)
        
        return ax
    
    def compare_particle_evolution(self, ax=None):
        """
        Compare total particle count evolution over time
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract particle count from parameters or diagnostics
        # Note: This requires tracking particle count in diagnostics
        # If not available, we can estimate from density integral
        
        # Calculate total particles (integral of density * volume element)
        dx = self.sim_no_ms.dx
        
        n_electrons_no_ms = np.sum(
            self.sim_no_ms.particle_data[ele]["n"], axis=1
        ) * dx
        n_electrons_with_ms = np.sum(
            self.sim_with_ms.particle_data[ele]["n"], axis=1
        ) * dx
        
        ax.plot(
            self.t[:self.sim_no_ms.n_files], 
            n_electrons_no_ms[:self.sim_no_ms.n_files], 
            'b-', 
            linewidth=2, 
            label='Electrons (No M&S)'
        )
        ax.plot(
            self.t[:self.sim_with_ms.n_files], 
            n_electrons_with_ms[:self.sim_with_ms.n_files], 
            'r--', 
            linewidth=2, 
            label='Electrons (With M&S)'
        )
        
        ax.set_xlabel('Time [s]', fontsize=12)
        ax.set_ylabel('Total Particles (Integrated Density)', fontsize=12)
        ax.set_title('Particle Conservation Check', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compute_relative_errors(self, time_index=-1, average=5):
        """
        Compute relative errors between the two simulations
        
        Returns dictionary with relative errors for various quantities
        """
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        i_start = max(0, i - average // 2)
        i_end = i_start + average
        
        errors = {}
        
        # Electron density error
        ne_no_ms = np.mean(
            self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        ne_with_ms = np.mean(
            self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        errors['electron_density'] = {
            'max_relative': np.max(np.abs((ne_with_ms - ne_no_ms) / ne_no_ms)),
            'mean_relative': np.mean(np.abs((ne_with_ms - ne_no_ms) / ne_no_ms)),
            'rms': np.sqrt(np.mean(((ne_with_ms - ne_no_ms) / ne_no_ms)**2))
        }
        
        # Ion density error
        ni_no_ms = np.mean(
            self.sim_no_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        ni_with_ms = np.mean(
            self.sim_with_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        errors['ion_density'] = {
            'max_relative': np.max(np.abs((ni_with_ms - ni_no_ms) / ni_no_ms)),
            'mean_relative': np.mean(np.abs((ni_with_ms - ni_no_ms) / ni_no_ms)),
            'rms': np.sqrt(np.mean(((ni_with_ms - ni_no_ms) / ni_no_ms)**2))
        }
        
        # Temperature errors (if available)
        if 'T' in self.sim_no_ms.particle_data[ele]:
            Te_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            Te_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            errors['electron_temperature'] = {
                'max_relative': np.max(np.abs((Te_with_ms - Te_no_ms) / Te_no_ms)),
                'mean_relative': np.mean(np.abs((Te_with_ms - Te_no_ms) / Te_no_ms)),
                'rms': np.sqrt(np.mean(((Te_with_ms - Te_no_ms) / Te_no_ms)**2))
            }
        
        return errors
    
    def plot_relative_error_profiles(self, time_index=-1, average=5):
        """
        Plot spatial profiles of relative errors
        """
        i = time_index if time_index >= 0 else self.sim_no_ms.n_files + time_index
        i_start = max(0, i - average // 2)
        i_end = i_start + average
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Electron density error
        ne_no_ms = np.mean(
            self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        ne_with_ms = np.mean(
            self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        rel_error_ne = (ne_with_ms - ne_no_ms) / ne_no_ms * 100
        
        axes[0, 0].plot(self.x, rel_error_ne, 'b-', linewidth=2)
        axes[0, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0, 0].set_xlabel('Position [m]')
        axes[0, 0].set_ylabel('Relative Error [%]')
        axes[0, 0].set_title('Electron Density Error')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Ion density error
        ni_no_ms = np.mean(
            self.sim_no_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        ni_with_ms = np.mean(
            self.sim_with_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        rel_error_ni = (ni_with_ms - ni_no_ms) / ni_no_ms * 100
        
        axes[0, 1].plot(self.x, rel_error_ni, 'r-', linewidth=2)
        axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Position [m]')
        axes[0, 1].set_ylabel('Relative Error [%]')
        axes[0, 1].set_title('Ion Density Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Temperature errors (if available)
        if 'T' in self.sim_no_ms.particle_data[ele]:
            Te_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            Te_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["T"][i_start:i_end], axis=0
            )
            rel_error_Te = (Te_with_ms - Te_no_ms) / Te_no_ms * 100
            
            axes[1, 0].plot(self.x, rel_error_Te, 'b-', linewidth=2)
            axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].set_xlabel('Position [m]')
            axes[1, 0].set_ylabel('Relative Error [%]')
            axes[1, 0].set_title('Electron Temperature Error')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'T' in self.sim_no_ms.particle_data[ion]:
            Ti_no_ms = np.mean(
                self.sim_no_ms.particle_data[ion]["T"][i_start:i_end], axis=0
            )
            Ti_with_ms = np.mean(
                self.sim_with_ms.particle_data[ion]["T"][i_start:i_end], axis=0
            )
            rel_error_Ti = (Ti_with_ms - Ti_no_ms) / Ti_no_ms * 100
            
            axes[1, 1].plot(self.x, rel_error_Ti, 'r-', linewidth=2)
            axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Position [m]')
            axes[1, 1].set_ylabel('Relative Error [%]')
            axes[1, 1].set_title('Ion Temperature Error')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_full_comparison_report(self, output_dir='benchmark_results', time_index=-1):
        """
        Generate a comprehensive comparison report with all plots
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("DPMSA BENCHMARK REPORT")
        print("="*60)
        
        # 1. Main comparison figure (6 subplots)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self.compare_densities(time_index=time_index, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.compare_temperature(time_index=time_index, ax=ax2)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self.compare_eepf(time_index=time_index, ax=ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self.compare_power_deposition(time_index=time_index, ax=ax4)
        
        ax5 = fig.add_subplot(gs[2, 0])
        self.compare_ionization_rate(time_index=time_index, ax=ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self.compare_particle_evolution(ax=ax6)
        
        plt.suptitle('DPMSA Benchmark: Complete Comparison', fontsize=16, y=0.995)
        plt.savefig(os.path.join(output_dir, 'full_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/full_comparison.png")
        
        # 2. Relative error profiles
        fig_errors = self.plot_relative_error_profiles(time_index=time_index)
        plt.savefig(os.path.join(output_dir, 'relative_errors.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/relative_errors.png")
        
        # 3. Compute and print numerical errors
        errors = self.compute_relative_errors(time_index=time_index)
        
        print("\n" + "-"*60)
        print("NUMERICAL ERROR ANALYSIS")
        print("-"*60)
        
        for quantity, error_dict in errors.items():
            print(f"\n{quantity.upper().replace('_', ' ')}:")
            print(f"  Max relative error:  {error_dict['max_relative']*100:.2f}%")
            print(f"  Mean relative error: {error_dict['mean_relative']*100:.2f}%")
            print(f"  RMS error:           {error_dict['rms']*100:.2f}%")
        
        # 4. Save error data to file
        with open(os.path.join(output_dir, 'error_summary.txt'), 'w') as f:
            f.write("DPMSA BENCHMARK - ERROR SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Simulation without M&S: {self.sim_no_ms.folder}\n")
            f.write(f"Simulation with M&S:    {self.sim_with_ms.folder}\n\n")
            
            for quantity, error_dict in errors.items():
                f.write(f"\n{quantity.upper().replace('_', ' ')}:\n")
                f.write(f"  Max relative error:  {error_dict['max_relative']*100:.4f}%\n")
                f.write(f"  Mean relative error: {error_dict['mean_relative']*100:.4f}%\n")
                f.write(f"  RMS error:           {error_dict['rms']*100:.4f}%\n")
        
        print(f"\n✓ Saved: {output_dir}/error_summary.txt")
        
        # 5. Save numerical data for further analysis
        np.savez(
            os.path.join(output_dir, 'benchmark_data.npz'),
            x=self.x.value,
            errors_electron_density=errors['electron_density'],
            errors_ion_density=errors['ion_density'],
            errors_electron_temperature=errors.get('electron_temperature', {}),
        )
        print(f"✓ Saved: {output_dir}/benchmark_data.npz")
        
        print("\n" + "="*60)
        print("BENCHMARK COMPLETE")
        print("="*60 + "\n")
        
        return errors


# Example usage function
def run_benchmark(path_without_ms, path_with_ms, output_dir='benchmark_results'):
    """
    Main function to run the complete benchmark
    
    Parameters:
    -----------
    path_without_ms : str
        Path to simulation data folder WITHOUT merging & splitting
    path_with_ms : str
        Path to simulation data folder WITH merging & splitting
    output_dir : str
        Directory to save benchmark results
    """
    
    # Create benchmark object
    benchmark = DPMSA_Benchmark(path_without_ms, path_with_ms)
    
    # Generate full report
    errors = benchmark.generate_full_comparison_report(output_dir=output_dir)
    
    # Show plots
    plt.show()
    
    return benchmark, errors


if __name__ == "__main__":
    """
    Example execution
    
    To use this script:
    1. Run your simulation WITHOUT merging/splitting
    2. Run your simulation WITH merging/splitting (same configuration)
    3. Update the paths below to your data folders
    4. Run this script
    """
    
    # UPDATE THESE PATHS to your actual simulation data folders
    path_no_ms = "data/2026-02-08_12h17"
    path_with_ms = "data/2026-02-08_13h14"
    
    # Check if paths exist
    if not os.path.exists(path_no_ms):
        print(f"ERROR: Path not found: {path_no_ms}")
        print("Please update the path_no_ms variable with your actual data folder")
    elif not os.path.exists(path_with_ms):
        print(f"ERROR: Path not found: {path_with_ms}")
        print("Please update the path_with_ms variable with your actual data folder")
    else:
        # Run benchmark
        benchmark, errors = run_benchmark(path_no_ms, path_with_ms)
