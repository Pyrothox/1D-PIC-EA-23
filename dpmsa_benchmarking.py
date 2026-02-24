#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPMSA Benchmarking Script - CORRECTED for actual HDF5 structure
Compares PIC simulations with and without Dynamic Particle Merging & Splitting Algorithm

Created: 2026-02-10
Author: Manef
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from pathlib import Path
import h5py

from pic.Hdf5Viewer import Hdf5Viewer
from plasmapy import particles
import astropy.units as u

# Define particles - use the exact names from your HDF5 files
ele = particles.electron
ion = particles.Particle("He+")
neutral = particles.Particle("He")


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
        try:
            self.sim_no_ms = Hdf5Viewer(sim_without_ms_path)
            print("✓ Successfully loaded simulation without M&S")
        except Exception as e:
            print(f"✗ Error loading simulation without M&S: {e}")
            raise
        
        print(f"\nLoading simulation with M&S from: {sim_with_ms_path}")
        try:
            self.sim_with_ms = Hdf5Viewer(sim_with_ms_path)
            print("✓ Successfully loaded simulation with M&S")
        except Exception as e:
            print(f"✗ Error loading simulation with M&S: {e}")
            raise
        
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
        
        # Check available diagnostics
        print(f"\nAvailable diagnostics:")
        print(f"  Particle species: {list(self.sim_no_ms.particle_data.keys())}")
        if self.sim_no_ms.particle_data:
            for species in self.sim_no_ms.particle_data.keys():
                print(f"  {species.symbol}: {list(self.sim_no_ms.particle_data[species].keys())}")
        print(f"  Files loaded: {self.sim_no_ms.n_files}/{len(self.t)-1}")
    
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
        
        i = time_index if time_index >= 0 else min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        i_start = max(0, i - average // 2)
        i_end = min(i_start + average, i + 1)
        
        if species in ['electron', 'both'] and 'n' in self.sim_no_ms.particle_data[ele]:
            ne_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
            )
            ne_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
            )
            
            ax.plot(self.x, ne_no_ms, 'b-', linewidth=2, label='Electron (No M&S)')
            ax.plot(self.x, ne_with_ms, 'b--', linewidth=2, label='Electron (With M&S)')
        
        if species in ['ion', 'both'] and 'n' in self.sim_no_ms.particle_data[ion]:
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
    
    def compare_velocity(self, time_index=-1, average=5, ax=None, component=0):
        """
        Compare electron and ion velocity profiles
        
        Parameters:
        -----------
        component : int
            Velocity component (0=x, 1=y, 2=z)
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        i_start = max(0, i - average // 2)
        i_end = min(i_start + average, i + 1)
        
        if 'V' in self.sim_no_ms.particle_data[ele]:
            Ve_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["V"][i_start:i_end, :, component], axis=0
            )
            Ve_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["V"][i_start:i_end, :, component], axis=0
            )
            
            ax.plot(self.x, Ve_no_ms, 'b-', linewidth=2, label='Electron (No M&S)')
            ax.plot(self.x, Ve_with_ms, 'b--', linewidth=2, label='Electron (With M&S)')
        
        if 'V' in self.sim_no_ms.particle_data[ion]:
            Vi_no_ms = np.mean(
                self.sim_no_ms.particle_data[ion]["V"][i_start:i_end, :, component], axis=0
            )
            Vi_with_ms = np.mean(
                self.sim_with_ms.particle_data[ion]["V"][i_start:i_end, :, component], axis=0
            )
            
            ax.plot(self.x, Vi_no_ms, 'r-', linewidth=2, label='Ion (No M&S)')
            ax.plot(self.x, Vi_with_ms, 'r--', linewidth=2, label='Ion (With M&S)')
        
        comp_labels = ['x', 'y', 'z']
        ax.set_xlabel('Position [m]', fontsize=12)
        ax.set_ylabel(f'Velocity {comp_labels[component]} [m/s]', fontsize=12)
        ax.set_title(f'Velocity Comparison at t = {self.t[i]:.2e}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_power_deposition(self, time_index=-1, average=5, ax=None):
        """
        Compare power deposition profiles
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        i = time_index if time_index >= 0 else min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        i_start = max(0, i - average // 2)
        i_end = min(i_start + average, i + 1)
        
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
                label='Electron (No M&S)'
            )
            ax.plot(
                self.x, 
                power_with_ms.to(u.kW / u.m**3), 
                'b--', 
                linewidth=2, 
                label='Electron (With M&S)'
            )
        
        if 'Power' in self.sim_no_ms.particle_data[ion]:
            power_no_ms_ion = np.mean(
                self.sim_no_ms.particle_data[ion]["Power"][i_start:i_end], axis=0
            )
            power_with_ms_ion = np.mean(
                self.sim_with_ms.particle_data[ion]["Power"][i_start:i_end], axis=0
            )
            
            ax.plot(
                self.x, 
                power_no_ms_ion.to(u.kW / u.m**3), 
                'r-', 
                linewidth=2, 
                label='Ion (No M&S)'
            )
            ax.plot(
                self.x, 
                power_with_ms_ion.to(u.kW / u.m**3), 
                'r--', 
                linewidth=2, 
                label='Ion (With M&S)'
            )
        
        ax.set_xlabel('Position [m]', fontsize=12)
        ax.set_ylabel('Power Deposition [kW/m³]', fontsize=12)
        ax.set_title(f'Power Deposition at t = {self.t[i]:.2e}', fontsize=14)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_particle_evolution(self, ax=None):
        """
        Compare total particle count evolution over time
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Calculate total particles (integral of density * volume element)
        dx = self.sim_no_ms.dx
        
        n_electrons_no_ms = np.sum(
            self.sim_no_ms.particle_data[ele]["n"], axis=1
        ) * dx
        n_electrons_with_ms = np.sum(
            self.sim_with_ms.particle_data[ele]["n"], axis=1
        ) * dx
        
        n_ions_no_ms = np.sum(
            self.sim_no_ms.particle_data[ion]["n"], axis=1
        ) * dx
        n_ions_with_ms = np.sum(
            self.sim_with_ms.particle_data[ion]["n"], axis=1
        ) * dx
        
        n_files = min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        
        ax.plot(
            self.t[:n_files], 
            n_electrons_no_ms[:n_files], 
            'b-', 
            linewidth=2, 
            label='Electrons (No M&S)'
        )
        ax.plot(
            self.t[:n_files], 
            n_electrons_with_ms[:n_files], 
            'b--', 
            linewidth=2, 
            label='Electrons (With M&S)'
        )
        
        ax.plot(
            self.t[:n_files], 
            n_ions_no_ms[:n_files], 
            'r-', 
            linewidth=2, 
            label='Ions (No M&S)'
        )
        ax.plot(
            self.t[:n_files], 
            n_ions_with_ms[:n_files], 
            'r--', 
            linewidth=2, 
            label='Ions (With M&S)'
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
        i = time_index if time_index >= 0 else min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        i_start = max(0, i - average // 2)
        i_end = min(i_start + average, i + 1)
        
        errors = {}
        
        # Electron density error
        ne_no_ms = np.mean(
            self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        ne_with_ms = np.mean(
            self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        )
        errors['electron_density'] = {
            'max_relative': float(np.max(np.abs((ne_with_ms - ne_no_ms) / ne_no_ms))),
            'mean_relative': float(np.mean(np.abs((ne_with_ms - ne_no_ms) / ne_no_ms))),
            'rms': float(np.sqrt(np.mean(((ne_with_ms - ne_no_ms) / ne_no_ms)**2)))
        }
        
        # Ion density error
        ni_no_ms = np.mean(
            self.sim_no_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        ni_with_ms = np.mean(
            self.sim_with_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        )
        errors['ion_density'] = {
            'max_relative': float(np.max(np.abs((ni_with_ms - ni_no_ms) / ni_no_ms))),
            'mean_relative': float(np.mean(np.abs((ni_with_ms - ni_no_ms) / ni_no_ms))),
            'rms': float(np.sqrt(np.mean(((ni_with_ms - ni_no_ms) / ni_no_ms)**2)))
        }
        
        # Power deposition errors (if available)
        if 'Power' in self.sim_no_ms.particle_data[ele]:
            Pe_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            ).value
            Pe_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            ).value
            # Avoid division by very small numbers
            mask = np.abs(Pe_no_ms) > 1e-10
            if np.any(mask):
                errors['electron_power'] = {
                    'max_relative': float(np.max(np.abs((Pe_with_ms[mask] - Pe_no_ms[mask]) / Pe_no_ms[mask]))),
                    'mean_relative': float(np.mean(np.abs((Pe_with_ms[mask] - Pe_no_ms[mask]) / Pe_no_ms[mask]))),
                    'rms': float(np.sqrt(np.mean(((Pe_with_ms[mask] - Pe_no_ms[mask]) / Pe_no_ms[mask])**2)))
                }
        
        return errors
    
    def plot_relative_error_profiles(self, time_index=-1, average=5):
        """
        Plot spatial profiles of relative errors
        """
        i = time_index if time_index >= 0 else min(self.sim_no_ms.n_files, self.sim_with_ms.n_files)
        i_start = max(0, i - average // 2)
        i_end = min(i_start + average, i + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Electron density error
        ne_no_ms = np.mean(
            self.sim_no_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        ).value
        ne_with_ms = np.mean(
            self.sim_with_ms.particle_data[ele]["n"][i_start:i_end], axis=0
        ).value
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
        ).value
        ni_with_ms = np.mean(
            self.sim_with_ms.particle_data[ion]["n"][i_start:i_end], axis=0
        ).value
        rel_error_ni = (ni_with_ms - ni_no_ms) / ni_no_ms * 100
        
        axes[0, 1].plot(self.x, rel_error_ni, 'r-', linewidth=2)
        axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Position [m]')
        axes[0, 1].set_ylabel('Relative Error [%]')
        axes[0, 1].set_title('Ion Density Error')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Velocity errors (if available)
        if 'V' in self.sim_no_ms.particle_data[ele]:
            Ve_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["V"][i_start:i_end, :, 0], axis=0
            ).value
            Ve_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["V"][i_start:i_end, :, 0], axis=0
            ).value
            mask = np.abs(Ve_no_ms) > 1e-6  # Avoid division by near-zero
            rel_error_Ve = np.zeros_like(Ve_no_ms)
            rel_error_Ve[mask] = (Ve_with_ms[mask] - Ve_no_ms[mask]) / Ve_no_ms[mask] * 100
            
            axes[1, 0].plot(self.x, rel_error_Ve, 'b-', linewidth=2)
            axes[1, 0].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 0].set_xlabel('Position [m]')
            axes[1, 0].set_ylabel('Relative Error [%]')
            axes[1, 0].set_title('Electron Velocity (x) Error')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'Power' in self.sim_no_ms.particle_data[ele]:
            Pe_no_ms = np.mean(
                self.sim_no_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            ).value
            Pe_with_ms = np.mean(
                self.sim_with_ms.particle_data[ele]["Power"][i_start:i_end], axis=0
            ).value
            mask = np.abs(Pe_no_ms) > 1e-6
            rel_error_Pe = np.zeros_like(Pe_no_ms)
            rel_error_Pe[mask] = (Pe_with_ms[mask] - Pe_no_ms[mask]) / Pe_no_ms[mask] * 100
            
            axes[1, 1].plot(self.x, rel_error_Pe, 'g-', linewidth=2)
            axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Position [m]')
            axes[1, 1].set_ylabel('Relative Error [%]')
            axes[1, 1].set_title('Electron Power Deposition Error')
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
        
        # 1. Main comparison figure (5 subplots)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        self.compare_densities(time_index=time_index, ax=ax1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        self.compare_velocity(time_index=time_index, ax=ax2, component=0)
        
        ax3 = fig.add_subplot(gs[1, 0])
        self.compare_power_deposition(time_index=time_index, ax=ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self.compare_particle_evolution(ax=ax4)
        
        # Add text summary
        ax5 = fig.add_subplot(gs[2, :])
        errors = self.compute_relative_errors(time_index=time_index)
        
        summary_text = "NUMERICAL ERROR SUMMARY\n\n"
        for quantity, error_dict in errors.items():
            summary_text += f"{quantity.upper().replace('_', ' ')}:\n"
            summary_text += f"  Max: {error_dict['max_relative']*100:.2f}%  "
            summary_text += f"Mean: {error_dict['mean_relative']*100:.2f}%  "
            summary_text += f"RMS: {error_dict['rms']*100:.2f}%\n\n"
        
        ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, 
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax5.axis('off')
        
        plt.suptitle('DPMSA Benchmark: Complete Comparison', fontsize=16, y=0.995)
        plt.savefig(os.path.join(output_dir, 'full_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/full_comparison.png")
        
        # 2. Relative error profiles
        fig_errors = self.plot_relative_error_profiles(time_index=time_index)
        plt.savefig(os.path.join(output_dir, 'relative_errors.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/relative_errors.png")
        
        # 3. Print numerical errors
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
            **{k: v for k, v in errors.items()}
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Benchmark PIC simulations with and without DPMSA'
    )
    parser.add_argument(
        'path_no_ms',
        type=str,
        help='Path to simulation data WITHOUT merging & splitting'
    )
    parser.add_argument(
        'path_with_ms',
        type=str,
        help='Path to simulation data WITH merging & splitting'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='benchmark_results',
        help='Output directory for benchmark results (default: benchmark_results)'
    )
    parser.add_argument(
        '--time-index',
        '-t',
        type=int,
        default=-1,
        help='Time index to compare (-1 for final time, default: -1)'
    )
    
    args = parser.parse_args()
    
    # Check if paths exist
    if not os.path.exists(args.path_no_ms):
        print(f"ERROR: Path not found: {args.path_no_ms}")
        exit(1)
    if not os.path.exists(args.path_with_ms):
        print(f"ERROR: Path not found: {args.path_with_ms}")
        exit(1)
    
    # Run benchmark
    print(f"\nStarting DPMSA Benchmark...")
    print(f"  Without M&S: {args.path_no_ms}")
    print(f"  With M&S:    {args.path_with_ms}")
    print(f"  Output:      {args.output}\n")
    
    benchmark, errors = run_benchmark(
        args.path_no_ms, 
        args.path_with_ms, 
        output_dir=args.output
    )
