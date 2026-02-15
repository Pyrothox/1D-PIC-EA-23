#!/usr/bin/env python3
"""
Check what diagnostics are actually saved in HDF5 files
"""

import h5py
import os
import sys

def check_hdf5_contents(data_folder):
    """
    Check what's actually saved in the HDF5 files
    """
    print(f"\nChecking data folder: {data_folder}")
    print("="*60)
    
    # Find first HDF5 file
    h5_files = [f for f in os.listdir(data_folder) if f.endswith('.h5') and f != 'restart.h5']
    
    if not h5_files:
        print("ERROR: No HDF5 files found!")
        return
    
    # Sort numerically
    h5_files = sorted(h5_files, key=lambda x: int(x.replace('.h5', '')))
    first_file = os.path.join(data_folder, h5_files[0])
    
    print(f"\nExamining file: {first_file}")
    print("-"*60)
    
    with h5py.File(first_file, 'r') as f:
        print("\nTop-level groups:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Check electron data
        if 'electron' in f:
            print("\nElectron group contents:")
            for key in f['electron'].keys():
                shape = f[f'electron/{key}'].shape
                dtype = f[f'electron/{key}'].dtype
                print(f"  - {key}: shape={shape}, dtype={dtype}")
        
        # Check He+ data  
        if 'He+' in f:
            print("\nHe+ group contents:")
            for key in f['He+'].keys():
                shape = f[f'He+/{key}'].shape
                dtype = f[f'He+/{key}'].dtype
                print(f"  - {key}: shape={shape}, dtype={dtype}")
        
        # Check for collision data
        if 'electron' in f:
            if 'collisions_rates' in f['electron']:
                print("\nCollision rates structure:")
                for key in f['electron/collisions_rates'].keys():
                    print(f"  - {key}")
                    for subkey in f[f'electron/collisions_rates/{key}'].keys():
                        shape = f[f'electron/collisions_rates/{key}/{subkey}'].shape
                        print(f"    - {subkey}: shape={shape}")
    
    print("\n" + "="*60)
    print(f"Total HDF5 files in folder: {len(h5_files)}")
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = input("Enter path to data folder: ")
    
    if os.path.exists(folder):
        check_hdf5_contents(folder)
    else:
        print(f"ERROR: Folder not found: {folder}")
