import numpy as np
import os
from os import path
import argparse

import pickle
from pic.plasma import Plasma

from pic.parsing import get_parameters

from datetime import datetime
from pathlib import Path
import signal
import sys


class MergingSplitting:
    
    def __init__(self, plasma: Plasma, merging_algorithm: str, params: dict):
        match merging_algorithm:
            case "classic":
                self.merging_splitting = MergingSplittingClassic(plasma, params)
            case "dpmsa":
                self.merging_splitting = DPMSA(plasma, params)
            case _:
                pass
        
    def execute(self):
        pass

    
class MergingSplittingClassic(MergingSplitting):
    def __init__(self, plasma: Plasma, params: dict):
        params = params
        self.plasma = plasma
        self.params = params

    def execute(self, nt: int):
        # Implement the classic merging and splitting algorithm here
        pass


class DPMSA:
    def __init__(self, plasma: Plasma, params: dict):
        params = params
        self.plasma = plasma
        self.params = params
        self.clusterSizeMin = params.get("cluster_size_min", 2)

    def execute(self, nt: int):
        # checking if time to merge 

        #looping on every class of macro particules
        species = self.plasma.species.values() 

        for specie in species:
            Npart = specie.Npart 
            
            # 1/ initialisating the mass center
            K0 = np.sqrt(Npart/2)
            centers = [np.random.randint(0, Npart-1)] #choix arbitraire du premier centre
            # We calculate the shortest distance D
            while len(centers) < K0:
                D = np.full((len(centers), Npart), np.inf)
                for k in range(len(centers)):
                    for i in range(Npart):
                        dist = np.abs(specie.x[i] - specie.x[centers[k]])
                        D[k, i] = dist
                D_min = np.min(D, axis=0)
                next_center = np.argmax(D_min)
                centers.append(next_center)
            centers = np.array(centers)

            # 2.1 Assigning particles to the nearest center  
            clusters = {center: [] for center in centers}
            for i in range(Npart):
                distances = []
                for center in centers:
                    dist = np.abs(specie.x[i] - specie.x[center])
                    distances.append(dist)
                nearest_center = centers[np.argmin(distances)]
                clusters[nearest_center].append(i)

            #2.2 Ensuring minimum cluster size
            for center, indices in clusters.items():
                if len(indices) < self.clusterSizeMin:
                    K0 = K0 - 1
                    # Find the nearest cluster to merge with (entire cluster)
                    
                    nearest_center = None
                    min_dist = np.inf
                    
                    for other_center in centers:
                        if other_center != center:
                            dist = np.abs(specie.x[center] - specie.x[other_center])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_center = other_center
                    # Merge clusters
                    clusters[nearest_center].extend(clusters[center])
                    clusters[center] = []


            # 3/ Merging particles in each cluster
            for center, indices in clusters.items(): 

                if len(indices) > 1:
                    total_weight = np.sum(specie.weights[indices])
                    new_position = np.sum(specie.x[indices] * specie.weights[indices]) / total_weight
                    new_velocity = np.sum(specie.V[indices] * specie.weights[indices][:, np.newaxis], axis=0) / total_weight
                    # Update the center particle
                    specie.x[center] = new_position % self.plasma.domain.length  # Periodic boundary conditions
                    specie.velocities[center] = new_velocity
                    specie.weights[center] = total_weight
                    # Remove merged particles
                    mask = np.ones(Npart, dtype=bool)
                    mask[indices] = False
                    specie.x = specie.x[mask]
                    specie.velocities = specie.velocities[mask]
                    specie.weights = specie.weights[mask]
                    specie.Npart = len(specie.weights)
            

            


        pass