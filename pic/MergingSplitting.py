import numpy as np
import os
from os import path
import argparse

from pic.plasma import Plasma
from pic.particles import ParticlesGroup

from pic.parsing import get_parameters

from datetime import datetime
from pathlib import Path
import signal
import sys


class Cluster:
    def __init__(self, center_index : ParticlesGroup, members : np.ndarray):
        self.center = center
        self.particle_indices = []

    def add_particle(self, particle_index: int):
        self.particle_indices.append(particle_index)















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


    def distance(self, x1, x2, v1, v2):
        # 2D distance in phase space
        return np.sqrt( (x1 - x2)^2 + (v1 - v2)^2 )

    def execute(self, nt: int):
        # checking if time to merge 

        #looping on every class of macro particules
        species = self.plasma.species.values() 

        for specie in species:
            Npart = specie.Npart 
            
            # 1/ initialisating the mass center
            K0 = np.sqrt(Npart/2)
            K = K0
            centers = [np.random.randint(0, Npart-1)] #choix arbitraire du premier centre
            # We calculate the shortest distance D
            while len(centers) < K0:
                D = np.full((len(centers), Npart), np.inf)
                for k in range(len(centers)):
                    x_center = specie.x[centers[k]]
                    v_center = specie.V[centers[k]]
                    for i in range(Npart):
                        dist = self.distance(x_center, specie.x[i],v_center, specie.V[i])
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
                    dist = self.distance(specie.x[center], specie.x[i], specie.V[center], specie.V[i])
                    distances.append(dist)
                nearest_center = centers[np.argmin(distances)]
                clusters[nearest_center].append(i)

            #2.2 Ensuring minimum cluster size
            for center, indices in clusters.items():
                if len(indices) < self.clusterSizeMin:
                    K = K - 1
                    # Find the nearest cluster to merge with (entire cluster)
                    
                    nearest_center = None
                    min_dist = np.inf
                    
                    for other_center in centers:
                        if other_center != center:
                            dist = distances = self.distance(specie.x[other_center], specie.x[center], specie.V[other_center], specie.V[center])
                            if dist < min_dist:
                                min_dist = dist
                                nearest_center = other_center
                    # Merge clusters
                    clusters[nearest_center].extend(clusters[center])
                    del(clusters[center])

            #2.3 Recalculate centers after merging. Multiple methods possible
            for center in list(clusters.keys()):
                indices = clusters[center]
                W_total = np.sum(specie.w[indices])
                probabilities = specie.w[indices] / W_total
                new_center = np.random.choice(indices, p=probabilities)
                if new_center != center:
                    clusters[new_center] = clusters.pop(center)

            # 5/ generating the new macro-particles
            for center, indices in clusters.items(): 

                if len(indices) > 1:
                    total_weight = np.sum(specie.weights[indices])
                    new_position = np.sum(specie.x[indices] * specie.weights[indices]) / total_weight
                    new_velocity = np.sum(specie.V[indices] * specie.weights[indices][:, np.newaxis], axis=0) / total_weight
                    # Update the center particle
                    specie.velocities[center] = new_velocity
                    specie.weights[center] = total_weight
                    # Remove merged particles
                    mask = np.ones(Npart, dtype=bool)
                    mask[indices] = False
                    specie.x = specie.x[mask]
                    specie.velocities = specie.velocities[mask]
                    specie.weights = specie.weights[mask]
                    specie.Npart = len(specie.weights)
            
    def merging(self, centers: np.ndarray, clusters: dict, dmean: float):
        # Merging clusters that are closer than dmean
        done_merging = False
        while not done_merging :
            done_merging = True
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    center_i = centers[i]
                    center_j = centers[j]
                    dist = self.distance(self.plasma.species.x[center_i], self.plasma.species.x[center_j],
                                            self.plasma.species.V[center_i], self.plasma.species.V[center_j])
                    if dist < dmean:
                        # merge clusters i and j
                        clusters[center_i].extend(clusters[center_j])
                        del clusters[j]
                        centers = np.delete(centers, j)
                        done_merging = False
                        break  # restart since centers have changed
                if not done_merging:
                    break # restart since centers have changed
        return centers, clusters
        
    def splitting(self, centers: np.ndarray, clusters: dict, Nmin:int):

        #computing momentum deviation for each cluster
        deviation = {center : 0.0 for center in centers}
        for center in centers:
            Wi = self.plasma.species[center].w
            Pi = self.plasma.species[center].V * Wi
            W = np.sum(Wi)
            P = np.sum(Wi*Pi, axis=0)/W
            deviation[center] = np.sqrt( np.sum( Wi*(Pi - P)**2 ) / W)


        #TODO weighted average for alpha
        alpha = np.mean( list(momentum_dev.values()) )

        for center in centers:
            if momentum_dev[center] > 2*alpha and len(clusters[center]) > 2*Nmin:
                # split the cluster into two
                indices = clusters[center]
                mid = len(indices) // 2
                new_center = indices[mid]
                clusters[new_center] = clusters[center][mid:]
                clusters[center] = clusters[center][:mid]
                centers = np.append(centers, new_center)


    def momentum_deviation(self, centers: np.ndarray, clusters: dict) :
        deviation = {center : 0.0 for center in centers}
        for center in centers:
            Wi = self.plasma.species[center].w
            Pi = self.plasma.species[center].V * Wi
            W = np.sum(Wi)
            P = np.sum(Wi*Pi, axis=0)/W
            deviation[center] = np.sqrt( np.sum( Wi*(Pi - P)**2 ) / W)
        return deviation