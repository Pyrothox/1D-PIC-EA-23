from __future__ import annotations

from typing import Tuple
import numpy as np
from numba import njit

import random


from scipy.constants import e

numbaparallel = False
fastmath = True

Complete_Distribution = False
Partial_Distribution = True


@njit("f8(f8,f8)")
def generate_maxw(T, m):
    """Generate a scalar following one Maxwellian distribution function"""
    v_Te = np.sqrt(e * T / m)
    W = 2
    while W >= 1 or W <= 0:
        R1 = random.random() * 2 - 1
        R2 = random.random() * 2 - 1

        W = R1**2 + R2**2
    W = np.sqrt((-2 * np.log(W)) / W)

    v = v_Te * R1 * W
    return v


def velocity_maxw_flux(T, m):
    """Generate one zcalar folowing a maxwellian flux distribution function"""
    import random

    v_Te = np.sqrt(e * T / m)

    R = random.random()
    v = v_Te * np.sqrt(-np.log(R))
    return v


def max_vect(N, T, m):
    """Generate an array of scalar folowing a Maxwellian DF"""
    return np.array([generate_maxw(T, m) for i in np.arange(N)])


@njit("f8[:,:](i8,f8,f8)")
def max_vect3D(N, T, m):
    """
    Computes an array of maxwellian speeds.

    Parameters
    ----------
    N : integer
        Number of vectors to be generated.
    T : float
        Temperature of the generated particles.
    m : float
        Mass of the particles.

    Returns
    -------
    v : np.array(N,3)
        Array of N maxellian speeds.

    """
    v = np.zeros((3, N), dtype=np.float64).T
    for i in range(N):
        for j in range(3):
            v[i, j] = generate_maxw(T, m)
    return v


@njit("f8[:,:](i8,f8[:],f8)")
def max_vect3D_multiple_T(N, T, m):
    """
    Computes an array of maxwellian speeds.

    Parameters
    ----------
    N : integer
        Number of vectors to be generated.
    T : np.array(N,3)
        Local temperature of the generated particles.
    m : float
        Mass of the particles.

    Returns
    -------
    v : np.array(N,3)
        Array of N maxellian speeds.

    """
    v = np.zeros((3, N), dtype=np.float64).T
    for i in range(N):
        for j in range(3):
            v[i, j] = generate_maxw(T[i], m)
    return v


@njit("(f8[:,:], i8[:], i8, f8, f8)")
def fill_max_vect(V, Idxs, N, T, m):
    for i in range(N):
        for j in range(3):
            V[Idxs[i], j] = generate_maxw(T, m)


@njit("(f8[:,:], i8[:], i8, f8[:], f8)")
def fill_max_vect_variable_T(V, Idxs, N, T, m):
    for i in range(N):
        for j in range(3):
            V[Idxs[i], j] = generate_maxw(T[i], m)


def fux_vect(N, T, m):
    """Generate an array of scalar folowing a Maxwellian flux DF"""

    from numpy.random import rand

    v_Te = np.sqrt(e * T / m)

    return v_Te * np.sqrt(-np.log(rand(N)))


@njit("f8[:](f8[:])")
def numba_ones_like(x):
    return np.ones_like(x)


@njit("f8[:](f8[:],i8)", fastmath=fastmath)
def numba_power(x, p):
    return x**p


@njit(
    [
        "f8[:](i8,f8[:],f8[:],f8[:],f8[:],f8)",
        "f8[:,:](i8,f8[:],f8[:,:],f8[:],f8[:,:],f8)",
        "f8[:,:,:](i8,f8[:],f8[:,:,:],f8[:],f8[:,:,:],f8)",
    ],
    fastmath=fastmath,
    parallel=numbaparallel,
)
def particle_to_grid(Np, partx, info, tabx, diag, dx):
    Jmax = len(tabx) - 1
    for i in range(Np):
        j = int(partx[i] / dx)
        if j > Jmax:
            j = Jmax
        deltax = abs(tabx[j] - partx[i]) / dx
        if j < 1:
            diag[j] += 2 * (1 - deltax) * info[i]
            diag[1] += deltax * info[i]

        elif j < Jmax - 1:
            diag[j] += (1 - deltax) * info[i]
            diag[j + 1] += deltax * info[i]
        elif j < Jmax:
            diag[j] += (1 - deltax) * info[i]
            diag[j + 1] += 2 * deltax * info[i]
        elif j == Jmax:
            diag[j] += 2 * info[i]

    return diag


@njit(
    fastmath=fastmath,
    parallel=numbaparallel,
)
def particle_to_grid_order0(Np, partx, tabx, diag, dx):
    Jmax = len(tabx) - 1
    for i in range(Np):
        j = int(partx[i] / dx)
        if j > Jmax:
            j = Jmax
        deltax = abs(tabx[j] - partx[i]) / dx
        if j < 1:
            diag[j] += 2 * (1 - deltax)
            diag[1] += deltax

        elif j < Jmax - 1:
            diag[j] += 1 - deltax
            diag[j + 1] += deltax
        elif j < Jmax:
            diag[j] += 1 - deltax
            diag[j + 1] += 2 * deltax
        elif j == Jmax:
            diag[j] += 2

    return diag

def get_particle_indexes_in_cells(Np, partx, tabx, cells_indexes, dx):
    """Spatially bin particles into cells and return their indexes.
    Groups all particles into their respective spatial cells based on position.
    Useful for cell-by-cell operations like local clustering, merging, or diagnostics.
    Particles outside domain bounds are clamped to nearest boundary cell.

    Parameters
    ----------
    Np : int
        Total number of particles
    partx : array, shape (Npart,)
        Particle x-positions [m]
    tabx : array, shape (Ncells,)
        Grid cell positions/boundaries
    cells_indexes : array, shape (Ncells, ?)
        Output array to store particle indexes per cell
    dx : float
        Grid spacing [m]
    
    Returns
    -------
    list of lists
        cells_indexes[j] contains sorted list of all particle indexes in cell j.
        Empty lists for cells with no particles.
    """

    Ncells = int(1.0 / dx)
    cells_indexes = [[] for _ in range(Ncells)]
    
    for i in range(Np):
        j = int(partx[i] / dx)
        # Clamp to valid cell range
        if j >= Ncells:
            j = Ncells - 1
        elif j < 0:
            j = 0
        cells_indexes[j].append(i)
    
    return cells_indexes


def numba_return_part_diag_weighted(Np, partx, partv, weight, tabx, diag, dx, power):
    """general function for the particle to grid diagnostics (weighted by per-particle weight).
    Returns raw sums (not normalized)."""

    if np.sum(weight) == 0:
        return diag

    if power == 0:
        return particle_to_grid(Np, partx, weight, tabx, diag, dx)
    elif power == 1:
        weighted_info = partv * weight
        return particle_to_grid(Np, partx, weighted_info, tabx, diag, dx)
    elif power > 0:
        weighted_info = numba_power(partv, power) * weight
        return particle_to_grid(Np, partx, weighted_info, tabx, diag, dx)

    else:
        print("Unknow dignostics !!")
        return diag

def numba_return_part_diag(Np, partx, partv, tabx, diag, dx, power):
    """general function for the particle to grid diagnostics"""

    if power == 0:
        return particle_to_grid_order0(Np, partx, tabx, diag, dx)
    elif power == 1:
        return particle_to_grid(Np, partx, partv, tabx, diag, dx)
    elif power > 0:
        info = numba_power(partv, power)
        return particle_to_grid(Np, partx, info, tabx, diag, dx)

    else:
        print("Unknow dignostics !!")
        return diag


@njit("i8[:](f8[:],f8)", fastmath=fastmath)
def normDx(tabx, dx):
    """devise a vector and use the `int` function to get integers"""
    tmp = tabx / dx
    normedx = np.zeros(tabx.size, dtype=np.int64)
    for i in range(tmp.size):
        normedx[i] = int(tmp[i])
    return normedx


@njit(parallel=False, fastmath=fastmath)
def numba_interp1D_normed(partx, tabE):
    """Compute the linear interpolation of the electric field in
     the X directions but with normed position
    This numba function should be faster than the
     scipy.interp1d and numpy.interp

    """
    partE = np.zeros_like(partx)
    for i in range(len(partx)):
        x = partx[i]
        j = int(x)  # position of the particle, in intex of tabx

        deltax = abs(x - j)  # length to cell center

        partE[i] = (1 - deltax) * tabE[j] + deltax * tabE[j + 1]

    return partE


@njit(parallel=False, fastmath=fastmath)
def numba_interp1D_normed_buff(partx, tabE, partE):
    """Compute the linear interpolation of the electric field in
     the X directions but with normed position
    This numba function should be faster than the
     scipy.interp1d and numpy.interp

    """
    for i in range(len(partx)):
        x = partx[i]
        j = int(x)  # position of the particle, in intex of tabx

        deltax = abs(x - j)  # length to cell center

        partE[i] = (1 - deltax) * tabE[j] + deltax * tabE[j + 1]

    return partE


@njit(
    "f8[:](f8[:],f8[:],f8[:],f8[:],i8)",
    fastmath=fastmath,
    parallel=numbaparallel,
)
def numba_thomas_solver(di, ai, bi, ciprim, Nx):
    """Solve thomas with the upward and download loops"""
    diprim = di
    diprim[0] /= bi[0]

    for i in np.arange(1, len(diprim)):
        diprim[i] -= ai[i] * diprim[i - 1]
        diprim[i] /= bi[i] - ai[i] * ciprim[i - 1]

    ## Init solution ##
    phi = np.zeros(Nx)
    phi[-1] = diprim[-1]

    # limit conditions
    for i in np.arange(Nx - 2, -1, -1):
        phi[i] = diprim[i] - ciprim[i] * phi[i + 1]

    return phi


@njit
def popout_weighted(x, V, w, val, absorbtion):
    """move elements that do not correspond to the condition
    x > val to the end of the table.

    Inputs :
    =========
    x, v (In and out) : array of float64

    val: float64 the threshold

    return:
    =======
    left_count, right_count, left_weight, right_weight

    """
    ## Init the parameters ##
    compt = 0
    left = 0
    right = 0
    left_w = 0.0
    right_w = 0.0
    N = len(x)
    zeros = np.zeros(3)
    ## Linear search from the end of the table ##
    pos = x[-1]
    if absorbtion:
        if (pos >= val) or (pos <= 0):  # check for the last particle
            # capture its weight before zeroing
            w_last = w[-1]
            x[-1] = -1
            V[-1] = zeros
            w[-1] = 0.0

            compt += 1
            if pos >= val:
                right += 1
                right_w += w_last
            else:
                left += 1
                left_w += w_last

        for i in np.arange(N - 2, -1, -1):
            pos = x[i]
            if (pos >= val) or (pos <= 0):  # Condition to move the element at the end
                # exchange the current element with the last
                tmp = x[-compt - 1]
                x[-compt - 1] = -1
                x[i] = tmp

                V[i, :] = V[-compt - 1, :]
                V[-compt - 1, :] = zeros
                tmp_w = w[-compt - 1]
                w[-compt - 1] = 0.0
                w[i] = tmp_w

                compt += 1
                if pos >= val:
                    right += 1
                    right_w += tmp_w
                else:
                    left += 1
                    left_w += tmp_w

    return left, right, left_w, right_w

@njit
def popout(x, V, val, absorbtion):
    """move elements that do not correspond to the condition
    x > val to the end of the table.

    Inputs :
    =========
    x, v (In and out) : array of float64

    val: float64 the threshold

    return:
    =======
    compt: int64 number of elements put at the end of x

    """
    ## Init the parameters ##
    compt = 0
    left = 0
    right = 0
    N = len(x)
    zeros = np.zeros(3)
    ## Linear search from the end of the table ##
    pos = x[-1]
    if absorbtion:
        if (pos >= val) or (pos <= 0):  # check for the last particle
            x[-1] = -1
            V[-1] = zeros
            
            compt += 1
            if pos >= val:
                right += 1
            else:
                left += 1

        for i in np.arange(N - 2, -1, -1):
            pos = x[i]
            if (pos >= val) or (pos <= 0):  # Condition to move the element at the end
                # exchange the current element with the last
                tmp = x[-compt - 1]
                x[-compt - 1] = -1
                x[i] = tmp

                V[i, :] = V[-compt - 1, :]
                V[-compt - 1, :] = zeros
                compt += 1
                if pos >= val:
                    right += 1
                else:
                    left += 1

    else:
        for i in range(len(x)):  # in case of reflection
            pos = x[i]
            if pos >= val:
                x[i] = 2 * val - pos
                V[i] = -V[i]

            if pos <= 0:
                x[i] = -x[i]
                V[i] = -V[i]

    return (left, right)


def popout_np(x, V, val, absorbtion):
    """move elements that do not correspond to the condition
    x > val to the end of the table.

    Inputs :
    =========
    x, v (In and out) : array of float64

    val: float64 the threshold

    return:
    =======
    compt: int64 number of elements put at the end of x

    """
    mask_0 = x <= 0
    mask_val = x >= val

    if absorbtion:
        mask_out = mask_0 + mask_val
        N_out = np.sum(mask_out)
        if not N_out:
            return 0
        mask_in = ~mask_out

        x[:-N_out] = x[mask_in]
        V[:-N_out] = V[mask_in]
        x[-N_out:] = -1.0
        V[-N_out:, ...] = 0.0
    else:
        x[mask_0] = -x[mask_0]
        V[mask_0] *= -1

        x[mask_val] = 2 * val - x[mask_val]
        V[mask_val] *= -1

        N_out = np.sum(mask_0 + mask_val)

    return N_out


def remove(Idxs, x, V, Npart):
    Idxs.sort()
    Idxs.reverse()
    return __remove_jit(Idxs, x, V, Npart)

def remove_weighted(Idxs, x, V, w, Npart):
    Idxs.sort()
    Idxs.reverse()
    return __remove_jit_weighted(Idxs, x, V, w, Npart)


def remove_array_weighted(Idxs, x, V, w, Npart):
    Idxs.sort()
    Idxs = Idxs[::-1]
    return __remove_jit_weighted(Idxs, x, V, w, Npart)

def remove_array(Idxs, x, V, Npart):
    Idxs.sort()
    Idxs = Idxs[::-1]
    return __remove_jit(Idxs, x, V, Npart)


@njit()
def __remove_jit(Idxs, x, V, Npart):
    for i in Idxs:
        x[i] = x[Npart - 1]
        V[i, :] = V[Npart - 1, :]
        x[Npart - 1] = -1.0
        V[Npart - 1, :] = 0.0
        Npart -= 1
    return Npart


@njit()
def __remove_jit_weighted(Idxs, x, V, w, Npart):
    for i in Idxs:
        x[i] = x[Npart - 1]
        V[i, :] = V[Npart - 1, :]
        w[i] = w[Npart - 1]
        x[Npart - 1] = -1.0
        V[Npart - 1, :] = 0.0
        w[Npart - 1] = 0.0
        Npart -= 1
    return Npart

@njit()
def fixed_angle_isotropic_scatter(
    v: np.ndarray, r1: np.ndarray, cos_khi: np.ndarray
) -> np.ndarray:
    """
    Computes velocities of scaterred particles with non random angles

    Parameters
    ----------
    v : np.array(N,3)
        Normalized velocities of the incident particles.
    r1 : np.array(N)
        First number for generating phi.
    cos_khi : np.array(N)
        Second cos of khi.

    Returns
    -------
    v_scat : TYPE
        DESCRIPTION.

    """
    cos_theta = v[:, 0]
    sin_theta = np.maximum(np.sqrt(1 - cos_theta**2), 1e-15)

    cos_phi = np.cos(2 * np.pi * r1)
    sin_phi = np.sin(2 * np.pi * r1)

    sin_khi = np.sin(np.arccos(cos_khi))

    a = np.zeros_like(v)
    a[:, 0] = sin_khi * sin_phi / sin_theta

    b = np.zeros_like(v)
    b[:, 0] = sin_khi * cos_phi / sin_theta

    v_scat = (
        v * np.expand_dims(cos_khi, -1) + np.cross(v, a) + np.cross(np.cross(v, b), v)
    )

    return v_scat


def isotropic_scatter(v: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes velocities of scaterred particles

    Parameters
    ----------
    v : np.array(N,3)
        Normalized velocities of the incident particles.

    Returns
    -------
    np.array(N,3)
        Normalized scaterred velocities.
    np.array(N)
        Cosines of the scatter angles.

    """

    r1, r2 = rng.random((2, v.shape[0]))
    cos_khi = 1 - 2 * r2
    return fixed_angle_isotropic_scatter(v, r1, cos_khi), cos_khi


def isotropic_scatter_e(
    v: np.ndarray, energy: np.ndarray, rng
) -> Tuple[np.ndarray, np.ndarray]:
    r1, r2 = rng.random((2, v.shape[0]))

    cos_khi = (2 + energy - 2 * np.power(1 + energy, r2)) / energy
    return fixed_angle_isotropic_scatter(v, r1, cos_khi), cos_khi


@njit("f8[:,:,:](f8[:,:])")
def tensor_2(a):
    N = a.shape[0]
    res = np.zeros((N, 3, 3))
    for i in range(N):
        res[i, :, :] = np.outer(a[i, :], a[i, :])
    return res


@njit
def remove_random(x, V, Npart, N_remove):
    for _ in range(N_remove):
        i = np.random.randint(0, Npart)
        x[i] = x[Npart - 1]
        V[i, :] = V[Npart - 1, :]
        x[Npart - 1] = -1.0
        V[Npart - 1, :] = 0.0
        Npart -= 1
    return Npart


@njit
def remove_random_weighted(x, V, w, Npart, N_remove):
    """Remove particles with probability proportional to their weight.
    
    Parameters
    ----------
    x : np.array(N)
        Particle positions
    V : np.array(N,3)
        Particle velocities
    w : np.array(N)
        Particle weights
    Npart : int
        Number of active particles
    N_remove : int
        Number of particles to remove
        
    Returns
    -------
    Npart : int
        Updated number of particles
    """
    for _ in range(N_remove):
        # Compute cumulative sum of weights
        cumsum = np.cumsum(w[:Npart])
        total = cumsum[-1]
        
        # Generate random number and find particle index
        rand_val = np.random.random() * total
        i = np.searchsorted(cumsum, rand_val)
        if i >= Npart:
            i = Npart - 1
        
        # Remove particle by swapping with last active particle
        x[i] = x[Npart - 1]
        V[i, :] = V[Npart - 1, :]
        w[i] = w[Npart - 1]
        
        x[Npart - 1] = -1.0
        V[Npart - 1, :] = 0.0
        w[Npart - 1] = 0.0
        
        Npart -= 1
    return Npart


@njit
def find_cells(x, cells, dx):
    x /= dx
    i_cell = 0
    for i in range(len(x)):
        if int(x[i]) >= i_cell:
            cells[i_cell] = i
            i_cell += 1
    cells[-1] = len(x)
    x *= dx


@njit
def histograms(positions, v_squared, probe_size, bin_size, out):
    max_j = out.shape[1] - 1
    for x, v in zip(positions, v_squared):
        i = int(x / probe_size)
        j = min(int(v / bin_size), max_j)
        out[i, j] += 1.0


@njit
def histograms_v(positions, v, probe_size, bin_size, out):
    v_max = bin_size * (out.shape[1] / 2.0)
    for x, v in zip(positions, v):
        i = int(x / probe_size)
        clippedValue = min(max(0.0, (v + v_max)), 2 * v_max)
        j = int(clippedValue / bin_size)
        out[i, j] += 1.0


@njit
def histograms_weighted(positions, v_squared, weights, probe_size, bin_size, out):
    """Weighted histogram: adds particle weight instead of 1"""
    max_j = out.shape[1] - 1
    for x, v, w in zip(positions, v_squared, weights):
        i = int(x / probe_size)
        j = min(int(v / bin_size), max_j)
        out[i, j] += w


@njit
def histograms_v_weighted(positions, v, weights, probe_size, bin_size, out):
    v_max = bin_size * (out.shape[1] / 2.0)
    for x, v, w in zip(positions, v, weights):
        i = int(x / probe_size)
        clippedValue = min(max(0.0, (v + v_max)), 2 * v_max)
        j = int(clippedValue / bin_size)
        out[i, j] += w


def random_round(x):
    flored = np.floor(x)
    decimal = x - flored
    return (int(flored) + 1) if np.random.rand() < decimal else int(flored)


@njit
def rearrange(x, v, n_part, n_coll):
    for j in range(n_coll):
        i = np.random.randint(j, n_part)
        temp_x = x[i]
        x[i] = x[j]
        x[j] = temp_x
        temp_v = v[i]
        v[i] = v[j]
        v[j] = temp_v
