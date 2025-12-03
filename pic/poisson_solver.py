"""Solver of the Poisson equation for a 1D PIC simulation"""

import numpy as np
from pic.functions import numba_thomas_solver


class Poisson_Solver(object):
    """docstring for Poisson_Solver."""

    def __init__(self, Nx, func_1, func_2):
        super(Poisson_Solver, self).__init__()
        self.Nx = Nx
        self.func_1 = func_1
        self.func_2 = func_2

    def solve(rho):
        """general solver, for Thomas and SOR"""
        pass


class Dirichlet(Poisson_Solver):
    def init_thomas(self):
        """Initialisation of the {b, a, c}_i values, and {c'}_i"""
        self.bi = -np.ones(self.Nx, dtype="float")
        self.bi[1:-1] *= 2

        [self.ai, self.ci] = np.ones((2, self.Nx), dtype="float")
        # ci[0] = 0 for left derichel condtion
        # ci[-1] = 0 useless but to be sure
        self.ci[-1] = 0.0
        self.ci[0] = 0.0
        self.ai[[0, -1]] = 0.0

        ciprim = np.copy(self.ci)  # copy the value, not the reference
        ciprim[0] /= self.bi[0]
        for i in np.arange(1, len(ciprim)):
            ciprim[i] /= self.bi[i] - self.ai[i] * ciprim[i - 1]

        self.ciprim = ciprim
        self.inited_thomas = True

    def thomas_solver(self, rho, time, norm=1):
        """solve phi for Rho using Thomas solver, need initialisation first"""

        di = -rho  # is Rho is normed but not signed
        di[0] = -self.func_1 / norm
        di[-1] = -self.func_2 / norm
        phi = numba_thomas_solver(di, self.ai, self.bi, self.ciprim, self.Nx)
        return phi


class Dirichlet_Dynamic(Poisson_Solver):
    def init_thomas(self):
        """Initialisation of the {b, a, c}_i values, and {c'}_i"""
        self.bi = -np.ones(self.Nx, dtype="float")
        self.bi[1:-1] *= 2

        [self.ai, self.ci] = np.ones((2, self.Nx), dtype="float")
        # ci[0] = 0 for left derichel condtion
        # ci[-1] = 0 useless but to be sure
        self.ci[-1] = 0.0
        self.ci[0] = 0.0
        self.ai[[0, -1]] = 0.0

        ciprim = np.copy(self.ci)  # copy the value, not the reference
        ciprim[0] /= self.bi[0]
        for i in np.arange(1, len(ciprim)):
            ciprim[i] /= self.bi[i] - self.ai[i] * ciprim[i - 1]

        self.ciprim = ciprim
        self.inited_thomas = True

    def thomas_solver(self, rho, time, norm=1):
        """solve phi for Rho using Thomas solver, need initialisation first"""

        di = -rho  # is Rho is normed but not signed
        di[0] = -self.func_1(time) / norm
        di[-1] = -self.func_2(time) / norm
        phi = numba_thomas_solver(di, self.ai, self.bi, self.ciprim, self.Nx)
        return phi


'''
class Neumann(Poisson_Solver): #not well defined --> Neumann-Dirichlet
    
    def init_thomas(self):
        """Initialisation of the {b, a, c}_i values, and {c'}_i """
        self.bi = - np.ones(self.Nx, dtype='float')
        self.bi[1:-1] *= 2

        [self.ai, self.ci] = np.ones((2, self.Nx), dtype='float')

        ciprim = np.copy(self.ci)  # copy the value, not the reference
        ciprim[0] /= self.bi[0]
        for i in np.arange(1, len(ciprim)):
            ciprim[i] /= self.bi[i] - self.ai[i]*ciprim[i-1]

        self.ciprim = ciprim
        self.inited_thomas = True

    def thomas_solver(self, rho, time, dx=1., q=1., qf=1., eps_0=1.):
        """solve phi for Rho using Thomas solver, need initialisation first
        """

        di = - rho  # is Rho is normed but not signed
        di[0] += self.func_1
        di[-1] += self.func_2
        phi = numba_thomas_solver(di, self.ai, self.bi, self.ciprim, self.Nx)
        return phi
'''
# scipy.linalg pour résoudre pour un probleme périodique
# tests unitaires (symetrie)
