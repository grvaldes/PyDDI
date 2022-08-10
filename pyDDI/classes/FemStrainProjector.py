import numpy as np
import scipy.linalg
from scipy import sparse


class FemStrainProjector:

    def __init__(self, GRAD, DIV, C0, Ce, CSTR):
        self.name = "FEM STRAIN PROJECTOR"
        self.Nquad = Ce.size 
        self.Ncomp = C0.shape[0]
        self.Nstr = DIV.shape[1]
        self.Nnodes = GRAD.shape[1]
        self.mask = np.full(self.Nnodes, True)
        self.mask[CSTR] = False
        self.CSTR = CSTR
        
        self.DIV = DIV
        self.GRAD = GRAD
        self.C = np.kron(C0,sparse.spdiags(Ce,0,self.Nquad,self.Nquad))
        self.DirichletCoupling = - self.DIV[self.mask,:] @ self.C @ self.GRAD[:,~self.mask]

        self.K = self.DIV[self.mask,:] @ (self.C @ self.GRAD[:,self.mask])
        self.renumbering = np.arange(self.K.shape[0])
        self.R = scipy.linalg.cholesky(self.K[self.renumbering,self.renumbering])


    def project(self, CSTR_VAL, strain = None):
        u = np.zeros(self.Nnodes)
        u[self.CSTR] = CSTR_VAL

        if strain == None:
            strain = np.zeros(self.C.shape[1]).reshape((-1,self.Ncomp))

        RHS = self.DirichletCoupling @ u[~self.mask] + self.DIV[self.mask,:] @ (self.C @ strain.ravel())

        tmp = np.zeros(self.renumbering.size)
        tmp[self.renumbering] = np.linalg.solve(self.R, np.linalg.solve(np.transpose(self.R), RHS[self.renumbering]))
        u[self.mask] = tmp
        strain_compat = (self.GRAD @ u).reshape((-1,self.Ncomp))

        return u, strain_compat

