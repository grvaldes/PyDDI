import numpy as np
import scipy.linalg
from scipy import sparse


class FemStressProjector:

    def __init__(self, GRAD, DIV, C0, Ce, CSTR, ResLists):
        self.name = "FEM STRESS PROJECTOR"
        self.Nquad = Ce.size 
        self.Ncomp = C0.shape[0]
        self.Nstr = DIV.shape[1]
        self.mask = np.full(DIV.shape[0], True)
        self.mask[CSTR] = False
        
        self.DIV = DIV
        self.B = GRAD
        self.CB = np.kron(C0,sparse.spdiags(Ce,0,self.Nquad,self.Nquad)) @ self.B

        for iSet in range(len(ResLists)):
            self.DIV = np.vstack((self.DIV, np.sum(self.DIV[ResLists[iSet],:], axis = 0)))
            self.CB = np.hstack((self.CB, np.sum(self.CB[:,ResLists[iSet]], axis = 1)))

        self.DIV = self.DIV[self.mask,:]
        self.CB = self.CB[:,self.mask]
        self.B = self.B[:,self.mask]

        self.K = self.DIV @ self.CB
        self.renumbering = np.arange(self.K.shape[0])
        self.R = scipy.linalg.cholesky(self.K[self.renumbering,self.renumbering])


    def project(self, F, Fres, stress = None):
        tmp = np.hstack((F.ravel(), Fres.ravel()))
        tmp = tmp[self.mask]

        if stress == None:
            stress = np.zeros(self.DIV.shape[1]).reshape((-1,self.Ncomp))

        RHS = tmp - self.DIV @ stress.ravel()
        eta = np.zeros(RHS.shape[0])
        eta[self.renumbering] = np.linalg.solve(self.R, np.linalg.solve(np.transpose(self.R), RHS[self.renumbering]))
        stress_compat = stress + (self.CB @ eta).reshape((-1,self.Ncomp))

        return eta, stress_compat

