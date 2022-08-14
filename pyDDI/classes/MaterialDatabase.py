import numpy as np
import scipy.linalg
import scipy.spatial
from scipy import sparse

class MaterialDatabase:

    def __init__(self, forces, fluxes, C = None, iC = None):
        assert forces.shape == fluxes.shape, "Force-Fluxes size mismatch"

        self.NS, self.dim_forces = forces.shape
        _, self.dim_fluxes = fluxes.shape
        self.forces = forces
        self.fluxes = fluxes

        if C == None:
            C = np.eye(self.dim_forces)
        else:
            assert C.shape == (self.dim_forces, self.dim_forces), "Force-C mismatch"
        
        if iC == None:
            iC = np.eye(self.dim_fluxes)
        else:
            assert C.shape == (self.dim_fluxes, self.dim_fluxes), "Force-inv(C) mismatch"

        self.sqC = scipy.linalg.sqrtm(C)
        self.sqiC = scipy.linalg.sqrtm(iC)
        self.searcher = scipy.spatial.KDTree(np.hstack((self.force @ self.sqC, self.fluxes @ self.sqiC)))
        self.acc_fluxes = np.zeros(self.fluxes.shape)
        self.w_acc_fluxes =  np.zeros(self.fluxes.shape[0])
        self.acc_forces = np.zeros(self.forces.shape)
        self.w_acc_forces = np.zeros(self.forces.shape[0])

    
    def project(self, ext_forces, ext_fluxes):
        D, ie = self.searcher.query(np.hstack((ext_forces @ self.sqC, ext_fluxes @ self.sqiC)))
        proj_forces = self.forces[ie,:]
        proj_fluxes = self.fluxes[ie,:]
        ieM = self.projectionMatrix(ie)

        return proj_forces, proj_fluxes, ieM, D


    def accumFluxes(self, ext_fluxes, ieM, w):
        self.acc_fluxes = self.acc_fluxes + np.transpose(ieM) @ (ext_fluxes * w)
        self.w_acc_fluxes = self.w_acc_fluxes + np.transpose(ieM) * w


    def updateFluxes(self, theta = 1):
        tmp = theta * (self.acc_fluxes / self.w_acc_fluxes) + (1 - theta) * self.fluxes
        tmp[np.isnan(tmp)] = 0
        relUpdate = np.linalg.norm(tmp - self.fluxes) / np.linalg.norm(self.fluxes)
        self.fluxes = tmp
        self.searcher = scipy.spatial.KDTree(np.hstack((self.forces @ self.sqC, self.fluxes @ self.sqiC)))
        self.acc_fluxes = np.zeros(self.fluxes.shape)
        self.w_acc_fluxes = np.zeros(self.fluxes.shape[0])

        return relUpdate


    def accumForces(self, ext_forces, ieM, w):
        self.acc_forces = self.acc_forces + np.transpose(ieM) @ (ext_forces * w)
        self.w_acc_forces = self.w_acc_forces + np.transpose(ieM) * w


    def updateForces(self, theta = 1):
        tmp = theta * (self.acc_forces / self.w_acc_forces) + (1 - theta) * self.forces
        tmp[np.isnan(tmp)] = 0
        relUpdate = np.linalg.norm(tmp - self.forces) / np.linalg.norm(self.forces)
        self.forces = tmp
        self.searcher = scipy.spatial.KDTree(np.hstack((self.forces @ self.sqC, self.fluxes @ self.sqiC)))
        self.acc_forces = np.zeros(self.forces.shape)
        self.w_acc_forces = np.zeros(self.forces.shape[0])

        return relUpdate


    def pruneDatabase(self, ieM):
        acc = np.zeros(self.NS)

        if type(ieM) is tuple:
            for iX in range(len(ieM)):
                acc = acc + np.transpose(ieM[iX]) @ np.ones(ieM[iX].shape[0])
        else:
            acc = acc + np.transpose(ieM) @ np.ones(ieM.shape[0])

        mask = np.nonzero(acc == 0)

        if any(mask):
            self.forces = self.forces[mask,:]
            self.fluxes = self.fluxes[mask,:]
            self.acc_forces = self.acc_forces[mask,:]
            self.acc_fluxes = self.acc_fluxes[mask,:]
            self.w_acc_forces = self.w_acc_forces[mask]
            self.w_acc_fluxes = self.w_acc_fluxes[mask]
            self.searcher = scipy.spatial.KDTree(np.hstack((self.forces @ self.sqC, self.fluxes @ self.sqiC)))
            self.NS = self.forces.shape[0]


    def projectionMatrix(self, ie):
        Ng = ie.size
        ieM = sparse.csr_matrix((np.ones(Ng), (np.arange(Ng), ie)), shape = (Ng, self.NS))

        return ieM