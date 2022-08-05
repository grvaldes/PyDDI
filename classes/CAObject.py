import numpy as np
from scipy import sparse

class CAObject:

    def __init__(self, ddi, centered = True):
        
        cM = 0

        for i in range(len(ddi.ieM)):
            cM += ddi.ieM[i]

        self.N = cM.copy()

        if centered:
            cM /= np.sum(cM)

        rm = np.sum(cM, 1)
        cm = np.sum(cM, 0)

        isqDr = sparse.diags(1 / np.sqrt(rm), 0, shape = (rm.shape, rm.shape))
        isqDc = sparse.diags(1 / np.sqrt(cm), 0, shape = (cm.shape, cm.shape))

        if centered:
            cM -= rm @ cm

        S = isqDr @ cM @ isqDc

        self.U, self.Da, self.V = np.linalg.svd(S,)

        if not centered:
            self.U = self.U[:,1:]
            self.V = self.V[:,1:]
            self.Da = self.Da[1:]

        self.Phi = isqDr @ self.U
        self.Gam = isqDc @ self.V
        self.F = self.Phi * self.Da
        self.G = self.Gam * self.Da
        self.L = self.Da ** 2

        self.mat = []
        self.grp = []

        self.kmF = None
        self.kmG = None
        self.cM = cM.copy()
        self.S = S.copy()
        self.rm = rm.copy()
        self.cm = cm.copy()
