import math
import numpy as np
from functions.tools import nestedKmeans

class DDIObject:

    def __init__(self, sys, prop, sols, initMethod = "kmeans"):

        self.ee = []
        self.se = None
        
        for i in range(len(sols)):
            self.ee[i] = sols[i].ee.copy()
        
        self.initMaterialStates(np.vstack(self.ee), prop.nStr, initMethod)

        if sys.materialType == "LinearViscoelastic":
            self.ss = np.zeros((self.es.shape[0], sys.comp))
        
        self.ieM = None
        self.wDS = None
        self.iter = None
        self.init_clock = None
        self.alg_clock = None

        self.sep = None
        self.L = None
        self.nv = None

    def initMaterialStates(self, allEps, nStr, initMethod):

        dim = allEps.shape[1]

        if initMethod == "None":
            self.es = np.zeros((nStr, dim))
        elif initMethod == "kmeans":
            _, self.es = nestedKmeans(allEps, nStr, 100)
        elif initMethod == "random":
            self.es = datasample() #TODO
        elif initMethod == "normal":
            mue = np.mean(allEps, 0)
            sde = np.std(allEps, 0)

            normAllEps = (allEps - mue) / sde
            _, self.es = nestedKmeans(normAllEps, nStr, 100)
        elif initMethod == "uniform":
            allMin = np.min(allEps, 0)
            allMax = np.max(allEps, 1)
            all1D = []
            NG = math.ceil(np.prod(nStr) ** (1 / dim))

            for i in range(dim):
                all1D.append(np.linspace(allMin[i], allMax[i], NG))

            allND = []

            for i in range(dim):
                allND.append(np.meshgrid(*all1D)[i].ravel())

            self.es = np.hstack(tuple(allND))

        else:
            raise Exception("Unknown initialization method.")

        self.ss = np.zeros(self.es.shape)