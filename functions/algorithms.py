import numpy as np
from classes import MaterialDatabase

def solveSystemDDI(sys, prop, th = 1, mxI = 500, mxJ = 500, EsTol = 1e-6, SsTol = 1e-6, rPCA = False, tPCA = 0.9):
    
    if rPCA:
        sexP = []

    wDS = np.zeros(mxI)

    mdb = MaterialDatabase(prop.es, prop.ss, prop.C0)

    return sol