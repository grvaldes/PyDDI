from hashlib import algorithms_guaranteed
import math
import numpy as np
import numpy.matlib as matlib
from classes.CAObject import CAObject
from functions.tools import *
from classes import DDIObject


def generateData():
    return 0


def assembleDDIObject(sys, prop, femData, CType = "Diagonal", C0 = None, nStr = 500, previousSteps = 0, skipSteps = 1):

    if C0 == None:
        C0 = prop.C0

    if type(C0) is not tuple:
        if sys.problemType == "PlaneStress":
            if CType == "PlaneStress":
                C0 = CPlaneStress(C0, 0.3)
            elif CType == "Diagonal":
                C0 = C0 * np.eye(3)
            else:
                C0 = np.eye(3)

    prop.C0 = C0.copy()
    prop.nStr = nStr

    if sys.materialType == "LinearViscoelastic":
        d2 = femData[0].shape[1]

        for i in range(len(femData)):
            femData[i].ee = matlib.repmat(femData[i].ee, 1, previousSteps + 1)

            for j in range(1,previousSteps + 1):
                if i - j * skipSteps > 0:
                    femData[i].ee[:,j * d2:(j + 1) * d2] = femData[i - j * skipSteps].eps[:,:d2]
                else:
                    femData[i].ee[:,j * d2:(j + 1) * d2] = np.zeros((femData[i].shape[0], d2))
    elif sys.materialType == "LinearViscoelastic2":
        for i in range(len(femData)):
            femData[i].ee = np.hstack((femData[i].ee, femData[i].deps))

    if type(prop.fT) is not list or type(prop.fT) is not tuple:
        prop.fT = [prop.fT for _ in range(prop.nX)]

    for i in range(prop.nX):
        if prop.fT[i] == "displacement":
            prop.rl.append(prop.Fr[math.ceil(i/ prop.steps)])
            prop.fc.append(np.zeros(len(prop.rl[i])))

            for j in range(len(prop.rl[i])):
                prop.fc[i][j] = np.sum(femData[i].r[prop.rl[i][j]])
        else:
            prop.fc.append([])
            prop.rl.append(0)

        prop.fp.append(femData[i].fp.copy())
        prop.rp.append(femData[i].rp.copy())
        prop.up.append(femData[i].ip.copy())


    return DDIObject(sys, prop, femData)


def performCorrespondenceAnalysis(sys, ddi, algorithm = "kmeans", centered = True, nModes = 2, nMats = 2, *, limit = None):
    
    CA = CAObject(ddi)

    if sys.verbose:
        print("The amount of materials to look for is ", nMats, ".")
    
    if algorithm == "kmeans":
        kmF, _ = simpleKmeans(CA.F[:,0:nModes], nMats, 100)
        kmG, _ = simpleKmeans(CA.G[:,0:nModes], nMats, 100)
    elif algorithm == "limit":
        if limit.shape[0] == limit.shape[1]:
            if limit.shape[0] == nModes:
                if nModes == 1:
                    kmF = np.zeros(CA.F.shape[0])
                    kmG = np.zeros(CA.G.shape[0])

                    for i in range(kmF.shape):
                        if CA.F[i,0] < limit:
                            kmF[i] = 1
                        else:
                            kmF[i] = 2

                    for i in range(kmG.shape):
                        if CA.G[i,0] < limit:
                            kmG[i] = 1
                        else:
                            kmG[i] = 2
                if nModes == 2:
                    m = np.polyfit(limit[:,0], limit[:,1], 1)
                    lim = lambda y : (y - m[1]) / m[0]

                    kmF = np.zeros(CA.F.shape[0])
                    kmG = np.zeros(CA.G.shape[0])

                    for i in range(kmF.shape):
                        if CA.F[i,0] < lim(CA.F[i,1]):
                            kmF[i] = 1
                        else:
                            kmF[i] = 2

                    for i in range(kmG.shape):
                        if CA.G[i,0] < lim(CA.G[i,1]):
                            kmG[i] = 1
                        else:
                            kmG[i] = 2
                elif nModes == 3:
                    v = np.cross(limit[0,:] - limit[1,:], limit[2,:] - limit[1,:])
                    d = np.dot(v, limit[1,:])

                    lim = lambda y, z : (d - v[1] * y - v[2] * z) / v[0]

                    kmF = np.zeros(CA.F.shape[0])
                    kmG = np.zeros(CA.G.shape[0])

                    for i in range(kmF.shape):
                        if CA.F[i,0] < lim(CA.F[i,1], CA.F[i,2]):
                            kmF[i] = 1
                        else:
                            kmF[i] = 2

                    for i in range(kmG.shape):
                        if CA.G[i,0] < lim(CA.G[i,1], CA.G[i,2]):
                            kmG[i] = 1
                        else:
                            kmG[i] = 2

    for i in range(np.max(kmF)):
        CA.mat.append(np.nonzero(kmG == i + 1))
        CA.grp.append(np.nonzero(kmF == i + 1))

    if sys.verbose:
        print(len(CA.grp), "materials were found.")

        for i in range(len(CA.grp)):
            print("Material", i+1, "has", np.size(CA.grp[i][CA.grp[i]]), "elements.")

    if sys.nIP > 1:
        # I have to do this
        CA.t_grp = []

    CA.kmF = kmF.copy()
    CA.kmG = kmG.copy()

    return CA