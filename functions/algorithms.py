import math
import numpy as np
import numpy.matlib as matlib
from functions.solvers import solveSystemLinearElasticFEM, solveSystemLinearViscoelasticFEM
from functions.tools import *
from classes import CAObject, PropertiesObject, DDIObject


def generateDataLinearElastic(sys, problemType, *, E = 1, Er = 1, forceType = None, C0 = 1, steps = None, forceVal = None, testVal = 'XXYYSS'):

    if sys.verbose:
        print("Generating the mechanical database through FEM.\n")


    LNG_X = np.max(sys.X[:,0]) - np.min(sys.X[:,0])
    LNG_Y = np.max(sys.X[:,1]) - np.min(sys.X[:,1])

    BND_south = np.nonzero(sys.X[:,1] == np.min(sys.X[:,1]) and sys.X[:,0] != np.min(sys.X[:,0]) and sys.X[:,0] != np.max(sys.X[:,0]))
    BND_north = np.nonzero(sys.X[:,1] == np.max(sys.X[:,1]) and sys.X[:,0] != np.min(sys.X[:,0]) and sys.X[:,0] != np.max(sys.X[:,0]))
    BND_east = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] != np.min(sys.X[:,1]) and sys.X[:,1] != np.max(sys.X[:,1]))
    BND_west = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] != np.min(sys.X[:,1]) and sys.X[:,1] != np.max(sys.X[:,1]))

    BND_ne = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] == np.max(sys.X[:,1]))
    BND_nw = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] == np.max(sys.X[:,1]))
    BND_se = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] == np.min(sys.X[:,1]))
    BND_sw = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] == np.min(sys.X[:,1]))

    F = []
    Fr = []
    uf = []
    rx = []

    prop = PropertiesObject(sys, problemType, E = E, Er = Er, testVal = testVal, steps = steps, forceType = forceType, forceVal = forceVal)

    SG = []
    EP = []

    if 'X' in testVal or 'Xt' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            SG.append(forceVal * (sys.BND_e[sys.BND_e != 0].size - 1) / LNG_Y)
            EP.append('X')

            F[-1][BND_east,0] = 1
            F[-1][BND_ne and BND_se,0] = 0.5
            rx[-1][sys.BND_w,:] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((sys.BND_e,np.full(sys.nNod, False)))])
            SG.append(forceVal)
            EP.append('X')

            rx[-1][sys.BND_w and sys.BND_e,:] = 1
            uf[-1][sys.BND_e,0] = forceVal * LNG_X
    if 'XX' in testVal or 'Xc' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            SG.append(forceVal * (sys.BND_e[sys.BND_e != 0].size - 1) / LNG_Y)
            EP.append('X')

            F[-1][BND_east,0] = -1
            F[-1][BND_ne and BND_se,0] = -0.5
            rx[-1][BND_west,0] = 1
            rx[-1][BND_nw,0] = 1
            rx[-1][BND_sw,:] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((sys.BND_e,np.full(sys.nNod, False)))])
            SG.append(forceVal)
            EP.append('X')

            rx[-1][sys.BND_w and sys.BND_e,:] = 1
            uf[-1][sys.BND_e,0] = -forceVal * LNG_X
    if 'Y' in testVal or 'Yt' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            SG.append(forceVal * (sys.BND_n[sys.BND_n != 0].size - 1) / LNG_X)
            EP.append('Y')

            F[-1][BND_north,1] = 1
            F[-1][BND_ne and BND_nw,1] = 0.5
            rx[-1][sys.BND_s,:] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((np.full(sys.nNod, False), sys.BND_n))])
            SG.append(forceVal)
            EP.append('Y')

            rx[-1][sys.BND_s and sys.BND_n,:] = 1
            uf[-1][sys.BND_n,1] = forceVal * LNG_Y
    if 'YY' in testVal or 'Yc' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            SG.append(forceVal * (sys.BND_n[sys.BND_n != 0].size - 1) / LNG_X)
            EP.append('Y')

            F[-1][BND_north,1] = -1
            F[-1][BND_ne and BND_nw,0] = -0.5
            rx[-1][BND_south,1] = 1
            rx[-1][BND_se,:] = 1
            rx[-1][BND_sw,1] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((np.full(sys.nNod, False), sys.BND_n))])
            SG.append(forceVal)
            EP.append('Y')

            rx[-1][sys.BND_s and sys.BND_n,:] = 1
            uf[-1][sys.BND_n,1] = -forceVal * LNG_X
    if 'SS' in testVal or 'Sh' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))

            F[-1][BND_east,1] = -0.25
            F[-1][BND_ne and BND_se,1] = -0.125
            rx[-1][sys.BND_w,:] = 1
            rx[-1][sys.BND_e,0] = 1

            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))

            F[-1][BND_east,1] = 0.25
            F[-1][BND_ne and BND_se,1] = 0.125
            rx[-1][sys.BND_w,:] = 1
            rx[-1][sys.BND_e,0] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((np.full(sys.nNod, False), sys.BND_e))])

            rx[-1][sys.BND_w,:] = 1
            rx[-1][sys.BND_e,:] = 1
            uf[-1][sys.BND_e,1] = -0.5 * forceVal * LNG_Y

            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((np.full(sys.nNod, False), sys.BND_e))])

            rx[-1][sys.BND_w,:] = 1
            rx[-1][sys.BND_e,:] = 1
            uf[-1][sys.BND_n,0] = 0.5 * forceVal * LNG_Y
            
        SG.append(0)
        SG.append(0)
        EP.append(0)
        EP.append(0)

    if 'SS' in testVal or 'Sv' in testVal:
        if forceType == "nodal":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            
            F[-1][BND_north,0] = -0.25
            F[-1][BND_ne and BND_nw,0] = -0.125
            rx[-1][sys.BND_s,:] = 1
            rx[-1][sys.BND_n,1] = 1
            
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))

            F[-1][BND_north,0] = 0.25
            F[-1][BND_ne and BND_nw,0] = 0.125
            rx[-1][sys.BND_s,:] = 1
            rx[-1][sys.BND_n,1] = 1
        elif forceType == "displacement":
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((sys.BND_n,np.full(sys.nNod, False)))])
            
            rx[-1][sys.BND_s,:] = 1
            rx[-1][sys.BND_n,:] = 1
            uf[-1][sys.BND_n,0] = -0.5 * forceVal * LNG_X
            
            F.append(np.zeros((sys.nNod, sys.nDim)))
            rx.append(np.full((sys.nNod, sys.nDim), False))
            uf.append(np.zeros((sys.nNod, sys.nDim)))
            Fr.append([np.hstack((sys.BND_n,np.full(sys.nNod, False)))])

            rx[-1][sys.BND_s,:] = 1
            rx[-1][sys.BND_n,:] = 1
            uf[-1][sys.BND_n,0] = 0.5 * forceVal * LNG_X
            
        SG.append(0)
        SG.append(0)
        EP.append(0)
        EP.append(0)

    prop.nX = prop.steps * len(prop.F)
    prop.F = [None for _ in range(prop.nX)]
    prop.uf = [None for _ in range(prop.nX)]
    prop.rx = [None for _ in range(prop.nX)]

    if forceType == "displacement":
        prop.Fr = [None for _ in range(prop.nX)]

    nX = 0
    nY = 0
    tX = np.zeros(len(F))
    tY = np.zeros(len(F))

    for i in range(len(F)-1,-1,-1):
        F[i] *= forceVal

        for j in range(prop.steps):
            prop.F[prop.steps * i + j] = F[i] * (j + 1) / prop.steps
            prop.uf[prop.steps * i + j] = uf[i] * (j + 1) / prop.steps
            prop.rx[prop.steps * i + j] = rx[i]

            if forceType == "displacement":
                prop.Fr[prop.steps * i + j] = Fr[i]

    femData = solveSystemLinearElasticFEM(sys, prop)

    for i in range(len(F)):
        if forceType == "nodal":
            if EP[i] == "X":
                EP[i] = np.mean(femData[prop.steps * (i + 1) - 1].u[np.hstack((sys.BND_e,np.full(sys.nNod, False)))]) / LNG_X
                nX += 1
                tX[i] = i
                C0[i] = np.abs(SG[i] / EP[i])
            elif EP[i] == "Y":
                EP[i] = np.mean(femData[prop.steps * (i + 1) - 1].u[np.hstack((np.full(sys.nNod, False), sys.BND_n))]) / LNG_Y
                nY += 1
                tY[i] = i
                C0[i] = np.abs(SG[i] / EP[i])
        elif forceType == "displacement":
            if EP[i] == "X":
                EP[i] = np.sum(femData[prop.steps * (i + 1) - 1].r[np.hstack((sys.BND_e,np.full(sys.nNod, False)))]) / LNG_Y
                nX += 1
                tX[i] = i
                C0[i] = np.abs(EP[i] / SG[i])
            elif EP[i] == "Y":
                EP[i] = np.sum(femData[prop.steps * (i + 1) - 1].r[np.hstack((np.full(sys.nNod, False), sys.BND_n))]) / LNG_X
                nY += 1
                tY[i] = i
                C0[i] = np.abs(EP[i] / SG[i])

    
    tX = tX[np.nonzero(tX)]
    tY = tY[np.nonzero(tY)]

    if nX > nY:
        prop.C0 = np.mean(*C0[tX])
    elif nY > nX:
        prop.C0 = np.mean(*C0[tY])
    else:
        prop.C0 = np.mean(*C0[tX and tY])

    return femData, prop


def generateDataLinearViscoelastic(sys, problemType, *, CData = None, Er = 1, dt = None, forceType = None, forceVal = None, testVal = 'X', previousSol = None):

    if sys.verbose:
        print("Generating the mechanical database through FEM.\n")


    LNG_X = np.max(sys.X[:,0]) - np.min(sys.X[:,0])
    LNG_Y = np.max(sys.X[:,1]) - np.min(sys.X[:,1])

    BND_south = np.nonzero(sys.X[:,1] == np.min(sys.X[:,1]) and sys.X[:,0] != np.min(sys.X[:,0]) and sys.X[:,0] != np.max(sys.X[:,0]))
    BND_north = np.nonzero(sys.X[:,1] == np.max(sys.X[:,1]) and sys.X[:,0] != np.min(sys.X[:,0]) and sys.X[:,0] != np.max(sys.X[:,0]))
    BND_east = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] != np.min(sys.X[:,1]) and sys.X[:,1] != np.max(sys.X[:,1]))
    BND_west = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] != np.min(sys.X[:,1]) and sys.X[:,1] != np.max(sys.X[:,1]))

    BND_ne = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] == np.max(sys.X[:,1]))
    BND_nw = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] == np.max(sys.X[:,1]))
    BND_se = np.nonzero(sys.X[:,0] == np.max(sys.X[:,0]) and sys.X[:,1] == np.min(sys.X[:,1]))
    BND_sw = np.nonzero(sys.X[:,0] == np.min(sys.X[:,0]) and sys.X[:,1] == np.min(sys.X[:,1]))

    F = []
    Fr = []
    uf = []
    rx = []

    prop = PropertiesObject(sys, problemType, CData = CData, Er = Er, dt = dt, forceVal = forceVal, testVal = testVal, steps = 1, forceType = forceType)

    if 'X' in testVal:
        rx.append(np.full((sys.nNod, sys.nDim), False))

        rx[-1][sys.BND_w,:] = 1
        if forceType == "displacement":
            rx[-1][sys.BND_e,:] = 1
    elif 'Y' in testVal:
        rx.append(np.full((sys.nNod, sys.nDim), False))

        rx[-1][sys.BND_s,:] = 1
        if forceType == "displacement":
            rx[-1][sys.BND_n,:] = 1

    nX = 0

    for n in range(len(forceVal)):
        step_n = np.round((forceVal[n][-1] - forceVal[n][-2]) / prop.dt)
        nX += step_n

        if forceType == "nodal":
            if 'X' in testVal:
                F.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                uf.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                Fr.append([np.array([]) for _ in range(step_n)])
            elif 'Y' in testVal:
                F.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                uf.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                Fr.append([np.array([]) for _ in range(step_n)])
        elif forceType == "diplacement":
            if 'X' in testVal:
                uf.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                Fr.append([np.array([]) for _ in range(step_n)])
                F.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
            elif 'Y' in testVal:
                uf.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])
                Fr.append([np.array([]) for _ in range(step_n)])
                F.append([np.zeros((sys.nNod, sys.nDim)) for _ in range(step_n)])

        for i in range(step_n):
            if forceVal[n][0] == "linear":
                if forceVal[n][1] == 'X':
                    FVALX = forceVal[n][2] + i * (forceVal[n][3] - forceVal[n][2]) / step_n
                    FVALY = 0
                elif forceVal[n][1] == 'Y':
                    FVALX = 0
                    FVALY = forceVal[n][2] + i * (forceVal[n][3] - forceVal[n][2]) / step_n
                elif forceVal[n][1] == 'XY':
                    FVALX = forceVal[n][2] + i * (forceVal[n][3] - forceVal[n][2]) / step_n
                    FVALY = forceVal[n][2] + i * (forceVal[n][3] - forceVal[n][2]) / step_n
            elif forceVal[n][0] == "constant":
                if forceVal[n][1] == 'X':
                    FVALX = forceVal[n][2]
                    FVALY = 0
                elif forceVal[n][1] == 'Y':
                    FVALX = 0
                    FVALY = forceVal[n][2]
                elif forceVal[n][1] == 'XY':
                    FVALX = forceVal[n][2]
                    FVALY = forceVal[n][2]
            elif forceVal[n][0] == "sine":
                if forceVal[n][1] == 'X':
                    FVALX = forceVal[n][3] + forceVal[n][2] * np.sin(forceVal[n][4] * i * dt)
                    FVALY = 0
                elif forceVal[n][1] == 'Y':
                    FVALX = 0
                    FVALY = forceVal[n][3] + forceVal[n][2] * np.sin(forceVal[n][4] * i * dt)
                elif forceVal[n][1] == 'XY':
                    FVALX = forceVal[n][3] + forceVal[n][2] * np.sin(forceVal[n][4] * i * dt)
                    FVALY = forceVal[n][3] + forceVal[n][2] * np.sin(forceVal[n][4] * i * dt)
            elif forceVal[n][0] == "cosine":
                if forceVal[n][1] == 'X':
                    FVALX = forceVal[n][3] + forceVal[n][2] * np.cos(forceVal[n][4] * i * dt)
                    FVALY = 0
                elif forceVal[n][1] == 'Y':
                    FVALX = 0
                    FVALY = forceVal[n][3] + forceVal[n][2] * np.cos(forceVal[n][4] * i * dt)
                elif forceVal[n][1] == 'XY':
                    FVALX = forceVal[n][3] + forceVal[n][2] * np.cos(forceVal[n][4] * i * dt)
                    FVALY = forceVal[n][3] + forceVal[n][2] * np.cos(forceVal[n][4] * i * dt)
            elif forceVal[n][0] == "function":
                if forceVal[n][1] == 'X':
                    FVALX = forceVal[n][2][i]
                    FVALY = 0
                    FVALM = 0
                elif forceVal[n][1] == 'Y':
                    FVALX = 0
                    FVALY = forceVal[n][2][i]
                    FVALM = 0
                elif forceVal[n][1] == 'XY':
                    FVALX = forceVal[n][2][i]
                    FVALY = forceVal[n][3][i]
                    FVALM = 0
                elif forceVal[n][1] == 'XYM':
                    FVALX = forceVal[n][2][i]
                    FVALY = forceVal[n][3][i]
                    FVALM = forceVal[n][4][i]
            
            if forceType == "nodal":
                if testVal == 'X':
                    F[n][i][BND_east,0] = FVALX
                    F[n][i][BND_east,1] = FVALY
                    F[n][i][BND_ne and BND_se,0] = 0.5 * FVALX
                    F[n][i][BND_ne and BND_se,1] = 0.5 * FVALY
                elif testVal == 'Y':
                    F[n][i][BND_north,0] = FVALX
                    F[n][i][BND_north,1] = FVALY
                    F[n][i][BND_ne and BND_nw,0] = 0.5 * FVALX
                    F[n][i][BND_ne and BND_nw,1] = 0.5 * FVALY
            elif forceType == "displacement":
                if testVal == 'X':
                    uf[n][i][BND_east,0] = FVALX * LNG_X
                    uf[n][i][BND_east,1] = FVALY * LNG_Y
                    Fr[n][i] = np.hstack((sys.BND_w, np.full(sys.nNod, False)))
                elif testVal == 'Y':
                    uf[n][i][BND_north,0] = FVALX * LNG_X
                    uf[n][i][BND_north,1] = FVALY * LNG_Y
                    Fr[n][i] = np.hstack((np.full(sys.nNod, False), sys.BND_n))

    prop.nX = nX

    prop.F = [item for n in F for item in n]
    prop.uf = [item for n in uf for item in n]
    prop.Fr = [item for n in Fr for item in n]
    prop.rx = [rx[0] for _ in range(len(prop.F))]

    if previousSol == None:
        femData = solveSystemLinearViscoelasticFEM(sys, prop, [])
    else:
        femData = previousSol
        femData = solveSystemLinearViscoelasticFEM(sys, prop, femData)

    return femData, prop


def generateDataNonlinearElastic(sys, problemType, *, E = 1, Er = 1, CData = None, dt = None, forceType = None, C0 = 1, curvType = None, curvC = None, curvPts = None, steps = None, forceVal = None, testVal = 'XXYYSS', previousSol = None):

    if sys.verbose:
        print("Generating the mechanical database through FEM.\n")

def generateDataDDCM(sys, problemType, *, E = 1, Er = 1, CData = None, dt = None, forceType = None, C0 = 1, curvType = None, curvC = None, curvPts = None, steps = None, forceVal = None, testVal = 'XXYYSS', previousSol = None):

    if sys.verbose:
        print("Generating the mechanical database through DDCM.\n")



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