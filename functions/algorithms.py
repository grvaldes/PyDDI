import math
import numpy as np
import numpy.matlib as matlib
from functions.solvers import solveSystemLinearElasticFEM, solveSystemLinearViscoelasticFEM
from functions.tools import *
from classes import CAObject, PropertiesObject, DDIObject


def generateData(sys, problemType, *, E = 1, Er = 1, CData = None, dt = None, forceType = None, C0 = 1, curvType = None, curvC = None, curvPts = None, steps = None, forceVal = None, testVal = 'XXYYSS', previousSol = None):

    if sys.verbose:
        if problemType == "DDCM":
            print("Generating the mechanical database through DDCM.\n")
        else:
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

    if problemType == "Linear":
        prop = PropertiesObject(sys, problemType, E = E, Er = Er, testVal = testVal, steps = steps, forceType = forceType, forceVal = forceVal)

        SG = []
        EP = []

        if 'X' in testVal or 'Xt' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                SG.append()
                EP.append('X')
            elif forceType == "displacement":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                SG.append()
                EP.append('X')

        if 'XX' in testVal or 'Xc' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                SG.append()
                EP.append('X')
            elif forceType == "displacement":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                SG.append()
                EP.append('X')

        if 'Y' in testVal or 'Yt' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                SG.append()
                EP.append('Y')
            elif forceType == "displacement":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                SG.append()
                EP.append('Y')

        if 'YY' in testVal or 'Yc' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                SG.append()
                EP.append('Y')
            elif forceType == "displacement":
                F.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                SG.append()
                EP.append('Y')

        if 'SS' in testVal or 'Sh' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                F.append(np.array([]))
                rx.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                uf.append(np.array([]))
            elif forceType == "displacement":
                F.append(np.array([]))
                F.append(np.array([]))
                rx.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                Fr.append(np.array([]))
                
            SG.append(0)
            SG.append(0)
            EP.append(0)
            EP.append(0)

        if 'SS' in testVal or 'Sv' in testVal:
            if forceType == "nodal":
                F.append(np.array([]))
                F.append(np.array([]))
                rx.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                uf.append(np.array([]))
            elif forceType == "displacement":
                F.append(np.array([]))
                F.append(np.array([]))
                rx.append(np.array([]))
                rx.append(np.array([]))
                uf.append(np.array([]))
                uf.append(np.array([]))
                Fr.append(np.array([]))
                Fr.append(np.array([]))
                
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
            F[i][:,1:] *= forceVal

            for j in range(prop.steps):
                prop.F[prop.steps * i + j][:,1:] = F[i][:,1:] * (j + 1) / prop.steps
                prop.uf[prop.steps * i + j][:,1:] = uf[i][:,1:] * (j + 1) / prop.steps
                prop.rx[prop.steps * i + j][:,1:] = rx[i]

                if forceType == "displacement":
                    prop.Fr[prop.steps * i + j][:,1:] = Fr[i]

        femData = solveSystemLinearElasticFEM(sys, prop)

        for i in range(len(F)):
            if forceType == "nodal":
                if EP[i] == "X":
                    EP[i] = np.mean(femData[prop.steps * (i + 1) - 1].u[sys.BND_e]) / LNG_X
                    nX += 1
                    tX[i] = i
                    C0[i] = np.abs(SG[i] / EP[i])
                elif EP[i] == "Y":
                    EP[i] = np.mean(femData[prop.steps * (i + 1) - 1].u[sys.nNod + sys.BND_n]) / LNG_Y
                    nY += 1
                    tY[i] = i
                    C0[i] = np.abs(SG[i] / EP[i])
            elif forceType == "displacement":
                if EP[i] == "X":
                    EP[i] = np.sum(femData[prop.steps * (i + 1) - 1].r[sys.BND_e]) / LNG_Y
                    nX += 1
                    tX[i] = i
                    C0[i] = np.abs(EP[i] / SG[i])
                elif EP[i] == "Y":
                    EP[i] = np.sum(femData[prop.steps * (i + 1) - 1].r[sys.nNod + sys.BND_n]) / LNG_X
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

    if problemType == "LinearViscoelastic" or problemType == "LinearViscoelastic2":
        prop = PropertiesObject(sys, problemType, CData = CData, Er = Er, dt = dt, forceVal = forceVal, testVal = testVal, steps = 1, forceType = forceType)

        if 'X' in testVal:
            rx.append(np.array([]))

            if forceType == "displacement":
                np.vstack((rx, np.array([])))
        elif 'Y' in testVal:
            rx.append(np.array([]))

            if forceType == "displacement":
                np.vstack((rx, np.array([])))

        nX = 0

        for n in range(len(forceVal)):
            step_n = np.round((forceVal[n][-1] - forceVal[n][-2]) / prop.dt)
            nX += step_n

            if forceType == "nodal":
                if 'X' in testVal:
                    F.append(np.zeros((np.sum(sys.BND_e), 3, step_n)))
                    uf.append(np.zeros((0, 0, step_n)))
                    Fr.append(np.zeros((0, 0, step_n)))
                elif 'Y' in testVal:
                    F.append(np.zeros((np.sum(sys.BND_n), 3, step_n)))
                    uf.append(np.zeros((0, 0, step_n)))
                    Fr.append(np.zeros((0, 0, step_n)))
            elif forceType == "diplacement":
                if 'X' in testVal:
                    uf.append(np.zeros((np.sum(sys.BND_e), 3, step_n)))
                    Fr.append(np.zeros((np.sum(sys.BND_e), 1, step_n)))
                    F.append(np.zeros((0, 0, step_n)))
                elif 'Y' in testVal:
                    uf.append(np.zeros((np.sum(sys.BND_n), 3, step_n)))
                    Fr.append(np.zeros((np.sum(sys.BND_n), 1, step_n)))
                    F.append(np.zeros((0, 0, step_n)))

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
                        F[n][:,:,i] = np.array([])
                    elif testVal == 'Y':
                        F[n][:,:,i] = np.array([])
                elif forceType == "displacement":
                    if testVal == 'X':
                        uf[n][:,:,i] = np.array([])
                        Fr[n][:,:,i] = np.array([])
                    elif testVal == 'Y':
                        uf[n][:,:,i] = np.array([])
                        Fr[n][:,:,i] = np.array([])

        prop.nX = nX

        prop.F = np.squeeze()
        prop.uf = np.squeeze()
        prop.Fr = np.squeeze()
        prop.rx = np.squeeze()

        if previousSol == None:
            femData = solveSystemLinearViscoelasticFEM(sys, prop, [])
        else:
            femData = previousSol
            femData = solveSystemLinearViscoelasticFEM(sys, prop, femData)

    if problemType == "NonLinear":
        prop = PropertiesObject(sys, problemType)
    if problemType == "DDCM":
        prop = PropertiesObject(sys, problemType)

    return femData, prop


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