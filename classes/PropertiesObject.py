import numpy as np
from functions.tools import CPlaneStress, generatePoints, parseConstitutive

class PropertiesObject:

    def __init__(self, sys, problemType, *, E = 1, Er = 1, CData = None, dt = None, forceType = None, C0 = 1, curvType = None, curvC = None, curvPts = None, steps = None, forceVal = None, testVal = 'XXYYSS'):

        self.nStr = None
        self.nX = None
        self.C0 = None
        self.Er = Er

        self.F = None
        self.Fr = None
        self.rx = None
        self.uf = None
        self.fp = []
        self.up = []
        self.rp = []
        self.rl = []
        self.fc = []

        if problemType == "DDCM":
            self.vC0 = np.ones(sys.nEl)
            self.pts = curvPts

            if steps == None:
                self.steps = 50
            else:
                self.steps = steps

            if sys.problemType == "PlaneStress":
                self.C0 = CPlaneStress(C0, 0.3)
            else:
                self.C0 = C0

            if len(sys.grp) > 1:
                self.ee = [0]
                self.se = [0]

                self.ee[0], self.se[0] = generatePoints(sys, curvType, curvC, curvPts)

                for i in range(len(sys.grp)):
                    self.vC0[sys.grp[i]] = Er[i]
                    ee, se = generatePoints(sys, curvType, Er[i]*curvC, curvPts)

                    self.ee.append(ee)
                    self.se.append(se)
            else:
                self.ee, self.se = generatePoints(sys, curvType, curvC, curvPts)

        elif problemType == "LinearFEM":
            self.vE = np.ones(sys.nEl)

            if testVal == None:
                self.testVal = 'XXYYSS'
            else:
                self.testVal = testVal

            if forceType == None:
                self.forceType = 'displacement'
            else:
                self.forceType = forceType

            if forceVal == None:    
                self.forceVal = 0.1
            else:
                self.forceVal = forceVal

            if steps == None:
                self.steps = 50
            else:
                self.steps = steps

            if sys.problemType == "PlaneStress":
                self.E = CPlaneStress(E, 0.3)
            else:
                self.E = E

            if len(sys.grp) > 1:
                for i in range(len(sys.grp)):
                    self.vE[sys.grp[i]] = Er[i]


        elif problemType == "ViscoElasticFEM":
            self.vE = np.ones(sys.nEl)
            self.dt = dt
            self.E0 = CData[0]
            self.Ei = []
            self.ldi = []

            for i in range((len(CData) - 1) / 2):
                self.Ei.append(CData[2 * i + 1])
                self.ldi.append(CData[2 * i + 2])

            if testVal == None:
                self.testVal = 'Y'
            else:
                self.testVal = testVal

            if forceType == None:
                self.forceType = 'displacement'
            else:
                self.forceType = forceType

            if steps == None:
                self.steps = 1
            else:
                self.steps = steps

            if len(sys.grp) > 1:
                for i in range(len(sys.grp)):
                    self.vE[sys.grp[i]] = Er[i]


        elif problemType == "NonLinearFEM":
            self.vE = np.ones(sys.nEl)

            if testVal == None:
                self.testVal = 'XXYYSS'
            else:
                self.testVal = testVal

            if forceType == None:
                self.forceType = 'displacement'
            else:
                self.forceType = forceType

            if forceVal == None:    
                self.forceVal = 0.1
            else:
                self.forceVal = forceVal

            if steps == None:
                self.steps = 50
            else:
                self.steps = steps

            if len(sys.grp) > 1:
                for i in range(len(sys.grp)):
                    self.vE[sys.grp[i]] = Er[i]

            self.E, self.dE = parseConstitutive(CData, sys.problemType)

            
