import time
import numpy as np
import scipy.linalg
from scipy import sparse
from classes import DataObject, MaterialDatabase, FemStressProjector, DDIObject


def solveSystemLinearElasticFEM(sys, prop):

    fp = np.full(sys.nDof, 0)
    up = np.full(sys.nDof, 0)
    rp = np.full(sys.nDof, False)

    for p in range(prop.F.shape[0]):
        fp[prop.F[p,0]::sys.Nnod] = prop.F[p,1:2+sys.nDim]
    for p in range(prop.rx.shape[0]):
        rp[prop.rx[p,0]::sys.Nnod] = prop.rx[p,1:2+sys.nDim]
    for p in range(prop.uf.shape[0]):
        up[prop.uf[p,0]::sys.Nnod] = prop.uf[p,1:2+sys.nDim]

    ud = np.nonzero(up)

    D = np.kron(prop.E, sparse.spdiags())

    sol = DataObject(sys, prop)

    sol.K = sys.BW @ D @ sys.B
    sol.f = fp - sol.K @ up
    sol.r = rp.copy()
    sol.u = up.copy()

    sol.fp = fp.copy()
    sol.rp = rp.copy()
    sol.up = up.copy()

    Kr = sol.K[~rp and ~ud,~rp and ~ud]
    fr = sol.f[~rp and ~ud]

    sol.u[~rp and ~ud] = np.linalg.solve(Kr, fr)
    sol.r[rp] = sol.K[rp,:] @ sol.u

    sol.eps = (sys.B @ sol.u).reshape((sys.ntIP,-1))
    sol.sig = (D @ sol.eps).reshape((sys.ntIP,-1))

    return sol
    

def solveSystemLinearViscoelasticFEM(sys, prop, sols = [], gam = 1, ths = 1e-8):
    dt = prop.dt
    ldi = prop.ldi
    sol = []
    
    mu = []
    alf = []
    Di = []
    Ki = []

    D0 = np.kron(prop.E0,sparse.spdiags())
    K0 = sys.BW @ D0 @ sys.B
    KT = K0.copy()

    for i in range(len(ldi)):
        mu.append(1 / (1 + gam * dt / ldi[i]))
        alf.append(1 - (1 - gam) * dt / ldi[i])

        Di.append(np.kron(prop.Ei[i],sparse.spdiags()))
        Ki.append(sys.BW @ Di[i] @ sys.B)

        KT += mu[i] * Ki[i]

    fn1 = np.full(sys.nDof, 0)
    pn1 = np.full(sys.nDof, 0)
    rn1 = np.full(sys.nDof, False)

    qn = []
    qn1 = []

    if sols == []:
        un = np.zeros(sys.nDof)
        en = np.zeros((sys.ntIP, sys.comp))

        for i in range(len(ldi)):
            qn[i] = np.zeros((sys.ntIP, sys.comp))
            qn1[i] = np.zeros((sys.ntIP, sys.comp))
    else:
        un = sols[-1].u
        en = sols[-1].eps

        for i in range(len(ldi)):
            qn[i] = sols[-1].qps[i]
            qn1[i] = np.zeros((sys.ntIP, sys.comp))

    for n in range(prop.nX):
        sol.append(DataObject(sys, prop))

        for p in range(prop.F[n].shape[0]):
            fn1[prop.F[n][p,0]::sys.Nnod] = prop.F[n][p,1:2+sys.nDim]
        for p in range(prop.rx[n].shape[0]):
            rn1[prop.rx[n][p,0]::sys.Nnod] = prop.rx[n][p,1:2+sys.nDim]
        for p in range(prop.uf[n].shape[0]):
            pn1[prop.uf[n][p,0]::sys.Nnod] = prop.uf[n][p,1:2+sys.nDim]

        pd1 = np.nonzero(pn1)

        Fs1 = fn1 - KT @ pn1

        for i in range(len(ldi)):
            Fs1 -= mu[i] * alf[i] * (sys.BW @ Di[i] @ qn[i].ravel()) + mu[i] * (Ki[i] @ un)

        un1 = pn1.copy()
        un1[~rn1 and ~pd1] = np.linalg.solve(KT[~rn1 and ~pd1, ~rn1 and ~pd1], Fs1[~rn1 and ~pd1])

        en1 = (sys.B @ un1).reshape((sys.ntIP, -1))
        sn1 = (D0 @ en1.ravel()).reshape((sys.ntIP, -1))

        for i in range(len(ldi)):
            qn1[i] = mu[i] * (alf[i] * qn[i] + en1 - en)
            sn1 += (Di[i] @ qn1[i].ravel()).reshape((sys.ntIP, -1))

        sol[n].eps = en1.copy()
        sol[n].sig = sn1.copy()
        sol[n].qps = []

        for i in range(len(ldi)):
            sol[n].qps.append(qn1[i])

        sol[n].f = Fs1.copy()
        
        sol[n].up = pn1.copy()
        sol[n].fp = fn1.copy()
        sol[n].rp = rn1.copy()

        sol[n].K = KT.copy()
        sol[n].u = un1.copy()
        sol[n].r = sys.BW @ sn1.ravel()
        
        sol[n].deps = (en1 - en) / dt
        sol[n].qps = []

        for i in range(len(ldi)):
            sol[n].qps.append((qn1[i] - qn[i]) / dt)
        
        un = un1.copy()
        en = en1.copy()
        qn = qn1.copy()

    return sol

        
def solveSystemNonLinearElasticFEM(sys, prop, tolerance = 1e-8, maxIter = 100):
    sol = []

    fn1 = np.full(sys.nDof, 0)
    pn1 = np.full(sys.nDof, 0)
    rn = np.full(sys.nDof, False)

    un = np.zeros(sys.nDof)
    eps = np.zeros((sys.ntIP, sys.comp))

    for p in range(prop.rx.shape[0]):
        rn[prop.rx[p,0]::sys.Nnod] = prop.rx[p,1:2+sys.nDim]

    for n in range(prop.steps):
        sol.append(DataObject(sys, prop))

        sol[n].r = rn.copy()
        sol[n].rp = rn.copy()

    for n in range(prop.steps):
        for p in range(prop.F.shape[0]):
            fn1[prop.F[p,0]::sys.Nnod] = (n + 1) * prop.F[p,1:2+sys.nDim] / prop.steps
        for p in range(prop.uf.shape[0]):
            pn1[prop.uf[p,0]::sys.Nnod] = (n + 1) * prop.uf[p,1:2+sys.nDim] / prop.steps

        pd1 = np.nonzero(pn1)

        if n == 0:
            ui = pn1.copy()
        else:
            ui = un.copy()

        i = 0
        while i < maxIter:
            i += 1

            if sys.comp == 1:
                DT = sparse.spdiags()
            elif sys.comp == 3:
                DT = sparse.csr_matrix((3 * sys.ntIP, 3 * sys.ntIP))

                for e in range(sys.nEl):
                    DT[e::sys.ntIP,e::sys.ntIP] = prop.dE(eps[e,0], eps[e,1], eps[e,2]) * prop.vE[e]

            KT = sys.BW @ DT @ sys.B
            psi = fn1 - KT @ ui
            psi = psi[~rn and ~pd1]
            ui1 = pn1.copy()

            du1 = np.zeros(ui.shape)
            du1[~rn and ~pd1] = np.linalg.solve(KT[~rn and ~pd1, ~rn and ~pd1], psi)
            ui1[~rn and ~pd1] = ui[~rn and ~pd1] + du1[~rn and ~pd1]

            if i == 0:
                psi0 = psi.copy()
            
            if (np.linalg.norm(psi) / np.linalg.norm(psi0) < tolerance) or np.linalg.norm(psi0) == 0:
                break

            eps = (sys.B @ ui1).reshape((sys.ntIP,-1))
            ui = ui1.copy()

        un1 = ui1.copy()

        sol[n].up = pn1.copy()
        sol[n].fp = fn1.copy()

        sol[n].K = KT.copy()
        sol[n].u = un1.copy()
        sol[n].f = fn1 - KT @ pn1
        sol[n].r[rn] = KT[rn,:] @ sol[n].u
        sol[n].eps = (sys.B @ sol[n].u).reshape((sys.ntIP,-1))

        if sys.comp == 1:
            sol[n].sig = 0
        elif sys.comp == 3:
            sol[n].sig = prop.E(sol[n].eps[:,0], sol[n].eps[:,1], sol[n].eps[:,2]) * prop.vE

        un = un1.copy()

    return sol


def solveSystemDDI(sys, prop, sol, th=1, mxI=500, mxJ=500, EsTol=1e-6, SsTol=1e-6, rPCA=False, tPCA=0.9):

    eex = sol.ee
    sex = []
    sts = []
    ieM = []
    wxe = np.diag(sys.W)

    if rPCA:
        sexP = []

        for i in range(prop.nX):
            sexP.append(0)

    wDS = np.zeros(mxI)

    tic = time.time()

    if type(prop.C0) == tuple:
        C0 = prop.C0[0]

        for c in range(1,len(prop.C0)):
            C0 = scipy.linalg.block_diag(C0, prop.C0[c])
        
        mdb = MaterialDatabase(prop.es, prop.ss, C0, np.linalg.inv(prop.C0[0]))
    else:
        C0 = np.kron(prop.C0, np.eye(prop.es.shape[1] / sys.comp))

        mdb = MaterialDatabase(prop.es, prop.ss, C0, np.linalg.inv(prop.C0))

    for i in range(prop.nX):
        sex.append(0)
        ieM.append(0)

        sts.append(FemStressProjector(sys.B, sys.BW, prop.C0, np.ones(sys.W.shape), prop.rp[i], prop.rl[i]))
        _, sex[i] = sts[i].project(prop.fp[i], prop.fc[i])
        _, _, ieM[i], _ = mdb.project(eex[i], sex[i])

    sol.init_clock = (time.time() - tic)

    if sys.verbose:
        print("\nPerforming DDI Algorithm")
        print("Parameters used: ")
        print("    C =", prop.C0)
        print("    N* =", np.prod(prop.nStr), "\n")

    rlEs = 1
    whl = 0

    if rPCA:
        nv = []
        L = []

    tic = time.time()
    while whl < mxI and rlEs > EsTol:
        whl += 1

        if sys.verbose:
            print("Starting iteration #", whl)

        iwhl = 0
        rlSs = 1

        while iwhl < mxJ and rlSs > SsTol:
            if not rPCA:
                for i in range(prop.nX):
                    _, sex[i] = sts[i].project(ieM[i] @ mdb.fluxes, prop.fp[i], prop.fc[i])
                    mdb.accumFluxes(sex[i], ieM[i], wxe)
            else:
                for i in range(prop.nX):
                    _, sex[i] = sts[i].project(ieM[i] @ mdb.fluxes, prop.fp[i], prop.fc[i])
                    sexP[i] = sex[i].ravel()

                X, nv[whl], L[whl] = 0#PCAfiltering() #TODO

                for i in range(prop.nX):
                    sex[i] = X[:,i].reshape((-1, sys.comp))
                    mdb.accumFluxes(sex[i], ieM[i], wxe)

            rlSs = mdb.updateFluxes(th)

        if sys.verbose:
            if iwhl == mxJ:
                print("    Number of max subiterations reached. Continuing with last value.")
            else:
                print("    Number of subiterations for stress:", iwhl)

        for i in range(prop.nX):
            mdb.accumForces(eex[i], ieM[i], wxe)

        rlEs = mdb.updateForces(th)

        if sys.verbose:
            print("    Convergence coefficient for strain:", rlEs)

        wDSt = 0

        for i in range(prop.nX):
            _, _, ieM[i], D = mdb.project(eex[i], sex[i])
            wDSt += wxe @ D

        wDS[whl] = wDSt
        mask = mdb.pruneDatabase(ieM)

        for i in range(prop.nX):
            ieM = ieM[i][:,mask]

    wDS = wDS[:whl+1]

    if rPCA:
        nv = nv[:whl]
        L = L[:whl]

        sol.sep = sex.copy()
        sol.L = L.copy()
        sol.nv = nv.copy()

        for i in range(prop.nX):
            _, sex[i] = sts[i].project(ieM[i] @ mdb.fluxes, prop.fp[i], prop.fc[i])
            _, _, ieM[i], _ = mdb.project(eex[i], sex[i])

    sol.alg_clock = (time.time() - tic)

    if sys.verbose:
        if whl == mxI:
            print("Maximum amount of iterations reached. The problem did not converge to the expected tolerance.")
        else:
            print("Convergence achieved in", whl, "iterations.\n")

    sol.es = mdb.forces.copy()
    sol.ss = mdb.fluxes.copy()
    sol.se = sex.copy()

    sol.ieM = ieM.copy()
    sol.wDS = wDS.copy()
    sol.iter = whl

    return sol
