import numpy as np

# @func CPlaneStress: computes the D tensor for the elastic plane stress case.    
# @Input E: Young's modulus
# @Input nu: Poisson ratio
# @Output D: D tensor (np.array)
def CPlaneStress(E,nu = 0.3):
    A = E / (1 - nu ** 2)
    B = A * nu
    C = E / (2 * (1 + nu))

    return np.array([[A,B,0],[B,A,0],[0,0,2*C]])


# @func 
# @Input 
# @Input 
# @Output 
def simpleKmeans(X,k,maxiter = 100,rand = True,ie = None):
    n = X.shape[0]
    ind = np.zeros(k)

    if rand:
        C, ind = datasample(X,1,1) #TODO
        minDist = np.inf * np.ones(n)

    for i in range(maxiter):
        A = 0
        C = A @ X
        ns = accumarray()
        C = C / ns
        KDTS = KDTreeSearcher(C)
        ie_n = knnsearch()

        if ie == ie_n:
            break

    return ie, C


# @func 
# @Input 
# @Input 
# @Output 
def nestedKmeans(X,kn,maxiter = 100,rand = True,ie = None):
    ie, C = simpleKmeans(X,kn[0],maxiter,rand,ie)

    kn = kn[1:]

    for m in range(kn.size):
        C_n = np.array([])
        ie_n = np.zeros(X.shape[0])

        for i in range(C.shape[0]):
            mask = np.nonzero(ie == i)
            t_ie, t_C = simpleKmeans(X[mask,:],kn[0],maxiter,rand,ie)

            ie_n[mask] = t_ie + C_n.shape[0]
            C_n = np.vstack(C_n,t_C)

        kn = kn[1:]
        ie = ie_n
        C = C_n

    return ie, C
    
    
# @func 
# @Input 
# @Input 
# @Output 
def myDataSample(X,prob): 
    cumProb = cumsum(prob)
    a = np.random.rand(1)
    pos = find(a < cumProb,1,'first')
    x = X[pos,:]

    return x, pos


# @func 
# @Input 
# @Input 
# @Output 
def triangleArea(X,T):
    A = np.zeros(T.shape[0])

    for e in range(T.shape[0]):
        Xe = X[T[e,:],:]
        A[e] = 0.5*np.linalg.det(np.array([Xe[:,0],X[:,1],np.ones(3,1)]))

    return A


# @func 
# @Input 
# @Input 
# @Output 
def doubleUnique(A,rows = True):
    if not rows:
        A = A.reshape(-1)

    ic = np.zeros(A.shape[0])
    id = np.zeros(A.shape[0])

    for i in range(A.shape[0]-1):
        for j in range(i+1,A.shape[0]):
            if id[j] == 0 and (A[i,:] == A[j,:] or A[i,:] == A[j,::-1]):
                ic[i] = i
                id[j] = j

    ia = np.nonzero(id == 0)
    C = A[ia,:]

    return ia, C


# @func 
# @Input 
# @Input 
# @Output 
def findNearest():
    return 0


# @func 
# @Input 
# @Input 
# @Output 
def generatePoints():
    return 0


# @func 
# @Input 
# @Input 
# @Output 
def generatePoints2DPS():
    return 0


# @func 
# @Input 
# @Input 
# @Output 
def normC(eps,sig,C,byElem = True):
    if C is tuple:
        C = C(1)

    v = np.array(sig.shape[0])

    if C.shape[0] == 1:
        assert C.shape[0] == C.shape[1], "C tensor is not square."
        assert C.shape[0] == eps.shape[1], "Strain values are not scalar."
        assert C.shape[0] == sig.shape[1], "Stress values are not scalar."
        assert eps.shape[0] == sig.shape[0], "Amount of points is not the same for strain and stress."

        for i in range(eps.shape[0]):
            v[i] = 0.5 * (eps[i] * C * eps[i] + sig[i] * (1 / C) * sig[i])

    elif C.shape[0] == 3:
        assert C.shape[0] == C.shape[1], "C tensor is not square."
        assert C.shape[0] == eps.shape[1], "Strain values are not vectors."
        assert C.shape[0] == sig.shape[1], "Stress values are not vectors."
        assert eps.shape[0] == sig.shape[0], "Amount of points is not the same for strain and stress."

        for i in range(eps.shape[0]):
            v[i] = 0.5 * (eps[i,:] @ (C @ eps[i,:].T) + sig[i,:] @ np.linalg.solve(C,sig[i,:]))

    if byElem:
        return v
    else:
        return np.sum(v)


# @func 
# @Input 
# @Input 
# @Output 
def parseConstitutive(data,typ):
    if typ == '2DTruss' or typ == '3DTruss':
        if 'linear' == data(0):
            C = lambda x : data(1) * x
            dC = lambda x : data(1) + 0 * x

        elif 'bilinear' == data(0): #TODO
            C = lambda x : np.multiply((data(1)(1)) * x,(np.abs(x) <= data(1)(3))) + np.multiply((np.sign(x) * data(1)(1) * data(1)(3) + data(1)(2) * (x - np.sign(x) * data(1)(3))),(np.abs(x) > data(1)(3)))
            dC = lambda x : (data(1)(1)) * (np.abs(x) <= data(1)(3)) + (data(1)(2)) * (np.abs(x) > data(1)(3))

        elif 'cubic' == data(0):
            C = lambda x : data(1)[0] * x ** 3 + data(1)[1] * x ** 2 + data(1)[2] * x
            dC = lambda x : data(1)[0] * 3 * x ** 2 + data(1)[1] * 2 * x + data(1)[2]

    elif typ == 'PlaneStress':
        if 'linear' == data(0):
            E = data(1)(1)
            nu = data(1)(2)
            I0 = np.eye(3)
            m = np.array([[1,1,0]]).T

            C = lambda x,y,z : np.transpose((E / (1 - nu ** 2) * ((1 - nu) * I0 + nu * (m @ m.T)) @ np.array([[x,y,z]]).T))
            dC = lambda x,y,z : E / (1 - nu ** 2) * ((1 - nu) * I0 + nu * (m @ m.T))
        
        elif 'cubic' == data(0): #TODO
            G = data(1)[0]
            a = data(1)[1]
            I0 = np.eye(3)
            m = np.array([[1,1,0]]).T

            #             C  = @(x,y,z) ((G*I0 - (m*m'))*[x y z]' + G*a*I0*[x.^3+z.^2.*(2*x+y) y.^3+z.^2.*(2*y+x) z.^3+z.*(x+x.*y+y)]' - a*(m*m'*[x y z]').^3)';
            C = lambda x,y,z : np.transpose(((G * I0 - (m * np.transpose(m))) * np.transpose(np.array([x,y,z])) + G * a * I0 * (np.transpose(np.array([x,y,z]))) ** 3 - a * (m * np.transpose(m) * np.transpose(np.array([x,y,z]))) ** 3))
            #             dC = @(x,y,z) (G*I0 + 3*G*a*[x^2+z^2 z*(x+y) 0;z*(x+y) y^2+z^2 0; 0 0 0]')';
            dC = lambda x,y,z : (G * I0 - (m * np.transpose(m))) + 3 * G * a * I0 * np.array([[x ** 2,z ** 2,0],[z ** 2,y ** 2,0],[0,0,0]]) - 3 * a * (np.array([[x + y,0,0],[0,x + y,0],[0,0,x + y]])) ** 2

    return C, dC


# @func 
# @Input 
# @Input 
# @Output 
def pointPrunner():
    return 0