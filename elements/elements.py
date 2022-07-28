import numpy as np
import numpy.matlib as matlib
from scipy import sparse

# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesBarTruss(X,EDGES,S):
    Nnodes = X.shape[0]
    Nelements = EDGES.shape[0]
    dim = X.shape[1]
    last = X[EDGES[:,1],:]
    first = X[EDGES[:,0],:]
    
    T = last - first
    L = np.sqrt(np.sum(T ** 2, 1))
    T = T / (L * np.ones(dim))
    tmp = T / (L * np.ones(dim))    

    idx_i = matlib.repmat(np.arange(Nelements).T,2*dim,1)
    idx_j = np.zeros((EDGES.size,dim))
    idx_j[:,1] = EDGES.reshape(-1, order = 'F')
    
    for d in range(1,dim):
        idx_j[:,d] =  EDGES.reshape(-1, order = 'F') + (d - 1) * Nnodes
    
    vals = np.hstack(-tmp,tmp)
    vals = vals.reshape(-1, order='F')

    B = sparse.csr_matrix((vals,(idx_i,idx_j)),shape = (Nelements,Nnodes*dim))
    W = sparse.diags(S*L,0,shape = (Nelements,Nelements))
    
    return W, B, T


# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesTri(XY,T20,order = 'T3',param = 1):
    Xe,We = hammerPointsTriangle(param)
    We = We.reshape(-1,order = 'F')
    Nipe = We.size

    if order == 'T3':
        phi = lambda xi : np.array([-xi[:,1] - xi[:,0] + 1.0, xi[:,0], xi[:,1]])
        dxi = lambda xi : np.transpose(np.array([-np.ones((Nipe,1)), np.ones((Nipe,1)), np.zeros((Nipe,1))]))
        deta = lambda xi : np.transpose(np.array([-np.ones((Nipe,1)), np.zeros((Nipe,1)), np.ones((Nipe,1))]))

    elif order == 'T6':
        phi = lambda xi : np.array([2 * (xi[:,0] + xi[:,1]) ** 2 - 3 * xi[:,1] - 3 * xi[:,0] + 1.0,\
            (2 * xi[:,0] - 1) * xi[:,0],\
                (2 * xi[:,1] - 1) * xi[:,1],\
                    4 * (xi[:,0] - xi[:,0] ** 2 - xi[:,0] * xi[:,1]),\
                        4 * xi[:,0] * xi[:,1],\
                            4 * (xi[:,1] - xi[:,1] ** 2 - xi[:,1] * xi[:,0])])
        dxi = lambda xi : np.transpose(np.array([-3 + 4 * xi[:,0] + 4 * xi[:,1],\
            4 * xi[:,0] - 1,0,\
                0 * xi[:,0],\
                    4 * (1 - xi[:,1] - 2 * xi[:,0]),\
                        4 * xi[:,1],\
                            -4 * xi[:,1]]))
        deta = lambda xi : np.transpose(np.array([-3 + 4 * xi[:,0] + 4 * xi[:,1],\
            0 * xi[:,1],\
                4 * xi[:,1] - 1,\
                    -4 * xi[:,0],\
                        4 * xi[:,0],\
                            4 * (1 - xi[:,0] - 2 * xi[:,1])]))

    PHIe = phi(Xe)
    dxie = dxi(Xe)
    detae = deta(Xe)

    NNodes = XY.shape[0]
    NElem = T20.shape[0]

    Ev = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    Dx = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    Dy = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    DxW = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    DyW = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    IFF = np.zeros((NNodes,1))
    W = np.zeros((NElem * Nipe,1))

    for e in range(NElem):
        loc_lines = (e * Nipe) + np.arange(Nipe)
        connec = T20[e,:]
        XY_loc = XY[connec,:]

        for n in range(Nipe):
            J = np.transpose(XY_loc) @ np.hstack(dxie[:,n],detae[:,n])
            dphidX = np.hstack(dxie[:,n],detae[:,n]) / J
            
            W[loc_lines[n]] = np.linalg.det(J) * We[n]
            Dx[loc_lines[n],:] = np.transpose(dphidX[:,0])
            Dy[loc_lines[n],:] = np.transpose(dphidX[:,1])
            DxW[loc_lines[n],:] = np.transpose(dphidX[:,0]) * W[loc_lines[n]]
            DyW[loc_lines[n],:] = np.transpose(dphidX[:,1]) * W[loc_lines[n]]
        
        
        Ev[loc_lines,:] = PHIe
        IFF[connec] = IFF[connec] + np.transpose(W[loc_lines].T @ Ev[loc_lines,:])
    

    tmp = np.repeat(T20.T,Nipe,1).T

    W = sparse.diags(W,0,shape = (NElem * Nipe,NElem * Nipe))
    Dx = sparse.csr_matrix((Dx, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    Dy = sparse.csr_matrix((Dy, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    DxW = sparse.csr_matrix((DxW, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    DyW = sparse.csr_matrix((DyW, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    Ev = sparse.csr_matrix((Ev, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))    

    return W, Dx, Dy, DxW, DyW, Ev, IFF


# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesQuad(XY,T20,order = 'Q4',param = 1):
    Xe,We = hammerPointsTriangle(param)
    We = We.reshape(-1,order = 'F')
    Nipe = We.size

    if order == 'Q4':
        phi = lambda xi : np.array([0.25 * (1 - xi[:,0]) * (1 - xi[:,1]), 0.25 * (1 + xi[:,0]) * (1 - xi[:,1]),\
            0.25 * (1 + xi[:,0]) * (1 + xi[:,1]), 0.25 * (1 - xi[:,0]) * (1 + xi[:,1])])
        dxi = lambda xi : np.transpose(np.array([- 0.25 * (1 - xi[:,1]), 0.25 * (1 - xi[:,1]),\
            0.25 * (1 + xi[:,1]), -0.25 * (1 + xi[:,1])]))
        deta = lambda xi : np.transpose(np.array([-0.25 * (1 - xi[:,0]), -0.25 * (1 + xi[:,0]),\
            0.25 * (1 + xi[:,0]), 0.25 * (1 - xi[:,0])]))

    elif order == 'Q8':
        phi = lambda xi : np.array([(1 - xi[:,0]) * (1 - xi[:,1]) * (-xi[:,0] - xi[:,1] - 1) / 4,\
                (1 + xi[:,0]) * (1 - xi[:,1]) * (xi[:,0] - xi[:,1] - 1) / 4,\
                    (1 + xi[:,0]) * (1 + xi[:,1]) * (xi[:,0] + xi[:,1] - 1) / 4,\
                        (1 - xi[:,0]) * (1 + xi[:,1]) * (-xi[:,0] + xi[:,1] - 1) / 4,\
                            (1 - xi[:,1]) * (1 - xi[:,0] ** 2) / 2,\
                                (1 + xi[:,0]) * (1 - xi[:,1] ** 2) / 2,\
                                    (1 + xi[:,1]) * (1 - xi[:,0] ** 2) / 2,\
                                        (1 - xi[:,0]) * (1 - xi[:,1] ** 2) / 2])
        dxi = lambda xi : np.transpose(np.array([-(1 - xi[:,1]) * (-2 * xi[:,0] - xi[:,1]) / 4,\
            (1 - xi[:,1]) * (2 * xi[:,0] - xi[:,1]) / 4,\
                (xi[:,1] + 1) * (2 * xi[:,0] + xi[:,1]) / 4,\
                    -(xi[:,1] + 1) * (-2 * xi[:,0] + xi[:,1]) / 4,\
                        - xi[:,0] * (1 - xi[:,1]),\
                            -(xi[:,1] ** 2 - 1) / 2,\
                                -xi[:,0] * (xi[:,1] + 1),\
                                    (xi[:,1] ** 2 - 1) / 2]))
        deta = lambda xi : np.transpose(np.array([-(1 - xi[:,0]) * (-xi[:,0] - 2 * xi[:,1]) / 4,\
            -(xi[:,0] + 1) * (xi[:,0] - 2 * xi[:,1]) / 4,\
                (xi[:,0] + 1) * (xi[:,0] + 2 * xi[:,1]) / 4,\
                    (1 - xi[:,0]) * (-xi[:,0] + 2 * xi[:,1]) / 4,\
                        (xi[:,0] ** 2 - 1) / 2,\
                            -xi[:,1] * (xi[:,0] + 1),\
                                -(xi[:,0] ** 2 - 1) / 2,\
                                    -xi[:,1] * (1 - xi[:,0])]))

    elif order == 'Q9':
        phi = lambda xi : np.array([0.25 * (xi[:,0] ** 2 - xi[:,0]) * (xi[:,1] ** 2 - xi[:,1]),\
            0.25 * (xi[:,0] ** 2 + xi[:,0]) * (xi[:,1] ** 2 - xi[:,1]),\
                0.25 * (xi[:,0] ** 2 + xi[:,0]) * (xi[:,1] ** 2 + xi[:,1]),\
                    0.25 * (xi[:,0] ** 2 - xi[:,0]) * (xi[:,1] ** 2 + xi[:,1]),\
                        0.5 * (xi[:,1] ** 2 + xi[:,1]) * (1 - xi[:,0] ** 2),\
                            0.5 * (xi[:,0] ** 2 - xi[:,0]) * (1 - xi[:,1] ** 2),\
                                0.5 * (xi[:,1] ** 2 - xi[:,1]) * (1 - xi[:,0] ** 2),\
                                    0.5 * (xi[:,0] ** 2 + xi[:,0]) * (1 - xi[:,1] ** 2),\
                                        (1 - xi[:,0] ** 2) * (1 - xi[:,1] ** 2)])
        dxi = lambda xi : np.transpose(np.array([0.25 * (2 * xi[:,0] - 1) * (xi[:,1] ** 2 - xi[:,1]),\
            0.25 * (2 * xi[:,0] + 1) * (xi[:,1] ** 2 - xi[:,1]),\
                0.25 * (2 * xi[:,0] + 1) * (xi[:,1] ** 2 + xi[:,1]),\
                    0.25 * (2 * xi[:,0] - 1) * (xi[:,1] ** 2 + xi[:,1]),\
                        -(xi[:,1] ** 2 + xi[:,1]) * xi[:,0],\
                            0.5 * (2 * xi[:,0] - 1) * (1 - xi[:,1] ** 2),\
                                -(xi[:,1] ** 2 - xi[:,1]) * xi[:,0],\
                                        0.5 * (2 * xi[:,0] + 1) * (1 - xi[:,1] ** 2),\
                                        -2 * xi[:,0] * (1 - xi[:,1] ** 2)]))
        deta = lambda xi : np.transpose(np.array([0.25 * (xi[:,0] ** 2 - xi[:,0]) * (2 * xi[:,1] - 1),\
            0.25 * (xi[:,0] ** 2 + xi[:,0]) * (2 * xi[:,1] - 1),\
                0.25 * (xi[:,0] ** 2 + xi[:,0]) * (2 * xi[:,1] + 1),\
                    0.25 * (xi[:,0] ** 2 - xi[:,0]) * (2 * xi[:,1] + 1),\
                            0.5 * (2 * xi[:,1] + 1) * (1 - xi[:,0] ** 2),\
                            -(xi[:,0] ** 2 - xi[:,0]) * xi[:,1],\
                                    0.5 * (2 * xi[:,1] - 1) * (1 - xi[:,0] ** 2),\
                                    -(xi[:,0] ** 2 + xi[:,0]) * xi[:,1],\
                                        -2 * xi[:,1] * (1 - xi[:,0] ** 2)]))


    PHIe = phi(Xe)
    dxie = dxi(Xe)
    detae = deta(Xe)

    NNodes = XY.shape[0]
    NElem = T20.shape[0]

    Ev = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    Dx = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    Dy = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    DxW = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    DyW = np.zeros((Nipe * T20.shape[0],T20.shape[1]))
    IFF = np.zeros((NNodes,1))
    W = np.zeros((NElem * Nipe,1))

    for e in range(NElem):
        loc_lines = (e * Nipe) + np.arange(Nipe)
        connec = T20[e,:]
        XY_loc = XY[connec,:]

        for n in range(Nipe):
            J = np.transpose(XY_loc) @ np.hstack(dxie[:,n],detae[:,n])
            dphidX = np.hstack(dxie[:,n],detae[:,n]) / J
            
            W[loc_lines[n]] = np.linalg.det(J) * We[n]
            Dx[loc_lines[n],:] = np.transpose(dphidX[:,0])
            Dy[loc_lines[n],:] = np.transpose(dphidX[:,1])
            DxW[loc_lines[n],:] = np.transpose(dphidX[:,0]) * W[loc_lines[n]]
            DyW[loc_lines[n],:] = np.transpose(dphidX[:,1]) * W[loc_lines[n]]
        
        
        Ev[loc_lines,:] = PHIe
        IFF[connec] = IFF[connec] + np.transpose(W[loc_lines].T @ Ev[loc_lines,:])
    

    tmp = np.repeat(T20.T,Nipe,1).T

    W = sparse.diags(W,0,shape = (NElem * Nipe,NElem * Nipe))
    Dx = sparse.csr_matrix((Dx, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    Dy = sparse.csr_matrix((Dy, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    DxW = sparse.csr_matrix((DxW, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    DyW = sparse.csr_matrix((DyW, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))
    Ev = sparse.csr_matrix((Ev, (np.repeat(np.arange(NElem * Nipe),T20.shape[1]),tmp)),shape = (NElem * Nipe,NNodes))

    return W, Dx, Dy, DxW, DyW, Ev, IFF



# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesEdge(X,T10,param = 1):
    Xe,We = gaussPoints(param)
    We = We.reshape(-1,order = 'F')
    Nipe = We.size

    phi = lambda xi : np.array([-xi * 1/2 + 1/2, xi * 1/2 + 1/2])
    dphidxi = np.array([[- 1.0 / 2.0],[1.0 / 2.0]])

    PHIe = phi(Xe)

    NNodes = X.shape[0]
    NElem = T10.shape[0]

    Ev = np.zeros((Nipe * T10.shape[0],2))
    Dx = np.zeros((Nipe * T10.shape[0],2))
    IFF = np.zeros((NNodes,1))
    W = np.zeros((NElem * Nipe,1))

    for e in range(NElem):
        loc_lines = (e * Nipe) + np.arange(Nipe)
        connec = T10[e,:]
        XY_loc = X[connec,:]

        J = np.transpose(XY_loc) @ dphidxi
        nJ = np.linalg.norm(J)
        W[loc_lines] = nJ * We
        dphidX = dphidxi / nJ
        
        Dx[loc_lines,:] = matlib.repmat(dphidX[:,0].T,Nipe,1)
        Ev[loc_lines,:] = PHIe
        IFF[connec] = IFF[connec] + nJ

    tmp = np.repeat(T10.T,Nipe,1).T

    W = sparse.diags(W,0,shape = (NElem * Nipe,NElem * Nipe))
    Dx = sparse.csr_matrix((Dx, (np.repeat(np.arange(NElem * Nipe),2),tmp)),shape = (NElem * Nipe,NNodes))
    Ev = sparse.csr_matrix((Ev, (np.repeat(np.arange(NElem * Nipe),2),tmp)),shape = (NElem * Nipe,NNodes))

    return W,Dx,Ev,IFF



# @func 
# @Input 
# @Input 
# @Output 
def hammerPointsTriangle(param):
    if 1 == param:
        Xe = np.transpose(np.array([[1],[1]])) / 3
        W = 0.5
    elif 3 == param:
        Xe = np.transpose(np.array([[0.5,0.5,0],[0,0.5,0.5]]))
        W = np.array([1,1,1]) / 6
    elif 4 == param:
        Xe = np.transpose(np.array([[1 / 3,1 / 5,3 / 5,1 / 5],[1 / 3,3 / 5,1 / 5,1 / 5]]))
        W = np.array([-9/32, 25/96, 25/96, 25/96])
    elif 6 == param:
        Xe = np.transpose(np.array([[0.0915762135,0.816847573,0.0915762135,0.4459484909,0.1081030182,0.4459484909],[0.0915762135,0.0915762135,0.816847573,0.4459484909,0.4459484909,0.1081030182]]))
        W = 0.5 * np.array([0.1099517437,0.1099517437,0.1099517437,0.2233815897,0.2233815897,0.2233815897])
    else:
        raise Exception('Unsupported value of the parameter')

    return Xe, W


# @func 
# @Input 
# @Input 
# @Output 
def GaussPointsQuad(param):
    if 1 == param:
        Xe = 0
        W = 2
    elif 4 == param:
        Xe = np.transpose(np.array([-1/np.sqrt(3), 1/np.sqrt(3)]))
        W = np.array([1,1])
    elif 9 == param:
        Xe = np.transpose(np.array([-np.sqrt(15)/5, 0, np.sqrt(15)/5]))
        W = np.array([5/9, 8/9, 5/9])
    elif 16 == param:
        Xe = np.transpose(np.array([-0.8611363115940525,-0.3399810435848563,0.3399810435848563,0.8611363115940525]))
        W = np.array([0.3478548451374541,0.6521451548625461,0.6521451548625461,0.3478548451374541])
    else:
        raise Exception('Unsupported value of the parameter')
    
    zgp,eta = np.meshgrid(Xe,Xe)
    Xe = np.array([zgp,eta])
    W = np.transpose(W) * W
    W = np.transpose(W)

    return Xe, W


# @func 
# @Input 
# @Input 
# @Output 
def gaussPoints(dim,NP): 
    if 1 == NP:
        x = 0
        w = 2
    elif 2 == NP:
        x = np.array([np.sqrt(3), -np.sqrt(3)]) / 3
        w = np.array([1,1])
    elif 3 == NP:
        x = np.array([np.sqrt(3/5), 0.0, -np.sqrt(3/5)])
        w = np.array([5/9, 8/9, 5/9])
    
    if 1 == dim:
        Xe = x
        W = w
    elif 2 == dim:
        r = np.arange(x.size)
        a = matlib.repmat(r,1,x.size)
        b = np.repeat(r,x.size)
        Xe = np.vstack(x[a],x[b])
        W = w[a] * w[b]
    elif 3 == dim:
        a = matlib.repmat(np.arange(x.size),1,x.size ** 2)
        b = np.repeat(np.arange(x.size ** 2),x.size)
        c = np
        Xe = np.vstack(x[a],x[b],x[c])
        W = w[a] * w[b] * w[c]
    
    Xe = np.transpose(Xe)

    return Xe, W