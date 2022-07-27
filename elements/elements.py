import numpy as np
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

    idx_i = np.matlib.repmat(np.arange(Nelements).T,2*dim,1)
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
        loc_lines = ((e - 1) * Nipe) + np.transpose((np.arange(1,Nipe+1)))
        #local connectivity
        connec = T20(e,:)
        #element nodes
        XY_loc = XY(connec,:)
        for n in np.arange(1,Nipe+1).reshape(-1):
            #Jacobian matrix
            J = np.transpose(XY_loc) * np.array([dxie(:,n),detae(:,n)])
            #shape function derivatives
            dphidX = np.array([dxie(:,n),detae(:,n)]) / J
            #quadrature weights
            W[loc_lines[n]] = det(J) * We(n)
            #assembly
            Dx[loc_lines[n],:] = np.transpose(dphidX(:,1))
            Dy[loc_lines[n],:] = np.transpose(dphidX(:,2))
            DxW[loc_lines[n],:] = np.transpose(dphidX(:,1)) * W(loc_lines(n))
            DyW[loc_lines[n],:] = np.transpose(dphidX(:,2)) * W(loc_lines(n))
        #assembly
        Ev[loc_lines,:] = PHIe
        #accumulating the integral of each shape function
        IFF[connec] = IFF(connec) + np.transpose((np.transpose(W(loc_lines)) * Ev(loc_lines,:)))
    
    #Change W to a diagonal matrix
    W = spdiags(W,0,NElem * Nipe,NElem * Nipe)
    #Change Dx,Dy and Ev to sparse matrices
    tmp = repelem(T20,Nipe,1)
    Dx = sparse(reshape(repelem(np.transpose((np.arange(1,(NElem * Nipe)+1))),1,T20.shape[2-1]),[],1),tmp,Dx,NElem * Nipe,NNodes)
    Dy = sparse(reshape(repelem(np.transpose((np.arange(1,(NElem * Nipe)+1))),1,T20.shape[2-1]),[],1),tmp,Dy,NElem * Nipe,NNodes)
    DxW = sparse(reshape(repelem(np.transpose((np.arange(1,(NElem * Nipe)+1))),1,T20.shape[2-1]),[],1),tmp,DxW,NElem * Nipe,NNodes)
    DyW = sparse(reshape(repelem(np.transpose((np.arange(1,(NElem * Nipe)+1))),1,T20.shape[2-1]),[],1),tmp,DyW,NElem * Nipe,NNodes)
    Ev = sparse(reshape(repelem(np.transpose((np.arange(1,(NElem * Nipe)+1))),1,T20.shape[2-1]),[],1),tmp,Ev,NElem * Nipe,NNodes)

    return W, Dx, Dy, DxW, DyW, Ev, IFF


# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesQuad(XY = None,T20 = None,order = None,param = None):

    return W, Dx, Dy, DxW, DyW, Ev, IFF



# @func 
# @Input 
# @Input 
# @Output 
def FeMatricesEdge(XY = None,T20 = None,order = None,param = None):

    return W,Dx,Ev,IFF



# @func 
# @Input 
# @Input 
# @Output 
def hammerPointsTriangle(param = 1):
    return Xe, W