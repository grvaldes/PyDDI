import numpy as np
from scipy import sparse
from functions.elements import *

class System:

    def __init__(self, problemType, elementType, materialType, meshFile, nGaussPoints = 1, verbose = True):
        self.problemType = problemType
        self.elementType = elementType
        self.materialType = materialType
        self.meshFile = meshFile
        self.nGaussPoints = nGaussPoints
        self.verbose = verbose

        assert type(meshFile) is str, "meshFile should be the name of a file in 'meshes/'"
        # LOAD MESH TODO
        msh = {}

        if verbose:
            print("Generating system object based on mesh {meshFile}.")
            print("The mesh is a {problemType} problem with {elementType} elements. The material is {materialType}.\n")

        if problemType == "PlaneStress" or problemType == "2DTruss":
            self.nDim = 2
            self.X = msh["POS"][:,:self.nDim]

            if problemType == "PlaneStress":
                self.comp == 3
            else:
                self.comp == 1
        elif problemType == "3DTruss":
            self.nDim = 3
            self.X = msh["POS"][:,:self.nDim]
            self.comp = 1
        else:
            raise Exception("Problem type not defined.")

        if elementType == "Triangle":
            self.T = msh["TRIANGLES"][:,:-1]
            self.grp = msh["TRIANGLES"][:,-1]
            self.ngh = msh["NEIGH"]

            self.W, Dx, Dy, DxW, DyW, _, _ = FeMatricesTri(self.X, self.T, "", self.nGaussPoints)

            self.B = sparse.csr_matrix(np.vstack((np.hstack((Dx,np.zeros(Dy.shape))),np.hstack((np.zeros(Dx.shape),Dy)),np.hstack((0.5*Dy,0.5*Dx)))))
            self.BW = sparse.csr_matrix(np.transpose(np.vstack((np.hstack((DxW,np.zeros(DyW.shape))),np.hstack((np.zeros(DxW.shape),DyW)),np.hstack((0.5*DyW,0.5*DxW))))))
        elif elementType == "Quad":
            self.T = msh["QUADS"][:,:-1]
            self.grp = msh["QUADS"][:,-1]
            self.ngh = msh["NEIGH"]

            self.W, Dx, Dy, DxW, DyW, _, _ = FeMatricesQuad(self.X, self.T, "", self.nGaussPoints)

            self.B = sparse.csr_matrix(np.vstack((np.hstack((Dx,np.zeros(Dy.shape))),np.hstack((np.zeros(Dx.shape),Dy)),np.hstack((0.5*Dy,0.5*Dx)))))
            self.BW = sparse.csr_matrix(np.transpose(np.vstack((np.hstack((DxW,np.zeros(DyW.shape))),np.hstack((np.zeros(DxW.shape),DyW)),np.hstack((0.5*DyW,0.5*DxW))))))
        elif elementType == "Bar":
            self.T = msh["LINES"][:,:-1]
            self.grp = msh["LINES"][:,-1]

            self.W, self.B, _ = FeMatricesBarTruss(self.X, self.T, np.ones(self.T.shape[0]))

            self.BW = np.transpose(self.B) @ self.W
        else:
            raise Exception("Element type not defined.")

        self.BND_s = np.nonzero(self.X[:,1] == np.min(self.X[:,1]))
        self.BND_n = np.nonzero(self.X[:,1] == np.max(self.X[:,1]))
        self.BND_e = np.nonzero(self.X[:,0] == np.min(self.X[:,0]))
        self.BND_w = np.nonzero(self.X[:,0] == np.max(self.X[:,0]))
        
        self.nEl = self.T.shape[0]
        self.nNodEl = self.T.shape[1]
        self.nNod = self.X.shape[0]

        self.nDof = self.nNod * self.nDim
        self.nIP = self.W.shape[0] / self.nEp
        self.ntIP = self.nEl * self.nIP

        if np.max(self.grp) == 0:
            self.grp = (np.nonzero(self.grp == 0))
        else:
            grp = ()
            for n in range(np.max(self.grp)):
                grp[n] = np.nonzero(self.grp == n)

                # MISSING NIP > 1
            
            self.grp = grp

        if verbose:
            print("    There are {len(self.grp)} materials in the mesh.")
            print("    # elements: {self.nEl}.")
            print("    # nodes: {self.nNod}.")
            print("    # integration points per element: {self.nIP}.\n")

        

