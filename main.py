import numpy as np
from pyDDI.functions import *
from pyDDI.classes.SystemObject import SystemObject
from pyDDI.functions.algorithms import generateDataLinearElastic

from meshes.testElems3 import msh

mshT3 = {
    'POS' : np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]),
    'TRIANGLES' : np.array([[0, 1, 4, 0], [0, 4, 3, 0], [1, 2, 5, 0], [1, 5, 4, 0], 
                            [3, 4, 7, 1], [3, 7, 6, 1], [4, 5, 8, 1], [4, 8, 7, 1]]),
    'LINES' : np.array([[0, 1, 1], [1, 2, 1], [0, 3, 1], [1, 4, 1], [2, 5, 1], [0, 4, 1], [1, 5, 1], [3, 4, 1],
                        [4, 5, 1], [3, 6, 2], [4, 7, 2], [5, 8, 2], [3, 7, 2], [4, 8, 2], [6, 7, 2], [7, 8, 2]]),
    'NEIGH' : 0,
}

mshT6 = {
    'POS' : np.array([[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], 
                      [0, 0.5], [1, 0.5], [2, 0.5],
                      [0, 1], [0.5, 1], [1, 1], [1.5, 1], [2, 1], 
                      [0, 1.5], [1, 1.5], [2, 1.5],
                      [0, 2], [0.5, 2], [1, 2], [1.5, 2], [2, 2],
                      [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]),
    'TRIANGLES' : np.array([[0, 2, 10, 1, 6, 21, 0], [0, 10, 8, 21, 9, 5, 0], 
                            [2, 4, 12, 3, 7, 22, 0], [2, 12, 10, 22, 11, 6, 0], 
                            [8, 10, 18, 9, 14, 23, 1], [8, 18, 16, 23, 17, 13, 1], 
                            [10, 12, 20, 11, 15, 24, 1], [10, 20, 18, 24, 19, 14, 1]]),
    'NEIGH' : 0,
}

mshQ4 = {
    'POS' : np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]),
    'QUADS' : np.array([[0, 1, 4, 3, 0], [1, 2, 5, 4, 0], [3, 4, 7, 6, 1], [4, 5, 8, 7, 1]]),
    'LINES' : np.array([[0, 1, 1], [1, 2, 1], [0, 3, 1], [1, 4, 1], [2, 5, 1], 
                        [3, 4, 1], [4, 5, 1], [3, 6, 2], [4, 7, 2], [5, 8, 2], 
                        [6, 7, 2], [7, 8, 2]]),
    'NEIGH' : 0,
}

mshQ8 = {
    'POS' : np.array([[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], 
                      [0, 0.5], [1, 0.5], [2, 0.5],
                      [0, 1], [0.5, 1], [1, 1], [1.5, 1], [2, 1], 
                      [0, 1.5], [1, 1.5], [2, 1.5],
                      [0, 2], [0.5, 2], [1, 2], [1.5, 2], [2, 2]]),
    'QUADS' : np.array([[0, 2, 10, 8, 1, 6, 9, 5, 0], [2, 4, 12, 10, 3, 7, 11, 6, 0], 
                        [8, 10, 18, 16, 9, 14, 17, 13, 1], [10, 12, 20, 18, 11, 15, 19, 14, 1]]),
    'NEIGH' : 0,
}

mshQ9 = {
    'POS' : np.array([[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], 
                      [0, 0.5], [1, 0.5], [2, 0.5],
                      [0, 1], [0.5, 1], [1, 1], [1.5, 1], [2, 1], 
                      [0, 1.5], [1, 1.5], [2, 1.5],
                      [0, 2], [0.5, 2], [1, 2], [1.5, 2], [2, 2],
                      [0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [1.5, 1.5]]),
    'QUADS' : np.array([[0, 2, 10, 8, 1, 6, 9, 5, 21, 0], [2, 4, 12, 10, 3, 7, 11, 6, 22, 0], 
                        [8, 10, 18, 16, 9, 14, 17, 13, 23, 1], [10, 12, 20, 18, 11, 15, 19, 14, 24, 1]]),
    'NEIGH' : 0,
}

sys = SystemObject('PlaneStress', 'Triangle', 'Linear', msh, nGaussPoints = 1, verbose = True)
print("\n\n")

femData, prop = generateDataLinearElastic(sys, 'Linear', Er = 20)

print("We're here")