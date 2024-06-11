import numpy as np

class Matrix:
    def __init__(self, array):
        self.array = np.array(array)

    def __xor__(self, other):
        return Matrix(np.kron(self.array, other.array))

    def __repr__(self):
        return repr(self.array)