import numpy as np
import numpy_Kron_overload as npk

################################################################################
# 1. array form of Possible Quantum Gates
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.array([[1, 0], [0, 1]], dtype=complex)
zero_state = np.array([[1, 0], [0, 0]], dtype=complex)
one_state = np.array([[0, 0], [0, 1]], dtype=complex)
hadamard = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
toffoli = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
cNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
cNOT_2 = np.array([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=complex)

################################################################################

################################################################################
# 2. Matrix form of Quantum Gates(defining the kron by ^)
Sigma_x = npk.Matrix(sigma_x)
Sigma_y = npk.Matrix(sigma_y)
Sigma_z = npk.Matrix(sigma_z)
Identity = npk.Matrix(identity)
Zero_state = npk.Matrix(zero_state)
One_state = npk.Matrix(one_state)
Hadamard = npk.Matrix(hadamard)
Toffoli = npk.Matrix(toffoli)
CNOT = npk.Matrix(cNOT)
CNOT_2 = npk.Matrix(cNOT_2)
################################################################################

################################################################################
# 3. Rebuilt Quantum Gates
I = np.eye(4)
Converse_Toffoli = Sigma_x ^ (One_state ^ One_state) + Identity ^ (I - (One_state ^ One_state))
Converse_CNOT = Sigma_x ^ One_state + Identity ^ Zero_state

################################################################################
# axis
a = np.array([1, 0, 0])
# sqrt(2)
root_two = np.sqrt(2)
################################################################################

################################################################################
# 4. Functions
def rotation_matrix(theta):
    """
    Given a rotation angle phi and a vector a, return the rotation operator Ra
    """
    a_dot_sigma = a[0] * Sigma_x + a[1] * Sigma_y + a[2] * Sigma_z

    Ra = Identity * np.cos(theta / 2) + 1j * a_dot_sigma * np.sin(theta / 2)

    return Ra
################################################################################