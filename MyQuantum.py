import numpy as np
import numpy_Kron_overload as npk
from qiskit import transpile
from qiskit_aer import Aer

# the order of operator in kronecker product follows the bit vector order
#  where in a graph, the higher the bit is, the leftmost the operator is
################################################################################
# 1. array form of Possible Quantum Gates
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.array([[1, 0], [0, 1]], dtype=complex)
swap_0_2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)

column_zero = np.array([[1], [0]], dtype=complex).reshape(-1, 1)
column_one = np.array([[0], [1]], dtype=complex).reshape(-1, 1)
row_zero = np.array([[1, 0]], dtype=complex)
row_one = np.array([[0, 1]], dtype=complex)

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
# 2. Matrix form of Quantum Gates(defining the kron by **)(Mainly Used)
Sigma_x = npk.Matrix(sigma_x)
Sigma_y = npk.Matrix(sigma_y)
Sigma_z = npk.Matrix(sigma_z)
Identity = npk.Matrix(identity)
Column_zero = npk.Matrix(column_zero)
Column_one = npk.Matrix(column_one)
Row_zero = npk.Matrix(row_zero)
Row_one = npk.Matrix(row_one)
Zero_state = npk.Matrix(zero_state)
One_state = npk.Matrix(one_state)
Hadamard = npk.Matrix(hadamard)
Toffoli = npk.Matrix(toffoli)
CNOT = npk.Matrix(cNOT)
CNOT_2 = npk.Matrix(cNOT_2)
SWAP_0_2 = npk.Matrix(swap_0_2)
################################################################################

################################################################################
# 3. Rebuilt Quantum Gates
I_4 = npk.Matrix(np.eye(4, dtype=complex))
I_8 = npk.Matrix(np.eye(8, dtype=complex))
Converse_Toffoli = Sigma_x ** (One_state ** One_state) + Identity ** (
        I_4 - (One_state ** One_state))
Converse_CNOT = Sigma_x ** One_state + Identity ** Zero_state
Converse_CNOT_2 = Sigma_x ** Zero_state + Identity ** One_state

################################################################################
# sqrt(2)
root_two = np.sqrt(2)
# axis
a = np.array([0, 1, 0])


################################################################################

################################################################################
# 4. Functions(Often Used high level Gates)
def rotation_matrix(theta):
    """
    Given a rotation angle phi and a vector a, return the rotation operator Ra
    """
    a_dot_sigma = a[0] * Sigma_x + a[1] * Sigma_y + a[2] * Sigma_z

    Ra = Identity * np.cos(theta / 2) + 1j * a_dot_sigma * np.sin(theta / 2)

    return Ra


def rotation(theta, a):
    """
    Given a rotation angle phi and a vector a, return the rotation operator Ra
    """
    a_dot_sigma = a[0] * Sigma_x + a[1] * Sigma_y + a[2] * Sigma_z

    Ra = Identity * np.cos(theta / 2) + 1j * a_dot_sigma * np.sin(theta / 2)

    return Ra

def Y_rotation(theta):
    """
    Given a rotation angle phi and a vector a, return the rotation operator Ra
    """
    Ra = np.cos(theta / 2) * Identity - 1j * np.sin(theta / 2) * Sigma_y

    return Ra


def Phase_shift(theta):
    """
    Given a phase shift angle theta, return the phase shift operator
    """
    Phase = npk.Matrix(np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex))

    return Phase


def add_one():
    """
    This function return the add_one for the LCU
    """
    T = Converse_Toffoli
    C = Identity ** Converse_CNOT
    N = np.kron(np.kron(Identity, Identity), sigma_x)

    add_one = N @ C @ T

    return add_one


def minus_one():
    A = Sigma_x ** (Zero_state ** Zero_state) + Identity ** (
            I_4 - Zero_state ** Zero_state)
    C = Identity ** Converse_CNOT_2
    N = Identity ** (Identity ** Sigma_x)

    minus_one = N @ C @ A

    return minus_one


def getM(size):
    def bitwise_inner_product(bin1: str, bin2: str) -> int:
        max_length = max(len(bin1), len(bin2))
        bin1 = bin1.zfill(max_length)
        bin2 = bin2.zfill(max_length)

        result = 0
        for i, j in zip(bin1, bin2):
            result += int(i) * int(j)

        return result

    def bigj(i: int, j: int) -> int:
        b = binary(i)
        g = grey_code(j)
        return bitwise_inner_product(b, g)

    def grey_code(num: int) -> str:
        return bin(num ^ (num >> 1))[2:]

    def binary(num: int) -> str:
        return bin(num)[2:]

    M = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            M[i][j] = (-1) ** (bigj(i, j))

    return M


def get_theta(alpha, M):
    return np.linalg.inv(M) @ alpha


def O_Lambda(THETAs):
    """
    This function return the Operator Lambda
    """
    axis = [0, 1, 0]
    Rearrange_THETAs = [THETAs[0], THETAs[7], THETAs[4], THETAs[3], THETAs[2],
                        THETAs[5], THETAs[6], THETAs[1]]
    R0 = rotation(Rearrange_THETAs[0], axis) ** I_8
    R1 = rotation(Rearrange_THETAs[1], axis) ** I_8
    R2 = rotation(Rearrange_THETAs[2], axis) ** I_8
    R3 = rotation(Rearrange_THETAs[3], axis) ** I_8
    R4 = rotation(Rearrange_THETAs[4], axis) ** I_8
    R5 = rotation(Rearrange_THETAs[5], axis) ** I_8
    R6 = rotation(Rearrange_THETAs[6], axis) ** I_8
    R7 = rotation(Rearrange_THETAs[7], axis) ** I_8

    CNOT_100 = (Sigma_x ** One_state + Identity ** Zero_state) ** I_4
    CNOT_010 = (Sigma_x ** Identity ** One_state + Identity ** Identity ** Zero_state) ** Identity
    CNOT_001 = (Sigma_x ** Identity ** Identity) ** One_state + Identity ** Identity ** Identity ** Zero_state

    Result = CNOT_001 @ R7 @ CNOT_100 @ R6 @ CNOT_010 @ R5 @ CNOT_100 @ R4 @ CNOT_001 @ R3 @ CNOT_100 @ R2 @ CNOT_010 @ R1 @ CNOT_100 @ R0

    return Result
def get_Rotations(PHI):
    THETA = get_theta(PHI, getM(8))


    CNOT_001 = (Sigma_x ** Identity ** Identity) ** One_state + Identity ** Identity ** Identity ** Zero_state
    CNOT_010 = (Sigma_x ** Identity ** One_state + Identity ** Identity ** Zero_state) ** Identity
    CNOT_100 = (Sigma_x ** One_state + Identity ** Zero_state) ** I_4

    R0 = Y_rotation(THETA[0]) ** I_8
    R1 = Y_rotation(THETA[1]) ** I_8
    R2 = Y_rotation(THETA[2]) ** I_8
    R3 = Y_rotation(THETA[3]) ** I_8
    R4 = Y_rotation(THETA[4]) ** I_8
    R5 = Y_rotation(THETA[5]) ** I_8
    R6 = Y_rotation(THETA[6]) ** I_8
    R7 = Y_rotation(THETA[7]) ** I_8


    R = CNOT_100 @ R7 @ CNOT_001 @ R6 @ CNOT_010 @ R5 @ CNOT_001 @ R4 @ CNOT_100 @ R3 @ CNOT_001 @ R2 @ CNOT_010 @ R1 @ CNOT_001 @ R0

    return R

def get_statevector(qc):
    backend = Aer.get_backend('statevector_simulator')
    tqc = transpile(qc, backend)
    result = backend.run(tqc).result()
    statevector = result.get_statevector(tqc)
    return statevector

# 从状态向量构建密度矩阵
def statevector_to_density_matrix(statevector):
    return np.outer(statevector, statevector.conj())

# 从密度矩阵获取量子电路的矩阵表示
def density_matrix_to_matrix(density_matrix):
    num_qubits = int(np.log2(density_matrix.shape[0]))
    matrix = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for i in range(2**num_qubits):
        for j in range(2**num_qubits):
            matrix[i, j] = density_matrix[i, j]
    return matrix


################################################################################
