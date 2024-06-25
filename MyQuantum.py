import numpy as np
import numpy_Kron_overload as npk

# the order of operator in kronecker product follows the bit vector order
#  where in a graph, the higher the bit is, the leftmost the operator is
################################################################################
# 1. array form of Possible Quantum Gates
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.array([[1, 0], [0, 1]], dtype=complex)

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

################################################################################
