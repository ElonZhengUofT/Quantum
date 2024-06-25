import numpy as np
import numpy_Kron_overload as npk
from MyQuantum import *


def QFT():
    """
    This function return the Quantum Fourier Transform
    """
    H = Hadamard ** Identity ** Identity
    B = (Phase_shift(
        np.pi / 2) ** One_state + Identity ** Zero_state) ** Identity
    C = Phase_shift(
        np.pi / 4) ** Identity ** One_state + Identity ** Identity ** Zero_state
    D = Identity ** Hadamard ** Identity
    E = Identity ** (
                Phase_shift(np.pi / 2) ** One_state + Identity ** Zero_state)
    F = Identity ** Identity ** Hadamard
    SWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    QFT = SWAP @ F @ E @ D @ C @ B @ H

#     H = Identity ** Identity ** Hadamard
    #     B = Identity ** (One_state ** Phase_shift(np.pi / 2) + Zero_state ** Identity)
    #     C = One_state ** One_state ** Phase_shift(np.pi / 4) + (I_4 - One_state ** One_state) ** Identity
    #     D = Identity ** Hadamard ** Identity
    #     E = (Zero_state ** Phase_shift(np.pi / 2) + One_state ** Identity) ** Identity
    #     F = Hadamard ** Identity ** Identity
    #     QFT = F @ E @ D @ C @ B @ H

    return QFT


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


def UT(A):
    """
    This function return the Unitary Transform
    """
    o = np.exp(np.pi * 1j / 4)
    F = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [1, o ** 1, o ** 2, o ** 3, o ** 4, o ** 5, o ** 6, o ** 7],
                  [1, o ** 2, o ** 4, o ** 6, o ** 8, o ** 10, o ** 12,
                   o ** 14],
                  [1, o ** 3, o ** 6, o ** 9, o ** 12, o ** 15, o ** 18,
                   o ** 21],
                  [1, o ** 4, o ** 8, o ** 12, o ** 16, o ** 20, o ** 24,
                   o ** 28],
                  [1, o ** 5, o ** 10, o ** 15, o ** 20, o ** 25, o ** 30,
                   o ** 35],
                  [1, o ** 6, o ** 12, o ** 18, o ** 24, o ** 30, o ** 36,
                   o ** 42],
                  [1, o ** 7, o ** 14, o ** 21, o ** 28, o ** 35, o ** 42,
                   o ** 49]], dtype=complex) / np.sqrt(8)

    Lambda = np.linalg.inv(F) @ A @ F / np.linalg.norm(A)
    #   print(np.round(Lambda, 5))
    ALPHA = []
    for i in range(8):
        ALPHA.append(2 * np.arccos(Lambda[i][i]))
    M = getM(8)
    #   print(ALPHA)
    THETAs = get_theta(ALPHA, M)
    #   print("THETA",np.round(THETAs, 5))

    #     O_L_should = np.zeros((16,16), dtype=complex)
    #     PHI = ALPHA
    #     r1 = rotation_matrix(PHI[0])
    #     r2 = rotation_matrix(PHI[1])
    #     r3 = rotation_matrix(PHI[2])
    #     r4 = rotation_matrix(PHI[3])
    #     r5 = rotation_matrix(PHI[4])
    #     r6 = rotation_matrix(PHI[5])
    #     r7 = rotation_matrix(PHI[6])
    #     r8 = rotation_matrix(PHI[7])
    #     r = [r1, r2, r3, r4, r5, r6, r7, r8]
    #
    #     for i in range(0, 8):
    #         print(i)
    #         O_L_should[i][i] = r[i][0, 0]
    #         O_L_should[i][i + 8] = r[i][0, 1]
    #         O_L_should[i + 8][i] = r[i][1, 0]
    #         O_L_should[i + 8][i + 8] = r[i][1, 1]
    #
    #
    #     print("O_L_should",np.round(O_L_should, 5).real)
    #     Def = O_L_should[0:8, 0:8] - Lambda
    #     print("Def",np.round(Def, 5).real)

    O_L = O_Lambda(THETAs)
    #   print("OL",np.round(O_L[0:8, 0:8], 5).real)
    Def = O_L[0:8, 0:8] - Lambda
    #   print("Def", np.round(Def, 5).real)
    Qft = QFT()
    print("QFT", np.round(Qft, 3).real)
    print("F", np.round(F, 3).real)
    Result = (Row_zero ** I_8) @ (Identity ** Qft) @ O_L @ (
                Identity ** np.linalg.inv(Qft)) @ (Column_zero ** I_8)
    return Result


if __name__ == '__main__':
    A = np.array([[1, 2, 1, 0, 0, 0, 1, 2],
                  [2, 1, 2, 1, 0, 0, 0, 1],
                  [1, 2, 1, 2, 1, 0, 0, 0],
                  [0, 1, 2, 1, 2, 1, 0, 0],
                  [0, 0, 1, 2, 1, 2, 1, 0],
                  [0, 0, 0, 1, 2, 1, 2, 1],
                  [1, 0, 0, 0, 1, 2, 1, 2],
                  [2, 1, 0, 0, 0, 1, 2, 1]], dtype=complex)
    Result = UT(A)
    print("A", np.round(A / np.linalg.norm(A), 3).real)
    print("Result", round(Result, 3).real)
    Validation = Result - A / np.linalg.norm(A)
    print(round(Validation, 3).real)
