import numpy as np
import numpy_Kron_overload as npk
from MyQuantum import *


def prep(theta_0, theta_1):
    """
    This function return the prep for the LCU
    >>> prep(0, 0)
    """
    R_0 = rotation_matrix(theta_0)
    R_1 = rotation_matrix(theta_1)

    A = np.kron(Identity, np.kron(Identity,R_0))
    A = Identity ^ Identity ^ R_0
    B = np.kron(Identity,np.kron(zero_state, Identity) + np.kron(one_state, Hadamard))
    C = Converse_Toffoli
    D = np.kron(Identity, (np.kron(zero_state, R_1) + np.kron(one_state, Identity)))
    E = np.kron(Hadamard, np.kron(one_state,zero_state)) + np.kron(Identity,I - np.kron(one_state,zero_state))


def add_one():
    """
    This function return the add_one for the LCU
    """
    T = Converse_Toffoli
    C =
    N = np.kron(np.kron(Identity,Identity), sigma_x)

    add_one = np.dot(N, np.dot(C, T))

    return add_one


add_one = add_one()


def minus_one():
    A = np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)
    C = np.kron(CNOT_2, Identity)
    N = np.kron(np.kron(sigma_x, Identity), Identity)

    minus_one = np.dot(N, np.dot(C, A))

    return minus_one


minus_one = minus_one()


def select():
    """
    This function return the select for the LCU
    """
    add_two = np.dot(add_one, add_one)
    minus_two = np.dot(minus_one, minus_one)

    A = np.zeros((64, 64), dtype=complex)
    zeroth = np.eye(8)
    first = np.eye(8)
    second = add_two
    third = minus_two
    fourth = add_one
    fifth = np.eye(8)
    sixth = np.eye(8)
    seventh = minus_one
    diag = [zeroth, first, second, third, fourth, fifth, sixth, seventh]

    for i in range(0, 64, 8):
        A[i:i + 8, i:i + 8] = diag[i // 8]

    R = rotation_matrix(2 * np.pi)

    B = np.kron(np.kron((np.kron(np.kron(np.kron(Identity, Identity), R),zero_state)+ np.kron(np.kron(Identity, Identity), Identity),one_state), Identity), Identity)

    select = np.dot(B, A)

    return select



def LCU_solution(A):
    """
    This function return the solution for the LCU
    """
    I = np.eye(5)

#     P = np.array([
    #         [0, 0, 0, 0, 1],
    #         [1, 0, 0, 0, 0],
    #         [0, 1, 0, 0, 0],
    #         [0, 0, 1, 0, 0],
    #         [0, 0, 0, 1, 0]], dtype=complex)

#     P2 = P @ P
    #     P3 = P @ P2
    #     P4 = P @ P3

    a = np.zeros(5, dtype=complex)

    a[0] = A[4, 4]
    a[1] = A[4, 3]
    a[2] = A[4, 2]
    a[3] = A[4, 1]
    a[4] = A[4, 0]

#     C = np.stack(
    #         [I.flatten(), P.flatten(), P2.flatten(), P3.flatten(), P4.flatten()]).T
    #
    #     A_flat = A.flatten()
    #
    #     a = np.linalg.lstsq(C, A_flat, rcond=None)[0]

# A_reconstructed = a[0] * I + a[1] * P + a[2] * P2 + a[3] * P3 + a[4] * P4
    #     print(A-A_reconstructed)

    return a
