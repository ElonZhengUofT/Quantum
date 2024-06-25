import numpy as np
import numpy_Kron_overload as npk
from MyQuantum import *


def LCU(A):
    """
    This function return the block encoding for a banded and cyclic matrix by LCU
    """
    theta_0, theta_1, coef_sum = LCU_solution(A)

    Prep = prep(theta_0, theta_1)
    PZ = Prep @ (Column_zero ** Column_zero ** Column_zero)
    # print(round(PZ, 3).real)
    Select = select()

    result =(Row_zero ** Row_zero ** Row_zero ** I_8) @ (np.linalg.inv(Prep) ** I_8) @ Select @ (Prep ** I_8) @ (Column_zero ** Column_zero ** Column_zero ** I_8)

    return result, coef_sum

def prep(theta_0, theta_1):
    """
    This function return the prep for the LCU
    >>> prep(0, 0)
    """
    R_0 = Y_rotation(theta_0)
    R_1 = Y_rotation(theta_1)

    A = Identity ** Identity ** R_0
    B = Identity ** (Hadamard ** One_state + Identity ** Zero_state)
    C = Converse_Toffoli
    D = Identity ** (R_1 ** Zero_state + Identity ** One_state)
    E = Hadamard ** (One_state ** Zero_state) + Identity ** (
            I_4 - One_state ** Zero_state)

    prep = E @ D @ C @ B @ A
    return prep


add_one = add_one()
minus_one = minus_one()


def select():
    """
    This function return the select for the LCU
    """
    add_two = add_one @ add_one
    minus_two = minus_one @ minus_one

    Mat = np.zeros((64, 64), dtype=complex)
    zeroth = I_8
    first = add_one
    second = add_two
    third = I_8
    fourth = I_8
    fifth = I_8
    sixth = minus_two
    seventh = minus_one
    diag = [zeroth, first, second, third, fourth, fifth, sixth, seventh]

    for i in range(0, 64, 8):
        Mat[i:i + 8, i:i + 8] = diag[i // 8]

    R = Y_rotation(np.pi * 2)
    B = I_4 ** (
                Zero_state ** R + One_state ** Identity) ** I_4
    Select = Mat
    return Select


def LCU_solution(A):
    """
    This function return the solution for the LCU
    """
    a_0 = A[0][0]
    a_1 = A[0][1]
    a_2 = A[0][2]
    coef_sum = a_0 + 2 * a_1 + 2 * a_2

    theta_0 = 2 * np.arcsin(np.sqrt(2 * a_1 / coef_sum))
    theta_1 = 2 * np.arcsin(np.sqrt(2 * a_2 / (coef_sum - 2 * a_1)))

    return theta_0, theta_1, coef_sum


if __name__ == "__main__":
    A = np.array([[1,2,1,0,0,0,1,2],
                        [2,1,2,1,0,0,0,1],
                        [1,2,1,2,1,0,0,0],
                        [0,1,2,1,2,1,0,0],
                        [0,0,1,2,1,2,1,0],
                        [0,0,0,1,2,1,2,1],
                        [1,0,0,0,1,2,1,2],
                        [2,1,0,0,0,1,2,1]], dtype=complex)


    Result, coef_sum = LCU(A)
    validation = Result  - A / coef_sum
    print(np.round(validation, 3).real)

    B = np.array(   [[1, 2, 3,0,0,0,3,2],
                            [2, 1, 2, 3,0,0,0,3],
                            [3, 2, 1, 2, 3,0,0,0],
                            [0, 3, 2, 1, 2, 3,0,0],
                            [0, 0, 3, 2, 1, 2, 3,0],
                            [0, 0, 0, 3, 2, 1, 2, 3],
                            [3, 0, 0, 0, 3, 2, 1, 2],
                            [2, 3, 0, 0, 0, 3, 2, 1]], dtype=complex)
    Result, coef_sum = LCU(B)
    validation = Result  - B / coef_sum
    print(np.round(validation, 3).real)

