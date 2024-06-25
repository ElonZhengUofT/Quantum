import Transformation_Matrix as tm
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import process_fidelity

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
Identity = np.array([[1, 0], [0, 1]], dtype=complex)
zero_state = np.array([[1, 0], [0, 0]], dtype=complex)
one_state = np.array([[0, 0], [0, 1]], dtype=complex)
a = np.array([1,1,1])/np.sqrt(3)

def C_R_1(phi):
    """
    return an operator for C_R_1 gate
    where its controller qubits are 0,0,0.
    When the control qubits are 0,0,0, the C_R_1 gate is equivalent to the R_1 gate.
    """
    R1 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), R1) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)R1 + (001)I + (010)I + (011)I + (100)I + (101)I + (110)I + (111)I
    return Operator(CR)

def C_R_2(phi):
    """
    controller qubits are 0,0,1
    """
    R2 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), R2) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)R2 + (010)I + (011)I + (100)I + (101)I + (110)I + (111)I
    return Operator(CR)

def C_R_3(phi):
    """
    controller qubits are 0,1,0
    """
    R3 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), R3) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)I + (010)R3 + (011)I + (100)I + (101)I + (110)I + (111)I
    return Operator(CR)

def C_R_4(phi):
    """
    controller qubits are 0,1,1
    """
    R4 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), R4) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)I + (010)I + (011)R4 + (100)I + (101)I + (110)I + (111)I
    return Operator(CR)

def C_R_5(phi):
    """
    controller qubits are 1,0,0
    """
    R5 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), R5) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)I + (010)I + (011)I + (100)R5 + (101)I + (110)I + (111)I
    return Operator(CR)

def C_R_6(phi):
    """
    controller qubits are 1,0,1
    """
    R6 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), R6) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)I + (010)I + (011)I + (100)I + (101)R6 + (110)I + (111)I
    return Operator(CR)

def C_R_7(phi):
    """
    controller qubits are 1,1,0
    """
    R7 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), R7) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), Identity)
    # (000)I + (001)I + (010)I + (011)I + (100)I + (101)I + (110)R7 + (111)I
    return Operator(CR)

def C_R_8(phi):
    """
    controller qubits are 1,1,1
    """
    R8 = rotation_matrix(phi)
    CR = np.kron(np.kron(np.kron(zero_state, zero_state),zero_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, zero_state),one_state), Identity) +\
        np.kron(np.kron(np.kron(zero_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(zero_state, one_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, zero_state),one_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),zero_state), Identity) +\
            np.kron(np.kron(np.kron(one_state, one_state),one_state), R8)
    # (000)I + (001)I + (010)I + (011)I + (100)I + (101)I + (110)I + (111)R8
    return Operator(CR)

def C_NOT_001():
    """
    return an operator for C_NOT_001 gate
    where its controller qubits are 0,0,1.
    When the control qubits are 0,0,1, the C_NOT_001 gate is equivalent to the NOT gate.
    """

    NOT = np.kron(np.kron(Identity,Identity),np.kron(zero_state, sigma_x) + np.kron(one_state, Identity))
    # I@I@(1X + 0I)
    return Operator(NOT)

def C_NOT_010():
    """
    return an operator for C_NOT_010 gate
    where its controller qubits are 0,1,0.
    When the control qubits are 0,1,0, the C_NOT_010 gate is equivalent to the NOT gate.
    """
    NOT = np.kron(Identity, np.kron(zero_state, np.kron(Identity,Identity)) + np.kron(one_state, np.kron(Identity,sigma_x)))
    # I@(0@I@I + 1@I@X)
    return Operator(NOT)

def C_NOT_100():
    """
    return an operator for C_NOT_011 gate
    where its controller qubits are 0,1,1.
    When the control qubits are 0,1,1, the C_NOT_011 gate is equivalent to the NOT gate.
    """
    NOT = np.kron(zero_state,np.kron(Identity,np.kron(Identity,Identity))) + np.kron(one_state,np.kron(Identity,np.kron(Identity,sigma_x)))
    # (0@I@I@I + 1@I@I@X)
    return Operator(NOT)

def rotation_matrix(phi):
    """
    Given a rotation angle phi and a vector a, return the rotation operator Ra
    """
    a_dot_sigma = a[0] * sigma_x + a[1] * sigma_y + a[2] * sigma_z

    Ra = Identity * np.cos(phi / 2) + 1j * a_dot_sigma * np.sin(phi / 2)

    return Ra


def test_getM_size_3(PHI):


    M = tm.getM(8)
    THETA = tm.get_theta(PHI, M)

    # U1 = C_R_1(PHI[0])
    #     U2 = C_R_2(PHI[1])
    #     U3 = C_R_3(PHI[2])
    #     U4 = C_R_4(PHI[3])
    #     U5 = C_R_5(PHI[4])
    #     U6 = C_R_6(PHI[5])
    #     U7 = C_R_7(PHI[6])
    #     U8 = C_R_8(PHI[7])
    #     U = U8.compose(U7).compose(U6).compose(U5).compose(U4).compose(U3).compose(U2).compose(U1)

    r1 = rotation_matrix(PHI[0])
    r2 = rotation_matrix(PHI[1])
    r3 = rotation_matrix(PHI[2])
    r4 = rotation_matrix(PHI[3])
    r5 = rotation_matrix(PHI[4])
    r6 = rotation_matrix(PHI[5])
    r7 = rotation_matrix(PHI[6])
    r8 = rotation_matrix(PHI[7])
    r = [r1, r2, r3, r4, r5, r6, r7, r8]

    U = np.zeros((16,16), dtype=complex)
    for i in range(0,16,2):
        U[i:i+2, i:i+2] = r[i//2]

    U = Operator(U)


    R1 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[0])))
    R2 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[1])))
    R3 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[2])))
    R4 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[3])))
    R5 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[4])))
    R6 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[5])))
    R7 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[6])))
    R8 = Operator(np.kron(np.kron(np.kron(Identity, Identity), Identity), rotation_matrix(THETA[7])))

    CN1 = C_NOT_001()
    CN2 = C_NOT_010()
    CN3 = C_NOT_100()

    R = CN3.compose(R8).compose(CN1).compose(R7).compose(CN2).compose(R6).compose(CN1).compose(R5).compose(CN3).compose(R4).compose(CN1).compose(R3).compose(CN2).compose(R2).compose(CN1).compose(R1)
    # build a 16 by 16 all-zero (0+0j) matrix
    zero_matrix = np.zeros((16,16), dtype=complex)
    zero_opertor = Operator(zero_matrix)
    D = R - U
    print(U)
    print("The difference between the two operators is", D)
    print("The process fidelity is", process_fidelity(U, R))


if __name__ == '__main__':
#     PHIs = [np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/2, np.pi/3, np.pi/3, np.pi/3, np.pi/3]),
#             np.array([np.pi/3, np.pi/3, np.pi/2, np.pi/2, np.pi/3, np.pi/3, np.pi/3, np.pi/3]),
#             np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/2, np.pi/2, np.pi/3, np.pi/3, np.pi/3]),
#             np.array([np.pi/3, np.pi/2, np.pi/3, np.pi/2, np.pi/3, np.pi/3, np.pi/3, np.pi/3]),
#             np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/2, np.pi/3, np.pi/3, np.pi/4, np.pi/3]),
#             np.array([np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3, np.pi/3])]
#     for PHI in PHIs:
#         test_getM_size_3(PHI)
    PHI =np.array([(1.9106332362490184+3.9252311467094385e-17j), (2.2133936772164815-0.22777661338978505j), (2.8549588374993706-0.2906413301006183j), (3.464980246312816-0.20645507040497738j), (3.727278197046944-5.439603600208736e-17j), (3.464980246312816+0.2064550704049773j), (2.8549588374993706+0.2906413301006183j), (2.2133936772164815+0.2277766133897849j)],dtype=complex)
    test_getM_size_3(PHI)