from qiskit import QuantumCircuit
import qiskit.visualization as qv
import numpy as np
from numpy import pi
from MyQuantum import *
import qiskit.quantum_info as qi

def O_val(a, b, c, d):
    alpha0 = 2 * np.arccos(a)
    alpha1 = 2 * np.arccos(b)
    alpha2 = 2 * np.arccos(c)
    alpha3 = 2 * np.arccos(d)
    alpha4 = 2 * np.arccos(0)
    alphas = [alpha0, alpha1, alpha2, alpha3, alpha4, alpha3, alpha2, alpha1]

    thetas = get_theta(alphas, getM(8))

    theta0, theta1, theta2, theta3, theta4, theta_3, theta_2, theta_1 = thetas

    qc = QuantumCircuit(4)

    qc.ry(theta0, 3)
    qc.cx(0,3)
    qc.ry(theta1, 3)
    qc.cx(1,3)
    qc.ry(theta2, 3)
    qc.cx(0,3)
    qc.ry(theta3, 3)
    qc.cx(2,3)
    qc.ry(theta4, 3)
    qc.cx(0,3)
    qc.ry(theta_3, 3)
    qc.cx(1,3)
    qc.ry(theta_2, 3)
    qc.cx(0,3)
    qc.ry(theta_1, 3)
    qc.cx(2,3)
    val = qc.to_gate(label='O_val')

    return val


if __name__ == '__main__':

    Mat = np.array([[1, 2, 3, 4, 0, 4, 3, 2],
                    [2, 1, 2, 3, 4, 0, 4, 3],
                    [3, 2, 1, 2, 3, 4, 0, 4],
                    [4, 3, 2, 1, 2, 3, 4, 0],
                    [0, 4, 3, 2, 1, 2, 3, 4],
                    [4, 0, 4, 3, 2, 1, 2, 3],
                    [3, 4, 0, 4, 3, 2, 1, 2],
                    [2, 3, 4, 0, 4, 3, 2, 1]])

    qc = O_val(1/ np.linalg.norm(Mat), 2/ np.linalg.norm(Mat), 3/ np.linalg.norm(Mat), 4/ np.linalg.norm(Mat))
    Result = qi.Operator(qc)
    print(Result)

    print(Mat/np.linalg.norm(Mat))

#     Result = qi.Operator(qc)
#     a = 1 / 10
#     b = 2 / 10
#     c = 3 / 10
#     d = 4 / 10
#     alpha0 = 2 * np.arccos(a)
#     alpha1 = 2 * np.arccos(b)
#     alpha2 = 2 * np.arccos(c)
#     alpha3 = 2 * np.arccos(d)
#     alphas = [alpha0, alpha1, alpha2, alpha3, alpha0, alpha3, alpha2, alpha1]
#     Compare = get_Rotations(alphas)
#     if np.allclose(Result, Compare, atol=1e-5):
#         print("O_val is correct")
#     else:
#         print("O_val is incorrect")