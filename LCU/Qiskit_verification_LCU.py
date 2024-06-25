from qiskit import QuantumCircuit, execute, Aer
import numpy as np
from qiskit.visualization import plot_histogram


def LCU_circuit():
    return


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


def Prep_circuit(theta_0, theta_1):
    qc = QuantumCircuit(3)
    qc.ry(theta_0, 0)
    qc.ch(0, 1)
    qc.mcx([0, 1], 2)
    qc.cry(theta_1, 0, 1   )
    
