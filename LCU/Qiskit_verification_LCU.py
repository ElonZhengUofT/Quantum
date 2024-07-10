from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import HGate


def LCU_circuit():
    theta_0, theta_1, coef_sum = LCU_solution(A)
    qc1 = Prep_circuit(theta_0, theta_1)

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
    # add a multi-controlled hadamard gate contorlled by the first two qubits
    qc.x(0)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    qc.x(0)

    backend = Aer.get_backend('unitary_simulator')
    tqc = transpile(qc, backend)
    result = backend.run(tqc).result()
    unitary = result.get_unitary(qc)
    print(np.round(unitary, 3))

    return qc


if __name__ == "__main__":
    A = np.array([[1,2,1,0,0,0,1,2],
                        [2,1,2,1,0,0,0,1],
                        [1,2,1,2,1,0,0,0],
                        [0,1,2,1,2,1,0,0],
                        [0,0,1,2,1,2,1,0],
                        [0,0,0,1,2,1,2,1],
                        [1,0,0,0,1,2,1,2],
                        [2,1,0,0,0,1,2,1]])

    LCU_circuit()


