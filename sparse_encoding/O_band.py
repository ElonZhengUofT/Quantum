from qiskit import QuantumCircuit, QuantumRegister
import qiskit.visualization as qv
import numpy as np
from numpy import pi
from MyQuantum import *
import qiskit.quantum_info as qi
from qiskit.circuit.library import MCMT
from qiskit.circuit import Parameter, Gate
from plus_minus import *
b_i = QuantumRegister(3, 'b_')
j_i = QuantumRegister(3, 'j_')
qc_band = QuantumCircuit(j_i, b_i)
minus3 = minus_3.control(3)
minus2 = minus_2.control(3)
minus1 = minus_1.control(3)
add1 = add_1.control(3)
add2 = add_2.control(3)
add3 = add_3.control(3)

qc_band.x(b_i[1])
qc_band.append(minus3, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[1])
qc_band.x(b_i[0])
qc_band.append(minus2, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[0])
qc_band.append(minus1, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[2])
qc_band.x(b_i[1])
qc_band.append(add1, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[1])
qc_band.x(b_i[0])
qc_band.append(add2, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[0])
qc_band.append(add3, [b_i[0], b_i[1], b_i[2], j_i[0], j_i[1], j_i[2]])
qc_band.x(b_i[2])

O_band = qc_band.to_gate(label='O_band')


if __name__ == '__main__':
    qv.circuit_drawer(qc_band)
    print(qc_band)


