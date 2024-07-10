from qiskit import QuantumCircuit
import qiskit.visualization as qv
import numpy as np
from numpy import pi
from MyQuantum import *
import qiskit.quantum_info as qi


qc_add1 = QuantumCircuit(3)
qc_add1.ccx(0, 1, 2)
qc_add1.cx(0, 1)
qc_add1.x(0)

qc_minus1 = QuantumCircuit(3)
qc_minus1.x(1)
qc_minus1.x(0)
qc_minus1.ccx(0, 1, 2)
qc_minus1.x(1)
qc_minus1.cx(0, 1)
qc_minus2 = qc_minus1.compose(qc_minus1, [0, 1, 2], front=True)
qc_minus3 = qc_minus1.compose(qc_minus2, [0, 1, 2], front=True)
qc_add2 = qc_add1.compose(qc_add1, [0, 1, 2], front=True)
qc_add3 = qc_add1.compose(qc_add2, [0, 1, 2], front=True)

minus_1 = qc_minus1.to_gate(label='minus_1')
minus_2 = qc_minus2.to_gate(label='minus_2')
minus_3 = qc_minus3.to_gate(label='minus_3')
add_1 = qc_add1.to_gate(label='add_1')
add_2 = qc_add2.to_gate(label='add_2')
add_3 = qc_add3.to_gate(label='add_3')










if __name__ == '__main__':
    qv.circuit_drawer(qc_add1)
    qv.circuit_drawer(qc_minus1)
    print(qc_add1)
    print(qc_minus1)

#     backend = AerSimulator()
    #     job1 = transpile(qc_add1, backend)
    #     add1 = backend.run(job1).result()
#     add1 = qi.Operator(qc_add1)
#     add_1 = add_one()
#     if np.array_equal(add1, add_1):
#         print("Add one is correct")
#     else:
#         print("Add one is incorrect")

#     job2 = transpile(qc_minus1, backend)
    #     minus1 = backend.run(job2).result()
#     minus1 = qi.Operator(qc_minus1)
#     minus_1 = minus_one()
#     if np.array_equal(minus1, minus_1):
#         print("Minus one is correct")
#     else:
#         print("Minus one is incorrect")
#
#     print(add1)
#     print(minus1)




