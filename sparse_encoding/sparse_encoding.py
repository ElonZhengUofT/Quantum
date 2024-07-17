import numpy as np
from qiskit import transpile, assemble
from O_val import *
from O_band import *
import qiskit.quantum_info as qi
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='e720215e3b824b343e4925053d61432717911a1241bee96d6cc5c02fa0e7e8f808703e5cdee9b9f065d2ffbdd4ef1f4dcd524c9756881dd6bf96c6cf7ec86aa8'
)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
estimator = Estimator(mode=backend)



def build_matrix_from_statevectors(num_qubits, circuit):
    backend = Aer.get_backend('statevector_simulator')
    matrix = []

    for i in range(2 ** num_qubits):
        test_circuit = QuantumCircuit(num_qubits)
        initial_state = [0] * (2 ** num_qubits)
        initial_state[i] = 1
        test_circuit.initialize(initial_state, range(num_qubits))
        test_circuit.compose(circuit, range(num_qubits), inplace=True)
        result = backend.run(transpile(test_circuit, backend)).result()
        state_vector = result.get_statevector()
        matrix.append(state_vector)

    return np.array(matrix).T

def build_matrix_from_real_qpu(num_qubits, circuit):
    matrix = []
    with Session(service=service, backend="ibm_kyoto"):
        sampler = Sampler()

    for i in range(2 ** num_qubits):
        test_circuit = QuantumCircuit(num_qubits)
        initial_state = [0] * (2 ** num_qubits)
        initial_state[i] = 1
        test_circuit.initialize(initial_state, range(num_qubits))
        test_circuit.compose(circuit, range(num_qubits), inplace=True)
        job = sampler.run(circuits=transpile(test_circuit, backend=Aer.get_backend('qasm_simulator')))
        result = job.result()
        vector = result.data()['\vector']
        matrix.append(vector)

    return np.array(matrix).T


def sparse_encoding(Mat):
    Matrix = Mat / np.linalg.norm(Mat)
    a, b, c, d = Matrix[0, 0], Matrix[0, 1], Matrix[0, 2], Matrix[0, 3]
    o_val = O_val(a, b, c, d)
    b_i = QuantumRegister(4, 'b_')
    j_i = QuantumRegister(3, 'j_')
    qc = QuantumCircuit(j_i, b_i)
    # initialize the state
    qc.reset(b_i)
    qc.h(b_i[2])
    qc.h(b_i[1])
    qc.h(b_i[0])

    qc.append(o_val, b_i)

    qc.append(O_band, j_i[:] + b_i[:3])

    qc.h(b_i[2])
    qc.h(b_i[1])
    qc.h(b_i[0])

    return qc

if __name__ == '__main__':


    Mat = np.array([[1, 2, 3, 4, 0, 4, 3, 2],
                    [2, 1, 2, 3, 4, 0, 4, 3],
                    [3, 2, 1, 2, 3, 4, 0, 4],
                    [4, 3, 2, 1, 2, 3, 4, 0],
                    [0, 4, 3, 2, 1, 2, 3, 4],
                    [4, 0, 4, 3, 2, 1, 2, 3],
                    [3, 4, 0, 4, 3, 2, 1, 2],
                    [2, 3, 4, 0, 4, 3, 2, 1]])

    qc = sparse_encoding(Mat)
    #     qv.circuit_drawer(qc)
    #     print(qc)
    #
    #     result = qi.Operator(qc)
    #     Result = result.data[0:8, 0:8] * 8
    #     print(np.round(Result, 4).real)
    #     print(np.round(Mat/np.linalg.norm(Mat), 4))
    #
    #     backend = Aer.get_backend('unitary_simulator')
    #     job = assemble(transpile(qc, backend))
    #     result = backend.run(job).result()
    #     unitary = result.get_unitary()
    #     print((np.round(unitary, 4)[0:8, 0:8]*8).real)
    #     print(np.round(Mat/np.linalg.norm(Mat), 4))

    file = open('matrix_record.txt', 'a+')
    file.seek(0)
    read = file.read()
    if read != '':
        result = eval(read)
        result = np.array(result)
        file.close()
    else:
        matrix = build_matrix_from_statevectors(7, qc)
        result = matrix[0:8, 0:8]
        file.write(repr(result.tolist()) + '\n')
        file.close()
    print(np.round(result, 4).real)
    print(np.round(Mat / (8 * np.linalg.norm(Mat)), 4).real)

    file = open('matrix_record_real_QPU.txt', 'a+')
    file.seek(0)
    read = file.read()
    if read != '':
        result = eval(read)
        result = np.array(result)
        file.close()
    else:
        bell = transpile(qc, backend=Aer.get_backend('qasm_simulator'))
        with Session(service=service, backend="ibm_canberra"):
            sampler = Sampler()
        job = sampler.run(circuits=bell)
        print(job.result())



# backend = Aer.get_backend('qasm_simulator')
#
#     tqc = transpile(qc, backend)
#
#     with Session(service=service, backend=backend) as session:
#         sampler = Sampler(session=session)
#         job = sampler.run(circuits=tqc, shots=1024)
#         result = job.result()
#
#     choi_matrix = result.data()['choi_matrix']
#     process_matrix = choi_matrix.data
#
#     print((np.round(process_matrix,4)[0:8, 0:8]*8).real)






