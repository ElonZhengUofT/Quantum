import numpy as np
from qiskit import transpile, assemble, QuantumCircuit, QuantumRegister, ClassicalRegister
from O_val import *
from O_band import *
import qiskit.quantum_info as qi
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Sampler
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='e720215e3b824b343e4925053d61432717911a1241bee96d6cc5c02fa0e7e8f808703e5cdee9b9f065d2ffbdd4ef1f4dcd524c9756881dd6bf96c6cf7ec86aa8'
)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
estimator = Estimator(mode=backend)
sampler = Sampler(mode=backend)

def build_matrix_from_real_qpu(num_qubits, circuit):
    matrix = []
    for i in range(2 ** num_qubits):
        test_circuit = QuantumCircuit(num_qubits)
        initial_state = [0] * (2 ** num_qubits)
        initial_state[i] = 1
        test_circuit.initialize(initial_state, range(num_qubits))
        test_circuit.compose(circuit, range(num_qubits), inplace=True)
        pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
        isa_circuit = pm.run(circuit)
        job = sampler.run([isa_circuit])
        result = job.result()
        counts = result.data(0)['counts']
        vector = np.zeros(2 ** num_qubits)
        for outcome, prob in counts.items():
            index = int(outcome[::-1], 2)
            vector[index] = prob

        matrix.append(vector)

    return np.array(matrix).T



def sparse_encoding(Mat):
    Matrix = Mat / np.linalg.norm(Mat)
    a, b, c, d = Matrix[0, 0], Matrix[0, 1], Matrix[0, 2], Matrix[0, 3]
    o_val = O_val(a, b, c, d)
    b_i = QuantumRegister(4, 'b_')
    j_i = QuantumRegister(3, 'j_')
    c_reg = ClassicalRegister(7, 'c_')
    qc = QuantumCircuit(j_i, b_i, c_reg)
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
    qc.measure(j_i, c_reg[:3])
    qc.measure(b_i, c_reg[3:])

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


    file = open('matrix_record_real_QPU.txt', 'a+')
    file.seek(0)
    read = file.read()
    if read != '':
        result = eval(read)
        result = np.array(result)
        file.close()
    else:
        result = build_matrix_from_real_qpu(7, qc)
        file.write(str(result))
        file.close()

    print(np.round(result, 4).real)
    print((Mat/np.linalg.norm(Mat)).real)






