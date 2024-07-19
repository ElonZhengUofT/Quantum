from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='e720215e3b824b343e4925053d61432717911a1241bee96d6cc5c02fa0e7e8f808703e5cdee9b9f065d2ffbdd4ef1f4dcd524c9756881dd6bf96c6cf7ec86aa8'
)
job = service.job('ctc0jyky6ybg008g5wn0')
job_result = job.result()
counts = job_result.metadata
print(counts)