from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# Eqn. (1)
def make_init_circuit(
    n_qubits: int,
    dry_run: bool = False
) -> QuantumCircuit | int:
    if dry_run:
        return n_qubits

    init_circuit = QuantumCircuit(n_qubits)
    x = ParameterVector('x', n_qubits)
    for i in range(n_qubits):
        init_circuit.ry(x[i], i)

    return init_circuit

# Fig. 1 (a) TTN classifier
def make_ansatz(
    n_qubits: int,
    insert_barrier: bool = False,
    dry_run: bool = False
) -> QuantumCircuit | int:
    def append_U(qc, i, j, thetas, count, last_unitary=False, reverse=False):
        qc.ry(thetas[count], i)
        count += 1
        qc.ry(thetas[count], j)
        count += 1

        if reverse:
            ansatz.cx(j, i)
        else:
            ansatz.cx(i, j)
        if last_unitary:
            qc.ry(thetas[count], j)
            count += 1
        return count

    length = 2*n_qubits//2  # U5 - U6
    length += 3*n_qubits//4  # U7

    if dry_run:
        return length

    thetas = ParameterVector('θ', length)

    count = 0
    ansatz = QuantumCircuit(n_qubits)
    # U5 - U6
    reverse = False
    for i in range(0, n_qubits, 2):
        if i+1 >= n_qubits:
            break
        count = append_U(ansatz, i, i+1, thetas, count, reverse=reverse)
        reverse = not reverse
    if insert_barrier:
        ansatz.barrier()
    # U7
    for i in range(1, n_qubits, 4):
        if i+1 >= n_qubits:
            break
        count = append_U(ansatz, i, i+1, thetas, count, last_unitary=True)
    if insert_barrier:
        ansatz.barrier()
    assert count == length, count
    return ansatz

def make_placeholder_circuit(
    n_qubits: int,
    insert_barrier: bool = False,
    dry_run: bool = False
) -> QuantumCircuit | int:
    if dry_run:
        length_feature = make_init_circuit(n_qubits, dry_run=True)
        length_ansatz = make_ansatz(n_qubits, dry_run=True)
        length = length_feature + length_ansatz
        return length

    qc = make_init_circuit(n_qubits)
    ansatz = make_ansatz(n_qubits, insert_barrier)
    qc.compose(ansatz, inplace=True)

    return qc
