from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# Eqn. (1)
def make_init_circuit(
    n_qubits: int,
    dry_run: bool = False
) -> QuantumCircuit | int:
    # dry_runがTrueの場合，必要なパラメータ数を返す
    if dry_run:
        return n_qubits
    
    # 量子回路を初期化
    init_circuit = QuantumCircuit(n_qubits)
    # 入力データ用のパラメータベクトルを作成
    x = ParameterVector('x', n_qubits)
    # 各量子ビットにRy回転ゲートを適用
    for i in range(n_qubits):
        init_circuit.ry(x[i], i)

    return init_circuit

# Fig. 1 (a) TTN classifier
def make_ansatz(
    n_qubits: int,
    insert_barrier: bool = False,
    dry_run: bool = False
) -> QuantumCircuit | int:
    # 単一のユニタリ操作を追加する内部関数
    def append_U(qc, i, j, thetas, count, last_unitary=False, reverse=False):
        # i番目の量子ビットにRy回転を適用
        qc.ry(thetas[count], i)
        count += 1
        # j番目の量子ビットにRy回転を適用
        qc.ry(thetas[count], j)
        count += 1

        # CXゲートを適用（制御ビットと標的ビットの順序は reverse フラグに依存）
        if reverse:
            ansatz.cx(j, i)
        else:
            ansatz.cx(i, j)
        # 最後のユニタリ操作の場合，追加のRy回転を適用
        if last_unitary:
            qc.ry(thetas[count], j)
            count += 1
        return count

    # 必要なパラメータ数を計算
    length = 2*n_qubits//2  # U5 - U6 層のパラメータ数
    length += 3*n_qubits//4  # U7 層のパラメータ数

    # dry_runがTrueの場合，パラメータ数を返す
    if dry_run:
        return length

    # パラメータベクトルを作成
    thetas = ParameterVector('θ', length)

    count = 0
    ansatz = QuantumCircuit(n_qubits)
    # U5 - U6 層の構築
    reverse = False
    for i in range(0, n_qubits, 2):
        if i+1 >= n_qubits:
            break
        count = append_U(ansatz, i, i+1, thetas, count, reverse=reverse)
        reverse = not reverse
    if insert_barrier:
        ansatz.barrier()
    # U7 層の構築
    for i in range(1, n_qubits, 4):
        if i+1 >= n_qubits:
            break
        count = append_U(ansatz, i, i+1, thetas, count, last_unitary=True)
    if insert_barrier:
        ansatz.barrier()
    # パラメータ数の整合性チェック
    assert count == length, count
    return ansatz

# 初期化回路とアンサッツ回路を組み合わせた全体の回路を生成する関数
def make_placeholder_circuit(
    n_qubits: int,
    insert_barrier: bool = False,
    dry_run: bool = False
) -> QuantumCircuit | int:
    # dry_runがTrueの場合，全体のパラメータ数を計算して返す
    if dry_run:
        # 初期化回路のパラメータ数を取得
        length_feature = make_init_circuit(n_qubits, dry_run=True)
        # アンサッツ回路のパラメータ数を取得
        length_ansatz = make_ansatz(n_qubits, dry_run=True)
        # 全体のパラメータ数を計算
        length = length_feature + length_ansatz
        return length

    # 初期化回路を生成
    qc = make_init_circuit(n_qubits)
    # アンサッツ回路を生成（バリア挿入オプション付き）
    ansatz = make_ansatz(n_qubits, insert_barrier)
    # 初期化回路とアンサッツ回路を組み合わせる（その場で変更）
    qc.compose(ansatz, inplace=True)

    # 完成した量子回路を返す
    return qc
