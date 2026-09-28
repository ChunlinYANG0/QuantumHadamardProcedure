from qiskit.quantum_info import Operator
from qiskit_aer import StatevectorSimulator
from QuantumHadamardProcedure.encoding import CompressedStatePreparation, BinaryTreeBlockEncoding, AllOneArray
import numpy as np


def clean_complex_matrix(matrix, tol=1e-12):
    """
    清洗复数矩阵：
      - 将实部和虚部中绝对值小于 tol 的分量置零。
      - 若虚部全部为零（或小于 tol），则返回实数数组（float64），否则返回复数数组。

    参数:
        matrix (np.ndarray): 复数数组或实数数组（会自动转为复数处理）
        tol (float): 绝对阈值

    返回:
        np.ndarray: 清洗后的数组（若虚部全零则为 float64，否则为 complex128）
    """
    # 强制转为复数类型（复制一份，避免修改原数组）
    mat = np.asarray(matrix, dtype=np.complex128)

    # 分别获取实部和虚部的副本（非视图，以便自由修改）
    real_part = mat.real.copy()
    imag_part = mat.imag.copy()

    # 清洗：将绝对值小于 tol 的分量置零
    real_part[np.abs(real_part) < tol] = 0.0
    imag_part[np.abs(imag_part) < tol] = 0.0

    # 判断虚部是否全为零（或极小）
    if np.all(np.abs(imag_part) <= tol):
        # 返回实数数组（float64），此时实部已经是清洗后的值
        return real_part
    else:
        # 重新组合为复数数组
        return real_part + 1j * imag_part


def test_state_preparation(n: int, num_control_qubits, control_states=None):
    state = 100 * np.random.randn(2 ** n) + 1j * 100 * np.random.randn(2 ** n)

    qsp = CompressedStatePreparation(
        state=state,
        num_target_qubits=n,
        epsilon=-1,
        num_control_qubits=num_control_qubits,
        control_states=control_states,
        decompose_mcx=False
    )
    qc = qsp.circuit()

    encoded_state = qsp.get_state(qc)[:2 ** n] * qsp.normalization_factor
    encoded_state = encoded_state.flatten()
    error = np.linalg.norm(state - encoded_state)

    # print('Circuit of state preparation:')
    # print(qc.draw(fold=240))
    # print('State:')
    # print(state)
    # print('Encoded state multiplied by normalization factor:')
    # print(encoded_state)
    # print('Error of state preparation:', error)
    if error > 1e-10:
        print(error)


def test_block_encoding(n, num_control_qubits, control_states=None, decompose_mcx=True):
    # matrix = np.array([[1, 3, 0, 0, 0, 0, 0, 0],
    #                    [3, 7, 8, 4, 0, 0, 0, 0],
    #                    [0, 8, 6, 0, 3, 8, 0, 0],
    #                    [0, 4, 0, 5, 0, 0, 0, 6],
    #                    [0, 0, 3, 0, 2, 0, 0, 0],
    #                    [0, 0, 8, 0, 0, 0, 0, 0],
    #                    [0, 0, 0, 0, 0, 0, 1, 0],
    #                    [0, 0, 0, 6, 0, 0, 0, 4]])
    # matrix = np.array([[0, 1, 0, 0],
    #                    [1, 1, 0, 0],
    #                    [0, 0, 1, 0],
    #                    [0, 0, 0, 0]])
    matrix = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
    be = BinaryTreeBlockEncoding(
        matrix=matrix,
        num_control_qubits=num_control_qubits,
        control_states=control_states,
        epsilon=-1,
        decompose_mcx=decompose_mcx
    )
    qc = be.circuit()
    normalization_factor = be.normalization_factor
    encoded_matrix = be.get_encoded_matrix(qc) * normalization_factor
    error = np.linalg.norm(matrix - encoded_matrix)

    # print('Circuit of block encoding:')
    # print(qc.draw(fold=240))
    # print(matrix)
    # print('Encoded matrix multiplied by normalization factor:')
    # print(clean_complex_matrix(encoded_matrix))
    print('Error of block encoding:', error)


def test_all_one_array(n, num_control_qubits, control_states=None, decompose_mcx=True):
    # state_or_matrix = 'state'
    state_or_matrix = 'matrix'

    if state_or_matrix == 'matrix':
        all_one_array = np.ones((2 ** n, 2 ** n))
    elif state_or_matrix == 'state':
        all_one_array = np.ones(2 ** n)
    else:
        raise ValueError('state_or_matrix must be either "matrix" or "state"')

    encoding = AllOneArray(
        num_working_qubits=n,
        state_or_matrix=state_or_matrix,
        num_control_qubits=num_control_qubits,
        control_state=control_states,
        decompose_mcx=decompose_mcx
    )
    qc = encoding.circuit()
    if state_or_matrix == 'state':
        simulator = StatevectorSimulator()
        result = simulator.run(qc).result()
        arr = result.get_statevector().data[:2 ** n]
        arr = arr * encoding.normalization_factor
        arr = clean_complex_matrix(arr)
    else:
        arr = Operator(qc).data[:2 ** n, :2 ** n] * encoding.normalization_factor
        arr = clean_complex_matrix(arr)

    print(qc.draw(fold=240))
    print(np.allclose(arr, all_one_array))


if __name__ == '__main__':
    n = 3
    num_control_qubits = 2
    control_states = 0
    # for n in range(1, 10):
    #     for _ in range(1000):
    # test_state_preparation(n, num_control_qubits, control_states)
    # print('=' * 150)
    # test_block_encoding(n, num_control_qubits, control_states)
    test_all_one_array(n, num_control_qubits, control_states, decompose_mcx=False)
