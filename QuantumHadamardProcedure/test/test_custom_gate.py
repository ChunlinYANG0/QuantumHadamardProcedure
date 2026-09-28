import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RZGate, RYGate, SwapGate, HGate
from qiskit.quantum_info import Operator
from QuantumHadamardProcedure.custum_gate import *
from QuantumHadamardProcedure import utils
from typing import Union, List, Tuple, Dict


def matrix_to_triplets(
        matrix: np.ndarray,
        include_value: bool = True,
        include_diagonal: bool = True,
        output_if_all_values_same: bool = False,
) -> Union[List[List], Tuple[List[List], bool, Optional[int | float | complex]]]:
    rows, cols = np.nonzero(matrix)

    if not include_diagonal:
        mask = (rows != cols)
        rows = rows[mask]
        cols = cols[mask]

    values = matrix[rows, cols]
    if include_value:
        triplets = [[int(r), int(c), v] for r, c, v in zip(rows, cols, values)]
    else:
        triplets = [[int(r), int(c)] for r, c in zip(rows, cols)]

    if output_if_all_values_same:
        if len(values) == 0:
            all_same = True
            same_value = None
        else:
            all_same = bool(np.all(values == values[0]))
            same_value = values[0]

        return triplets, all_same, same_value
    else:
        return triplets

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


def test_MultiCXGate():
    num_control_qubits = 3

    target = 0
    controls = list(range(1, 1+num_control_qubits))
    qc1 = QuantumCircuit(num_control_qubits+2)
    qc1.mcx(controls, target)
    print(qc1.draw(fold=240))
    unitary1 = Operator(qc1).data
    unitary1 = clean_complex_matrix(unitary1)
    unitary1 = matrix_to_triplets(unitary1, include_value=False, include_diagonal=False)
    print(unitary1)

    mcx = MultiCXGate(num_control_qubits)
    qc2 = mcx.circuit()
    print(qc2.draw(fold=240))
    unitary2 = Operator(qc2).data
    unitary2 = clean_complex_matrix(unitary2)
    unitary2 = matrix_to_triplets(unitary2, include_value=False, include_diagonal=False)
    print(unitary2)


def test_MultiCSWAP():
    num_control_qubits = 3

    qc1 = QuantumCircuit(2+num_control_qubits)
    swap = SwapGate().control(num_control_qubits)
    qc1.append(swap, list(range(2, 2+num_control_qubits))+[0, 1])
    print(qc1.draw(fold=240))
    unitary1 = Operator(qc1).data
    unitary1 = clean_complex_matrix(unitary1)

    mcswap = MultiCSWAP(num_control_qubits)
    qc2 = mcswap.circuit()
    print(qc2.draw(fold=240))
    unitary2 = Operator(qc2).data[:2**(2+num_control_qubits), :2**(2+num_control_qubits)]
    unitary2 = clean_complex_matrix(unitary2)

    print(np.allclose(unitary1, unitary2))


def test_MultiCRGate():
    basis_gate = 'ry'
    theta = np.random.randint(1, 100) / 100
    target = 0
    controls = [1, 2, 3]
    num_control_qubits = len(controls)

    qc1 = QuantumCircuit(len(controls)+2)
    if basis_gate == 'rx':
        mcr1 = RXGate(theta=theta).control(len(controls))
    elif basis_gate == 'ry':
        mcr1 = RYGate(theta=theta).control(len(controls))
    elif basis_gate == 'rz':
        mcr1 = RZGate(theta).control(len(controls))
    else:
        raise ValueError('basis_gate must be rx or ry or rz')
    qc1.append(mcr1, controls + [target])
    print(qc1.draw(fold=240))
    unitary1 = Operator(qc1).data
    unitary1 = clean_complex_matrix(unitary1)

    mcr = MultiCRGate(
        rotation_gate=basis_gate,
        rotation_angle=theta,
        num_control_qubits=num_control_qubits,
        decompose_mcx=False
    )
    qc2 = mcr.circuit()
    print(qc2.draw(fold=240))
    unitary2 = Operator(qc2).data
    unitary2 = clean_complex_matrix(unitary2)

    print(np.allclose(unitary1, unitary2))


def test_MultiCHGate():
    num_control_qubits = 2
    num_qubits = 1 + num_control_qubits if num_control_qubits <=2 else num_control_qubits + 2

    qc1 = QuantumCircuit(num_qubits)
    h = HGate().control(num_control_qubits)
    qc1.append(h, list(range(1,1+num_control_qubits))+[0])
    print(qc1.draw(fold=240))
    unitary1 = Operator(qc1).data
    unitary1 = clean_complex_matrix(unitary1)

    mch = MultiCHGate(
        num_control_qubits=num_control_qubits,
        decompose_mcx=False
    )
    qc_mch = mch.circuit()
    print(qc_mch.draw(fold=240))
    unitary2 = Operator(qc_mch).data
    unitary2 = clean_complex_matrix(unitary2)

    print(np.allclose(unitary1, unitary2))


def test_UniformlyCRGate():
    basis_gate = 'ry'
    num_uniformly_control_qubits = 5

    uniformly_controls = list(range(1, 1+num_uniformly_control_qubits))
    theta_list = [0.1 * (i+1) for i in range(2**num_uniformly_control_qubits)]

    qc1 = QuantumCircuit(num_uniformly_control_qubits+1)
    for i in range(len(theta_list)):
        ctrl_state = bin(i)[2:].zfill(num_uniformly_control_qubits)
        if basis_gate == 'ry':
            ucr = RYGate(theta_list[i]).control(num_uniformly_control_qubits, ctrl_state=ctrl_state)
        elif basis_gate == 'rz':
            ucr = RZGate(theta_list[i]).control(num_uniformly_control_qubits, ctrl_state=ctrl_state)
        else:
            raise ValueError('basis_gate must be ry or rz')
        qc1.append(ucr, uniformly_controls + [0])
    print(qc1.draw(fold=240))
    unitary1 = Operator(qc1).data
    unitary1 = clean_complex_matrix(unitary1)

    ucr2 = UniformlyCRGate(
        rotation_gate=basis_gate,
        rotation_angles=theta_list,
        num_target_qubits=1 + num_uniformly_control_qubits,
        epsilon=-1
    )
    qc2_ucr = ucr2.circuit()
    print(qc2_ucr.draw(fold=240))
    unitary2 = Operator(qc2_ucr).data
    unitary2 = clean_complex_matrix(unitary2)

    print(np.allclose(unitary1, unitary2))

    # ucr3 = UniformlyCRGate(
    #     rotation_gate=basis_gate,
    #     rotation_angles=theta_list,
    #     num_target_qubits=1 + num_uniformly_control_qubits,
    #     num_control_qubits=2,
    #     epsilon=-1,
    #     control_states=0
    # )
    # qc3_ucr = ucr3.circuit()
    # print(qc3_ucr.draw(fold=240))
    # unitary3 = Operator(qc3_ucr).data[:2**(num_uniformly_control_qubits+1), :2**(num_uniformly_control_qubits+1)]
    # unitary3 = clean_complex_matrix(unitary3)
    #
    # print(np.allclose(unitary1, unitary3))


def test_GHZGate():
    n = 10

    ghz = GHZGate(n)
    qc_ghz = ghz.circuit()
    print(qc_ghz.draw(fold=240))


def test_MultiCGHZGate():
    n = 10
    num_control_qubits = 0

    cghz = MultiCGHZGate(
        num_target_qubits=n,
        num_control_qubits=num_control_qubits,
        decompose_mcx=False
    )
    qc_cghz = cghz.circuit()
    print(qc_cghz.draw(fold=240))


def test_MultiShiftGate():
    num_target_qubits = 1
    num_control_qubits = 0

    mcshift = MultiShiftGate(
        num_target_qubits=num_target_qubits,
        num_control_qubits=num_control_qubits,
        decompose_mcx=False
    )
    qc_mcshift = mcshift.circuit()
    print(qc_mcshift.draw(fold=240))
    unitary = Operator(qc_mcshift).data[:2 ** num_target_qubits, :2 ** num_target_qubits]
    unitary = clean_complex_matrix(unitary)
    print(unitary)


if __name__ == '__main__':
    # test_MultiCXGate()
    # test_MultiCSWAP()
    # test_MultiCRGate()
    # test_MultiCHGate()
    test_UniformlyCRGate()
    # test_GHZGate()
    # test_MultiCGHZGate()
    # test_MultiShiftGate()