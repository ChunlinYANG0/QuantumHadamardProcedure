from qiskit.quantum_info import Operator
from qiskit_aer import StatevectorSimulator
from QuantumHadamardProcedure.qhp import *
import random
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


def hadamard_poly(poly, array):
    array = np.asarray(array)
    poly = poly[::-1]
    if poly[0] == 0:
        result = np.zeros_like(array)
    else:
        result = poly[0] * np.ones_like(array)

    for k in range(1, len(poly)):
        result = result + poly[k] * (array ** k)

    return result


def test_QHPoly(n, poly, state_or_matrix, mode, num_subpolys=1):
    if state_or_matrix == 'state':
        arr = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
        # arr = np.array(range(2 ** n))
    elif state_or_matrix == 'matrix':
        arr = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
    else:
        raise ValueError("state_or_matrix must be 'matrix' or 'state'")

    arr_hpoly = hadamard_poly(poly, arr)

    qhpoly = QHPoly(
        arr=arr,
        poly=poly,
        mode=mode,
        num_subpolys=num_subpolys,
        decompose_mcx=False
    )
    qc = qhpoly.circuit()

    if state_or_matrix == 'state':
        simulator = StatevectorSimulator()
        result = simulator.run(qc).result()
        arr_qhpoly = result.get_statevector().data[:2**n] * qhpoly.normalization_factor
    else:
        arr_qhpoly = Operator(qc).data[:2**n, :2**n] * qhpoly.normalization_factor
    error = np.linalg.norm(arr_hpoly - arr_qhpoly)

    # print(qc.draw(fold=240))
    # print('Original arr:')
    # print(arr)
    # print('Arr after Hadamard poly:')
    # print(arr_hpoly)
    # print('Arr after QHPoly:')
    # print(arr_qhpoly)
    print('Error after QHPoly:', error)
    print('='*100)

    # if error > 1e-10:
    #     print(error)
    #     print('n', n)
    #     print('poly', poly)
    #     print('='*50)


if __name__ == '__main__':
    # state_or_matrix = 'state'
    state_or_matrix = 'matrix'
    mode = 'lcu'
    # mode = 'binary'
    # mode = 'factorization'
    # mode = 'tradeoff'
    # num_subpolys = 2

    for n in range(2, 3):
        for length in range(3, 7):
            poly = np.random.randint(-100, 100, length) / 10
            while poly[0] == 0:
                poly[0] = random.randint(-10, 10)
            test_QHPoly(
                n=n,
                poly=poly,
                state_or_matrix=state_or_matrix,
                mode=mode,
                # num_subpolys=num_subpolys
            )