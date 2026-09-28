from qiskit.quantum_info import Operator
from qiskit_aer import StatevectorSimulator
from QuantumHadamardProcedure.qhp import *
import numpy as np


def hadamard_product(array_list):
    result_array = array_list[0]

    for arr in array_list[1:]:
        result_array = result_array * arr

    return result_array


def test_QHProd(
        n, num_arr, state_or_matrix,
        num_control_qubits=0, control_states=None,
        use_cache=False, decompose_mcx=True
):
    if state_or_matrix == 'matrix':
        arr_list = [np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n) for _ in range(num_arr)]
    elif state_or_matrix == 'state':
        arr_list = [np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n) for _ in range(num_arr)]
    else:
        raise ValueError('Unsupported state or matrix')

    arr_hprod = hadamard_product(arr_list)

    qhprod = QHProd(
        arr_list=arr_list,
        state_or_matrix=state_or_matrix,
        num_control_qubits=num_control_qubits,
        control_states=control_states,
        use_cache=use_cache,
        decompose_mcx=decompose_mcx,
    )
    qc_qhprod = qhprod.circuit()
    print(qc_qhprod.draw(fold=240))
    if state_or_matrix == 'matrix':
        unitary = Operator(qc_qhprod).data
        matrix = unitary[:2**n, :2**n] * qhprod.normalization_factor
        print(np.allclose(matrix, arr_hprod))
    else:
        simulator = StatevectorSimulator()
        result = simulator.run(qc_qhprod).result()
        state = result.get_statevector().data * qhprod.normalization_factor
        print(np.allclose(state[:2**n], arr_hprod))


if __name__ == '__main__':
    n = 2
    num_arr = 5
    state_or_matrix = 'state'
    # num_control_qubits = 2
    # control_states = None
    # control_states = 0
    use_cache = False
    decompose_mcx = False

    test_QHProd(
        n=n,
        num_arr=num_arr,
        state_or_matrix=state_or_matrix,
        # num_control_qubits=num_control_qubits,
        # control_states=control_states,
        use_cache=use_cache,
        decompose_mcx=decompose_mcx,
    )