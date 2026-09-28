from qiskit import QuantumCircuit
from qiskit_aer import StatevectorSimulator
from QuantumHadamardProcedure.custum_gate import (
    BaseMultiControlledGate, MultiCRGate, UniformlyCRGate, MultiCSWAP, MultiCHGate
)
from QuantumHadamardProcedure import utils
import numpy as np
import math
from typing import List


class CompressedStatePreparation(BaseMultiControlledGate):
    """

    target qubit     —/—| U |——
                          |
    control qubits   —/———●————

    borrowed ancilla ——————————

    """

    def __init__(
            self,
            state: np.ndarray | List,
            num_target_qubits: int = None,
            is_real: bool = None,
            epsilon: float = 1e-12,
            num_control_qubits: int = 0,
            control_states: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        self.state = np.asarray(state)
        if num_target_qubits is None:
            num_target_qubits = utils.get_num_qubits(state)

        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        self.dim = 2 ** self.num_target_qubits
        if len(self.state) == 0:
            raise ValueError("Input state must be non-empty.")
        self.normalization_factor = np.linalg.norm(state)
        if self.normalization_factor == 0:
            raise ValueError("Input state must be non-zero.")
        elif self.normalization_factor != 1:
            self.state = self.state / self.normalization_factor

        if is_real is None:
            self._is_real = utils.is_all_real(self.state)
        else:
            self._is_real = is_real

        self.state = utils.pad_to_power_of_two(self.state)
        # if length != self.dim:
        #     pad_width = self.dim - length
        #     self.state = np.pad(self.state, (0, pad_width), mode='constant')

        if self._is_real:
            self.state = np.real_if_close(self.state)
            self.norm_angles = utils.binarytree_norm(self.state, True)
            self.phase_angles = None
        else:
            state_abs = np.abs(self.state)
            state_phase = np.angle(self.state)
            self.norm_angles = utils.binarytree_norm(state_abs).flatten()
            self.phase_angles = utils.binarytree_phase(state_phase).flatten()

        self.num_borrowed_ancilla = 1 if ((num_target_qubits == 1 and num_control_qubits >= 3)
                                          or (num_target_qubits == 2 and num_control_qubits >= 2)) else 0
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.epsilon = epsilon
        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        if not self._is_real:
            rz = MultiCRGate(
                rotation_gate='rz',
                rotation_angle=self.phase_angles[0].item(),
                num_control_qubits=self.num_control_qubits,
                epsilon=self.epsilon,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_rz = rz.circuit()
            qubit_rz = [self.target_qubits[-1]] + self.control_qubits
            if rz.num_borrowed_ancilla != 0:
                extra_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else self.target_qubits[-2]
                qubit_rz.append(extra_ancilla)
            qc.compose(qc_rz, qubit_rz, inplace=True)

        ry = MultiCRGate(
            rotation_gate='ry',
            rotation_angle=self.norm_angles[0].item(),
            num_control_qubits=self.num_control_qubits,
            epsilon=self.epsilon,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx
        )
        qc_ry = ry.circuit()
        qubit_ry = [self.target_qubits[-1]] + self.control_qubits
        if ry.num_borrowed_ancilla != 0:
            extra_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else self.target_qubits[-2]
            qubit_ry.append(extra_ancilla)
        qc.compose(qc_ry, qubit_ry, inplace=True)

        angle_index = 1
        for layer in range(1, self.num_target_qubits):
            ucry = UniformlyCRGate(
                rotation_gate='ry',
                rotation_angles=self.norm_angles[angle_index: angle_index + 2 ** layer],
                num_target_qubits=1 + layer,
                epsilon=self.epsilon,
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_ucry = ucry.circuit()
            qubits_ucry = self.target_qubits[- 1 - layer:] + self.control_qubits
            if ucry.num_borrowed_ancilla != 0:
                extra_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else self.target_qubits[-3]
                qubits_ucry.append(extra_ancilla)
            qc.compose(qc_ucry, qubits_ucry, inplace=True)
            angle_index += 2 ** layer

        if not self._is_real:
            rz = MultiCRGate(
                rotation_gate='rz',
                rotation_angle=float(self.phase_angles[1]),
                num_control_qubits=self.num_control_qubits,
                epsilon=self.epsilon,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_rz = rz.circuit()
            qubit_rz = [self.target_qubits[-1]] + self.control_qubits
            if rz.num_borrowed_ancilla != 0:
                extra_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else self.target_qubits[-2]
                qubit_rz.append(extra_ancilla)
            qc.compose(qc_rz, qubit_rz, inplace=True)

            angle_index = 2
            for layer in range(1, self.num_target_qubits):
                ucrz = UniformlyCRGate(
                    rotation_gate='rz',
                    rotation_angles=self.phase_angles[angle_index: angle_index + 2 ** layer],
                    num_target_qubits=1 + layer,
                    epsilon=self.epsilon,
                    num_control_qubits=self.num_control_qubits,
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx
                )
                qc_ucrz = ucrz.circuit()
                qubits_ucrz = self.target_qubits[- 1 - layer:] + self.control_qubits
                if ucrz.num_borrowed_ancilla != 0:
                    extra_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else self.target_qubits[-3]
                    qubits_ucrz.append(extra_ancilla)
                qc.compose(qc_ucrz, qubits_ucrz, inplace=True)
                angle_index += 2 ** layer

        self._apply_x(qc)

        return qc

    @staticmethod
    def get_state(
            qc: QuantumCircuit
    ) -> np.ndarray:
        simulator = StatevectorSimulator()
        result = simulator.run(qc).result()
        state = result.get_statevector().data

        return state


class BinaryTreeBlockEncoding(BaseMultiControlledGate):
    def __init__(
            self,
            matrix: np.ndarray,
            epsilon: float = 1e-12,
            num_control_qubits: int = 0,
            control_states: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Matrix must be np.ndarray.")
        else:
            if matrix.ndim != 2:
                raise ValueError("Input is not a matrix.")

        self._is_real = utils.is_all_real(matrix)

        if self._is_real:
            matrix = np.real_if_close(matrix)
            self.normalization_factor = np.linalg.norm(matrix)
            self.matrix = matrix / self.normalization_factor
            self._norm_angles, self._col_norm_angles = utils.rotation_angles_matrix(self.matrix, True)
            self._phase_angles = None
        else:
            self.normalization_factor = np.linalg.norm(matrix)
            self.matrix = matrix / self.normalization_factor
            self._norm_angles, self._phase_angles, self._col_norm_angles = utils.rotation_angles_matrix(self.matrix)
        self._norm_angles = np.nan_to_num(self._norm_angles)

        self.num_working_qubits = utils.get_num_qubits(self.matrix)
        self.num_ancilla = self.num_working_qubits

        super().__init__(
            num_target_qubits=2 * self.num_working_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        self.num_borrowed_ancilla = 1 if self.num_working_qubits == 1 and self.num_control_qubits >= 2 else 0
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla

        self.working_qubits = self.target_qubits[:self.num_working_qubits]
        self.ancilla = self.target_qubits[self.num_working_qubits:]
        self.control_qubits = list(range(self.num_target_qubits, self.num_target_qubits + self.num_control_qubits))
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.epsilon = epsilon
        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        # qubits:  target qubits (working)
        #          target qubits (ancilla)
        #          control qubits
        #          borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        qc_ul = self._build_ul_circuit()
        qc_swap = self._build_swap_circuit()
        qc_ur = self._build_ur_circuit()

        qc.compose(qc_ul, inplace=True)
        qc.compose(qc_swap, inplace=True)
        qc.compose(qc_ur, inplace=True)

        self._apply_x(qc)

        return qc

    def _build_ul_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        if not self._is_real:
            ucrz = UniformlyCRGate(
                rotation_gate='rz',
                rotation_angles=self._phase_angles[0, :],
                num_target_qubits=1 + self.num_working_qubits,
                epsilon=self.epsilon,
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_ucrz = ucrz.circuit()
            if ucrz.num_borrowed_ancilla:
                extra_ancilla = self.borrowed_ancilla
            else:
                extra_ancilla = None
            qubits_ucrz = (
                    [self.target_qubits[-1]]
                    + self.working_qubits + self.control_qubits
                    + ([extra_ancilla] if extra_ancilla is not None else [])
            )
            qc.compose(qc_ucrz, qubits_ucrz, inplace=True)

        angle_index = 0
        for layer in range(self.num_working_qubits):
            ucry = UniformlyCRGate(
                rotation_gate='ry',
                rotation_angles=self._norm_angles[angle_index: angle_index + 2 ** layer, :].flatten(),
                num_target_qubits=1 + self.num_working_qubits + layer,
                epsilon=self.epsilon,
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_ucry = ucry.circuit()
            if ucry.num_borrowed_ancilla:
                extra_ancilla = self.borrowed_ancilla
            else:
                extra_ancilla = None
            qubits_ucry = (
                [self.target_qubits[-1 - layer]]
                + self.working_qubits + (self.target_qubits[- layer:] if layer != 0 else []) + self.control_qubits
                + ([extra_ancilla] if extra_ancilla is not None else [])
            )
            qc.compose(qc_ucry, qubits_ucry, inplace=True)
            angle_index += 2 ** layer

        if not self._is_real:
            angle_index = 1
            for layer in range(self.num_working_qubits):
                ucr = UniformlyCRGate(
                    rotation_gate='rz',
                    rotation_angles=self._phase_angles[angle_index: angle_index + 2 ** layer, :].flatten(),
                    num_target_qubits=1 + self.num_working_qubits + layer,
                    epsilon=self.epsilon,
                    num_control_qubits=self.num_control_qubits,
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx
                )
                qc_ucr = ucr.circuit()
                if ucr.num_borrowed_ancilla:
                    extra_ancilla = self.borrowed_ancilla
                else:
                    extra_ancilla = None
                qubits_ucr = (
                    [self.target_qubits[-1 - layer]]
                    + self.working_qubits + (self.target_qubits[- layer:] if layer != 0 else []) + self.control_qubits
                    + ([extra_ancilla] if extra_ancilla is not None else [])
                )
                qc.compose(qc_ucr, qubits_ucr, inplace=True)
                angle_index += 2 ** layer

        return qc

    def _build_swap_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        for layer in range(self.num_working_qubits):
            swap = MultiCSWAP(
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_swap = swap.circuit()
            qubits_swap = [layer, layer + self.num_working_qubits] + self.control_qubits
            if swap.num_borrowed_ancilla != 0:
                exact_ancilla = self.borrowed_ancilla if self.borrowed_ancilla is not None else layer + 1
                qubits_swap.append(exact_ancilla)
            qc.compose(qc_swap, qubits_swap, inplace=True)

        return qc

    def _build_ur_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        mcry = MultiCRGate(
            rotation_gate='ry',
            rotation_angle=self._col_norm_angles[0],
            num_control_qubits=self.num_control_qubits,
            epsilon=self.epsilon,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx
        )
        qc_mcry = mcry.circuit()
        if mcry.num_borrowed_ancilla:
            extra_ancilla = self.num_working_qubits - 1
        else:
            extra_ancilla = None
        qubits_mcry = (
                [self.target_qubits[-1]]
                + self.control_qubits
                + ([extra_ancilla] if extra_ancilla is not None else [])
        )
        qc.compose(qc_mcry, qubits_mcry, inplace=True)

        angle_index = 1
        for layer in range(1, self.num_working_qubits):
            ucr = UniformlyCRGate(
                rotation_gate='ry',
                rotation_angles=self._col_norm_angles[angle_index: angle_index + 2 ** layer],
                num_target_qubits=1+layer,
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_ucr = ucr.circuit()
            if ucr.num_borrowed_ancilla:
                extra_ancilla = self.num_working_qubits - 1
            else:
                extra_ancilla = None
            qubits_ucr = (
                self.target_qubits[-1 - layer:]
                + self.control_qubits
                + ([extra_ancilla] if extra_ancilla is not None else [])
            )
            qc.compose(qc_ucr, qubits_ucr, inplace=True)
            angle_index += 2 ** layer

        return qc.inverse()

    def get_encoded_matrix(
            self,
            qc: QuantumCircuit
    ) -> np.ndarray:
        from qiskit.quantum_info import Operator
        return Operator(qc).data[:2 ** self.num_working_qubits, :2 ** self.num_working_qubits]


class AllOneArray(BaseMultiControlledGate):
    def __init__(
            self,
            num_working_qubits: int,
            state_or_matrix: str = 'state',
            num_control_qubits: int = 0,
            control_state: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        self.num_working_qubits = num_working_qubits
        if state_or_matrix == 'state':
            num_target_qubits = self.num_working_qubits
            self.normalization_factor = math.sqrt(2 ** self.num_working_qubits)
        elif state_or_matrix == 'matrix':
            num_target_qubits = 2 * self.num_working_qubits
            self.normalization_factor = 2 ** self.num_working_qubits
        else:
            raise ValueError('state_or_matrix must be "state" or "matrix".')
        self.state_or_matrix = state_or_matrix

        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        self.num_borrowed_ancilla = (
            1 if (self.state_or_matrix == 'state' and self.num_target_qubits == 1 and self.num_control_qubits >= 3) or
                 (self.state_or_matrix == 'matrix' and self.num_working_qubits == 1 and self.num_control_qubits >= 2)
            else 0
        )
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla

        self.borrowed_ancilla = self.num_qubits - 1

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        qc_h = self._build_h_circuit()
        if self.state_or_matrix == 'state':
            self._apply_x(qc)
            qc.compose(qc_h, inplace=True)
            self._apply_x(qc)

            return qc

        self._apply_x(qc)

        qc_swap = self._build_swap_circuit()

        qc.compose(qc_h, inplace=True)
        qc.compose(qc_swap, inplace=True)
        qc.compose(qc_h, inplace=True)

        self._apply_x(qc)

        return qc

    def _build_h_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        start_qubit = self.num_target_qubits - self.num_working_qubits
        if self.num_control_qubits == 0:
            for layer in range(self.num_working_qubits):
                qc.h(start_qubit + layer)

            return qc

        for layer in range(self.num_working_qubits):
            mch = MultiCHGate(
                num_control_qubits=self.num_control_qubits,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_mch = mch.circuit()
            if mch.num_borrowed_ancilla:
                if self.num_borrowed_ancilla:
                    exact_ancilla = self.borrowed_ancilla
                else:
                    exact_ancilla = layer + 1 if layer != self.num_working_qubits - 1 else layer - 1
            else:
                exact_ancilla = None
            qubits_mch = (
                    [layer] + self.control_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
            )
            qc.compose(qc_mch, qubits_mch, inplace=True)

        return qc

    def _build_swap_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        swap = MultiCSWAP(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx
        )
        qc_swap = swap.circuit()

        for i in range(self.num_working_qubits):
            if swap.num_borrowed_ancilla:
                exact_ancilla = self.borrowed_ancilla if self.num_borrowed_ancilla else i + 1
            else:
                exact_ancilla = None
            qubits_swap = (
                    [i, i+self.num_working_qubits]
                    + self.control_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
            )
            qc.compose(qc_swap, qubits_swap, inplace=True)

        return qc