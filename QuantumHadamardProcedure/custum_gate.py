from qiskit import QuantumCircuit
from QuantumHadamardProcedure import utils
import math
import numpy as np
from typing import List, Optional, Dict


class BaseMultiControlledGate:
    """

    target qubits  —/——| U |———
                         |
    control qubits —/————●————

    """
    def __init__(
            self,
            num_target_qubits: int,
            num_control_qubits: int,
            control_states: str | int | None = None
    ):
        if num_target_qubits <= 0:
            raise ValueError("num_target_qubits must be a positive integer")

        if num_control_qubits != 0:
            if num_control_qubits < 0:
                raise ValueError("num_control_qubits must be a non-negative integer")

            if isinstance(control_states, int):
                control_states = bin(control_states)[2:].zfill(num_control_qubits)
            elif control_states is None:
                control_states = "1" * num_control_qubits
            elif isinstance(control_states, str):
                if len(control_states) != num_control_qubits:
                    raise ValueError(
                        f"The length of control_states {control_states} "
                        f"must be equal to num_control_qubits {num_control_qubits}"
                    )
            else:
                raise ValueError("control_states must be a string or an integer or None.")

            zero_control_qubits = [
                num_target_qubits + idx for idx, bit in enumerate(reversed(control_states)) if bit == "0"
            ] if control_states != "1" * num_control_qubits else []

        else:
            if control_states is not None:
                raise ValueError("control_states is given but num_control_qubits is None.")
            zero_control_qubits = []

        self.control_states = control_states

        self.num_target_qubits = num_target_qubits
        self.num_control_qubits = num_control_qubits
        self.zero_control_qubits = zero_control_qubits

        self.target_qubits = list(range(num_target_qubits))
        self.control_qubits = list(range(num_target_qubits, num_target_qubits + num_control_qubits))

    def _apply_x(
            self,
            qc: QuantumCircuit
    ):
        if self.zero_control_qubits:
            qc.x(self.zero_control_qubits)


class CircuitCache:
    _cache_mcx: Dict[int, QuantumCircuit] = {}

    @classmethod
    def get_mcx_circuit(
            cls,
            num_control_qubits: int,
            use_cache: bool = False,
            decompose: bool = True
    ) -> QuantumCircuit:

        if not decompose:
            num_qubits = num_control_qubits + 1 + (1 if num_control_qubits >= 3 else 0)
            qc = QuantumCircuit(num_qubits)
            qc.mcx(
                control_qubits=list(range(1, 1+ num_control_qubits)),
                target_qubit=0
            )
            return qc

        key = num_control_qubits

        if use_cache and key in cls._cache_mcx:
            return cls._cache_mcx[key].copy()

        mcx = MultiCXGate(
            num_control_qubits=num_control_qubits
        )
        qc = mcx._build_circuit()

        if use_cache:
            cls._cache_mcx[key] = qc.copy()

        return qc


class MultiCXGate(BaseMultiControlledGate):
    """

    target qubit     ————⨁————
                         |
    control qubits   —/——●————

    borrowed ancilla ——————————

    """
    def __init__(
            self,
            num_control_qubits: int,
            control_states: str | int | None = None,
            use_cache: bool = False,
            decompose: bool = True
    ):
        super().__init__(
            num_target_qubits=1,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        (self.num_borrowed_ancilla, self.borrowed_ancilla) = (
            (1, 1 + num_control_qubits) if num_control_qubits >= 3 else (0, None)
        )
        self.num_qubits = 1 + self.num_control_qubits + self.num_borrowed_ancilla

        self.use_cache = use_cache
        self.decompose = decompose

    def circuit(self) -> QuantumCircuit:
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        if not self.decompose:
            qc.mcx(
                control_qubits=self.control_qubits,
                target_qubit=self.target_qubits[0],
                ctrl_state=self.control_states,
            )
            return qc

        self._apply_x(qc)

        qc_mcx = CircuitCache.get_mcx_circuit(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose
        )
        qc.compose(qc_mcx, inplace=True)

        self._apply_x(qc)

        return qc

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        if self.num_control_qubits == 1:
            qc.cx(self.control_qubits[0], self.target_qubits[0])
        elif self.num_control_qubits == 2:
            qc.ccx(self.control_qubits[0], self.control_qubits[1], self.target_qubits[0])
        else:
            qc_mcx = self._build_mcx_recursive(
                control_qubits=self.control_qubits,
                target_qubit=self.target_qubits[0],
                borrowed_ancilla=self.borrowed_ancilla
            )
            qc.compose(qc_mcx, inplace=True)

        return qc

    def _build_mcx_recursive(
            self,
            control_qubits: List[int],
            target_qubit: int,
            borrowed_ancilla: Optional[int]
    ) -> QuantumCircuit:
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        n = len(control_qubits)
        if n == 1:
            qc.cx(control_qubits[0], target_qubit)
            return qc

        if n == 2:
            qc.ccx(control_qubits[0], control_qubits[1], target_qubit)
            return qc

        # 分组控制比特
        num_qubits_per_register_R = max(1, math.floor(math.sqrt(n)) - 2)
        num_registers = n // (num_qubits_per_register_R + 2)
        num_remaining_control_qubits = n - num_registers * (num_qubits_per_register_R + 2)
        registers_list = []
        start = 0
        for _ in range(num_registers):
            length = num_qubits_per_register_R + 2
            registers_list.append(control_qubits[start: start + length])
            start += length
        remaining_control_qubits = control_qubits[start:] if num_remaining_control_qubits > 0 else []

        ctrl_qubits_for_first = []
        for register in registers_list:
            ctrl_qubits_for_first.extend([register[0], register[-1]])
        ctrl_qubits_for_first.extend(remaining_control_qubits)
        qc1 = self._build_mcx_recursive(
            control_qubits=ctrl_qubits_for_first,
            target_qubit=borrowed_ancilla,
            borrowed_ancilla=target_qubit
        )

        qc2 = QuantumCircuit(self.num_qubits)
        for register in registers_list:
            inner = self._build_mcx_recursive(
                control_qubits=register[1:-1],
                target_qubit=register[0],
                borrowed_ancilla=register[-1]
            )
            qc2.compose(inner, inplace=True)

        qc3 = QuantumCircuit(self.num_qubits)
        ctrl_qubits_for_third = [register[0] for register in registers_list] + [borrowed_ancilla]
        for qubit in ctrl_qubits_for_third[:-1]:
            qc3.x(qubit)
        mcx3 = self._build_mcx_recursive(
            control_qubits=ctrl_qubits_for_third,
            target_qubit=target_qubit,
            borrowed_ancilla=registers_list[0][-1]
        )
        qc3.compose(mcx3, inplace=True)
        for qubit in ctrl_qubits_for_third[:-1]:
            qc3.x(qubit)

        qc.compose(qc1, inplace=True)
        qc.compose(qc2, inplace=True)
        qc.compose(qc3, inplace=True)
        qc.compose(qc2, inplace=True)
        qc.compose(qc1, inplace=True)
        qc.compose(qc2, inplace=True)
        qc.compose(qc3, inplace=True)
        qc.compose(qc2, inplace=True)

        return qc


class MultiCZGate(BaseMultiControlledGate):
    """

    target qubit     ————| Z |————
                           |
    control qubits   —/————●——————

    borrowed ancilla —————————————

    """
    def __init__(
            self,
            num_control_qubits,
            control_state: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=1,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        self.num_borrowed_ancilla = 1 if self.num_control_qubits >= 3 else 0
        self.num_qubits = 1 + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        qc.h(0)
        mcx = MultiCXGate(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose_mcx
        )
        qc_mcx = mcx.circuit()
        qc.compose(qc_mcx, inplace=True)
        qc.h(0)

        self._apply_x(qc)

        return qc


class MultiCHGate(BaseMultiControlledGate):
    """

    target qubit     ————| H |————
                           |
    control qubits   —/————●——————

    borrowed ancilla —————————————

    """
    def __init__(
            self,
            num_control_qubits,
            control_state: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=1,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        self.num_borrowed_ancilla = 1 if self.num_control_qubits >= 3 else 0
        self.num_qubits = 1 + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        qc.ry(-math.pi / 4, 0)
        mcz = MultiCZGate(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx
        )
        qc_mcz = mcz.circuit()
        qc.compose(qc_mcz, inplace=True)
        qc.ry(math.pi / 4, 0)

        self._apply_x(qc)

        return qc


class MultiCRGate(BaseMultiControlledGate):
    """

    target qubit     ——| R |——
                         |
    control qubits   —/——●————

    borrowed ancilla —————————

    """
    def __init__(
            self,
            rotation_gate: str,
            rotation_angle: float,
            num_control_qubits: int,
            control_states: str | int = None,
            epsilon: float = 1e-12,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=1,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        self.rotation_gate = rotation_gate.lower()
        if self.rotation_gate not in {'rx', 'ry', 'rz'}:
            raise ValueError("rotation_gate must be 'rx', 'ry', or 'rz'")

        self.rotation_angle = rotation_angle

        (self.num_borrowed_ancilla, self.borrowed_ancilla) = (
            (1, 1 + num_control_qubits) if num_control_qubits >= 3 else (0, None)
        )
        self.num_qubits = 1 + self.num_control_qubits + self.num_borrowed_ancilla

        self.epsilon = epsilon
        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self):
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        if abs(self.rotation_angle) > self.epsilon:
            if self.num_control_qubits == 0:
                if self.rotation_gate == 'rx':
                    qc.rx(self.rotation_angle, 0)
                elif self.rotation_gate == 'ry':
                    qc.ry(self.rotation_angle, 0)
                elif self.rotation_gate == 'rz':
                    qc.rz(self.rotation_angle, 0)

                return qc

            elif self.num_control_qubits <= 1:
                self._apply_x(qc)

                mcr = UniformlyCRGate(
                    rotation_gate=self.rotation_gate,
                    rotation_angles=[0] * (2 ** self.num_control_qubits - 1) + [self.rotation_angle],
                    num_target_qubits=1 + self.num_control_qubits,
                    epsilon=self.epsilon
                )
                qc_mcr = mcr.circuit()
                qc.compose(qc_mcr, inplace=True)

                self._apply_x(qc)

                return qc

            self._apply_x(qc)

            if self.rotation_gate == 'rx':
                qc_mcr = self._multi_crx()
            elif self.rotation_gate == 'ry':
                qc_mcr = self._multi_cry()
            elif self.rotation_gate == 'rz':
                qc_mcr = self._multi_crz()
            else:
                raise ValueError("rotation_gate must be 'rx', 'ry', or 'rz'")
            qc.compose(qc_mcr, inplace=True)

            self._apply_x(qc)

        return qc

    def _multi_crx(self) -> QuantumCircuit:
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)
        qc_mcx = CircuitCache.get_mcx_circuit(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose_mcx
        )

        qc.rz(-math.pi / 2, self.target_qubits)
        qc.compose(qc_mcx, inplace=True)
        qc.ry(self.rotation_angle / 2, self.target_qubits)
        qc.compose(qc_mcx, inplace=True)
        qc.ry(-self.rotation_angle / 2, self.target_qubits)
        qc.rz(math.pi / 2, self.target_qubits)

        return qc

    def _multi_cry(self) -> QuantumCircuit:
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)
        qc_mcx = CircuitCache.get_mcx_circuit(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose_mcx
        )

        qc.ry(self.rotation_angle / 2, self.target_qubits)
        qc.compose(qc_mcx, inplace=True)
        qc.ry(-self.rotation_angle / 2, self.target_qubits)
        qc.compose(qc_mcx, inplace=True)

        return qc

    def _multi_crz(self) -> QuantumCircuit:
        # qubits: target qubit
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)
        qc_mcx = CircuitCache.get_mcx_circuit(
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose_mcx
        )

        qc.compose(qc_mcx, inplace=True)
        qc.rz(-self.rotation_angle / 2, self.target_qubits)
        qc.compose(qc_mcx, inplace=True)
        qc.rz(self.rotation_angle / 2, self.target_qubits)

        return qc


class MultiCSWAP(BaseMultiControlledGate):
    """

    target qubit 0   ————×————
                         |
    target qubit 1   ————×————
                         |
    control qubits   —/——●————

    borrowed ancilla ——————————

    """
    def __init__(
            self,
            num_control_qubits: int,
            control_state: str | int | None = None,
            use_cache: bool = True,
            decompose_mcx: bool = True,
    ):
        super().__init__(
            num_target_qubits=2,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        (self.num_borrowed_ancilla, self.borrowed_ancilla) = (
            (1, 2 + num_control_qubits) if num_control_qubits >= 2 else (0, None)
        )
        self.num_qubits = 2 + self.num_control_qubits + self.num_borrowed_ancilla

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        # qubits: target qubits
        #         control qubits
        #         borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        mcx = MultiCXGate(
            num_control_qubits=1+self.num_control_qubits,
            use_cache=self.use_cache,
            decompose=self.decompose_mcx
        )
        qc_mcx = mcx.circuit()

        qubits_middle_mcx = [1, 0] + self.control_qubits + ([self.borrowed_ancilla] if self.num_borrowed_ancilla else [])

        qc.compose(qc_mcx, inplace=True)
        qc.compose(qc_mcx, qubits_middle_mcx, inplace=True)
        qc.compose(qc_mcx, inplace=True)

        self._apply_x(qc)

        return qc


class UniformlyCRGate(BaseMultiControlledGate):
    """

    target qubits (rotation)          ——| UCR |——
                                           |
    target qubits (uniformly control) —/———⊘————
                                           |
    control qubits                    —/———●————

    borrowed ancilla                  —————————

    """
    def __init__(
            self,
            rotation_gate: str,
            rotation_angles: List[float] | np.ndarray,
            num_target_qubits: int,
            epsilon: float = 1e-12,
            num_control_qubits: int = 0,
            control_states: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        self.rotation_gate = rotation_gate.lower()
        if self.rotation_gate not in {'ry', 'rz'}:
            raise ValueError("rotation_gate must be 'ry' or 'rz'")

        if num_target_qubits < 2:
            raise ValueError("num_target_qubits must be not less than 2.")

        if isinstance(rotation_angles, list):
            rotation_angles = np.asarray(rotation_angles)
        self.rotation_angles = rotation_angles
        self.num_rotation_angles = len(rotation_angles)
        self.uniformly_rotation_angles = utils.get_uniformly_controlled_rotation_angles(self.rotation_angles)

        self.num_borrowed_ancilla = 1 if num_target_qubits == 2 and num_control_qubits >= 2 else 0
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.epsilon = epsilon
        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        # qubit: target qubits (rotation)
        #        target qubits (uniformly control)
        #        control qubits
        #        borrowed ancilla
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        active_controls = []

        for i in range(len(self.uniformly_rotation_angles)):
            angle = self.uniformly_rotation_angles[i]

            if abs(angle) > self.epsilon:
                for qubit in active_controls:
                    mcx = MultiCXGate(
                        num_control_qubits=1 + self.num_control_qubits,
                        use_cache=self.use_cache,
                        decompose=self.decompose_mcx
                    )
                    qc_mcx = mcx.circuit()
                    extra_ancilla = None
                    if self.borrowed_ancilla is not None:
                        extra_ancilla = self.borrowed_ancilla
                    elif self.num_control_qubits >= 2:
                        extra_ancilla = (
                            self.target_qubits[1] if self.target_qubits[1] != qubit else self.target_qubits[2]
                        )
                    qubits_mcx = [0, qubit] + self.control_qubits + (
                        [extra_ancilla] if extra_ancilla is not None else []
                    )
                    qc.compose(qc_mcx, qubits_mcx, inplace=True)
                active_controls.clear()

                if abs(self.uniformly_rotation_angles[i]) > self.epsilon:
                    mcr = MultiCRGate(
                        rotation_gate=self.rotation_gate,
                        rotation_angle=float(self.uniformly_rotation_angles[i]),
                        num_control_qubits=self.num_control_qubits,
                        use_cache=self.use_cache,
                        decompose_mcx=self.decompose_mcx
                    )
                    qc_mcr = mcr.circuit()
                    qubits_mcr = [0] + self.control_qubits + (
                        [self.target_qubits[1]] if self.num_control_qubits > 2 else [])
                    qc.compose(qc_mcr, qubits_mcr, inplace=True)

            if i < self.num_rotation_angles - 1:
                ctrl_qubit = 1 + utils.different_gray_codes_index(i, i + 1)
            else:
                ctrl_qubit = self.num_target_qubits - 1

            if ctrl_qubit in active_controls:
                active_controls.remove(ctrl_qubit)
            else:
                active_controls.append(ctrl_qubit)

        for qubit in active_controls:
            mcx = MultiCXGate(
                num_control_qubits=1 + self.num_control_qubits,
                use_cache=self.use_cache,
                decompose=self.decompose_mcx
            )
            qc_mcx = mcx.circuit()
            extra_ancilla = None
            if self.borrowed_ancilla is not None:
                extra_ancilla = self.borrowed_ancilla
            elif self.num_control_qubits >= 2:
                extra_ancilla = (
                    self.target_qubits[1] if self.target_qubits[1] != qubit else self.target_qubits[2]
                )
            qubits_mcx = [0, qubit] + self.control_qubits + (
                [extra_ancilla] if extra_ancilla is not None else []
            )
            qc.compose(qc_mcx, qubits_mcx, inplace=True)

        self._apply_x(qc)

        return qc


class GHZGate:
    def __init__(
            self,
            num_target_qubits: int
    ):
        if num_target_qubits < 2:
            raise ValueError("num_target_qubits must be at least 2")
        self.num_target_qubits = num_target_qubits

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_target_qubits)
        qc = self._build_circuit_recursive(
            qc=qc,
            control_qubit=0,
            target_qubits=list(range(1, self.num_target_qubits)),
        )

        return qc

    def _build_circuit_recursive(
            self,
            qc,
            control_qubit: int,
            target_qubits: List[int],
    ) -> QuantumCircuit:
        n = len(target_qubits)

        if n == 0:
            return qc
        if n == 1:
            qc.cx(control_qubit, target_qubits[0])
            return qc

        mid = n // 2

        qc.cx(control_qubit, target_qubits[0])
        qc.cx(control_qubit, target_qubits[mid])

        self._build_circuit_recursive(
            qc=qc,
            control_qubit=target_qubits[0],
            target_qubits=target_qubits[1:mid],
        )

        self._build_circuit_recursive(
            qc=qc,
            control_qubit=target_qubits[mid],
            target_qubits=target_qubits[mid + 1:],
        )

        return qc

    def control(self):
        pass

class MultiCGHZGate(BaseMultiControlledGate):
    """

    target qubits    ——| GHZ |——
                          |
    control qubits   —/———●————

    borrowed ancilla —————————

    """
    def __init__(
            self,
            num_target_qubits: int,
            num_control_qubits: int = 0,
            control_state: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        if self.num_target_qubits <= 1:
            raise ValueError("num_target_qubits of GHZ must be no less than 2")

        self.num_borrowed_ancilla = 1 if self.num_target_qubits == 2 and self.num_control_qubits >= 2 else 0
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla

        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        if self.num_control_qubits == 0:
            ghz = GHZGate(num_target_qubits=self.num_target_qubits)
            qc = ghz.circuit()

            return qc

        qc = QuantumCircuit(self.num_qubits)
        self._apply_x(qc)

        for qubit in self.target_qubits[1:]:
            mcx = MultiCXGate(
                num_control_qubits=1 + self.num_control_qubits,
                use_cache=self.use_cache,
                decompose=self.decompose_mcx
            )
            qc_mcx = mcx.circuit()
            extra_ancilla = None
            if self.borrowed_ancilla is not None:
                extra_ancilla = self.borrowed_ancilla
            elif self.num_control_qubits >= 2:
                extra_ancilla = qubit - 1 if qubit != 1 else 2
            qubits_mcx = [qubit, 0] + self.control_qubits + ([extra_ancilla] if extra_ancilla is not None else [])
            qc.compose(qc_mcx, qubits_mcx, inplace=True)

        self._apply_x(qc)
        return qc


class MultiCShiftGate(BaseMultiControlledGate):
    def __init__(
            self,
            num_target_qubits: int,
            num_control_qubits: int = 0,
            control_state: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_state
        )

        self.num_borrowed_ancilla = 1 if self.num_target_qubits + self.num_control_qubits >= 4 else 0
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

    def circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        for layer in range(self.num_target_qubits-1, -1, -1):
            mcx = MultiCXGate(
                num_control_qubits=layer + self.num_control_qubits,
                use_cache=self.use_cache,
                decompose=self.decompose_mcx
            )
            qc_mcx = mcx.circuit()
            qubit_mcx = [layer] + self.target_qubits[:layer] + self.control_qubits
            if mcx.num_borrowed_ancilla != 0:
                extra_ancilla = self.borrowed_ancilla
                qubit_mcx.append(extra_ancilla)
            qc.compose(qc_mcx, qubit_mcx, inplace=True)

        self._apply_x(qc)

        return qc