from qiskit import QuantumCircuit
from QuantumHadamardProcedure import utils
from QuantumHadamardProcedure.encoding import CompressedStatePreparation, BinaryTreeBlockEncoding, AllOneArray
from QuantumHadamardProcedure.custum_gate import (
    BaseMultiControlledGate, MultiCRGate, MultiCGHZGate, MultiCXGate, GHZGate
)
from typing import List
import numpy as np
import math


class QHP(BaseMultiControlledGate):
    """
                            __________________________________
                            |          ______________        |
    target qubits (working) |  —/——⨁——|             |——⨁——  |
                            |      |  | U_{A_{m-1}} |   |    |
    target qubits (ancilla) |  —/——|——|_____________|———|——  |
                            |      |        ...         |    |
                            |      |   ______________   |    |
    target qubits (working) |  —/——●——|             |———●——  |
                            |         |   U_{A_0}   |        |
    target qubits (ancilla) |  —/—————|_____________|——————  |
                            |________________________________|
                                            |
    control qubits             —/———————————●——————————————

    borrowed ancilla           —/——————————————————————————

    """

    def __init__(
            self,
            arr_list: List[np.ndarray],
            state_or_matrix: str = None,
            num_control_qubits: int = 0,
            control_states: str | int | None = None,
            use_cache: bool = False,
            decompose_mcx: bool = True,
    ):
        self.num_arr = len(arr_list)
        if self.num_arr <= 1:
            raise ValueError("Number of matrices must be greater than 1.")

        if state_or_matrix is None:
            ndim = [np.ndim(arr) for arr in arr_list]
            if np.all(ndim == 1):
                state_or_matrix = 'state'
            elif np.all(ndim == 2):
                state_or_matrix = 'matrix'
            else:
                raise ValueError("Only matrices or states are supported.")
        elif state_or_matrix not in ['state', 'matrix']:
            raise ValueError("Only matrices or states are supported.")
        self.state_or_matrix = state_or_matrix

        self.num_working_qubits = utils.get_num_qubits(arr_list[0])
        self.num_qubits_encodings_arr = (
            self.num_working_qubits if self.state_or_matrix == 'state' else 2 * self.num_working_qubits
        )
        num_target_qubits = self.num_arr * self.num_qubits_encodings_arr

        super().__init__(
            num_target_qubits=num_target_qubits,
            num_control_qubits=num_control_qubits,
            control_states=control_states
        )

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

        encoding_circuit_list = []
        normalization_list = []
        if self.state_or_matrix == 'state':
            for k in range(self.num_arr):
                encoding = {}
                qsp = CompressedStatePreparation(
                    state=arr_list[k],
                    num_target_qubits=self.num_working_qubits,
                    num_control_qubits=self.num_control_qubits,
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
                encoding['circuit'] = qsp.circuit()
                encoding['num_borrowed_ancilla'] = qsp.num_borrowed_ancilla
                encoding_circuit_list.append(encoding)
                normalization_list.append(qsp.normalization_factor)
                arr_list[k] = qsp.state
        else:
            for k in range(self.num_arr):
                encoding = {}
                be = BinaryTreeBlockEncoding(
                    matrix=arr_list[k],
                    num_control_qubits=self.num_control_qubits,
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
                encoding['circuit'] = be.circuit()
                encoding['num_borrowed_ancilla'] = be.num_borrowed_ancilla
                encoding_circuit_list.append(encoding)
                normalization_list.append(be.normalization_factor)
                arr_list[k] = be.matrix

        self.encoding_circuit_list = encoding_circuit_list
        self.normalization_factor = np.prod(normalization_list)
        self.arr_list = arr_list

        num_borrowed_ancilla = 0
        if self.state_or_matrix == 'state':
            if self.num_working_qubits == 1 and self.num_arr == 2 and self.num_control_qubits >= 2:
                num_borrowed_ancilla = 1
        self.num_borrowed_ancilla = num_borrowed_ancilla
        self.num_qubits = self.num_target_qubits + self.num_control_qubits + self.num_borrowed_ancilla
        self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

    def circuit(self):
        qc = QuantumCircuit(self.num_qubits)

        self._apply_x(qc)

        qc_ghz = self._build_ghz_circuit()
        qc_encoding = self._build_encoding_circuit()

        qc.compose(qc_ghz, inplace=True)
        qc.compose(qc_encoding, inplace=True)
        qc.compose(qc_ghz.inverse(), inplace=True)

        self._apply_x(qc)

        return qc

    def _build_ghz_circuit(self):
        qc = QuantumCircuit(self.num_qubits)

        ghz = MultiCGHZGate(
            num_target_qubits=self.num_arr,
            num_control_qubits=self.num_control_qubits,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx,
        )
        qc_ghz = ghz.circuit()

        for i in range(self.num_working_qubits):
            if self.state_or_matrix == 'state':
                if ghz.num_borrowed_ancilla:
                    if self.borrowed_ancilla is not None:
                        exact_ancilla = self.borrowed_ancilla
                    else:
                        exact_ancilla = i + 1 if i != self.num_working_qubits - 1 else i - 1
                else:
                    exact_ancilla = None
                qubits_ghz = (
                        [i + k * self.num_working_qubits for k in range(self.num_arr)]
                        + self.control_qubits
                        + ([exact_ancilla] if exact_ancilla is not None else [])
                )
            else:
                if ghz.num_borrowed_ancilla:
                    exact_ancilla = i + 1
                else:
                    exact_ancilla = None
                qubits_ghz = (
                        [i + k * self.num_qubits_encodings_arr for k in range(self.num_arr)]
                        + self.control_qubits
                        + ([exact_ancilla] if exact_ancilla is not None else [])
                )
            qc.compose(qc_ghz, qubits_ghz, inplace=True)

        return qc

    def _build_encoding_circuit(self):
        qc = QuantumCircuit(self.num_qubits)

        for k in range(self.num_arr):
            encoding_circuit = self.encoding_circuit_list[k]
            if encoding_circuit['num_borrowed_ancilla']:
                if self.borrowed_ancilla is not None:
                    exact_ancilla = self.borrowed_ancilla
                else:
                    exact_ancilla = (
                        self.num_qubits_encodings_arr * (k + 1) if k != self.num_arr - 1 else
                        self.num_qubits_encodings_arr * (k - 1)
                    )
            else:
                exact_ancilla = None
            qubits_encoding = (
                    self.target_qubits[k * self.num_qubits_encodings_arr: (k + 1) * self.num_qubits_encodings_arr]
                    + self.control_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
            )
            qc_encoding = encoding_circuit['circuit']
            qc.compose(qc_encoding, qubits_encoding, inplace=True)

        return qc


class QHMF:
    def __init__(
            self,
            arr: np.ndarray,
            poly: List | np.ndarray,    # sorted by descending power
            mode: str,
            num_subpolys: int = 1,
            use_cache: bool = False,
            decompose_mcx: bool = True
    ):
        if not isinstance(arr, np.ndarray):
            raise ValueError("arr must be numpy array")

        if np.ndim(arr) == 1:
            state_or_matrix = 'state'
        elif np.ndim(arr) == 2:
            state_or_matrix = 'matrix'
        else:
            raise ValueError("Only matrices or states are supported.")
        self.state_or_matrix = state_or_matrix
        self._normalization_of_arr = np.linalg.norm(arr)
        self.arr = arr

        if (not isinstance(poly, list)) and (not isinstance(poly, np.ndarray)):
            raise ValueError("poly must be a list or numpy array")
        poly = np.asarray(poly[::-1])   # sorted by ascending power

        self.num_working_qubits = utils.get_num_qubits(arr)
        self.num_qubits_encoding_arr = (
            self.num_working_qubits if self.state_or_matrix == 'state'  # using the compressed state preparation method
            else 2 * self.num_working_qubits    # using the binary tree block encoding method
        )

        self._normalization_of_all_one_arr = (
            math.sqrt(2 ** self.num_working_qubits) if self.state_or_matrix == 'state'
            else 2 ** self.num_working_qubits
        )

        self.use_cache = use_cache
        self.decompose_mcx = decompose_mcx

        if mode not in ('lcu', 'binary', 'factorization', 'tradeoff'):
            raise ValueError("mode must be either 'lcu' or 'binary' or 'factorization' or 'tradeoff'")
        self.mode = mode

        if mode == 'lcu':
            degree = len(poly) - 1
            while degree >= 0 and poly[degree] == 0:
                degree -= 1
            poly = poly[:degree + 1]
            self.poly = poly
            self.degree = len(self.poly) - 1

            max_num_hprod_arr = 0
            hpower_terms = {}
            lcu_terms = []
            for k in range(self.degree + 1):
                term = []
                if self.poly[k] != 0:
                    k_bin = bin(k)
                    max_num_hprod_arr = max(max_num_hprod_arr, k_bin.count('1'))
                    for pos, bit in enumerate(k_bin[:1:-1]):
                        if bit == '1':
                            if pos not in hpower_terms:
                                arr_hpower = self.arr ** (2 ** pos)
                                hpower_terms[pos] = (arr_hpower, np.linalg.norm(arr_hpower))
                            term.append(hpower_terms[pos][0])
                lcu_terms.append(term)
            self.hprod_terms = hpower_terms
            self.lcu_terms = lcu_terms

            self.num_hprod_qubits = max_num_hprod_arr * self.num_qubits_encoding_arr
            self.num_index_qubits = len(bin(self.degree)[2:])
            num_borrowed_ancilla = 0
            if self.state_or_matrix == 'state':
                if self.num_index_qubits >= 3 and self.num_working_qubits == 1 and max_num_hprod_arr == 1:
                    num_borrowed_ancilla = 1
                elif self.num_index_qubits >= 2 and self.num_working_qubits == 1 and max_num_hprod_arr == 2:
                    num_borrowed_ancilla = 1
                elif self.num_index_qubits >= 2 and self.num_working_qubits == 2 and max_num_hprod_arr == 1:
                    num_borrowed_ancilla = 1
            self.num_borrowed_ancilla = num_borrowed_ancilla
            self.num_qubits = self.num_hprod_qubits + self.num_index_qubits + self.num_borrowed_ancilla

            self.qubits = list(range(self.num_qubits))
            self.working_qubits = self.qubits[:self.num_working_qubits]
            self.encoding_arr_qubits = self.qubits[:self.num_qubits_encoding_arr]
            self.hprod_qubits = self.qubits[:self.num_hprod_qubits]
            self.index_qubits = self.qubits[self.num_hprod_qubits:self.num_hprod_qubits + self.num_index_qubits]
            self.borrowed_ancilla = self.num_qubits - 1 if self.num_borrowed_ancilla else None

            # Compute normalization factor
            normalization_of_terms = [self._normalization_of_all_one_arr, self._normalization_of_arr]
            for k in range(2, self.degree + 1):
                normalization = 1
                if self.poly[k] != 0:
                    for pos, bit in enumerate(bin(k)[:1:-1]):
                        if bit == '1':
                            normalization = normalization * self.hprod_terms[pos][1]
                normalization_of_terms.append(normalization)
            self._normalization_of_terms = np.asarray(normalization_of_terms)
            self.normalization_factor = np.dot(np.abs(self.poly), self._normalization_of_terms)

            # Compute the amplitudes of state preparation unitaries P_L, P_R
            sqrt_poly = np.sqrt(np.abs(self.poly) / self.normalization_factor)
            self.pl_amplitudes = sqrt_poly * np.sqrt(self._normalization_of_terms)
            phase = np.angle(self.poly)
            self.pr_amplitudes = self.pl_amplitudes * np.exp(-1j * phase)

        elif mode == 'binary':
            self.poly = utils.pad_to_power_of_two(poly)
            self.degree = len(poly) - 1

            self._d = len(bin(self.degree)[2:])
            self.hprod_terms = (
                    [(arr, self._normalization_of_arr)]
                    + [(arr ** (2 ** l), np.linalg.norm(arr ** (2 ** l))) for l in range(1, self._d)]
            )

            normalization_of_subpolys = np.zeros((self._d, 2 ** (self._d-1)))
            for kl in range(2 ** (self._d-1)):
                normalization = (
                        self._normalization_of_all_one_arr * abs(self.poly[2 * kl])
                        + self._normalization_of_arr * abs(self.poly[2 * kl + 1])
                )
                normalization_of_subpolys[self._d-1, kl] = normalization
            for l in range(self._d-2, -1, -1):
                for kl in range(2 ** l):
                    normalization = (
                        normalization_of_subpolys[l+1, 2*kl]
                        + normalization_of_subpolys[l+1, 2*kl+1] * self.hprod_terms[self._d-1-l][1]
                    )
                    normalization_of_subpolys[l, kl] = normalization
            self.normalization_factor = normalization_of_subpolys[0, 0]

            if self._d >= 2:
                gamma_angles = np.zeros((self._d-1, 2 ** (self._d - 2)))
                for l in range(self._d-1):
                    for kl in range(2 ** l):
                        if normalization_of_subpolys[l, kl] == 0:
                            gamma = None
                        else:
                            gamma = 2 * math.acos(math.sqrt(
                                normalization_of_subpolys[l+1, 2*kl] / normalization_of_subpolys[l, kl]
                            ))
                        gamma_angles[l, kl] = gamma
            else:
                gamma_angles = None

            theta_angles = []
            phi_angles = []
            psi_angles = []
            for k in range(2 ** (self._d-1)):
                if normalization_of_subpolys[self._d-1, k] == 0:
                    theta = 0
                else:
                    theta = 2 * math.acos(math.sqrt(
                        self._normalization_of_all_one_arr * abs(self.poly[2*k]) / normalization_of_subpolys[self._d-1, k]
                    ))
                phi = - np.angle(self.poly[2*k]) - np.angle(self.poly[2*k+1])
                psi = - np.angle(self.poly[2*k]) + np.angle(self.poly[2*k+1])
                theta_angles.append(theta)
                phi_angles.append(phi)
                psi_angles.append(psi)
            self.gamma_angles = gamma_angles
            self.theta_angles = np.asarray(theta_angles)
            self.phi_angles = np.asarray(phi_angles)
            self.psi_angles = np.asarray(psi_angles)

            self.num_qubits = self._d * (self.num_qubits_encoding_arr + 1)

        elif mode == 'factorization':
            degree = len(poly) - 1
            while degree >= 0 and poly[degree] == 0:
                degree -= 1
            poly = poly[:degree + 1]
            self.poly = poly
            self.degree = len(self.poly) - 1

            self.num_qubits = self.degree * (self.num_qubits_encoding_arr + 1)

            self.roots = np.roots(self.poly[::-1])
            theta_list = []
            phi_list = []
            for root in self.roots:
                theta = 2 * math.atan(math.sqrt(
                    self._normalization_of_all_one_arr * abs(root) / self._normalization_of_arr
                ))
                phi = np.angle(root)
                theta_list.append(theta)
                phi_list.append(phi)
            varphi = -np.sum(phi_list) - 2 * np.angle(self.poly[-1])
            self.theta_list = theta_list
            self.phi_list = phi_list
            self.varphi = varphi

            self.normalization_factor = (abs(self.poly[-1]) * np.prod(
                    np.abs(self.roots) * self._normalization_of_all_one_arr + self._normalization_of_arr)
            )

        else:
            degree = len(poly) - 1
            while degree >= 0 and poly[degree] == 0:
                degree -= 1
            poly = poly[:degree + 1]
            self.poly = poly
            self.degree = len(self.poly) - 1

            if num_subpolys < 1 or num_subpolys > self.degree:
                raise ValueError("num_subpolys must be between 1 and degree of polynomial")
            self.num_subpolys = num_subpolys

            if self.num_subpolys == 1:
                qhmf = QHMF(
                    arr=self.arr,
                    poly=self.poly[::-1],
                    mode='binary',
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
                self._qhmf = [qhmf]
                self.num_qubits = qhmf.num_qubits
                self.normalization_factor = qhmf.normalization_factor
            elif self.num_subpolys == self.degree:
                qhmf = QHMF(
                    arr=self.arr,
                    poly=self.poly[::-1],
                    mode='factorization',
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
                self._qhmf = [qhmf]
                self.num_qubits = qhmf.num_qubits
                self.normalization_factor = qhmf.normalization_factor
            else:
                roots = np.roots(self.poly[::-1])
                roots_of_subpolys = np.array_split(roots, self.num_subpolys)
                self.subpolys = [np.polynomial.polynomial.polyfromroots(r) for r in roots_of_subpolys]
                self.subpolys[0] = self.subpolys[0] * self.poly[-1]

                self._qhmf = [QHMF(
                    arr=self.arr,
                    poly=subpoly[::-1],
                    mode='binary',
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx
                ) for subpoly in self.subpolys]

                self.num_qubits_of_subpolys = [qhmf.num_qubits for qhmf in self._qhmf]
                self.num_qubits = np.sum(self.num_qubits_of_subpolys)
                self.normalization_factor = np.prod([qhmf.normalization_factor for qhmf in self._qhmf])

    def circuit(self):
        if self.mode == 'lcu':
            qc = self._build_lcu_circuit()
        elif self.mode == 'binary':
            qc = self._build_binary_circuit()
        elif self.mode == 'factorization':
            qc = self._build_factorization_circuit()
        else:
            qc = self._build_tradeoff_circuit()

        return qc

    def _build_lcu_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        pl = CompressedStatePreparation(
            state=self.pl_amplitudes,
            num_target_qubits=self.num_index_qubits,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx,
        )
        qc_pl = pl.circuit()
        qubits_prep = self.index_qubits
        qc.compose(qc_pl, qubits_prep, inplace=True)

        if self.poly[0] != 0:
            all_one_arr = AllOneArray(
                num_working_qubits=self.num_working_qubits,
                state_or_matrix=self.state_or_matrix,
                num_control_qubits=self.num_index_qubits,
                control_state=0,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx
            )
            qc_all_one_arr = all_one_arr.circuit()
            if all_one_arr.num_borrowed_ancilla:
                exact_ancilla = self.borrowed_ancilla if self.num_borrowed_ancilla else self.num_qubits_encoding_arr
            else:
                exact_ancilla = None
            qubits_all_one_arr = (
                    self.encoding_arr_qubits + self.index_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
            )
            qc.compose(qc_all_one_arr, qubits_all_one_arr, inplace=True)

        for k in range(1, self.degree + 1):
            ck = self.poly[k]
            if ck != 0:
                term = self.lcu_terms[k]
                if len(term) == 1:
                    if self.state_or_matrix == 'state':
                        encoding = CompressedStatePreparation(
                            state=term[0],
                            num_target_qubits=self.num_working_qubits,
                            num_control_qubits=self.num_index_qubits,
                            control_states=k,
                            use_cache=self.use_cache,
                            decompose_mcx=self.decompose_mcx,
                        )
                    else:
                        encoding = BinaryTreeBlockEncoding(
                            matrix=term[0],
                            num_control_qubits=self.num_index_qubits,
                            control_states=k,
                            use_cache=self.use_cache,
                            decompose_mcx=self.decompose_mcx,
                        )
                    qc_encoding = encoding.circuit()
                    if encoding.num_borrowed_ancilla:
                        exact_ancilla = (
                            self.borrowed_ancilla if self.num_borrowed_ancilla else self.num_qubits_encoding_arr
                        )
                    else:
                        exact_ancilla = None
                    qubits_encoding = (
                            self.encoding_arr_qubits
                            + self.index_qubits
                            + ([exact_ancilla] if exact_ancilla is not None else [])
                    )
                    qc.compose(qc_encoding, qubits_encoding, inplace=True)

                else:
                    qhp = QHP(
                        arr_list=term,
                        state_or_matrix=self.state_or_matrix,
                        num_control_qubits=self.num_index_qubits,
                        control_states=k,
                        use_cache=self.use_cache,
                        decompose_mcx=self.decompose_mcx,
                    )
                    qc_qhp = qhp.circuit()
                    if qhp.num_borrowed_ancilla:
                        exact_ancilla = (
                            self.borrowed_ancilla if self.num_borrowed_ancilla
                            else self.num_qubits_encoding_arr * len(term)
                        )
                    else:
                        exact_ancilla = None
                    qubits_qhp = (
                            self.hprod_qubits[:self.num_qubits_encoding_arr * len(term)]
                            + self.index_qubits
                            + ([exact_ancilla] if exact_ancilla is not None else [])
                    )
                    qc.compose(qc_qhp, qubits_qhp, inplace=True)

        pr = CompressedStatePreparation(
            state=self.pr_amplitudes,
            num_target_qubits=self.num_index_qubits,
            use_cache=self.use_cache,
            decompose_mcx=self.decompose_mcx,
        )
        qc_pr = pr.circuit()
        qc.compose(qc_pr.inverse(), qubits_prep, inplace=True)

        return qc

    def _build_binary_circuit(self) -> QuantumCircuit:
        qc = self._build_binary_circuit_recursive(
            l=0,
            kl=0,
            control_qubits=[]
        )

        return qc

    def _build_binary_circuit_recursive(self, l, kl, control_qubits) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        if l == self._d - 1:
            mcrz_phi = MultiCRGate(
                rotation_gate='rz',
                rotation_angle=self.phi_angles[kl],
                num_control_qubits=len(control_qubits),
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx,
            )
            qc_mcrz_phi = mcrz_phi.circuit()
            if mcrz_phi.num_borrowed_ancilla != 0:
                exact_ancilla = self.num_qubits_encoding_arr - 1
            else:
                exact_ancilla = None
            qubits_mcr = (
                    [self.num_qubits_encoding_arr]
                    + control_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
            )
            qc.compose(qc_mcrz_phi, qubits_mcr, inplace=True)

            mcry = MultiCRGate(
                rotation_gate='ry',
                rotation_angle=self.theta_angles[kl],
                num_control_qubits=len(control_qubits),
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx,
            )
            qc_mcry = mcry.circuit()
            qc.compose(qc_mcry, qubits_mcr, inplace=True)

            all_one_arr = AllOneArray(
                num_working_qubits=self.num_working_qubits,
                state_or_matrix=self.state_or_matrix,
                num_control_qubits=1+len(control_qubits),
                decompose_mcx=self.decompose_mcx,
            )
            qc_all_one_arr = all_one_arr.circuit()
            qubits_encoding = list(range(self.num_qubits_encoding_arr + 1)) + control_qubits
            if all_one_arr.num_borrowed_ancilla:
                exact_ancilla = self.num_qubits_encoding_arr + 1
            else:
                exact_ancilla = None
            qubits_encoding_all_one_arr = qubits_encoding + ([exact_ancilla] if exact_ancilla is not None else [])
            qc.x(self.num_qubits_encoding_arr)
            qc.compose(qc_all_one_arr, qubits_encoding_all_one_arr, inplace=True)
            qc.x(self.num_qubits_encoding_arr)

            if self.state_or_matrix == 'state':
                encoding_arr = CompressedStatePreparation(
                    state=self.arr,
                    num_target_qubits=self.num_qubits_encoding_arr,
                    num_control_qubits=1+len(control_qubits),
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
            else:
                encoding_arr = BinaryTreeBlockEncoding(
                    matrix=self.arr,
                    num_control_qubits=1+len(control_qubits),
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
            qc_encoding_arr = encoding_arr.circuit()
            if encoding_arr.num_borrowed_ancilla != 0:
                exact_ancilla = self.num_qubits_encoding_arr + 1
            else:
                exact_ancilla = None
            qubits_encoding_arr = qubits_encoding + ([exact_ancilla] if exact_ancilla is not None else [])
            qc.compose(qc_encoding_arr, qubits_encoding_arr, inplace=True)

            mcrz_psi = MultiCRGate(
                rotation_gate='rz',
                rotation_angle=self.psi_angles[kl],
                num_control_qubits=len(control_qubits),
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx,
            )
            qc_mcrz_psi = mcrz_psi.circuit()
            qc.compose(qc_mcrz_psi, qubits_mcr, inplace=True)

            qc.compose(qc_mcry.inverse(), qubits_mcr, inplace=True)

            return qc

        else:
            if self.gamma_angles[l, kl] is not None:
                mcry = MultiCRGate(
                    rotation_gate='ry',
                    rotation_angle=self.gamma_angles[l, kl].item(),
                    num_control_qubits=len(control_qubits),
                    use_cache=self.use_cache,
                    decompose_mcx=self.decompose_mcx,
                )
                qc_mcry = mcry.circuit()
                idx_qubit = (self._d - l) * (self.num_qubits_encoding_arr + 1) - 1
                if mcry.num_borrowed_ancilla != 0:
                    exact_ancilla = idx_qubit - 1
                else:
                    exact_ancilla = None
                qubits_mcry = [idx_qubit] + control_qubits + ([exact_ancilla] if exact_ancilla is not None else [])
                qc.compose(qc_mcry, qubits_mcry, inplace=True)

                qc_poly0 = self._build_binary_circuit_recursive(
                    l=l+1,
                    kl=2*kl,
                    control_qubits=[idx_qubit]+control_qubits
                )
                num_qubits_encoding_poly = (self.num_qubits_encoding_arr + 1) * (self._d - l - 1)
                qubits_encoding_poly = list(range(num_qubits_encoding_poly))
                qc.x(idx_qubit)
                qc.compose(qc_poly0, inplace=True)
                qc.x(idx_qubit)

                mcx = MultiCXGate(
                    num_control_qubits=2+len(control_qubits),
                    use_cache=self.use_cache,
                    decompose=self.decompose_mcx
                )
                qc_mcx = mcx.circuit()
                for i in range(self.num_qubits_encoding_arr):
                    if mcx.num_borrowed_ancilla != 0:
                        exact_ancilla = i + 1
                    else:
                        exact_ancilla = None
                    qubits_mcx = (
                            [len(qubits_encoding_poly) + i, i, idx_qubit] + control_qubits
                            + ([exact_ancilla] if exact_ancilla is not None else [])
                    )
                    qc.compose(qc_mcx, qubits_mcx, inplace=True)

                qc_poly1 = self._build_binary_circuit_recursive(
                    l=l+1,
                    kl=2*kl+1,
                    control_qubits=[idx_qubit]+control_qubits
                )
                qc.compose(qc_poly1, inplace=True)

                if self.state_or_matrix == 'state':
                    encoding = CompressedStatePreparation(
                        state=self.hprod_terms[self._d-1-l][0],
                        num_target_qubits=self.num_qubits_encoding_arr,
                        num_control_qubits=1+len(control_qubits),
                        use_cache=self.use_cache,
                        decompose_mcx=self.decompose_mcx
                    )
                else:
                    encoding = BinaryTreeBlockEncoding(
                        matrix=self.hprod_terms[self._d-1-l][0],
                        num_control_qubits=1+len(control_qubits),
                        use_cache=self.use_cache,
                        decompose_mcx=self.decompose_mcx
                    )
                qc_encoding = encoding.circuit()
                if encoding.num_borrowed_ancilla != 0:
                    exact_ancilla = num_qubits_encoding_poly - 1
                else:
                    exact_ancilla = None
                qubits_encoding = (
                    list(range(num_qubits_encoding_poly, num_qubits_encoding_poly + self.num_qubits_encoding_arr + 1))
                    + control_qubits
                    + ([exact_ancilla] if exact_ancilla is not None else [])
                )
                qc.compose(qc_encoding, qubits_encoding, inplace=True)

                for i in range(self.num_qubits_encoding_arr):
                    if mcx.num_borrowed_ancilla != 0:
                        exact_ancilla = i + 1
                    else:
                        exact_ancilla = None
                    qubits_mcx = (
                            [len(qubits_encoding_poly) + i, i, idx_qubit]
                            + control_qubits
                            + ([exact_ancilla] if exact_ancilla is not None else [])
                    )
                    qc.compose(qc_mcx, qubits_mcx, inplace=True)

                qc.compose(qc_mcry.inverse(), qubits_mcry, inplace=True)

            return qc

    def _build_factorization_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        if self.degree >= 2:
            qc_ghz = self._build_ghz_circuit(
                num_registers=self.degree,
                num_qubits_per_register=self.num_working_qubits
            )
            qubits_ghz = []
            for i in range(self.degree):
                start = i * (self.num_qubits_encoding_arr + 1)
                qubits_ghz.extend(range(start, start + self.num_working_qubits))
        else:
            qc_ghz = None
            qubits_ghz = []

        if self.state_or_matrix == 'state':
            encoding = CompressedStatePreparation(
                state=self.arr,
                num_target_qubits=self.num_working_qubits,
                num_control_qubits=1,
                control_states=0,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx,
            )
        else:
            encoding = BinaryTreeBlockEncoding(
                matrix=self.arr,
                num_control_qubits=1,
                control_states=0,
                use_cache=self.use_cache,
                decompose_mcx=self.decompose_mcx,
            )
        qc_arr = encoding.circuit()

        all_one_arr = AllOneArray(
            num_working_qubits=self.num_working_qubits,
            state_or_matrix=self.state_or_matrix,
            num_control_qubits=1,
            control_state=1,
            decompose_mcx=self.decompose_mcx,
        )
        qc_all_one_arr = all_one_arr.circuit()

        if qc_ghz:
            qc.compose(qc_ghz, qubits_ghz, inplace=True)

        for i in range(self.degree):
            qc.ry(self.theta_list[i], self.num_qubits_encoding_arr + i * (self.num_qubits_encoding_arr + 1))
            qc.rz(self.phi_list[i], self.num_qubits_encoding_arr + i * (self.num_qubits_encoding_arr + 1))

            qubits_encoding = list(range(
                i * (self.num_qubits_encoding_arr + 1), (i + 1) * (self.num_qubits_encoding_arr + 1)
            ))
            qc.compose(qc_arr, qubits_encoding, inplace=True)
            qc.compose(qc_all_one_arr, qubits_encoding, inplace=True)

            qc.ry(self.theta_list[i], self.num_qubits_encoding_arr + i * (self.num_qubits_encoding_arr + 1))

        qc.rz(self.varphi, self.num_qubits_encoding_arr)

        if qc_ghz:
            qc.compose(qc_ghz.inverse(), qubits_ghz, inplace=True)

        return qc

    def _build_tradeoff_circuit(self) -> QuantumCircuit:
        if self.num_subpolys == 1:
            qhmf = self._qhmf[0]
            qc = qhmf.circuit()
        elif self.num_subpolys == self.degree:
            qhmf = self._qhmf[0]
            qc = qhmf.circuit()
        else:
            qc = QuantumCircuit(self.num_qubits)

            qc_ghz = self._build_ghz_circuit(
                num_registers=self.num_subpolys,
                num_qubits_per_register=self.num_working_qubits
            )
            qubits_ghz = []
            for i in range(self.num_subpolys):
                start = int(np.sum(self.num_qubits_of_subpolys[:i]))
                qubits_ghz.extend(range(start, start + self.num_working_qubits))
            qc.compose(qc_ghz, qubits_ghz, inplace=True)

            for idx in range(self.num_subpolys):
                qhmf = self._qhmf[idx]
                qc_qhmf = qhmf.circuit()
                start = int(np.sum(self.num_qubits_of_subpolys[:idx]))
                qubits_qhmf = list(range(start, start + self.num_qubits_of_subpolys[idx]))
                qc.compose(qc_qhmf, qubits_qhmf, inplace=True)

            qc.compose(qc_ghz.inverse(), qubits_ghz, inplace=True)

        return qc

    @staticmethod
    def _build_ghz_circuit(
            num_registers: int,
            num_qubits_per_register: int
    ) -> QuantumCircuit:
        num_qubits = num_registers * num_qubits_per_register
        qc = QuantumCircuit(num_qubits)

        ghz = GHZGate(num_target_qubits=num_registers)
        qc_ghz = ghz.circuit()

        for i in range(num_qubits_per_register):
            qubit_ghz = [i + k * num_qubits_per_register for k in range(num_registers)]
            qc.compose(qc_ghz, qubit_ghz, inplace=True)

        return qc