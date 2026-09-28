import numpy as np
import sympy as sp
from typing import Tuple, List, cast
from chiku import remez
from numpy.polynomial import Polynomial


def is_all_real(arr, tol=1e-10):
    """Check whether all elements of an array are real.

    For complex arrays, elements with imaginary part smaller than `tol` are
    considered real.

    Args:
        arr: Input array.
        tol: Tolerance for the imaginary part. Defaults to 1e-10.

    Returns:
        bool: True if all elements are real within tolerance, False otherwise.
    """
    arr = np.asarray(arr)
    if not np.iscomplexobj(arr):
        return True
    return np.all(np.abs(arr.imag) < tol)


def get_num_qubits(arr: np.ndarray) -> int:
    """Return the number of qubits needed to represent the dimension of `arr`.

    The dimension is taken as the maximum shape value for 1D or 2D arrays.
    The number of qubits is the bit length of (dim - 1).

    Args:
        arr: Input array with ndim 1 or 2.

    Returns:
        int: Number of qubits.

    Raises:
        ValueError: If `arr.ndim` is not 1 or 2.
    """
    if arr.ndim <= 2:
        dim = max(arr.shape)
    else:
        raise ValueError("arr.ndim must be 1 or 2")

    if dim > 0:
        n = len(bin(dim-1)[2:])
        return n
    else:
        return 0


def pad_to_power_of_two(arr: np.ndarray | List) -> np.ndarray | List:
    """Pad an array or list to the next power of two in each dimension.

    The padding is applied symmetrically with zeros. If the input is already
    square and its first dimension is a power of two, it is returned unchanged.

    Args:
        arr: Input array or list.

    Returns:
        Padded array or list with the same type as the input.
    """
    is_list = isinstance(arr, list)
    if is_list:
        arr_np = np.asarray(arr)
    elif isinstance(arr, np.ndarray):
        arr_np = cast(np.ndarray, arr)
    else:
        raise ValueError("arr must be list or np.ndarray or list")

    if arr_np.size == 0 or arr_np.ndim == 0:
        return arr if is_list else arr_np

    shape = arr_np.shape
    first_dim = shape[0]
    all_equal = all(d == first_dim for d in shape)
    is_power_of_two = (first_dim > 0) and (first_dim & (first_dim - 1) == 0)
    if all_equal and is_power_of_two:
        return arr if is_list else arr_np

    max_dim = max(shape)
    target = 1 << (max_dim - 1).bit_length()
    pad_width = [(0, target - d) for d in shape]
    padded = np.pad(arr_np, pad_width=pad_width, mode='constant', constant_values=0)

    return padded.tolist() if is_list else padded


def gray_code(x: int) -> int:
    """Return the Gray code of a non-negative integer.

    Args:
        x: A non-negative integer.

    Returns:
        int: The Gray code of `x`.
    """
    return x ^ (x >> 1)


def different_gray_codes_index(
        number1: int,
        number2: int
) -> int:
    """Get the index of the differing bit between two Gray codes.

    Args:
        number1: A non-negative integer.
        number2: A non-negative integer.

    Returns:
        int: The bit index where the Gray codes differ, or -1 if they are equal.
    """
    gray_code_number1 = gray_code(number1)
    gray_code_number2 = gray_code(number2)
    diff = gray_code_number1 ^ gray_code_number2  # Calculate XOR of the two Gray codes
    if diff == 0:
        return -1
    bit_pos = diff.bit_length() - 1

    return bit_pos


def gray_permutation(arr: np.ndarray) -> np.ndarray:
    """Reorder the rows of an array according to the Gray code permutation.

    Args:
        arr: Input array of shape (n, ...). The first dimension is permuted.

    Returns:
        np.ndarray: Array with rows reordered by the Gray code sequence.
    """
    n = arr.shape[0]
    perm = np.array([gray_code(i) for i in range(n)])

    return arr[perm]


def angle_search_binary_tree(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform binary tree angle decomposition on each column of `arr`.

    Each column is treated as a vector whose length must be a power of two.
    The decomposition yields angles for every level of the binary tree and
    the final norms.

    Args:
        arr: Input array of shape (m, n). m must be a power of two.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - final_norms: Array of shape (n,) containing the total norm of each column.
            - all_angles: Array of shape (m-1, n) containing angles for all levels
              ordered from bottom to top.

    Raises:
        ValueError: If the number of rows `m` is not a power of two.
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    m, n = arr.shape
    if m & (m - 1) != 0:
        raise ValueError("Number of rows m must be a power of 2")
    if m == 0:
        return np.array([]), np.array([])

    angle_layers = []
    current = arr  # shape (m, n)

    while current.shape[0] > 1:
        even = current[0::2, :]  # (m/2, n)
        odd = current[1::2, :]  # (m/2, n)
        norms = np.sqrt(even ** 2 + odd ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            angles = np.arccos(np.clip(even / norms, -1.0, 1.0))
            # angles[norms < 1e-12] = 0.0
        angle_layers.append(angles)  # angles for current layer
        current = norms  # input for next layer

    # current now has shape (1, n), i.e., the total norm for each column
    final_norms = current.flatten()
    # Concatenate angles from bottom to top
    all_angles = np.concatenate(angle_layers[::-1], axis=0)  # 垂直堆叠

    return final_norms, all_angles


def positive_transform(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute norms and angles from pairs of elements in a real array.

    For each column, pairs (even, odd) are transformed to (norm, angle) where
    norm = hypot(even, odd) and angle = atan2(odd, even) modulo 2π.

    Args:
        arr: Input array of shape (m, n). m must be even.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - norms: Array of shape (m//2, n).
            - angles: Array of shape (m//2, n).
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    even = arr[0::2, :]
    odd = arr[1::2, :]

    norms = np.hypot(even, odd)
    angles = np.arctan2(odd, even) % (2 * np.pi)

    return norms, angles


def binarytree_vector(
        vector: np.ndarray,
        mode: str,
        is_real=False
) -> np.ndarray:
    """Convert a real or complex vector into its binary tree representation.

    Depending on `mode`, the function returns either the norm angles or the
    phase angles of the binary tree decomposition.

    Args:
        vector: Input vector of length a power of two.
        mode: Either 'norm' or 'phase'.
        is_real: If True, treat the vector as real. Defaults to False.

    Returns:
        np.ndarray: The binary tree representation (norm angles or phase angles).

    Raises:
        ValueError: If the vector length is not a power of two, or if `mode`
            is not 'norm' or 'phase'.
    """
    vector = np.asarray(vector).flatten()
    n = len(vector)
    if (n & (n - 1)) != 0:
        raise ValueError("Vector length must be a power of 2")

    if mode == 'norm':
        if is_real:
            norms, norm_angles = positive_transform(vector)
        else:
            norms, norm_angles = angle_search_binary_tree(vector)

        # Continue building up to the root node
        while len(norms) > 1:
            norms, angle_list = angle_search_binary_tree(norms)
            norm_angles = np.append(angle_list, norm_angles)

        return norm_angles * 2

    elif mode == 'phase':
        levels = int(np.log2(n))
        current = vector.copy()

        temp = None

        for i in range(1, levels + 1):
            l = 1 << (levels - i)
            temp = current[:2 * l].copy()

            for j in range(l):
                a = temp[2 * j]
                b = temp[2 * j + 1]
                current[j] = (a + b) / 2
                current[l + j] = -a + b

        # Handle the root node
        if temp is None:
            current[0] = -current[0]
        else:
            current[0] = -temp[0] - temp[1]

        return current

    else:
        raise ValueError("The mode must be 'norm' or 'phase'")


def binarytree_norm(
        arr: np.ndarray,
        is_real: bool = False
) -> np.ndarray:
    """Compute norm angles for a matrix using binary tree decomposition.

    Args:
        arr: Input array of shape (m, n). m must be a power of two.
        is_real: If True, treat the input as real. Defaults to False.

    Returns:
        np.ndarray: Norm angles of shape (m-1, n), multiplied by 2.
    """
    all_angles = []  # collect angles from bottom to top (reversed later)

    # First layer
    if is_real:
        norms, angles = positive_transform(arr)
    else:
        norms, angles = angle_search_binary_tree(arr)
    all_angles.append(angles)  # bottom layer angles
    current = norms

    # Subsequent layers
    while current.ndim > 1 and current.shape[0] > 1:
        norms, angles = angle_search_binary_tree(current)
        all_angles.append(angles)
        current = norms

    # Merge angles (order from root to leaves)
    result_angles = np.concatenate(all_angles[::-1], axis=0)

    return result_angles * 2


def binarytree_phase(arr: np.ndarray) -> np.ndarray:
    """Compute phase angles for a matrix using binary tree decomposition.

    Args:
        arr: Input array of shape (m, n). m must be a power of two.

    Returns:
        np.ndarray: Phase angles of shape (m, n).
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    m = arr.shape[0]

    levels = int(np.log2(m))
    current = arr.copy()

    temp = None

    for i in range(1, levels + 1):
        l = 1 << (levels - i)
        temp = current[:2 * l, :].copy()

        a = temp[0::2, :]
        b = temp[1::2, :]
        current[:l, :] = (a + b) / 2
        current[l:2*l, :] = - a + b

    if temp is None:
        current[0, :] = -current[0, :]
    else:
        current[0, :] = -temp[0, :] - temp[1, :]

    return current


def rotation_angles_matrix(
        matrix: np.ndarray,
        is_real: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rotation angle matrices from a given matrix.

    For real matrices, returns norm angles and column norm angles.
    For complex matrices, returns norm angles, phase angles, and column norm angles.

    Args:
        matrix: Input matrix of shape (2^n, 2^n).
        is_real: If True, treat the matrix as real. Defaults to False.

    Returns:
        If `is_real` is True:
            Tuple[np.ndarray, np.ndarray]: (norm_angles, col_norm_angles)
        Otherwise:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (norm_angles, phase_angles, col_norm_angles)
    """
    if is_real:
        norm_angles = binarytree_norm(matrix, is_real=True)

        col_norm = np.linalg.norm(matrix, axis=0)
        col_norm_angles = binarytree_norm(col_norm, is_real=True)

        return norm_angles, col_norm_angles.flatten()

    else:
        norm_matrix = np.abs(matrix)
        phase_matrix = np.angle(matrix)

        norm_angles = binarytree_norm(norm_matrix, is_real=False)  # (m-1, n)
        phase_angles = binarytree_phase(phase_matrix)  # (m, n)

        col_norm = np.linalg.norm(matrix, axis=0)
        col_norm_angles = binarytree_vector(col_norm, 'norm', is_real=True)  # 长度 (m-1)

        return norm_angles, phase_angles, col_norm_angles.flatten()


def scaled_fast_walsh_hadamard_transform(arr: np.ndarray) -> np.ndarray:
    """Compute the Scaled Fast Walsh-Hadamard Transform of an array.

    Args:
        arr: Input array of shape (n, ...). n must be a power of two.

    Returns:
        np.ndarray: Transformed array of the same shape as input.

    Raises:
        ValueError: If the first dimension `n` is not a power of two.
    """
    arr = np.asarray(arr, dtype=float).copy()
    n = arr.shape[0]
    if n & (n - 1) != 0:
        raise ValueError("n must be power of 2")

    for h in range(1, int(np.log2(n)) + 1):
        step = 1 << h  # 2^h
        half = 1 << (h - 1)  # 2^{h-1}

        for i in range(0, n, step):
            for j in range(i, i + half):
                x = arr[j].copy()
                y = arr[j + half]
                arr[j] = (x + y) / 2
                arr[j + half] = (x - y) / 2

    return arr


def get_uniformly_controlled_rotation_angles(rotation_angles: np.ndarray | List[float]) -> np.ndarray:
    """Compute uniformly controlled rotation angles from given rotation angles.

    The input angles are transformed using the scaled fast Walsh-Hadamard
    transform and then permuted by the Gray code permutation.

    Args:
        rotation_angles: Input rotation angles as an array or list.

    Returns:
        np.ndarray: Flattened uniformly controlled rotation angles.
    """
    rotation_angles = np.asarray(rotation_angles).reshape((-1, 1))
    uniformly_rotation_angles = gray_permutation(
        scaled_fast_walsh_hadamard_transform(rotation_angles)
    )

    return uniformly_rotation_angles.flatten()


def taylor_approximation(func, x0: float, degree: int) -> List:
    """Compute the Taylor polynomial coefficients of a function.

    The polynomial is expanded around `x0` up to the given `degree`.

    Args:
        func: A callable that takes a sympy symbol and returns a sympy expression.
        x0: Expansion point.
        degree: Degree of the Taylor polynomial.

    Returns:
        List: Coefficients sorted by descending power.
    """
    x = sp.symbols('x')
    expr = func(x)

    series = expr.series(x, x0, degree + 1).removeO()

    t = sp.symbols('t')
    poly_t = series.subs(x, t + x0).expand()
    poly = [float(poly_t.coeff(t, k)) for k in range(degree, -1, -1)]

    return poly


def chebyshev_approximation(func, degree, domain):
    """Compute a Chebyshev approximation of a function.

    Args:
        func: The function to approximate.
        degree: Degree of the Chebyshev polynomial.
        domain: Tuple (a, b) defining the interval.

    Returns:
        Tuple: (poly, coeffs) where `poly` is a numpy Polynomial and `coeffs`
        are the coefficients sorted by descending power.
    """
    poly_chebyshev = np.polynomial.Chebyshev.interpolate(func, degree, domain=domain)
    poly = poly_chebyshev.convert(kind=Polynomial)
    coeffs = poly.coef[::-1]  # sorted by descending power

    return poly, coeffs


def minmax_approximation(func, degree, domain):
    """Compute a minimax (Remez) approximation of a function.

    Args:
        func: The function to approximate.
        degree: Degree of the polynomial.
        domain: Tuple (a, b) defining the interval.

    Returns:
        Tuple: (poly, coeffs) where `poly` is a numpy Polynomial and `coeffs`
        are the coefficients sorted by descending power.
    """
    a, b = domain
    r = remez.RemezSolver(func, degree=degree, frange=(a, b))
    coeffs = np.asarray(r.get_coeffs())
    poly = Polynomial(coeffs)
    coeffs = poly.coef[::-1]  # sorted by descending power

    return poly, coeffs


def hadamard_polynomial(poly, array):
    """Evaluate a polynomial on an array using element-wise operations.

    The polynomial coefficients are given in descending order. The evaluation
    uses the Horner-like scheme: result = poly[0] + poly[1]*x + poly[2]*x^2 + ...

    Args:
        poly: Coefficients of the polynomial in descending order.
        array: Input array.

    Returns:
        np.ndarray: Result of applying the polynomial element-wise to `array`.
    """
    array = np.asarray(array)
    poly = poly[::-1]
    if poly[0] == 0:
        result = np.zeros_like(array)
    else:
        result = poly[0] * np.ones_like(array)

    for k in range(1, len(poly)):
        result = result + poly[k] * (array ** k)

    return result