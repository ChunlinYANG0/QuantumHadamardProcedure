import copy

import numpy as np
from typing import Tuple
from QuantumHadamardProcedure import utils
from dible import anglecompute


# def angle_search_binary_tree(vector):
#     """
#     Compute the angles of a normalized real vector in its binary tree structure representation.
#
#     This function takes a normalized real vector and returns the norms and angles at each level of the binary tree.
#     The binary tree structure is a hierarchical representation of the vector, where each node represents a rotation or scaling operation.
#
#     Args:
#         vector (numpy.array): A normalized real vector in ℝ^(2^n), where n is a non-negative integer.
#
#     Returns:
#         tuple: A tuple containing two numpy arrays: the first array contains the norms at each level of the binary tree,
#                and the second array contains the corresponding angles.
#     """
#     length = len(vector)
#     if length == 2:
#         norms = np.array([np.linalg.norm(vector)])
#         if norms[0] == 0:
#             angles = np.array([0])
#         else:
#             angles = np.array([np.arccos(vector[0] / norms[0])])
#     else:
#         first_half_norm, first_half_angles = angle_search_binary_tree(vector[:length//2])
#         second_half_norm, second_half_angles = angle_search_binary_tree(vector[length//2:])
#         norms = np.append(first_half_norm, second_half_norm)
#         angles = np.append(first_half_angles, second_half_angles)
#
#     return norms, angles
#
#
# def positive_transform(vector):
#     """
#     Transforms an input vector into arrays of sum of squared amplitudes and corresponding angles.
#
#     This function takes a vector and treats every two elements as the real and imaginary parts of a complex number.
#     It calculates the sum of squared amplitudes and the corresponding angles for each pair of elements.
#     If a pair of elements are both zero, their sum of squared amplitudes is zero and the angle is also zero.
#
#     Args:
#         vector (numpy.array): The input vector, which should have an even length.
#
#     Returns:
#         tuple: A tuple containing two arrays.
#             - sum_square_amplitude (numpy.ndarray): An array of the sum of squared amplitudes for each pair of elements,
#                 with length half of the input vector.
#             - angles (numpy.ndarray): An array of the corresponding angles for each pair of elements,
#                 with length half of the input vector, in the range [0, 2π).
#     """
#     length = len(vector)
#     angles = np.zeros(length // 2)
#     sum_square_amplitude = np.zeros(length // 2)
#     for i in range(length // 2):
#         pair = np.array([vector[2 * i], vector[2 * i + 1]])
#         if np.all(pair == 0):
#             sum_square_amplitude[i] = 0
#             angles[i] = 0
#         else:
#             sum_square_amplitude[i] = np.linalg.norm(pair)
#             pair = pair / sum_square_amplitude[i]
#             complex_num = pair[0] + pair[1] * 1j
#             angles[i] = np.mod(np.angle(complex_num), 2 * np.pi)
#
#     return sum_square_amplitude, angles
#
#
# def binarytree_vector(vector, mode, is_real=False):
#     """
#     Convert a real or complex vector into its binary tree structure representation.
#
#     This function takes a vector and a mode, and returns either the norm binary tree vector "norm_angles"
#         or the phase binary tree vector "phase_angles".
#     The binary tree structure is a hierarchical representation of the vector,
#         where each node represents a rotation or scaling operation.
#
#     Args:
#         vector (numpy.array): A real or complex vector.
#         mode (str): The mode of the output, either 'norm' for the norm binary tree vector or 'phase' for the phase binary tree vector.
#         is_real (bool, optional): A flag indicating whether the input vector is real. Defaults to False.
#
#     Returns:
#         numpy.array: The binary tree structure of the input vector as a norm binary tree vector or a phase binary tree vector.
#
#     Raises:
#         ValueError: If the mode is not 'norm' or 'phase'.
#
#     Note:
#         The function assumes that the input vector is normalized and that its length is a power of 2.
#     """
#     vector = vector.flatten()
#     n = int(np.log2(len(vector)))
#     if mode == 'norm':
#         if is_real:
#             norms, norm_angles = positive_transform(vector)
#         else:
#             norms, norm_angles = angle_search_binary_tree(vector)
#         while len(norms) != 1:
#             norms, angle_list = angle_search_binary_tree(norms)
#             norm_angles = np.append(angle_list, norm_angles)
#         norm_angles = norm_angles * 2
#
#         return norm_angles
#
#     elif mode == 'phase':
#         # phase_angles = np.dot(tools.phase_angle_matrix_inverse(len(vector)), vector)
#         global temp
#         for i in range(1, n + 1):
#             l = 2 ** (n - i)
#             temp = copy.deepcopy(vector[: 2 * l])
#             for j in range(1, l + 1):
#                 vector[j - 1] = (temp[2 * j - 2] + temp[2 * j - 1]) / 2
#                 vector[l + j - 1] = - temp[2 * j - 2] + temp[2 * j - 1]
#         vector[0] = -temp[0] - temp[1]
#         return vector
#
#     else:
#         raise ValueError("The mode must be 'norm' or 'phase'")
#
#
# def rotation_angles_matrix(matrix, is_real=False):
#     """
#     Compute two rotation angles matrices from a given complex matrix.
#
#     This function takes a complex matrix A of size 2^n x 2^n and returns two rotation angles matrices.
#     The first matrix, "norm_angles", contains the norms of the columns of the matrix.
#     The second matrix, "phase_angles", contains the corresponding phase angles.
#
#     Args:
#         matrix (numpy.array): A complex matrix A in ℂ^{2^n × 2^n}.
#         is_real (bool, optional): A flag indicating whether the input matrix is real. Defaults to False.
#
#     Returns:
#         tuple: A tuple containing two numpy arrays: the first array is "norm_angles" and the second is "phase_angles".
#
#     Note:
#         The function assumes that the input matrix is of size 2^n x 2^n, where n is a non-negative integer.
#     """
#     if is_real:
#         norm_angles = np.zeros([matrix.shape[0] - 1, matrix.shape[1] + 1])
#
#         # Compute the rotation angle matrix "norm_angles"
#         for col in range(matrix.shape[1]):
#             norm_angles[:, col] = binarytree_vector(matrix[:, col], 'norm', True)
#         column_norm = np.array([np.linalg.norm(matrix[:, j]) for j in range(matrix.shape[1])])
#         norm_angles[:, matrix.shape[1]] = binarytree_vector(column_norm, 'norm')
#
#         return norm_angles
#
#     else:
#         norm_matrix = np.abs(matrix)
#         phase_matrix = np.angle(matrix)
#
#         norm_angles = np.zeros([matrix.shape[0]-1, matrix.shape[1]+1])
#         phase_angles = np.zeros([matrix.shape[0], matrix.shape[1]])
#
#         # Compute the two rotation angles matrices "norm_angles" and "phase_angles"
#         for col in range(matrix.shape[1]):
#             norm_angles[:, col] = binarytree_vector(norm_matrix[:, col], 'norm')
#             phase_angles[:, col] = binarytree_vector(phase_matrix[:, col], 'phase')
#         column_norm = np.array([np.linalg.norm(norm_matrix[:, j]) for j in range(matrix.shape[1])])
#         norm_angles[:, matrix.shape[1]] = binarytree_vector(column_norm, 'norm')
#
#         return norm_angles, phase_angles


def test_angle_search_binary_tree(n_min, n_max, num):
    for n in range(n_min, n_max + 1):
        for _ in range(num):
            vec = np.random.randn(2 ** n, 2 ** n)

            norms1, angles1 = [], []
            for j in range(2 ** n):
                norm, angle = anglecompute.angle_search_binary_tree(vec[:, j])
                norms1.append(norm)
                angles1.append(angle)
            norms1 = np.asarray(norms1).flatten()
            angles1 = np.asarray(angles1).T

            norms2, angles2 = utils.angle_search_binary_tree(vec)

            print(norms1)
            print(norms2)
            print('---')
            print(angles1)
            print(angles2)

            if not np.allclose(norms1, norms2):
                print('norm')
                print(norms1)
                print(norms2)
                break

            if not np.allclose(angles1.flatten(), angles2.flatten()):
                print('phase')
                print(angles1)
                print(angles2)
                break


def test_positive_transform(n_min, n_max, num):
    for n in range(n_min, n_max + 1):
        for _ in range(num):
            vec = np.random.randn(2**n)

            norms1, angles1 = anglecompute.positive_transform(vec)
            norms2, angles2 = utils.positive_transform(vec)

            if not np.allclose(norms1.flatten(), norms2.flatten()):
                print('norm')
                print(norms1)
                print(norms2)
                break

            if not np.allclose(angles1.flatten(), angles2.flatten()):
                print('angle')
                print(angles1)
                print(angles2)
                break


def test_binary_vector(n_min, n_max, num):
    for n in range(n_min, n_max + 1):
        for _ in range(num):
            vec = np.random.randn(2**n)

            norms1 = anglecompute.binarytree_vector(vec, 'norm', True)
            norms2 = utils.binarytree_norm(vec, True)

            if not np.allclose(norms1.flatten(), norms2.flatten()):
                print(norms1)
                print(norms2)
                break

        for _ in range(num):
            vec = np.random.randn(2**n)

            angles1 = anglecompute.binarytree_vector(vec, 'phase')
            angles2 = utils.binarytree_phase(vec)

            if not np.allclose(angles1.flatten(), angles2.flatten()):
                print(angles1)
                print(angles2)
                break


def test_rotation_angles_matrix(n_min, n_max, num):
    for n in range(n_min, n_max + 1):
        for _ in range(num):
            mat = np.random.randn(2 ** n, 2 ** n)

            norm1 = anglecompute.rotation_angles_matrix(mat, True)
            norm2, col_norm_angles2 = utils.rotation_angles_matrix(mat, True)
            norm2 = np.concatenate((norm2, col_norm_angles2.reshape(-1, 1)), axis=1)

            if not np.allclose(norm1.flatten(), norm2.flatten()):
                print('norm')
                print(norm1)
                print(norm2)
                break

        for _ in range(num):
            mat = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)

            norm1, phase1 = anglecompute.rotation_angles_matrix(mat)
            norm2, phase2, col_norm_angles2 = utils.rotation_angles_matrix(mat)
            norm2 = np.concatenate((norm2, col_norm_angles2.reshape(-1, 1)), axis=1)

            if not np.allclose(norm1.flatten(), norm2.flatten()):
                print('norm')
                print(norm1)
                print(norm2)
                break

            if not np.allclose(phase1.flatten(), phase2.flatten()):
                print('phase')
                print(phase1)
                print(phase2)
                break


if __name__ == '__main__':
    n_min, n_max = 2, 8
    num = 50

    # test_angle_search_binary_tree(n_min, n_max, num)
    # test_positive_transform(n_min, n_max, num)
    # test_binary_vector(n_min, n_max, num)
    test_rotation_angles_matrix(n_min, n_max, num)


