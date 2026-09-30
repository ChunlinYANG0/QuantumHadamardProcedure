from qiskit import transpile
from QuantumHadamardProcedure import qhp, utils, functions
import numpy as np
import matplotlib.pyplot as plt


def hadamard_poly(poly, array):
    array = np.asarray(array)
    poly = poly[::-1]
    if poly[0] == 0:
        result = np.zeros_like(array)
    else:
        result = poly[0] * np.ones_like(array)

    for k in range(1, len(poly)):
        result = result + poly[k] * array ** k

    return result


def circuit_complexity(n, poly, state_or_matrix):
    if state_or_matrix == 'state':
        arr = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
    elif state_or_matrix == 'matrix':
        arr = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
    else:
        raise ValueError("state_or_matrix must be 'matrix' or 'state'")

    use_cache = True
    decompose_mcx = True
    optimization_level = 1

    qhmf_factorizaton = qhp.QHMF(
        arr=arr,
        poly=poly,
        mode='factorization',
        use_cache=use_cache,
        decompose_mcx=decompose_mcx
    )
    qhmf_binary = qhp.QHMF(
        arr=arr,
        poly=poly,
        mode='binary',
        use_cache=use_cache,
        decompose_mcx=decompose_mcx
    )
    qhmf_lcu = qhp.QHMF(
        arr=arr,
        poly=poly,
        mode='lcu',
        use_cache=use_cache,
        decompose_mcx=decompose_mcx
    )

    qc_factorizaton = transpile(qhmf_factorizaton.circuit(), optimization_level=optimization_level)
    qc_binary = transpile(qhmf_binary.circuit(), optimization_level=optimization_level)
    qc_lcu = transpile(qhmf_lcu.circuit(), optimization_level=optimization_level)

    depth = [qc_factorizaton.depth(), qc_binary.depth(),  qc_lcu.depth()]
    size = [qc_factorizaton.size(), qc_binary.size(), qc_lcu.size()]
    num_qubits = [qc_factorizaton.num_qubits, qc_binary.num_qubits, qc_lcu.num_qubits]
    normalization = [
        qhmf_factorizaton.normalization_factor,
        qhmf_binary.normalization_factor,
        qhmf_lcu.normalization_factor
    ]

    return depth, size, num_qubits, normalization


def draw(x, y_list, func_name):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    ax_list = axes.flatten()

    sample_labels = ['Factorization', 'Binary tree', 'LCU']
    colors = [
        "#009E73",
        "#56B4E9",
        "#E69F00",
    ]
    markers = ['s', 'o', '^']

    y_data0 = np.array(y_list[0])
    for col in range(y_data0.shape[1]):
        ax_list[0].plot(
            x, y_data0[:, col],
            marker=markers[col % len(markers)],
            markersize=5,
            linestyle='-',
            color=colors[col % len(colors)],
            label=sample_labels[col % len(sample_labels)]
        )
    ax_list[0].set_xlabel('n', fontsize=12)
    ax_list[0].set_yscale("log")
    ax_list[0].set_ylabel('Depth (log scale)', fontsize=12)
    ax_list[0].grid(True)

    y_data1 = np.array(y_list[1])
    for col in range(y_data1.shape[1]):
        ax_list[1].plot(
            x, y_data1[:, col],
            marker=markers[col % len(markers)],
            markersize=5,
            linestyle='-',
            color=colors[col % len(colors)],
            label=sample_labels[col % len(sample_labels)]
        )
    ax_list[1].set_xlabel('n', fontsize=12)
    ax_list[1].set_yscale("log")
    ax_list[1].set_ylabel('Size (log scale)', fontsize=12)
    ax_list[1].grid(True)

    y_data2 = np.array(y_list[2])
    for col in range(y_data2.shape[1]):
        ax_list[2].plot(
            x, y_data2[:, col],
            marker=markers[col % len(markers)],
            markersize=5,
            linestyle='-',
            color=colors[col % len(colors)],
            label=sample_labels[col % len(sample_labels)]
        )
    ax_list[2].set_xlabel('n', fontsize=12)
    ax_list[2].set_ylabel('Number of qubits', fontsize=12)
    ax_list[2].grid(True)

    y_data3 = np.array(y_list[3])
    for col in range(y_data3.shape[1]):
        ax_list[3].plot(
            x, y_data3[:, col],
            marker=markers[col % len(markers)],
            markersize=5,
            linestyle='-',
            color=colors[col % len(colors)],
            label=sample_labels[col % len(sample_labels)]
        )
    ax_list[3].set_xlabel('n', fontsize=12)
    ax_list[3].set_yscale("log")
    ax_list[3].set_ylabel('Normalization (log scale)', fontsize=12)
    ax_list[3].grid(True)

    handles = [plt.Line2D([0], [0], color=c, marker=m, linestyle='-', label=l)
               for c, m, l in zip(colors, markers, sample_labels)]
    fig.legend(handles=handles,
               labels=sample_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.0),
               ncol=3,
               fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f'../image/complexity_comparison_{func_name}.png', dpi=300)
    plt.show()


def comparison(n_min, n_max, poly, state_or_matrix, func_name):
    n_list = list(range(n_min, n_max + 1))
    depth_list = []
    size_list = []
    num_qubits_list = []
    normalization_list = []

    for n in n_list:
        (
            depth,
            size,
            num_qubits,
            normalization
        ) = circuit_complexity(n, poly, state_or_matrix)
        depth_list.append(depth)
        size_list.append(size)
        num_qubits_list.append(num_qubits)
        normalization_list.append(normalization)
        print('n =', n)

    draw(n_list, [depth_list, size_list, num_qubits_list, normalization_list], func_name)


def sigmoid():
    degree = 5
    domain = (-4, 4)
    _, poly = utils.chebyshev_approximation(functions.sigmoid, degree, domain)
    state_or_matrix = 'state'
    comparison(1, 10, poly, state_or_matrix, func_name='sigmoid')


def tanh():
    degree = 13
    domain = (-4, 4)
    _, poly = utils.chebyshev_approximation(functions.tanh, degree, domain)
    state_or_matrix = 'state'
    comparison(1, 10, poly, state_or_matrix, func_name='tanh')


if __name__ == '__main__':
    sigmoid()
    tanh()