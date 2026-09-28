from qiskit import transpile
from QuantumHadamardProcedure import qhp, functions, utils
import numpy as np
from matplotlib import pyplot as plt


def circuit_complexity(arr, poly, m):
    qhpoly = qhp.QHPoly(
        arr=arr,
        poly=poly,
        mode='tradeoff',
        num_subpolys=m,
        use_cache=True,
        decompose_mcx=True
    )
    qc = transpile(qhpoly.circuit(), optimization_level=1)

    return qc.depth(), qc.size(), qc.num_qubits, qhpoly.normalization_factor


def comparison(n_min, n_max, func, degree, state_or_matrix):
    _, poly = utils.chebyshev_approximation(
        func=func,
        degree=degree,
        domain=(-4, 4)
    )

    depth_lists = []
    size_lists = []
    num_qubits_lists = []
    normalization_factor_lists = []

    for n in range(n_min, n_max + 1):
        if state_or_matrix == 'state':
            arr = np.random.randn(2 ** n) + 1j * np.random.randn(2 ** n)
        elif state_or_matrix == 'matrix':
            arr = np.random.randn(2 ** n, 2 ** n) + 1j * np.random.randn(2 ** n, 2 ** n)
        else:
            raise ValueError("state_or_matrix must be 'matrix' or 'state'")

        depth_list = []
        size_list = []
        num_qubits_list = []
        normalization_factor_list = []
        for m in range(1, degree + 1, 2):
            depth, size, num_qubits, normalization_factor = circuit_complexity(arr, poly, m)
            depth_list.append(depth)
            size_list.append(size)
            num_qubits_list.append(num_qubits)
            normalization_factor_list.append(normalization_factor)
        depth_lists.append(depth_list)
        size_lists.append(size_list)
        num_qubits_lists.append(num_qubits_list)
        normalization_factor_lists.append(normalization_factor_list)

    draw(
        range(n_min, n_max + 1),
        [depth_lists, size_lists, num_qubits_lists, normalization_factor_lists],
        'tanh'
    )


def draw(x, y_list, func_name):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    ax_list = axes.flatten()

    colors = [
        "#000000",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#CC79A7",
        "#0072B2",
        "#D55E00"
    ]

    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.08, right=0.985,
                        hspace=0.35, wspace=0.28)
    handles, labels = [], []

    y_data0 = np.array(y_list[0])
    for col in range(y_data0.shape[1]):
        line, = ax_list[0].plot(
            x, y_data0[:, col],
            linestyle='-',
            linewidth=1,
            color=colors[col],
            label=f'm={2 * col + 1}'
        )
        handles.append(line)
        labels.append(f'm={2 * col + 1}')
    ax_list[0].set_xlabel('n', fontsize=12)
    ax_list[0].set_yscale("log")
    ax_list[0].set_ylabel('Depth (log scale)', fontsize=12)
    ax_list[0].annotate("m=5,7,9,11",
                        xy=(x[7], 2 * 1e4),
                        xytext=(x[6], 1e2),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=9, color="gray",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax_list[0].grid(True)

    y_data1 = np.array(y_list[1])
    for col in range(y_data1.shape[1]):
        ax_list[1].plot(
            x, y_data1[:, col],
            linestyle='-',
            linewidth=1,
            color=colors[col],
            label=f'm={2 * col + 1}'
        )
    ax_list[1].set_xlabel('n', fontsize=12)
    ax_list[1].set_yscale("log")
    ax_list[1].set_ylabel('Size (log scale)', fontsize=12)
    ax_list[1].grid(True)

    y_data2 = np.array(y_list[2])
    for col in range(y_data2.shape[1]):
        ax_list[2].plot(
            x, y_data2[:, col],
            linestyle='-',
            linewidth=1,
            color=colors[col],
            label=f'm={2 * col + 1}'
        )
    ax_list[2].set_xlabel('n', fontsize=12)
    ax_list[2].set_ylabel('Number of qubits', fontsize=12)
    ax_list[2].annotate("m=7,9,11,13",
                        xy=(x[6], 110),
                        xytext=(x[2], 120),
                        arrowprops=dict(arrowstyle="->", color="gray"),
                        fontsize=9, color="gray",
                        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    ax_list[2].grid(True)

    y_data3 = np.array(y_list[3])
    for col in range(y_data3.shape[1]):
        ax_list[3].plot(
            x, y_data3[:, col],
            linestyle='-',
            linewidth=1,
            color=colors[col],
            label=f'm={2 * col + 1}'
        )
    ax_list[3].set_xlabel('n', fontsize=12)
    ax_list[3].set_yscale("log")
    ax_list[3].set_ylabel('Normalization (log scale)', fontsize=12)
    ax_list[3].grid(True)

    fig.legend(handles, labels,
               loc="upper center",
               bbox_to_anchor=(0.5, 0.98),
               ncol=4,
               frameon=True,
               fontsize=11)

    plt.savefig(f'../image/complexity_tradeoff_{func_name}.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    degree = 13
    func = functions.tanh

    comparison(n_min=1, n_max=10, func=func, degree=degree, state_or_matrix='state')