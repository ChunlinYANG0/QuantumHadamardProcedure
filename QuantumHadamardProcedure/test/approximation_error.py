from QuantumHadamardProcedure import functions, utils
import numpy as np
import matplotlib.pyplot as plt


def compute_errors(func, degrees, domain, method='Chebyshev', num_points=20001):
    if method not in ['Chebyshev', 'minmax']:
        raise ValueError(f'method must be one of {["Chebyshev", "minmax"]}')

    x = np.linspace(domain[0], domain[1], num_points)
    y_true = func(x)

    approx_vals = {}
    abs_error_vals = {}
    max_abs_errors = []
    rmse = []

    for d in degrees:
        if method == 'Chebyshev':
            poly, _ = utils.chebyshev_approximation(func, d, domain)
        else:
            poly, _ = utils.minmax_approximation(func, d, domain)

        y_approx = poly(x)
        abs_err = np.abs(y_true - y_approx)
        max_abs_err = np.max(abs_err)
        rmse_val = np.sqrt(np.mean(abs_err ** 2))

        approx_vals[d] = y_approx
        abs_error_vals[d] = abs_err
        max_abs_errors.append(max_abs_err)
        rmse.append(rmse_val)

    return approx_vals, abs_error_vals, max_abs_errors, rmse


def draw(func, degrees, domain, method='Chebyshev', num_points=20001):
    if method not in ['Chebyshev', 'minmax']:
        raise ValueError(f'method must be one of {["Chebyshev", "minmax"]}')

    x = np.linspace(domain[0], domain[1], num_points)
    y_true = func(x)

    approx_vals, abs_error_vals, max_abs_errors, rmse = compute_errors(func, degrees, domain, method, num_points)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    if func.__name__ == 'log_transformation':
        label = 'Log transformation'
    else:
        label = 'Gamma correction'
    plt.plot(x, y_true, 'k--', linewidth=2, label=label)

    colors = plt.cm.tab10(np.linspace(0, 1, len(degrees)))
    title = 'Chebyshev Approximations' if method == 'Chebyshev' else 'Minmax Approximations'

    for d, color in zip(degrees, colors):
        plt.plot(x, approx_vals[d], color=color, linewidth=1, label=f'Degree {d}')

    plt.xlabel('x', fontsize=14)
    plt.ylabel('f(x), P(x)', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim(domain)

    plt.subplot(1, 2, 2)
    for d, color in zip(degrees, colors):
        plt.semilogy(x, abs_error_vals[d], color=color, linewidth=1, label=f'Degree {d}')

    plt.xlabel('x', fontsize=12)
    plt.ylabel(r'Absolute Error $|f(x)-P(x)|$ (log-scale axis)', fontsize=12)
    plt.title('Approximation Errors', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim(domain)

    plt.tight_layout()
    plt.savefig(f'../image/chebyshev_approx_{func.__name__}.png', dpi=600)
    plt.show()


def print_table(func, name, degrees, domain, method='Chebyshev'):
    if method not in ['Chebyshev', 'minmax']:
        raise ValueError(f'method must be one of {["Chebyshev", "minmax"]}')

    print("=" * 70)
    print(f"Interpolation error of {name} on [{domain[0]}, {domain[1]}] ({method})")
    print("=" * 70)
    print(f"{'Degree':<10} {'Maximum Absolute Error':<25} {'RMSE':<20}")
    print("-" * 70)

    _, _, max_abs_errors, rmse = compute_errors(func, degrees, domain, method)
    for idx, deg in enumerate(degrees):
        # {:.3e} 表示科学计数法下小数点后 3 位，即共 4 位有效数字
        print(f"{deg:<10} {max_abs_errors[idx]:<25.3e} {rmse[idx]:<20.3e}")
    print()


def get_sigmoid():
    draw(
        functions.sigmoid,
        degrees=[1, 3, 5, 7],
        domain=(-4, 4),
    )
    print_table(
        functions.sigmoid, "Sigmoid",
        degrees=[1, 3, 5, 7, 9, 11, 13, 15],
        domain=[-4, 4]
    )


def get_tanh():
    draw(
        functions.tanh,
        degrees=[5, 7, 9, 11, 13],
        domain=(-4, 4),
    )
    print_table(
        functions.tanh, "Tanh",
        degrees=[1, 3, 5, 7, 9, 11, 13, 15],
        domain=[-4, 4]
    )


def get_log():
    draw(
        functions.log_transformation,
        degrees=[2, 3, 4, 5, 6],
        domain=(0, 1),
    )
    print_table(
        functions.log_transformation,'Log',
        degrees=[1, 2, 3, 4, 5, 6],
        domain=[0, 1],
    )


def get_gamma_correction_1_cheby():
    draw(
        functions.gamma_correction_1,
        degrees=list(range(1, 9)),
        domain=(0, 1),
    )
    print_table(
        functions.gamma_correction_1, 'Gamma Correction',
        degrees=list(range(1, 9)),
        domain=[0, 1],
    )


def get_gamma_correction_1_minmax():
    draw(
        functions.gamma_correction_1,
        degrees=list(range(1, 9)),
        domain=(0, 1),
        method='minmax',
    )
    print_table(
        functions.gamma_correction_1, 'Gamma Correction',
        degrees=list(range(1, 9)),
        domain=[0, 1],
        method='minmax',
    )


def get_gamma_correction_2():
    draw(
        functions.gamma_correction_2,
        degrees=[3, 4, 5, 6, 7],
        domain=(0, 1),
    )
    print_table(
        functions.gamma_correction_2,'Gamma Correction',
        degrees=[1, 2, 3, 4, 5, 6, 7],
        domain=[0, 1],
    )


if __name__ == '__main__':
    get_sigmoid()
    get_tanh()
    get_log()
    get_gamma_correction_1_cheby()
    get_gamma_correction_1_minmax()