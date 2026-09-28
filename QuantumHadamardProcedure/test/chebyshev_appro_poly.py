from QuantumHadamardProcedure import utils, functions


def print_poly_table(polys, sig=6, width=16):
    names = [n for n, _ in polys]
    max_len = max(len(c) for _, c in polys)

    header = f"{'power':>{width}}" + "".join(f"{n:>{width}}" for n in names)
    sep = "-" * len(header)
    print(header)
    print(sep)

    for k in range(max_len):
        row = f"{'x^' + str(k):>{width}}"
        for _, c in polys:
            if k < len(c):
                row += f"{c[k]:>{width}.{sig}g}"
            else:
                row += f"{'-':>{width}}"
        print(row)


_, poly_sigmoid = utils.chebyshev_approximation(functions.sigmoid, degree=5, domain=(-4, 4))
_, poly_tanh = utils.chebyshev_approximation(functions.tanh, degree=13, domain=(-4, 4))
_, poly_log_transformation = utils.chebyshev_approximation(functions.log_transformation, degree=5, domain=(0, 1))
_, poly_gamma_correction_1 = utils.chebyshev_approximation(functions.gamma_correction_1, degree=5, domain=(0, 1))
_, poly_gamma_correction_2 = utils.chebyshev_approximation(functions.gamma_correction_2, degree=6, domain=(0, 1))

print('Polynomial approximation (ascending power):')
print_poly_table([
    ('sigmoid', poly_sigmoid[::-1]),
    ('tanh', poly_tanh[::-1]),
    ('log', poly_log_transformation[::-1]),
    ('gamma(2.2)', poly_gamma_correction_2[::-1]),
    ('gamma(0.4)', poly_gamma_correction_1[::-1]),
])
