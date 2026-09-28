from qiskit_aer import StatevectorSimulator
from QuantumHadamardProcedure import utils, qhp, functions
import numpy as np
import time
from PIL import Image
import matplotlib.pyplot as plt


def matrix_to_image(matrix, file_name=None):
    img_array = np.array(matrix, dtype=np.uint8)

    if file_name is not None:
        Image.fromarray(img_array).save(f'../image/{file_name}.png')

    plt.figure(figsize=(6, 6))
    plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)


def calculate_psnr(img1, img2, max_val=255.0, color_mode='average'):
    """Calculate the PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (numpy.ndarray): First input image. Must have the same shape as `img2`.
            Can be a grayscale image with shape (H, W) or a color image with shape
            (H, W, C). Data type can be uint8, uint16, or float.
        img2 (numpy.ndarray): Second input image, with the same shape and type as `img1`.
        max_val (float, optional): Maximum possible pixel value. For 8-bit images, use 255;
            for 16-bit images, use 65535; for float images normalized to [0, 1], use 1.0.
            Defaults to 255.0.
        color_mode (str, optional): Method for handling color images. Options are:
            - 'average': Compute PSNR for each channel separately and then average them (default).
            - 'overall': Compute MSE over all channels combined, then compute PSNR.
            Defaults to 'average'.

    Returns:
        float: The PSNR value in decibels (dB). Returns `inf` if the two images are identical.

    Raises:
        ValueError: If `img1` and `img2` have different shapes.
        ValueError: If `color_mode` is not 'average' or 'overall'.
        ValueError: If the image dimensions are not 2 or 3.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.shape != img2.shape:
        raise ValueError("The shapes of the two images must be the same")

    if img1.ndim == 2:
        # Grayscale image
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((max_val ** 2) / mse)

    elif img1.ndim == 3:
        # Color image
        if color_mode == 'overall':
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return float('inf')
            return 10 * np.log10((max_val ** 2) / mse)

        elif color_mode == 'average':
            psnr_list = []
            for c in range(img1.shape[2]):
                mse = np.mean((img1[:, :, c] - img2[:, :, c]) ** 2)
                if mse == 0:
                    psnr_list.append(float('inf'))
                else:
                    psnr_list.append(10 * np.log10((max_val ** 2) / mse))
            return float(np.mean(psnr_list))
        else:
            raise ValueError("color_mode must be either 'average' or 'overall'")
    else:
        raise ValueError("Unsupported image dimensions, should be 2 or 3")


def quantum_hadamard_poly(poly, array):
    shape = array.shape
    array = np.asarray(array).reshape(-1)

    qhpoly = qhp.QHPoly(
        arr=array,
        poly=poly,
        mode='lcu',
        use_cache=True
    )
    qc = qhpoly.circuit()

    simulator = StatevectorSimulator()
    result = simulator.run(qc).result()
    arr_qhpoly = result.get_statevector().data[:len(array)] * qhpoly.normalization_factor
    arr_qhpoly = arr_qhpoly.reshape(shape)

    return arr_qhpoly


def processing_log(matrix, file_name):
    matrix = matrix / 255

    dim = matrix.shape[0]
    dim_block = 8
    num_block_per_col = dim // dim_block
    num_blocks = num_block_per_col ** 2
    blocks = (
        matrix.reshape(num_block_per_col, dim_block, num_block_per_col, dim_block).swapaxes(1, 2)
        .reshape(-1, dim_block, dim_block)
    )

    _, poly_log = utils.chebyshev_approximation(
        functions.log_transformation,
        degree=5,
        domain=(0, 1)
    )

    start_time = time.time()
    print('time =', start_time - start_time)
    for i in range(num_blocks):
        block = quantum_hadamard_poly(poly_log, blocks[i]) * 255
        blocks[i] = block.real
        print('Process: ', i, ' /', num_blocks - 1)
        temp_time = time.time()
        print('time =', (temp_time - start_time) // 60, ' min', (temp_time - start_time) % 60, 'sec')
        print('-' * 50)

    matrix_processed = (
        blocks.reshape(num_block_per_col, num_block_per_col, dim_block, dim_block).swapaxes(1, 2).reshape(dim, dim)
    )
    matrix_to_image(matrix_processed, file_name=file_name)


def processing_gamma1(matrix, file_name):
    matrix = matrix / 255

    dim = matrix.shape[0]
    dim_block = 8
    num_block_per_col = dim // dim_block
    num_blocks = num_block_per_col ** 2
    blocks = (
        matrix.reshape(num_block_per_col, dim_block, num_block_per_col, dim_block).swapaxes(1, 2)
        .reshape(-1, dim_block, dim_block)
    )

    _, poly_gamma = utils.chebyshev_approximation(
        functions.gamma_correction_1,
        degree=5,
        domain=(0, 1)
    )

    start_time = time.time()
    print('time =', start_time - start_time)
    for i in range(num_blocks):
        block = quantum_hadamard_poly(poly_gamma, blocks[i]) * 255
        blocks[i] = block.real
        print('Process: ', i, ' /', num_blocks - 1)
        temp_time = time.time()
        print('time =', (temp_time - start_time) // 60, ' min', (temp_time - start_time) % 60, 'sec')
        print('-' * 50)

    matrix_processed = (
        blocks.reshape(num_block_per_col, num_block_per_col, dim_block, dim_block).swapaxes(1, 2).reshape(dim, dim)
    )
    matrix_to_image(matrix_processed, file_name=file_name)


def processing_gamma2(matrix, file_name):
    matrix = matrix / 255

    dim = matrix.shape[0]
    dim_block = 8
    num_block_per_col = dim // dim_block
    num_blocks = num_block_per_col ** 2
    blocks = (
        matrix.reshape(num_block_per_col, dim_block, num_block_per_col, dim_block).swapaxes(1, 2)
        .reshape(-1, dim_block, dim_block)
    )

    _, poly_gamma = utils.chebyshev_approximation(
        functions.gamma_correction_2,
        degree=6,
        domain=(0, 1)
    )

    start_time = time.time()
    print('time =', start_time - start_time)
    for i in range(num_blocks):
        block = quantum_hadamard_poly(poly_gamma, blocks[i]) * 255
        blocks[i] = block.real
        print('Process: ', i, ' /', num_blocks - 1)
        temp_time = time.time()
        print('time =', (temp_time - start_time) // 60, ' min', (temp_time - start_time) % 60, 'sec')
        print('-' * 50)

    matrix_processed = (
        blocks.reshape(num_block_per_col, num_block_per_col, dim_block, dim_block).swapaxes(1, 2).reshape(dim, dim)
    )
    matrix_to_image(matrix_processed, file_name=file_name)


def standard_log_transformation(img):
    """
    s = (255 / log(1 + 255)) * log(1 + r)
    """
    img_f = img.astype(np.float64) / 255.0
    out = np.log2(1.0 + img_f)
    out = np.clip(out * 255.0, 0, 255)
    return out


def standard_gamma_correction(img, gamma):
    """
    s = 255 * (r / 255) ** gamma
    """
    img_f = img.astype(np.float64) / 255.0
    out_f = np.power(img_f, gamma)
    out = np.clip(out_f * 255.0, 0, 255)
    return out


if __name__ == '__main__':
    matrix_origin = np.array(Image.open('../image/lena.png'))

    processing_log(matrix_origin, file_name='image_qhpoly_log_lena')
    print('=' * 200)
    processing_gamma1(matrix_origin, file_name='image_qhpoly_gamma1_lena')
    print('=' * 200)
    processing_gamma2(matrix_origin, file_name='image_qhpoly_gamma2_lena')
    print('=' * 200)

    matrix_standard_log = standard_log_transformation(matrix_origin)
    matrix_standard_gamma2 = standard_gamma_correction(matrix_origin, gamma=2.2)
    matrix_standard_gamma1 = standard_gamma_correction(matrix_origin, gamma=0.4)

    matrix_to_image(matrix_standard_log, file_name='../image/lena_log')
    matrix_to_image(matrix_standard_gamma1, file_name='../image/lena_gamma1')
    matrix_to_image(matrix_standard_gamma2, file_name='../image/lena_gamma2')

    matrix_log = np.array(Image.open('../image/image_qhpoly_log_lena.png'))
    matrix_gamma2 = np.array(Image.open('../image/image_qhpoly_gamma2_lena.png'))
    matrix_gamma1 = np.array(Image.open('../image/image_qhpoly_gamma1_lena.png'))

    psnr_log = calculate_psnr(matrix_standard_log, matrix_log)
    psnr_gamma2 = calculate_psnr(matrix_standard_gamma2, matrix_gamma2)
    psnr_gamma1 = calculate_psnr(matrix_standard_gamma1, matrix_gamma1)

    print('PSNR for log transformation:', psnr_log)
    print('PSNR for gamma correction with gamma=2.2:', psnr_gamma2)
    print('PSNR for gamma correction with gamma=0.4:', psnr_gamma1)

    print(np.max(np.abs(matrix_standard_log - matrix_log)))
    print(np.max(np.abs(matrix_gamma2 - matrix_standard_gamma2)))
