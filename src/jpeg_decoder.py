"""
This module implements the jpeg decoder.

:author: Simon Sand
"""
from pathlib import Path

from PIL import Image
import numpy
from numpy import ndarray
from src.terminal_output import show_progress_indicator, StepName


def dequantize(block: ndarray, quantization_table: numpy.ndarray) -> numpy.ndarray:
    """
    Apply dequantization table to a 8x8 block.

    :param block: 8x8 quantized block
    :param quantization_table: quantization table
    :return: dequantized 8x8 block
    """
    dequantized_block = numpy.zeros((8, 8))
    for count_x, x in enumerate(block):
        for count_y, y in enumerate(x):
            dequantized_block[count_x, count_y] = y * quantization_table[count_x, count_y]
    return dequantized_block


def idct_slow(block: numpy.ndarray) -> numpy.ndarray:
    """
    Apply slow inverse discrete cosinus transformation (like in the lecture) to an 8x8 block.

    :param block: 8x8 quantized block
    :return: inversed dct 8x8 block
    """
    idct_coefficients = numpy.zeros((8, 8), dtype=float)
    for x in range(8):
        for y in range(8):
            idct_coefficients[x][y] = calc_idct_coefficient(x, y, block)
    return idct_coefficients


def calc_idct_coefficient(x: int, y: int, block: numpy.ndarray) -> float:
    """
    Calculating the inverse discrete cosinus transformation coefficient.

    :param x: coordinate x-axis
    :param y: coordinate y-axis
    :param block: 8x8 quantized block
    :return: IDCT coefficient for x and y
    """
    s = 0
    for u in range(8):
        for v in range(8):
            if u == 0:
                cu = 1 / numpy.sqrt(2)
            else:
                cu = 1
            if v == 0:
                cv = 1 / numpy.sqrt(2)
            else:
                cv = 1
            s += (
                cu
                * cv
                * block[u][v]
                * numpy.cos(((2 * x + 1) * u * numpy.pi) / 16)
                * numpy.cos(((2 * y + 1) * v * numpy.pi) / 16)
            )
    return round(0.25 * s)


def idct(block: numpy.ndarray, dct_basis_functions: numpy.ndarray) -> numpy.ndarray:
    """
    Every 8x8 block can be multiplied (matrix multiplication) with the same basis matrix and the
    transposed basis matrix. This is __much__ faster than the formula used in the slow_dct function.

    :param block: 8x8 pixel block
    :param dct_basis_functions:
    :return: 8x8 matrix with DCT coefficients
    """
    return dct_basis_functions.transpose().dot(block).dot(dct_basis_functions)


def save_rgb(image: numpy.array, image_path: Path):
    """
    Saves the processed image

    :param image: array of the image
    :param image_path: path to save image
    """
    im = Image.fromarray(image)
    im.save(f"{image_path.stem}_processed.png")


@show_progress_indicator(StepName.CALC_PSNR)
def calc_psnr(image_before: ndarray, image_after: ndarray, height, width) -> float:
    """
    Calculates the PSNR value

    :param image_before: original image
    :param image_after: processed image
    :param height: image height
    :param width: image width
    :return: PSNR value from given original and processed image
    """
    s = numpy.array([0, 0, 0])
    for y in range(height):
        for x in range(width):
            s += (image_before[y][x] - image_after[y][x]) ** 2
    sum_rgb = s[0] + s[1] + s[2]
    mse = sum_rgb / (height * width) / 3
    psnr = 10 * numpy.log10((255 ** 2) / mse)
    return psnr
