"""
This module implements the jpeg encoder.

:author: Robin Senn
"""

import numpy
import itertools

from typing import Tuple, Iterator, Optional
from src.terminal_output import show_progress_indicator, StepName


def split_component(component: numpy.ndarray, height: int, width: int) -> Iterator[Tuple[numpy.ndarray, int, int]]:
    """
    Split component into 8x8 blocks. Split the component row-oriented.
    The first returned block is block (0,0) of the component, the second (0,1), the third (0,2).

    :return: Tuple of 8x8 block as 2d array, row number (y), column number (x)
    """
    for y, row in enumerate(numpy.vsplit(component, height / 8)):
        for x, block in enumerate(numpy.hsplit(row, width / 8)):
            yield block, y, x


def block_string_repr(block_no: int, block: numpy.ndarray) -> str:
    return f"\nBlock: {block_no}\n" + " ".join(
        f"{count}: {element:.6}"
        for count, element in enumerate(itertools.chain(*block.astype(float).tolist()), start=1)
    )


@show_progress_indicator(StepName.QUANTIZATION_TABLES)
def calculate_quantization_tables(quality: Optional[int]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculate the quantization tables for the given quality value.

    :param quality: quality 0 < quality < 100
    :return: tuple with luminance quantization table as pos. 0, chrominance table at pos. 1
    """
    luminance = numpy.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ],
        dtype=float,
    )

    chrominance = numpy.array(
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
        ],
        dtype=float,
    )
    if quality is None:
        #  print("Angegebener Qualitätswert: Standardtabelle")
        return luminance, chrominance

    # print(f"Angegebener Qualitätswert: {quality}")
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality if quality != 100 else 1
    for table in [chrominance, luminance]:
        with numpy.nditer(table, op_flags=["readwrite"]) as it:
            for x in it:
                x[...] = (scale * x + 50) / 100
    return luminance, chrominance


@show_progress_indicator(StepName.QUANTIZATION_TABLES)
def check_quantization_tables(luminance_table: numpy.ndarray, chrominance_table: numpy.ndarray) -> float:
    """
    Reverse calculate the quality that the quantization tables are representing.

    :return: None
    """
    mean_luminance = numpy.mean(list(numpy.concatenate(luminance_table.tolist()).flat)[1:])
    mean_chrominance = numpy.mean(list(numpy.concatenate(chrominance_table.tolist()).flat)[1:])
    mu = numpy.mean([mean_luminance, mean_chrominance, mean_chrominance])
    d = numpy.abs(mean_luminance - mean_chrominance) * 0.49 + numpy.abs(mean_luminance - mean_chrominance) * 0.49
    return 100 - mu + d


def calc_dct_basis_functions() -> numpy.ndarray:
    """
    Calculate the DCT basis functions with 1D DCT.
    """
    dct_basis_functions = numpy.zeros((8, 8), dtype=float)
    for x in range(8):
        for y in range(8):
            cx = 0.5 * (1 / numpy.sqrt(2) if x == 0 else 1)
            dct_basis_functions[x][y] = cx * numpy.cos(((2 * y + 1) * numpy.pi * x) / 16)
    return dct_basis_functions


def dct(block: numpy.ndarray, dct_basis_functions: numpy.ndarray) -> numpy.ndarray:
    """
    Every 8x8 block can be multiplied (matrix multiplication) with the same basis matrix and the
    transposed basis matrix. This is __much__ faster than the formula used in the slow_dct function.

    :param block: 8x8 pixel block
    :param dct_basis_functions:
    :return: 8x8 matrix with DCT coefficients
    """
    return dct_basis_functions.dot(block).dot(dct_basis_functions.transpose())


def slow_dct(block: numpy.ndarray) -> numpy.ndarray:
    """
    The DCT formula 1:1 from the lecture script.

    :param block: 8x8 pixel block
    :return: 8x8 matrix with DCT coefficients
    """
    dct_coefficients = numpy.zeros((8, 8), dtype=float)
    for u in range(8):
        for v in range(8):
            cu = 1 / numpy.sqrt(2) if u == 0 else 1
            cv = 1 / numpy.sqrt(2) if v == 0 else 1
            dct_coefficients[u][v] = _calc_dct_coefficient(u, v, cu, cv, block)
    return dct_coefficients


def _calc_dct_coefficient(u: int, v: int, cu: float, cv: float, block: numpy.ndarray) -> float:
    """
    Calculate the DCT coefficient for one element (index) of the 8x8 pixel block.

    :param u: row of the element
    :param v: column of the element
    :param cu: constant 1/sqrt(2) if u == 0 else 1
    :param cv: constant 1/sqrt(2) if v == 0 else 1
    :param block: 8x8 pixel block
    :return: DCT coefficient for position u,v
    """
    s = sum(
        (
            sum(
                (
                    y
                    * numpy.cos(((2 * count_x + 1) * u * numpy.pi) / 16)
                    * numpy.cos(((2 * count_y + 1) * v * numpy.pi) / 16)
                )
                for count_y, y in enumerate(x)
            )
            for count_x, x in enumerate(block)
        )
    )
    return 0.25 * cu * cv * s


def quantize(block: numpy.ndarray, quantization_table: numpy.ndarray) -> numpy.ndarray:
    """
    Apply quantization table to a 8x8 block.

    :param block: 8x8 block of dct coefficients
    :param quantization_table: quantization table
    :return: quantized 8x8 block
    """
    quantized_block = numpy.zeros((8, 8))
    for count_x, x in enumerate(block):
        for count_y, y in enumerate(x):
            quantized_block[count_x, count_y] = numpy.round(y / quantization_table[count_x, count_y])
    return quantized_block
