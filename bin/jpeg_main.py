"""
This module uses the encoder and decoder function to perform
the JPEG compression.

:author: Robin Senn, Quinten Stampa
"""
import copy

import numpy
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
from src import jpeg_encoder, jpeg_decoder
from src.terminal_output import (
    show_progress_indicator,
    StepName,
    initialize,
    teardown,
    print_results_to_console_and_file,
)
from color_space_transform import transform_ycbcr_rgb, transform_rgb_ycbcr
from multiprocessing import shared_memory


def main(
    image_path: Path,
    quality: Optional[int] = None,
    output: bool = False,
    dct_slow: bool = False,
    ela: bool = False,
    ela_multiplier: float = 1,
) -> None:
    """
    Perform the calculation of quantization tables and transformation RGB->YCbCr.
    Split each YCbCr component into 8x8 blocks and pass each block to an available
    worker process along with the calculated quantization table.

    :param image_path: path, where the image is located
    :param quality: the quality value to calculate the quantization tables. If not given,
     the standard tables are used.
    :param output: write the blocks of a component to a file for each transformation step.
     By default, no output is written.
    :param dct_slow: use the formula from the lecture notes 1:1. Very slow.
     By default, the faster method is used.
    :param ela: perform ELA and generate ELA image.
    :param ela_multiplier: multiplier for ELA
    :return: None
    """

    initialize(ela, output)

    image = extend_symmetrically(numpy.asarray(Image.open(str(image_path)), dtype=numpy.uint8))
    img_height = image.shape[0]
    img_width = image.shape[1]

    ycbcr_components = color_space_transformation_rgb_ycbcr(image, img_height, img_width)
    luminance_table, chrominance_table = jpeg_encoder.calculate_quantization_tables(quality)
    dct_basis_functions = jpeg_encoder.calc_dct_basis_functions()

    intermediate_result_names = ["idct"]
    recalculated_quality_value = None
    if output:
        intermediate_result_names.extend(["cst", "dct", "qdct", "dqdct"])
        recalculated_quality_value = jpeg_encoder.check_quantization_tables(luminance_table, chrominance_table)

    component_arrays = block_transformation(
        intermediate_result_names,
        ycbcr_components,
        img_height,
        img_width,
        dct_basis_functions,
        chrominance_table,
        luminance_table,
        dct_slow,
    )

    # transform y,cb,cr components, that are the result of inverse dct to rgb and save it as a png
    rgb_image = color_space_transformation_ycbcr_rgb(component_arrays, img_height, img_width)
    jpeg_decoder.save_rgb(rgb_image, image_path)

    # perform ELA if param ela is true (| input_array - output_array | * multiplier)
    if ela:
        generate_ela_image(image, rgb_image, image_path, quality, ela_multiplier)

    if output:
        psnr = jpeg_decoder.calc_psnr(image, rgb_image, img_height, img_width)
        entropies = component_entropies(image, intermediate_result_names, component_arrays)
        write_components_to_file(intermediate_result_names, image_path, component_arrays, img_height, img_width)
        print_results_to_console_and_file(image_path, psnr, entropies, recalculated_quality_value, quality)

    teardown()


@show_progress_indicator(StepName.EXTEND_SYMMETRICALLY)
def extend_symmetrically(image: numpy.ndarray):
    """
    Check if height and width are divisible by eight for the split in 8x8 blocks.
    If not add new mirrored rows/columns.

    :param image: ndarray that contains the image.
    """

    img_height = image.shape[0]
    img_width = image.shape[1]

    if img_height % 8 != 0:
        count_new_rows = 8 - img_height % 8
        extended_image = image
        for i in range(1, count_new_rows + 1):
            extended_image = numpy.vstack((extended_image, numpy.array([image[-i, :, :]])))
        image = extended_image

    # check if width is divisible by eight for the split in 8x8 blocks
    # if not add new mirrored columns
    if img_width % 8 != 0:
        count_new_columns = 8 - img_width % 8
        extended_image = image
        for i in range(1, count_new_columns + 1):
            extended_image = numpy.hstack((extended_image, numpy.transpose(numpy.array([image[:, -i, :]]), (1, 0, 2))))
        image = extended_image
    return image


@show_progress_indicator(StepName.TRANS_RGB_YCBCR)
def color_space_transformation_rgb_ycbcr(image: numpy.ndarray, img_height: int, img_width: int):
    return transform_rgb_ycbcr(image, img_height, img_width)


@show_progress_indicator(StepName.TRANS_YCBCR_RGB)
def color_space_transformation_ycbcr_rgb(component_arrays, img_height, img_width):
    return transform_ycbcr_rgb(*(c["idct"] for c in component_arrays.values()), img_height, img_width)


@show_progress_indicator(StepName.TRANS_BLOCK)
def block_transformation(
    intermediate_result_names,
    ycbcr_components,
    img_height,
    img_width,
    dct_basis_functions,
    chrominance_table,
    luminance_table,
    dct_slow,
):
    """
    Public api for parallel transformation of the 8x8 pixel blocks of each component (y,cb,cr).

    """
    # initialize shared memory blocks for every component and for every intermediate step that
    # is to be printed later
    shm_components = {
        comp_name: {
            step_name: shared_memory.SharedMemory(create=True, size=ycbcr_components[0].nbytes)
            for step_name in intermediate_result_names
        }
        for comp_name in ["y", "cb", "cr"]
    }

    # Process each 8x8 block individually in a process.
    # By that, parallelism is only limited by number of cpus on the system.
    for component, component_name in zip(ycbcr_components, ["y", "cb", "cr"]):
        with multiprocessing.Pool(multiprocessing.cpu_count()) as t:
            t.starmap(
                _transform_block,
                (
                    (
                        shm_components[component_name],
                        *block,
                        dct_basis_functions,
                        luminance_table if component_name == "y" else chrominance_table,
                        dct_slow,
                        img_height,
                        img_width,
                    )
                    for block in jpeg_encoder.split_component(component, img_height, img_width)
                ),
            )

    # attach ndarrays to the shared memory so that the components can be used for further processing
    # deepcopy the ndarrays, so that the shared memory blocks can be unlinked after
    component_arrays = {
        comp_name: {
            step_name: copy.deepcopy(
                numpy.ndarray(
                    (img_height, img_width),
                    dtype=float,
                    buffer=shm_components[comp_name][step_name].buf,
                )
            )
            for step_name in intermediate_result_names
        }
        for comp_name in ["y", "cb", "cr"]
    }

    # clean up shared memory
    for step_name in intermediate_result_names:
        for component_name in ["y", "cb", "cr"]:
            shm_components[component_name][step_name].unlink()

    return component_arrays


def _transform_block(
    component_shm_names: Dict[str, shared_memory.SharedMemory],
    block: numpy.ndarray,
    block_y: int,
    block_x: int,
    dct_basis_functions: numpy.ndarray,
    quantization_table: numpy.ndarray,
    dct_slow: bool,
    img_height: int,
    img_width: int,
) -> None:
    """
    Perform the encoding and decoding steps i.e. DCT, quantization, inverse DCT, dequantization for one block.
    Write the transformed block and possibly the intermediate results in the ndarrays in shared
    memory.

    :param component_shm_names: shared memory blocks, where the transformed blocks should be written
    :param block: 8x8 pixel block
    :param block_y: row of block in original image
    :param block_x: column of block in original image
    :param dct_basis_functions: static matrix to perform faster DCT
    :param quantization_table: quantization table for luminance or chrominance
    :param dct_slow: see documentation for main()
    :param img_height: image height in pixel
    :param img_width: image width in pixel
    :return: None
    """
    coefficient_block = jpeg_encoder.dct(block, dct_basis_functions) if not dct_slow else jpeg_encoder.slow_dct(block)
    quantized_block = jpeg_encoder.quantize(coefficient_block, quantization_table)
    dequantized_block = jpeg_decoder.dequantize(quantized_block, quantization_table)
    idct_block = (
        jpeg_decoder.idct(dequantized_block, dct_basis_functions)
        if not dct_slow
        else jpeg_decoder.idct_slow(dequantized_block)
    )

    blocks = {
        "cst": block,
        "dct": coefficient_block,
        "qdct": quantized_block,
        "dqdct": dequantized_block,
        "idct": idct_block,
    }

    for step_name, step_shm in component_shm_names.items():
        shm = shared_memory.SharedMemory(name=step_shm.name)
        array = numpy.ndarray((img_height, img_width), dtype=float, buffer=shm.buf)
        for y in range(8):
            for x in range(8):
                array[block_y * 8 + y][block_x * 8 + x] = blocks[step_name][y, x]
        shm.close()


@show_progress_indicator(StepName.CAL_ELA)
def generate_ela_image(
    image: numpy.ndarray,
    rgb_image: numpy.ndarray,
    image_path: Path,
    quality: int,
    ela_multiplier: float,
):
    """
    Writes the ELA result image as a file in PNG format.

    """
    ela_array = numpy.absolute(image.astype(int) - rgb_image.astype(int)) * ela_multiplier
    ela_array = ela_array.astype(numpy.uint8)
    ela_image = Image.fromarray(ela_array)
    ela_image.save(f"{image_path.stem}_ela_q{quality}_m{ela_multiplier}.png")


@show_progress_indicator(StepName.CALC_ENTROPY)
def component_entropies(
    original_image: numpy.ndarray, intermediate_result_names: List[str], component_arrays
) -> Dict[str, float]:
    """
    Public API for calculating the entropy for one component.
    """
    # split image in r, g, b components
    b, g, r = original_image[:, :, 0], original_image[:, :, 1], original_image[:, :, 2]
    rgb_arrays = {"b": b, "g": g, "r": r}

    entropies = {}
    # calculate entropy for b, g, r
    for component_name in ["b", "g", "r"]:
        comp = rgb_arrays[component_name]
        entropies[component_name] = _component_entropy(comp)

    # calculate entropy for y, cb, components
    for component_name in ["y", "cb", "cr"]:
        for step_name in intermediate_result_names:
            comp = component_arrays[component_name][step_name]
            entropies[f"{step_name}_{component_name}"] = _component_entropy(comp)
    return entropies


def _component_entropy(array: numpy.ndarray) -> float:
    """
    Calculate the entropy of given array.

    Is called for:
        - R-,G-,B-Components
        - Y-,Cb-,Cr-Components
        - DCT-Coefficients of Y-,Cb-,Cr-Components
        - quantized DCT-Coefficients of Y-,Cb-,Cr-Components
        - dequantized DCT-Coefficients of Y-,Cb-,Cr-Components

    :param array: self explanatory
    """
    entropy_sum = 0

    array = numpy.round(array, 0)
    checked_values = []
    for row in array:
        for value in row:
            if value not in checked_values:
                p = ((array == value).sum()) / array.size
                checked_values.append(value)
                if p != 0:
                    try:
                        entropy_sum += p * numpy.log2(p)
                    except RuntimeWarning:
                        print(value)
    return round(-entropy_sum, 3)


@show_progress_indicator(StepName.OUT_COMPONENTS)
def write_components_to_file(
    intermediate_result_names: List[str], image_path: Path, component_arrays, img_height: int, img_width: int
) -> None:
    """
    Write the pixel values of each component to a txt file. Required by as part of the assignment.
    """
    # write the intermediate results to text files
    intermediate_result_names.remove("idct")
    for component_name in ["y", "cb", "cr"]:
        for step_name in intermediate_result_names:
            with open(f"{image_path.stem}_{component_name}_{step_name}.txt", "w+") as f:
                # split the components again
                comp = component_arrays[component_name][step_name]
                if step_name == "cst":
                    cmp = Image.fromarray(comp.astype(numpy.uint8), "L")
                    cmp.save(f"{image_path.stem}_{component_name}.png")
                for block_no, b in enumerate(jpeg_encoder.split_component(comp, img_height, img_width), start=1):
                    f.write(jpeg_encoder.block_string_repr(block_no, b[0]))
