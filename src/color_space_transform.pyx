#!python
#cython: language_level=3
"""
Functions for color space transformation implemented as cython functions as they were the
slowest of the whole program.

:author: Robin Senn
"""
import numpy
cimport numpy as cnp
from libc.math cimport round

cpdef tuple transform_rgb_ycbcr(cnp.ndarray[cnp.uint8_t, ndim=3] image, int height, int width):
    cdef cnp.ndarray[cnp.float_t, ndim=2] ycbcr_conversion_matrix = \
        numpy.array([[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]], dtype=numpy.float)
    cdef cnp.ndarray[cnp.int_t, ndim=2] y_component = numpy.empty((height, width), dtype=numpy.int)
    cdef cnp.ndarray[cnp.int_t, ndim=2] cb_component = numpy.empty((height, width), dtype=numpy.int)
    cdef cnp.ndarray[cnp.int_t, ndim=2] cr_component = numpy.empty((height, width), dtype=numpy.int)

    cdef Py_ssize_t y, x
    for y in range(height):
        for x in range(width):
            y_val, cb, cr = numpy.dot(ycbcr_conversion_matrix, image[y, x])
            y_component[y, x] = <cnp.int_t>round(y_val)
            cb_component[y, x] = <cnp.int_t>round(cb + 128)
            cr_component[y, x] = <cnp.int_t>round(cr + 128)
    return y_component, cb_component, cr_component


cpdef cnp.ndarray transform_ycbcr_rgb(
        cnp.ndarray[cnp.float_t, ndim=2] y_array,
        cnp.ndarray[cnp.float_t, ndim=2] cb_array,
        cnp.ndarray[cnp.float_t, ndim=2] cr_array,
        int height,
        int width,
):
    cdef cnp.ndarray[cnp.uint8_t, ndim=3] image = numpy.empty((height, width, 3), dtype=numpy.uint8)
    cdef cnp.ndarray[cnp.float_t, ndim=2] rgb_conversion_matrix =\
        numpy.array([[1, 0, 1.402], [1, -0.34414, -0.71414], [1, 1.772, 0]], dtype=numpy.float)
    cdef cnp.ndarray[cnp.long_t, ndim=1] pixel_array = numpy.empty(3, dtype=numpy.int)

    cdef cnp.int_t y_value, cb_value, cr_value, r, g, b
    cdef Py_ssize_t x, y, i
    for y in range(height):
        for x in range(width):
            y_value = <cnp.int_t>y_array[y, x]
            cb_value = <cnp.int_t>cb_array[y, x] - 128
            cr_value = <cnp.int_t>cr_array[y, x] - 128

            pixel_array[0] = y_value
            pixel_array[1] = cb_value
            pixel_array[2] = cr_value

            # r, g, b
            rgb_vals = numpy.dot(rgb_conversion_matrix, pixel_array)

            # Correcting rgb values out of scope
            for i in range(3):
                val = rgb_vals[i]
                if val <= 0:
                    val = 0
                if val >= 255:
                    val = 255
                image[y, x, i] = val
    return image