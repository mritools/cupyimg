"""Additional tests for convolve & correlate.

These tests compare directly to the output of NumPy
"""
import itertools

import numpy as np
import cupy as cp
import pytest

from cupyimg.numpy import correlate, convolve


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, function",
    itertools.product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 6, 7],
        ["full", "valid", "same"],
        ["correlate", "convolve"],
    ),
)
def test_convolve_and_correlate(dtype_x, dtype_h, len_x, mode, function):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)

        if function == "convolve":
            func_cpu = np.convolve
            func_gpu = convolve
        elif function == "correlate":
            func_cpu = np.correlate
            func_gpu = correlate

        y = func_cpu(x_cpu, h_cpu, mode=mode)

        y2 = func_gpu(cp.asarray(x_cpu), cp.asarray(h_cpu), mode=mode)
        cp.testing.assert_allclose(y, y2)


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode, function",
    itertools.product(
        [np.float32, np.complex64, np.float64, np.complex128],
        [np.float32, np.complex64],
        [2, 3, 6, 7],
        ["full", "valid", "same"],
        ["correlate", "convolve"],
    ),
)
def test_convolve_and_correlate_complex(
    dtype_x, dtype_h, len_x, mode, function
):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    if x_cpu.dtype.kind == "c":
        x_cpu = x_cpu + 1j * x_cpu

    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        if h_cpu.dtype.kind == "c":
            h_cpu = h_cpu + 1j * h_cpu

        if function == "convolve":
            func_cpu = np.convolve
            func_gpu = convolve
        elif function == "correlate":
            func_cpu = np.correlate
            func_gpu = correlate

        y = func_cpu(x_cpu, h_cpu, mode=mode)

        y2 = func_gpu(cp.asarray(x_cpu), cp.asarray(h_cpu), mode=mode)
        cp.testing.assert_allclose(y, y2)
