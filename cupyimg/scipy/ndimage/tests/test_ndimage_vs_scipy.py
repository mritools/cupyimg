"""Tests that compare directly to SciPy."""
import itertools

import cupy as cp
import numpy as np
from scipy import ndimage as ndi

from cupyimg.scipy.ndimage import convolve1d, correlate1d
from cupyimg.scipy.signal import upfirdn
from cupyimg.scipy.ndimage._ni_support import _get_ndimage_mode_kwargs

import pytest


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode",
    itertools.product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 6, 7],
        ["constant", "mirror", "nearest", "reflect", "wrap"],
    ),
)
def test_convolve1d(dtype_x, dtype_h, len_x, mode):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, len_x):
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        min_origin = -(len_h // 2)
        max_origin = (len_h - 1) // 2
        for origin in range(min_origin, max_origin + 1):
            y = ndi.convolve1d(x_cpu, h_cpu, mode=mode, cval=0, origin=origin)

            # test via convolve1d
            y3 = convolve1d(
                cp.asarray(x_cpu),
                cp.asarray(h_cpu),
                mode=mode,
                cval=0,
                origin=origin,
            )
            cp.testing.assert_allclose(y, y3)

            # test using upfirdn directly
            offset = len(h_cpu) // 2 + origin
            mode_kwargs = _get_ndimage_mode_kwargs(mode, cval=0)
            y2 = upfirdn(
                cp.asarray(h_cpu),
                cp.asarray(x_cpu),
                offset=offset,
                **mode_kwargs,
            )[:len_x]
            cp.testing.assert_allclose(y, y2)

        for origin in [min_origin - 1, max_origin + 1]:
            with pytest.raises(ValueError):
                convolve1d(
                    cp.asarray(x_cpu),
                    cp.asarray(h_cpu),
                    mode=mode,
                    cval=0,
                    origin=origin,
                )


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode",
    itertools.product(
        [np.float32, np.float64],
        [np.float32, np.float64],
        [2, 3, 6, 7],
        ["constant", "mirror", "nearest", "reflect", "wrap"],
    ),
)
def test_correlate1d(dtype_x, dtype_h, len_x, mode):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, 2 * len_x + 2):  # include cases for len_h > len_x
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)
        min_origin = -(len_h // 2)
        max_origin = (len_h - 1) // 2

        for origin in range(min_origin, max_origin + 1):
            y = ndi.correlate1d(x_cpu, h_cpu, mode=mode, cval=0, origin=origin)

            # test via convolve1d
            y3 = correlate1d(
                cp.asarray(x_cpu),
                cp.asarray(h_cpu),
                mode=mode,
                cval=0,
                origin=origin,
            )
            cp.testing.assert_allclose(y, y3)

        for origin in [min_origin - 1, max_origin + 1]:
            with pytest.raises(ValueError):
                correlate1d(
                    cp.asarray(x_cpu),
                    cp.asarray(h_cpu),
                    mode=mode,
                    cval=0,
                    origin=origin,
                )


@pytest.mark.parametrize(
    "dtype_x, dtype_h, len_x, mode",
    itertools.product(
        [np.float32, np.float64, np.complex64, np.complex128],
        [np.float32, np.float64, np.complex64, np.complex128],
        [6],
        ["constant", "mirror", "nearest", "reflect", "wrap"],
    ),
)
def test_correlate1d_complex(dtype_x, dtype_h, len_x, mode):
    x_cpu = np.arange(1, 1 + len_x, dtype=dtype_x)
    for len_h in range(1, 2 * len_x + 2):  # include cases for len_h > len_x
        h_cpu = np.arange(1, 1 + len_h, dtype=dtype_h)

        y = ndi.correlate1d(x_cpu.real, h_cpu.real, mode=mode, cval=0)
        y = y + 1j * ndi.correlate1d(x_cpu.imag, h_cpu.imag, mode=mode, cval=0)

        # test via convolve1d
        y3 = correlate1d(
            cp.asarray(x_cpu), cp.asarray(h_cpu), mode=mode, cval=0
        )
        cp.testing.assert_allclose(y, y3)
