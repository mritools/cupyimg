import itertools

import numpy as np
import cupy as cp
import pytest
from scipy import ndimage as ndi

from cupyimg.scipy.ndimage import spline_filter1d


@pytest.mark.parametrize(
    "dtype, order, axis, mode",
    itertools.product(
        [np.float32, np.float64],
        [2, 3, 4, 5],
        [0, -1],
        ["mirror"],  # "constant", "nearest", "reflect", "wrap"],
    ),
)
def test_spline_filter_1d_real(dtype, order, axis, mode):
    rstate = np.random.RandomState(1234)
    if dtype == cp.float32:
        atol = rtol = 1e-5
    else:
        atol = rtol = 1e-11
    x = rstate.randn(156, 256).astype(dtype)
    xd = cp.asarray(x)
    y = ndi.spline_filter1d(x, order=order, axis=axis, output=dtype)
    yd = spline_filter1d(xd, order=order, axis=axis, output=dtype)
    cp.testing.assert_allclose(y, yd, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, order, axis, mode",
    itertools.product(
        [np.float32, np.float64],
        [2, 3, 4, 5],
        [0, -1],
        ["mirror", "constant", "nearest", "reflect", "wrap"],
    ),
)
def test_spline_filter_1d_real_ndimage(dtype, order, axis, mode):
    rstate = np.random.RandomState(1234)
    atol = rtol = 1e-6
    x = rstate.randn(156, 256).astype(dtype)
    xd = cp.asarray(x)
    y = ndi.spline_filter1d(x, order=order, axis=axis, output=dtype)
    yd = spline_filter1d(
        xd, order=order, axis=axis, output=dtype, allow_float32=False,
    )
    cp.testing.assert_allclose(y, yd, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, order, axis, mode",
    itertools.product(
        [np.complex64, np.complex128],
        [2, 3, 4, 5],
        [0, -1],
        ["mirror", "constant", "nearest", "reflect", "wrap"],
    ),
)
def test_spline_filter_1d_complex(dtype, order, axis, mode):
    order = 3
    rstate = np.random.RandomState(1234)
    if dtype == cp.complex64:
        atol = rtol = 1e-5
        real_dtype = cp.float32
    else:
        atol = rtol = 1e-11
        real_dtype = cp.float64

    x = rstate.randn(156, 256).astype(real_dtype)
    x = x + 1j * rstate.randn(156, 256).astype(real_dtype)
    y = ndi.spline_filter1d(x.real, order=order, axis=axis, output=real_dtype)
    y = y + 1j * ndi.spline_filter1d(
        x.imag, order=order, axis=axis, output=real_dtype
    )

    xd = cp.asarray(x)
    yd = spline_filter1d(xd, order=order, axis=axis, output=dtype)
    cp.testing.assert_allclose(y, yd, atol=atol, rtol=rtol)
