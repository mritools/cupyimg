"""Test CuPy-specific functionality not covered elsewhere."""
import itertools

import cupy as cp
import pytest

from cupyimg.scipy.ndimage import correlate, convolve

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@pytest.mark.parametrize(
    "w, func",
    itertools.product(
        [
            cp.ones((5, 5), dtype=float),
            cp.asarray([[1, 0], [0, 0], [0, -1]]),
            cp.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        ],
        [correlate, convolve],
    ),
)
def test_convolve_with_masked_weights(w, func):
    rstate = cp.random.RandomState(5)
    x = rstate.randn(16, 7)
    y1 = func(x, w, use_weights_mask=False)
    y2 = func(x, w, use_weights_mask=True)
    cp.testing.assert_array_equal(y1, y2)


@pytest.mark.parametrize(
    "dtype, func",
    itertools.product(
        [cp.float32, cp.float64, cp.complex64, cp.complex128],
        [correlate, convolve],
    ),
)
def test_convolve_precision(dtype, func):
    rstate = cp.random.RandomState(5)
    x = rstate.randn(16, 7).astype(dtype)
    w = rstate.randn(3, 4).astype(dtype)

    # Note: single_precision flag only affects what dtype w is cast to
    #       internally. The output precision should match the input precision.
    y1 = func(x, w, allow_float32=False)
    assert y1.dtype == x.dtype

    y2 = func(x, w, allow_float32=True)
    assert y2.dtype == x.dtype

    # not identical due to differing internal precision used above
    cp.testing.assert_allclose(y1, y2, rtol=1e-4)
