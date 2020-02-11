""" Some tests for filters """
# import sys

import cupy as cp
import numpy as np
import pytest

from numpy.testing import assert_equal
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
)

from pytest import raises as assert_raises

import cupyimg.scipy.ndimage as sndi
from scipy.ndimage.filters import _gaussian_kernel1d

# from scipy._lib._numpy_compat import suppress_warnings

# def test_ticket_701():
#     # Test generic filter sizes
#     arr = cp.arange(4).reshape((2,2))
#     func = lambda x: cp.min(x)
#     res = sndi.generic_filter(arr, func, size=(1,1))
#     # The following raises an error unless ticket 701 is fixed
#     res2 = sndi.generic_filter(arr, func, size=1)
#     assert_equal(res, res2)


# def test_gh_5430():
#     # At least one of these raises an error unless gh-5430 is
#     # fixed. In py2k an int is implemented using a C long, so
#     # which one fails depends on your system. In py3k there is only
#     # one arbitrary precision integer type, so both should fail.
#     sigma = np.int32(1)
#     out = sndi._ni_support._normalize_sequence(sigma, 1)
#     assert_equal(out, [sigma])
#     sigma = np.int64(1)
#     out = sndi._ni_support._normalize_sequence(sigma, 1)
#     assert_equal(out, [sigma])
#     # This worked before; make sure it still works
#     sigma = 1
#     out = sndi._ni_support._normalize_sequence(sigma, 1)
#     assert_equal(out, [sigma])
#     # This worked before; make sure it still works
#     sigma = [1, 1]
#     out = sndi._ni_support._normalize_sequence(sigma, 2)
#     assert_equal(out, sigma)
#     # Also include the OPs original example to make sure we fixed the issue
#     x = cp.random.normal(size=(256, 256))
#     perlin = cp.zeros_like(x)
#     for i in 2**cp.arange(6):
#         perlin += sndi.filters.gaussian_filter(x, i, mode="wrap") * i**2
#     # This also fixes gh-4106, show that the OPs example now runs.
#     x = np.int64(21)
#     sndi._ni_support._normalize_sequence(x, 0)


def test_gaussian_kernel1d():
    radius = 10
    sigma = 2
    sigma2 = sigma * sigma
    x = cp.arange(-radius, radius + 1, dtype=np.double)
    phi_x = cp.exp(-0.5 * x * x / sigma2)
    phi_x /= phi_x.sum()
    assert_allclose(phi_x, _gaussian_kernel1d(sigma, 0, radius))
    assert_allclose(-phi_x * x / sigma2, _gaussian_kernel1d(sigma, 1, radius))
    assert_allclose(
        phi_x * (x * x / sigma2 - 1) / sigma2,
        _gaussian_kernel1d(sigma, 2, radius),
    )
    assert_allclose(
        phi_x * (3 - x * x / sigma2) * x / (sigma2 * sigma2),
        _gaussian_kernel1d(sigma, 3, radius),
    )


def test_orders_gauss():
    # Check order inputs to Gaussians
    arr = cp.zeros((1,))
    assert_equal(0, sndi.gaussian_filter(arr, 1, order=0).get())
    assert_equal(0, sndi.gaussian_filter(arr, 1, order=3).get())
    assert_raises(ValueError, sndi.gaussian_filter, arr, 1, -1)
    assert_equal(0, sndi.gaussian_filter1d(arr, 1, axis=-1, order=0).get())
    assert_equal(0, sndi.gaussian_filter1d(arr, 1, axis=-1, order=3).get())
    assert_raises(ValueError, sndi.gaussian_filter1d, arr, 1, -1, -1)


def test_valid_origins():
    """Regression test for #1311."""
    # func = lambda x: cp.mean(x)
    data = cp.array([1, 2, 3, 4, 5], dtype=np.float64)
    # assert_raises(ValueError, sndi.generic_filter, data, func, size=3,
    #               origin=2)
    # func2 = lambda x, y: cp.mean(x + y)
    # assert_raises(ValueError, sndi.generic_filter1d, data, func,
    #               filter_size=3, origin=2)
    assert_raises(
        ValueError, sndi.percentile_filter, data, 0.2, size=3, origin=2
    )

    for filter in [
        sndi.uniform_filter,
        sndi.minimum_filter,
        sndi.maximum_filter,
        sndi.maximum_filter1d,
        sndi.median_filter,
        sndi.minimum_filter1d,
    ]:
        # This should work, since for size == 3, the valid range for origin is
        # -1 to 1.
        list(filter(data, 3, origin=-1))
        list(filter(data, 3, origin=1))
        # Just check this raises an error instead of silently accepting or
        # segfaulting.
        assert_raises(ValueError, filter, data, 3, origin=2)


def test_bad_convolve_and_correlate_origins():
    """Regression test for gh-822."""
    # Before gh-822 was fixed, these would generate seg. faults or
    # other crashes on many system.
    assert_raises(
        ValueError,
        sndi.correlate1d,
        cp.array([0, 1, 2, 3, 4, 5]),
        cp.array([1, 1, 2, 0]),
        origin=2,
    )
    assert_raises(
        ValueError,
        sndi.correlate,
        cp.array([0, 1, 2, 3, 4, 5]),
        cp.array([0, 1, 2]),
        origin=[2],
    )
    assert_raises(
        ValueError,
        sndi.correlate,
        cp.ones((3, 5)),
        cp.ones((2, 2)),
        origin=[0, 1],
    )

    assert_raises(
        ValueError, sndi.convolve1d, cp.arange(10), cp.ones(3), origin=-2
    )
    assert_raises(
        ValueError, sndi.convolve, cp.arange(10), cp.ones(3), origin=[-2]
    )
    assert_raises(
        ValueError,
        sndi.convolve,
        cp.ones((3, 5)),
        cp.ones((2, 2)),
        origin=[0, -2],
    )


def test_multiple_modes():
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying a single mode.
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    mode1 = "reflect"
    mode2 = ["reflect", "reflect"]

    assert_array_equal(
        sndi.gaussian_filter(arr, 1, mode=mode1),
        sndi.gaussian_filter(arr, 1, mode=mode2),
    )
    assert_array_equal(
        sndi.prewitt(arr, mode=mode1), sndi.prewitt(arr, mode=mode2)
    )
    assert_array_equal(sndi.sobel(arr, mode=mode1), sndi.sobel(arr, mode=mode2))
    assert_array_equal(
        sndi.laplace(arr, mode=mode1), sndi.laplace(arr, mode=mode2)
    )
    assert_array_equal(
        sndi.gaussian_laplace(arr, 1, mode=mode1),
        sndi.gaussian_laplace(arr, 1, mode=mode2),
    )
    assert_array_equal(
        sndi.maximum_filter(arr, size=5, mode=mode1),
        sndi.maximum_filter(arr, size=5, mode=mode2),
    )
    assert_array_equal(
        sndi.minimum_filter(arr, size=5, mode=mode1),
        sndi.minimum_filter(arr, size=5, mode=mode2),
    )
    assert_array_equal(
        sndi.gaussian_gradient_magnitude(arr, 1, mode=mode1),
        sndi.gaussian_gradient_magnitude(arr, 1, mode=mode2),
    )
    assert_array_equal(
        sndi.uniform_filter(arr, 5, mode=mode1),
        sndi.uniform_filter(arr, 5, mode=mode2),
    )


def test_multiple_modes_sequentially():
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying the filters with
    # different modes sequentially
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    modes = ["reflect", "wrap"]

    expected = sndi.gaussian_filter1d(arr, 1, axis=0, mode=modes[0])
    expected = sndi.gaussian_filter1d(expected, 1, axis=1, mode=modes[1])
    assert_array_equal(expected, sndi.gaussian_filter(arr, 1, mode=modes))

    expected = sndi.uniform_filter1d(arr, 5, axis=0, mode=modes[0])
    expected = sndi.uniform_filter1d(expected, 5, axis=1, mode=modes[1])
    assert_array_equal(expected, sndi.uniform_filter(arr, 5, mode=modes))

    expected = sndi.maximum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = sndi.maximum_filter1d(expected, size=5, axis=1, mode=modes[1])
    assert_array_equal(expected, sndi.maximum_filter(arr, size=5, mode=modes))

    expected = sndi.minimum_filter1d(arr, size=5, axis=0, mode=modes[0])
    expected = sndi.minimum_filter1d(expected, size=5, axis=1, mode=modes[1])
    assert_array_equal(expected, sndi.minimum_filter(arr, size=5, mode=modes))


def test_multiple_modes_prewitt():
    # Test prewitt filter for multiple extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array([[1.0, -3.0, 2.0], [1.0, -2.0, 1.0], [1.0, -1.0, 0.0]])

    modes = ["reflect", "wrap"]

    assert_array_equal(expected, sndi.prewitt(arr, mode=modes))


def test_multiple_modes_sobel():
    # Test sobel filter for multiple extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array([[1.0, -4.0, 3.0], [2.0, -3.0, 1.0], [1.0, -1.0, 0.0]])

    modes = ["reflect", "wrap"]

    assert_array_equal(expected, sndi.sobel(arr, mode=modes))


def test_multiple_modes_laplace():
    # Test laplace filter for multiple extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array([[-2.0, 2.0, 1.0], [-2.0, -3.0, 2.0], [1.0, 1.0, 0.0]])

    modes = ["reflect", "wrap"]

    assert_array_equal(expected, sndi.laplace(arr, mode=modes))


def test_multiple_modes_gaussian_laplace():
    # Test gaussian_laplace filter for multiple extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array(
        [
            [-0.28438687, 0.01559809, 0.19773499],
            [-0.36630503, -0.20069774, 0.07483620],
            [0.15849176, 0.18495566, 0.21934094],
        ]
    )

    modes = ["reflect", "wrap"]

    assert_array_almost_equal(
        expected, sndi.gaussian_laplace(arr, 1, mode=modes)
    )


def test_multiple_modes_gaussian_gradient_magnitude():
    # Test gaussian_gradient_magnitude filter for multiple
    # extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array(
        [
            [0.04928965, 0.09745625, 0.06405368],
            [0.23056905, 0.14025305, 0.04550846],
            [0.19894369, 0.14950060, 0.06796850],
        ]
    )

    modes = ["reflect", "wrap"]

    calculated = sndi.gaussian_gradient_magnitude(arr, 1, mode=modes)

    assert_array_almost_equal(expected, calculated)


def test_multiple_modes_uniform():
    # Test uniform filter for multiple extrapolation modes
    arr = cp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])

    expected = cp.array(
        [[0.32, 0.40, 0.48], [0.20, 0.28, 0.32], [0.28, 0.32, 0.40]]
    )

    modes = ["reflect", "wrap"]

    assert_array_almost_equal(expected, sndi.uniform_filter(arr, 5, mode=modes))


def test_gaussian_truncate():
    # Test that Gaussian filters can be truncated at different widths.
    # These tests only check that the result has the expected number
    # of nonzero elements.
    arr = cp.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (sndi.gaussian_filter(arr, 5, truncate=2) > 0).sum().get()
    assert_equal(num_nonzeros_2, 21 ** 2)
    num_nonzeros_5 = (sndi.gaussian_filter(arr, 5, truncate=5) > 0).sum().get()
    assert_equal(num_nonzeros_5, 51 ** 2)

    # Test truncate when sigma is a sequence.
    f = sndi.gaussian_filter(arr, [0.5, 2.5], truncate=3.5).get()
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)

    # Test gaussian_filter1d.
    x = cp.zeros(51)
    x[25] = 1
    f = sndi.gaussian_filter1d(x, sigma=2, truncate=3.5).get()
    n = (f > 0).sum()
    assert_equal(n, 15)

    # Test gaussian_laplace
    y = sndi.gaussian_laplace(x, sigma=2, truncate=3.5).get()
    nonzero_indices = np.nonzero(y != 0)[0]
    n = nonzero_indices.ptp() + 1
    assert_equal(n, 15)

    # Test gaussian_gradient_magnitude
    y = sndi.gaussian_gradient_magnitude(x, sigma=2, truncate=3.5).get()
    nonzero_indices = np.nonzero(y != 0)[0]
    n = nonzero_indices.ptp() + 1
    assert_equal(n, 15)


class TestThreading(object):
    def check_func_thread(self, n, fun, args, out):
        from threading import Thread

        thrds = [
            Thread(target=fun, args=args, kwargs={"output": out[x]})
            for x in range(n)
        ]
        [t.start() for t in thrds]
        [t.join() for t in thrds]

    def check_func_serial(self, n, fun, args, out):
        for i in range(n):
            fun(*args, output=out[i])

    def test_correlate1d(self):
        d = cp.random.randn(5000)
        os = cp.empty((4, d.size))
        ot = cp.empty_like(os)
        self.check_func_serial(4, sndi.correlate1d, (d, cp.arange(5)), os)
        self.check_func_thread(4, sndi.correlate1d, (d, cp.arange(5)), ot)
        assert_array_equal(os, ot)

    def test_correlate(self):
        d = cp.random.randn(500, 500)
        k = cp.random.randn(10, 10)
        os = cp.empty([4] + list(d.shape))
        ot = cp.empty_like(os)
        self.check_func_serial(4, sndi.correlate, (d, k), os)
        self.check_func_thread(4, sndi.correlate, (d, k), ot)
        assert_array_equal(os, ot)

    # TODO: median filter case currently fails
    # def test_median_filter(self):
    #     d = cp.random.randn(500, 500)
    #     os = cp.empty([4] + list(d.shape))
    #     ot = cp.empty_like(os)
    #     self.check_func_serial(4, sndi.median_filter, (d, 3), os)
    #     self.check_func_thread(4, sndi.median_filter, (d, 3), ot)
    #     assert_array_equal(os, ot)

    def test_uniform_filter1d(self):
        d = cp.random.randn(5000)
        os = cp.empty((4, d.size))
        ot = cp.empty_like(os)
        self.check_func_serial(4, sndi.uniform_filter1d, (d, 5), os)
        self.check_func_thread(4, sndi.uniform_filter1d, (d, 5), ot)
        assert_array_equal(os, ot)

    def test_minmax_filter(self):
        d = cp.random.randn(500, 500)
        os = cp.empty([4] + list(d.shape))
        ot = cp.empty_like(os)
        self.check_func_serial(4, sndi.maximum_filter, (d, 3), os)
        self.check_func_thread(4, sndi.maximum_filter, (d, 3), ot)
        assert_array_equal(os, ot)
        self.check_func_serial(4, sndi.minimum_filter, (d, 3), os)
        self.check_func_thread(4, sndi.minimum_filter, (d, 3), ot)
        assert_array_equal(os, ot)


def test_minmaximum_filter1d():
    # Regression gh-3898
    in_ = cp.arange(10)
    out = sndi.minimum_filter1d(in_, 1)
    assert_array_equal(in_, out)
    out = sndi.maximum_filter1d(in_, 1)
    assert_array_equal(in_, out)
    # Test reflect
    out = sndi.minimum_filter1d(in_, 5, mode="reflect")
    assert_array_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = sndi.maximum_filter1d(in_, 5, mode="reflect")
    assert_array_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    # Test constant
    out = sndi.minimum_filter1d(in_, 5, mode="constant", cval=-1)
    assert_array_equal([-1, -1, 0, 1, 2, 3, 4, 5, -1, -1], out)
    out = sndi.maximum_filter1d(in_, 5, mode="constant", cval=10)
    assert_array_equal([10, 10, 4, 5, 6, 7, 8, 9, 10, 10], out)
    # Test nearest
    out = sndi.minimum_filter1d(in_, 5, mode="nearest")
    assert_array_equal([0, 0, 0, 1, 2, 3, 4, 5, 6, 7], out)
    out = sndi.maximum_filter1d(in_, 5, mode="nearest")
    assert_array_equal([2, 3, 4, 5, 6, 7, 8, 9, 9, 9], out)
    # Test wrap
    out = sndi.minimum_filter1d(in_, 5, mode="wrap")
    assert_array_equal([0, 0, 0, 1, 2, 3, 4, 5, 0, 0], out)
    out = sndi.maximum_filter1d(in_, 5, mode="wrap")
    assert_array_equal([9, 9, 4, 5, 6, 7, 8, 9, 9, 9], out)


@pytest.mark.xfail(True, reason="integer rounding bug")
def test_uniform_filter1d_roundoff_errors():
    # gh-6930
    in_ = cp.array(np.repeat([0, 1, 0], [9, 9, 9]))
    for filter_size in range(3, 10):
        out = sndi.uniform_filter1d(in_, filter_size)
        assert_equal(out.sum().get(), 10 - filter_size)


def test_footprint_all_zeros():
    # regression test for gh-6876: footprint of all zeros segfaults
    arr = cp.random.randint(0, 100, (100, 100))
    kernel = cp.zeros((3, 3), bool)
    with assert_raises(ValueError):
        sndi.maximum_filter(arr, footprint=kernel)


# def test_gaussian_filter():
#     # Test gaussian filter with np.float16
#     # gh-8207
#     data = cp.array([1], dtype=np.float16)
#     sigma = 1.0
#     # grlee77: TODO: raise RuntimeError on float16 input to correlate1d?
#     # with assert_raises(RuntimeError):


def test_rank_filter_noninteger_rank():
    # regression test for issue 9388: ValueError for
    # non integer rank when performing rank_filter
    arr = cp.random.random((10, 20, 30))
    assert_raises(
        TypeError,
        sndi.rank_filter,
        arr,
        0.5,
        footprint=cp.ones((1, 1, 10), dtype=bool),
    )


# def test_size_footprint_both_set():
#     # test for input validation, expect user warning when
#     # size and footprint is set
#     with suppress_warnings() as sup:
#         sup.filter(UserWarning,
#                    "ignoring size because footprint is set")
#         arr = cp.random.random((10, 20, 30))
#         sndi.rank_filter(
#             arr, 5, size=2, footprint=cp.ones((1, 1, 10), dtype=bool)
#         )
