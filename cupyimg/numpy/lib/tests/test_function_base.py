import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_raises_regex, assert_raises, assert_equal

from cupyimg.numpy.lib import gradient


class TestGradient(object):
    def test_basic(self):
        v = [[1, 1], [3, 4]]
        x = cp.asarray(v)
        dx = [
            cp.asarray([[2.0, 3.0], [2.0, 3.0]]),
            cp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        ]
        assert_array_equal(gradient(x), dx)
        assert_array_equal(gradient(v), dx)

    def test_args(self):
        dx = cp.cumsum(cp.ones(5))
        dx_uneven = [1.0, 2.0, 5.0, 9.0, 11.0]
        f_2d = cp.arange(25).reshape(5, 5)

        # distances must be scalars or have size equal to gradient[axis]
        gradient(cp.arange(5), 3.0)
        gradient(cp.arange(5), cp.array(3.0))
        gradient(cp.arange(5), dx)
        # dy is set equal to dx because scalar
        gradient(f_2d, 1.5)
        gradient(f_2d, cp.array(1.5))

        gradient(f_2d, dx_uneven, dx_uneven)
        # mix between even and uneven spaces and
        # mix between scalar and vector
        gradient(f_2d, dx, 2)

        # 2D but axis specified
        gradient(f_2d, dx, axis=1)

        # 2d coordinate arguments are not yet allowed
        assert_raises_regex(
            ValueError,
            ".*scalars or 1d",
            gradient,
            f_2d,
            cp.stack([dx] * 2, axis=-1),
            1,
        )

    def test_badargs(self):
        f_2d = cp.arange(25).reshape(5, 5)
        x = cp.cumsum(cp.ones(5))

        # wrong sizes
        assert_raises(ValueError, gradient, f_2d, x, cp.ones(2))
        assert_raises(ValueError, gradient, f_2d, 1, cp.ones(2))
        assert_raises(ValueError, gradient, f_2d, cp.ones(2), cp.ones(2))
        # wrong number of arguments
        assert_raises(TypeError, gradient, f_2d, x)
        assert_raises(TypeError, gradient, f_2d, x, axis=(0, 1))
        assert_raises(TypeError, gradient, f_2d, x, x, x)
        assert_raises(TypeError, gradient, f_2d, 1, 1, 1)
        assert_raises(TypeError, gradient, f_2d, x, x, axis=1)
        assert_raises(TypeError, gradient, f_2d, 1, 1, axis=1)

    @pytest.mark.skip(reason="cupy doesn't support datetime64 or timedelta64")
    def test_datetime64(self):
        # Make sure gradient() can handle special types like datetime64
        x = cp.array(
            [
                "1910-08-16",
                "1910-08-11",
                "1910-08-10",
                "1910-08-12",
                "1910-10-12",
                "1910-12-12",
                "1912-12-12",
            ],
            dtype="datetime64[D]",
        )
        dx = cp.array([-5, -3, 0, 31, 61, 396, 731], dtype="timedelta64[D]")
        assert_array_equal(gradient(x), dx)
        assert dx.dtype == cp.dtype("timedelta64[D]")

    @pytest.mark.skip(reason="cupy doesn't support masked arrays")
    def test_masked(self):
        # Make sure that gradient supports subclasses like masked arrays
        x = cp.ma.array([[1, 1], [3, 4]], mask=[[False, False], [False, False]])
        out = gradient(x)[0]
        assert_equal(type(out), type(x))
        # And make sure that the output and input don't have aliased mask
        # arrays
        assert x._mask is not out._mask
        # Also check that edge_order=2 doesn't alter the original mask
        x2 = cp.ma.arange(5)
        x2[2] = cp.ma.masked
        gradient(x2, edge_order=2)
        assert_array_equal(x2.mask, [False, False, True, False, False])

    def test_second_order_accurate(self):
        # Testing that the relative numerical error is less that 3% for
        # this example problem. This corresponds to second order
        # accurate finite differences for all interior and boundary
        # points.
        x = cp.linspace(0, 1, 10)
        dx = x[1] - x[0]
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = cp.abs((gradient(y, dx, edge_order=2) / analytical) - 1)
        assert cp.all(num_error < 0.03).item() is True

        # test with unevenly spaced
        cp.random.seed(0)
        x = cp.sort(cp.random.random(10))
        y = 2 * x ** 3 + 4 * x ** 2 + 2 * x
        analytical = 6 * x ** 2 + 8 * x + 2
        num_error = cp.abs((gradient(y, x, edge_order=2) / analytical) - 1)
        assert cp.all(num_error < 0.03).item() is True

    def test_spacing(self):
        f = cp.array([0, 2.0, 3.0, 4.0, 5.0, 5.0])
        f = cp.tile(f, (6, 1)) + f.reshape(-1, 1)
        x_uneven = cp.array([0.0, 0.5, 1.0, 3.0, 5.0, 7.0])
        x_even = cp.arange(6.0)

        fdx_even_ord1 = cp.tile([2.0, 1.5, 1.0, 1.0, 0.5, 0.0], (6, 1))
        fdx_even_ord2 = cp.tile([2.5, 1.5, 1.0, 1.0, 0.5, -0.5], (6, 1))
        fdx_uneven_ord1 = cp.tile([4.0, 3.0, 1.7, 0.5, 0.25, 0.0], (6, 1))
        fdx_uneven_ord2 = cp.tile([5.0, 3.0, 1.7, 0.5, 0.25, -0.25], (6, 1))

        # evenly spaced
        for edge_order, exp_res in [(1, fdx_even_ord1), (2, fdx_even_ord2)]:
            res1 = gradient(f, 1.0, axis=(0, 1), edge_order=edge_order)
            res2 = gradient(
                f, x_even, x_even, axis=(0, 1), edge_order=edge_order
            )
            res3 = gradient(f, x_even, x_even, axis=None, edge_order=edge_order)
            assert_array_equal(res1, res2)
            assert_array_equal(res2, res3)
            assert_array_almost_equal(res1[0], exp_res.T)
            assert_array_almost_equal(res1[1], exp_res)

            res1 = gradient(f, 1.0, axis=0, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=0, edge_order=edge_order)
            assert res1.shape == res2.shape
            assert_array_almost_equal(res2, exp_res.T)

            res1 = gradient(f, 1.0, axis=1, edge_order=edge_order)
            res2 = gradient(f, x_even, axis=1, edge_order=edge_order)
            assert res1.shape == res2.shape
            assert_array_equal(res2, exp_res)

        # unevenly spaced
        for edge_order, exp_res in [(1, fdx_uneven_ord1), (2, fdx_uneven_ord2)]:
            res1 = gradient(
                f, x_uneven, x_uneven, axis=(0, 1), edge_order=edge_order
            )
            res2 = gradient(
                f, x_uneven, x_uneven, axis=None, edge_order=edge_order
            )
            assert_array_equal(res1, res2)
            assert_array_almost_equal(res1[0], exp_res.T)
            assert_array_almost_equal(res1[1], exp_res)

            res1 = gradient(f, x_uneven, axis=0, edge_order=edge_order)
            assert_array_almost_equal(res1, exp_res.T)

            res1 = gradient(f, x_uneven, axis=1, edge_order=edge_order)
            assert_array_almost_equal(res1, exp_res)

        # mixed
        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=1)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=1)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_array_almost_equal(res1[0], fdx_even_ord1.T)
        assert_array_almost_equal(res1[1], fdx_uneven_ord1)

        res1 = gradient(f, x_even, x_uneven, axis=(0, 1), edge_order=2)
        res2 = gradient(f, x_uneven, x_even, axis=(1, 0), edge_order=2)
        assert_array_equal(res1[0], res2[1])
        assert_array_equal(res1[1], res2[0])
        assert_array_almost_equal(res1[0], fdx_even_ord2.T)
        assert_array_almost_equal(res1[1], fdx_uneven_ord2)

    def test_specific_axes(self):
        # Testing that gradient can work on a given axis only
        v = [[1, 1], [3, 4]]
        x = cp.array(v)
        dx = [
            cp.array([[2.0, 3.0], [2.0, 3.0]]),
            cp.array([[0.0, 0.0], [1.0, 1.0]]),
        ]
        assert_array_equal(gradient(x, axis=0), dx[0])
        assert_array_equal(gradient(x, axis=1), dx[1])
        assert_array_equal(gradient(x, axis=-1), dx[1])
        assert_array_equal(gradient(x, axis=(1, 0)), [dx[1], dx[0]])

        # test axis=None which means all axes
        assert_array_almost_equal(gradient(x, axis=None), [dx[0], dx[1]])
        # and is the same as no axis keyword given
        assert_array_almost_equal(gradient(x, axis=None), gradient(x))

        # test vararg order
        assert_array_equal(
            gradient(x, 2, 3, axis=(1, 0)), [dx[1] / 2.0, dx[0] / 3.0]
        )
        # test maximal number of varargs
        assert_raises(TypeError, gradient, x, 1, 2, axis=1)

        assert_raises(np.AxisError, gradient, x, axis=3)
        assert_raises(np.AxisError, gradient, x, axis=-3)
        # assert_raises(TypeError, gradient, x, axis=[1,])

    @pytest.mark.skip(reason="cupy doesn't implement timedelta64")
    def test_timedelta64(self):
        # Make sure gradient() can handle special types like timedelta64
        x = cp.array([-5, -3, 10, 12, 61, 321, 300], dtype="timedelta64[D]")
        dx = cp.array([2, 7, 7, 25, 154, 119, -21], dtype="timedelta64[D]")
        assert_array_equal(gradient(x), dx)
        assert dx.dtype == cp.dtype("timedelta64[D]")

    def test_inexact_dtypes(self):
        for dt in [np.float16, np.float32, np.float64]:
            # dtypes should not be promoted in a different way to what diff does
            x = cp.array([1, 2, 3], dtype=dt)
            assert_equal(gradient(x).dtype, cp.diff(x).dtype)

    def test_values(self):
        # needs at least 2 points for edge_order ==1
        gradient(cp.arange(2), edge_order=1)
        # needs at least 3 points for edge_order ==1
        gradient(cp.arange(3), edge_order=2)

        assert_raises(ValueError, gradient, cp.arange(0), edge_order=1)
        assert_raises(ValueError, gradient, cp.arange(0), edge_order=2)
        assert_raises(ValueError, gradient, cp.arange(1), edge_order=1)
        assert_raises(ValueError, gradient, cp.arange(1), edge_order=2)
        assert_raises(ValueError, gradient, cp.arange(2), edge_order=2)
