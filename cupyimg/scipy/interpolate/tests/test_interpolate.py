import itertools

import cupy as cp
import numpy as np
from numpy.testing import assert_equal
from cupy.testing import assert_array_almost_equal, assert_allclose
from pytest import raises as assert_raises

# scipy functions used as a reference in tests
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from cupyimg.scipy.interpolate import RegularGridInterpolator, interpn


class TestRegularGridInterpolator(object):
    def _get_sample_4d(self, xp=cp):
        # create a 4-D grid of 3 points in each dimension
        points = [(0.0, 0.5, 1.0)] * 4
        values = xp.asarray([0.0, 0.5, 1.0])
        values0 = values[:, xp.newaxis, xp.newaxis, xp.newaxis]
        values1 = values[xp.newaxis, :, xp.newaxis, xp.newaxis]
        values2 = values[xp.newaxis, xp.newaxis, :, xp.newaxis]
        values3 = values[xp.newaxis, xp.newaxis, xp.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return points, values

    def _get_sample_4d_2(self):
        # create another 4-D grid of 3 points in each dimension
        points = [(0.0, 0.5, 1.0)] * 2 + [(0.0, 5.0, 10.0)] * 2
        values = cp.asarray([0.0, 0.5, 1.0])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return points, values

    def test_list_input(self):
        points, values = self._get_sample_4d()

        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )

        for method in ["linear", "nearest"]:
            interp = RegularGridInterpolator(
                points, values.tolist(), method=method
            )
            v1 = interp(sample.tolist())
            interp = RegularGridInterpolator(points, values, method=method)
            v2 = interp(sample)
            assert_allclose(v1, v2)

    def test_complex(self):
        points, values = self._get_sample_4d()
        values = values - 2j * values
        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )

        for method in ["linear", "nearest"]:
            interp = RegularGridInterpolator(points, values, method=method)
            rinterp = RegularGridInterpolator(
                points, values.real, method=method
            )
            iinterp = RegularGridInterpolator(
                points, values.imag, method=method
            )

            v1 = interp(sample)
            v2 = rinterp(sample) + 1j * iinterp(sample)
            assert_allclose(v1, v2)

    def test_linear_xi1d(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([0.1, 0.1, 10.0, 9.0])
        wanted = 1001.1
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_xi3d(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )
        wanted = cp.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")
        sample = cp.asarray([0.1, 0.1, 0.9, 0.9])
        wanted = 1100.0
        assert_array_almost_equal(interp(sample), wanted)
        sample = cp.asarray([0.1, 0.1, 0.1, 0.1])
        wanted = 0.0
        assert_array_almost_equal(interp(sample), wanted)
        sample = cp.asarray([0.0, 0.0, 0.0, 0.0])
        wanted = 0.0
        assert_array_almost_equal(interp(sample), wanted)
        sample = cp.asarray([1.0, 1.0, 1.0, 1.0])
        wanted = 1111.0
        assert_array_almost_equal(interp(sample), wanted)
        sample = cp.asarray([0.1, 0.4, 0.6, 0.9])
        wanted = 1055.0
        assert_array_almost_equal(interp(sample), wanted)

    def test_linear_edges(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        wanted = cp.asarray([0.0, 1111.0])
        assert_array_almost_equal(interp(sample), wanted)

    def test_valid_create(self):
        # create a 2-D grid of 3 points in each dimension
        points = [(0.0, 0.5, 1.0), (0.0, 1.0, 0.5)]
        values = cp.asarray([0.0, 0.5, 1.0])
        values0 = values[:, cp.newaxis]
        values1 = values[cp.newaxis, :]
        values = values0 + values1 * 10
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [((0.0, 0.5, 1.0),), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 0.75, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(ValueError, RegularGridInterpolator, points, values)
        points = [(0.0, 0.5, 1.0), (0.0, 0.5, 1.0)]
        assert_raises(
            ValueError,
            RegularGridInterpolator,
            points,
            values,
            method="undefmethod",
        )

    def test_valid_call(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        assert_raises(ValueError, interp, sample, "undefmethod")
        sample = cp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        assert_raises(ValueError, interp, sample)
        sample = cp.asarray([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.1]])
        assert_raises(ValueError, interp, sample)

    def test_out_of_bounds_extrap(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(
            points, values, bounds_error=False, fill_value=None
        )
        sample = cp.asarray(
            [
                [-0.1, -0.1, -0.1, -0.1],
                [1.1, 1.1, 1.1, 1.1],
                [21, 2.1, -1.1, -11],
                [2.1, 2.1, -1.1, -1.1],
            ]
        )
        wanted = cp.asarray([0.0, 1111.0, 11.0, 11.0])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = cp.asarray([-111.1, 1222.1, -11068.0, -1186.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_extrap2(self):
        points, values = self._get_sample_4d_2()
        interp = RegularGridInterpolator(
            points, values, bounds_error=False, fill_value=None
        )
        sample = cp.asarray(
            [
                [-0.1, -0.1, -0.1, -0.1],
                [1.1, 1.1, 1.1, 1.1],
                [21, 2.1, -1.1, -11],
                [2.1, 2.1, -1.1, -1.1],
            ]
        )
        wanted = cp.asarray([0.0, 11.0, 11.0, 11.0])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        wanted = cp.asarray([-12.1, 133.1, -1069.0, -97.9])
        assert_array_almost_equal(interp(sample, method="linear"), wanted)

    def test_out_of_bounds_fill(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(
            points, values, bounds_error=False, fill_value=cp.nan
        )
        sample = cp.asarray(
            [
                [-0.1, -0.1, -0.1, -0.1],
                [1.1, 1.1, 1.1, 1.1],
                [2.1, 2.1, -1.1, -1.1],
            ]
        )
        wanted = cp.asarray([cp.nan, cp.nan, cp.nan])
        assert_array_almost_equal(interp(sample, method="nearest"), wanted)
        assert_array_almost_equal(interp(sample, method="linear"), wanted)
        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )
        wanted = cp.asarray([1001.1, 846.2, 555.5])
        assert_array_almost_equal(interp(sample), wanted)

    def test_nearest_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values, method="nearest")

        points_cpu, values_cpu = self._get_sample_4d(xp=np)
        points_qhull = itertools.product(*points_cpu)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values_cpu.reshape(-1)
        interp_qhull = NearestNDInterpolator(points_qhull, values_qhull)
        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )
        assert_array_almost_equal(interp(sample), interp_qhull(sample.get()))

    def test_linear_compare_qhull(self):
        points, values = self._get_sample_4d()
        interp = RegularGridInterpolator(points, values)

        points_cpu, values_cpu = self._get_sample_4d(xp=np)
        points_qhull = itertools.product(*points_cpu)
        points_qhull = [p for p in points_qhull]
        points_qhull = np.asarray(points_qhull)
        values_qhull = values_cpu.reshape(-1)
        interp_qhull = LinearNDInterpolator(points_qhull, values_qhull)
        sample = cp.asarray(
            [[0.1, 0.1, 1.0, 0.9], [0.2, 0.1, 0.45, 0.8], [0.5, 0.5, 0.5, 0.5]]
        )
        assert_array_almost_equal(interp(sample), interp_qhull(sample.get()))

    def test_invalid_fill_value(self):
        cp.random.seed(1234)
        x = cp.linspace(0, 2, 5)
        y = cp.linspace(0, 1, 7)
        values = cp.random.rand(5, 7)

        # integers can be cast to floats
        RegularGridInterpolator((x, y), values, fill_value=1)

        # complex values cannot
        assert_raises(
            ValueError,
            RegularGridInterpolator,
            (x, y),
            values,
            fill_value=1 + 2j,
        )

    def test_fillvalue_type(self):
        # from #3703; test that interpolator object construction succeeds
        values = cp.ones((10, 20, 30), dtype=">f4")
        points = [cp.arange(n) for n in values.shape]
        xi = [(1, 1, 1)]
        interpolator = RegularGridInterpolator(points, values)
        interpolator = RegularGridInterpolator(points, values, fill_value=0.0)


class TestInterpN(object):
    def _sample_2d_data(self):
        x = cp.arange(1, 6)
        x = cp.array([0.5, 2.0, 3.0, 4.0, 5.5])
        y = cp.arange(1, 6)
        y = cp.array([0.5, 2.0, 3.0, 4.0, 5.5])
        z = cp.array(
            [
                [1, 2, 1, 2, 1],
                [1, 2, 1, 2, 1],
                [1, 2, 3, 2, 1],
                [1, 2, 2, 2, 1],
                [1, 2, 1, 2, 1],
            ]
        )
        return x, y, z

    # def test_spline_2d(self):
    #     x, y, z = self._sample_2d_data()
    #     lut = RectBivariateSpline(x, y, z)

    #     xi = cp.array([[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3],
    #                    [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]).T
    #     assert_array_almost_equal(interpn((x, y), z, xi, method="splinef2d"),
    #                               lut.ev(xi[:, 0], xi[:, 1]))

    def test_list_input(self):
        x, y, z = self._sample_2d_data()
        xi = cp.asarray(
            [[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]
        ).T

        for method in ["nearest", "linear"]:  # 'splinef2d']:
            v1 = interpn((x, y), z, xi, method=method)
            v2 = interpn(
                (x.tolist(), y.tolist()), z.tolist(), xi.tolist(), method=method
            )
            assert_allclose(v1, v2, err_msg=method)

    # def test_spline_2d_outofbounds(self):
    #     x = cp.asarray([.5, 2., 3., 4., 5.5])
    #     y = cp.asarray([.5, 2., 3., 4., 5.5])
    #     z = cp.asarray([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1],
    #                     [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    #     lut = RectBivariateSpline(x, y, z)

    #     xi = cp.asarray([[1, 2.3, 6.3, 0.5, 3.3, 1.2, 3],
    #                      [1, 3.3, 1.2, -4.0, 5.0, 1.0, 3]]).T
    #     actual = interpn((x, y), z, xi, method="splinef2d",
    #                      bounds_error=False, fill_value=999.99)
    #     expected = lut.ev(xi[:, 0], xi[:, 1])
    #     expected[2:4] = 999.99
    #     assert_array_almost_equal(actual, expected)

    #     # no extrapolation for splinef2d
    #     assert_raises(ValueError, interpn, (x, y), z, xi, method="splinef2d",
    #                   bounds_error=False, fill_value=None)

    def _sample_4d_data(self):
        points = [(0.0, 0.5, 1.0)] * 2 + [(0.0, 5.0, 10.0)] * 2
        values = cp.asarray([0.0, 0.5, 1.0])
        values0 = values[:, cp.newaxis, cp.newaxis, cp.newaxis]
        values1 = values[cp.newaxis, :, cp.newaxis, cp.newaxis]
        values2 = values[cp.newaxis, cp.newaxis, :, cp.newaxis]
        values3 = values[cp.newaxis, cp.newaxis, cp.newaxis, :]
        values = values0 + values1 * 10 + values2 * 100 + values3 * 1000
        return points, values

    def test_linear_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values)
        sample = cp.asarray([[0.1, 0.1, 10.0, 9.0]])
        wanted = interpn(points, values, sample, method="linear")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_linear_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = cp.asarray([[0.1, -0.1, 10.1, 9.0]])
        wanted = 999.99
        actual = interpn(
            points,
            values,
            sample,
            method="linear",
            bounds_error=False,
            fill_value=999.99,
        )
        assert_array_almost_equal(actual, wanted)

    def test_nearest_4d(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        interp_rg = RegularGridInterpolator(points, values, method="nearest")
        sample = cp.asarray([[0.1, 0.1, 10.0, 9.0]])
        wanted = interpn(points, values, sample, method="nearest")
        assert_array_almost_equal(interp_rg(sample), wanted)

    def test_4d_nearest_outofbounds(self):
        # create a 4-D grid of 3 points in each dimension
        points, values = self._sample_4d_data()
        sample = cp.asarray([[0.1, -0.1, 10.1, 9.0]])
        wanted = 999.99
        actual = interpn(
            points,
            values,
            sample,
            method="nearest",
            bounds_error=False,
            fill_value=999.99,
        )
        assert_array_almost_equal(actual, wanted)

    def test_xi_1d(self):
        # verify that 1-D xi works as expected
        points, values = self._sample_4d_data()
        sample = cp.asarray([0.1, 0.1, 10.0, 9.0])
        v1 = interpn(points, values, sample, bounds_error=False)
        v2 = interpn(points, values, sample[None, :], bounds_error=False)
        assert_allclose(v1, v2)

    def test_xi_nd(self):
        # verify that higher-d xi works as expected
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        sample = cp.asarray(np.random.rand(2, 3, 4))

        v1 = interpn(
            points, values, sample, method="nearest", bounds_error=False
        )
        assert_equal(v1.shape, (2, 3))

        v2 = interpn(
            points,
            values,
            sample.reshape(-1, 4),
            method="nearest",
            bounds_error=False,
        )
        assert_allclose(v1, v2.reshape(v1.shape))

    def test_xi_broadcast(self):
        # verify that the interpolators broadcast xi
        x, y, values = self._sample_2d_data()
        points = (x, y)

        xi = cp.linspace(0, 1, 2)
        yi = cp.linspace(0, 3, 3)

        for method in ["nearest", "linear"]:  # 'splinef2d']:
            sample = (xi[:, None], yi[None, :])
            v1 = interpn(
                points, values, sample, method=method, bounds_error=False
            )
            assert_equal(v1.shape, (2, 3))

            xx, yy = np.meshgrid(xi, yi)
            sample = cp.c_[xx.T.ravel(), yy.T.ravel()]

            v2 = interpn(
                points, values, sample, method=method, bounds_error=False
            )
            assert_allclose(v1, v2.reshape(v1.shape))

    def test_nonscalar_values(self):
        # Verify that non-scalar valued values also works
        points, values = self._sample_4d_data()

        np.random.seed(1234)
        values = cp.asarray(np.random.rand(3, 3, 3, 3, 6))
        sample = cp.asarray(np.random.rand(7, 11, 4))

        for method in ["nearest", "linear"]:
            v = interpn(
                points, values, sample, method=method, bounds_error=False
            )
            assert_equal(v.shape, (7, 11, 6), err_msg=method)

            vs = [
                interpn(
                    points,
                    values[..., j],
                    sample,
                    method=method,
                    bounds_error=False,
                )
                for j in range(6)
            ]
            v2 = cp.asarray(vs).transpose(1, 2, 0)

            assert_allclose(v, v2, err_msg=method)

        # # Vector-valued splines supported with fitpack
        # assert_raises(ValueError, interpn, points, values, sample,
        #               method='splinef2d')

    def test_complex(self):
        x, y, values = self._sample_2d_data()
        points = (x, y)
        values = values - 2j * values

        sample = cp.asarray(
            [[1, 2.3, 5.3, 0.5, 3.3, 1.2, 3], [1, 3.3, 1.2, 4.0, 5.0, 1.0, 3]]
        ).T

        for method in ["linear", "nearest"]:
            v1 = interpn(points, values, sample, method=method)
            v2r = interpn(points, values.real, sample, method=method)
            v2i = interpn(points, values.imag, sample, method=method)
            v2 = v2r + 1j * v2i
            assert_allclose(v1, v2)

        # # Complex-valued data not supported by spline2fd
        # _assert_warns(cp.ComplexWarning, interpn, points, values,
        #               sample, method='splinef2d')
