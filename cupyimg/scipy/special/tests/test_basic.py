import itertools
import unittest

import cupy
import numpy
from cupy import testing
import scipy.special  # NOQA
import scipy.special._ufuncs as cephes

import cupyimg.scipy.special  # NOQA
from cupyimg.testing import numpy_cupyimg_allclose


@testing.gpu
@testing.with_requires("scipy")
class TestSpecialConvex(unittest.TestCase):
    def test_huber_basic(self):
        huber = cupyimg.scipy.special.huber
        assert huber(-1, 1.5) == cupy.inf
        testing.assert_allclose(huber(2, 1.5), 0.5 * 1.5 ** 2)
        testing.assert_allclose(huber(2, 2.5), 2 * (2.5 - 0.5 * 2))

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupyimg_allclose(scipy_name="scp")
    def test_huber(self, xp, scp, dtype):
        z = testing.shaped_random((10, 2), xp=xp, dtype=dtype)
        return scp.special.huber(z[:, 0], z[:, 1])

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupyimg_allclose(scipy_name="scp")
    def test_entr(self, xp, scp, dtype):
        values = (0, 0.5, 1.0, cupy.inf)
        signs = [-1, 1]
        arr = []
        for sgn, v in itertools.product(signs, values):
            arr.append(sgn * v)
        z = xp.asarray(arr, dtype=dtype)
        return scp.special.entr(z)

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupyimg_allclose(scipy_name="scp")
    def test_rel_entr(self, xp, scp, dtype):
        values = (0, 0.5, 1.0)
        signs = [-1, 1]
        arr = []
        arr = []
        for sgna, va, sgnb, vb in itertools.product(
            signs, values, signs, values
        ):
            arr.append((sgna * va, sgnb * vb))
        z = xp.asarray(numpy.array(arr, dtype=dtype))
        return scp.special.kl_div(z[:, 0], z[:, 1])

    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupyimg_allclose(scipy_name="scp")
    def test_pseudo_huber(self, xp, scp, dtype):
        z = testing.shaped_random((10, 2), xp=numpy, dtype=dtype).tolist()
        z = xp.asarray(z + [[0, 0.5], [0.5, 0]], dtype=dtype)
        return scp.special.pseudo_huber(z[:, 0], z[:, 1])


# TODO: update/expand lpmv tests. The below is adapted from SciPy
@testing.gpu
@testing.with_requires("scipy")
class TestLegendreFunctions(unittest.TestCase):
    def test_lpmv_basic(self):
        scp = cupyimg.scipy
        lp = scp.special.lpmv(0, 2, 0.5)
        cupy.testing.assert_array_almost_equal(lp, -0.125, 7)
        lp = scp.special.lpmv(0, 40, 0.001)
        cupy.testing.assert_array_almost_equal(lp, 0.1252678976534484, 7)

        # XXX: this is outside the domain of the current implementation,
        #      so ensure it returns a NaN rather than a wrong answer.
        olderr = numpy.seterr(all="ignore")
        try:
            lp = scp.special.lpmv(-1, -1, 0.001)
        finally:
            numpy.seterr(**olderr)
        assert lp != 0 or cupy.isnan(lp)


def test_gammasgn_vs_cephes():
    vals = cupy.asarray([-4, -3.5, -2.3, 1, 4.2], cupy.float64)
    scp = cupyimg.scipy
    cupy.testing.assert_array_equal(
        scp.special.gammasgn(vals),
        numpy.sign(cephes.rgamma(cupy.asnumpy(vals))),
    )


@testing.gpu
@testing.with_requires("scipy")
class TestBasic(unittest.TestCase):
    @testing.for_dtypes(["e", "f", "d"])
    @numpy_cupyimg_allclose(scipy_name="scp")
    def test_gammasgn(self, xp, scp, dtype):
        vals = xp.asarray([-4, -3.5, -2.3, 1, 4.2], dtype=dtype)
        return scp.special.gammasgn(vals)
