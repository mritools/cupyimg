import unittest

import numpy

import cupy
from cupy import testing
import cupyimg.scipy.ndimage  # NOQA
from cupyimg.testing import numpy_cupyimg_allclose

# import cupyx.scipy.ndimage  # NOQA

try:
    import scipy.ndimage  # NOQA
except ImportError:
    pass


@testing.parameterize(
    *(
        testing.product(
            {
                "shape": [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
                "ksize": [3, 4],
                "mode": ["reflect"],
                "cval": [0.0],
                "origin": [0, 1, None],
                "adtype": [
                    numpy.int8,
                    numpy.int16,
                    numpy.int32,
                    numpy.float32,
                    numpy.float64,
                ],
                "wdtype": [None, numpy.int32, numpy.float64],
                "output": [None, numpy.int32, numpy.float64],
                "filter": ["convolve", "correlate"],
            }
        )
        + testing.product(
            {
                "shape": [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
                "ksize": [3, 4],
                "mode": ["constant"],
                "cval": [-1.0, 0.0, 1.0],
                "origin": [0],
                "adtype": [numpy.int32, numpy.float64],
                "wdtype": [None],
                "output": [None],
                "filter": ["convolve", "correlate"],
            }
        )
        + testing.product(
            {
                "shape": [(3, 4), (2, 3, 4), (1, 2, 3, 4)],
                "ksize": [3, 4],
                "mode": ["nearest", "mirror", "wrap"],
                "cval": [0.0],
                "origin": [0],
                "adtype": [numpy.int32, numpy.float64],
                "wdtype": [None],
                "output": [None],
                "filter": ["convolve", "correlate"],
            }
        )
    )
)
@testing.with_requires("scipy")
class TestConvolveAndCorrelate(unittest.TestCase):
    def _filter(self, xp, scp, a, w):
        filter = getattr(scp.ndimage, self.filter)
        if self.origin is None:
            origin = (-1, 1, -1, 1)[: a.ndim]
        else:
            origin = self.origin
        return filter(
            a,
            w,
            output=self.output,
            mode=self.mode,
            cval=self.cval,
            origin=origin,
        )

    @numpy_cupyimg_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_convolve_and_correlate(self, xp, scp):
        if self.adtype == self.wdtype or self.adtype == self.output:
            return xp.array(True)
        a = testing.shaped_random(self.shape, xp, self.adtype)
        if self.wdtype is None:
            wdtype = self.adtype
        else:
            wdtype = self.wdtype
        w = testing.shaped_random((self.ksize,) * a.ndim, xp, wdtype)
        return self._filter(xp, scp, a, w)


@testing.parameterize(
    *testing.product(
        {
            "ndim": [2, 3],
            "dtype": [numpy.int32, numpy.float64],
            "filter": ["convolve", "correlate"],
        }
    )
)
@testing.with_requires("scipy")
class TestConvolveAndCorrelateSpecialCases(unittest.TestCase):
    def _filter(self, scp, a, w, mode="reflect", origin=0):
        filter = getattr(scp.ndimage, self.filter)
        return filter(a, w, mode=mode, origin=origin)

    @numpy_cupyimg_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_weights_with_size_zero_dim(self, xp, scp):
        a = testing.shaped_random((3,) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((0,) + (3,) * self.ndim, xp, self.dtype)
        return self._filter(scp, a, w)

    def test_invalid_shape_weights(self):
        a = testing.shaped_random((3,) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3,) * (self.ndim - 1), cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyimg.scipy, a, w)
        w = testing.shaped_random(
            (0,) + (3,) * (self.ndim - 1), cupy, self.dtype
        )
        with self.assertRaises(RuntimeError):
            self._filter(cupyimg.scipy, a, w)

    def test_invalid_mode(self):
        a = testing.shaped_random((3,) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3,) * self.ndim, cupy, self.dtype)
        with self.assertRaises(RuntimeError):
            self._filter(cupyimg.scipy, a, w, mode="unknown")

    def test_invalid_origin(self):
        a = testing.shaped_random((3,) * self.ndim, cupy, self.dtype)
        w = testing.shaped_random((3,) * self.ndim, cupy, self.dtype)
        for origin in (-3, -2, 2, 3):
            if self.filter == "correlate" and origin == 2:
                continue
            if self.filter == "convolve" and origin == -2:
                continue
            with self.assertRaises(ValueError):
                self._filter(cupyimg.scipy, a, w, origin=origin)

    @numpy_cupyimg_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_noncontig_input(self, xp, scp):
        a = testing.shaped_random((16,) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((3,) * self.ndim, xp, self.dtype)
        return self._filter(scp, a[..., :4], w)

    @numpy_cupyimg_allclose(atol=1e-5, rtol=1e-5, scipy_name="scp")
    def test_noncontig_weights(self, xp, scp):
        a = testing.shaped_random((8,) * self.ndim, xp, self.dtype)
        w = testing.shaped_random((4,) * self.ndim, xp, self.dtype)
        return self._filter(scp, a, w[..., :2])
