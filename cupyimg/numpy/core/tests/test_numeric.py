"""Test cases adapted directly from NumPy."""

import cupy as cp
from cupy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from cupyimg.numpy import convolve, correlate


class TestCorrelate(object):
    def _setup(self, dt):
        self.x = cp.array([1, 2, 3, 4, 5], dtype=dt)
        self.xs = cp.arange(1, 20)[::3]
        self.y = cp.array([-1, -2, -3], dtype=dt)
        self.z1 = cp.array(
            [-3.0, -8.0, -14.0, -20.0, -26.0, -14.0, -5.0], dtype=dt
        )
        self.z1_4 = cp.array([-2.0, -5.0, -8.0, -11.0, -14.0, -5.0], dtype=dt)
        self.z1r = cp.array(
            [-15.0, -22.0, -22.0, -16.0, -10.0, -4.0, -1.0], dtype=dt
        )
        self.z2 = cp.array(
            [-5.0, -14.0, -26.0, -20.0, -14.0, -8.0, -3.0], dtype=dt
        )
        self.z2r = cp.array(
            [-1.0, -4.0, -10.0, -16.0, -22.0, -22.0, -15.0], dtype=dt
        )
        self.zs = cp.array(
            [-3.0, -14.0, -30.0, -48.0, -66.0, -84.0, -102.0, -54.0, -19.0],
            dtype=dt,
        )

    @pytest.mark.parametrize("dtype", [float, cp.float32])
    def test_float(self, dtype):
        self._setup(dtype)
        z = correlate(self.x, self.y, "full")
        assert_array_almost_equal(z, self.z1)
        z = correlate(self.x, self.y[:-1], "full")
        assert_array_almost_equal(z, self.z1_4)
        z = correlate(self.y, self.x, "full")
        assert_array_almost_equal(z, self.z2)
        z = correlate(self.x[::-1], self.y, "full")
        assert_array_almost_equal(z, self.z1r)
        z = correlate(self.y, self.x[::-1], "full")
        assert_array_almost_equal(z, self.z2r)
        z = correlate(self.xs, self.y, "full")
        assert_array_almost_equal(z, self.zs)

    @pytest.mark.skip(reason="object case not supported")
    def test_object(self):
        from decimal import Decimal

        self._setup(Decimal)
        z = correlate(self.x, self.y, "full")
        assert_array_almost_equal(z, self.z1)
        z = correlate(self.y, self.x, "full")
        assert_array_almost_equal(z, self.z2)

    def test_no_overwrite(self):
        d = cp.ones(100)
        k = cp.ones(3)
        correlate(d, k)
        assert_array_equal(d, cp.ones(100))
        assert_array_equal(k, cp.ones(3))

    def test_complex(self):
        x = cp.array([1, 2, 3, 4 + 1j], dtype=complex)
        y = cp.array([-1, -2j, 3 + 1j], dtype=complex)
        r_z = cp.array(
            [3 - 1j, 6, 8 + 1j, 11 + 5j, -5 + 8j, -4 - 1j], dtype=complex
        )
        r_z = r_z[::-1].conj()
        z = correlate(y, x, mode="full")
        assert_array_almost_equal(z, r_z)


class TestConvolve(object):
    def test_object(self):
        d = [1.0] * 100
        k = [1.0] * 3
        assert_array_almost_equal(convolve(d, k)[2:-2], cp.full(98, 3))

    def test_no_overwrite(self):
        d = cp.ones(100)
        k = cp.ones(3)
        convolve(d, k)
        assert_array_equal(d, cp.ones(100))
        assert_array_equal(k, cp.ones(3))
