import cupy as cp
import numpy as np
import pytest
from cupy.testing import assert_array_almost_equal
from numpy.testing import assert_almost_equal

from cupyimg.scipy import stats


class TestEntropy(object):
    def test_entropy_positive(self):
        # See ticket #497
        pk = cp.asarray([0.5, 0.2, 0.3])
        qk = cp.asarray([0.1, 0.25, 0.65])
        eself = stats.entropy(pk, pk)
        edouble = stats.entropy(pk, qk)
        assert 0.0 == eself
        assert edouble >= 0.0

    def test_entropy_base(self):
        pk = cp.ones(16, float)
        S = stats.entropy(pk, base=2.0)
        assert abs(S - 4.0) < 1.0e-5

        qk = cp.ones(16, float)
        qk[:8] = 2.0
        S = stats.entropy(pk, qk)
        S2 = stats.entropy(pk, qk, base=2.0)
        assert abs(S / S2 - np.log(2.0)) < 1.0e-5

    def test_entropy_zero(self):
        # Test for PR-479
        assert_almost_equal(
            stats.entropy(cp.asarray([0, 1, 2]).get()),
            0.63651416829481278,
            decimal=12,
        )

    def test_entropy_2d(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(
            stats.entropy(pk, qk), [0.1933259, 0.18609809]
        )

    def test_entropy_2d_zero(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.0, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(stats.entropy(pk, qk), [np.inf, 0.18609809])

        pk[0][0] = 0.0
        assert_array_almost_equal(
            stats.entropy(pk, qk), [0.17403988, 0.18609809]
        )

    def test_entropy_base_2d_nondefault_axis(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        assert_array_almost_equal(
            stats.entropy(pk, axis=1),
            cp.asarray([0.63651417, 0.63651417, 0.66156324]),
        )

    def test_entropy_2d_nondefault_axis(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(
            stats.entropy(pk, qk, axis=1),
            cp.asarray([0.231049, 0.231049, 0.127706]),
        )

    def test_entropy_raises_value_error(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.1, 0.2], [0.6, 0.3]])
        with pytest.raises(ValueError):
            stats.entropy(pk, qk)

    def test_base_entropy_with_axis_0_is_equal_to_default(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        assert_array_almost_equal(stats.entropy(pk, axis=0), stats.entropy(pk))

    def test_entropy_with_axis_0_is_equal_to_default(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(
            stats.entropy(pk, qk, axis=0), stats.entropy(pk, qk)
        )

    def test_base_entropy_transposed(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        assert_array_almost_equal(
            stats.entropy(pk.T).T, stats.entropy(pk, axis=1)
        )

    def test_entropy_transposed(self):
        pk = cp.asarray([[0.1, 0.2], [0.6, 0.3], [0.3, 0.5]])
        qk = cp.asarray([[0.2, 0.1], [0.3, 0.6], [0.5, 0.3]])
        assert_array_almost_equal(
            stats.entropy(pk.T, qk.T).T, stats.entropy(pk, qk, axis=1)
        )
