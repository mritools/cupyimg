# Code adapted from "upfirdn" python library with permission:
#
# Copyright (c) 2009, Motorola, Inc
#
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the name of Motorola nor the names of its contributors may be
# used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from itertools import product

import numpy as np
from numpy.testing import assert_equal
import cupy as cp
from cupy.testing import assert_allclose
import pytest
from pytest import raises as assert_raises

from fast_upfirdn import upfirdn_modes, upfirdn_out_len as _output_len


from cupyimg.scipy.signal import upfirdn
from scipy.signal import firwin, lfilter


def _pad_test(x, npre, npost, mode):
    # test array extension by convolving with an impulse padded with zeros
    h = cp.zeros((npre + npost + 1))
    h[npre] = 1
    return upfirdn(h, x, up=1, down=1, mode=mode)


# TODO: implement apply_along_axis for CuPy
#       keep this on CPU until apply_along_axis is available
def upfirdn_naive(x, h, up=1, down=1):
    """Naive upfirdn processing in Python

    Note: arg order (x, h) differs to facilitate apply_along_axis use.
    """
    h = np.asarray(h)
    out = np.zeros(len(x) * up, x.dtype)
    out[::up] = x
    out = np.convolve(h, out)[::down][: _output_len(len(h), len(x), up, down)]
    return out


class UpFIRDnCase(object):
    """Test _UpFIRDn object"""

    def __init__(self, up, down, h, x_dtype):
        self.up = up
        self.down = down
        self.h = cp.atleast_1d(h)
        self.x_dtype = x_dtype
        self.rng = cp.random.RandomState(17)

    def __call__(self):
        # tiny signal
        self.scrub(cp.ones(1, self.x_dtype))
        # ones
        self.scrub(cp.ones(10, self.x_dtype))  # ones
        # randn
        x = self.rng.randn(10).astype(self.x_dtype)
        if self.x_dtype in (cp.complex64, cp.complex128):
            x += 1j * self.rng.randn(10)
        self.scrub(x)
        # ramp
        self.scrub(cp.arange(10).astype(self.x_dtype))
        # 3D, random
        size = (2, 3, 5)
        x = self.rng.randn(*size).astype(self.x_dtype)
        if self.x_dtype in (cp.complex64, cp.complex128):
            x += 1j * self.rng.randn(*size)
        for axis in range(len(size)):
            self.scrub(x, axis=axis)
        x = x[:, ::2, 1::3].T
        for axis in range(len(size)):
            self.scrub(x, axis=axis)

    def scrub(self, x, axis=-1):
        yr = np.apply_along_axis(
            upfirdn_naive, axis, x.get(), self.h.get(), self.up, self.down
        )
        yr = cp.asarray(yr)
        y = upfirdn(self.h, x, self.up, self.down, axis=axis)
        dtypes = (self.h.dtype, x.dtype)
        if all(d == cp.complex64 for d in dtypes):
            assert_equal(y.dtype, cp.complex64)
        elif cp.complex64 in dtypes and cp.float32 in dtypes:
            assert_equal(y.dtype, cp.complex64)
        elif all(d == cp.float32 for d in dtypes):
            assert_equal(y.dtype, cp.float32)
        elif cp.complex128 in dtypes or cp.complex64 in dtypes:
            assert_equal(y.dtype, cp.complex128)
        else:
            assert_equal(y.dtype, cp.float64)
        assert_allclose(yr, y)


class TestUpfirdn(object):
    def test_valid_input(self):
        assert_raises(ValueError, upfirdn, [1], [1], 1, 0)  # up or down < 1
        assert_raises(ValueError, upfirdn, [], [1], 1, 1)  # h.ndim != 1
        assert_raises(ValueError, upfirdn, [[1]], [1], 1, 1)

    def test_vs_lfilter(self):
        # Check that up=1.0 gives same answer as lfilter + slicing
        random_state = cp.random.RandomState(17)
        try_types = (int, cp.float32, cp.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (cp.complex64, cp.complex128):
                x += 1j * random_state.randn(size)

            tol = cp.finfo(cp.float32).eps * 100

            for down in down_factors:
                h = firwin(31, 1.0 / down, window="hamming")
                yl = cp.asarray(lfilter(h, 1.0, x.get())[::down])
                h = cp.asarray(h)
                y = upfirdn(h, x, up=1, down=down)
                assert_allclose(yl, y[: yl.size], atol=tol, rtol=tol)

    def test_vs_naive(self):
        tests = []
        try_types = (int, cp.float32, cp.complex64, float, complex)

        # Simple combinations of factors
        for x_dtype, h in product(try_types, (1.0, 1j)):
            tests.append(UpFIRDnCase(1, 1, h, x_dtype))
            tests.append(UpFIRDnCase(2, 2, h, x_dtype))
            tests.append(UpFIRDnCase(3, 2, h, x_dtype))
            tests.append(UpFIRDnCase(2, 3, h, x_dtype))

        # mixture of big, small, and both directions (net up and net down)
        # use all combinations of data and filter dtypes
        factors = (100, 10)  # up/down factors
        cases = product(factors, factors, try_types, try_types)
        for case in cases:
            tests += self._random_factors(*case)

        for test in tests:
            test()

    def _random_factors(self, p_max, q_max, h_dtype, x_dtype):
        n_rep = 3
        longest_h = 25
        random_state = np.random.RandomState(17)
        tests = []

        for _ in range(n_rep):
            # Randomize the up/down factors somewhat
            p_add = q_max if p_max > q_max else 1
            q_add = p_max if q_max > p_max else 1
            p = random_state.randint(p_max) + p_add
            q = random_state.randint(q_max) + q_add

            # Generate random FIR coefficients
            len_h = random_state.randint(longest_h) + 1
            h = cp.atleast_1d(random_state.randint(len_h))
            h = h.astype(h_dtype)
            if h_dtype == complex:
                h += 1j * cp.atleast_1d(random_state.randint(len_h))

            tests.append(UpFIRDnCase(p, q, h, x_dtype))

        return tests

    @pytest.mark.parametrize("mode", upfirdn_modes)
    def test_extensions(self, mode):
        """Test vs. manually computed results for modes not in numpy's pad."""
        x = cp.array([1, 2, 3, 1], dtype=float)
        npre, npost = 6, 6

        y = _pad_test(x, npre=npre, npost=npost, mode=mode)
        if mode == "antisymmetric":
            y_expected = cp.asarray(
                [3, 1, -1, -3, -2, -1, 1, 2, 3, 1, -1, -3, -2, -1, 1, 2]
            )
        elif mode == "antireflect":
            y_expected = cp.asarray(
                [1, 2, 3, 1, -1, 0, 1, 2, 3, 1, -1, 0, 1, 2, 3, 1]
            )
        elif mode == "smooth":
            y_expected = cp.asarray(
                [-5, -4, -3, -2, -1, 0, 1, 2, 3, 1, -1, -3, -5, -7, -9, -11]
            )
        elif mode == "line":
            lin_slope = (x[-1] - x[0]) / (len(x) - 1)
            left = x[0] + cp.arange(-npre, 0, 1) * lin_slope
            right = x[-1] + cp.arange(1, npost + 1) * lin_slope
            y_expected = cp.concatenate((left, x, right))
        else:
            y_expected = cp.pad(x, (npre, npost), mode=mode)
        assert_allclose(y, y_expected)

    @pytest.mark.parametrize(
        "size, h_len, mode, dtype",
        product(
            [8],
            [4, 5, 26],  # include cases with h_len > 2*size
            upfirdn_modes,
            [cp.float32, cp.float64, cp.complex64, cp.complex128],
        ),
    )
    def test_modes(self, size, h_len, mode, dtype):
        random_state = cp.random.RandomState(5)
        x = random_state.randn(size).astype(dtype)
        if dtype in (cp.complex64, cp.complex128):
            x += 1j * random_state.randn(size)
        h = cp.arange(1, 1 + h_len, dtype=x.real.dtype)

        y = upfirdn(h, x, up=1, down=1, mode=mode)
        # expected result: pad the input, filter with zero padding, then crop
        npad = h_len - 1
        if mode in ["antisymmetric", "antireflect", "smooth", "line"]:
            # use _pad_test test function for modes not supported by cp.pad.
            xpad = _pad_test(x, npre=npad, npost=npad, mode=mode)
        else:
            xpad = cp.pad(x, npad, mode=mode)
        ypad = upfirdn(h, xpad, up=1, down=1, mode="constant")
        y_expected = ypad[npad:-npad]

        atol = rtol = cp.finfo(dtype).eps * 1e2
        assert_allclose(y, y_expected, atol=atol, rtol=rtol)
