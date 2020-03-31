# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

# import sys
# import platform

import cupy
import numpy
import pytest
from cupy import fft
from cupy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_allclose,
)
from numpy.testing import assert_equal
from pytest import raises as assert_raises

# import pytest
# from pytest import raises as assert_raises
# from scipy._lib._numpy_compat import suppress_warnings
from cupyimg.scipy import ndimage


eps = 1e-12
brute_force_implemented = False  # used to skip tests for missing feature


def sumsq(a, b):
    return math.sqrt(((a - b) ** 2).sum())


class TestNdimage:
    def setup_method(self):
        # list of numarray data types
        self.integer_types = [
            numpy.int8,
            numpy.uint8,
            numpy.int16,
            numpy.uint16,
            numpy.int32,
            numpy.uint32,
            numpy.int64,
            numpy.uint64,
        ]

        self.float_types = [numpy.float32, numpy.float64]

        self.types = self.integer_types + self.float_types

        # list of boundary modes:
        self.modes = ["nearest", "wrap", "reflect", "mirror", "constant"]

    def test_correlate01(self):
        array = cupy.asarray([1, 2])
        weights = cupy.asarray([2])
        expected = [2, 4]

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate02(self):
        array = cupy.asarray([1, 2, 3])
        kernel = cupy.asarray([1])

        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)

        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate03(self):
        array = cupy.asarray([1])
        weights = cupy.asarray([1, 1])
        expected = [2]

        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)

        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate04(self):
        array = cupy.asarray([1, 2])
        tcor = [2, 3]
        tcov = [3, 4]
        weights = cupy.asarray([1, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate05(self):
        array = cupy.asarray([1, 2, 3])
        tcor = [2, 3, 5]
        tcov = [3, 5, 6]
        kernel = cupy.asarray([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(tcov, output)

    def test_correlate06(self):
        array = cupy.asarray([1, 2, 3])
        tcor = [9, 14, 17]
        tcov = [7, 10, 15]
        weights = cupy.asarray([1, 2, 3])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate07(self):
        array = cupy.asarray([1, 2, 3])
        expected = [5, 8, 11]
        weights = cupy.asarray([1, 2, 1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, expected)

    def test_correlate08(self):
        array = cupy.asarray([1, 2, 3])
        tcor = [1, 2, 5]
        tcov = [3, 6, 7]
        weights = cupy.asarray([1, 2, -1])
        output = ndimage.correlate(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve(array, weights)
        assert_array_almost_equal(output, tcov)
        output = ndimage.correlate1d(array, weights)
        assert_array_almost_equal(output, tcor)
        output = ndimage.convolve1d(array, weights)
        assert_array_almost_equal(output, tcov)

    def test_correlate09(self):
        array = cupy.asarray([])
        kernel = cupy.asarray([1, 1])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.correlate1d(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve1d(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate10(self):
        array = cupy.asarray([[]])
        kernel = cupy.asarray([[1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal(array, output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal(array, output)

    def test_correlate11(self):
        array = cupy.asarray([[1, 2, 3], [4, 5, 6]])
        kernel = cupy.asarray([[1, 1], [1, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[12, 16, 18], [18, 22, 24]], output)

    def test_correlate12(self):
        array = cupy.asarray([[1, 2, 3], [4, 5, 6]])
        kernel = cupy.asarray([[1, 0], [0, 1]])
        output = ndimage.correlate(array, kernel)
        assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
        output = ndimage.convolve(array, kernel)
        assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)

    def test_correlate13(self):
        kernel = cupy.asarray([[1, 0], [0, 1]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            for type2 in self.types:
                output = ndimage.correlate(array, kernel, output=type2)
                assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
                assert_equal(output.dtype.type, type2)

                output = ndimage.convolve(array, kernel, output=type2)
                assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
                assert_equal(output.dtype.type, type2)

    def test_correlate14(self):
        kernel = cupy.asarray([[1, 0], [0, 1]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros(array.shape, type2)
                ndimage.correlate(array, kernel, output=output)
                assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
                assert_equal(output.dtype.type, type2)

                ndimage.convolve(array, kernel, output=output)
                assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
                assert_equal(output.dtype.type, type2)

    def test_correlate15(self):
        kernel = cupy.asarray([[1, 0], [0, 1]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            output = ndimage.correlate(array, kernel, output=numpy.float32)
            assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
            assert_equal(output.dtype.type, numpy.float32)

            output = ndimage.convolve(array, kernel, output=numpy.float32)
            assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
            assert_equal(output.dtype.type, numpy.float32)

    def test_correlate16(self):
        kernel = cupy.asarray([[0.5, 0], [0, 0.5]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            output = ndimage.correlate(array, kernel, output=numpy.float32)
            assert_array_almost_equal([[1, 1.5, 2.5], [2.5, 3, 4]], output)
            assert_equal(output.dtype.type, numpy.float32)

            output = ndimage.convolve(array, kernel, output=numpy.float32)
            assert_array_almost_equal([[3, 4, 4.5], [4.5, 5.5, 6]], output)
            assert_equal(output.dtype.type, numpy.float32)

    def test_correlate17(self):
        array = cupy.asarray([1, 2, 3])
        tcor = [3, 5, 6]
        tcov = [2, 3, 5]
        kernel = cupy.asarray([1, 1])
        output = ndimage.correlate(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)
        output = ndimage.correlate1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcor, output)
        output = ndimage.convolve1d(array, kernel, origin=-1)
        assert_array_almost_equal(tcov, output)

    def test_correlate18(self):
        kernel = cupy.asarray([[1, 0], [0, 1]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            output = ndimage.correlate(
                array, kernel, output=numpy.float32, mode="nearest", origin=-1
            )
            assert_array_almost_equal([[6, 8, 9], [9, 11, 12]], output)
            assert_equal(output.dtype.type, numpy.float32)

            output = ndimage.convolve(
                array, kernel, output=numpy.float32, mode="nearest", origin=-1
            )
            assert_array_almost_equal([[2, 3, 5], [5, 6, 8]], output)
            assert_equal(output.dtype.type, numpy.float32)

    def test_correlate19(self):
        kernel = cupy.asarray([[1, 0], [0, 1]])
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [4, 5, 6]], type1)
            output = ndimage.correlate(
                array,
                kernel,
                output=numpy.float32,
                mode="nearest",
                origin=[-1, 0],
            )
            assert_array_almost_equal([[5, 6, 8], [8, 9, 11]], output)
            assert_equal(output.dtype.type, numpy.float32)

            output = ndimage.convolve(
                array,
                kernel,
                output=numpy.float32,
                mode="nearest",
                origin=[-1, 0],
            )
            assert_array_almost_equal([[3, 5, 6], [6, 8, 9]], output)
            assert_equal(output.dtype.type, numpy.float32)

    def test_correlate20(self):
        weights = cupy.asarray([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [2, 4, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros((2, 3), type2)
                ndimage.correlate1d(array, weights, axis=0, output=output)
                assert_array_almost_equal(output, expected)
                ndimage.convolve1d(array, weights, axis=0, output=output)
                assert_array_almost_equal(output, expected)

    def test_correlate21(self):
        array = cupy.asarray([[1, 2, 3], [2, 4, 6]])
        expected = [[5, 10, 15], [7, 14, 21]]
        weights = cupy.asarray([1, 2, 1])
        output = ndimage.correlate1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)
        output = ndimage.convolve1d(array, weights, axis=0)
        assert_array_almost_equal(output, expected)

    def test_correlate22(self):
        weights = cupy.asarray([1, 2, 1])
        expected = [[6, 12, 18], [6, 12, 18]]
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [2, 4, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros((2, 3), type2)
                ndimage.correlate1d(
                    array, weights, axis=0, mode="wrap", output=output
                )
                assert_array_almost_equal(output, expected)
                ndimage.convolve1d(
                    array, weights, axis=0, mode="wrap", output=output
                )
                assert_array_almost_equal(output, expected)

    def test_correlate23(self):
        weights = cupy.asarray([1, 2, 1])
        expected = [[5, 10, 15], [7, 14, 21]]
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [2, 4, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros((2, 3), type2)
                ndimage.correlate1d(
                    array, weights, axis=0, mode="nearest", output=output
                )
                assert_array_almost_equal(output, expected)
                ndimage.convolve1d(
                    array, weights, axis=0, mode="nearest", output=output
                )
                assert_array_almost_equal(output, expected)

    def test_correlate24(self):
        weights = cupy.asarray([1, 2, 1])
        tcor = [[7, 14, 21], [8, 16, 24]]
        tcov = [[4, 8, 12], [5, 10, 15]]
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [2, 4, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros((2, 3), type2)
                ndimage.correlate1d(
                    array,
                    weights,
                    axis=0,
                    mode="nearest",
                    output=output,
                    origin=-1,
                )
                assert_array_almost_equal(output, tcor)
                ndimage.convolve1d(
                    array,
                    weights,
                    axis=0,
                    mode="nearest",
                    output=output,
                    origin=-1,
                )
                assert_array_almost_equal(output, tcov)

    def test_correlate25(self):
        weights = cupy.asarray([1, 2, 1])
        tcor = [[4, 8, 12], [5, 10, 15]]
        tcov = [[7, 14, 21], [8, 16, 24]]
        for type1 in self.types:
            array = cupy.asarray([[1, 2, 3], [2, 4, 6]], type1)
            for type2 in self.types:
                output = cupy.zeros((2, 3), type2)
                ndimage.correlate1d(
                    array,
                    weights,
                    axis=0,
                    mode="nearest",
                    output=output,
                    origin=1,
                )
                assert_array_almost_equal(output, tcor)
                ndimage.convolve1d(
                    array,
                    weights,
                    axis=0,
                    mode="nearest",
                    output=output,
                    origin=1,
                )
                assert_array_almost_equal(output, tcov)

    def test_gauss01(self):
        input = cupy.asarray([[1, 2, 3], [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 0)
        assert_array_almost_equal(output, input)

    def test_gauss02(self):
        input = cupy.asarray([[1, 2, 3], [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 1.0)
        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

    def test_gauss03(self):
        # single precision data"
        input = cupy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output = ndimage.gaussian_filter(input, [1.0, 1.0])

        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

        # input.sum() is 49995000.0.  With single precision floats, we can't
        # expect more than 8 digits of accuracy, so use decimal=0 in this test.
        assert_allclose(output.sum(dtype="d"), input.sum(dtype="d"), rtol=1e-7)
        assert sumsq(input, output) > 1.0

    def test_gauss04(self):
        input = cupy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert sumsq(input, output) > 1.0

    def test_gauss05(self):
        input = cupy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(
            input, [1.0, 1.0], order=1, output=otype
        )
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert sumsq(input, output) > 1.0

    def test_gauss06(self):
        input = cupy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0], output=otype)
        output2 = ndimage.gaussian_filter(input, 1.0, output=otype)
        assert_array_almost_equal(output1, output2)

    def test_prewitt01(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
            t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
            output = ndimage.prewitt(array, 0)
            assert_array_almost_equal(t, output)

    def test_prewitt02(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
            t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 1)
            output = cupy.zeros(array.shape, type_)
            ndimage.prewitt(array, 0, output)
            assert_array_almost_equal(t, output)

    def test_prewitt03(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
            t = ndimage.correlate1d(t, [1.0, 1.0, 1.0], 0)
            output = ndimage.prewitt(array, 1)
            assert_array_almost_equal(t, output)

    def test_prewitt04(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.prewitt(array, -1)
            output = ndimage.prewitt(array, 1)
            assert_array_almost_equal(t, output)

    def test_sobel01(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
            t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
            output = ndimage.sobel(array, 0)
            assert_array_almost_equal(t, output)

    def test_sobel02(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 0)
            t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 1)
            output = cupy.zeros(array.shape, type_)
            ndimage.sobel(array, 0, output)
            assert_array_almost_equal(t, output)

    def test_sobel03(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.correlate1d(array, [-1.0, 0.0, 1.0], 1)
            t = ndimage.correlate1d(t, [1.0, 2.0, 1.0], 0)
            output = cupy.zeros(array.shape, type_)
            output = ndimage.sobel(array, 1)
            assert_array_almost_equal(t, output)

    def test_sobel04(self):
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            t = ndimage.sobel(array, -1)
            output = ndimage.sobel(array, 1)
            assert_array_almost_equal(t, output)

    def test_laplace01(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
            tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
            output = ndimage.laplace(array)
            assert_array_almost_equal(tmp1 + tmp2, output)

    def test_laplace02(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.correlate1d(array, [1, -2, 1], 0)
            tmp2 = ndimage.correlate1d(array, [1, -2, 1], 1)
            output = cupy.zeros(array.shape, type_)
            ndimage.laplace(array, output=output)
            assert_array_almost_equal(tmp1 + tmp2, output)

    def test_gaussian_laplace01(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
            tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
            output = ndimage.gaussian_laplace(array, 1.0)
            assert_array_almost_equal(tmp1 + tmp2, output)

    def test_gaussian_laplace02(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.gaussian_filter(array, 1.0, [2, 0])
            tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 2])
            output = cupy.zeros(array.shape, type_)
            ndimage.gaussian_laplace(array, 1.0, output)
            assert_array_almost_equal(tmp1 + tmp2, output)

    def test_generic_laplace01(self):
        def derivative2(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = cupy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 2
            return ndimage.gaussian_filter(
                input, sigma, order, output, mode, cval
            )

        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = cupy.zeros(array.shape, type_)
            tmp = ndimage.generic_laplace(
                array,
                derivative2,
                extra_arguments=(1.0,),
                extra_keywords={"b": 2.0},
            )
            ndimage.gaussian_laplace(array, 1.0, output)
            assert_array_almost_equal(tmp, output)

    def test_gaussian_gradient_magnitude01(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
            tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
            output = ndimage.gaussian_gradient_magnitude(array, 1.0)
            expected = tmp1 * tmp1 + tmp2 * tmp2
            expected = numpy.sqrt(expected).astype(type_)
            assert_array_almost_equal(expected, output)

    def test_gaussian_gradient_magnitude02(self):
        for type_ in [numpy.int32, numpy.float32, numpy.float64]:
            array = (
                cupy.asarray(
                    [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
                )
                * 100
            )
            tmp1 = ndimage.gaussian_filter(array, 1.0, [1, 0])
            tmp2 = ndimage.gaussian_filter(array, 1.0, [0, 1])
            output = cupy.zeros(array.shape, type_)
            ndimage.gaussian_gradient_magnitude(array, 1.0, output)
            expected = tmp1 * tmp1 + tmp2 * tmp2
            expected = numpy.sqrt(expected).astype(type_)
            assert_array_almost_equal(expected, output)

    def test_generic_gradient_magnitude01(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], numpy.float64
        )

        def derivative(input, axis, output, mode, cval, a, b):
            sigma = [a, b / 2.0]
            input = cupy.asarray(input)
            order = [0] * input.ndim
            order[axis] = 1
            return ndimage.gaussian_filter(
                input, sigma, order, output, mode, cval
            )

        tmp1 = ndimage.gaussian_gradient_magnitude(array, 1.0)
        tmp2 = ndimage.generic_gradient_magnitude(
            array, derivative, extra_arguments=(1.0,), extra_keywords={"b": 2.0}
        )
        assert_array_almost_equal(tmp1, tmp2)

    def test_uniform01(self):
        array = cupy.asarray([2, 4, 6])
        size = 2
        output = ndimage.uniform_filter1d(array, size, origin=-1)
        assert_array_almost_equal([3, 5, 6], output)

    def test_uniform02(self):
        array = cupy.asarray([1, 2, 3])
        filter_shape = [0]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform03(self):
        array = cupy.asarray([1, 2, 3])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal(array, output)

    def test_uniform04(self):
        array = cupy.asarray([2, 4, 6])
        filter_shape = [2]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 5], output)

    def test_uniform05(self):
        array = cupy.asarray([])
        filter_shape = [1]
        output = ndimage.uniform_filter(array, filter_shape)
        assert_array_almost_equal([], output)

    def test_uniform06(self):
        filter_shape = [2, 2]
        for type1 in self.types:
            array = cupy.asarray([[4, 8, 12], [16, 20, 24]], type1)
            for type2 in self.types:
                output = ndimage.uniform_filter(
                    array, filter_shape, output=type2
                )
                assert_array_almost_equal([[4, 6, 10], [10, 12, 16]], output)
                assert_equal(output.dtype.type, type2)

    def test_minimum_filter01(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        filter_shape = [2]
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter02(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        filter_shape = [3]
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([1, 1, 2, 3, 4], output)

    def test_minimum_filter03(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        filter_shape = [2]
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([3, 2, 2, 1, 1], output)

    def test_minimum_filter04(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        filter_shape = [3]
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal([2, 2, 1, 1, 1], output)

    def test_minimum_filter05(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        filter_shape = [2, 3]
        output = ndimage.minimum_filter(array, filter_shape)
        assert_array_almost_equal(
            [[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], output
        )

    def test_minimum_filter06(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal(
            [[2, 2, 1, 1, 1], [2, 2, 1, 1, 1], [5, 3, 3, 1, 1]], output
        )

    def test_minimum_filter07(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(array, footprint=footprint)
        assert_array_almost_equal(
            [[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]], output
        )

    def test_minimum_filter08(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal(
            [[3, 1, 3, 1, 1], [5, 3, 3, 1, 1], [3, 3, 1, 1, 1]], output
        )

    def test_minimum_filter09(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.minimum_filter(
            array, footprint=footprint, origin=[-1, 0]
        )
        assert_array_almost_equal(
            [[2, 3, 1, 3, 1], [5, 5, 3, 3, 1], [5, 3, 3, 1, 1]], output
        )

    def test_maximum_filter01(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        filter_shape = [2]
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([1, 2, 3, 4, 5], output)

    def test_maximum_filter02(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        filter_shape = [3]
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([2, 3, 4, 5, 5], output)

    def test_maximum_filter03(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        filter_shape = [2]
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_maximum_filter04(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        filter_shape = [3]
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal([3, 5, 5, 5, 4], output)

    def test_maximum_filter05(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        filter_shape = [2, 3]
        output = ndimage.maximum_filter(array, filter_shape)
        assert_array_almost_equal(
            [[3, 5, 5, 5, 4], [7, 9, 9, 9, 5], [8, 9, 9, 9, 7]], output
        )

    def test_maximum_filter06(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal(
            [[3, 5, 5, 5, 4], [7, 9, 9, 9, 5], [8, 9, 9, 9, 7]], output
        )

    def test_maximum_filter07(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(array, footprint=footprint)
        assert_array_almost_equal(
            [[3, 5, 5, 5, 4], [7, 7, 9, 9, 5], [7, 9, 8, 9, 7]], output
        )

    def test_maximum_filter08(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(array, footprint=footprint, origin=-1)
        assert_array_almost_equal(
            [[7, 9, 9, 5, 5], [9, 8, 9, 7, 5], [8, 8, 7, 7, 7]], output
        )

    def test_maximum_filter09(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.maximum_filter(
            array, footprint=footprint, origin=[-1, 0]
        )
        assert_array_almost_equal(
            [[7, 7, 9, 9, 5], [7, 9, 8, 9, 7], [8, 8, 8, 7, 7]], output
        )

    def test_rank01(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, 2)
        assert_array_almost_equal(array, output)

    def test_rank02(self):
        array = cupy.asarray([1, 2, 3, 4, 5])
        output = ndimage.rank_filter(array, 1, size=[3])
        assert_array_almost_equal(array, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(array, output)
        output = ndimage.median_filter(array, (3,))
        assert_array_almost_equal(array, output)

    def test_rank03(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        output = ndimage.rank_filter(array, 1, size=[2])
        assert_array_almost_equal([3, 3, 5, 5, 4], output)
        output = ndimage.percentile_filter(array, 100, size=2)
        assert_array_almost_equal([3, 3, 5, 5, 4], output)

    def test_rank04(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        expected = cupy.asarray([3, 3, 2, 4, 4])
        output = ndimage.rank_filter(array, 1, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 50, size=3)
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank05(self):
        array = cupy.asarray([3, 2, 5, 1, 4])
        expected = cupy.asarray([3, 3, 2, 4, 4])
        output = ndimage.rank_filter(array, -2, size=3)
        assert_array_almost_equal(expected, output)

    def test_rank06(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]]
        )
        expected = cupy.asarray(
            [[2, 2, 1, 1, 1], [3, 3, 2, 1, 1], [5, 5, 3, 3, 1]]
        )
        output = ndimage.rank_filter(array, 1, size=[2, 3])
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 17, size=(2, 3))
        assert_array_almost_equal(expected, output)

    def test_rank07(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]]
        )
        expected = cupy.asarray(
            [[3, 5, 5, 5, 4], [5, 5, 7, 5, 4], [6, 8, 8, 7, 5]]
        )
        output = ndimage.rank_filter(array, -2, size=[2, 3])
        assert_array_almost_equal(expected, output)

    def test_rank08(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]]
        )
        expected = cupy.asarray(
            [[3, 3, 2, 4, 4], [5, 5, 5, 4, 4], [5, 6, 7, 5, 5]]
        )
        output = ndimage.percentile_filter(array, 50.0, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.rank_filter(array, 3, size=(2, 3))
        assert_array_almost_equal(expected, output)
        output = ndimage.median_filter(array, size=(2, 3))
        assert_array_almost_equal(expected, output)

    def test_rank09(self):
        expected = cupy.asarray(
            [[3, 3, 2, 4, 4], [3, 5, 2, 5, 1], [5, 5, 8, 3, 5]]
        )
        footprint = cupy.asarray([[1, 0, 1], [0, 1, 0]])
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = ndimage.rank_filter(array, 1, footprint=footprint)
            assert_array_almost_equal(expected, output)
            output = ndimage.percentile_filter(array, 35, footprint=footprint)
            assert_array_almost_equal(expected, output)

    def test_rank10(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        expected = cupy.asarray(
            [[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.rank_filter(array, 0, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 0.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank11(self):
        array = cupy.asarray(
            [[3, 2, 5, 1, 4], [7, 6, 9, 3, 5], [5, 8, 3, 7, 1]]
        )
        expected = cupy.asarray(
            [[3, 5, 5, 5, 4], [7, 7, 9, 9, 5], [7, 9, 8, 9, 7]]
        )
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.rank_filter(array, -1, footprint=footprint)
        assert_array_almost_equal(expected, output)
        output = ndimage.percentile_filter(array, 100.0, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_rank12(self):
        expected = cupy.asarray(
            [[3, 3, 2, 4, 4], [3, 5, 2, 5, 1], [5, 5, 8, 3, 5]]
        )
        footprint = cupy.asarray([[1, 0, 1], [0, 1, 0]])
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = ndimage.rank_filter(array, 1, footprint=footprint)
            assert_array_almost_equal(expected, output)
            output = ndimage.percentile_filter(array, 50.0, footprint=footprint)
            assert_array_almost_equal(expected, output)
            output = ndimage.median_filter(array, footprint=footprint)
            assert_array_almost_equal(expected, output)

    def test_rank13(self):
        expected = cupy.asarray(
            [[5, 2, 5, 1, 1], [5, 8, 3, 5, 5], [6, 6, 5, 5, 5]]
        )
        footprint = cupy.asarray([[1, 0, 1], [0, 1, 0]])
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = ndimage.rank_filter(
                array, 1, footprint=footprint, origin=-1
            )
            assert_array_almost_equal(expected, output)

    def test_rank14(self):
        expected = cupy.asarray(
            [[3, 5, 2, 5, 1], [5, 5, 8, 3, 5], [5, 6, 6, 5, 5]]
        )
        footprint = cupy.asarray([[1, 0, 1], [0, 1, 0]])
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = ndimage.rank_filter(
                array, 1, footprint=footprint, origin=[-1, 0]
            )
            assert_array_almost_equal(expected, output)

    def test_rank15(self):
        "rank filter 15"
        expected = cupy.asarray(
            [[2, 3, 1, 4, 1], [5, 3, 7, 1, 1], [5, 5, 3, 3, 3]]
        )
        footprint = cupy.asarray([[1, 0, 1], [0, 1, 0]])
        for type_ in self.types:
            array = cupy.asarray(
                [[3, 2, 5, 1, 4], [5, 8, 3, 7, 1], [5, 6, 9, 3, 5]], type_
            )
            output = ndimage.rank_filter(
                array, 0, footprint=footprint, origin=[-1, 0]
            )
            assert_array_almost_equal(expected, output)

    # def test_generic_filter1d01(self):
    #     weights = cupy.asarray([1.1, 2.2, 3.3])

    #     def _filter_func(input, output, fltr, total):
    #         fltr = fltr / total
    #         for ii in range(input.shape[0] - 2):
    #             output[ii] = input[ii] * fltr[0]
    #             output[ii] += input[ii + 1] * fltr[1]
    #             output[ii] += input[ii + 2] * fltr[2]
    #     for type_ in self.types:
    #         a = cupy.arange(12, dtype=type_)
    #         a.shape = (3, 4)
    #         r1 = ndimage.correlate1d(a, weights / weights.sum(), 0, origin=-1)
    #         r2 = ndimage.generic_filter1d(
    #             a, _filter_func, 3, axis=0, origin=-1,
    #             extra_arguments=(weights,),
    #             extra_keywords={'total': weights.sum()})
    #         assert_array_almost_equal(r1, r2)

    # def test_generic_filter01(self):
    #     filter_ = cupy.asarray([[1.0, 2.0], [3.0, 4.0]])
    #     footprint = cupy.asarray([[1, 0], [0, 1]])
    #     cf = cupy.asarray([1., 4.])

    #     def _filter_func(buffer, weights, total=1.0):
    #         weights = cf / total
    #         return (buffer * weights).sum()
    #     for type_ in self.types:
    #         a = cupy.arange(12, dtype=type_)
    #         a.shape = (3, 4)
    #         r1 = ndimage.correlate(a, filter_ * footprint)
    #         if type_ in self.float_types:
    #             r1 /= 5
    #         else:
    #             r1 //= 5
    #         r2 = ndimage.generic_filter(
    #             a, _filter_func, footprint=footprint, extra_arguments=(cf,),
    #             extra_keywords={'total': cf.sum()})
    #         assert_array_almost_equal(r1, r2)

    def test_extend01(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([1, 0])
        # fmt: off
        expected_values = [
            [1, 1, 2],
            [3, 1, 2],
            [1, 1, 2],
            [2, 1, 2],
            [0, 1, 2],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend02(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([1, 0, 0, 0, 0, 0, 0, 0])
        expected_values = [
            [1, 1, 1],
            [3, 1, 2],
            [3, 3, 2],
            [1, 2, 3],
            [0, 0, 0],
        ]
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend03(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([0, 0, 1])
        # fmt: off
        expected_values = [
            [2, 3, 3],
            [2, 3, 1],
            [2, 3, 3],
            [2, 3, 2],
            [2, 3, 0],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend04(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        expected_values = [
            [3, 3, 3],
            [2, 3, 1],
            [2, 1, 1],
            [1, 2, 3],
            [0, 0, 0],
        ]
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate1d(array, weights, 0, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend05(self):
        array = cupy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = cupy.asarray([[1, 0], [0, 0]])
        expected_values = [
            [[1, 1, 2], [1, 1, 2], [4, 4, 5]],
            [[9, 7, 8], [3, 1, 2], [6, 4, 5]],
            [[1, 1, 2], [1, 1, 2], [4, 4, 5]],
            [[5, 4, 5], [2, 1, 2], [5, 4, 5]],
            [[0, 0, 0], [0, 1, 2], [0, 4, 5]],
        ]
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend06(self):
        array = cupy.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = cupy.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        expected_values = [
            [[5, 6, 6], [8, 9, 9], [8, 9, 9]],
            [[5, 6, 4], [8, 9, 7], [2, 3, 1]],
            [[5, 6, 6], [8, 9, 9], [8, 9, 9]],
            [[5, 6, 5], [8, 9, 8], [5, 6, 5]],
            [[5, 6, 0], [8, 9, 0], [0, 0, 0]],
        ]
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend07(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        # fmt: off
        expected_values = [
            [3, 3, 3],
            [2, 3, 1],
            [2, 1, 1],
            [1, 2, 3],
            [0, 0, 0],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend08(self):
        array = cupy.asarray([[1], [2], [3]])
        weights = cupy.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        # fmt: off
        expected_values = [
            [[3], [3], [3]],
            [[2], [3], [1]],
            [[2], [1], [1]],
            [[1], [2], [3]],
            [[0], [0], [0]],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend09(self):
        array = cupy.asarray([1, 2, 3])
        weights = cupy.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1])
        # fmt: off
        expected_values = [
            [3, 3, 3],
            [2, 3, 1],
            [2, 1, 1],
            [1, 2, 3],
            [0, 0, 0],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_extend10(self):
        array = cupy.asarray([[1], [2], [3]])
        weights = cupy.asarray([[0], [0], [0], [0], [0], [0], [0], [0], [1]])
        # fmt: off
        expected_values = [
            [[3], [3], [3]],
            [[2], [3], [1]],
            [[2], [1], [1]],
            [[1], [2], [3]],
            [[0], [0], [0]],
        ]
        # fmt: on
        for mode, expected_value in zip(self.modes, expected_values):
            output = ndimage.correlate(array, weights, mode=mode, cval=0)
            assert_array_equal(output, expected_value)

    def test_fourier_shift_real01(self):
        for shape in [(32, 16), (31, 15)]:
            for type_, dec in zip([numpy.float32, numpy.float64], [4, 11]):
                expected = cupy.arange(shape[0] * shape[1], dtype=type_)
                expected.shape = shape
                a = fft.rfft(expected, shape[0], 0)
                a = fft.fft(a, shape[1], 1)
                a = ndimage.fourier_shift(a, [1, 1], shape[0], 0)
                a = fft.ifft(a, shape[1], 1)
                a = fft.irfft(a, shape[0], 0)
                assert_array_almost_equal(
                    a[1:, 1:], expected[:-1, :-1], decimal=dec
                )
                assert_array_almost_equal(
                    a.imag, cupy.zeros(shape), decimal=dec
                )

    def test_fourier_shift_complex01(self):
        for shape in [(32, 16), (31, 15)]:
            for type_, dec in zip([numpy.complex64, numpy.complex128], [4, 11]):
                expected = cupy.arange(shape[0] * shape[1], dtype=type_)
                expected.shape = shape
                a = fft.fft(expected, shape[0], 0)
                a = fft.fft(a, shape[1], 1)
                a = ndimage.fourier_shift(a, [1, 1], -1, 0)
                a = fft.ifft(a, shape[1], 1)
                a = fft.ifft(a, shape[0], 0)
                assert_array_almost_equal(
                    a.real[1:, 1:], expected[:-1, :-1], decimal=dec
                )
                assert_array_almost_equal(
                    a.imag, cupy.zeros(shape), decimal=dec
                )

    def test_spline01(self):
        for type_ in self.types:
            data = cupy.ones([], type_)
            for order in range(2, 6):
                out = ndimage.spline_filter(data, order=order)
                assert_array_almost_equal(out, 1)

    def test_spline02(self):
        for type_ in self.types:
            data = cupy.asarray([1], type_)
            for order in range(2, 6):
                out = ndimage.spline_filter(data, order=order)
                assert_array_almost_equal(out, [1])

    def test_spline03(self):
        for type_ in self.types:
            data = cupy.ones([], type_)
            for order in range(2, 6):
                out = ndimage.spline_filter(data, order, output=type_)
                assert_array_almost_equal(out, 1)

    def test_spline04(self):
        for type_ in self.types:
            data = cupy.ones([4], type_)
            for order in range(2, 6):
                out = ndimage.spline_filter(data, order)
                assert_array_almost_equal(out, [1, 1, 1, 1])

    def test_spline05(self):
        for type_ in self.types:
            data = cupy.ones([4, 4], type_)
            for order in range(2, 6):
                out = ndimage.spline_filter(data, order=order)
                # fmt: off
                assert_array_almost_equal(out, [[1, 1, 1, 1],
                                                [1, 1, 1, 1],
                                                [1, 1, 1, 1],
                                                [1, 1, 1, 1]])
                # fmt: on

    def test_generate_structure01(self):
        struct = ndimage.generate_binary_structure(0, 1)
        assert_array_almost_equal(struct, 1)

    def test_generate_structure02(self):
        struct = ndimage.generate_binary_structure(1, 1)
        assert_array_almost_equal(struct, [1, 1, 1])

    def test_generate_structure03(self):
        struct = ndimage.generate_binary_structure(2, 1)
        # fmt: off
        assert_array_almost_equal(struct, [[0, 1, 0],
                                           [1, 1, 1],
                                           [0, 1, 0]])
        # fmt: on

    def test_generate_structure04(self):
        struct = ndimage.generate_binary_structure(2, 2)
        # fmt: off
        assert_array_almost_equal(struct, [[1, 1, 1],
                                           [1, 1, 1],
                                           [1, 1, 1]])
        # fmt: on

    def test_iterate_structure01(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        out = ndimage.iterate_structure(struct, 2)
        assert_array_almost_equal(
            out,
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
        )

    def test_iterate_structure02(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1],
                               [0, 1]])
        # fmt: on
        out = ndimage.iterate_structure(struct, 2)
        assert_array_almost_equal(
            out, [[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]]
        )

    def test_iterate_structure03(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        out = ndimage.iterate_structure(struct, 2, 1)
        expected = [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]
        assert_array_almost_equal(out[0], expected)
        assert_equal(out[1], [2, 2])

    def test_binary_erosion01(self):
        for type_ in self.types:
            data = cupy.ones([], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, 1)

    def test_binary_erosion02(self):
        for type_ in self.types:
            data = cupy.ones([], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, 1)

    def test_binary_erosion03(self):
        for type_ in self.types:
            data = cupy.ones([1], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [0])

    def test_binary_erosion04(self):
        for type_ in self.types:
            data = cupy.ones([1], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [1])

    def test_binary_erosion05(self):
        for type_ in self.types:
            data = cupy.ones([3], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [0, 1, 0])

    def test_binary_erosion06(self):
        for type_ in self.types:
            data = cupy.ones([3], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [1, 1, 1])

    def test_binary_erosion07(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    def test_binary_erosion08(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    def test_binary_erosion09(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [0, 0, 0, 0, 0])

    def test_binary_erosion10(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [1, 0, 0, 0, 1])

    def test_binary_erosion11(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_erosion(data, struct, border_value=1)
            assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    def test_binary_erosion12(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_erosion(
                data, struct, border_value=1, origin=-1
            )
            assert_array_almost_equal(out, [0, 1, 0, 1, 1])

    def test_binary_erosion13(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_erosion(data, struct, border_value=1, origin=1)
            assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    def test_binary_erosion14(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            struct = cupy.asarray([1, 1])
            out = ndimage.binary_erosion(data, struct, border_value=1)
            assert_array_almost_equal(out, [1, 1, 0, 0, 1])

    def test_binary_erosion15(self):
        for type_ in self.types:
            data = cupy.ones([5], type_)
            data[2] = 0
            struct = cupy.asarray([1, 1])
            out = ndimage.binary_erosion(
                data, struct, border_value=1, origin=-1
            )
            assert_array_almost_equal(out, [1, 0, 0, 1, 1])

    def test_binary_erosion16(self):
        for type_ in self.types:
            data = cupy.ones([1, 1], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [[1]])

    def test_binary_erosion17(self):
        for type_ in self.types:
            data = cupy.ones([1, 1], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [[0]])

    def test_binary_erosion18(self):
        for type_ in self.types:
            data = cupy.ones([1, 3], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [[0, 0, 0]])

    def test_binary_erosion19(self):
        for type_ in self.types:
            data = cupy.ones([1, 3], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, [[1, 1, 1]])

    def test_binary_erosion19_noncontig(self):
        # TODO: grlee77: add other cases with non-contiguous input
        for type_ in self.types:
            data = cupy.ones([1, 3, 4, 5, 0, 0, 0, 5], type_)
            out = ndimage.binary_erosion(data[::-1], border_value=1)
            expected = ndimage.binary_erosion(
                cupy.ascontiguousarray(data[::-1]), border_value=1
            )
            assert_array_almost_equal(out, expected)

    def test_binary_erosion20(self):
        for type_ in self.types:
            data = cupy.ones([3, 3], type_)
            out = ndimage.binary_erosion(data)
            assert_array_almost_equal(out, [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

    def test_binary_erosion21(self):
        for type_ in self.types:
            data = cupy.ones([3, 3], type_)
            out = ndimage.binary_erosion(data, border_value=1)
            # fmt: off
            assert_array_almost_equal(out, [[1, 1, 1],
                                            [1, 1, 1],
                                            [1, 1, 1]])
            # fmt: on

    def test_binary_erosion22(self):
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_erosion(data, border_value=1)
            assert_array_almost_equal(out, expected)

            # grlee77 add non-contiguous test case
            out2 = ndimage.binary_erosion(data[::-1], border_value=1)
            expected2 = ndimage.binary_erosion(
                cupy.ascontiguousarray(data[::-1]), border_value=1
            )
            assert_array_almost_equal(out2, expected2)

    def test_binary_erosion23(self):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_erosion(data, struct, border_value=1)
            assert_array_almost_equal(out, expected)

    def test_binary_erosion24(self):
        struct = cupy.asarray([[0, 1], [1, 1]])
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_erosion(data, struct, border_value=1)
            assert_array_almost_equal(out, expected)

            # grlee77: add cases with non-contiguous input
            out2 = ndimage.binary_erosion(data[::-1], struct, border_value=1)
            expected2 = ndimage.binary_erosion(
                data[::-1].copy(), struct, border_value=1
            )
            assert_array_almost_equal(out2, expected2)

            out2 = ndimage.binary_erosion(data, struct[::-1], border_value=1)
            expected2 = ndimage.binary_erosion(
                data, struct[::-1].copy(), border_value=1
            )
            assert_array_almost_equal(out2, expected2)

    def test_binary_erosion25(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_erosion(data, struct, border_value=1)
            assert_array_almost_equal(out, expected)

    def test_binary_erosion26(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 1, 1, 1, 0, 1, 1],
                    [0, 0, 1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_erosion(
                data, struct, border_value=1, origin=(-1, -1)
            )
            assert_array_almost_equal(out, expected)

    def test_binary_erosion27(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=2,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion28(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=2,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion29(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=3,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion30(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=3,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion31(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=1,
            output=out,
            origin=(-1, -1),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion32(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=2,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion33(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        mask = cupy.asarray(
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ]
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            mask=mask,
            iterations=-1,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion34(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        mask = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_erosion(data, struct, border_value=1, mask=mask)
        assert_array_almost_equal(out, expected)

    def test_binary_erosion35(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        mask = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        tmp = cupy.asarray(
            [
                [0, 0, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 1],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1],
            ]
        )
        expected = cupy.logical_and(tmp, mask)
        tmp = cupy.logical_and(data, cupy.logical_not(mask))
        expected = cupy.logical_or(expected, tmp)
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=1,
            output=out,
            origin=(-1, -1),
            mask=mask,
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion36(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]])
        # fmt: on
        mask = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        tmp = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 1, 1],
                [0, 0, 1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        expected = cupy.logical_and(tmp, mask)
        tmp = cupy.logical_and(data, cupy.logical_not(mask))
        expected = cupy.logical_or(expected, tmp)
        out = ndimage.binary_erosion(
            data, struct, mask=mask, border_value=1, origin=(-1, -1)
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion37(self):
        a = cupy.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        b = cupy.zeros_like(a)
        out = ndimage.binary_erosion(
            a,
            structure=a,
            output=b,
            iterations=1,
            border_value=True,
            brute_force=True,
        )
        assert out is b

        b = cupy.zeros_like(a)
        out = ndimage.binary_erosion(
            a,
            structure=a,
            output=b,
            iterations=2,
            border_value=True,
            brute_force=True,
        )
        assert out is b

        if brute_force_implemented:
            assert_array_equal(
                ndimage.binary_erosion(
                    a, structure=a, iterations=0, border_value=True
                ),
                b,
            )

    def test_binary_erosion38(self):
        data = cupy.asarray([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
        iterations = 2.0
        with assert_raises(TypeError):
            ndimage.binary_erosion(data, iterations=iterations)

    def test_binary_erosion39(self):
        iterations = cupy.int32(3)
        struct = cupy.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=iterations,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_erosion40(self):
        iterations = cupy.int64(3)
        struct = cupy.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        expected = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
        data = cupy.asarray(
            [
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_erosion(
            data,
            struct,
            border_value=1,
            iterations=iterations,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation01(self):
        for type_ in self.types:
            data = cupy.ones([], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, 1)

    def test_binary_dilation02(self):
        for type_ in self.types:
            data = cupy.zeros([], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, 0)

    def test_binary_dilation03(self):
        for type_ in self.types:
            data = cupy.ones([1], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [1])

    def test_binary_dilation04(self):
        for type_ in self.types:
            data = cupy.zeros([1], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [0])

    def test_binary_dilation05(self):
        for type_ in self.types:
            data = cupy.ones([3], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [1, 1, 1])

    def test_binary_dilation06(self):
        for type_ in self.types:
            data = cupy.zeros([3], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [0, 0, 0])

    def test_binary_dilation07(self):
        for type_ in self.types:
            data = cupy.zeros([3], type_)
            data[1] = 1
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [1, 1, 1])

    def test_binary_dilation08(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            data[3] = 1
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [1, 1, 1, 1, 1])

    def test_binary_dilation09(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [1, 1, 1, 0, 0])

    def test_binary_dilation10(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            out = ndimage.binary_dilation(data, origin=-1)
            assert_array_almost_equal(out, [0, 1, 1, 1, 0])

    def test_binary_dilation11(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            out = ndimage.binary_dilation(data, origin=1)
            assert_array_almost_equal(out, [1, 1, 0, 0, 0])

    def test_binary_dilation12(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_dilation(data, struct)
            assert_array_almost_equal(out, [1, 0, 1, 0, 0])

    def test_binary_dilation13(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_dilation(data, struct, border_value=1)
            assert_array_almost_equal(out, [1, 0, 1, 0, 1])

    def test_binary_dilation14(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_dilation(data, struct, origin=-1)
            assert_array_almost_equal(out, [0, 1, 0, 1, 0])

    def test_binary_dilation15(self):
        for type_ in self.types:
            data = cupy.zeros([5], type_)
            data[1] = 1
            struct = cupy.asarray([1, 0, 1])
            out = ndimage.binary_dilation(
                data, struct, origin=-1, border_value=1
            )
            assert_array_almost_equal(out, [1, 1, 0, 1, 0])

    def test_binary_dilation16(self):
        for type_ in self.types:
            data = cupy.ones([1, 1], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [[1]])

    def test_binary_dilation17(self):
        for type_ in self.types:
            data = cupy.zeros([1, 1], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [[0]])

    def test_binary_dilation18(self):
        for type_ in self.types:
            data = cupy.ones([1, 3], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [[1, 1, 1]])

    def test_binary_dilation19(self):
        for type_ in self.types:
            data = cupy.ones([3, 3], type_)
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def test_binary_dilation20(self):
        for type_ in self.types:
            data = cupy.zeros([3, 3], type_)
            data[1, 1] = 1
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])

    def test_binary_dilation21(self):
        struct = ndimage.generate_binary_structure(2, 2)
        for type_ in self.types:
            data = cupy.zeros([3, 3], type_)
            data[1, 1] = 1
            out = ndimage.binary_dilation(data, struct)
            assert_array_almost_equal(out, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def test_binary_dilation22(self):
        expected = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation23(self):
        expected = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data, border_value=1)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation24(self):
        expected = [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data, origin=(1, 1))
            assert_array_almost_equal(out, expected)

    def test_binary_dilation25(self):
        expected = [
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data, origin=(1, 1), border_value=1)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation26(self):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data, struct)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation27(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1]])
        # fmt: on
        expected = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]

        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(data, struct)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation28(self):
        # fmt: off
        expected = [[1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]]
        # fmt: on

        for type_ in self.types:
            # fmt: off
            data = cupy.asarray([[0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], type_)
            # fmt: on
            out = ndimage.binary_dilation(data, border_value=1)
            assert_array_almost_equal(out, expected)

    def test_binary_dilation29(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_dilation(
            data,
            struct,
            iterations=2,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation30(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_dilation(
            data,
            struct,
            iterations=2,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation31(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1]])
        # fmt: on
        expected = [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_dilation(
            data,
            struct,
            iterations=3,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation32(self):
        # fmt: off
        struct = cupy.asarray([[0, 1],
                               [1, 1]])
        # fmt: on
        expected = [
            [0, 0, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = cupy.zeros(data.shape, bool)
        ndimage.binary_dilation(
            data,
            struct,
            iterations=3,
            output=out,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation33(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        mask = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )

        out = ndimage.binary_dilation(
            data,
            struct,
            iterations=-1,
            mask=mask,
            border_value=0,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation34(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        mask = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.zeros(mask.shape, bool)
        out = ndimage.binary_dilation(
            data,
            struct,
            iterations=-1,
            mask=mask,
            border_value=1,
            brute_force=(not brute_force_implemented),
        )
        assert_array_almost_equal(out, expected)

    def test_binary_dilation35(self):
        tmp = cupy.asarray(
            [
                [1, 1, 0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 1, 0, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        mask = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        expected = cupy.logical_and(tmp, mask)
        tmp = cupy.logical_and(data, cupy.logical_not(mask))
        expected = cupy.logical_or(expected, tmp)
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_dilation(
                data, mask=mask, origin=(1, 1), border_value=1
            )
            assert_array_almost_equal(out, expected)

    def test_binary_propagation01(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        mask = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )

        out = ndimage.binary_propagation(
            data, struct, mask=mask, border_value=0
        )
        assert_array_almost_equal(out, expected)

    def test_binary_propagation02(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        mask = cupy.asarray(
            [
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.zeros(mask.shape, bool)
        out = ndimage.binary_propagation(
            data, struct, mask=mask, border_value=1
        )
        assert_array_almost_equal(out, expected)

    def test_binary_opening01(self):
        expected = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_opening(data)
            assert_array_almost_equal(out, expected)

    def test_binary_opening02(self):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_opening(data, struct)
            assert_array_almost_equal(out, expected)

    def test_binary_closing01(self):
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_closing(data)
            assert_array_almost_equal(out, expected)

    def test_binary_closing02(self):
        struct = ndimage.generate_binary_structure(2, 2)
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_closing(data, struct)
            assert_array_almost_equal(out, expected)

    def test_binary_fill_holes01(self):
        expected = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_binary_fill_holes02(self):
        expected = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_binary_fill_holes03(self):
        expected = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 0, 1, 1, 1],
                [0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        data = cupy.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [0, 0, 1, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            bool,
        )
        out = ndimage.binary_fill_holes(data)
        assert_array_almost_equal(out, expected)

    def test_grey_erosion01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        output = ndimage.grey_erosion(array, footprint=footprint)
        assert_array_almost_equal(
            [[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]], output
        )

    def test_grey_erosion02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        output = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(
            [[2, 2, 1, 1, 1], [2, 3, 1, 3, 1], [5, 5, 3, 3, 1]], output
        )

    def test_grey_erosion03(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(
            [[1, 1, 0, 0, 0], [1, 2, 0, 2, 0], [4, 4, 2, 2, 0]], output
        )

    def test_grey_dilation01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[0, 1, 1], [1, 0, 1]])
        output = ndimage.grey_dilation(array, footprint=footprint)
        assert_array_almost_equal(
            [[7, 7, 9, 9, 5], [7, 9, 8, 9, 7], [8, 8, 8, 7, 7]], output
        )

    def test_grey_dilation02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[0, 1, 1], [1, 0, 1]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        output = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(
            [[7, 7, 9, 9, 5], [7, 9, 8, 9, 7], [8, 8, 8, 7, 7]], output
        )

    def test_grey_dilation03(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[0, 1, 1], [1, 0, 1]])
        structure = cupy.asarray([[1, 1, 1], [1, 1, 1]])
        output = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(
            [[8, 8, 10, 10, 6], [8, 10, 9, 10, 8], [9, 9, 9, 8, 8]], output
        )

    def test_grey_opening01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        tmp = ndimage.grey_erosion(array, footprint=footprint)
        expected = ndimage.grey_dilation(tmp, footprint=footprint)
        output = ndimage.grey_opening(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_grey_opening02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        expected = ndimage.grey_dilation(
            tmp, footprint=footprint, structure=structure
        )
        output = ndimage.grey_opening(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    def test_grey_closing01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        tmp = ndimage.grey_dilation(array, footprint=footprint)
        expected = ndimage.grey_erosion(tmp, footprint=footprint)
        output = ndimage.grey_closing(array, footprint=footprint)
        assert_array_almost_equal(expected, output)

    def test_grey_closing02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        expected = ndimage.grey_erosion(
            tmp, footprint=footprint, structure=structure
        )
        output = ndimage.grey_closing(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    def test_morphological_gradient01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp1 = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        tmp2 = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        expected = tmp1 - tmp2
        output = cupy.zeros(array.shape, array.dtype)
        ndimage.morphological_gradient(
            array, footprint=footprint, structure=structure, output=output
        )
        assert_array_almost_equal(expected, output)

    def test_morphological_gradient02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp1 = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        tmp2 = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        expected = tmp1 - tmp2
        output = ndimage.morphological_gradient(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    def test_morphological_laplace01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp1 = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        tmp2 = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        expected = tmp1 + tmp2 - 2 * array
        output = cupy.zeros(array.shape, array.dtype)
        ndimage.morphological_laplace(
            array, footprint=footprint, structure=structure, output=output
        )
        assert_array_almost_equal(expected, output)

    def test_morphological_laplace02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp1 = ndimage.grey_dilation(
            array, footprint=footprint, structure=structure
        )
        tmp2 = ndimage.grey_erosion(
            array, footprint=footprint, structure=structure
        )
        expected = tmp1 + tmp2 - 2 * array
        output = ndimage.morphological_laplace(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    def test_white_tophat01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_opening(
            array, footprint=footprint, structure=structure
        )
        expected = array - tmp
        output = cupy.zeros(array.shape, array.dtype)
        ndimage.white_tophat(
            array, footprint=footprint, structure=structure, output=output
        )
        assert_array_almost_equal(expected, output)

    def test_white_tophat02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_opening(
            array, footprint=footprint, structure=structure
        )
        expected = array - tmp
        output = ndimage.white_tophat(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    @pytest.mark.xfail(
        True, reason="known bug for white_tophat with boolean input"
    )
    def test_white_tophat03(self):
        array = cupy.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=numpy.bool_,
        )
        structure = cupy.ones((3, 3), dtype=numpy.bool_)
        expected = cupy.asarray(
            [
                [0, 1, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 1, 1, 0],
                [1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 1],
                [0, 1, 1, 0, 1, 0, 1],
                [0, 1, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1, 1, 1],
            ],
            dtype=numpy.bool_,
        )

        output = ndimage.white_tophat(array, structure=structure)
        assert_array_equal(expected, output)

    @pytest.mark.xfail(
        True, reason="known bug for white_tophat with boolean input"
    )
    def test_white_tophat04(self):
        array = cupy.eye(5, dtype=numpy.bool_)
        structure = cupy.ones((3, 3), dtype=numpy.bool_)

        # Check that type mismatch is properly handled
        output = cupy.empty_like(array, dtype=numpy.float)
        ndimage.white_tophat(array, structure=structure, output=output)

    def test_black_tophat01(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_closing(
            array, footprint=footprint, structure=structure
        )
        expected = tmp - array
        output = cupy.zeros(array.shape, array.dtype)
        ndimage.black_tophat(
            array, footprint=footprint, structure=structure, output=output
        )
        assert_array_almost_equal(expected, output)

    def test_black_tophat02(self):
        # fmt: off
        array = cupy.asarray([[3, 2, 5, 1, 4],
                              [7, 6, 9, 3, 5],
                              [5, 8, 3, 7, 1]])
        # fmt: on
        footprint = cupy.asarray([[1, 0, 1], [1, 1, 0]])
        structure = cupy.asarray([[0, 0, 0], [0, 0, 0]])
        tmp = ndimage.grey_closing(
            array, footprint=footprint, structure=structure
        )
        expected = tmp - array
        output = ndimage.black_tophat(
            array, footprint=footprint, structure=structure
        )
        assert_array_almost_equal(expected, output)

    @pytest.mark.xfail(
        True, reason="known bug for white_tophat with boolean input"
    )
    def test_black_tophat03(self):
        array = cupy.asarray(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=numpy.bool_,
        )
        structure = cupy.ones((3, 3), dtype=numpy.bool_)
        expected = cupy.asarray(
            [
                [0, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 0],
            ],
            dtype=numpy.bool_,
        )

        output = ndimage.black_tophat(array, structure=structure)
        assert_array_equal(expected, output)

    @pytest.mark.xfail(
        True, reason="known bug for white_tophat with boolean input"
    )
    def test_black_tophat04(self):
        array = cupy.eye(5, dtype=numpy.bool_)
        structure = cupy.ones((3, 3), dtype=numpy.bool_)

        # Check that type mismatch is properly handled
        output = cupy.empty_like(array, dtype=numpy.float)
        ndimage.black_tophat(array, structure=structure, output=output)

    def test_hit_or_miss01(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1],
                    [0, 0, 1, 1, 1],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = cupy.zeros(data.shape, bool)
            ndimage.binary_hit_or_miss(data, struct, output=out)
            assert_array_almost_equal(expected, out)

    def test_hit_or_miss02(self):
        # fmt: off
        struct = cupy.asarray([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 1, 0, 0, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_hit_or_miss(data, struct)
            assert_array_almost_equal(expected, out)

    def test_hit_or_miss03(self):
        # fmt: off
        struct1 = cupy.asarray([[0, 0, 0],
                                [1, 1, 1],
                                [0, 0, 0]])

        struct2 = cupy.asarray([[1, 1, 1],
                                [0, 0, 0],
                                [1, 1, 1]])
        # fmt: on
        expected = [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        for type_ in self.types:
            data = cupy.asarray(
                [
                    [0, 1, 0, 0, 1, 1, 1, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                ],
                type_,
            )
            out = ndimage.binary_hit_or_miss(data, struct1, struct2)
            assert_array_almost_equal(expected, out)


class TestDilateFix:
    def setup_method(self):
        # dilation related setup
        # fmt: off
        self.array = cupy.asarray([[0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 0],
                                   [0, 0, 1, 1, 0],
                                   [0, 0, 0, 0, 0]], dtype=numpy.uint8)
        # fmt: on
        self.sq3x3 = cupy.ones((3, 3))
        dilated3x3 = ndimage.binary_dilation(self.array, structure=self.sq3x3)
        self.dilated3x3 = dilated3x3.view(cupy.uint8)

    def test_dilation_square_structure(self):
        result = ndimage.grey_dilation(self.array, structure=self.sq3x3)
        # +1 accounts for difference between grey and binary dilation
        assert_array_almost_equal(result, self.dilated3x3 + 1)

    def test_dilation_scalar_size(self):
        result = ndimage.grey_dilation(self.array, size=3)
        assert_array_almost_equal(result, self.dilated3x3)


class TestBinaryOpeningClosing:
    def setup_method(self):
        a = cupy.zeros((5, 5), dtype=bool)
        a[1:4, 1:4] = True
        a[4, 4] = True
        self.array = a
        self.sq3x3 = cupy.ones((3, 3))
        self.opened_old = ndimage.binary_opening(
            self.array, self.sq3x3, 1, None, 0
        )
        self.closed_old = ndimage.binary_closing(
            self.array, self.sq3x3, 1, None, 0
        )

    def test_opening_new_arguments(self):
        opened_new = ndimage.binary_opening(
            self.array, self.sq3x3, 1, None, 0, None, 0, False
        )
        assert_array_equal(opened_new, self.opened_old)

    def test_closing_new_arguments(self):
        closed_new = ndimage.binary_closing(
            self.array, self.sq3x3, 1, None, 0, None, 0, False
        )
        assert_array_equal(closed_new, self.closed_old)
