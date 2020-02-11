import itertools
import math

import cupy as cp
import numpy as np
import pytest
from cupy.testing import (
    assert_allclose,
    assert_array_equal,
    assert_array_almost_equal,
)
from fast_upfirdn import upfirdn_modes
from numpy.testing import assert_, assert_equal

from cupyimg.scipy import signal as signal_cp
from cupyimg.scipy.signal import (
    choose_conv_method,
    convolve,
    convolve2d,
    correlate,
    correlate2d,
    fftconvolve,
    hilbert,
    hilbert2,
    oaconvolve,
    resample,
    resample_poly,
    wiener,
)
from cupyimg.scipy.ndimage import correlate1d

signal = pytest.importorskip("scipy.signal")


class _TestConvolve(object):
    def test_basic(self):
        a = cp.asarray([3, 4, 5, 6, 5, 4])
        b = cp.asarray([1, 2, 3])
        c = convolve(a, b)
        assert_array_equal(c, cp.asarray([3, 10, 22, 28, 32, 32, 23, 12]))

    def test_same(self):
        a = cp.asarray([3, 4, 5])
        b = cp.asarray([1, 2, 3, 4])
        c = convolve(a, b, mode="same")
        assert_array_equal(c, cp.asarray([10, 22, 34]))

    def test_same_eq(self):
        a = cp.asarray([3, 4, 5])
        b = cp.asarray([1, 2, 3])
        c = convolve(a, b, mode="same")
        assert_array_equal(c, cp.asarray([10, 22, 22]))

    def test_complex(self):
        x = cp.asarray([1 + 1j, 2 + 1j, 3 + 1j])
        y = cp.asarray([1 + 1j, 2 + 1j])
        z = convolve(x, y)
        assert_array_equal(z, cp.asarray([2j, 2 + 6j, 5 + 8j, 5 + 5j]))

    def test_zero_rank(self):
        a = cp.asarray(1289)
        b = cp.asarray(4567)
        c = convolve(a, b)
        assert_array_equal(c, a * b)

    def test_broadcastable(self):
        a = cp.arange(27).reshape(3, 3, 3)
        b = cp.arange(3)
        for i in range(3):
            b_shape = [1] * 3
            b_shape[i] = 3
            x = convolve(a, b.reshape(b_shape), method="direct")
            y = convolve(a, b.reshape(b_shape), method="fft")
            assert_allclose(x, y)

    def test_single_element(self):
        a = cp.asarray([4967])
        b = cp.asarray([3920])
        c = convolve(a, b)
        assert_array_equal(c, a * b)

    def test_2d_arrays(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve(a, b)
        d = cp.asarray(
            [[2, 7, 16, 17, 12], [10, 30, 62, 58, 38], [12, 31, 58, 49, 30]]
        )
        assert_array_equal(c, d)

    def test_input_swapping(self):
        small = cp.arange(8).reshape(2, 2, 2)
        big = 1j * cp.arange(27).reshape(3, 3, 3)
        big += cp.arange(27)[::-1].reshape(3, 3, 3)

        out_array = cp.asarray(
            [
                [
                    [0 + 0j, 26 + 0j, 25 + 1j, 24 + 2j],
                    [52 + 0j, 151 + 5j, 145 + 11j, 93 + 11j],
                    [46 + 6j, 133 + 23j, 127 + 29j, 81 + 23j],
                    [40 + 12j, 98 + 32j, 93 + 37j, 54 + 24j],
                ],
                [
                    [104 + 0j, 247 + 13j, 237 + 23j, 135 + 21j],
                    [282 + 30j, 632 + 96j, 604 + 124j, 330 + 86j],
                    [246 + 66j, 548 + 180j, 520 + 208j, 282 + 134j],
                    [142 + 66j, 307 + 161j, 289 + 179j, 153 + 107j],
                ],
                [
                    [68 + 36j, 157 + 103j, 147 + 113j, 81 + 75j],
                    [174 + 138j, 380 + 348j, 352 + 376j, 186 + 230j],
                    [138 + 174j, 296 + 432j, 268 + 460j, 138 + 278j],
                    [70 + 138j, 145 + 323j, 127 + 341j, 63 + 197j],
                ],
                [
                    [32 + 72j, 68 + 166j, 59 + 175j, 30 + 100j],
                    [68 + 192j, 139 + 433j, 117 + 455j, 57 + 255j],
                    [38 + 222j, 73 + 499j, 51 + 521j, 21 + 291j],
                    [12 + 144j, 20 + 318j, 7 + 331j, 0 + 182j],
                ],
            ]
        )

        assert_array_equal(convolve(small, big, "full"), out_array)
        assert_array_equal(convolve(big, small, "full"), out_array)
        assert_array_equal(
            convolve(small, big, "same"), out_array[1:3, 1:3, 1:3]
        )
        assert_array_equal(
            convolve(big, small, "same"), out_array[0:3, 0:3, 0:3]
        )
        assert_array_equal(
            convolve(small, big, "valid"), out_array[1:3, 1:3, 1:3]
        )
        assert_array_equal(
            convolve(big, small, "valid"), out_array[1:3, 1:3, 1:3]
        )

    def test_invalid_params(self):
        a = cp.asarray([3, 4, 5])
        b = cp.asarray([1, 2, 3])
        with pytest.raises(ValueError):
            convolve(a, b, mode="spam")
        with pytest.raises(ValueError):
            convolve(a, b, mode="eggs", method="fft")
        with pytest.raises(ValueError):
            convolve(a, b, mode="ham", method="direct")
        with pytest.raises(ValueError):
            convolve(a, b, mode="full", method="bacon")
        with pytest.raises(ValueError):
            convolve(a, b, mode="same", method="bacon")


class TestConvolve(_TestConvolve):
    def test_valid_mode2(self):
        # See gh-5897
        a = cp.asarray([1, 2, 3, 6, 5, 3])
        b = cp.asarray([2, 3, 4, 5, 3, 4, 2, 2, 1])
        expected = cp.asarray([70, 78, 73, 65])

        out = convolve(a, b, "valid")
        assert_array_equal(out, expected)

        out = convolve(b, a, "valid")
        assert_array_equal(out, expected)

        a = cp.asarray([1 + 5j, 2 - 1j, 3 + 0j])
        b = cp.asarray([2 - 3j, 1 + 0j])
        expected = cp.asarray([2 - 3j, 8 - 10j])

        out = convolve(a, b, "valid")
        assert_array_equal(out, expected)

        out = convolve(b, a, "valid")
        assert_array_equal(out, expected)

    def test_same_mode(self):
        a = cp.asarray([1, 2, 3, 3, 1, 2])
        b = cp.asarray([1, 4, 3, 4, 5, 6, 7, 4, 3, 2, 1, 1, 3])
        c = convolve(a, b, "same")
        d = cp.asarray([57, 61, 63, 57, 45, 36])
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = cp.arange(1, 7).reshape((2, 3))
        b = cp.arange(-6, 0).reshape((3, 2))

        with pytest.raises(ValueError):
            convolve(*(a, b), **{"mode": "valid"})
        with pytest.raises(ValueError):
            convolve(*(b, a), **{"mode": "valid"})

    def test_convolve_method(self, n=100):

        types = sum([t for _, t in np.sctypes.items()], [])
        types = {np.dtype(t).name for t in types}

        # These types include 'bool' and all precisions (int8, float32, etc)
        # The removed types throw errors in correlate or fftconvolve
        for dtype in [
            "complex256",
            "complex192",
            "float128",
            "float96",
            "str",
            "void",
            "bytes",
            "object",
            "unicode",
            "string",
        ]:
            if dtype in types:
                types.remove(dtype)

        # TODO: grlee77
        # only single and double precision float and complex are supported
        # by mode 'fft'. fix this
        float_types = ["float32", "float64", "complex64", "complex128"]
        args = [
            (t1, t2, mode)
            for t1 in types
            for t2 in types
            for mode in ["valid", "full", "same"]
        ]

        # These are random arrays, which means test is much stronger than
        # convolving testing by convolving two np.ones arrays
        np.random.seed(42)
        array_types = {
            "i": cp.asarray(np.random.choice([0, 1], size=n)),
            "f": cp.asarray(np.random.randn(n)),
        }
        array_types["b"] = array_types["u"] = array_types["i"]
        array_types["c"] = array_types["f"] + 0.5j * array_types["f"]

        for t1, t2, mode in args:
            x1 = array_types[np.dtype(t1).kind].astype(t1)
            x2 = array_types[np.dtype(t2).kind].astype(t2)

            results = {
                key: convolve(x1.copy(), x2.copy(), method=key, mode=mode)
                for key in ["fft", "direct"]
            }
            print(
                f"{x1.dtype}, {x2.dtype}, {results['fft'].dtype}, {results['direct'].dtype}"
            )
            assert results["fft"].dtype == results["direct"].dtype

            if "bool" in t1 and "bool" in t2:
                assert choose_conv_method(x1, x2) == "direct"
                continue

            # Found by experiment. Found approx smallest value for (rtol, atol)
            # threshold to have tests pass.
            if any([t in {"complex64", "float32"} for t in [t1, t2]]):
                # grlee77: Note: had to increase atol from 1e-6 to 1e-5
                kwargs = {"rtol": 1.0e-4, "atol": 1e-5}
            elif "float16" in [t1, t2]:
                # atol is default for np.allclose
                kwargs = {"rtol": 1e-3, "atol": 1e-3}
            else:
                # defaults for np.allclose (different from assert_allclose)
                kwargs = {"rtol": 1e-5, "atol": 1e-8}

            if "fft" in results:
                assert_allclose(results["fft"], results["direct"], **kwargs)

    def test_convolve_method_large_input(self):
        # This is really a test that convolving two large integers goes to the
        # direct method even if they're in the fft method.
        for n in [10, 20, 50, 51, 52, 53, 54, 60, 62]:
            z = cp.asarray([2 ** n], dtype=np.int64)
            fft = convolve(z, z, method="fft")
            direct = convolve(z, z, method="direct")

            # this is the case when integer precision gets to us
            # issue #6076 has more detail, hopefully more tests after resolved
            if n < 50:
                assert_array_equal(fft, direct)
                assert_array_equal(fft, 2 ** (2 * n))
                assert_array_equal(direct, 2 ** (2 * n))

    def test_mismatched_dims(self):
        # Input arrays should have the same number of dimensions
        with pytest.raises(ValueError):
            convolve([1], 2, method="direct")
        with pytest.raises(ValueError):
            convolve(1, [2], method="direct")
        with pytest.raises(ValueError):
            convolve([1], 2, method="fft")
        with pytest.raises(ValueError):
            convolve(1, [2], method="fft")
        with pytest.raises(ValueError):
            convolve([1], [[2]])
        with pytest.raises(ValueError):
            convolve([3], 2)


class _TestConvolve2d(object):
    def test_2d_arrays(self):
        a = cp.asarray([[1, 2, 3], [3, 4, 5]])
        b = cp.asarray([[2, 3, 4], [4, 5, 6]])
        # fmt: off
        d = cp.asarray([[2, 7, 16, 17, 12],
                        [10, 30, 62, 58, 38],
                        [12, 31, 58, 49, 30]])
        # fmt: on
        e = convolve2d(a, b)
        assert_array_equal(e, d)

    def test_valid_mode(self):
        e = cp.asarray([[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]])
        f = cp.asarray([[1, 2, 3], [3, 4, 5]])
        h = cp.asarray([[62, 80, 98, 116, 134]])

        g = convolve2d(e, f, "valid")
        assert_array_equal(g, h)

        # See gh-5897
        g = convolve2d(f, e, "valid")
        assert_array_equal(g, h)

    def test_valid_mode_complx(self):
        e = cp.asarray([[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]])
        f = cp.asarray([[1, 2, 3], [3, 4, 5]], dtype=complex) + 1j
        h = cp.asarray(
            [
                [
                    62.0 + 24.0j,
                    80.0 + 30.0j,
                    98.0 + 36.0j,
                    116.0 + 42.0j,
                    134.0 + 48.0j,
                ]
            ]
        )

        g = convolve2d(e, f, "valid")
        assert_array_almost_equal(g, h)

        # See gh-5897
        g = convolve2d(f, e, "valid")
        assert_array_equal(g, h)

    def test_fillvalue(self):
        a = cp.asarray([[1, 2, 3], [3, 4, 5]])
        b = cp.asarray([[2, 3, 4], [4, 5, 6]])
        fillval = 1
        c = convolve2d(a, b, "full", "fill", fillval)
        # fmt: off
        d = cp.asarray([[24, 26, 31, 34, 32],
                        [28, 40, 62, 64, 52],
                        [32, 46, 67, 62, 48]])
        # fmt: on
        assert_array_equal(c, d)

    def test_fillvalue_deprecations(self):

        # grlee77: changed to ValueError
        with pytest.raises(ValueError):
            convolve2d([[1]], [[1, 2]], fillvalue=1j)

        # # Deprecated 2017-07, scipy version 1.0.0
        # with suppress_warnings() as sup:
        #     sup.filter(np.ComplexWarning, "Casting complex values to real")
        #     r = sup.record(DeprecationWarning, "could not cast `fillvalue`")
        #     convolve2d([[1]], [[1, 2]], fillvalue=1j)
        #     assert len(r) == 1
        #     warnings.filterwarnings(
        #         "error", message="could not cast `fillvalue`",
        #         category=DeprecationWarning)
        #     assert_raises(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
        #                   fillvalue=1j)

        # with suppress_warnings():
        #     warnings.filterwarnings(
        #         "always", message="`fillvalue` must be scalar or an array ",
        #         category=DeprecationWarning)
        #     assert_warns(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
        #                  fillvalue=[1, 2])
        #     warnings.filterwarnings(
        #         "error", message="`fillvalue` must be scalar or an array ",
        #         category=DeprecationWarning)
        #     assert_raises(DeprecationWarning, convolve2d, [[1]], [[1, 2]],
        #                   fillvalue=[1, 2])

        # grlee77: changed to ValueError
        with pytest.raises(ValueError):
            convolve2d([[1]], [[1, 2]], fillvalue=[1, 2])

    def test_fillvalue_empty(self):
        # Check that fillvalue being empty raises an error:
        with pytest.raises(ValueError):
            convolve2d([[1]], [[1, 2]], fillvalue=[])

    def test_wrap_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, "full", "wrap")
        # fmt: off
        d = cp.asarray([[80, 80, 74, 80, 80],
                        [68, 68, 62, 68, 68],
                        [80, 80, 74, 80, 80]])
        # fmt: on
        assert_array_equal(c, d)

    def test_sym_boundary(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [[2, 3, 4], [4, 5, 6]]
        c = convolve2d(a, b, "full", "symm")
        d = cp.asarray(
            [[34, 30, 44, 62, 66], [52, 48, 62, 80, 84], [82, 78, 92, 110, 114]]
        )
        assert_array_equal(c, d)

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = cp.arange(1, 7).reshape((2, 3))
        b = cp.arange(-6, 0).reshape((3, 2))

        with pytest.raises(ValueError):
            convolve2d(*(a, b), **{"mode": "valid"})
            convolve2d(*(b, a), **{"mode": "valid"})


class TestConvolve2d(_TestConvolve2d):
    def test_same_mode(self):
        e = cp.asarray([[1, 2, 3], [3, 4, 5]])
        f = cp.asarray([[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]])
        g = convolve2d(e, f, "same")
        # fmt: off
        h = cp.asarray([[22, 28, 34],
                        [80, 98, 116]])
        # fmt: on
        assert_array_equal(g, h)

    def test_valid_mode2(self):
        # See gh-5897
        e = cp.asarray([[1, 2, 3], [3, 4, 5]])
        f = cp.asarray([[2, 3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9, 10]])
        expected = cp.asarray([[62, 80, 98, 116, 134]])

        out = convolve2d(e, f, "valid")
        assert_array_equal(out, expected)

        out = convolve2d(f, e, "valid")
        assert_array_equal(out, expected)

        e = cp.asarray([[1 + 1j, 2 - 3j], [3 + 1j, 4 + 0j]])
        f = cp.asarray([[2 - 1j, 3 + 2j, 4 + 0j], [4 - 0j, 5 + 1j, 6 - 3j]])
        expected = cp.asarray([[27 - 1j, 46.0 + 2j]])

        out = convolve2d(e, f, "valid")
        assert_array_equal(out, expected)

        # See gh-5897
        out = convolve2d(f, e, "valid")
        assert_array_equal(out, expected)

    def test_consistency_convolve_funcs(self):
        # Compare np.convolve, convolve, convolve2d
        a = np.arange(5)
        b = np.array([3.2, 1.4, 3])
        ag = cp.asarray(a)
        bg = cp.asarray(b)
        for mode in ["full", "valid", "same"]:
            assert_array_almost_equal(
                np.convolve(a, b, mode=mode), convolve(ag, bg, mode=mode)
            )
            assert_array_almost_equal(
                cp.squeeze(
                    convolve2d(
                        cp.asarray(np.array([a])),
                        cp.asarray(np.array([b])),
                        mode=mode,
                    )
                ),
                convolve(ag, bg, mode=mode),
            )

    def test_invalid_dims(self):
        with pytest.raises(ValueError):
            convolve2d(3, 4)
        with pytest.raises(ValueError):
            convolve2d([3], [4])
        with pytest.raises(ValueError):
            convolve2d([[[3]]], [[[4]]])


class TestFFTConvolve(object):
    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_real(self, axes):
        a = cp.asarray([1, 2, 3])
        expected = cp.asarray([1, 4, 10, 12, 9.0])

        if axes == "":
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [1, [1], -1, [-1]])
    def test_real_axes(self, axes):
        a = cp.asarray([1, 2, 3])
        expected = cp.asarray([1, 4, 10, 12, 9.0])

        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_complex(self, axes):
        a = cp.asarray([1 + 1j, 2 + 2j, 3 + 3j])
        expected = cp.asarray([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        if axes == "":
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [1, [1], -1, [-1]])
    def test_complex_axes(self, axes):
        a = cp.asarray([1 + 1j, 2 + 2j, 3 + 3j])
        expected = cp.asarray([0 + 2j, 0 + 8j, 0 + 20j, 0 + 24j, 0 + 18j])

        a = np.tile(a, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            "",
            None,
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
            [-2, 1],
            [1, -2],
            [-2, -1],
            [-1, -2],
        ],
    )
    def test_2d_real_same(self, axes):
        # fmt: off
        a = cp.asarray([[1, 2, 3],
                        [4, 5, 6]])
        expected = cp.asarray([[1, 4, 10, 12, 9],
                               [8, 26, 56, 54, 36],
                               [16, 40, 73, 60, 36]])
        # fmt: on
        if axes == "":
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [1, 2],
            [2, 1],
            [1, -1],
            [-1, 1],
            [-2, 2],
            [2, -2],
            [-2, -1],
            [-1, -2],
        ],
    )
    def test_2d_real_same_axes(self, axes):
        # fmt: off
        a = cp.asarray([[1, 2, 3],
                        [4, 5, 6]])
        expected = cp.asarray([[1, 4, 10, 12, 9],
                               [8, 26, 56, 54, 36],
                               [16, 40, 73, 60, 36]])
        # fmt: on
        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            "",
            None,
            [0, 1],
            [1, 0],
            [0, -1],
            [-1, 0],
            [-2, 1],
            [1, -2],
            [-2, -1],
            [-1, -2],
        ],
    )
    def test_2d_complex_same(self, axes):
        # fmt: off
        a = cp.asarray(
            [
                [1 + 2j, 3 + 4j, 5 + 6j],
                [2 + 1j, 4 + 3j, 6 + 5j]
            ]
        )
        # fmt: on
        expected = cp.asarray(
            [
                [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
                [10j, 44j, 118j, 156j, 122j],
                [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j],
            ]
        )

        if axes == "":
            out = fftconvolve(a, a)
        else:
            out = fftconvolve(a, a, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "axes",
        [
            [1, 2],
            [2, 1],
            [1, -1],
            [-1, 1],
            [-2, 2],
            [2, -2],
            [-2, -1],
            [-1, -2],
        ],
    )
    def test_2d_complex_same_axes(self, axes):
        # fmt: off
        a = cp.asarray(
            [
                [1 + 2j, 3 + 4j, 5 + 6j],
                [2 + 1j, 4 + 3j, 6 + 5j],
            ]
        )
        # fmt: on
        expected = cp.asarray(
            [
                [-3 + 4j, -10 + 20j, -21 + 56j, -18 + 76j, -11 + 60j],
                [10j, 44j, 118j, 156j, 122j],
                [3 + 4j, 10 + 20j, 21 + 56j, 18 + 76j, 11 + 60j],
            ]
        )

        a = np.tile(a, [2, 1, 1])
        expected = np.tile(expected, [2, 1, 1])

        out = fftconvolve(a, a, axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_real_same_mode(self, axes):
        a = cp.asarray([1, 2, 3])
        b = cp.asarray([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = cp.asarray([35.0, 41.0, 47.0])
        expected_2 = cp.asarray(
            [9.0, 20.0, 25.0, 35.0, 41.0, 47.0, 39.0, 28.0, 2.0]
        )

        if axes == "":
            out = fftconvolve(a, b, "same")
        else:
            out = fftconvolve(a, b, "same", axes=axes)
        assert_array_almost_equal(out, expected_1)

        if axes == "":
            out = fftconvolve(b, a, "same")
        else:
            out = fftconvolve(b, a, "same", axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize("axes", [1, -1, [1], [-1]])
    def test_real_same_mode_axes(self, axes):
        a = cp.asarray([1, 2, 3])
        b = cp.asarray([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected_1 = cp.asarray([35.0, 41.0, 47.0])
        expected_2 = cp.asarray(
            [9.0, 20.0, 25.0, 35.0, 41.0, 47.0, 39.0, 28.0, 2.0]
        )

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected_1 = np.tile(expected_1, [2, 1])
        expected_2 = np.tile(expected_2, [2, 1])

        out = fftconvolve(a, b, "same", axes=axes)
        assert_array_almost_equal(out, expected_1)

        out = fftconvolve(b, a, "same", axes=axes)
        assert_array_almost_equal(out, expected_2)

    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_valid_mode_real(self, axes):
        # See gh-5897
        a = cp.asarray([3, 2, 1])
        b = cp.asarray([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = cp.asarray([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])

        if axes == "":
            out = fftconvolve(a, b, "valid")
        else:
            out = fftconvolve(a, b, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == "":
            out = fftconvolve(b, a, "valid")
        else:
            out = fftconvolve(b, a, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [1, [1]])
    def test_valid_mode_real_axes(self, axes):
        # See gh-5897
        a = cp.asarray([3, 2, 1])
        b = cp.asarray([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = cp.asarray([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_valid_mode_complex(self, axes):
        a = cp.asarray([3 - 1j, 2 + 7j, 1 + 0j])
        b = cp.asarray([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = cp.asarray([45.0 + 12.0j, 30.0 + 23.0j, 48 + 32.0j])

        if axes == "":
            out = fftconvolve(a, b, "valid")
        else:
            out = fftconvolve(a, b, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

        if axes == "":
            out = fftconvolve(b, a, "valid")
        else:
            out = fftconvolve(b, a, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [1, [1], -1, [-1]])
    def test_valid_mode_complex_axes(self, axes):
        a = cp.asarray([3 - 1j, 2 + 7j, 1 + 0j])
        b = cp.asarray([3 + 2j, 3 - 3j, 5 + 0j, 6 - 1j, 8 + 0j])
        expected = cp.asarray([45.0 + 12.0j, 30.0 + 23.0j, 48 + 32.0j])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

        out = fftconvolve(b, a, "valid", axes=axes)
        assert_array_almost_equal(out, expected)

    def test_valid_mode_ignore_nonaxes(self):
        # See gh-5897
        a = cp.asarray([3, 2, 1])
        b = cp.asarray([3, 3, 5, 6, 8, 7, 9, 0, 1])
        expected = cp.asarray([24.0, 31.0, 41.0, 43.0, 49.0, 25.0, 12.0])

        a = np.tile(a, [2, 1])
        b = np.tile(b, [1, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, "valid", axes=1)
        assert_array_almost_equal(out, expected)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert fftconvolve([], []).size == 0
        assert fftconvolve([5, 6], []).size == 0
        assert fftconvolve([], [7]).size == 0

    def test_zero_rank(self):
        a = cp.asarray(4967)
        b = cp.asarray(3920)
        out = fftconvolve(a, b)
        assert_array_equal(out, a * b)

    def test_single_element(self):
        a = cp.asarray([4967])
        b = cp.asarray([3920])
        out = fftconvolve(a, b)
        assert_array_equal(out, a * b)

    @pytest.mark.parametrize("axes", ["", None, 0, [0], -1, [-1]])
    def test_random_data(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, "full")

        if axes == "":
            out = fftconvolve(a, b, "full")
        else:
            out = fftconvolve(a, b, "full", axes=axes)
        assert cp.allclose(out, expected, rtol=1e-10)

    @pytest.mark.parametrize("axes", [1, [1], -1, [-1]])
    def test_random_data_axes(self, axes):
        np.random.seed(1234)
        a = np.random.rand(1233) + 1j * np.random.rand(1233)
        b = np.random.rand(1321) + 1j * np.random.rand(1321)
        expected = np.convolve(a, b, "full")

        a = np.tile(a, [2, 1])
        b = np.tile(b, [2, 1])
        expected = np.tile(expected, [2, 1])

        out = fftconvolve(a, b, "full", axes=axes)
        assert cp.allclose(out, expected, rtol=1e-10)

    @pytest.mark.parametrize(
        "axes",
        [
            [1, 4],
            [4, 1],
            [1, -1],
            [-1, 1],
            [-4, 4],
            [4, -4],
            [-4, -1],
            [-1, -4],
        ],
    )
    def test_random_data_multidim_axes(self, axes):
        a_shape, b_shape = (123, 22), (132, 11)
        np.random.seed(1234)
        a = np.random.rand(*a_shape) + 1j * np.random.rand(*a_shape)
        b = np.random.rand(*b_shape) + 1j * np.random.rand(*b_shape)
        expected = convolve2d(a, b, "full")

        a = a[:, :, None, None, None]
        b = b[:, :, None, None, None]
        expected = expected[:, :, None, None, None]

        a = np.rollaxis(a.swapaxes(0, 2), 1, 5)
        b = np.rollaxis(b.swapaxes(0, 2), 1, 5)
        expected = np.rollaxis(expected.swapaxes(0, 2), 1, 5)

        # use 1 for dimension 2 in a and 3 in b to test broadcasting
        a = np.tile(a, [2, 1, 3, 1, 1])
        b = np.tile(b, [2, 1, 1, 4, 1])
        expected = np.tile(expected, [2, 1, 3, 4, 1])

        out = fftconvolve(a, b, "full", axes=axes)
        assert_allclose(out, expected, rtol=1e-10, atol=1e-10)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "n",
        list(range(1, 100))
        + list(range(1000, 1500))
        + np.random.RandomState(1234).randint(1001, 10000, 5).tolist(),
    )
    def test_many_sizes(self, n):
        a = np.random.rand(n) + 1j * np.random.rand(n)
        b = np.random.rand(n) + 1j * np.random.rand(n)
        expected = np.convolve(a, b, "full")

        out = fftconvolve(a, b, "full")
        assert_allclose(out, expected, atol=1e-10)

        out = fftconvolve(a, b, "full", axes=[0])
        assert_allclose(out, expected, atol=1e-10)


def fftconvolve_err(*args, **kwargs):
    raise RuntimeError("Fell back to fftconvolve")


def gen_oa_shapes(sizes):
    return [
        (a, b) for a, b in itertools.product(sizes, repeat=2) if abs(a - b) > 3
    ]


def gen_oa_shapes_2d(sizes):
    shapes0 = gen_oa_shapes(sizes)
    shapes1 = gen_oa_shapes(sizes)
    shapes = [
        ishapes0 + ishapes1 for ishapes0, ishapes1 in zip(shapes0, shapes1)
    ]

    modes = ["full", "valid", "same"]
    return [
        ishapes + (imode,)
        for ishapes, imode in itertools.product(shapes, modes)
        if imode != "valid"
        or (ishapes[0] > ishapes[1] and ishapes[2] > ishapes[3])
        or (ishapes[0] < ishapes[1] and ishapes[2] < ishapes[3])
    ]


def gen_oa_shapes_eq(sizes):
    return [(a, b) for a, b in itertools.product(sizes, repeat=2) if a >= b]


class TestOAConvolve(object):
    @pytest.mark.slow()
    @pytest.mark.parametrize(
        "shape_a_0, shape_b_0",
        gen_oa_shapes_eq(list(range(100)) + list(range(100, 1000, 23))),
    )
    def test_real_manylens(self, shape_a_0, shape_b_0):
        a = cp.random.rand(shape_a_0)
        b = cp.random.rand(shape_b_0)

        expected = fftconvolve(a, b)
        out = oaconvolve(a, b)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "shape_a_0, shape_b_0", gen_oa_shapes([50, 47, 6, 4])
    )
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_1d_noaxes(
        self, shape_a_0, shape_b_0, is_complex, mode, monkeypatch
    ):
        a = cp.random.rand(shape_a_0)
        b = cp.random.rand(shape_b_0)
        if is_complex:
            a = a + 1j * cp.random.rand(shape_a_0)
            b = b + 1j * cp.random.rand(shape_b_0)

        expected = fftconvolve(a, b, mode=mode)

        monkeypatch.setattr(
            signal_cp.signaltools, "fftconvolve", fftconvolve_err
        )
        out = oaconvolve(a, b, mode=mode)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [0, 1])
    @pytest.mark.parametrize(
        "shape_a_0, shape_b_0", gen_oa_shapes([50, 47, 6, 4])
    )
    @pytest.mark.parametrize("shape_a_extra", [1, 3])
    @pytest.mark.parametrize("shape_b_extra", [1, 3])
    @pytest.mark.parametrize("is_complex", [True, False])
    @pytest.mark.parametrize("mode", ["full", "valid", "same"])
    def test_1d_axes(
        self,
        axes,
        shape_a_0,
        shape_b_0,
        shape_a_extra,
        shape_b_extra,
        is_complex,
        mode,
        monkeypatch,
    ):
        ax_a = [shape_a_extra] * 2
        ax_b = [shape_b_extra] * 2
        ax_a[axes] = shape_a_0
        ax_b[axes] = shape_b_0

        a = cp.random.rand(*ax_a)
        b = cp.random.rand(*ax_b)
        if is_complex:
            a = a + 1j * cp.random.rand(*ax_a)
            b = b + 1j * cp.random.rand(*ax_b)

        expected = fftconvolve(a, b, mode=mode, axes=axes)

        monkeypatch.setattr(
            signal_cp.signaltools, "fftconvolve", fftconvolve_err
        )
        out = oaconvolve(a, b, mode=mode, axes=axes)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize(
        "shape_a_0, shape_b_0, " "shape_a_1, shape_b_1, mode",
        gen_oa_shapes_2d([50, 47, 6, 4]),
    )
    @pytest.mark.parametrize("is_complex", [True, False])
    def test_2d_noaxes(
        self,
        shape_a_0,
        shape_b_0,
        shape_a_1,
        shape_b_1,
        mode,
        is_complex,
        monkeypatch,
    ):
        a = cp.random.rand(shape_a_0, shape_a_1)
        b = cp.random.rand(shape_b_0, shape_b_1)
        if is_complex:
            a = a + 1j * cp.random.rand(shape_a_0, shape_a_1)
            b = b + 1j * cp.random.rand(shape_b_0, shape_b_1)

        expected = fftconvolve(a, b, mode=mode)

        monkeypatch.setattr(
            signal_cp.signaltools, "fftconvolve", fftconvolve_err
        )
        out = oaconvolve(a, b, mode=mode)

        assert_array_almost_equal(out, expected)

    @pytest.mark.parametrize("axes", [[0, 1], [0, 2], [1, 2]])
    @pytest.mark.parametrize(
        "shape_a_0, shape_b_0, " "shape_a_1, shape_b_1, mode",
        gen_oa_shapes_2d([50, 47, 6, 4]),
    )
    @pytest.mark.parametrize("shape_a_extra", [1, 3])
    @pytest.mark.parametrize("shape_b_extra", [1, 3])
    @pytest.mark.parametrize("is_complex", [True, False])
    def test_2d_axes(
        self,
        axes,
        shape_a_0,
        shape_b_0,
        shape_a_1,
        shape_b_1,
        mode,
        shape_a_extra,
        shape_b_extra,
        is_complex,
        monkeypatch,
    ):
        ax_a = [shape_a_extra] * 3
        ax_b = [shape_b_extra] * 3
        ax_a[axes[0]] = shape_a_0
        ax_b[axes[0]] = shape_b_0
        ax_a[axes[1]] = shape_a_1
        ax_b[axes[1]] = shape_b_1

        a = cp.random.rand(*ax_a)
        b = cp.random.rand(*ax_b)
        if is_complex:
            a = a + 1j * cp.random.rand(*ax_a)
            b = b + 1j * cp.random.rand(*ax_b)

        expected = fftconvolve(a, b, mode=mode, axes=axes)

        monkeypatch.setattr(
            signal_cp.signaltools, "fftconvolve", fftconvolve_err
        )
        out = oaconvolve(a, b, mode=mode, axes=axes)

        assert_array_almost_equal(out, expected)

    def test_empty(self):
        # Regression test for #1745: crashes with 0-length input.
        assert oaconvolve(cp.asarray([]), cp.asarray([])).size == 0
        assert oaconvolve(cp.asarray([5, 6]), cp.asarray([])).size == 0
        assert oaconvolve(cp.asarray([]), cp.asarray([7])).size == 0

    def test_zero_rank(self):
        a = cp.asarray(4967)
        b = cp.asarray(3920)
        out = oaconvolve(a, b)
        assert_equal(out, a * b)

    def test_single_element(self):
        a = cp.asarray([4967])
        b = cp.asarray([3920])
        out = oaconvolve(a, b)
        assert_equal(out, a * b)


class TestAllFreqConvolves(object):
    @pytest.mark.parametrize("convapproach", [fftconvolve, oaconvolve])
    def test_invalid_shapes(self, convapproach):
        a = cp.arange(1, 7).reshape((2, 3))
        b = cp.arange(-6, 0).reshape((3, 2))
        with pytest.raises(
            ValueError,
            match="For 'valid' mode, one must be at least "
            "as large as the other in every dimension",
        ):
            convapproach(a, b, mode="valid")

    @pytest.mark.parametrize("convapproach", [fftconvolve, oaconvolve])
    def test_invalid_shapes_axes(self, convapproach):
        a = cp.zeros([5, 6, 2, 1])
        b = cp.zeros([5, 6, 3, 1])
        with pytest.raises(
            ValueError,
            match=r"incompatible shapes for in1 and in2:"
            r" \(5L?, 6L?, 2L?, 1L?\) and"
            r" \(5L?, 6L?, 3L?, 1L?\)",
        ):
            convapproach(a, b, axes=[0, 1])

    @pytest.mark.parametrize("a,b", [([1], 2), (1, [2]), ([3], [[2]])])
    @pytest.mark.parametrize("convapproach", [fftconvolve, oaconvolve])
    def test_mismatched_dims(self, a, b, convapproach):
        with pytest.raises(
            ValueError,
            match="in1 and in2 should have the same" " dimensionality",
        ):
            convapproach(cp.asarray(a), cp.asarray(b))

    @pytest.mark.parametrize("convapproach", [fftconvolve, oaconvolve])
    def test_invalid_flags(self, convapproach):
        with pytest.raises(
            ValueError,
            match="acceptable mode flags are 'valid'," " 'same', or 'full'",
        ):
            convapproach(cp.asarray([1]), cp.asarray([2]), mode="chips")

        with pytest.raises(
            ValueError, match="when provided, axes cannot be empty"
        ):
            convapproach(cp.asarray([1]), cp.asarray([2]), axes=[])

        with pytest.raises(
            ValueError, match="axes must be a scalar or " "iterable of integers"
        ):
            convapproach(
                cp.asarray([1]), cp.asarray([2]), axes=[[1, 2], [3, 4]]
            )

        with pytest.raises(
            ValueError, match="axes must be a scalar or " "iterable of integers"
        ):
            convapproach(
                cp.asarray([1]), cp.asarray([2]), axes=[1.0, 2.0, 3.0, 4.0]
            )

        with pytest.raises(
            ValueError, match="axes exceeds dimensionality of input"
        ):
            convapproach(cp.asarray([1]), cp.asarray([2]), axes=[1])

        with pytest.raises(
            ValueError, match="axes exceeds dimensionality of input"
        ):
            convapproach(cp.asarray([1]), cp.asarray([2]), axes=[-2])

        with pytest.raises(ValueError, match="all axes must be unique"):
            convapproach(cp.asarray([1]), cp.asarray([2]), axes=[0, 0])

    # # grlee77: don't support longfloat or longcomplex
    # @pytest.mark.parametrize('dtype', [np.longfloat, np.longcomplex])
    # def test_longdtype_input(self, dtype):
    #     x = np.random.random((27, 27)).astype(dtype)
    #     y = np.random.random((4, 4)).astype(dtype)
    #     if np.iscomplexobj(dtype()):
    #         x += .1j
    #         y -= .1j

    #     res = fftconvolve(x, y)
    #     assert_allclose(res, convolve(x, y, method='direct'))
    #     assert res.dtype == dtype


class TestWiener(object):
    def test_basic(self):
        # fmt: off
        g = cp.asarray(
            [
                [5, 6, 4, 3],
                [3, 5, 6, 2],
                [2, 3, 5, 6],
                [1, 6, 9, 7],
            ],
            'd'
        )
        # fmt: on
        h = cp.asarray(
            [
                [2.16374269, 3.2222222222, 2.8888888889, 1.6666666667],
                [2.666666667, 4.33333333333, 4.44444444444, 2.8888888888],
                [2.222222222, 4.4444444444, 5.4444444444, 4.801066874837],
                [1.33333333333, 3.92735042735, 6.0712560386, 5.0404040404],
            ]
        )
        assert_array_almost_equal(wiener(g), h, decimal=6)
        assert_array_almost_equal(wiener(g, mysize=3), h, decimal=6)


padtype_options = [
    "mean",
    "minimum",
    "maximum",
]  # , "median", "line"]  #TODO: grlee77: support maximum, line
padtype_options += upfirdn_modes


class TestResample(object):
    """Note: The tests in this class are adapted from scipy.signal's tests"""

    def test_basic(self):
        # Some basic tests

        # Regression test for issue #3603.
        # window.shape must equal to sig.shape[0]
        sig = cp.arange(128)
        num = 256
        win = cp.asarray(signal.get_window(("kaiser", 8.0), 160))
        with pytest.raises(ValueError):
            resample(sig, num, window=win)

        # Other degenerate conditions
        with pytest.raises(ValueError):
            resample_poly(sig, "yo", 1)
        with pytest.raises(ValueError):
            resample_poly(sig, 1, 0)
        with pytest.raises(ValueError):
            resample_poly(sig, 2, 1, padtype="")
        with pytest.raises(ValueError):
            resample_poly(sig, 2, 1, padtype="mean", cval=10)

        # test for issue #6505 - should not modify window.shape when axis ≠ 0
        sig2 = cp.tile(cp.arange(160), (2, 1))
        resample(sig2, num, axis=-1, window=win)
        assert win.shape == (160,)

    @pytest.mark.parametrize("window", (None, "hamming"))
    @pytest.mark.parametrize("N", (20, 19))
    @pytest.mark.parametrize("num", (100, 101, 10, 11))
    def test_rfft(self, N, num, window):
        # Make sure the speed up using rfft gives the same result as the normal
        # way using fft
        x = np.linspace(0, 10, N, endpoint=False)
        y = np.cos(-(x ** 2) / 6.0)
        y = cp.asarray(y)
        assert_allclose(
            resample(y, num, window=window),
            resample(y + 0j, num, window=window).real,
        )

        y = cp.asarray([np.cos(-(x ** 2) / 6.0), np.sin(-(x ** 2) / 6.0)])
        y_complex = y + 0j
        assert_allclose(
            resample(y, num, axis=1, window=window),
            resample(y_complex, num, axis=1, window=window).real,
            atol=1e-9,
        )

    @pytest.mark.parametrize("nx", (1, 2, 3, 5, 8))
    @pytest.mark.parametrize("ny", (1, 2, 3, 5, 8))
    @pytest.mark.parametrize("dtype", ("float", "complex"))
    def test_dc(self, nx, ny, dtype):
        x = cp.asarray([1] * nx, dtype)
        y = resample(x, ny)
        assert_allclose(y, cp.asarray([1] * ny))

    @pytest.mark.parametrize("padtype", padtype_options)
    def test_mutable_window(self, padtype):
        # Test that a mutable window is not modified
        impulse = cp.zeros(3)
        window = cp.random.RandomState(0).randn(2)
        window_orig = window.copy()
        resample_poly(impulse, 5, 1, window=window, padtype=padtype)

        cp.testing.assert_array_equal(window, window_orig)

    @pytest.mark.parametrize("padtype", padtype_options)
    def test_output_float32(self, padtype):
        # Test that float32 inputs yield a float32 output
        x = cp.arange(10, dtype=np.float32)
        h = cp.array([1, 1, 1], dtype=np.float32)
        y = resample_poly(x, 1, 2, window=h, padtype=padtype)
        assert y.dtype == np.float32

    @pytest.mark.parametrize(
        "method, ext, padtype",
        [("fft", False, None)]
        + list(
            itertools.product(["polyphase"], [False, True], padtype_options)
        ),
    )
    def test_resample_methods(self, method, ext, padtype):
        hann = signal.windows.hann

        # Test resampling of sinusoids
        rate = 100
        rates_to = [49, 50, 51, 99, 100, 101, 199, 200, 201]

        # Sinusoids, windowed to avoid edge artifacts
        t = np.arange(rate) / float(rate)
        freqs = np.array((1.0, 10.0, 40.0))[:, np.newaxis]
        x = np.sin(2 * np.pi * freqs * t) * hann(rate)

        for rate_to in rates_to:
            t_to = np.arange(rate_to) / float(rate_to)
            y_tos = np.sin(2 * np.pi * freqs * t_to) * hann(rate_to)
            if method == "fft":
                y_resamps = resample(cp.asarray(x), rate_to, axis=-1).get()
            else:
                if ext and rate_to != rate:
                    # Match default window design
                    g = math.gcd(rate_to, rate)
                    up = rate_to // g
                    down = rate // g
                    max_rate = max(up, down)
                    f_c = 1.0 / max_rate
                    half_len = 10 * max_rate
                    window = signal.firwin(
                        2 * half_len + 1, f_c, window=("kaiser", 5.0)
                    )
                    window = cp.asarray(window)
                    polyargs = {"window": window, "padtype": padtype}
                else:
                    polyargs = {"padtype": padtype}

                x = cp.asarray(x)
                y_resamps = resample_poly(x, rate_to, rate, axis=-1, **polyargs)
                y_resamps = y_resamps.get()

                for y_to, y_resamp, freq in zip(y_tos, y_resamps, freqs):
                    if freq >= 0.5 * rate_to:
                        y_to.fill(0.0)  # mostly low-passed away
                        if padtype in ["minimum", "maximum"]:
                            np.testing.assert_allclose(
                                y_resamp, y_to, atol=3e-1
                            )
                        else:
                            np.testing.assert_allclose(
                                y_resamp, y_to, atol=1e-3
                            )
                    else:
                        np.testing.assert_array_equal(
                            y_to.shape, y_resamp.shape
                        )
                        corr = np.corrcoef(y_to, y_resamp)[0, 1]
                        assert corr > 0.99
        # Random data
        rng = np.random.RandomState(0)
        x = hann(rate) * np.cumsum(rng.randn(rate))  # low-pass, wind
        xg = cp.asarray(x)
        for rate_to in rates_to:
            # random data
            t_to = np.arange(rate_to) / float(rate_to)
            y_to = np.interp(t_to, t, x)
            if method == "fft":
                y_resamp = resample(xg, rate_to)
            else:
                y_resamp = resample_poly(xg, rate_to, rate, padtype=padtype)
            y_resamp = y_resamp.get()
            assert_array_equal(y_to.shape, y_resamp.shape)
            corr = np.corrcoef(y_to, y_resamp)[0, 1]
            assert_(corr > 0.99, msg=corr)

        # More tests of fft method (Master 0.18.1 fails these)
        if method == "fft":
            x1 = cp.asarray([1.0 + 0.0j, 0.0 + 0.0j])
            y1_test = resample(x1, 4)
            # upsampling a complex array
            y1_true = np.asarray(
                [1.0 + 0.0j, 0.5 + 0.0j, 0.0 + 0.0j, 0.5 + 0.0j]
            )
            assert_allclose(y1_test, y1_true, atol=1e-12)
            x2 = np.asarray([1.0, 0.5, 0.0, 0.5])
            y2_test = resample(x2, 2)  # downsampling a real array
            y2_true = np.asarray([1.0, 0.0])
            assert_allclose(y2_test, y2_true, atol=1e-12)

    def test_poly_vs_filtfilt(self):
        # Check that up=1.0 gives same answer as filtfilt + slicing
        random_state = np.random.RandomState(17)
        try_types = (int, np.float32, np.complex64, float, complex)
        size = 10000
        down_factors = [2, 11, 79]

        for dtype in try_types:
            x = random_state.randn(size).astype(dtype)
            if dtype in (np.complex64, np.complex128):
                x += 1j * random_state.randn(size)

            if x.real.dtype.char == "f":
                atol = rtol = 1e-6  # Note: scipy's tests use 1e-7
            else:
                atol = rtol = 1e-13

            # resample_poly assumes zeros outside of signl, whereas filtfilt
            # can only constant-pad. Make them equivalent:
            x[0] = 0
            x[-1] = 0

            for down in down_factors:
                h = signal.firwin(31, 1.0 / down, window="hamming")
                yf = signal.filtfilt(h, 1.0, x, padtype="constant")[::down]

                # Need to pass convolved version of filter to resample_poly,
                # since filtfilt does forward and backward, but resample_poly
                # only goes forward
                hc = signal.convolve(h, h[::-1])
                x_gpu, hc = cp.asarray(x), cp.asarray(hc)
                y = resample_poly(x_gpu, 1, down, window=hc).get()
                np.testing.assert_allclose(yf, y, atol=atol, rtol=rtol)

    def test_correlate1d(self):
        for down in [2, 4]:
            for nx in range(1, 40, down):
                for nweights in (32, 33):
                    x = np.random.random((nx,))
                    weights = np.random.random((nweights,))
                    x, weights = cp.asarray(x), cp.asarray(weights)
                    y_g = correlate1d(x, weights[::-1], mode="constant")
                    y_s = resample_poly(x, up=1, down=down, window=weights)
                    cp.testing.assert_allclose(y_g[::down], y_s)


# grlee77: Note: omit Decimal, np.longdouble, np.ulonglong, np.ulonglong
@pytest.mark.parametrize(
    "dt",
    [
        np.float32,
        np.float64,
        np.ubyte,
        np.byte,
        np.ushort,
        np.short,
        np.uint,
        int,
    ],
)
class TestCorrelateReal(object):
    def _setup_rank1(self, dt):
        a = cp.linspace(0, 3, 4).astype(dt)
        b = cp.linspace(1, 2, 2).astype(dt)

        y_r = cp.asarray([0, 2, 5, 8, 3]).astype(dt)
        return a, b, y_r

    def equal_tolerance(self, res_dt):
        # default value of keyword
        decimal = 6
        try:
            dt_info = np.finfo(res_dt)
            if hasattr(dt_info, "resolution"):
                decimal = int(-0.5 * np.log10(dt_info.resolution))
        except Exception:
            pass
        return decimal

    def equal_tolerance_fft(self, res_dt):
        # FFT implementations convert longdouble arguments down to
        # double so don't expect better precision, see gh-9520
        if res_dt == np.longdouble:
            return self.equal_tolerance(np.double)
        else:
            return self.equal_tolerance(res_dt)

    def test_method(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y_fft = correlate(a, b, method="fft")
        y_direct = correlate(a, b, method="direct")

        assert_array_almost_equal(
            y_r, y_fft, decimal=self.equal_tolerance_fft(y_fft.dtype)
        )
        assert_array_almost_equal(
            y_r, y_direct, decimal=self.equal_tolerance(y_direct.dtype)
        )
        assert_equal(y_fft.dtype, dt)
        assert_equal(y_direct.dtype, dt)

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, "valid")
        assert_array_almost_equal(y, y_r[1:4])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, "valid")
        assert_array_almost_equal(y, y_r[1:4][::-1])
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, "same")
        assert_array_almost_equal(y, y_r[:-1])
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt)
        y = correlate(a, b, "full")
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)

    def _setup_rank3(self, dt):
        a = np.linspace(0, 39, 40).reshape((2, 4, 5), order="F").astype(dt)
        b = np.linspace(0, 23, 24).reshape((2, 3, 4), order="F").astype(dt)
        # fmt: off
        y_r = cp.asarray(
            [[[0., 184., 504., 912., 1360., 888., 472., 160.],
              [46., 432., 1062., 1840., 2672., 1698., 864., 266.],
              [134., 736., 1662., 2768., 3920., 2418., 1168., 314.],
              [260., 952., 1932., 3056., 4208., 2580., 1240., 332.],
              [202., 664., 1290., 1984., 2688., 1590., 712., 150.],
              [114., 344., 642., 960., 1280., 726., 296., 38.]],

             [[23., 400., 1035., 1832., 2696., 1737., 904., 293.],
              [134., 920., 2166., 3680., 5280., 3306., 1640., 474.],
              [325., 1544., 3369., 5512., 7720., 4683., 2192., 535.],
              [571., 1964., 3891., 6064., 8272., 4989., 2324., 565.],
              [434., 1360., 2586., 3920., 5264., 3054., 1312., 230.],
              [241., 700., 1281., 1888., 2496., 1383., 532., 39.]],

             [[22., 214., 528., 916., 1332., 846., 430., 132.],
              [86., 484., 1098., 1832., 2600., 1602., 772., 206.],
              [188., 802., 1698., 2732., 3788., 2256., 1018., 218.],
              [308., 1006., 1950., 2996., 4052., 2400., 1078., 230.],
              [230., 692., 1290., 1928., 2568., 1458., 596., 78.],
              [126., 354., 636., 924., 1212., 654., 234., 0.]],
             ],
            dtype=dt
        )
        # fmt: on
        return a, b, y_r

    def test_rank3_valid(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5])
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, "valid")
        assert_array_almost_equal(y, y_r[1:2, 2:4, 3:5][::-1, ::-1, ::-1])
        assert_equal(y.dtype, dt)

    def test_rank3_same(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b, "same")
        assert_array_almost_equal(y, y_r[0:-1, 1:-1, 1:-2])
        assert_equal(y.dtype, dt)

    def test_rank3_all(self, dt):
        a, b, y_r = self._setup_rank3(dt)
        y = correlate(a, b)
        assert_array_almost_equal(y, y_r)
        assert_equal(y.dtype, dt)


class TestCorrelate(object):
    # Tests that don't depend on dtype

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = cp.arange(1, 7).reshape((2, 3))
        b = cp.arange(-6, 0).reshape((3, 2))

        with pytest.raises(ValueError):
            correlate(a, b, mode="valid")
        with pytest.raises(ValueError):
            correlate(b, a, mode="valid")

    def test_invalid_params(self):
        a = cp.asarray([3, 4, 5])
        b = cp.asarray([1, 2, 3])
        with pytest.raises(ValueError):
            correlate(a, b, mode="spam")
        with pytest.raises(ValueError):
            correlate(a, b, mode="eggs", method="fft")
        with pytest.raises(ValueError):
            correlate(a, b, mode="ham", method="direct")
        with pytest.raises(ValueError):
            correlate(a, b, mode="full", method="bacon")
        with pytest.raises(ValueError):
            correlate(a, b, mode="same", method="bacon")

    def test_mismatched_dims(self):
        # Input arrays should have the same number of dimensions
        with pytest.raises(ValueError):
            correlate([1], 2, method="direct")
        with pytest.raises(ValueError):
            correlate(1, [2], method="direct")
        with pytest.raises(ValueError):
            correlate([1], 2, method="fft")
        with pytest.raises(ValueError):
            correlate(1, [2], method="fft")
        with pytest.raises(ValueError):
            correlate([1], [[2]])
        with pytest.raises(ValueError):
            correlate([3], 2)

    def test_numpy_fastpath(self):
        a = cp.asarray([1, 2, 3])
        b = cp.asarray([4, 5])
        assert_allclose(correlate(a, b, mode="same"), cp.asarray([5, 14, 23]))

        a = cp.asarray([1, 2, 3])
        b = cp.asarray([4, 5, 6])
        assert_allclose(correlate(a, b, mode="same"), cp.asarray([17, 32, 23]))
        assert_allclose(
            correlate(a, b, mode="full"), cp.asarray([6, 17, 32, 23, 12])
        )
        assert_allclose(correlate(a, b, mode="valid"), cp.asarray([32]))


# grlee77: omit np.clongdouble
@pytest.mark.parametrize("dt", [np.csingle, np.cdouble])
class TestCorrelateComplex(object):
    # The decimal precision to be used for comparing results.
    # This value will be passed as the 'decimal' keyword argument of
    # assert_array_almost_equal().
    # Since correlate may chose to use FFT method which converts
    # longdoubles to doubles internally don't expect better precision
    # for longdouble than for double (see gh-9520).

    def decimal(self, dt):
        if dt == np.clongdouble:
            dt = np.cdouble
        return int(2 * np.finfo(dt).precision / 3)

    def _setup_rank1(self, dt, mode):
        np.random.seed(9)
        a = np.random.randn(10).astype(dt)
        a += 1j * np.random.randn(10).astype(dt)
        b = np.random.randn(8).astype(dt)
        b += 1j * np.random.randn(8).astype(dt)
        a = cp.asarray(a)
        b = cp.asarray(b)

        y_r = (
            correlate(a.real, b.real, mode=mode)
            + correlate(a.imag, b.imag, mode=mode)
        ).astype(dt)
        y_r += 1j * (
            -correlate(a.real, b.imag, mode=mode)
            + correlate(a.imag, b.real, mode=mode)
        )
        return a, b, y_r

    def test_rank1_valid(self, dt):
        a, b, y_r = self._setup_rank1(dt, "valid")
        y = correlate(a, b, "valid")
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

        # See gh-5897
        y = correlate(b, a, "valid")
        assert_array_almost_equal(y, y_r[::-1].conj(), decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_same(self, dt):
        a, b, y_r = self._setup_rank1(dt, "same")
        y = correlate(a, b, "same")
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_rank1_full(self, dt):
        a, b, y_r = self._setup_rank1(dt, "full")
        y = correlate(a, b, "full")
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt))
        assert_equal(y.dtype, dt)

    def test_swap_full(self, dt):
        d = cp.asarray([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 2.0j], dtype=dt)
        k = cp.asarray(
            [1.0 + 3.0j, 2.0 + 4.0j, 3.0 + 5.0j, 4.0 + 6.0j], dtype=dt
        )
        y = correlate(d, k)
        assert_array_equal(
            y,
            cp.asarray(
                [
                    0.0 + 0.0j,
                    10.0 - 2.0j,
                    28.0 - 6.0j,
                    22.0 - 6.0j,
                    16.0 - 6.0j,
                    8.0 - 4.0j,
                ]
            ),
        )

    def test_swap_same(self, dt):
        d = cp.asarray([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 2.0j], dtype=dt)
        k = cp.asarray(
            [1.0 + 3.0j, 2.0 + 4.0j, 3.0 + 5.0j, 4.0 + 6.0j], dtype=dt
        )
        y = correlate(d, k, mode="same")
        assert_array_equal(
            y, cp.asarray([10.0 - 2.0j, 28.0 - 6.0j, 22.0 - 6.0j])
        )

    def test_rank3(self, dt):
        a = np.random.randn(10, 8, 6).astype(dt)
        a += 1j * np.random.randn(10, 8, 6).astype(dt)
        b = np.random.randn(8, 6, 4).astype(dt)
        b += 1j * np.random.randn(8, 6, 4).astype(dt)
        a = cp.asarray(a)
        b = cp.asarray(b)
        y_r = (correlate(a.real, b.real) + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, "full")
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)

    def test_rank0(self, dt):
        a = np.array(np.random.randn()).astype(dt)
        a += 1j * np.array(np.random.randn()).astype(dt)
        b = np.array(np.random.randn()).astype(dt)
        b += 1j * np.array(np.random.randn()).astype(dt)
        a = cp.asarray(a)
        b = cp.asarray(b)

        y_r = (correlate(a.real, b.real) + correlate(a.imag, b.imag)).astype(dt)
        y_r += 1j * (-correlate(a.real, b.imag) + correlate(a.imag, b.real))

        y = correlate(a, b, "full")
        assert_array_almost_equal(y, y_r, decimal=self.decimal(dt) - 1)
        assert_equal(y.dtype, dt)

        assert_equal(
            correlate(cp.asarray([1]), cp.asarray([2j])),
            correlate(cp.asarray(1), cp.asarray(2j)),
        )
        assert_equal(
            correlate(cp.asarray([2j]), cp.asarray([3j])),
            correlate(cp.asarray(2j), cp.asarray(3j)),
        )
        assert_equal(
            correlate(cp.asarray([3j]), cp.asarray([4])),
            correlate(cp.asarray(3j), cp.asarray(4)),
        )


class TestCorrelate2d(object):
    def test_consistency_correlate_funcs(self):
        # Compare np.correlate, correlate, correlate2d
        a = np.arange(5)
        b = np.array([3.2, 1.4, 3])
        ag = cp.asarray(a)
        bg = cp.asarray(b)
        for mode in ["full", "valid", "same"]:
            assert_array_almost_equal(
                np.correlate(a, b, mode=mode), correlate(ag, bg, mode=mode)
            )
            assert_array_almost_equal(
                cp.squeeze(
                    correlate2d(
                        cp.asarray(np.array([a])),
                        cp.asarray(np.array([b])),
                        mode=mode,
                    )
                ),
                correlate(a, b, mode=mode),
            )

        # See gh-5897
        if mode == "valid":
            assert_array_almost_equal(
                np.correlate(b, a, mode=mode), correlate(b, a, mode=mode)
            )
            assert_array_almost_equal(
                cp.squeeze(
                    correlate2d(
                        cp.asarray(np.array([b])),
                        cp.asarray(np.array([a])),
                        mode=mode,
                    )
                ),
                correlate(b, a, mode=mode),
            )

    def test_invalid_shapes(self):
        # By "invalid," we mean that no one
        # array has dimensions that are all at
        # least as large as the corresponding
        # dimensions of the other array. This
        # setup should throw a ValueError.
        a = cp.arange(1, 7).reshape((2, 3))
        b = cp.arange(-6, 0).reshape((3, 2))

        with pytest.raises(ValueError):
            correlate2d(a, b, mode="valid")
        with pytest.raises(ValueError):
            correlate2d(b, a, mode="valid")

    def test_complex_input(self):
        assert_array_equal(
            correlate2d(cp.asarray([[1]]), cp.asarray([[2j]])), cp.asarray(-2j)
        )
        assert_array_equal(
            correlate2d(cp.asarray([[2j]]), cp.asarray([[3j]])), cp.asarray(6)
        )
        assert_array_equal(
            correlate2d(cp.asarray([[3j]]), cp.asarray([[4]])), cp.asarray(12j)
        )


def test_choose_conv_method():
    for mode in ["valid", "same", "full"]:
        for ndim in [1, 2]:
            n, k, true_method = 8, 6, "direct"
            x = cp.random.randn(*((n,) * ndim))
            h = cp.random.randn(*((k,) * ndim))

            method = choose_conv_method(x, h, mode=mode)
            assert_equal(method, true_method)

            method_try, times = choose_conv_method(
                x, h, mode=mode, measure=True
            )
            assert_(method_try in {"fft", "direct"})
            assert_(type(times) is dict)
            assert_("fft" in times.keys() and "direct" in times.keys())

        x = cp.asarray([2 ** 51], dtype=np.int64)
        h = x.copy()
        assert_equal(choose_conv_method(x, h, mode=mode), "direct")


class TestHilbert(object):
    def test_bad_args(self):
        x = cp.array([1.0 + 0.0j])
        with pytest.raises(ValueError):
            hilbert(x)
        x = cp.arange(8.0)
        with pytest.raises(ValueError):
            hilbert(x, N=0)

    def test_hilbert_theoretical(self):
        # test cases by Ariel Rokem
        decimal = 14

        pi = np.pi
        t = cp.arange(0, 2 * pi, pi / 256)
        a0 = np.sin(t)
        a1 = np.cos(t)
        a2 = np.sin(2 * t)
        a3 = np.cos(2 * t)
        a = cp.asarray(np.vstack([a0, a1, a2, a3]))

        h = hilbert(a)
        h_abs = cp.abs(h)
        h_angle = cp.angle(h)
        h_real = cp.real(h)

        # The real part should be equal to the original signals:
        assert_array_almost_equal(h_real, a, decimal)
        # The absolute value should be one everywhere, for this input:
        assert_array_almost_equal(h_abs, np.ones(a.shape), decimal)
        # For the 'slow' sine - the phase should go from -pi/2 to pi/2 in
        # the first 256 bins:
        assert_array_almost_equal(
            h_angle[0, :256], cp.arange(-pi / 2, pi / 2, pi / 256), decimal
        )
        # For the 'slow' cosine - the phase should go from 0 to pi in the
        # same interval:
        assert_array_almost_equal(
            h_angle[1, :256], cp.arange(0, pi, pi / 256), decimal
        )
        # The 'fast' sine should make this phase transition in half the time:
        assert_array_almost_equal(
            h_angle[2, :128], cp.arange(-pi / 2, pi / 2, pi / 128), decimal
        )
        # Ditto for the 'fast' cosine:
        assert_array_almost_equal(
            h_angle[3, :128], cp.arange(0, pi, pi / 128), decimal
        )

        # The imaginary part of hilbert(cos(t)) = sin(t) Wikipedia
        assert_array_almost_equal(h[1].imag, a0, decimal)

    def test_hilbert_axisN(self):
        # tests for axis and N arguments
        a = cp.arange(18).reshape(3, 6)
        # test axis
        aa = hilbert(a, axis=-1)
        assert_array_equal(hilbert(a.T, axis=0), aa.T)
        # test 1d
        assert_array_almost_equal(hilbert(a[0]), aa[0], 14)

        # test N
        aan = hilbert(a, N=20, axis=-1)
        assert aan.shape == (3, 20)
        assert hilbert(a.T, N=20, axis=0).shape == (20, 3)
        # the next test is just a regression test,
        # no idea whether numbers make sense
        # fmt: off
        a0hilb = cp.asarray(
            [0.000000000000000e+00 - 1.72015830311905j,
             1.000000000000000e+00 - 2.047794505137069j,
             1.999999999999999e+00 - 2.244055555687583j,
             3.000000000000000e+00 - 1.262750302935009j,
             4.000000000000000e+00 - 1.066489252384493j,
             5.000000000000000e+00 + 2.918022706971047j,
             8.881784197001253e-17 + 3.845658908989067j,
            -9.444121133484362e-17 + 0.985044202202061j,
            -1.776356839400251e-16 + 1.332257797702019j,
            -3.996802888650564e-16 + 0.501905089898885j,
             1.332267629550188e-16 + 0.668696078880782j,
            -1.192678053963799e-16 + 0.235487067862679j,
            -1.776356839400251e-16 + 0.286439612812121j,
             3.108624468950438e-16 + 0.031676888064907j,
             1.332267629550188e-16 - 0.019275656884536j,
            -2.360035624836702e-16 - 0.1652588660287j,
             0.000000000000000e+00 - 0.332049855010597j,
             3.552713678800501e-16 - 0.403810179797771j,
             8.881784197001253e-17 - 0.751023775297729j,
             9.444121133484362e-17 - 0.79252210110103j]
        )
        # fmt: on
        assert_array_almost_equal(aan[0], a0hilb, 14, "N regression")


class TestHilbert2(object):
    def test_bad_args(self):
        # x must be real.
        x = cp.array([[1.0 + 0.0j]])
        with pytest.raises(ValueError):
            hilbert2(x)

        # x must be rank 2.
        x = cp.arange(24).reshape(2, 3, 4)
        with pytest.raises(ValueError):
            hilbert2(x)

        # Bad value for N.
        x = cp.arange(16).reshape(4, 4)
        with pytest.raises(ValueError):
            hilbert2(x, N=0)
        with pytest.raises(ValueError):
            hilbert2(x, N=(2, 0))
        with pytest.raises(ValueError):
            hilbert2(x, N=(2,))
