import unittest

import numpy as np
import scipy

import cupyimg.scipy.fft as cp_fft
from cupy import testing

try:
    import scipy.fft as sp_fft
except ImportError:
    import scipy.fftpack as sp_fft

if np.lib.NumpyVersion(scipy.__version__) < np.lib.NumpyVersion("1.5.0"):
    # used to avoid SciPy bug in complex dtype cases with output_x=True
    # https://github.com/scipy/scipy/pull/11904
    scipy_cplx_bug = True
else:
    scipy_cplx_bug = False


def _fft_module(xp):
    # Test cupyx.scipy against numpy since scipy.fft is not yet released
    if xp != np:
        return cp_fft
    else:
        return sp_fft


@testing.parameterize(
    *testing.product(
        {
            "n": [None, 0, 5, 10, 15],
            "type": [2, 3],
            "shape": [(9,), (10,), (10, 9), (10, 10)],
            "axis": [-1, 0],
            "norm": [None, "ortho"],
            "overwrite_x": [True, False],
        }
    )
)
@testing.gpu
class TestDctDst(unittest.TestCase):
    def _run_transform(self, dct_func, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if scipy_cplx_bug and x.dtype.kind == "c":
            # skip cases where SciPy has a bug
            return x
        x_orig = x.copy()
        out = dct_func(
            x,
            type=self.type,
            n=self.n,
            axis=self.axis,
            norm=self.norm,
            overwrite_x=self.overwrite_x,
        )
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_dct(self, xp, dtype):
        func = _fft_module(xp).dct
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_idct(self, xp, dtype):
        func = _fft_module(xp).idct
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_dst(self, xp, dtype):
        func = _fft_module(xp).dst
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_idst(self, xp, dtype):
        func = _fft_module(xp).idst
        return self._run_transform(func, xp, dtype)


@testing.parameterize(
    *(
        # 2D cases
        testing.product(
            {
                "shape": [(3, 4)],
                "type": [2, 3],
                # Note: non-integer s or s == 0 will cause a ValueError
                "s": [None, (1, 5), (-1, -1), (0, 5), (1.5, 2.5)],
                "axes": [None, (-2, -1), (-1, -2), (0,)],
                "norm": [None, "ortho"],
                "overwrite_x": [True, False],
            }
        )
        # 3D cases
        + testing.product(
            {
                "shape": [(2, 3, 4)],
                "type": [2, 3],
                # Note: len(s) < ndim is allowed
                #       len(s) > ndim raises a ValueError
                "s": [None, (1, 5), (1, 4, 10), (2, 2, 2, 2)],
                "axes": [None, (-2, -1), (-1, -2, -3)],
                "norm": [None, "ortho"],
                "overwrite_x": [True, False],
            }
        )
        # 4D cases
        + testing.product(
            {
                "shape": [(2, 3, 4, 5)],
                "type": [2, 3],
                "s": [None],
                "axes": [None, (0, 1, 2, 3)],
                "norm": [None, "ortho"],
                "overwrite_x": [True, False],
            }
        )
    )
)
@testing.gpu
class TestDctnDstn(unittest.TestCase):
    def _run_transform(self, dct_func, xp, dtype):
        x = testing.shaped_random(self.shape, xp, dtype)
        if scipy_cplx_bug and x.dtype.kind == "c":
            # skip cases where SciPy has a bug
            return x
        x_orig = x.copy()
        out = dct_func(
            x,
            type=self.type,
            s=self.s,
            axes=self.axes,
            norm=self.norm,
            overwrite_x=self.overwrite_x,
        )
        if not self.overwrite_x:
            testing.assert_array_equal(x, x_orig)
        return out

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_dctn(self, xp, dtype):
        func = _fft_module(xp).dctn
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_idctn(self, xp, dtype):
        func = _fft_module(xp).idctn
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_dstn(self, xp, dtype):
        func = _fft_module(xp).dstn
        return self._run_transform(func, xp, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_allclose(
        rtol=1e-4, atol=1e-5, accept_error=ValueError, contiguous_check=False
    )
    def test_idstn(self, xp, dtype):
        func = _fft_module(xp).idstn
        return self._run_transform(func, xp, dtype)
