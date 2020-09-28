"""Misc utility functions that are not from SciPy, NumPy or scikit-image.

"""
import os
import tempfile

import cupy

from cupyimg.scipy.ndimage import convolve1d

__all__ = ["convolve_separable"]


def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with size > 1 at the specified axis."""
    if arr.ndim != 1:
        raise ValueError("expected a 1d array")
    if axis < -ndim or axis > ndim - 1:
        raise ValueError("invalid axis")
    if ndim < 1:
        raise ValueError("ndim must be >= 1")
    axis = axis % ndim
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)


def _prod(iterable):
    """
    Product of a list of numbers.
    Faster than np.prod for short lists like array shapes.
    """
    product = 1
    for x in iterable:
        product *= x
    return product


def convolve_separable(x, w, axes=None, **kwargs):
    """n-dimensional convolution via separable application of convolve1d

    Parameters
    ----------
    x : cupy.ndarray
        The input array.
    w : cupy.ndarray or sequence of cupy.ndarray
        If a single array is given, this same filter will be applied along
        all axes. A sequence of arrays can be provided in order to apply a
        separate filter along each axis. In this case the length of ``w`` must
        match the number of axes filtered.
    axes : tuple of int or None
        The axes of ``x`` to be filtered. The default (None) is to filter all
        axes of ``x``.

    Returns
    -------
    out : cupy.ndarray
        The filtered array.

    """
    if axes is None:
        axes = range(x.ndim)
    axes = tuple(axes)
    ndim = x.ndim
    if any(ax < -ndim or ax > ndim - 1 for ax in axes):
        raise ValueError("axis out of range")

    if isinstance(w, cupy.ndarray):
        w = [w] * len(axes)
    elif len(w) != len(axes):
        raise ValueError("user should supply one filter per axis")

    for ax, w0 in zip(axes, w):
        if not isinstance(w0, cupy.ndarray) or w0.ndim != 1:
            raise ValueError("w must be a 1d array (or sequence of 1d arrays)")
        x = convolve1d(x, w0, axis=ax, **kwargs)
    return x


class cache_source(object):
    """Context for use in temporarily caching source files in a new location.

    Can be used to inspect the source of previously compiled kernels.

    Notes
    -----
    If no files are being produced when using this context, try restarting the
    Python process. CuPy tends to cache previously compiled kernels in each
    process so that they don't have to be repeatedly retrieved from disk
    """

    def __init__(self, temp_dir=None):
        # store values of environment variables prior to entering the context
        from cupy.cuda.compiler import get_cache_dir

        val = os.environ.get("CUPY_CACHE_SAVE_CUDA_SOURCE")
        if val is None or len(val) == 0:
            self.cache_val = False
        else:
            try:
                self.cache_val = int(val) == 1
            except ValueError:
                self.cache_val = False
        self.cache_dir = get_cache_dir()

        # create a directory for caching the kernels
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            if not os.path.isdir(temp_dir):
                # raise ValueError("specified temp_dir doesn't exist")
                os.makedirs(temp_dir)
            self.temp_dir = temp_dir

    def __enter__(self):
        # enable caching of source files to the specified directory
        os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = "1"
        os.environ["CUPY_CACHE_DIR"] = self.temp_dir
        return self.temp_dir

    def __exit__(self, type, value, traceback):
        # restore values of environment variables prior to entering the context
        os.environ["CUPY_CACHE_SAVE_CUDA_SOURCE"] = str(int(self.cache_val))
        os.environ["CUPY_CACHE_DIR"] = self.cache_dir


try:
    from cupy._util import (
        _normalize_axis_index,
        _normalize_axis_indices,
    )  # NOQA

except ImportError:
    try:
        from cupy.util import (
            _normalize_axis_index,
            _normalize_axis_indices,
        )  # NOQA
    except ImportError:
        import numpy

        def _normalize_axis_index(axis, ndim):  # NOQA
            """
            Normalizes an axis index, ``axis``, such that is a valid positive
            index into the shape of array with ``ndim`` dimensions. Raises a
            ValueError with an appropriate message if this is not possible.

            Args:
                axis (int):
                    The un-normalized index of the axis. Can be negative
                ndim (int):
                    The number of dimensions of the array that ``axis`` should
                    be normalized against

            Returns:
                int:
                    The normalized axis index, such that
                    `0 <= normalized_axis < ndim`
            """
            if axis < 0:
                axis += ndim
            if not (0 <= axis < ndim):
                raise numpy.AxisError("axis out of bounds")
            return axis

        def _normalize_axis_indices(axes, ndim):  # NOQA
            """Normalize axis indices.

            Args:
                axis (int, tuple of int or None):
                    The un-normalized indices of the axis. Can be negative.
                ndim (int):
                    The number of dimensions of the array that ``axis`` should
                    be normalized against

            Returns:
                tuple of int:
                    The tuple of normalized axis indices.
            """
            if axes is None:
                axes = tuple(range(ndim))
            elif not isinstance(axes, tuple):
                axes = (axes,)

            res = []
            for axis in axes:
                axis = _normalize_axis_index(axis, ndim)
                if axis in res:
                    raise ValueError("Duplicate value in 'axis'")
                res.append(axis)

            return tuple(sorted(res))
