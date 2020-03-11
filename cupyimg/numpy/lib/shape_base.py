from numpy.lib.index_tricks import ndindex
from numpy.core.multiarray import normalize_axis_index
import cupy

__all__ = ["apply_along_axis"]


def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Apply a function to 1-D slices along the given axis.

    Execute `func1d(a, *args, **kwargs)` where `func1d` operates on 1-D arrays
    and `a` is a 1-D slice of `arr` along `axis`.

    This is equivalent to (but faster than) the following use of `ndindex` and
    `s_`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                f = func1d(arr[ii + s_[:,] + kk])
                Nj = f.shape
                for jj in ndindex(Nj):
                    out[ii + jj + kk] = f[jj]

    Equivalently, eliminating the inner loop, this can be expressed as::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nk):
                out[ii + s_[...,] + kk] = func1d(arr[ii + s_[:,] + kk])

    Parameters
    ----------
    func1d : function (M,) -> (Nj...)
        This function should accept 1-D arrays. It is applied to 1-D
        slices of `arr` along the specified axis. It must return a 1-D
        ``cupy.ndarray``.
    axis : integer
        Axis along which `arr` is sliced.
    arr : ndarray (Ni..., M, Nk...)
        Input array.
    args : any
        Additional arguments to `func1d`.
    kwargs : any
        Additional named arguments to `func1d`.

        .. versionadded:: 1.9.0


    Returns
    -------
    out : ndarray  (Ni..., Nj..., Nk...)
        The output array. The shape of `out` is identical to the shape of
        `arr`, except along the `axis` dimension. This axis is removed, and
        replaced with new dimensions equal to the shape of the return value
        of `func1d`. So if `func1d` returns a scalar `out` will have one
        fewer dimensions than `arr`.

    See Also
    --------
    apply_over_axes : Apply a function repeatedly over multiple axes.

    Examples
    --------
    >>> import cupy
    >>> from cupyimg.numpy import apply_along_axis
    >>> def my_func(a):
    ...     '''Average first and last element of a 1-D array'''
    ...     return (a[0] + a[-1]) * 0.5
    >>> b = cupy.asarray([[1,2,3], [4,5,6], [7,8,9]])
    >>> apply_along_axis(my_func, 0, b)
    array([4., 5., 6.])
    >>> apply_along_axis(my_func, 1, b)
    array([2.,  5.,  8.])

    For a function that returns a higher dimensional array, those dimensions
    are inserted in place of the `axis` dimension.

    >>> b = cupy.asarray([[1,2,3], [4,5,6], [7,8,9]])
    >>> apply_along_axis(cupy.diag, -1, b)
    array([[[1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]],
           [[4, 0, 0],
            [0, 5, 0],
            [0, 0, 6]],
           [[7, 0, 0],
            [0, 8, 0],
            [0, 0, 9]]])
    """
    # handle negative axes
    nd = arr.ndim
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_view = cupy.transpose(
        arr, in_dims[:axis] + in_dims[axis + 1 :] + [axis]
    )

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = ndindex(inarr_view.shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError(
            "Cannot apply_along_axis when any iteration dimensions are 0"
        )
    res = func1d(inarr_view[ind0], *args, **kwargs)

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = cupy.zeros(inarr_view.shape[:-1] + res.shape, res.dtype)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0:axis]
        + buff_dims[buff.ndim - res.ndim : buff.ndim]
        + buff_dims[axis : buff.ndim - res.ndim]
    )

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = func1d(inarr_view[ind], *args, **kwargs)

    # finally, rotate the inserted axes back to where they belong
    return cupy.transpose(buff, buff_permute)
