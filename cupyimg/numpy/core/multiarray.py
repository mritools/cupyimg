import functools
import numbers
import operator

import cupy


# Note: ravel_multi_index has been submitted upstream
#       https://github.com/cupy/cupy/pull/3104
#       That PR includes tests vs. NumPy behavior


def ravel_multi_index(multi_index, dims, mode='wrap', order='C'):
    """
    Converts a tuple of index arrays into an array of flat indices, applying
    boundary modes to the multi-index.

    Args:
        multi_index (tuple of cupy.ndarray) : A tuple of integer arrays, one
            array for each dimension.
        dims (tuple of ints): The shape of array into which the indices from
            ``multi_index`` apply.
        mode ('raise', 'wrap' or 'clip'), optional: Specifies how out-of-bounds
            indices are handled.  Can specify either one mode or a tuple of
            modes, one mode per index:

            - *'raise'* -- raise an error
            - *'wrap'* -- wrap around (default)
            - *'clip'* -- clip to the range

            In 'clip' mode, a negative index which would normally wrap will
            clip to 0 instead.
        order ('C' or 'F'), optional: Determines whether the multi-index should
            be viewed as indexing in row-major (C-style) or column-major
            (Fortran-style) order.

    Returns:
        raveled_indices (cupy.ndarray): An array of indices into the flattened
            version of an array of dimensions ``dims``.

    .. warning::

        This function may synchronize the device when ``mode == 'raise'``.

    Notes
    -----
    Note that the default `mode` (``'wrap'``) is different than in NumPy. This
    is done to avoid potential device synchronization.

    Examples
    --------
    >>> cupy.ravel_multi_index(cupy.asarray([[3,6,6],[4,5,1]]), (7,6))
    array([22, 41, 37])
    >>> cupy.ravel_multi_index(cupy.asarray([[3,6,6],[4,5,1]]), (7,6),
    ...                        order='F')
    array([31, 41, 13])
    >>> cupy.ravel_multi_index(cupy.asarray([[3,6,6],[4,5,1]]), (4,6),
    ...                        mode='clip')
    array([22, 23, 19])
    >>> cupy.ravel_multi_index(cupy.asarray([[3,6,6],[4,5,1]]), (4,4),
    ...                        mode=('clip', 'wrap'))
    array([12, 13, 13])
    >>> cupy.ravel_multi_index(cupy.asarray((3,1,4,1)), (6,7,8,9))
    array(1621)

    .. seealso:: :func:`numpy.ravel_multi_index`, :func:`unravel_index`
    """

    ndim = len(dims)
    if len(multi_index) != ndim:
        raise ValueError(
            "parameter multi_index must be a sequence of "
            "length {}".format(ndim))

    for d in dims:
        if not isinstance(d, numbers.Integral):
            raise TypeError(
                "{} object cannot be interpreted as an integer".format(
                    type(d)))

    if isinstance(mode, str):
        mode = (mode, ) * ndim

    if functools.reduce(operator.mul, dims) > cupy.iinfo(cupy.int64).max:
        raise ValueError("invalid dims: array size defined by dims is larger "
                         "than the maximum possible size")

    s = 1
    ravel_strides = [1] * ndim
    if order is None:
        order = "C"
    if order == "C":
        for i in range(ndim - 2, -1, -1):
            s = s * dims[i + 1]
            ravel_strides[i] = s
    elif order == "F":
        for i in range(1, ndim):
            s = s * dims[i - 1]
            ravel_strides[i] = s
    else:
        raise TypeError("order not understood")

    multi_index = cupy.broadcast_arrays(*multi_index)
    raveled_indices = cupy.zeros(multi_index[0].shape, dtype=cupy.int64)
    for d, stride, idx, _mode in zip(dims, ravel_strides, multi_index, mode):

        if not isinstance(idx, cupy.ndarray):
            raise TypeError("elements of multi_index must be cupy arrays")
        if not cupy.can_cast(idx, cupy.int64, 'same_kind'):
            raise TypeError(
                'multi_index entries could not be cast from dtype(\'{}\') to '
                'dtype(\'{}\') according to the rule \'same_kind\''.format(
                    idx.dtype, cupy.int64().dtype))
        idx = idx.astype(cupy.int64, copy=False)

        if _mode == "raise":
            if cupy.any(cupy.logical_or(idx >= d, idx < 0)):
                raise ValueError("invalid entry in coordinates array")
        elif _mode == "clip":
            idx = cupy.clip(idx, 0, d - 1)
        elif _mode == 'wrap':
            idx = idx % d
        else:
            raise TypeError("Unrecognized mode: {}".format(_mode))
        raveled_indices += stride * idx
    return raveled_indices
