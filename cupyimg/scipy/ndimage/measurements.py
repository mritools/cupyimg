import cupy
import numpy

from .morphology import generate_binary_structure

__all__ = [
    "labeled_comprehension",
    "sum",
    "mean",
    "variance",
    "standard_deviation",
    "minimum",
    "maximum",
    "median",
    "minimum_position",
    "maximum_position",
    "extrema",
    "center_of_mass",
    "histogram",
    "label",
]

# TODO: grlee77: 'find_objects', 'watershed_ift'


def _safely_castable_to_int(dt):
    """Test whether the NumPy data type `dt` can be safely cast to an int."""
    int_size = cupy.dtype(int).itemsize
    safe = (
        cupy.issubdtype(dt, cupy.signedinteger) and dt.itemsize <= int_size
    ) or (cupy.issubdtype(dt, cupy.unsignedinteger) and dt.itemsize < int_size)
    return safe


def labeled_comprehension(
    input, labels, index, func, out_dtype, default, pass_positions=False
):
    """
    Roughly equivalent to [func(input[labels == i]) for i in index].

    Sequentially applies an arbitrary function (that works on array_like input)
    to subsets of an N-D image array specified by `labels` and `index`.
    The option exists to provide the function with positional parameters as the
    second argument.

    Parameters
    ----------
    input : array_like
        Data from which to select `labels` to process.
    labels : array_like or None
        Labels to objects in `input`.
        If not None, array must be same shape as `input`.
        If None, `func` is applied to raveled `input`.
    index : int, sequence of ints or None
        Subset of `labels` to which to apply `func`.
        If a scalar, a single value is returned.
        If None, `func` is applied to all non-zero values of `labels`.
    func : callable
        Python function to apply to `labels` from `input`.
    out_dtype : dtype
        Dtype to use for `result`.
    default : int, float or None
        Default return value when a element of `index` does not exist
        in `labels`.
    pass_positions : bool, optional
        If True, pass linear indices to `func` as a second argument.
        Default is False.

    Returns
    -------
    result : ndarray
        Result of applying `func` to each of `labels` to `input` in `index`.

    Examples
    --------
    >>> import cupy as cp
    >>> a = cp.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from cupyimg.scipy import  ndimage
    >>> lbl, nlbl = ndimage.label(a)
    >>> lbls = cp.arange(1, nlbl+1)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, cp.mean, float, 0)
    array([ 2.75,  5.5 ,  6.  ])

    Falling back to `default`:

    >>> lbls = cp.arange(1, nlbl+2)
    >>> ndimage.labeled_comprehension(a, lbl, lbls, cp.mean, float, -1)
    array([ 2.75,  5.5 ,  6.  , -1.  ])

    Passing positions:

    >>> def fn(val, pos):
    ...     print("fn says: %s : %s" % (val, pos))
    ...     return (val.sum()) if (pos.sum() % 2 == 0) else (-val.sum())
    ...
    >>> ndimage.labeled_comprehension(a, lbl, lbls, fn, float, 0, True)
    fn says: [1 2 5 3] : [0 1 4 5]
    fn says: [4 7] : [ 7 11]
    fn says: [9 3] : [12 13]
    array([ 11.,  11., -12.,   0.])

    """

    as_scalar = cupy.isscalar(index)
    input = cupy.asarray(input)

    if pass_positions:
        positions = cupy.arange(input.size).reshape(input.shape)

    if labels is None:
        if index is not None:
            raise ValueError("index without defined labels")
        if not pass_positions:
            return func(input.ravel())
        else:
            return func(input.ravel(), positions.ravel())

    try:
        input, labels = cupy.broadcast_arrays(input, labels)
    except ValueError:
        raise ValueError(
            "input and labels must have the same shape "
            "(excepting dimensions with width 1)"
        )

    if index is None:
        if not pass_positions:
            return func(input[labels > 0])
        else:
            return func(input[labels > 0], positions[labels > 0])

    index = cupy.atleast_1d(index)
    if cupy.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError(
            "Cannot convert index values from <%s> to <%s> "
            "(labels' type) without loss of precision"
            % (index.dtype, labels.dtype)
        )

    index = index.astype(labels.dtype)

    # optimization: find min/max in index, and select those parts of labels, input, and positions
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)

    # this also ravels the arrays
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]

    # sort everything by labels
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]

    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        """labels must be sorted"""
        nidx = sorted_index.size

        # Find boundaries for each stretch of constant labels
        # This could be faster, but we already paid N log N to sort labels.
        lo = cupy.searchsorted(labels, sorted_index, side="left")
        hi = cupy.searchsorted(labels, sorted_index, side="right")

        for i, l, h in zip(range(nidx), lo, hi):
            if l == h:
                continue
            output[i] = cupy.asnumpy(func(*[inp[l:h] for inp in inputs]))

    temp = numpy.empty(index.shape, out_dtype)
    temp[:] = default
    if not pass_positions:
        do_map([input], temp)
    else:
        do_map([input, positions], temp)

    output = numpy.zeros(index.shape, out_dtype)
    output[cupy.asnumpy(index_order)] = temp
    if as_scalar:
        output = output[0]

    return output


def _stats(input, labels=None, index=None, centered=False):
    """Count, sum, and optionally compute (sum - centre)^2 of input by label

    Parameters
    ----------
    input : array_like, N-D
        The input data to be analyzed.
    labels : array_like (N-D), optional
        The labels of the data in `input`. This array must be broadcast
        compatible with `input`; typically, it is the same shape as `input`.
        If `labels` is None, all nonzero values in `input` are treated as
        the single labeled group.
    index : label or sequence of labels, optional
        These are the labels of the groups for which the stats are computed.
        If `index` is None, the stats are computed for the single group where
        `labels` is greater than 0.
    centered : bool, optional
        If True, the centered sum of squares for each labeled group is
        also returned. Default is False.

    Returns
    -------
    counts : int or ndarray of ints
        The number of elements in each labeled group.
    sums : scalar or ndarray of scalars
        The sums of the values in each labeled group.
    sums_c : scalar or ndarray of scalars, optional
        The sums of mean-centered squares of the values in each labeled group.
        This is only returned if `centered` is True.

    """

    def single_group(vals):
        if centered:
            vals_c = vals - vals.mean()
            return vals.size, vals.sum(), (vals_c * vals_c.conj()).sum()
        else:
            return vals.size, vals.sum()

    if labels is None:
        return single_group(input)

    # ensure input and labels match sizes
    input, labels = cupy.broadcast_arrays(input, labels)

    if index is None:
        return single_group(input[labels > 0])

    if cupy.isscalar(index):
        return single_group(input[labels == index])
    else:
        index = cupy.asarray(index)

    def _sum_centered(labels):
        # `labels` is expected to be an ndarray with the same shape as `input`.
        # It must contain the label indices (which are not necessarily the labels
        # themselves).
        means = sums / counts
        centered_input = input - means[labels]
        # bincount expects 1-D inputs, so we ravel the arguments.
        bc = cupy.bincount(
            labels.ravel(),
            weights=(centered_input * centered_input.conj()).ravel(),
        )
        return bc

    # Remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.

    if (
        not _safely_castable_to_int(labels.dtype)
        or labels.min() < 0
        or labels.max() > labels.size
    ):
        # Use cupy.unique to generate the label indices.  `new_labels` will
        # be 1-D, but it should be interpreted as the flattened N-D array of
        # label indices.
        unique_labels, new_labels = cupy.unique(labels, return_inverse=True)
        counts = cupy.bincount(new_labels)
        sums = cupy.bincount(new_labels, weights=input.ravel())
        if centered:
            # Compute the sum of the mean-centered squares.
            # We must reshape new_labels to the N-D shape of `input` before
            # passing it _sum_centered.
            sums_c = _sum_centered(new_labels.reshape(labels.shape))
        idxs = cupy.searchsorted(unique_labels, index)
        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = unique_labels[idxs] == index
    else:
        # labels are an integer type allowed by bincount, and there aren't too
        # many, so call bincount directly.
        counts = cupy.bincount(labels.ravel())
        sums = cupy.bincount(labels.ravel(), weights=input.ravel())
        if centered:
            sums_c = _sum_centered(labels)
        # make sure all index values are valid
        idxs = cupy.asanyarray(index, cupy.int).copy()
        found = (idxs >= 0) & (idxs < counts.size)
        idxs[~found] = 0

    counts = counts[idxs]
    counts[~found] = 0
    sums = sums[idxs]
    sums[~found] = 0

    # TODO: grlee77: call .item() on 0-dim arrays to give scalar values as
    #                described in the docstring?
    if not centered:
        return (counts, sums)
    else:
        sums_c = sums_c[idxs]
        sums_c[~found] = 0
        return (counts, sums, sums_c)


def sum(input, labels=None, index=None):
    """
    Calculate the sum of the values of the array.

    Parameters
    ----------
    input : array_like
        Values of `input` inside the regions defined by `labels`
        are summed together.
    labels : array_like of ints, optional
        Assign labels to the values of the array. Has to have the same shape as
        `input`.
    index : array_like, optional
        A single label number or a sequence of label numbers of
        the objects to be measured.

    Returns
    -------
    sum : ndarray or scalar
        An array of the sums of values of `input` inside the regions defined
        by `labels` with the same shape as `index`. If 'index' is None or scalar,
        a scalar is returned.

    See also
    --------
    mean, median

    Examples
    --------
    >>> from cupyimg.scipy import  ndimage
    >>> input =  [0,1,2,3]
    >>> labels = [1,1,2,2]
    >>> ndimage.sum(input, labels, index=[1,2])
    [1.0, 5.0]
    >>> ndimage.sum(input, labels, index=1)
    1
    >>> ndimage.sum(input, labels)
    6


    """
    count, sum = _stats(input, labels, index)
    return sum


def mean(input, labels=None, index=None):
    """
    Calculate the mean of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array on which to compute the mean of elements over distinct
        regions.
    labels : array_like, optional
        Array of labels of same shape, or broadcastable to the same shape as
        `input`. All elements sharing the same label form one region over
        which the mean of the elements is computed.
    index : int or sequence of ints, optional
        Labels of the objects over which the mean is to be computed.
        Default is None, in which case the mean for all values where label is
        greater than 0 is calculated.

    Returns
    -------
    out : list
        Sequence of same length as `index`, with the mean of the different
        regions labeled by the labels in `index`.

    See also
    --------
    variance, standard_deviation, minimum, maximum, sum, label

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.arange(25).reshape((5,5))
    >>> labels = cp.zeros_like(a)
    >>> labels[3:5,3:5] = 1
    >>> index = cp.unique(labels)
    >>> labels
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 0, 1, 1]])
    >>> index
    array([0, 1])
    >>> ndimage.mean(a, labels=labels, index=index)
    [10.285714285714286, 21.0]

    """

    count, sum = _stats(input, labels, index)
    return sum / cupy.asanyarray(count).astype(cupy.float)


def variance(input, labels=None, index=None):
    """
    Calculate the variance of the values of an N-D image array, optionally at
    specified sub-regions.

    Parameters
    ----------
    input : array_like
        Nd-image data to process.
    labels : array_like, optional
        Labels defining sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output.  If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    variance : float or ndarray
        Values of variance, for each sub-region if `labels` and `index` are
        specified.

    See Also
    --------
    label, standard_deviation, maximum, minimum, extrema

    Examples
    --------
    >>> import cupy as cp
    >>> a = cp.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from cupyimg.scipy import  ndimage
    >>> ndimage.variance(a)
    7.609375

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.variance(a, lbl, index=cp.arange(1, nlbl+1))
    array([ 2.1875,  2.25  ,  9.    ])

    If no index is given, all non-zero `labels` are processed:

    >>> ndimage.variance(a, lbl)
    6.1875

    """
    count, sum, sum_c_sq = _stats(input, labels, index, centered=True)
    return sum_c_sq / cupy.asanyarray(count).astype(float)


def standard_deviation(input, labels=None, index=None):
    """
    Calculate the standard deviation of the values of an N-D image array,
    optionally at specified sub-regions.

    Parameters
    ----------
    input : array_like
        N-D image data to process.
    labels : array_like, optional
        Labels to identify sub-regions in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        `labels` to include in output. If None (default), all values where
        `labels` is non-zero are used.

    Returns
    -------
    standard_deviation : float or ndarray
        Values of standard deviation, for each sub-region if `labels` and
        `index` are specified.

    See Also
    --------
    label, variance, maximum, minimum, extrema

    Examples
    --------
    >>> import cupy as cp
    >>> a = cp.array([[1, 2, 0, 0],
    ...               [5, 3, 0, 4],
    ...               [0, 0, 0, 7],
    ...               [9, 3, 0, 0]])
    >>> from cupyimg.scipy import  ndimage
    >>> ndimage.standard_deviation(a)
    2.7585095613392387

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.standard_deviation(a, lbl, index=cp.arange(1, nlbl+1))
    array([ 1.479,  1.5  ,  3.   ])

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.standard_deviation(a, lbl)
    2.4874685927665499

    """
    return cupy.sqrt(variance(input, labels, index))


def _select(
    input,
    labels=None,
    index=None,
    find_min=False,
    find_max=False,
    find_min_positions=False,
    find_max_positions=False,
    find_median=False,
):
    """Returns min, max, or both, plus their positions (if requested), and
    median."""

    input = cupy.asanyarray(input)

    find_positions = find_min_positions or find_max_positions
    positions = None
    if find_positions:
        positions = cupy.arange(input.size).reshape(input.shape)

    def single_group(vals, positions):
        result = []
        if find_min:
            result += [vals.min()]
        if find_min_positions:
            result += [positions[vals == vals.min()][0]]
        if find_max:
            result += [vals.max()]
        if find_max_positions:
            result += [positions[vals == vals.max()][0]]
        if find_median:
            # TODO: grlee77: implement cupy.median
            result += [numpy.median(cupy.asnumpy(vals))]
        return result

    if labels is None:
        return single_group(input, positions)

    # ensure input and labels match sizes
    input, labels = cupy.broadcast_arrays(input, labels)

    if index is None:
        mask = labels > 0
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    if cupy.isscalar(index):
        mask = labels == index
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)

    index = cupy.asarray(index)

    # remap labels to unique integers if necessary, or if the largest
    # label is larger than the number of values.
    if (
        not _safely_castable_to_int(labels.dtype)
        or labels.min() < 0
        or labels.max() > labels.size
    ):
        # remap labels, and indexes
        unique_labels, labels = cupy.unique(labels, return_inverse=True)
        idxs = cupy.searchsorted(unique_labels, index)

        # make all of idxs valid
        idxs[idxs >= unique_labels.size] = 0
        found = unique_labels[idxs] == index
    else:
        # labels are an integer type, and there aren't too many
        idxs = cupy.asanyarray(index, cupy.int).copy()
        found = (idxs >= 0) & (idxs <= labels.max())

    idxs[~found] = labels.max() + 1

    if find_median:
        # TODO: grlee77: fix cupy.lexsort for multiple keys
        order = cupy.asarray(
            numpy.lexsort(
                (cupy.asnumpy(input.ravel()), cupy.asnumpy(labels.ravel()))
            )
        )
    else:
        order = input.ravel().argsort()
    input = input.ravel()[order]
    labels = labels.ravel()[order]
    labels = cupy.asnumpy(labels)
    idxs = cupy.asnumpy(idxs)
    if find_positions:
        positions = positions.ravel()[order]
        positions = cupy.asnumpy(positions)
    if find_min or find_max or find_median:
        input = cupy.asnumpy(input)

    result = []
    if find_min:
        mins = numpy.zeros(labels.max().item() + 2, input.dtype)
        mins[labels[::-1]] = input[::-1]
        result += [mins[idxs]]
    if find_min_positions:
        minpos = numpy.zeros(labels.max().item() + 2, int)
        # Note: CuPy cannot index with repeated integer values:
        #       https://docs-cupy.chainer.org/en/stable/reference/difference.html#duplicate-values-in-indices
        # For now, I have transferred labels/positions back to host above
        # to avoid this.
        minpos[labels[::-1]] = positions[::-1]
        result += [minpos[idxs]]
    if find_max:
        maxs = numpy.zeros(labels.max().item() + 2, input.dtype)
        maxs[labels] = input
        result += [maxs[idxs]]
    if find_max_positions:
        maxpos = numpy.zeros(labels.max().item() + 2, int)
        maxpos[labels] = positions
        result += [maxpos[idxs]]
    if find_median:
        locs = numpy.arange(len(labels))
        lo = numpy.zeros(labels.max().item() + 2, cupy.int)
        lo[labels[::-1]] = locs[::-1]
        hi = numpy.zeros(labels.max().item() + 2, cupy.int)
        hi[labels] = locs
        lo = lo[idxs]
        hi = hi[idxs]
        # lo is an index to the lowest value in input for each label,
        # hi is an index to the largest value.
        # move them to be either the same ((hi - lo) % 2 == 0) or next
        # to each other ((hi - lo) % 2 == 1), then average.
        step = (hi - lo) // 2
        lo += step
        hi -= step
        result += [(input[lo] + input[hi]) / 2.0]

    return result


def minimum(input, labels=None, index=None):
    """
    Calculate the minimum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        minimal values of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        minimum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the minimum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        minima. If index is None, the minimum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    minimum : float or list of floats
        List of minima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the minimal value of `input` if `labels` is None,
        and the minimal value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, maximum, median, minimum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `cp.asarray` to convert the list to an array.

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.asarray([[1, 2, 0, 0],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.minimum(a, labels=labels, index=cp.arange(1, labels_nb + 1))
    [1.0, 4.0, 3.0]
    >>> ndimage.minimum(a)
    0.0
    >>> ndimage.minimum(a, labels=labels)
    1.0

    """
    # TODO: grlee77: should this call item() to get a scalar or leave as
    #                0-dim ndarray?
    return _select(input, labels, index, find_min=True)[0]


def maximum(input, labels=None, index=None):
    """
    Calculate the maximum of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        maximal values of `input` over the region is computed.
    labels : array_like, optional
        An array of integers marking different regions over which the
        maximum value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the maximum
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        maxima. If index is None, the maximum over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    output : float or list of floats
        List of maxima of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the maximal value of `input` if `labels` is None,
        and the maximal value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `cp.asarray` to convert the list to an array.

    Examples
    --------
    >>> a = cp.arange(16).reshape((4,4))
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> labels = cp.zeros_like(a)
    >>> labels[:2,:2] = 1
    >>> labels[2:, 1:3] = 2
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 2, 2, 0],
           [0, 2, 2, 0]])
    >>> from cupyimg.scipy import ndimage
    >>> ndimage.maximum(a)
    15.0
    >>> ndimage.maximum(a, labels=labels, index=[1,2])
    [5.0, 14.0]
    >>> ndimage.maximum(a, labels=labels)
    14.0

    >>> b = cp.asarray([[1, 2, 0, 0],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(b)
    >>> labels
    array([[1, 1, 0, 0],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.maximum(b, labels=labels, index=cp.arange(1, labels_nb + 1))
    [5.0, 7.0, 9.0]

    """
    return _select(input, labels, index, find_max=True)[0]


def median(input, labels=None, index=None):
    """
    Calculate the median of the values of an array over labeled regions.

    Parameters
    ----------
    input : array_like
        Array_like of values. For each region specified by `labels`, the
        median value of `input` over the region is computed.
    labels : array_like, optional
        An array_like of integers marking different regions over which the
        median value of `input` is to be computed. `labels` must have the
        same shape as `input`. If `labels` is not specified, the median
        over the whole array is returned.
    index : array_like, optional
        A list of region labels that are taken into account for computing the
        medians. If index is None, the median over all elements where `labels`
        is non-zero is returned.

    Returns
    -------
    median : float or list of floats
        List of medians of `input` over the regions determined by `labels` and
        whose index is in `index`. If `index` or `labels` are not specified, a
        float is returned: the median value of `input` if `labels` is None,
        and the median value of elements where `labels` is greater than zero
        if `index` is None.

    See also
    --------
    label, minimum, maximum, extrema, sum, mean, variance, standard_deviation

    Notes
    -----
    The function returns a Python list and not a NumPy array, use
    `cp.array` to convert the list to an array.

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.asarray([[1, 2, 0, 1],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])
    >>> labels, labels_nb = ndimage.label(a)
    >>> labels
    array([[1, 1, 0, 2],
           [1, 1, 0, 2],
           [0, 0, 0, 2],
           [3, 3, 0, 0]])
    >>> ndimage.median(a, labels=labels, index=cp.arange(1, labels_nb + 1))
    [2.5, 4.0, 6.0]
    >>> ndimage.median(a)
    1.0
    >>> ndimage.median(a, labels=labels)
    3.0

    """
    return _select(input, labels, index, find_median=True)[0]


def minimum_position(input, labels=None, index=None):
    """
    Find the positions of the minimums of the values of an array at labels.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the minimum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first minimum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the minima. If `index` is None, the ``first`` minimum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        Tuple of ints or list of tuples of ints that specify the location
        of minima of `input` over the regions determined by `labels` and
        whose index is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the first minimal value of `input`.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> a = cp.asarray([[10, 20, 30],
    ...                 [40, 80, 100],
    ...                 [1, 100, 200]])
    >>> b = cp.asarray([[1, 2, 0, 1],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])

    >>> from cupyimg.scipy import ndimage

    >>> ndimage.minimum_position(a)
    (2, 0)
    >>> ndimage.minimum_position(b)
    (0, 2)

    Features to process can be specified using `labels` and `index`:

    >>> label, pos = ndimage.label(a)
    >>> ndimage.minimum_position(a, label, index=cp.arange(1, pos+1))
    [(2, 0)]

    >>> label, pos = ndimage.label(b)
    >>> ndimage.minimum_position(b, label, index=cp.arange(1, pos+1))
    [(0, 0), (0, 3), (3, 1)]

    """
    dims = numpy.array(cupy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    result = _select(input, labels, index, find_min_positions=True)[0]

    # have to transfer 0-dim array back to CPU?
    # may want to modify to avoid this
    if isinstance(result, cupy.ndarray) and result.ndim == 0:
        result = result.item()

    if cupy.isscalar(result):
        return tuple((result // dim_prod) % dims)
    result = cupy.asnumpy(result)

    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]


def maximum_position(input, labels=None, index=None):
    """
    Find the positions of the maximums of the values of an array at labels.

    For each region specified by `labels`, the position of the maximum
    value of `input` within the region is returned.

    Parameters
    ----------
    input : array_like
        Array_like of values.
    labels : array_like, optional
        An array of integers marking different regions over which the
        position of the maximum value of `input` is to be computed.
        `labels` must have the same shape as `input`. If `labels` is not
        specified, the location of the first maximum over the whole
        array is returned.

        The `labels` argument only works when `index` is specified.
    index : array_like, optional
        A list of region labels that are taken into account for finding the
        location of the maxima. If `index` is None, the first maximum
        over all elements where `labels` is non-zero is returned.

        The `index` argument only works when `labels` is specified.

    Returns
    -------
    output : list of tuples of ints
        List of tuples of ints that specify the location of maxima of
        `input` over the regions determined by `labels` and whose index
        is in `index`.

        If `index` or `labels` are not specified, a tuple of ints is
        returned specifying the location of the ``first`` maximal value
        of `input`.

    See also
    --------
    label, minimum, median, maximum_position, extrema, sum, mean, variance,
    standard_deviation

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.asarray([[1, 2, 0, 0],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])
    >>> ndimage.maximum_position(a)
    (3, 0)

    Features to process can be specified using `labels` and `index`:

    >>> lbl = cp.asarray([[0, 1, 2, 3],
    ...                   [0, 1, 2, 3],
    ...                   [0, 1, 2, 3],
    ...                   [0, 1, 2, 3]])
    >>> ndimage.maximum_position(a, lbl, 1)
    (1, 1)

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.maximum_position(a, lbl)
    (2, 3)

    If there are no maxima, the position of the first element is returned:

    >>> ndimage.maximum_position(a, lbl, 2)
    (0, 2)

    """
    dims = numpy.array(cupy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    result = _select(input, labels, index, find_max_positions=True)[0]

    # have to transfer 0-dim array back to CPU?
    # may want to modify to avoid this
    if isinstance(result, cupy.ndarray) and result.ndim == 0:
        result = result.item()

    if cupy.isscalar(result):
        return tuple((result // dim_prod) % dims)
    result = cupy.asnumpy(result)

    return [tuple(v) for v in (result.reshape(-1, 1) // dim_prod) % dims]


def extrema(input, labels=None, index=None):
    """
    Calculate the minimums and maximums of the values of an array
    at labels, along with their positions.

    Parameters
    ----------
    input : ndarray
        N-D image data to process.
    labels : ndarray, optional
        Labels of features in input.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        Labels to include in output.  If None (default), all values where
        non-zero `labels` are used.

    Returns
    -------
    minimums, maximums : int or ndarray
        Values of minimums and maximums in each feature.
    min_positions, max_positions : tuple or list of tuples
        Each tuple gives the N-D coordinates of the corresponding minimum
        or maximum.

    See Also
    --------
    maximum, minimum, maximum_position, minimum_position, center_of_mass

    Examples
    --------
    >>> a = cp.asarray([[1, 2, 0, 0],
    ...                 [5, 3, 0, 4],
    ...                 [0, 0, 0, 7],
    ...                 [9, 3, 0, 0]])
    >>> from cupyimg.scipy import ndimage
    >>> ndimage.extrema(a)
    (0, 9, (0, 2), (3, 0))

    Features to process can be specified using `labels` and `index`:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.extrema(a, lbl, index=cp.arange(1, nlbl+1))
    (array([1, 4, 3]),
     array([5, 7, 9]),
     [(0, 0), (1, 3), (3, 1)],
     [(1, 0), (2, 3), (3, 0)])

    If no index is given, non-zero `labels` are processed:

    >>> ndimage.extrema(a, lbl)
    (1, 9, (0, 0), (3, 0))

    """
    dims = numpy.array(cupy.asarray(input).shape)
    # see numpy.unravel_index to understand this line.
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]

    minimums, min_positions, maximums, max_positions = _select(
        input,
        labels,
        index,
        find_min=True,
        find_max=True,
        find_min_positions=True,
        find_max_positions=True,
    )

    # have to transfer 0-dim array back to CPU?
    # may want to modify to avoid this
    if isinstance(minimums, cupy.ndarray) and minimums.ndim == 0:
        minimums = minimums.item()
    if isinstance(maximums, cupy.ndarray) and maximums.ndim == 0:
        maximums = maximums.item()
    if isinstance(min_positions, cupy.ndarray) and min_positions.ndim == 0:
        min_positions = min_positions.item()
    if isinstance(max_positions, cupy.ndarray) and max_positions.ndim == 0:
        max_positions = max_positions.item()

    if cupy.isscalar(minimums):
        return (
            minimums,
            maximums,
            tuple((min_positions // dim_prod) % dims),
            tuple((max_positions // dim_prod) % dims),
        )

    min_positions = cupy.asnumpy(min_positions)
    max_positions = cupy.asnumpy(max_positions)
    min_positions = [
        tuple(v) for v in (min_positions.reshape(-1, 1) // dim_prod) % dims
    ]
    max_positions = [
        tuple(v) for v in (max_positions.reshape(-1, 1) // dim_prod) % dims
    ]

    return minimums, maximums, min_positions, max_positions


def center_of_mass(input, labels=None, index=None):
    """
    Calculate the center of mass of the values of an array at labels.

    Parameters
    ----------
    input : ndarray
        Data from which to calculate center-of-mass. The masses can either
        be positive or negative.
    labels : ndarray, optional
        Labels for objects in `input`, as generated by `ndimage.label`.
        Only used with `index`. Dimensions must be the same as `input`.
    index : int or sequence of ints, optional
        Labels for which to calculate centers-of-mass. If not specified,
        all labels greater than zero are used. Only used with `labels`.

    Returns
    -------
    center_of_mass : tuple, or list of tuples
        Coordinates of centers-of-mass.

    Examples
    --------
    >>> a = cp.asarray(([0,0,0,0],
    ...                 [0,1,1,0],
    ...                 [0,1,1,0],
    ...                 [0,1,1,0]))
    >>> from cupyimg.scipy import ndimage
    >>> ndimage.measurements.center_of_mass(a)
    (2.0, 1.5)

    Calculation of multiple objects in an image

    >>> b = cp.asarray(([0,1,1,0],
    ...                 [0,1,0,0],
    ...                 [0,0,0,0],
    ...                 [0,0,1,1],
    ...                 [0,0,1,1]))
    >>> lbl = ndimage.label(b)[0]
    >>> ndimage.measurements.center_of_mass(b, lbl, [1,2])
    [(0.33333333333333331, 1.3333333333333333), (3.5, 2.5)]

    Negative masses are also accepted, which can occur for example when
    bias is removed from measured data due to random noise.

    >>> c = cp.asarray(([-1,0,0,0],
    ...                 [0,-1,-1,0],
    ...                 [0,1,-1,0],
    ...                 [0,1,1,0]))
    >>> ndimage.measurements.center_of_mass(c)
    (-4.0, 1.0)

    If there are division by zero issues, the function does not raise an
    error but rather issues a RuntimeWarning before returning inf and/or NaN.

    >>> d = cp.asarray([-1, 1])
    >>> ndimage.measurements.center_of_mass(d)
    (inf,)
    """
    normalizer = sum(input, labels, index)
    grids = cupy.ogrid[[slice(0, i) for i in input.shape]]

    results = [
        sum(input * grids[dir].astype(float), labels, index) / normalizer
        for dir in range(input.ndim)
    ]

    # have to transfer 0-dim array back to CPU?
    # may want to modify to avoid this
    is_0dim_array = (
        isinstance(results[0], cupy.ndarray) and results[0].ndim == 0
    )

    if cupy.isscalar(results[0]) or is_0dim_array:
        return tuple(cupy.asnumpy(results))

    results = [cupy.asnumpy(r) for r in results]
    return [tuple(v) for v in numpy.array(results).T]
    # return [tuple(v) for v in cupy.asnumpy(cupy.stack(results, axis=-1))]


def histogram(input, min, max, bins, labels=None, index=None):
    """
    Calculate the histogram of the values of an array, optionally at labels.

    Histogram calculates the frequency of values in an array within bins
    determined by `min`, `max`, and `bins`. The `labels` and `index`
    keywords can limit the scope of the histogram to specified sub-regions
    within the array.

    Parameters
    ----------
    input : array_like
        Data for which to calculate histogram.
    min, max : int
        Minimum and maximum values of range of histogram bins.
    bins : int
        Number of bins.
    labels : array_like, optional
        Labels for objects in `input`.
        If not None, must be same shape as `input`.
    index : int or sequence of ints, optional
        Label or labels for which to calculate histogram. If None, all values
        where label is greater than zero are used

    Returns
    -------
    hist : ndarray
        Histogram counts.

    Examples
    --------
    >>> a = cp.asarray([[ 0.    ,  0.2146,  0.5962,  0.    ],
    ...                 [ 0.    ,  0.7778,  0.    ,  0.    ],
    ...                 [ 0.    ,  0.    ,  0.    ,  0.    ],
    ...                 [ 0.    ,  0.    ,  0.7181,  0.2787],
    ...                 [ 0.    ,  0.    ,  0.6573,  0.3094]])
    >>> from cupyimg.scipy import ndimage
    >>> ndimage.measurements.histogram(a, 0, 1, 10)
    array([13,  0,  2,  1,  0,  1,  1,  2,  0,  0])

    With labels and no indices, non-zero elements are counted:

    >>> lbl, nlbl = ndimage.label(a)
    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl)
    array([0, 0, 2, 1, 0, 1, 1, 2, 0, 0])

    Indices can be used to count only certain objects:

    >>> ndimage.measurements.histogram(a, 0, 1, 10, lbl, 2)
    array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])

    """
    _bins = cupy.linspace(min, max, bins + 1)

    def _hist(vals):
        return cupy.histogram(vals, _bins)[0]

    return labeled_comprehension(
        input, labels, index, _hist, object, None, pass_positions=False
    )


"""
Elementwise Kernel's for use by label
"""


def _kernel_init():
    return cupy.ElementwiseKernel(
        "X x",
        "Y y",
        "if (x == 0) { y = -1; } else { y = i; }",
        "cupyx_nd_label_init",
    )


def _kernel_connect(greyscale_mode=False, int_t="int"):  # cas_dtype='int'):
    """
    Notes
    -----
    dirs is a (n_neig//2, ndim) of relative offsets to the neighboring voxels.
    For example, for structure = np.ones((3, 3)):
        dirs = array([[-1, -1],
                      [-1,  0],
                      [-1,  1],
                      [ 0, -1]], dtype=int32)
    (Implementation assumes a centro-symmetric structure)
    ndirs = dirs.shape[0]

    In the dirs loop below, there is a loop over the ndim neighbors:
        Here, index j corresponds to the current pixel and k is the current
        neighbor location.
    """
    in_params = "raw int32 shape, raw int32 dirs, int32 ndirs, int32 ndim"
    if greyscale_mode:
        # greyscale mode -> different values receive different labels
        x_condition = "if (x[k] != x[j]) continue;"
        in_params = "raw X x, " + in_params
    else:
        # binary mode -> all non-background voxels treated the same
        x_condition = ""

    # Note: atomicCAS is implemented for int, unsigned short, unsigned int, and
    # unsigned long long

    code = """
        if (y[i] < 0) continue;
        for (int dr = 0; dr < ndirs; dr++) {{
            {int_t} j = i;
            {int_t} rest = j;
            {int_t} stride = 1;
            {int_t} k = 0;
            for (int dm = ndim-1; dm >= 0; dm--) {{
                int pos = rest % shape[dm] + dirs[dm + dr * ndim];
                if (pos < 0 || pos >= shape[dm]) {{
                    k = -1;
                    break;
                }}
                k += pos * stride;
                rest /= shape[dm];
                stride *= shape[dm];
            }}
            if (k < 0) continue;
            if (y[k] < 0) continue;
            {x_condition}
            while (1) {{
                while (j != y[j]) {{ j = y[j]; }}
                while (k != y[k]) {{ k = y[k]; }}
                if (j == k) break;
                if (j < k) {{
                    {int_t} old = atomicCAS( &y[k], (Y)k, (Y)j );
                    if (old == k) break;
                    k = old;
                }}
                else {{
                    {int_t} old = atomicCAS( &y[j], (Y)j, (Y)k );
                    if (old == j) break;
                    j = old;
                }}
            }}
        }}
        """.format(
        x_condition=x_condition, int_t=int_t
    )

    return cupy.ElementwiseKernel(
        in_params, "raw Y y", code, "cupyx_nd_label_connect",
    )


def _kernel_count():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw int32 count",
        """
        if (y[i] < 0) continue;
        int j = i;
        while (j != y[j]) { j = y[j]; }
        if (j != i) y[i] = j;
        else atomicAdd(&count[0], 1);
        """,
        "cupyx_nd_label_count",
    )


def _kernel_labels():
    return cupy.ElementwiseKernel(
        "",
        "raw Y y, raw int32 count, raw int32 labels",
        """
        if (y[i] != i) continue;
        int j = atomicAdd(&count[1], 1);
        labels[j] = i;
        """,
        "cupyx_nd_label_labels",
    )


def _kernel_finalize():
    return cupy.ElementwiseKernel(
        "int32 maxlabel",
        "raw int32 labels, raw Y y",
        """
        if (y[i] < 0) {
            y[i] = 0;
            continue;
        }
        int yi = y[i];
        int j_min = 0;
        int j_max = maxlabel - 1;
        int j = (j_min + j_max) / 2;
        while (j_min < j_max) {
            if (yi == labels[j]) break;
            if (yi < labels[j]) j_max = j - 1;
            else j_min = j + 1;
            j = (j_min + j_max) / 2;
        }
        y[i] = j + 1;
        """,
        "cupyx_nd_label_finalize",
    )


int_types = {
    "i": "int",
    "H": "unsigned short",
    "I": "unsigned int",
    "L": "unsigned long long",
}


def _label(x, structure, y, greyscale_mode=False):
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]
    dirs = cupy.array(dirs, dtype=numpy.int32)
    ndirs = indxs.shape[0]
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    count = cupy.zeros(2, dtype=numpy.int32)
    _kernel_init()(x, y)
    try:
        int_t = int_types[y.dtype.char]
    except KeyError:
        raise ValueError("y must have int32, uint16, uint32 or uint64 dtype")
    if int_t != "int":
        raise NotImplementedError(
            "Currently only 32-bit integer case is implemented"
        )
    if greyscale_mode:
        _kernel_connect(True, int_t)(
            x, y_shape, dirs, ndirs, x.ndim, y, size=y.size
        )
    else:
        _kernel_connect(False, int_t)(
            y_shape, dirs, ndirs, x.ndim, y, size=y.size
        )
    _kernel_count()(y, count, size=y.size)
    maxlabel = int(count[0])  # synchronize
    labels = cupy.empty(maxlabel, dtype=numpy.int32)
    _kernel_labels()(y, count, labels, size=y.size)
    _kernel_finalize()(maxlabel, cupy.sort(labels), y, size=y.size)
    return maxlabel


def label(input, structure=None, output=None, *, greyscale_mode=False):
    """Labels features in an array

    Args:
        input (cupy.ndarray): The input array.
        structure (array_like or None): A structuring element that defines
            feature connections. ```structure``` must be centersymmetric. If
            None, structure is automatically generated with a squared
            connectivity equal to one.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        greyscale_mode (boolean): If True, the function will behave like
            ``skimage.measure.label`` where differening non-background values
            will receive different labels.

    Returns:
        label (cupy.ndarray): An integer array where each unique feature in
            ```input``` has a unique label in the array.
        num_features (int): Number of features found.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`scipy.ndimage.label`
    """
    assert isinstance(input, cupy.ndarray)
    if input.dtype.char in "FD":
        raise TypeError("Complex type not supported")
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1, on_cpu=True)
    elif isinstance(structure, cupy.ndarray):
        structure = cupy.asnumpy(structure)
    structure = numpy.array(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError("structure and input must have equal rank")
    for i in structure.shape:
        if i != 3:
            raise ValueError("structure dimensions must be equal to 3")

    if isinstance(output, cupy.ndarray):
        if output.shape != input.shape:
            raise ValueError("output shape not correct")
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = cupy.empty(input.shape, numpy.int32)
        else:
            output = cupy.empty(input.shape, output)

    if input.size == 0:
        # 0-dim array
        maxlabel = 0
    elif input.ndim == 0:
        # 0-dim array
        maxlabel = 0 if input.item() == 0 else 1  # synchronize
        output[...] = maxlabel
    else:
        if output.dtype != numpy.int32:
            y = cupy.empty(input.shape, numpy.int32)
        else:
            y = output
        maxlabel = _label(input, structure, y, greyscale_mode=greyscale_mode)
        if output.dtype != numpy.int32:
            output[...] = y[...]

    if caller_provided_output:
        return maxlabel
    else:
        return output, maxlabel
