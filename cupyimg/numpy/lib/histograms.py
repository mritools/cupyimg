import functools
import operator
import warnings

import numpy

import cupy
from cupyimg import numpy as cnp

from cupy import core

__all__ = ["histogram", "histogram2d", "histogramdd"]

# rename range for use in functions that take a range argument
_range = range

_preamble = """
__device__ long long atomicAdd(long long *address, long long val) {
    return atomicAdd(reinterpret_cast<unsigned long long*>(address),
                     static_cast<unsigned long long>(val));
}"""

# TODO(unno): use searchsorted
_histogram_kernel = core.ElementwiseKernel(
    "S x, raw T bins, int32 n_bins",
    "raw U y",
    """
    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    int high = n_bins - 1;
    int low = 0;

    while (high - low > 1) {
        int mid = (high + low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    atomicAdd(&y[low], U(1));
    """,
    preamble=_preamble,
)


_weighted_histogram_kernel = core.ElementwiseKernel(
    "S x, raw T bins, int32 n_bins, raw W weights",
    "raw Y y",
    """
    if (x < bins[0] or bins[n_bins - 1] < x) {
        return;
    }
    int high = n_bins - 1;
    int low = 0;

    while (high - low > 1) {
        int mid = (high + low) / 2;
        if (bins[mid] <= x) {
            low = mid;
        } else {
            high = mid;
        }
    }
    atomicAdd(&y[low], (Y)weights[i]);
    """,
    preamble=_preamble,
)


def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """

    # Ensure that the array is a "subtractable" dtype
    if a.dtype == cupy.bool_:
        warnings.warn(
            "Converting input from {} to {} for compatibility.".format(
                a.dtype, cupy.uint8
            ),
            RuntimeWarning,
            stacklevel=3,
        )
        a = a.astype(cupy.uint8)

    if weights is not None:
        if not isinstance(weights, cupy.ndarray):
            raise ValueError("weights must be a cupy.ndarray")
        if weights.shape != a.shape:
            raise ValueError("weights should have the same shape as a.")
        weights = weights.ravel()
    a = a.ravel()
    return a, weights


def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError("max must be larger than min in range parameter.")
        if not (numpy.isfinite(first_edge) and numpy.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(
                    first_edge, last_edge
                )
            )
    elif a.size == 0:
        first_edge = 0.0
        last_edge = 1.0
    else:
        first_edge = float(a.min())
        last_edge = float(a.max())
        if not (cupy.isfinite(first_edge) and cupy.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(
                    first_edge, last_edge
                )
            )

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def _get_bin_edges(a, bins, range):
    """
    Computes the bins used internally by `histogram`.

    Args:
        a (ndarray): Ravelled data array
        bins (int or ndarray): Forwarded argument from `histogram`.
        range (None or tuple): Forwarded argument from `histogram`.

    Returns:
        bin_edges (ndarray): Array of bin edges
        uniform_bins (Number, Number, int): The upper bound, lowerbound, and
        number of bins, used in the implementation of `histogram` that works on
        uniform bins.
    """
    # parse the overloaded bins argument
    n_equal_bins = None
    bin_edges = None

    if isinstance(bins, int):  # cupy.ndim(bins) == 0:
        try:
            n_equal_bins = operator.index(bins)
        except TypeError:
            raise TypeError("`bins` must be an integer, a string, or an array")
        if n_equal_bins < 1:
            raise ValueError("`bins` must be positive, when an integer")

        first_edge, last_edge = _get_outer_edges(a, range)

    elif isinstance(bins, cupy.ndarray):
        if bins.ndim == 1:  # cupy.ndim(bins) == 0:
            bin_edges = cupy.asarray(bins)
            if (bin_edges[:-1] > bin_edges[1:]).any():  # synchronize!
                raise ValueError(
                    "`bins` must increase monotonically, when an array"
                )

    elif isinstance(bins, str):
        raise NotImplementedError("only integer and array bins are implemented")

    if n_equal_bins is not None:
        # numpy's gh-10322 means that type resolution rules are dependent on
        # array shapes. To avoid this causing problems, we pick a type now and
        # stick with it throughout.
        bin_type = cupy.result_type(first_edge, last_edge, a)
        if cupy.issubdtype(bin_type, cupy.integer):
            bin_type = cupy.result_type(bin_type, float)

        # bin edges must be computed
        bin_edges = cupy.linspace(
            first_edge,
            last_edge,
            n_equal_bins + 1,
            endpoint=True,
            dtype=bin_type,
        )
        return bin_edges, (first_edge, last_edge, n_equal_bins)
    else:
        return bin_edges, None


def histogram(x, bins=10, range=None, weights=None, density=False):
    """Computes the histogram of a set of data.

    Args:
        x (cupy.ndarray): Input array.
        bins (int or cupy.ndarray): If ``bins`` is an int, it represents the
            number of bins. If ``bins`` is an :class:`~cupy.ndarray`, it
            represents a bin edges.
        range (2-tuple of float, optional): The lower and upper range of the
            bins.  If not provided, range is simply ``(a.min(), a.max())``.
            Values outside the range are ignored. The first element of the
            range must be less than or equal to the second. `range` affects the
            automatic bin computation as well. While bin width is computed to
            be optimal based on the actual data within `range`, the bin count
            will fill the entire range including portions containing no data.
        density (bool, optional): If False, the default, returns the number of
            samples in each bin. If True, returns the probability *density*
            function at the bin, ``bin_count / sample_count / bin_volume``.
        weights (cupy.ndarray, optional): An array of weights, of the same
            shape as `x`.  Each value in `x` only contributes its associated
            weight towards the bin count (instead of 1).
    Returns:
        tuple: ``(hist, bin_edges)`` where ``hist`` is a :class:`cupy.ndarray`
        storing the values of the histogram, and ``bin_edges`` is a
        :class:`cupy.ndarray` storing the bin edges.

    .. warning::

        This function may synchronize the device.

    .. seealso:: :func:`numpy.histogram`
    """

    if x.dtype.kind == "c":
        # TODO(unno): comparison between complex numbers is not implemented
        raise NotImplementedError("complex number is not supported")

    if not isinstance(x, cupy.ndarray):
        raise ValueError("x must be a cupy.ndarray")

    x, weights = _ravel_and_check_weights(x, weights)
    bin_edges, uniform_bins = _get_bin_edges(x, bins, range)

    if weights is None:
        y = cupy.zeros(bin_edges.size - 1, dtype="l")
        _histogram_kernel(x, bin_edges, bin_edges.size, y)
    else:
        simple_weights = cupy.can_cast(
            weights.dtype, cupy.double
        ) or cupy.can_cast(weights.dtype, complex)
        if not simple_weights:
            # object dtype such as Decimal are supported in NumPy, but not here
            raise NotImplementedError(
                "only weights with dtype that can be cast to float or complex "
                "are supported"
            )
        if weights.dtype.kind == "c":
            y = cupy.zeros(bin_edges.size - 1, dtype=complex)
            _weighted_histogram_kernel(
                x, bin_edges, bin_edges.size, weights.real, y.real
            )
            _weighted_histogram_kernel(
                x, bin_edges, bin_edges.size, weights.imag, y.imag
            )
        else:
            if weights.dtype.kind in "bui":
                y = cupy.zeros(bin_edges.size - 1, dtype=int)
            else:
                y = cupy.zeros(bin_edges.size - 1, dtype=float)
            _weighted_histogram_kernel(x, bin_edges, bin_edges.size, weights, y)

    if density:
        db = cupy.array(cupy.diff(bin_edges), float)
        return y / db / y.sum(), bin_edges
    return y, bin_edges


def histogramdd(sample, bins=10, range=None, weights=None, density=False):
    """
    Compute the multidimensional histogram of some data.

    Parameters
    ----------
    sample : (N, D) array, or (D, N) array_like
        The data to be histogrammed.

        Note the unusual interpretation of sample when an array_like:

        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(cupy.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.

        The first form should be preferred.

    bins : sequence or int, optional
        The bin specification:

        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).

    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        The values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.

    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights
        for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.

    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram

    Examples
    --------
    >>> r = cupy.random.randn(100,3)
    >>> H, edges = cupy.histogramdd(r, bins = (5, 8, 4))
    >>> H.shape, edges[0].size, edges[1].size, edges[2].size
    ((5, 8, 4), 6, 9, 5)

    """
    if isinstance(sample, cupy.ndarray):
        # Sample is an ND-array.
        if sample.ndim == 1:
            sample = sample[:, cupy.newaxis]
        nsamples, ndim = sample.shape
    else:
        sample = cupy.stack(sample, axis=-1)
        nsamples, ndim = sample.shape

    nbin = numpy.empty(ndim, int)
    edges = ndim * [None]
    dedges = ndim * [None]
    if weights is not None:
        weights = cupy.asarray(weights)

    try:
        nbins = len(bins)
        if nbins != ndim:
            raise ValueError(
                "The dimension of bins must be equal to the dimension of the "
                " sample x."
            )
    except TypeError:
        # bins is an integer
        bins = ndim * [bins]

    # normalize the range argument
    if range is None:
        range = (None,) * ndim
    elif len(range) != ndim:
        raise ValueError("range argument must have one entry per dimension")

    # Create edge arrays
    for i in _range(ndim):
        if cnp.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    "`bins[{}]` must be positive, when an integer".format(i)
                )
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            num = int(bins[i] + 1)  # synchronize!
            edges[i] = cupy.linspace(smin, smax, num)
        elif cnp.ndim(bins[i]) == 1:
            edges[i] = cupy.asarray(bins[i])
            if (edges[i][:-1] > edges[i][1:]).any():
                raise ValueError(
                    "`bins[{}]` must be monotonically increasing, when an array".format(
                        i
                    )
                )
        else:
            raise ValueError(
                "`bins[{}]` must be a scalar or 1d array".format(i)
            )

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = cupy.diff(edges[i])

    # Compute the bin number each sample falls into.
    ncount = tuple(
        # avoid cupy.digitize to work around gh-11022
        cupy.searchsorted(edges[i], sample[:, i], side="right")
        for i in _range(ndim)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(ndim):
        # Find which points are on the rightmost edge.
        on_edge = sample[:, i] == edges[i][-1]
        # Shift these points one bin to the left.
        ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = cnp.ravel_multi_index(ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.
    hist = cupy.bincount(xy, weights, minlength=numpy.prod(nbin))

    # Shape into a proper matrix
    hist = hist.reshape(nbin)

    # This preserves the (bad) behavior observed in gh-7845, for now.
    hist = hist.astype(float)  # Note: NumPy uses casting='safe' here too

    # Remove outliers (indices 0 and -1 for each dimension).
    core = ndim * (slice(1, -1),)
    hist = hist[core]

    if density:
        # calculate the probability density function
        s = hist.sum()
        for i in _range(ndim):
            shape = [1] * ndim
            shape[i] = nbin[i] - 2
            hist = hist / dedges[i].reshape(shape)
        hist /= s

    if any(hist.shape != numpy.asarray(nbin) - 2):
        raise RuntimeError("Internal Shape Error")
    return hist, edges


def histogram2d(x, y, bins=10, range=None, weights=None, density=None):
    """Compute the bi-dimensional histogram of two data samples.

    See documentation for numpy.histogram2d
    """
    try:
        n = len(bins)
    except TypeError:
        n = 1

    if n != 1 and n != 2:
        if isinstance(bins, cupy.ndarray):
            xedges = yedges = bins
            bins = [xedges, yedges]
        else:
            raise ImportError("array-like bins not supported in CuPy")

    hist, edges = histogramdd([x, y], bins, range, weights, density)
    return hist, edges[0], edges[1]
