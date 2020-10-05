__all__ = ["upfirdn"]

_upfirdn_modes = [
    "constant",
    "wrap",
    "edge",
    "smooth",
    "symmetric",
    "reflect",
    "antisymmetric",
    "antireflect",
    "line",
]


def upfirdn(
    h,
    x,
    up=1,
    down=1,
    axis=-1,
    mode="zero",
    cval=0,
    *,
    prepadded=False,
    out=None,
    offset=0,
    crop=False,
    take=None,
):
    """Upsample, FIR filter, and downsample

    Parameters
    ----------
    h : array_like
        1-D FIR (finite-impulse response) filter coefficients.
    x : array_like
        Input signal array.
    up : int, optional
        Upsampling rate. Default is 1.
    down : int, optional
        Downsampling rate. Default is 1.
    axis : int, optional
        The axis of the input data array along which to apply the
        linear filter. The filter is applied to each subarray along
        this axis. Default is -1.
    mode : str, optional
        The signal extension mode to use. The set
        ``{"constant", "symmetric", "reflect", "edge", "wrap"}`` correspond to
        modes provided by `numpy.pad`. ``"smooth"`` implements a smooth
        extension by extending based on the slope of the last 2 points at each
        end of the array. ``"antireflect"`` and ``"antisymmetric"`` are
        anti-symmetric versions of ``"reflect"`` and ``"symmetric"``. The mode
        `"line"` extends the signal based on a linear trend defined by the
        first and last points along the ``axis``.
    cval : float, optional
        The constant value to use when ``mode == "constant"``.

    Additional Parameters
    ---------------------
    prepadded : bool, optional
        If this is True, it is assumed that the internal computation
        ``h = _pad_h(h, up=up)`` has already been performed on ``h``.
    out : ndarray
        TODO
    offset : int, optional
        TODO
    crop : bool, optional
        TODO
    take : int or None, optional
        TODO

    Returns
    -------
    y : ndarray
        The output signal array. Dimensions will be the same as `x` except
        for along `axis`, which will change size according to the `h`,
        `up`,  and `down` parameters.

    Notes
    -----
    The algorithm is an implementation of the block diagram shown on page 129
    of the Vaidyanathan text [1]_ (Figure 4.3-8d).

    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,
       Prentice Hall, 1993.

    The direct approach of upsampling by factor of P with zero insertion,
    FIR filtering of length ``N``, and downsampling by factor of Q is
    O(N*Q) per output sample. The polyphase implementation used here is
    O(N/P).

    The parameters under "Additional Parameters" above are not supported in the
    SciPy implementation.

    Examples
    --------
    Simple operations:

    >>> from cupyimg.scipy.signal import upfirdn
    >>> upfirdn(cupy.ones(3), cupy.ones(3))   # FIR filter
    array([ 1.,  2.,  3.,  2.,  1.])
    >>> upfirdn(cupy.array([1]), cupy.array([1, 2, 3]), 3)  # upsampling with zeros insertion
    array([ 1.,  0.,  0.,  2.,  0.,  0.,  3.,  0.,  0.])
    >>> upfirdn(cupy.array([1, 1, 1]), cupy.array([1, 2, 3]), 3)  # upsampling with sample-and-hold
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  3.,  3.,  3.])
    >>> upfirdn(cupy.array([.5, 1, .5]), cupy.array([1, 1, 1]), 2)  # linear interpolation
    array([ 0.5,  1. ,  1. ,  1. ,  1. ,  1. ,  0.5,  0. ])
    >>> upfirdn(cupy.array([1]), cupy.arange(10), 1, 3)  # decimation by 3
    array([ 0.,  3.,  6.,  9.])
    >>> upfirdn(cupy.array([.5, 1, .5]), cupy.arange(10), 2, 3)  # linear interp, rate 2/3
    array([ 0. ,  1. ,  2.5,  4. ,  5.5,  7. ,  8.5,  0. ])

    Apply a single filter to multiple signals:

    >>> x = cupy.reshape(cupy.arange(8), (4, 2))
    >>> x
    array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]])

    Apply along the last dimension of ``x``:

    >>> h = cupy.array([1, 1])
    >>> upfirdn(h, x, 2)
    array([[ 0.,  0.,  1.,  1.],
           [ 2.,  2.,  3.,  3.],
           [ 4.,  4.,  5.,  5.],
           [ 6.,  6.,  7.,  7.]])

    Apply along the 0th dimension of ``x``:

    >>> upfirdn(h, x, 2, axis=0)
    array([[ 0.,  1.],
           [ 0.,  1.],
           [ 2.,  3.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 6.,  7.]])

    """
    from fast_upfirdn.cupy import upfirdn as upfirdn_cupy

    upfirdn_kwargs = dict(
        up=up,
        down=down,
        axis=axis,
        mode=mode,
        cval=cval,
        offset=offset,
        crop=int(crop),
        take=take,
    )

    y = upfirdn_cupy(h, x, prepadded=prepadded, out=out, **upfirdn_kwargs)
    return y
