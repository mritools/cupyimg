"""Implementations of functions from the NumPy API via upfirdn.

"""


import cupy


__all__ = ["convolve", "correlate"]


def convolve(a, v, mode="full", *, xp=None):
    """
    Returns the discrete, linear convolution of two one-dimensional sequences.

    The convolution operator is often seen in signal processing, where it
    models the effect of a linear time-invariant system on a signal [1]_.  In
    probability theory, the sum of two independent random variables is
    distributed according to the convolution of their individual
    distributions.

    If `v` is longer than `a`, the arrays are swapped before computation.

    Parameters
    ----------
    a : (N,) array_like
        First one-dimensional input array.
    v : (M,) array_like
        Second one-dimensional input array.
    mode : {'full', 'valid', 'same'}, optional
        'full':
          By default, mode is 'full'.  This returns the convolution
          at each point of overlap, with an output shape of (N+M-1,). At
          the end-points of the convolution, the signals do not overlap
          completely, and boundary effects may be seen.

        'same':
          Mode 'same' returns output of length ``max(M, N)``.  Boundary
          effects are still visible.

        'valid':
          Mode 'valid' returns output of length
          ``max(M, N) - min(M, N) + 1``.  The convolution product is only given
          for points where the signals overlap completely.  Values outside
          the signal boundary have no effect.

    Returns
    -------
    out : ndarray
        Discrete, linear convolution of `a` and `v`.

    See Also
    --------
    scipy.signal.fftconvolve : Convolve two arrays using the Fast Fourier
                               Transform.
    scipy.linalg.toeplitz : Used to construct the convolution operator.
    polymul : Polynomial multiplication. Same output as convolve, but also
              accepts poly1d objects as input.

    Notes
    -----
    The main difference in functionality as compared to NumPy is that this
    version only operates using np.float32, np.float64, np.complex64 and
    np.complex128.

    The discrete convolution operation is defined as

    .. math:: (a * v)[n] = \\sum_{m = -\\infty}^{\\infty} a[m] v[n - m]

    It can be shown that a convolution :math:`x(t) * y(t)` in time/space
    is equivalent to the multiplication :math:`X(f) Y(f)` in the Fourier
    domain, after appropriate padding (padding is necessary to prevent
    circular convolution).  Since multiplication is more efficient (faster)
    than convolution, the function `scipy.signal.fftconvolve` exploits the
    FFT to calculate the convolution of large data-sets.

    References
    ----------
    .. [1] Wikipedia, "Convolution",
        https://en.wikipedia.org/wiki/Convolution

    Examples
    --------
    Note how the convolution operator flips the second array
    before "sliding" the two across one another:

    >>> from cupyimg.numpy import convolve
    >>> convolve(cupy.array([1, 2, 3]), cupy.array([0, 1, 0.5]))
    array([0. , 1. , 2.5, 4. , 1.5])

    Only return the middle values of the convolution.
    Contains boundary effects, where zeros are taken
    into account:

    >>> convolve(cupy.array([1, 2, 3]), cupy.array([0, 1, 0.5]), 'same')
    array([1. ,  2.5,  4. ])

    The two arrays are of the same length, so there
    is only one position where they completely overlap:

    >>> convolve(cupy.array([1, 2, 3]),cupy.array([0, 1, 0.5]), 'valid')
    array([2.5])
    """
    from fast_upfirdn.cupy import convolve1d

    a = cupy.array(a, copy=False, ndmin=1)
    v = cupy.array(v, copy=False, ndmin=1)

    if len(a) < len(v):
        v, a = a, v
    if len(a) == 0:
        raise ValueError("a cannot be empty")
    if len(v) == 0:
        raise ValueError("v cannot be empty")
    if mode == "full":
        offset = 0
        size = len(a) + len(v) - 1
        crop = False
    elif mode == "same":
        offset = (len(v) - 1) // 2  # needed - 1 here to match NumPy
        size = len(a)
        crop = True
    elif mode == "valid":
        offset = len(v) - 1
        size = len(a) - len(v) + 1
        crop = True
    else:
        raise ValueError("unrecognized mode: {}".format(mode))

    out = convolve1d(v, a, offset=offset, mode="constant", cval=0, crop=crop)
    return out[:size]


def correlate(a, v, mode="valid", *, xp=None):
    """
    Cross-correlation of two 1-dimensional sequences.

    This function computes the correlation as generally defined in signal
    processing texts::

        c_{av}[k] = sum_n a[n+k] * conj(v[n])

    with a and v sequences being zero-padded where necessary and conj being
    the conjugate.

    Parameters
    ----------
    a, v : array_like
        Input sequences.
    mode : {'valid', 'same', 'full'}, optional
        Refer to the `convolve` docstring.  Note that the default
        is 'valid', unlike `convolve`, which uses 'full'.
    old_behavior : bool
        `old_behavior` was removed in NumPy 1.10. If you need the old
        behavior, use `multiarray.correlate`.

    Returns
    -------
    out : ndarray
        Discrete cross-correlation of `a` and `v`.

    See Also
    --------
    convolve : Discrete, linear convolution of two one-dimensional sequences.

    Notes
    -----
    The main difference in functionality vs. NumPy is that this version only
    operates using np.float32, np.float64, np.complex64 and np.complex128.

    The definition of correlation above is not unique and sometimes correlation
    may be defined differently. Another common definition is::

        c'_{av}[k] = sum_n a[n] conj(v[n+k])

    which is related to ``c_{av}[k]`` by ``c'_{av}[k] = c_{av}[-k]``.

    Examples
    --------
    >>> from cupyimg.numpy import correlate
    >>> correlate(cupy.array([1, 2, 3]), cupy.array([0, 1, 0.5]))
    array([3.5])
    >>> correlate(cupy.array([1, 2, 3]), cupy.array([0, 1, 0.5]), "same")
    array([2. ,  3.5,  3. ])
    >>> correlate(cupy.array([1, 2, 3]), cupy.array([0, 1, 0.5]), "full")
    array([0.5,  2. ,  3.5,  3. ,  0. ])

    Using complex sequences:

    >>> correlate(cupy.array([1+1j, 2, 3-1j]), cupy.array([0, 1, 0.5j]), 'full')
    array([ 0.5-0.5j,  1.0+0.j ,  1.5-1.5j,  3.0-1.j ,  0.0+0.j ])

    Note that you get the time reversed, complex conjugated result
    when the two input sequences change places, i.e.,
    ``c_{va}[k] = c^{*}_{av}[-k]``:

    >>> correlate(cupy.array([0, 1, 0.5j]), cupy.array([1+1j, 2, 3-1j]), 'full')
    array([ 0.0+0.j ,  3.0+1.j ,  1.5+1.5j,  1.0+0.j ,  0.5+0.5j])

    """
    v = v[::-1]
    if v.dtype.kind == "c":
        v = cupy.conj(v)
    return convolve(a, v, mode=mode, xp=xp)
