import math
import timeit

import cupy
import numpy as np
from fast_upfirdn import upfirdn_out_len, upfirdn_modes
from cupyx.scipy import fft as sp_fft
from cupyimg.scipy import ndimage as ndi
from cupyimg._misc import _prod

# from cupyimg import numpy as cnp
try:
    # TODO: duplicate here instead of importing this private function
    from scipy.fft._helper import _init_nd_shape_and_axes
except ImportError:
    from scipy.fftpack.helper import _init_nd_shape_and_axes
from scipy.signal import get_window
from scipy.special import lambertw

from ._upfirdn import upfirdn

# TODO: add next_fast_len to cupyx.scipy.fft
try:
    from scipy.fft import next_fast_len
except ImportError:
    from scipy.fftpack import next_fast_len


__all__ = [
    "choose_conv_method",
    "convolve",
    "correlate",
    "convolve2d",
    "correlate2d",
    "fftconvolve",
    "hilbert",
    "hilbert2",
    "oaconvolve",
    "resample",
    "resample_poly",
    "wiener",
]


_modedict = {"valid": 0, "same": 1, "full": 2}

# _boundarydict = {'fill': 0, 'pad': 0, 'wrap': 2, 'circular': 2, 'symm': 1,
#                  'symmetric': 1, 'reflect': 4}


# convert convolve2d boundary names to the corresponding scipy.ndimage one
_ndi_boundarydict = {
    "fill": "constant",
    "pad": "constant",
    "wrap": "wrap",
    "circular": "wrap",
    "symm": "reflect",
    "symmetric": "reflect",
}

# convert convolve2d boundary name to the corresponding numpy.pad one
_np_pad_boundarydict = {
    "fill": "constant",
    "pad": "constant",
    "wrap": "wrap",
    "circular": "wrap",
    "symm": "symmetric",
    "symmetric": "symmetric",
}


def _convolveND(in1, in2, mode="same", boundary="fill", fillvalue=0):
    origin = []
    for s in in2.shape:
        if s % 2:
            origin.append(0)
        else:
            origin.append(-1)
    origin = tuple(origin)

    if not np.isscalar(fillvalue):
        raise ValueError("non-scalar fillvalue not supported")
    if fillvalue and np.iscomplexobj(fillvalue):
        raise ValueError("complex-valued fillvalue not supported")

    ndi_mode = _ndi_boundarydict[boundary]
    if mode == "full":
        pad_mode = _np_pad_boundarydict[boundary]
        if pad_mode == "constant":
            pad_width = [
                ((s - 1) // 2, math.ceil((s - 1) / 2)) for s in in2.shape
            ]
            sl_out = None
            in1 = cupy.pad(
                in1,
                pad_width=pad_width,
                mode=pad_mode,
                constant_values=fillvalue,
            )
        else:
            pad_width = [((s - 1), (s - 1)) for s in in2.shape]

            def _get_sl(s):
                if s == 1:
                    return slice(None)
                elif s == 2:
                    return slice(1, None)
                else:
                    return slice(
                        (s - 1) - (s - 1) // 2,
                        -(s - 1) + math.ceil((s - 1) / 2),
                    )

            sl_out = tuple([_get_sl(s) for s in in2.shape])
            in1 = cupy.pad(in1, pad_width=pad_width, mode=pad_mode)
    out = ndi.convolve(
        in1,
        in2,
        mode=ndi_mode,
        origin=origin,
        cval=fillvalue,
        dtype_mode="numpy",
    )
    if mode == "valid":
        # shape_out = tuple([i - j for i, j in zip(in1.shape, in2.shape)])
        sl_out = []
        for s in in2.shape:
            if s == 1:
                sl = slice(None)
            elif s == 2:
                sl = slice(1, None)
            else:
                sl = slice(math.ceil((s - 1) / 2), -math.floor((s - 1) / 2))
            sl_out.append(sl)
        sl_out = tuple(sl_out)
        out = out[sl_out]
    elif mode == "full" and sl_out is not None:
        out = out[sl_out]
    return out


def _correlateND(in1, in2, mode="same"):
    origin = (0,) * in1.ndim

    if mode == "full":
        pad_width = [((s - 1) // 2, math.ceil((s - 1) / 2)) for s in in2.shape]
        in1 = cupy.pad(in1, pad_width=pad_width)

    out = ndi.correlate(
        in1, in2, mode="constant", origin=origin, cval=0, dtype_mode="numpy"
    )
    if mode == "valid":
        sl_out = []
        for s in in2.shape:
            if s == 1:
                sl = slice(None)
            elif s == 2:
                sl = slice(1, None)
            else:
                sl = slice(math.ceil((s - 1) / 2), -((s - 1) // 2))
            sl_out.append(sl)
        sl_out = tuple(sl_out)
        out = out[sl_out]
    return out


# def _valfrommode(mode):
#     try:
#         return _modedict[mode]
#     except KeyError:
#         raise ValueError("Acceptable mode flags are 'valid',"
#                          " 'same', or 'full'.")


# def _bvalfromboundary(boundary):
#     try:
#         return _boundarydict[boundary] << 2
#     except KeyError:
#         raise ValueError("Acceptable boundary flags are 'fill', 'circular' "
#                          "(or 'wrap'), and 'symmetric' (or 'symm').")


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """Determine if inputs arrays need to be swapped in `"valid"` mode.

    If in `"valid"` mode, returns whether or not the input arrays need to be
    swapped depending on whether `shape1` is at least as large as `shape2` in
    every calculated dimension.

    This is important for some of the correlation and convolution
    implementations in this module, where the larger array input needs to come
    before the smaller array input when operating in this mode.

    Note that if the mode provided is not 'valid', False is immediately
    returned.

    """
    if mode != "valid":
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least "
            "as large as the other in every dimension"
        )

    return not ok1


def correlate(in1, in2, mode="full", method="auto"):
    r"""
    Cross-correlate two N-dimensional arrays.

    Cross-correlate `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the correlation.

        ``direct``
           The correlation is determined directly from sums, the definition of
           correlation.
        ``fft``
           The Fast Fourier Transform is used to perform the correlation more
           quickly (only available for numerical arrays.)
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See `convolve` Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    correlate : array
        An N-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    See Also
    --------
    choose_conv_method : contains more documentation on `method`.

    Notes
    -----
    The correlation z of two d-dimensional arrays x and y is defined as::

        z[...,k,...] = sum[..., i_l, ...] x[..., i_l,...] * conj(y[..., i_l - k,...])

    This way, if x and y are 1-D arrays and ``z = correlate(x, y, 'full')``
    then

    .. math::

          z[k] = (x * y)(k - N + 1)
               = \sum_{l=0}^{||x||-1}x_l y_{l-k+N-1}^{*}

    for :math:`k = 0, 1, ..., ||x|| + ||y|| - 2`

    where :math:`||x||` is the length of ``x``, :math:`N = \max(||x||,||y||)`,
    and :math:`y_m` is 0 when m is outside the range of y.

    ``method='fft'`` only works for numerical arrays as it relies on
    `fftconvolve`. In certain cases (i.e., arrays of objects or when
    rounding integers can lose precision), ``method='direct'`` is always used.

    Examples
    --------
    Implement a matched filter using cross-correlation, to recover a signal
    that has passed through a noisy channel.

    >>> from scipy import signal
    >>> sig = cupy.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    >>> sig_noise = sig + cupy.random.randn(len(sig))
    >>> corr = signal.correlate(sig_noise, cupy.ones(128), mode='same') / 128

    >>> import matplotlib.pyplot as plt
    >>> clock = cupy.arange(64, len(sig), 128)
    >>> fig, (ax_orig, ax_noise, ax_corr) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.plot(clock, sig[clock], 'ro')
    >>> ax_orig.set_title('Original signal')
    >>> ax_noise.plot(sig_noise)
    >>> ax_noise.set_title('Signal with noise')
    >>> ax_corr.plot(corr)
    >>> ax_corr.plot(clock, corr[clock], 'ro')
    >>> ax_corr.axhline(0.5, ls=':')
    >>> ax_corr.set_title('Cross-correlated with rectangular pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        _modedict[mode]
    except KeyError:
        raise ValueError(
            "Acceptable mode flags are 'valid'," " 'same', or 'full'."
        )

    # this either calls fftconvolve or this function with method=='direct'
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == "direct":
        # fastpath to faster numpy.correlate for 1d inputs when possible
        # f _np_conv_ok(in1, in2, mode):
        #     return cnp.correlate(in1, in2, mode)

        # TODO: grlee77: check performance with in2 larger than in1, etc. too
        #                potentially modify ndimage.convolve/correlate to
        #                handle modes full and valid to avoid the need to explicitly
        #                pad within _convolvedND
        swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)

        if swapped_inputs:
            in1, in2 = in2, in1

        z = _correlateND(in1, in2, mode)
        return z

    else:
        raise ValueError(
            "Acceptable method flags are 'auto'," " 'direct', or 'fft'."
        )


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    """Handle the axes argument for frequency-domain convolution.

    Returns the inputs and axes in a standard form, eliminating redundant axes,
    swapping the inputs if necessary, and checking for various potential
    errors.

    Parameters
    ----------
    in1 : array
        First input.
    in2 : array
        Second input.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the FFTs.
    sorted_axes : bool, optional
        If `True`, sort the axes.
        Default is `False`, do not sort.

    Returns
    -------
    in1 : array
        The first input, possible swapped with the second input.
    in2 : array
        The second input, possible swapped with the first input.
    axes : list of ints
        Axes over which to compute the FFTs.

    """
    s1 = in1.shape
    s2 = in2.shape
    noaxes = axes is None

    _, axes = _init_nd_shape_and_axes(in1, shape=None, axes=axes)

    if not noaxes and not len(axes):
        raise ValueError("when provided, axes cannot be empty")

    # Axes of length 1 can rely on broadcasting rules for multipy,
    # no fft needed.
    axes = [a for a in axes if s1[a] != 1 and s2[a] != 1]

    if sorted_axes:
        axes.sort()

    if not all(
        s1[a] == s2[a] or s1[a] == 1 or s2[a] == 1
        for a in range(in1.ndim)
        if a not in axes
    ):
        raise ValueError(
            "incompatible shapes for in1 and in2:" " {0} and {1}".format(s1, s2)
        )

    # Check that input sizes are compatible with 'valid' mode.
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        # Convolution is commutative; order doesn't have any effect on output.
        in1, in2 = in2, in1

    return in1, in2, axes


def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    """Convolve two arrays in the frequency domain.

    This function implements only base the FFT-related operations.
    Specifically, it converts the signals to the frequency domain, multiplies
    them, then converts them back to the time domain.  Calculations of axes,
    shapes, convolution mode, etc. are implemented in higher level-functions,
    such as `fftconvolve` and `oaconvolve`.  Those functions should be used
    instead of this one.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    axes : array_like of ints
        Axes over which to compute the FFTs.
    shape : array_like of ints
        The sizes of the FFTs.
    calc_fast_len : bool, optional
        If `True`, set each value of `shape` to the next fast FFT length.
        Default is `False`, use `axes` as-is.

    Returns
    -------
    out : array
        An N-dimensional array containing the discrete linear convolution of
        `in1` with `in2`.

    """
    if not len(axes):
        return in1 * in2

    complex_result = in1.dtype.kind == "c" or in2.dtype.kind == "c"

    if calc_fast_len:
        # Speed up FFT by padding to optimal size.
        fshape = [next_fast_len(shape[a], not complex_result) for a in axes]
    else:
        fshape = shape

    if not complex_result:
        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
    else:
        fft, ifft = sp_fft.fftn, sp_fft.ifftn

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)

    if calc_fast_len:
        fslice = tuple([slice(sz) for sz in shape])
        ret = ret[fslice]

    return ret


def _apply_conv_mode(ret, s1, s2, mode, axes):
    """Calculate the convolution result shape based on the `mode` argument.

    Returns the result sliced to the correct size for the given mode.

    Parameters
    ----------
    ret : array
        The result array, with the appropriate shape for the 'full' mode.
    s1 : list of int
        The shape of the first input.
    s2 : list of int
        The shape of the second input.
    mode : str {'full', 'valid', 'same'}
        A string indicating the size of the output.
        See the documentation `fftconvolve` for more information.
    axes : list of ints
        Axes over which to compute the convolution.

    Returns
    -------
    ret : array
        A copy of `res`, sliced to the correct size for the given `mode`.

    """
    if mode == "full":
        return ret.copy()
    elif mode == "same":
        return _centered(ret, s1).copy()
    elif mode == "valid":
        shape_valid = [
            ret.shape[a] if a not in axes else s1[a] - s2[a] + 1
            for a in range(ret.ndim)
        ]
        return _centered(ret, shape_valid).copy()
    else:
        raise ValueError(
            "acceptable mode flags are 'valid'," " 'same', or 'full'"
        )


def fftconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using FFT.

    Convolve `in1` and `in2` using the fast Fourier transform method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    but can be slower when only a few output values are needed, and can only
    output float arrays (int or object array inputs will be cast to float).

    As of v0.19, `convolve` automatically chooses this method or the direct
    method based on an estimation of which is faster.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Examples
    --------
    Autocorrelation of white noise is an impulse.

    >>> from scipy import signal
    >>> sig = cupy.random.randn(1000)
    >>> autocorr = signal.fftconvolve(sig, sig[::-1], mode='full')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(cupy.arange(-len(sig)+1,len(sig)), autocorr)
    >>> ax_mag.set_title('Autocorrelation')
    >>> fig.tight_layout()
    >>> fig.show()

    Gaussian blur implemented using FFT convolution.  Notice the dark borders
    around the image, due to the zero-padding beyond its boundaries.
    The `convolve2d` function allows for other types of image boundaries,
    but is far slower.

    >>> from scipy import misc
    >>> face = misc.face(gray=True)
    >>> kernel = cupy.outer(signal.gaussian(70, 8), signal.gaussian(70, 8))
    >>> blurred = signal.fftconvolve(face, kernel, mode='same')

    >>> fig, (ax_orig, ax_kernel, ax_blurred) = plt.subplots(3, 1,
    ...                                                      figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_kernel.imshow(kernel, cmap='gray')
    >>> ax_kernel.set_title('Gaussian kernel')
    >>> ax_kernel.set_axis_off()
    >>> ax_blurred.imshow(blurred, cmap='gray')
    >>> ax_blurred.set_title('Blurred')
    >>> ax_blurred.set_axis_off()
    >>> fig.show()

    """
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return cupy.array([])

    in1, in2, axes = _init_freq_conv_axes(
        in1, in2, mode, axes, sorted_axes=False
    )

    s1 = in1.shape
    s2 = in2.shape

    shape = [
        max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1
        for i in range(in1.ndim)
    ]

    ret = _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=True)

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _calc_oa_lens(s1, s2):
    """Calculate the optimal FFT lengths for overlapp-add convolution.

    The calculation is done for a single dimension.

    Parameters
    ----------
    s1 : int
        Size of the dimension for the first array.
    s2 : int
        Size of the dimension for the second array.

    Returns
    -------
    block_size : int
        The size of the FFT blocks.
    overlap : int
        The amount of overlap between two blocks.
    in1_step : int
        The size of each step for the first array.
    in2_step : int
        The size of each step for the first array.

    """
    # Set up the arguments for the conventional FFT approach.
    fallback = (s1 + s2 - 1, None, s1, s2)

    # Use conventional FFT convolve if sizes are same.
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback

    if s2 > s1:
        s1, s2 = s2, s1
        swapped = True
    else:
        swapped = False

    # There cannot be a useful block size if s2 is more than half of s1.
    if s2 >= s1 / 2:
        return fallback

    # Derivation of optimal block length
    # For original formula see:
    # https://en.wikipedia.org/wiki/Overlap-add_method
    #
    # Formula:
    # K = overlap = s2-1
    # N = block_size
    # C = complexity
    # e = exponential, exp(1)
    #
    # C = (N*(log2(N)+1))/(N-K)
    # C = (N*log2(2N))/(N-K)
    # C = N/(N-K) * log2(2N)
    # C1 = N/(N-K)
    # C2 = log2(2N) = ln(2N)/ln(2)
    #
    # dC1/dN = (1*(N-K)-N)/(N-K)^2 = -K/(N-K)^2
    # dC2/dN = 2/(2*N*ln(2)) = 1/(N*ln(2))
    #
    # dC/dN = dC1/dN*C2 + dC2/dN*C1
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + N/(N*ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + 1/(ln(2)*(N-K))
    # dC/dN = -K*ln(2N)/(ln(2)*(N-K)^2) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (-K*ln(2N) + (N-K)/(ln(2)*(N-K)^2)
    # dC/dN = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    #
    # Solve for minimum, where dC/dN = 0
    # 0 = (N - K*ln(2N) - K)/(ln(2)*(N-K)^2)
    # 0 * ln(2)*(N-K)^2 = N - K*ln(2N) - K
    # 0 = N - K*ln(2N) - K
    # 0 = N - K*(ln(2N) + 1)
    # 0 = N - K*ln(2Ne)
    # N = K*ln(2Ne)
    # N/K = ln(2Ne)
    #
    # e^(N/K) = e^ln(2Ne)
    # e^(N/K) = 2Ne
    # 1/e^(N/K) = 1/(2*N*e)
    # e^(N/-K) = 1/(2*N*e)
    # e^(N/-K) = K/N*1/(2*K*e)
    # N/K*e^(N/-K) = 1/(2*e*K)
    # N/-K*e^(N/-K) = -1/(2*e*K)
    #
    # Using Lambert W function
    # https://en.wikipedia.org/wiki/Lambert_W_function
    # x = W(y) It is the solution to y = x*e^x
    # x = N/-K
    # y = -1/(2*e*K)
    #
    # N/-K = W(-1/(2*e*K))
    #
    # N = -K*W(-1/(2*e*K))
    overlap = s2 - 1
    opt_size = -overlap * lambertw(-1 / (2 * math.e * overlap), k=-1).real
    block_size = next_fast_len(math.ceil(opt_size))

    # Use conventional FFT convolve if there is only going to be one block.
    if block_size >= s1:
        return fallback

    if not swapped:
        in1_step = block_size - s2 + 1
        in2_step = s2
    else:
        in1_step = s2
        in2_step = block_size - s2 + 1

    return block_size, overlap, in1_step, in2_step


def oaconvolve(in1, in2, mode="full", axes=None):
    """Convolve two N-dimensional arrays using the overlap-add method.

    Convolve `in1` and `in2` using the overlap-add method, with
    the output size determined by the `mode` argument.

    This is generally much faster than `convolve` for large arrays (n > ~500),
    and generally much faster than `fftconvolve` when one array is much
    larger than the other, but can be slower when only a few output values are
    needed or when the arrays are very similar in shape, and can only
    output float arrays (int or object array inputs will be cast to float).

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    axes : int or array_like of ints or None, optional
        Axes over which to compute the convolution.
        The default is over all axes.

    Returns
    -------
    out : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    convolve : Uses the direct convolution or FFT convolution algorithm
               depending on which is faster.
    fftconvolve : An implementation of convolution using FFT.

    Notes
    -----
    .. versionadded:: 1.4.0

    Examples
    --------
    Convolve a 100,000 sample signal with a 512-sample filter.

    >>> from cupyimg.scipy import signal
    >>> from scipy import signal as signal_sp
    >>> sig = cupy.random.randn(100000)
    >>> filt = cupy.asarray(signal_sp.firwin(512, 0.01))
    >>> fsig = signal.oaconvolve(sig, filt)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag) = plt.subplots(2, 1)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('White noise')
    >>> ax_mag.plot(fsig)
    >>> ax_mag.set_title('Filtered noise')
    >>> fig.tight_layout()
    >>> fig.show()

    References
    ----------
    .. [1] Wikipedia, "Overlap-add_method".
           https://en.wikipedia.org/wiki/Overlap-add_method
    .. [2] Richard G. Lyons. Understanding Digital Signal Processing,
           Third Edition, 2011. Chapter 13.10.
           ISBN 13: 978-0137-02741-5

    """
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return cupy.asarray([])
    elif in1.shape == in2.shape:  # Equivalent to fftconvolve
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    in1, in2, axes = _init_freq_conv_axes(
        in1, in2, mode, axes, sorted_axes=True
    )

    if not axes:
        return in1 * in2

    s1 = in1.shape
    s2 = in2.shape

    # Calculate this now since in1 is changed later
    shape_final = [
        None if i not in axes else s1[i] + s2[i] - 1 for i in range(in1.ndim)
    ]

    # Calculate the block sizes for the output, steps, first and second inputs.
    # It is simpler to calculate them all together than doing them in separate
    # loops due to all the special cases that need to be handled.
    optimal_sizes = (
        (-1, -1, s1[i], s2[i]) if i not in axes else _calc_oa_lens(s1[i], s2[i])
        for i in range(in1.ndim)
    )
    block_size, overlaps, in1_step, in2_step = zip(*optimal_sizes)

    # Fall back to fftconvolve if there is only one block in every dimension.
    if in1_step == s1 and in2_step == s2:
        return fftconvolve(in1, in2, mode=mode, axes=axes)

    # Figure out the number of steps and padding.
    # This would get too complicated in a list comprehension.
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue

        if s1[i] > in1_step[i]:
            curnstep1 = math.ceil((s1[i] + 1) / in1_step[i])
            if (block_size[i] - overlaps[i]) * curnstep1 < shape_final[i]:
                curnstep1 += 1

            curpad1 = curnstep1 * in1_step[i] - s1[i]
        else:
            curnstep1 = 1
            curpad1 = 0

        if s2[i] > in2_step[i]:
            curnstep2 = math.ceil((s2[i] + 1) / in2_step[i])
            if (block_size[i] - overlaps[i]) * curnstep2 < shape_final[i]:
                curnstep2 += 1

            curpad2 = curnstep2 * in2_step[i] - s2[i]
        else:
            curnstep2 = 1
            curpad2 = 0

        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]

    # Pad the array to a size that can be reshaped to the desired shape
    # if necessary.
    if not all(curpad == (0, 0) for curpad in pad_size1):
        in1 = cupy.pad(in1, pad_size1, mode="constant", constant_values=0)

    if not all(curpad == (0, 0) for curpad in pad_size2):
        in2 = cupy.pad(in2, pad_size2, mode="constant", constant_values=0)

    # Reshape the overlap-add parts to input block sizes.
    split_axes = [iax + i for i, iax in enumerate(axes)]
    fft_axes = [iax + 1 for iax in split_axes]

    # We need to put each new dimension before the corresponding dimension
    # being reshaped in order to get the data in the right layout at the end.
    reshape_size1 = list(in1_step)
    reshape_size2 = list(in2_step)
    for i, iax in enumerate(split_axes):
        reshape_size1.insert(iax, nsteps1[i])
        reshape_size2.insert(iax, nsteps2[i])

    in1 = in1.reshape(*reshape_size1)
    in2 = in2.reshape(*reshape_size2)

    # Do the convolution.
    fft_shape = [block_size[i] for i in axes]
    ret = _freq_domain_conv(in1, in2, fft_axes, fft_shape, calc_fast_len=False)

    # Do the overlap-add.
    for ax, ax_fft, ax_split in zip(axes, fft_axes, split_axes):
        overlap = overlaps[ax]
        if overlap is None:
            continue

        ret, overpart = cupy.split(ret, [-overlap], ax_fft)
        overpart = cupy.split(overpart, [-1], ax_split)[0]

        ret_overpart = cupy.split(ret, [overlap], ax_fft)[0]
        ret_overpart = cupy.split(ret_overpart, [1], ax_split)[1]
        ret_overpart += overpart

    # Reshape back to the correct dimensionality.
    shape_ret = [
        ret.shape[i] if i not in fft_axes else ret.shape[i] * ret.shape[i - 1]
        for i in range(ret.ndim)
        if i not in split_axes
    ]
    ret = ret.reshape(*shape_ret)

    # Slice to the correct size.
    slice_final = tuple([slice(islice) for islice in shape_final])
    ret = ret[slice_final]

    return _apply_conv_mode(ret, s1, s2, mode, axes)


def _numeric_arrays(arrays, kinds="buifc"):
    """
    See if a list of arrays are all numeric.

    Parameters
    ----------
    ndarrays : array or list of arrays
        arrays to check if numeric.
    numeric_kinds : string-like
        The dtypes of the arrays to be checked. If the dtype.kind of
        the ndarrays are not in this string the function returns False and
        otherwise returns True.
    """
    if type(arrays) == cupy.ndarray:
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True


def _conv_ops(x_shape, h_shape, mode):
    """
    Find the number of operations required for direct/fft methods of
    convolution. The direct operations were recorded by making a dummy class to
    record the number of operations by overriding ``__mul__`` and ``__add__``.
    The FFT operations rely on the (well-known) computational complexity of the
    FFT (and the implementation of ``_freq_domain_conv``).

    """
    x_size, h_size = _prod(x_shape), _prod(h_shape)
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "valid":
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "same":
        out_shape = x_shape
    else:
        raise ValueError(
            "Acceptable mode flags are 'valid',"
            " 'same', or 'full', not mode={}".format(mode)
        )

    s1, s2 = x_shape, h_shape
    if len(x_shape) == 1:
        s1, s2 = s1[0], s2[0]
        if mode == "full":
            direct_ops = s1 * s2
        elif mode == "valid":
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == "same":
            direct_ops = (
                s1 * s2 if s1 < s2 else s1 * s2 - (s2 // 2) * ((s2 + 1) // 2)
            )
    else:
        if mode == "full":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "valid":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "same":
            direct_ops = _prod(s1) * _prod(s2)

    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_ops, direct_ops


def _fftconv_faster(x, h, mode):
    """
    See if using fftconvolve or convolve is faster.

    Parameters
    ----------
    x : cupy.ndarray
        Signal
    h : cupy.ndarray
        Kernel
    mode : str
        Mode passed to convolve

    Returns
    -------
    fft_faster : bool

    Notes
    -----
    See docstring of `choose_conv_method` for details on tuning hardware.

    See pull request 11031 for more detail:
    https://github.com/scipy/scipy/pull/11031.

    """
    fft_ops, direct_ops = _conv_ops(x.shape, h.shape, mode)
    offset = -1e-3 if x.ndim == 1 else -1e-4
    constants = (
        {
            "valid": (1.89095737e-9, 2.1364985e-10, offset),
            "full": (1.7649070e-9, 2.1414831e-10, offset),
            "same": (3.2646654e-9, 2.8478277e-10, offset)
            if h.size <= x.size
            else (3.21635404e-9, 1.1773253e-8, -1e-5),
        }
        if x.ndim == 1
        else {
            "valid": (1.85927e-9, 2.11242e-8, offset),
            "full": (1.99817e-9, 1.66174e-8, offset),
            "same": (2.04735e-9, 1.55367e-8, offset),
        }
    )
    O_fft, O_direct, O_offset = constants[mode]
    return O_fft * fft_ops < O_direct * direct_ops + O_offset


def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()


def _np_conv_ok(volume, kernel, mode):
    """
    See if numpy supports convolution of `volume` and `kernel` (i.e. both are
    1D ndarrays and of the appropriate shape).  NumPy's 'same' mode uses the
    size of the larger input, while SciPy's uses the size of the first input.

    Invalid mode strings will return False and be caught by the calling func.
    """
    if volume.ndim == kernel.ndim == 1:
        if mode in ("full", "valid"):
            return True
        elif mode == "same":
            return volume.size >= kernel.size
    else:
        return False


def _timeit_fast(stmt="pass", setup="pass", repeat=3):
    """
    Returns the time the statement/function took, in seconds.

    Faster, less precise version of IPython's timeit. `stmt` can be a statement
    written as a string or a callable.

    Will do only 1 loop (like IPython's timeit) with no repetitions
    (unlike IPython) for very slow functions.  For fast functions, only does
    enough loops to take 5 ms, which seems to produce similar results (on
    Windows at least), and avoids doing an extraneous cycle that isn't
    measured.

    """
    timer = timeit.Timer(stmt, setup)

    # determine number of calls per rep so total time for 1 rep >= 5 ms
    x = 0
    for p in range(0, 10):
        number = 10 ** p
        x = timer.timeit(number)  # seconds
        if x >= 5e-3 / 10:  # 5 ms for final test, 1/10th that for this one
            break
    if x > 1:  # second
        # If it's macroscopic, don't bother with repetitions
        best = x
    else:
        number *= 10
        r = timer.repeat(repeat, number)
        best = min(r)

    sec = best / number
    return sec


# TODO: grlee77: tune this for CUDA when measure=False rather than falling
#                back to the choices made by SciPy


def choose_conv_method(in1, in2, mode="full", measure=False):
    """
    Find the fastest convolution/correlation method.

    This primarily exists to be called during the ``method='auto'`` option in
    `convolve` and `correlate`. It can also be used to determine the value of
    ``method`` for many different convolutions of the same dtype/shape.
    In addition, it supports timing the convolution to adapt the value of
    ``method`` to a particular set of inputs and/or hardware.

    Parameters
    ----------
    in1 : array_like
        The first argument passed into the convolution function.
    in2 : array_like
        The second argument passed into the convolution function.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    measure : bool, optional
        If True, run and time the convolution of `in1` and `in2` with both
        methods and return the fastest. If False (default), predict the fastest
        method using precomputed values.

    Returns
    -------
    method : str
        A string indicating which convolution method is fastest, either
        'direct' or 'fft'
    times : dict, optional
        A dictionary containing the times (in seconds) needed for each method.
        This value is only returned if ``measure=True``.

    See Also
    --------
    convolve
    correlate

    Notes
    -----
    Generally, this method is 99% accurate for 2D signals and 85% accurate
    for 1D signals for randomly chosen input sizes. For precision, use
    ``measure=True`` to find the fastest method by timing the convolution.
    This can be used to avoid the minimal overhead of finding the fastest
    ``method`` later, or to adapt the value of ``method`` to a particular set
    of inputs.

    Experiments were run on an Amazon EC2 r5a.2xlarge machine to test this
    function. These experiments measured the ratio between the time required
    when using ``method='auto'`` and the time required for the fastest method
    (i.e., ``ratio = time_auto / min(time_fft, time_direct)``). In these
    experiments, we found:

    * There is a 95% chance of this ratio being less than 1.5 for 1D signals
      and a 99% chance of being less than 2.5 for 2D signals.
    * The ratio was always less than 2.5/5 for 1D/2D signals respectively.
    * This function is most inaccurate for 1D convolutions that take between 1
      and 10 milliseconds with ``method='direct'``. A good proxy for this
      (at least in our experiments) is ``1e6 <= in1.size * in2.size <= 1e7``.

    The 2D results almost certainly generalize to 3D/4D/etc because the
    implementation is the same (the 1D implementation is different).

    All the numbers above are specific to the EC2 machine. However, we did find
    that this function generalizes fairly decently across hardware. The speed
    tests were of similar quality (and even slightly better) than the same
    tests performed on the machine to tune this function's numbers (a mid-2014
    15-inch MacBook Pro with 16GB RAM and a 2.5GHz Intel i7 processor).

    There are cases when `fftconvolve` supports the inputs but this function
    returns `direct` (e.g., to protect against floating point integer
    precision).

    .. versionadded:: 0.19

    Examples
    --------
    Estimate the fastest method for a given input:

    >>> from scipy import signal
    >>> img = cupy.random.rand(32, 32)
    >>> filter = cupy.random.rand(8, 8)
    >>> method = signal.choose_conv_method(img, filter, mode='same')
    >>> method
    'fft'

    This can then be applied to other arrays of the same dtype and shape:

    >>> img2 = cupy.random.rand(32, 32)
    >>> filter2 = cupy.random.rand(8, 8)
    >>> corr2 = signal.correlate(img2, filter2, mode='same', method=method)
    >>> conv2 = signal.convolve(img2, filter2, mode='same', method=method)

    The output of this function (``method``) works with `correlate` and
    `convolve`.

    """
    volume = cupy.asarray(in1)
    kernel = cupy.asarray(in2)

    if measure:
        times = {}
        for method in ["fft", "direct"]:
            times[method] = _timeit_fast(
                lambda: convolve(volume, kernel, mode=mode, method=method)
            )

        chosen_method = "fft" if times["fft"] < times["direct"] else "direct"
        return chosen_method, times

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds="ui") for x in [volume, kernel]]):
        max_value = int(cupy.abs(volume).max()) * int(cupy.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2 ** np.finfo("float").nmant - 1:
            return "direct"

    if _numeric_arrays([volume, kernel], kinds="b"):
        return "direct"

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return "fft"

    return "direct"


def convolve(in1, in2, mode="full", method="auto"):
    """
    Convolve two N-dimensional arrays.

    Convolve `in1` and `in2`, with the output size determined by the
    `mode` argument.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    method : str {'auto', 'direct', 'fft'}, optional
        A string indicating which method to use to calculate the convolution.

        ``direct``
           The convolution is determined directly from sums, the definition of
           convolution.
        ``fft``
           The Fourier Transform is used to perform the convolution by calling
           `fftconvolve`.
        ``auto``
           Automatically chooses direct or Fourier method based on an estimate
           of which is faster (default).  See Notes for more detail.

           .. versionadded:: 0.19.0

    Returns
    -------
    convolve : array
        An N-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    See Also
    --------
    numpy.polymul : performs polynomial multiplication (same operation, but
                    also accepts poly1d objects)
    choose_conv_method : chooses the fastest appropriate convolution method
    fftconvolve : Always uses the FFT method.
    oaconvolve : Uses the overlap-add method to do convolution, which is
                 generally faster when the input arrays are large and
                 significantly different in size.

    Notes
    -----
    By default, `convolve` and `correlate` use ``method='auto'``, which calls
    `choose_conv_method` to choose the fastest method using pre-computed
    values (`choose_conv_method` can also measure real-world timing with a
    keyword argument). Because `fftconvolve` relies on floating point numbers,
    there are certain constraints that may force `method=direct` (more detail
    in `choose_conv_method` docstring).

    Examples
    --------
    Smooth a square pulse using a Hann window:

    >>> from scipy import signal
    >>> sig = cupy.repeat([0., 1., 0.], 100)
    >>> win = signal.hann(50)
    >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
    >>> ax_orig.plot(sig)
    >>> ax_orig.set_title('Original pulse')
    >>> ax_orig.margins(0, 0.1)
    >>> ax_win.plot(win)
    >>> ax_win.set_title('Filter impulse response')
    >>> ax_win.margins(0, 0.1)
    >>> ax_filt.plot(filtered)
    >>> ax_filt.set_title('Filtered signal')
    >>> ax_filt.margins(0, 0.1)
    >>> fig.tight_layout()
    >>> fig.show()

    """
    volume = cupy.asarray(in1)
    kernel = cupy.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError(
            "volume and kernel should have the same " "dimensionality"
        )

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = np.result_type(volume.dtype, kernel.dtype)
        if result_type.kind in {"u", "i"}:
            out = cupy.around(out)
        return out.astype(result_type)
    elif method == "direct":
        # fastpath to faster numpy.convolve for 1d inputs when possible
        # if _np_conv_ok(volume, kernel, mode):
        #    return cnp.convolve(volume, kernel, mode, dtype_mode='numpy')

        return correlate(volume, _reverse_and_conj(kernel), mode, "direct")
    else:
        raise ValueError(
            "Acceptable method flags are 'auto'," " 'direct', or 'fft'."
        )


def wiener(im, mysize=None, noise=None):
    """
    Perform a Wiener filter on an N-dimensional array.

    Apply a Wiener filter to the N-dimensional array `im`.

    Parameters
    ----------
    im : ndarray
        An N-dimensional array.
    mysize : int or array_like, optional
        A scalar or an N-length list giving the size of the Wiener filter
        window in each dimension.  Elements of mysize should be odd.
        If mysize is a scalar, then this scalar is used as the size
        in each dimension.
    noise : float, optional
        The noise-power to use. If None, then noise is estimated as the
        average of the local variance of the input.

    Returns
    -------
    out : ndarray
        Wiener filtered result with the same shape as `im`.

    """
    im = cupy.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), im.ndim)

    # Estimate the local mean
    lMean = correlate(im, cupy.ones(mysize), "same") / np.prod(mysize, axis=0)

    # Estimate the local variance
    lVar = (
        correlate(im ** 2, cupy.ones(mysize), "same") / np.prod(mysize, axis=0)
        - lMean ** 2
    )

    # Estimate the noise power if needed.
    if noise is None:
        noise = cupy.mean(cupy.ravel(lVar), axis=0)

    res = im - lMean
    res *= 1 - noise / lVar
    res += lMean
    out = cupy.where(lVar < noise, lMean, res)

    return out


def convolve2d(in1, in2, mode="full", boundary="fill", fillvalue=0):
    """
    Convolve two 2-dimensional arrays.

    Convolve `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    out : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.

    Examples
    --------
    Compute the gradient of an image by 2D convolution with a complex Scharr
    operator.  (Horizontal operator is real, vertical is imaginary.)  Use
    symmetric boundary condition to avoid creating edges at the image
    boundaries.

    >>> from scipy import signal
    >>> from scipy import misc
    >>> ascent = misc.ascent()
    >>> scharr = cupy.asarray([[ -3-3j, 0-10j,  +3 -3j],
    ...                        [-10+0j, 0+ 0j, +10 +0j],
    ...                        [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
    >>> grad = signal.convolve2d(ascent, scharr, boundary='symm', mode='same')

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
    >>> ax_orig.imshow(ascent, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_mag.imshow(cupy.absolute(grad), cmap='gray')
    >>> ax_mag.set_title('Gradient magnitude')
    >>> ax_mag.set_axis_off()
    >>> ax_ang.imshow(cupy.angle(grad), cmap='hsv') # hsv is cyclic, like angles
    >>> ax_ang.set_title('Gradient orientation')
    >>> ax_ang.set_axis_off()
    >>> fig.show()

    """
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError("convolve2d inputs must both be 2D arrays")

    if _inputs_swap_needed(mode, in1.shape, in2.shape):
        in1, in2 = in2, in1
    if in1.dtype != in2.dtype:
        dtype = np.promote_types(in1.dtype, in2.dtype)
        in1 = in1.astype(dtype, copy=False)
        in2 = in2.astype(dtype, copy=False)
    out = _convolveND(
        in1, in2, mode=mode, boundary=boundary, fillvalue=fillvalue
    )
    # val = _valfrommode(mode)
    # bval = _bvalfromboundary(boundary)
    # out = sigtools._convolve2d(in1, in2, 1, val, bval, fillvalue)
    return out


def correlate2d(in1, in2, mode="full", boundary="fill", fillvalue=0):
    """
    Cross-correlate two 2-dimensional arrays.

    Cross correlate `in1` and `in2` with output size determined by `mode`, and
    boundary conditions determined by `boundary` and `fillvalue`.

    Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear cross-correlation
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``
           pad input arrays with fillvalue. (default)
        ``wrap``
           circular boundary conditions.
        ``symm``
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    correlate2d : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        cross-correlation of `in1` with `in2`.

    Examples
    --------
    Use 2D cross-correlation to find the location of a template in a noisy
    image:

    >>> from scipy import signal
    >>> from scipy import misc
    >>> face = misc.face(gray=True) - misc.face(gray=True).mean()
    >>> template = cupy.copy(face[300:365, 670:750])  # right eye
    >>> template -= template.mean()
    >>> face = face + cupy.random.randn(*face.shape) * 50  # add noise
    >>> corr = signal.correlate2d(face, template, boundary='symm', mode='same')
    >>> y, x = cupy.unravel_index(cupy.argmax(corr), corr.shape)  # find the match

    >>> import matplotlib.pyplot as plt
    >>> fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
    ...                                                     figsize=(6, 15))
    >>> ax_orig.imshow(face, cmap='gray')
    >>> ax_orig.set_title('Original')
    >>> ax_orig.set_axis_off()
    >>> ax_template.imshow(template, cmap='gray')
    >>> ax_template.set_title('Template')
    >>> ax_template.set_axis_off()
    >>> ax_corr.imshow(corr, cmap='gray')
    >>> ax_corr.set_title('Cross-correlation')
    >>> ax_corr.set_axis_off()
    >>> ax_orig.plot(x, y, 'ro')
    >>> fig.show()

    """
    in1 = cupy.asarray(in1)
    in2 = cupy.asarray(in2)

    if not in1.ndim == in2.ndim == 2:
        raise ValueError("correlate2d inputs must both be 2D arrays")
    swapped_inputs = _inputs_swap_needed(mode, in1.shape, in2.shape)
    if swapped_inputs:
        in1, in2 = in2, in1
    if in1.dtype != in2.dtype:
        dtype = np.promote_types(in1.dtype, in2.dtype)
        in1 = in1.astype(dtype, copy=False)
        in2 = in2.astype(dtype, copy=False)
    if cupy.iscomplexobj(in2):
        in2 = in2.conj()
    out = _convolveND(
        in1, in2[::-1, ::-1], mode=mode, boundary=boundary, fillvalue=fillvalue
    )
    return out


def hilbert(x, N=None, axis=-1):
    """
    Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the last axis by default.

    Parameters
    ----------
    x : array_like
        Signal data.  Must be real.
    N : int, optional
        Number of Fourier components.  Default: ``x.shape[axis]``
    axis : int, optional
        Axis along which to do the transformation.  Default: -1.

    Returns
    -------
    xa : ndarray
        Analytic signal of `x`, of each 1-D array along `axis`

    Notes
    -----
    The analytic signal ``x_a(t)`` of signal ``x(t)`` is:

    .. math:: x_a = F^{-1}(F(x) 2U) = x + i y

    where `F` is the Fourier transform, `U` the unit step function,
    and `y` the Hilbert transform of `x`. [1]_

    In other words, the negative half of the frequency spectrum is zeroed
    out, turning the real-valued signal into a complex signal.  The Hilbert
    transformed signal can be obtained from ``cupy.imag(hilbert(x))``, and the
    original signal from ``cupy.real(hilbert(x))``.

    Examples
    ---------
    In this example we use the Hilbert transform to determine the amplitude
    envelope and instantaneous frequency of an amplitude-modulated signal.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.signal import hilbert, chirp

    >>> duration = 1.0
    >>> fs = 400.0
    >>> samples = int(fs*duration)
    >>> t = cupy.arange(samples) / fs

    We create a chirp of which the frequency increases from 20 Hz to 100 Hz and
    apply an amplitude modulation.

    >>> signal = chirp(t, 20.0, t[-1], 100.0)
    >>> signal *= (1.0 + 0.5 * cupy.sin(2.0*np.pi*3.0*t) )

    The amplitude envelope is given by magnitude of the analytic signal. The
    instantaneous frequency can be obtained by differentiating the
    instantaneous phase in respect to time. The instantaneous phase corresponds
    to the phase angle of the analytic signal.

    >>> analytic_signal = hilbert(signal)
    >>> amplitude_envelope = cupy.abs(analytic_signal)
    >>> instantaneous_phase = cupy.unwrap(cupy.angle(analytic_signal))
    >>> instantaneous_frequency = (cupy.diff(instantaneous_phase) /
    ...                            (2.0*np.pi) * fs)

    >>> fig = plt.figure()
    >>> ax0 = fig.add_subplot(211)
    >>> ax0.plot(t, signal, label='signal')
    >>> ax0.plot(t, amplitude_envelope, label='envelope')
    >>> ax0.set_xlabel("time in seconds")
    >>> ax0.legend()
    >>> ax1 = fig.add_subplot(212)
    >>> ax1.plot(t[1:], instantaneous_frequency)
    >>> ax1.set_xlabel("time in seconds")
    >>> ax1.set_ylim(0.0, 120.0)

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal
    .. [2] Leon Cohen, "Time-Frequency Analysis", 1995. Chapter 2.
    .. [3] Alan V. Oppenheim, Ronald W. Schafer. Discrete-Time Signal
           Processing, Third Edition, 2009. Chapter 12.
           ISBN 13: 978-1292-02572-8

    """
    x = cupy.asarray(x)
    if cupy.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape[axis]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = sp_fft.fft(x, N, axis=axis)
    h = cupy.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1 : N // 2] = 2
    else:
        h[0] = 1
        h[1 : (N + 1) // 2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]
    x = sp_fft.ifft(Xf * h, axis=axis)
    return x


def hilbert2(x, N=None):
    """
    Compute the '2-D' analytic signal of `x`

    Parameters
    ----------
    x : array_like
        2-D signal data.
    N : int or tuple of two ints, optional
        Number of Fourier components. Default is ``x.shape``

    Returns
    -------
    xa : ndarray
        Analytic signal of `x` taken along axes (0,1).

    References
    ----------
    .. [1] Wikipedia, "Analytic signal",
        https://en.wikipedia.org/wiki/Analytic_signal

    """
    x = cupy.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be 2-D.")
    if cupy.iscomplexobj(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.shape
    elif isinstance(N, int):
        if N <= 0:
            raise ValueError("N must be positive.")
        N = (N, N)
    elif len(N) != 2 or np.any(np.asarray(N) <= 0):
        raise ValueError(
            "When given as a tuple, N must hold exactly "
            "two positive integers"
        )

    Xf = sp_fft.fft2(x, N, axes=(0, 1))
    h1 = cupy.zeros(N[0], "d")
    h2 = cupy.zeros(N[1], "d")
    for p in range(2):
        h = eval("h%d" % (p + 1))
        N1 = N[p]
        if N1 % 2 == 0:
            h[0] = h[N1 // 2] = 1
            h[1 : N1 // 2] = 2
        else:
            h[0] = 1
            h[1 : (N1 + 1) // 2] = 2
        exec("h%d = h" % (p + 1), globals(), locals())

    h = h1[:, np.newaxis] * h2[np.newaxis, :]
    k = x.ndim
    while k > 2:
        h = h[:, np.newaxis]
        k -= 1
    x = sp_fft.ifft2(Xf * h, axes=(0, 1))
    return x


def resample(x, num, t=None, axis=0, window=None):
    """
    Resample `x` to `num` samples using Fourier method along the given axis.

    The resampled signal starts at the same value as `x` but is sampled
    with a spacing of ``len(x) / num * (spacing of x)``.  Because a
    Fourier method is used, the signal is assumed to be periodic.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.

    Returns
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample_poly : Resample using polyphase filtering and an FIR filter.

    Notes
    -----
    The argument `window` controls a Fourier-domain window that tapers
    the Fourier spectrum before zero-padding to alleviate ringing in
    the resampled values for sampled signals you didn't intend to be
    interpreted as band-limited.

    If `window` is a function, then it is called with a vector of inputs
    indicating the frequency bins (i.e. fftfreq(x.shape[axis]) ).

    If `window` is an array of the same length as `x.shape[axis]` it is
    assumed to be the window to be applied directly in the Fourier
    domain (with dc and low-frequency first).

    For any other type of `window`, the function `scipy.signal.get_window`
    is called to generate the window.

    The first sample of the returned vector is the same as the first
    sample of the input vector.  The spacing between samples is changed
    from ``dx`` to ``dx * len(x) / num``.

    If `t` is not None, then it is used solely to calculate the resampled
    positions `resampled_t`

    As noted, `resample` uses FFT transformations, which can be very
    slow if the number of input or output samples is large and prime;
    see `scipy.fft.fft`.

    Examples
    --------
    Note that the end of the resampled data rises to meet the first
    sample of the next cycle:

    >>> from scipy import signal

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f = signal.resample(y, 100)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
    >>> plt.legend(['data', 'resampled'], loc='best')
    >>> plt.show()
    """
    x = cupy.asarray(x)
    X = sp_fft.fft(x, axis=axis)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = cupy.isrealobj(x)

    # Forward transform
    if real_input:
        X = sp_fft.rfft(x, axis=axis)
    else:  # Full complex FFT
        X = sp_fft.fft(x, axis=axis)

    # Apply window to spectrum
    if window is not None:
        if callable(window):
            W = window(sp_fft.fftfreq(Nx))
        elif isinstance(window, cupy.ndarray):
            if window.shape != (Nx,):
                raise ValueError("window must have the same length as data")
            W = window
        else:
            W = sp_fft.ifftshift(cupy.asarray(get_window(window, Nx)))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[: newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = cupy.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N // 2, N // 2 + 1)
                Y[tuple(sl)] *= 2.0
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N // 2, -N // 2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N // 2, N // 2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num - N // 2, num - N // 2 + 1)
                Y[tuple(sl)] = temp

    # Inverse transform
    if real_input:
        y = sp_fft.irfft(Y, num, axis=axis)
    else:
        y = sp_fft.ifft(Y, axis=axis, overwrite_x=True)

    y *= float(num) / float(Nx)

    if t is None:
        return y
    else:
        new_t = cupy.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def _resample_poly_window(up, down, window=("kaiser", 5.0)):
    """Design a linear-phase low-pass FIR filter for resample_poly."""
    try:
        from scipy.signal import firwin
    except ImportError:
        print("Use of resample_poly requires SciPy.")
        raise

    max_rate = max(up, down)
    f_c = 1.0 / max_rate  # cutoff of FIR filter (rel. to Nyquist)
    half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
    h = firwin(2 * half_len + 1, f_c, window=window)
    return h, half_len


def resample_poly(
    x, up, down, axis=0, window=("kaiser", 5.0), padtype="constant", cval=None
):
    """
    Resample `x` along the given axis using polyphase filtering.

    The signal `x` is upsampled by the factor `up`, a zero-phase low-pass
    FIR filter is applied, and then it is downsampled by the factor `down`.
    The resulting sample rate is ``up / down`` times the original sample
    rate. By default, values beyond the boundary of the signal are assumed
    to be zero during the filtering step.

    Parameters
    ----------
    x : array_like
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Default is 0.
    window : string, tuple, or array_like, optional
        Desired window to use to design the low-pass filter, or the FIR filter
        coefficients to employ. See below for details.
    padtype : string, optional
        `constant`, `line`, `mean`, `median`, `maximum`, `minimum` or any of
        the other signal extension modes supported by `scipy.signal.upfirdn`.
        Changes assumptions on values beyond the boundary. If `constant`,
        assumed to be `cval` (default zero). If `line` assumed to continue a
        linear trend defined by the first and last points. `mean`, `median`,
        `maximum` and `minimum` work as in `cupy.pad` and assume that the values
        beyond the boundary are the mean, median, maximum or minimum
        respectively of the array along the axis.
    cval : float, optional
        Value to use if `padtype='constant'`. Default is zero.

    Returns
    -------
    resampled_x : array
        The resampled array.

    See Also
    --------
    decimate : Downsample the signal after applying an FIR or IIR filter.
    resample : Resample up or down using the FFT method.

    Notes
    -----
    This polyphase method will likely be faster than the Fourier method
    in `scipy.signal.resample` when the number of samples is large and
    prime, or when the number of samples is large and `up` and `down`
    share a large greatest common denominator. The length of the FIR
    filter used will depend on ``max(up, down) // gcd(up, down)``, and
    the number of operations during polyphase filtering will depend on
    the filter length and `down` (see `scipy.signal.upfirdn` for details).

    The argument `window` specifies the FIR low-pass filter design.

    If `window` is an array_like it is assumed to be the FIR filter
    coefficients. Note that the FIR filter is applied after the upsampling
    step, so it should be designed to operate on a signal at a sampling
    frequency higher than the original by a factor of `up//gcd(up, down)`.
    This function's output will be centered with respect to this array, so it
    is best to pass a symmetric filter with an odd number of samples if, as
    is usually the case, a zero-phase filter is desired.

    For any other type of `window`, the functions `scipy.signal.get_window`
    and `scipy.signal.firwin` are called to generate the appropriate filter
    coefficients.

    The first sample of the returned vector is the same as the first
    sample of the input vector. The spacing between samples is changed
    from ``dx`` to ``dx * down / float(up)``.

    Examples
    --------
    By default, the end of the resampled data rises to meet the first
    sample of the next cycle for the FFT method, and gets closer to zero
    for the polyphase method:

    >>> from scipy import signal

    >>> x = cupy.linspace(0, 10, 20, endpoint=False)
    >>> y = cupy.cos(-x**2/6.0)
    >>> f_fft = signal.resample(y, 100)
    >>> f_poly = signal.resample_poly(y, 100, 20)
    >>> xnew = cupy.linspace(0, 10, 100, endpoint=False)

    >>> import matplotlib.pyplot as plt
    >>> plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
    >>> plt.plot(x, y, 'ko-')
    >>> plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
    >>> plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    >>> plt.show()

    This default behaviour can be changed by using the padtype option:

    >>> import numpy as np
    >>> from scipy import signal

    >>> N = 5
    >>> x = cupy.linspace(0, 1, N, endpoint=False)
    >>> y = 2 + x**2 - 1.7*cupy.sin(x) + .2*cupy.cos(11*x)
    >>> y2 = 1 + x**3 + 0.1*cupy.sin(x) + .1*cupy.cos(11*x)
    >>> Y = cupy.stack([y, y2], axis=-1)
    >>> up = 4
    >>> xr = cupy.linspace(0, 1, N*up, endpoint=False)

    >>> y2 = signal.resample_poly(Y, up, 1, padtype='constant')
    >>> y3 = signal.resample_poly(Y, up, 1, padtype='mean')
    >>> y4 = signal.resample_poly(Y, up, 1, padtype='line')

    >>> import matplotlib.pyplot as plt
    >>> for i in [0,1]:
    ...     plt.figure()
    ...     plt.plot(xr, y4[:,i], 'g.', label='line')
    ...     plt.plot(xr, y3[:,i], 'y.', label='mean')
    ...     plt.plot(xr, y2[:,i], 'r.', label='constant')
    ...     plt.plot(x, Y[:,i], 'k-')
    ...     plt.legend()
    >>> plt.show()

    """
    if up != int(up):
        raise ValueError("up must be an integer")
    if down != int(down):
        raise ValueError("down must be an integer")
    up = int(up)
    down = int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    if cval is not None and padtype != "constant":
        raise ValueError("cval has no effect when padtype is ", padtype)

    # Determine our up and down factors
    # Use a rational approximation to save computation time on really long
    # signals
    g_ = math.gcd(up, down)
    up //= g_
    down //= g_
    if up == down == 1:
        return x.copy()
    n_in = x.shape[axis]
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)

    if isinstance(window, (list, cupy.ndarray)):
        window = cupy.asarray(
            window
        )  # use array to force a copy (we modify it)
        if window.ndim > 1:
            raise ValueError("window must be 1-D")
        half_len = (window.size - 1) // 2
        h = window
    else:
        # Design a linear-phase low-pass FIR filter
        h, half_len = _resample_poly_window(up, down, window=window)
        h = cupy.asarray(h, dtype=x.real.dtype)
    h_tmp = h * up

    # Zero-pad our filter to put the output samples at the center
    n_pre_pad = down - half_len % down
    n_post_pad = 0
    n_pre_remove = (half_len + n_pre_pad) // down
    # print(f"n_pre_pad={n_pre_pad}, n_pre_remove={n_pre_remove}")
    # We should rarely need to do this given our filter lengths...
    while (
        upfirdn_out_len(len(h_tmp) + n_pre_pad + n_post_pad, n_in, up, down)
        < n_out + n_pre_remove
    ):
        n_post_pad += 1

    h = cupy.zeros(len(h_tmp) + n_pre_pad + n_post_pad, dtype=h.dtype)
    h[n_pre_pad : n_pre_pad + h_tmp.size] = h_tmp
    n_pre_remove_end = n_pre_remove + n_out

    # Remove background depending on the padtype option
    funcs = {
        "mean": cupy.mean,
        # 'median': cupy.median,   # TODO: needs cupy.median implementation
        "minimum": cupy.amin,
        "maximum": cupy.amax,
    }
    upfirdn_kwargs = {"mode": "constant", "cval": 0}
    if padtype in funcs:
        background_values = funcs[padtype](x, axis=axis, keepdims=True)
    elif padtype in ["median"]:
        raise NotImplementedError("padtype=median not yet supported")
    elif padtype in upfirdn_modes:
        upfirdn_kwargs = {"mode": padtype}
        if padtype == "constant":
            if cval is None:
                cval = 0
            upfirdn_kwargs["cval"] = cval
    else:
        raise ValueError(
            "padtype must be one of: maximum, mean, median, minimum, "
            + ", ".join(upfirdn_modes)
        )

    if padtype in funcs:
        x = x - background_values

    # filter then remove excess
    y = upfirdn(h, x, up, down, axis=axis, **upfirdn_kwargs)

    keep = [slice(None)] * x.ndim
    keep[axis] = slice(n_pre_remove, n_pre_remove_end)
    y_keep = y[tuple(keep)]

    # Add background back
    if padtype in funcs:
        y_keep += background_values

    return y_keep


# def convolve(in1, in2, mode='full', method='auto'):
#     """
#     Convolve two N-dimensional arrays.

#     Convolve `in1` and `in2`, with the output size determined by the
#     `mode` argument.

#     Parameters
#     ----------
#     in1 : array_like
#         First input.
#     in2 : array_like
#         Second input. Should have the same number of dimensions as `in1`.
#     mode : str {'full', 'valid', 'same'}, optional
#         A string indicating the size of the output:

#         ``full``
#            The output is the full discrete linear convolution
#            of the inputs. (Default)
#         ``valid``
#            The output consists only of those elements that do not
#            rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
#            must be at least as large as the other in every dimension.
#         ``same``
#            The output is the same size as `in1`, centered
#            with respect to the 'full' output.
#     method : str {'auto', 'direct', 'fft'}, optional
#         A string indicating which method to use to calculate the convolution.

#         ``direct``
#            The convolution is determined directly from sums, the definition of
#            convolution.
#         ``fft``
#            The Fourier Transform is used to perform the convolution by calling
#            `fftconvolve`.
#         ``auto``
#            Automatically chooses direct or Fourier method based on an estimate
#            of which is faster (default).  See Notes for more detail.

#            .. versionadded:: 0.19.0

#     Returns
#     -------
#     convolve : array
#         An N-dimensional array containing a subset of the discrete linear
#         convolution of `in1` with `in2`.

#     See Also
#     --------
#     numpy.polymul : performs polynomial multiplication (same operation, but
#                     also accepts poly1d objects)
#     choose_conv_method : chooses the fastest appropriate convolution method
#     fftconvolve : Always uses the FFT method.
#     oaconvolve : Uses the overlap-add method to do convolution, which is
#                  generally faster when the input arrays are large and
#                  significantly different in size.

#     Notes
#     -----
#     By default, `convolve` and `correlate` use ``method='auto'``, which calls
#     `choose_conv_method` to choose the fastest method using pre-computed
#     values (`choose_conv_method` can also measure real-world timing with a
#     keyword argument). Because `fftconvolve` relies on floating point numbers,
#     there are certain constraints that may force `method=direct` (more detail
#     in `choose_conv_method` docstring).

#     Examples
#     --------
#     Smooth a square pulse using a Hann window:

#     >>> from scipy import signal
#     >>> sig = cupy.repeat([0., 1., 0.], 100)
#     >>> win = signal.hann(50)
#     >>> filtered = signal.convolve(sig, win, mode='same') / sum(win)

#     >>> import matplotlib.pyplot as plt
#     >>> fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
#     >>> ax_orig.plot(sig)
#     >>> ax_orig.set_title('Original pulse')
#     >>> ax_orig.margins(0, 0.1)
#     >>> ax_win.plot(win)
#     >>> ax_win.set_title('Filter impulse response')
#     >>> ax_win.margins(0, 0.1)
#     >>> ax_filt.plot(filtered)
#     >>> ax_filt.set_title('Filtered signal')
#     >>> ax_filt.margins(0, 0.1)
#     >>> fig.tight_layout()
#     >>> fig.show()

#     """
#     volume = cupy.asarray(in1)
#     kernel = cupy.asarray(in2)

#     if volume.ndim == kernel.ndim == 0:
#         return volume * kernel
#     elif volume.ndim != kernel.ndim:
#         raise ValueError("volume and kernel should have the same "
#                          "dimensionality")

#     if _inputs_swap_needed(mode, volume.shape, kernel.shape):
#         # Convolution is commutative; order doesn't have any effect on output
#         volume, kernel = kernel, volume

#     if method == 'auto':
#         method = choose_conv_method(volume, kernel, mode=mode)

#     if method == 'fft':
#         out = fftconvolve(volume, kernel, mode=mode)
#         result_type = np.result_type(volume.dtype, kernel.dtype)
#         if result_type.kind in {'u', 'i'}:
#             out = cupy.around(out)
#         return out.astype(result_type)
#     elif method == 'direct':
#         # fastpath to faster numpy.convolve for 1d inputs when possible
#         #if _np_conv_ok(volume, kernel, mode):
#         #    return cnp.convolve(volume, kernel, mode)

#         return correlate(volume, _reverse_and_conj(kernel), mode, 'direct')
#     else:
#         raise ValueError(
#             "Acceptable method flags are 'auto', 'direct', or 'fft'."
#         )
