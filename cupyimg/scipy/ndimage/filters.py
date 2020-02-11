"""Functions based on separable convolution found in scipy.ndimage

These functions are intended to operate exactly like their SciPy counterparts
aside from the following differences:

1.) By default, convolution computations are done in the nearest floating point
    precision (single or double) for the input dtype.
    (``scipy.ndimage`` does all convolutions in double precision)
2.) Complex-valued inputs (complex64 or complex128) are supported by many
    functions (e.g. convolve, correlate).
    (``scipy.ndimage`` does not support complex-valued inputs)
3.) convolve1d has a ``crop`` kwarg. If set to False, a full convolution
    instead of one truncated to the size of the input is given.
4.) In-place operation via ``output`` is not fully supported.

"""
import functools
import operator
import warnings

import cupy
import numpy

from ._kernels.filters import (
    _get_correlete_kernel,
    _get_correlete_kernel_masked,
    _get_min_or_max_kernel,
    _get_min_or_max_kernel_masked,
    _get_min_or_max_kernel_masked_v2,
    _get_rank_kernel,
    _get_rank_kernel_masked,
)
from . import _ni_support


_partial = functools.partial


__all__ = [
    # from scipy.ndimage API
    "correlate1d",
    "convolve1d",
    "gaussian_filter1d",
    "gaussian_filter",
    "prewitt",
    "sobel",
    "generic_laplace",
    "laplace",
    "gaussian_laplace",
    "generic_gradient_magnitude",
    "gaussian_gradient_magnitude",
    "correlate",
    "convolve",
    "uniform_filter1d",
    "uniform_filter",
    "minimum_filter1d",
    "maximum_filter1d",
    "minimum_filter",
    "maximum_filter",
    "rank_filter",
    "median_filter",
    "percentile_filter",
]

# TODO: grlee77: 'generic_filter1d', 'generic_filter'


def correlate1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    *,
    backend="ndimage",
    dtype_mode="float",
):
    """Calculate a one-dimensional correlation along the given axis.


    See ``scipy.ndimage.correlate1d``

    This version supports only ``numpy.float32``, ``numpy.float64``,
    ``numpy.complex64`` and ``numpy.complex128`` dtypes.
    """
    if _ni_support._invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )

    weights = weights[::-1]
    origin = -origin
    if not len(weights) & 1:
        origin -= 1
    if cupy.iscomplexobj(weights):
        # numpy.correlate conjugates weights rather than input. Do the same here
        weights = weights.conj()

    return convolve1d(
        input,
        weights,
        axis=axis,
        output=output,
        mode=mode,
        cval=cval,
        origin=origin,
        backend=backend,
        dtype_mode=dtype_mode,
    )


def convolve1d(
    input,
    weights,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0,
    origin=0,
    *,
    crop=True,  # if False, will get a "full" convolution instead
    backend="ndimage",
    dtype_mode="float",
):
    """Calculate a one-dimensional convolution along the given axis.

    see ``scipy.ndimage.convolve1d``

    Notes
    -----
    crop=True gives an output the same size as the input ``input``.

    Setting crop=False gives an output that is the full size of the
    convolution. ``output.shape[axis] = input.shape[axis] + len(weights) - 1``
    """
    from cupyimg._misc import _reshape_nd

    if _ni_support._invalid_origin(origin, len(weights)):
        raise ValueError(
            "Invalid origin; origin must satisfy "
            "-(len(weights) // 2) <= origin <= "
            "(len(weights)-1) // 2"
        )

    axis = _ni_support._check_axis(axis, input.ndim)

    if backend == "fast_upfirdn":
        try:
            from fast_upfirdn.cupy import convolve1d as _convolve1d_gpu
        except ImportError as err:
            msg = (
                "Use of fast_upfirdn backend requires installation of "
                "fast_upfirdn."
            )
            raise ImportError(msg) from err

        w_len_half = len(weights) // 2
        if crop:
            offset = w_len_half + origin
        else:
            if origin != 0:
                raise ValueError("uncropped case requires origin == 0")
            offset = 0

        mode_kwargs = _ni_support._get_ndimage_mode_kwargs(mode, cval)

        if dtype_mode == "float":
            dtype = functools.reduce(
                numpy.promote_types,
                [input.dtype, numpy.float32, weights.real.dtype],
            )
        elif dtype_mode == "ndimage":
            dtype = functools.reduce(
                numpy.promote_types,
                [input.dtype, numpy.float64, weights.real.dtype],
            )
        else:
            raise ValueError(
                "dtype_mode={} not supported by backend fast_upfirdn".format(
                    dtype_mode
                )
            )
        if input.dtype != dtype:
            warnings.warn("input of dtype {input.dtype} promoted to {dtype}")
        input = input.astype(dtype, copy=False)
        if output is not None:
            output = _ni_support._get_output(output, input)

        return _convolve1d_gpu(
            weights,
            input,
            axis=axis,
            offset=offset,
            crop=crop,
            out=output,
            **mode_kwargs,
        )
    else:
        if not crop:
            raise ValueError("crop=False requires backend='fast_upfirdn'")
        if output is input:
            # warnings.warn(
            #     "in-place convolution is not supported. A copy of input "
            #     "will be made."
            # )
            input = input.copy()
        weights = cupy.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("expected a 1d weights array")
        if input.ndim > 1:
            weights = _reshape_nd(weights, ndim=input.ndim, axis=axis)
            _origin = origin
            origin = [0] * input.ndim
            origin[axis] = _origin
        return convolve(
            input,
            weights,
            mode=mode,
            output=output,
            origin=origin,
            dtype_mode=dtype_mode,
        )


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1D Gaussian convolution kernel.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    exponent_range = numpy.arange(order + 1)
    sigma2 = sigma * sigma
    x = numpy.arange(-radius, radius + 1)
    phi_x = numpy.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = numpy.zeros(order + 1)
        q[0] = 1
        D = numpy.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = numpy.diag(
            numpy.ones(order) / -sigma2, -1
        )  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


def gaussian_filter1d(
    input,
    sigma,
    axis=-1,
    order=0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
):
    """One-dimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter1d``

    This version supports only ``numpy.float32``, ``numpy.float64``,
    ``numpy.complex64`` and ``numpy.complex128`` dtypes.

    """
    dtype_weights = numpy.promote_types(input.real.dtype, numpy.float32)

    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    # Since we are calling correlate, not convolve, revert the kernel
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    weights = cupy.asarray(weights, dtype=dtype_weights)
    return correlate1d(input, weights, axis, output, mode, cval, 0)


def gaussian_filter(
    input, sigma, order=0, output=None, mode="reflect", cval=0.0, truncate=4.0
):
    """Multidimensional Gaussian filter.

    See ``scipy.ndimage.gaussian_filter``
    """
    output = _ni_support._get_output(output, input)
    orders = _ni_support._normalize_sequence(order, input.ndim)
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [
        (axes[ii], sigmas[ii], orders[ii], modes[ii])
        for ii in range(len(axes))
        if sigmas[ii] > 1e-15
    ]
    if len(axes) > 0:
        for axis, sigma, order, mode in axes:
            gaussian_filter1d(
                input, sigma, axis, order, output, mode, cval, truncate
            )
            input = output
    else:
        output[...] = input[...]
    return output


def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Apply a Prewitt filter.

    See ``scipy.ndimage.prewitt``
    """
    dtype_weights = numpy.promote_types(input.real.dtype, numpy.float32)
    axis = _ni_support._check_axis(axis, input.ndim)
    output = _ni_support._get_output(output, input)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    filt1 = [-1, 0, 1]
    filt2 = [1, 1, 1]

    filt1, filt2 = map(
        _partial(cupy.asarray, dtype=dtype_weights), [filt1, filt2]
    )
    ndim = input.ndim
    if axis < -ndim or axis >= ndim:
        raise ValueError("invalid axis")
    axis = axis % ndim

    correlate1d(input, filt1, axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, filt2, ii, output, modes[ii], cval, 0)
    return output


def sobel(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Apply a sobel filter.

    See ``scipy.ndimage.sobel``
    """
    dtype_weights = numpy.promote_types(input.real.dtype, numpy.float32)
    output = _ni_support._get_output(output, input)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    filt1 = [-1, 0, 1]
    filt2 = [1, 2, 1]
    filt1, filt2 = map(
        _partial(cupy.asarray, dtype=dtype_weights), [filt1, filt2]
    )
    ndim = input.ndim
    if axis < -ndim or axis >= ndim:
        raise ValueError("invalid axis")
    axis = axis % ndim
    correlate1d(input, filt1, axis, output, modes[axis], cval, 0)
    axes = [ii for ii in range(input.ndim) if ii != axis]
    for ii in axes:
        correlate1d(output, filt2, ii, output, modes[ii], cval, 0)
    return output


def generic_laplace(
    input,
    derivative2,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
):
    """
    N-dimensional Laplace filter using a provided second derivative function.

    See ``scipy.ndimage.generic_laplace``
    """
    if extra_keywords is None:
        extra_keywords = {}
    output = _ni_support._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        modes = _ni_support._normalize_sequence(mode, len(axes))
        derivative2(
            input,
            axes[0],
            output,
            modes[0],
            cval,
            *extra_arguments,
            **extra_keywords,
        )
        for ii in range(1, len(axes)):
            tmp = derivative2(
                input,
                axes[ii],
                # None,
                output.dtype,
                modes[ii],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
            output += tmp
    else:
        output[...] = input[...]
    return output


def laplace(input, output=None, mode="reflect", cval=0.0):
    """N-dimensional Laplace filter based on approximate second derivatives.

    See ``scipy.ndimage.laplace``
    """

    def derivative2(input, axis, output, mode, cval):
        h_dtype = numpy.promote_types(input.real.dtype, numpy.float32)
        h = cupy.asarray([1, -2, 1], dtype=h_dtype)
        return correlate1d(input, h, axis, output, mode, cval, 0)

    return generic_laplace(input, derivative2, output, mode, cval)


def gaussian_laplace(
    input, sigma, output=None, mode="reflect", cval=0.0, **kwargs
):
    """Multidimensional Laplace filter using Gaussian second derivatives.

    See ``scipy.ndimage.gaussian_laplace``

    """

    def derivative2(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 2
        return gaussian_filter(
            input, sigma, order, output, mode, cval, **kwargs
        )

    return generic_laplace(
        input,
        derivative2,
        output,
        mode,
        cval,
        extra_arguments=(sigma,),
        extra_keywords=kwargs,
    )


def generic_gradient_magnitude(
    input,
    derivative,
    output=None,
    mode="reflect",
    cval=0.0,
    extra_arguments=(),
    extra_keywords=None,
):
    """Gradient magnitude using a provided gradient function.

    See ``scipy.ndimage.generic_gradient_magnitude``

    """
    if extra_keywords is None:
        extra_keywords = {}
    output = _ni_support._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        modes = _ni_support._normalize_sequence(mode, len(axes))
        derivative(
            input,
            axes[0],
            output,
            modes[0],
            cval,
            *extra_arguments,
            **extra_keywords,
        )

        cupy.multiply(output, output, output)
        for ii in range(1, len(axes)):
            tmp = derivative(
                input,
                axes[ii],
                output.dtype,
                modes[ii],
                cval,
                *extra_arguments,
                **extra_keywords,
            )
            cupy.multiply(tmp, tmp, tmp)
            output += tmp
        # This allows the sqrt to work with a different default casting
        cupy.sqrt(output, output, casting="unsafe")
    else:
        output[...] = input[...]
    return output


def gaussian_gradient_magnitude(
    input, sigma, output=None, mode="reflect", cval=0.0, **kwargs
):
    """Multidimensional gradient magnitude using Gaussian derivatives.

    See ``scipy.ndimage.gaussian_gradient_magnitude``
    """

    def derivative(input, axis, output, mode, cval, sigma, **kwargs):
        order = [0] * input.ndim
        order[axis] = 1
        return gaussian_filter(
            input, sigma, order, output, mode, cval, **kwargs
        )

    return generic_gradient_magnitude(
        input,
        derivative,
        output,
        mode,
        cval,
        extra_arguments=(sigma,),
        extra_keywords=kwargs,
    )


def _get_output_v2(output, input, weights_dtype, shape=None):
    if shape is None:
        shape = input.shape
    if isinstance(output, cupy.ndarray):
        if output.shape != tuple(shape):
            raise RuntimeError("output shape is not correct")
        if output is input:
            raise RuntimeError("in-place convolution not supported")
        if (
            input.dtype.kind == "c" or weights_dtype.kind == "c"
        ) and output.dtype.kind != "c":
            raise RuntimeError(
                "output must have complex dtype if either the input or "
                "weights are complex-valued."
            )
    else:
        dtype = output
        if dtype is None:
            if weights_dtype.kind == "c":
                dtype = cupy.promote_types(input.dtype, cupy.complex64)
            else:
                dtype = input.dtype
        elif (
            input.dtype.kind == "c" or weights_dtype.kind == "c"
        ) and output.dtype.kind != "c":
            raise RuntimeError(
                "output must have complex dtype if either the input or "
                "weights are complex-valued."
            )
        output = cupy.zeros(shape, dtype)
    return output


def _correlate_or_convolve(
    input,
    weights,
    output,
    mode,
    cval,
    origin,
    convolution,
    dtype_mode,
    use_weights_mask,
):
    if not hasattr(origin, "__getitem__"):
        origin = [origin] * input.ndim
    else:
        origin = list(origin)
    wshape = [ii for ii in weights.shape if ii > 0]
    if len(wshape) != input.ndim:
        raise RuntimeError("filter weights array has incorrect shape.")
    if convolution:
        # weights are reversed in order for convolution and the origin
        # must be adjusted.
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        for ii in range(len(origin)):
            origin[ii] = -origin[ii]
            if weights.shape[ii] % 2 == 0:
                origin[ii] -= 1
    elif cupy.iscomplexobj(weights):
        # numpy.correlate conjugates weights rather than input. Do the same here
        weights = weights.conj()

    for _origin, lenw in zip(origin, wshape):
        if (lenw // 2 + _origin < 0) or (lenw // 2 + _origin >= lenw):
            raise ValueError("invalid origin")
    if mode not in ("reflect", "constant", "nearest", "mirror", "wrap"):
        msg = "boundary mode not supported (actual: {}).".format(mode)
        raise RuntimeError(msg)

    if dtype_mode == "numpy":
        # numpy.convolve and correlate do not always cast to floats
        dtype = numpy.promote_types(input.dtype, weights.dtype)
        output_dtype = dtype
        if dtype.char == "e":
            # promote internal float type to float32 for accuracy
            dtype = "f"
        if output is not None:
            raise ValueError(
                "dtype_mode == 'numpy' does not support the output " "argument"
            )
        weights_dtype = dtype
        if weights.dtype != dtype:
            weights = weights.astype(dtype)
        if input.dtype != dtype:
            input = input.astype(dtype)
        output = cupy.zeros(input.shape, output_dtype)
        weights_dtype = dtype
    else:
        # scipy.ndimage always use double precision for the weights
        if weights.dtype.kind == "c" or input.dtype.kind == "c":
            if dtype_mode == "ndimage":
                weights_dtype = numpy.complex128
            elif dtype_mode == "float":
                weights_dtype = numpy.promote_types(
                    input.real.dtype, numpy.complex64
                )
        else:
            if dtype_mode == "ndimage":
                weights_dtype = numpy.float64
            elif dtype_mode == "float":
                weights_dtype = numpy.promote_types(
                    input.real.dtype, numpy.float32
                )
        weights_dtype = cupy.dtype(weights_dtype)

        #    if output is input:
        #        input = input.copy()
        output = _get_output_v2(output, input, weights_dtype)
    if weights.size == 0:
        return output

    input = cupy.ascontiguousarray(input)
    weights = cupy.ascontiguousarray(weights, weights_dtype)

    unsigned_output = output.dtype.kind in ["u", "b"]

    if use_weights_mask:
        # The kernel needs only the non-zero kernel values and their coordinates.
        # This allows us to use a single for loop to compute the ndim convolution.
        # The loop will be over only the the non-zero entries of the filter.
        wlocs = cupy.nonzero(weights)
        wvals = weights[wlocs]  # (nnz,) array of non-zero values
        wlocs = cupy.stack(
            wlocs
        )  # (ndim, nnz) array of indices for these values

        return _get_correlete_kernel_masked(
            mode,
            cval,
            input.shape,
            weights.shape,
            wvals.size,
            tuple(origin),
            unsigned_output,
        )(input, wlocs, wvals, output)
    else:
        return _get_correlete_kernel(
            input.ndim,
            mode,
            cval,
            input.shape,
            weights.shape,
            tuple(origin),
            unsigned_output,
        )(input, weights, output)


def _prep_size_footprint(
    ndim, size=None, footprint=None, allow_separable=False
):
    """In cases where separable filtering is possible, this function returns
    footprint=None to indicate that a separable filter should be used.
    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            "ignoring size because footprint is set", UserWarning, stacklevel=3
        )
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint provided")
        fshape = tuple(_ni_support._normalize_sequence(size, ndim))
        if not allow_separable:
            footprint = cupy.ones(fshape, dtype=bool)
        filter_size = functools.reduce(operator.mul, fshape)
    else:
        footprint = cupy.asarray(footprint, dtype=bool)
        if not footprint.flags.c_contiguous:
            footprint = footprint.copy()
        filter_size = int(cupy.where(footprint, 1, 0).sum())
        fshape = footprint.shape
        if filter_size == 0:
            raise ValueError("All-zero footprint is not supported.")
        if allow_separable and filter_size == footprint.size:
            footprint = None

    # ? TODO: is there a use case for allowing 0 size axes?
    # fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != ndim:
        raise RuntimeError("footprint array has incorrect shape.")

    return fshape, footprint, filter_size


def correlate(
    input,
    weights,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    use_weights_mask=False,
    dtype_mode="ndimage",
):
    """Multi-dimensional correlate.

    The array is correlated with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        use_weights_mask (bool, optional): When true, the weights array
            is preprocessed to extract only its non-zero values. This can be
            beneficial when there are relatively few non-zero coefficients in
            ``weights``, but does involve some overhead in the fully dense
            case.

    Returns:
        cupy.ndarray: The result of correlate.

    .. note::
        This function supports complex-valued inputs, but the implementation in
        scipy does not. It also supports single precision computation if the
        user specifies ``dtype_mode='float'``. Otherwise, the convolution is
        done in double precision.
        If ``weights`` is complex-valued, it will be conjugated. This matches
        convention used by ``numpy.correlate`` and ``scipy.signal.correlate``.

    .. seealso:: :func:`scipy.ndimage.correlate`
    """
    return _correlate_or_convolve(
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        False,
        dtype_mode,
        use_weights_mask,
    )


def convolve(
    input,
    weights,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    *,
    use_weights_mask=False,
    dtype_mode="ndimage",
):
    """Multi-dimensional convolution.

    The array is convolved with the given kernel.

    Args:
        input (cupy.ndarray): The input array.
        weights (cupy.ndarray): Array of weights, same number of dimensions as
            input
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.
        use_weights_mask (bool, optional): When true, the weights array
            is preprocessed to extract only its non-zero values. This can be
            beneficial when there are relatively few non-zero coefficients in
            ``weights``, but does involve some overhead in the fully dense
            case.

    Returns:
        cupy.ndarray: The result of convolution.

    .. note::
        This function supports complex-valued inputs, but the implementation in
        scipy does not. It also supports single precision computation if the
        user specifies ``dtype_mode='float'``. Otherwise, the convolution is
        done in double precision.

    .. seealso:: :func:`scipy.ndimage.convolve`
    """
    return _correlate_or_convolve(
        input,
        weights,
        output,
        mode,
        cval,
        origin,
        True,
        dtype_mode,
        use_weights_mask,
    )


# TODO: grlee77: incorporate https://github.com/scipy/scipy/pull/7516
def uniform_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Calculate a one-dimensional uniform filter along the given axis.

    See ``scipy.ndimage.uniform_filter1d``
    """
    dtype_weights = numpy.promote_types(input.real.dtype, numpy.float32)
    if size < 1:
        raise RuntimeError("incorrect filter size")
    output = _ni_support._get_output(output, input)
    if (size // 2 + origin < 0) or (size // 2 + origin >= size):
        raise ValueError("invalid origin")
    weights = cupy.full((size,), 1 / size, dtype=dtype_weights)
    return correlate1d(
        input, weights, axis, output, mode, cval, origin, dtype_mode="ndimage"
    )


def uniform_filter(
    input, size=3, output=None, mode="reflect", cval=0.0, origin=0
):
    """Multi-dimensional uniform filter.

    See ``scipy.ndimage.uniform_filter``
    """
    output = _ni_support._get_output(output, input)
    sizes = _ni_support._normalize_sequence(size, input.ndim)
    origins = _ni_support._normalize_sequence(origin, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))
    axes = [
        (axes[ii], sizes[ii], origins[ii], modes[ii])
        for ii in range(len(axes))
        if sizes[ii] > 1
    ]
    if len(axes) > 0:
        for axis, size, origin, mode in axes:
            uniform_filter1d(input, int(size), axis, output, mode, cval, origin)
            input = output
    else:
        output[...] = input[...]
    return output


def minimum_filter1d(
    input, size=None, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Calculate a 1-D minimum filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): length along which to calculate the 1D minimum
        axis (int): axis along which to calculate the 1D minimum
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    ndim = input.ndim
    axis = _ni_support._check_axis(axis, ndim)
    fshape = (1,) * axis + (size,) + (1,) * (ndim - axis - 1)
    footprint = cupy.ones(fshape, dtype=numpy.bool)
    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, True
    )


def maximum_filter1d(
    input, size=None, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Calculate a 1-D maximum filter along the given axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): length along which to calculate the 1D maximum
        axis (int): axis along which to calculate the 1D maximum
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    ndim = input.ndim
    axis = _ni_support._check_axis(axis, ndim)
    fshape = (1,) * axis + (size,) + (1,) * (ndim - axis - 1)
    footprint = cupy.ones(fshape, dtype=numpy.bool)
    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, False
    )


def _min_or_max_filter(
    input,
    size,
    footprint,
    structure,
    output,
    mode,
    cval,
    origin,
    minimum,
    morphology_mode=False,
):
    if cupy.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    if not hasattr(origin, "__getitem__"):
        origin = [origin] * input.ndim
    else:
        origin = list(origin)
    ndim = input.ndim
    separable = False

    # The separable kernel case doesn't work correctly for morphology, so
    # disable it explicitly here.
    # TODO: it seems that when structure is None and the footprint is True
    #       everywhere it should work?
    allow_separable = not morphology_mode

    if structure is None:
        # only grey-scale morphology functions use structure.
        # (minimum_filter and maximum_filter and rank just use footprint)
        fshape, footprint, filter_size = _prep_size_footprint(
            ndim, size, footprint, allow_separable=allow_separable
        )
        if footprint is None:
            separable = True
            masked = False
        else:
            masked = False
    else:
        # Note: conversion of the structure values to float is done later
        separable = False
        if footprint is None:
            footprint = cupy.ones(structure.shape, bool)
        else:
            footprint = cupy.asarray(footprint, dtype=bool)
        fshape = footprint.shape
        masked = True

    for _origin, lenw in zip(origin, fshape):
        if (lenw // 2 + _origin < 0) or (lenw // 2 + _origin >= lenw):
            raise ValueError("invalid origin")

    if separable:
        modes = _ni_support._normalize_sequence(mode, ndim)
        # all elements true -> can use separable application of 1d filters
        fshape_1d = [1] * ndim
        for ax, sz in zip(range(ndim), fshape):
            fshape_1d[ax] = sz
            fp = cupy.ones(tuple(fshape_1d), dtype=cupy.bool)
            fshape_1d[ax] = 1
            m = modes[ax]
            if m not in ("reflect", "constant", "nearest", "mirror", "wrap"):
                msg = "boundary mode not supported (actual: {}).".format(mode)
                raise RuntimeError(msg)
            if ax == 0:
                result = _min_or_max_filter_inner(
                    input, fp, None, output, m, cval, origin, minimum, masked
                )
            else:
                result = _min_or_max_filter_inner(
                    result, fp, None, None, m, cval, origin, minimum, masked
                )
        return result

    if mode not in ("reflect", "constant", "nearest", "mirror", "wrap"):
        msg = "boundary mode not supported (actual: {}).".format(mode)
        raise RuntimeError(msg)

    return _min_or_max_filter_inner(
        input, footprint, structure, output, mode, cval, origin, minimum, masked
    )


def _min_or_max_filter_inner(
    input, footprint, structure, output, mode, cval, origin, minimum, masked
):
    if footprint is None:
        raise ValueError("footprint must not be None")
    fshape = footprint.shape

    output = _ni_support._get_output(output, input)

    input = cupy.ascontiguousarray(input)

    # The kernel needs only the non-zero footprint coordinates
    wlocs = cupy.nonzero(footprint)

    unsigned_output = output.dtype.kind in ["u", "b"]
    origin = tuple(origin)
    if masked:
        if structure is not None:
            if input.dtype.kind == "b":
                raise NotImplementedError(
                    "Filtering with a structure is not currently supported "
                    "for boolean dtype."
                )
            wvals = structure[wlocs]
            # Note: SciPy always uses double, but it doesn't seem necessary
            wvals_dtype = numpy.promote_types(wvals.dtype, numpy.float32)
            wvals = wvals.astype(wvals_dtype, copy=False)
            if minimum:
                wvals = -wvals
        wlocs = cupy.stack(wlocs)  # (ndim, nnz)
        nnz = wlocs.shape[1]

        if structure is None:
            return _get_min_or_max_kernel_masked(
                mode,
                cval,
                input.shape,
                fshape,
                nnz,
                origin,
                minimum,
                unsigned_output,
            )(input, wlocs, output)
        else:
            return _get_min_or_max_kernel_masked_v2(
                mode,
                cval,
                input.shape,
                fshape,
                nnz,
                origin,
                minimum,
                unsigned_output,
            )(input, wlocs, wvals, output)
    else:
        return _get_min_or_max_kernel(
            mode, cval, input.shape, fshape, origin, minimum, unsigned_output
        )(input, footprint, output)


def minimum_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """Multi-dimensional minimum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or None): see footprint. Ignored if footprint is given.
        footprint (cupy.ndarray or None): Either size or footprint must be
            defined. size gives the shape that is taken from the input array,
            at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly)
            a shape, but also which of the elements within this shape will get
            passed to the filter function. Thus ``size=(n, m)`` is equivalent to
            ``footprint=cupy.ones((n, m))``. We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
            When footprint is given, size is ignored.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    # ndim = input.ndim
    # fshape, footprint, filter_size = _prep_size_footprint(
    #     ndim, size, footprint, allow_separable=True
    # )
    # if footprint is None:
    #     modes = _ni_support._normalize_sequence(mode, ndim)
    #     # all elements true -> can use separable application of 1d filters
    #     fshape_1d = [1, ] * ndim
    #     for ax, sz in zip(range(ndim), fshape):
    #         fshape_1d[ax] = sz
    #         footprint = cupy.ones(tuple(fshape_1d), dtype=cupy.bool)
    #         fshape_1d[ax] = 1
    #         if ax == 0:
    #             result = _min_or_max_filter(
    #                 input, size, footprint, None, output, modes[ax], cval, origin, True
    #             )
    #         else:
    #             result = _min_or_max_filter(
    #                 result, size, footprint, None, None, modes[ax], cval, origin, True
    #             )
    #     return result

    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, True
    )


def maximum_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """Multi-dimensional maximum filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or None): see footprint. Ignored if footprint is given.
        footprint (cupy.ndarray or None): Either size or footprint must be
            defined. size gives the shape that is taken from the input array,
            at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly)
            a shape, but also which of the elements within this shape will get
            passed to the filter function. Thus ``size=(n, m)`` is equivalent to
            ``footprint=cupy.ones((n, m))``. We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
            When footprint is given, size is ignored.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str or sequence of str): The array borders are handled according
            to the given mode (``'reflect'``, ``'constant'``, ``'nearest'``,
            ``'mirror'``, ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, False
    )


def _rank_filter(
    input, rank, size, footprint, output, mode, cval, origin, operation="rank"
):
    if cupy.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    if not hasattr(origin, "__getitem__"):
        origin = [origin] * input.ndim
    else:
        origin = list(origin)
    fshape, footprint, filter_size = _prep_size_footprint(
        input.ndim, size, footprint, allow_separable=False
    )
    for _origin, lenw in zip(origin, fshape):
        if (lenw // 2 + _origin < 0) or (lenw // 2 + _origin >= lenw):
            raise ValueError("invalid origin")
    if mode not in ("reflect", "constant", "nearest", "mirror", "wrap"):
        msg = "boundary mode not supported (actual: {}).".format(mode)
        raise RuntimeError(msg)
    if operation == "median":
        rank = filter_size // 2
    elif operation == "percentile":
        percentile = rank
        if percentile < 0.0:
            percentile += 100.0
        if percentile < 0 or percentile > 100:
            raise RuntimeError("invalid percentile")
        if percentile == 100.0:
            rank = filter_size - 1
        else:
            rank = int(float(filter_size) * percentile / 100.0)
    if rank < 0:
        rank += filter_size
    if rank < 0 or rank >= filter_size:
        raise RuntimeError("rank not within filter footprint size")

    if rank == 0:
        return minimum_filter(
            input, None, footprint, output, mode, cval, origin
        )
    if rank == filter_size - 1:
        return maximum_filter(
            input, None, footprint, output, mode, cval, origin
        )
    # if footprint.dtype.char != '?':
    #     raise ValueError("filter footprint must be boolean")

    output = _ni_support._get_output(output, input)

    input = cupy.ascontiguousarray(input)

    # unsigned_output =  output.dtype.kind in ['u', 'b']

    if filter_size != functools.reduce(operator.mul, fshape):
        # The kernel needs only the non-zero footprint coordinates
        wlocs = cupy.stack(cupy.nonzero(footprint))  # (ndim, nnz)

        return _get_rank_kernel_masked(
            mode,
            cval,
            input.shape,
            footprint.shape,
            wlocs.shape[1],
            tuple(origin),
            rank,
        )(input, wlocs, output)
    else:
        footprint = cupy.ascontiguousarray(footprint)
        return _get_rank_kernel(
            mode, cval, input.shape, footprint.shape, tuple(origin), rank
        )(input, footprint, output)


def rank_filter(
    input,
    rank,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """Multi-dimensional rank filter.

    Args:
        input (cupy.ndarray): The input array.
        rank (int): The rank parameter may be less then zero, i.e., ``rank=-1``
            indicates the largest element.
        size (int or None): see footprint. Ignored if footprint is given.
        footprint (cupy.ndarray or None): Either size or footprint must be
            defined. size gives the shape that is taken from the input array,
            at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly)
            a shape, but also which of the elements within this shape will get
            passed to the filter function. Thus ``size=(n, m)`` is equivalent to
            ``footprint=cupy.ones((n, m))``. We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
            When footprint is given, size is ignored.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = operator.index(rank)
    return _rank_filter(
        input, rank, size, footprint, output, mode, cval, origin
    )


def median_filter(
    input,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """Multi-dimensional median filter.

    Args:
        input (cupy.ndarray): The input array.
        size (int or None): see footprint. Ignored if footprint is given.
        footprint (cupy.ndarray or None): Either size or footprint must be
            defined. size gives the shape that is taken from the input array,
            at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly)
            a shape, but also which of the elements within this shape will get
            passed to the filter function. Thus ``size=(n, m)`` is equivalent to
            ``footprint=cupy.ones((n, m))``. We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
            When footprint is given, size is ignored.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(
        input, None, size, footprint, output, mode, cval, origin, "median"
    )


def percentile_filter(
    input,
    percentile,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """Multi-dimensional percentile filter.

    Args:
        input (cupy.ndarray): The input array.
        percentile (float): The percentile parameter may be less then zero,
            i.e., percentile = -20 equals percentile = 80
        size (int or None): see footprint. Ignored if footprint is given.
        footprint (cupy.ndarray or None): Either size or footprint must be
            defined. size gives the shape that is taken from the input array,
            at every element position, to define the input to the filter
            function. footprint is a boolean array that specifies (implicitly)
            a shape, but also which of the elements within this shape will get
            passed to the filter function. Thus ``size=(n, m)`` is equivalent to
            ``footprint=cupy.ones((n, m))``. We adjust size to the number of
            dimensions of the input array, so that, if the input array is shape
            (10,10,10), and size is 2, then the actual size used is (2,2,2).
            When footprint is given, size is ignored.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``constant``. Default is ``0.0``.
        origin (scalar or tuple of scalar): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of convolution.

    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    return _rank_filter(
        input,
        percentile,
        size,
        footprint,
        output,
        mode,
        cval,
        origin,
        "percentile",
    )
