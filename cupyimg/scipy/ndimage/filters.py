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

# from ._kernels.filters import (
#     #    _get_min_or_max_kernel,
#     #    _get_min_or_max_kernel_masked,
#     #    _get_min_or_max_kernel_masked_v2,
# #    _get_rank_kernel,
# #    _get_rank_kernel_masked,
# )
from . import _util
from ._kernels.filters_v2 import _correlate_or_convolve
from cupyimg import _misc
from cupyimg.scipy.ndimage import _filters_core
from cupyimg.scipy.ndimage import _filters_optimal_medians

median_preambles = _filters_optimal_medians._opt_med_preambles
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

    origin = _util._check_origin(origin, len(weights))

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

    axis = _misc._normalize_axis_index(axis, input.ndim)
    if backend == "fast_upfirdn":
        origin = _util._check_origin(origin, len(weights))

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

        mode_kwargs = _util._get_ndimage_mode_kwargs(mode, cval)

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
            output = _util._get_output(output, input)
            needs_temp = cupy.shares_memory(output, input, "MAY_SHARE_BOUNDS")
            if needs_temp:
                output, temp = (
                    _util._get_output(output.dtype, input),
                    output,
                )

        output = _convolve1d_gpu(
            weights,
            input,
            axis=axis,
            offset=offset,
            crop=crop,
            out=output,
            **mode_kwargs,
        )
        if needs_temp:
            temp[...] = output[...]
            output = temp
        return output
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


# def _correlate_or_convolve(input, weights, output, mode, cval, origin,
#                            convolution=False):
#     origins, int_type = _filters_core._check_nd_args(input, weights,
#                                                      mode, origin)
#     if weights.size == 0:
#         return cupy.zeros_like(input)

#     _util._check_cval(mode, cval, _util._is_integer_output(output, input))

#     if convolution:
#         weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
#         origins = list(origins)
#         for i, wsize in enumerate(weights.shape):
#             origins[i] = -origins[i]
#             if wsize % 2 == 0:
#                 origins[i] -= 1
#         origins = tuple(origins)
#     offsets = _filters_core._origins_to_offsets(origins, weights.shape)
#     kernel = _get_correlate_kernel(mode, weights.shape, int_type,
#                                    offsets, cval)
#     output = _filters_core._call_kernel(kernel, input, weights, output)
#     return output


@cupy._util.memoize(for_each_device=True)
def _get_correlate_kernel(mode, w_shape, int_type, offsets, cval):
    return _filters_core._generate_nd_kernel(
        "correlate",
        "W sum = (W)0;",
        "sum += cast<W>({value}) * wval;",
        "y = cast<Y>(sum);",
        mode,
        w_shape,
        int_type,
        offsets,
        cval,
        ctype="W",
    )


def _run_1d_correlates(
    input, params, get_weights, output, mode, cval, origin=0
):
    """
    Enhanced version of _run_1d_filters that uses correlate1d as the filter
    function. The params are a list of values to pass to the get_weights
    callable given. If duplicate param values are found, the weights are
    reused from the first invocation of get_weights. The get_weights callable
    must return a 1D array of weights to give to correlate1d.
    """
    wghts = {}
    for param in params:
        if param not in wghts:
            wghts[param] = get_weights(param)
    wghts = [wghts[param] for param in params]
    return _filters_core._run_1d_filters(
        [None if w is None else correlate1d for w in wghts],
        input,
        wghts,
        output,
        mode,
        cval,
        origin,
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
    output = _util._get_output(output, input)
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
    output = _util._get_output(output, input)
    sizes = _util._normalize_sequence(size, input.ndim)
    origins = _util._normalize_sequence(origin, input.ndim)
    modes = _util._normalize_sequence(mode, input.ndim)
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
    output = _util._get_output(output, input)
    orders = _util._normalize_sequence(order, input.ndim)
    sigmas = _util._normalize_sequence(sigma, input.ndim)
    modes = _util._normalize_sequence(mode, input.ndim)
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


def prewitt(input, axis=-1, output=None, mode="reflect", cval=0.0):
    """Apply a Prewitt filter.

    See ``scipy.ndimage.prewitt``
    """
    dtype_weights = numpy.promote_types(input.real.dtype, numpy.float32)
    axis = _misc._normalize_axis_index(axis, input.ndim)
    output = _util._get_output(output, input)
    modes = _util._normalize_sequence(mode, input.ndim)
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
    output = _util._get_output(output, input)
    modes = _util._normalize_sequence(mode, input.ndim)
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
    output = _util._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        modes = _util._normalize_sequence(mode, len(axes))
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
    output = _util._get_output(output, input)
    axes = list(range(input.ndim))
    if len(axes) > 0:
        modes = _util._normalize_sequence(mode, len(axes))
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
        fshape = tuple(_util._normalize_sequence(size, ndim))
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
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.minimum_filter`
    """
    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, "min"
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
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.maximum_filter`
    """
    return _min_or_max_filter(
        input, size, footprint, None, output, mode, cval, origin, "max"
    )


def _min_or_max_filter(
    input, size, ftprnt, structure, output, mode, cval, origin, func
):
    # structure is used by morphology.grey_erosion() and grey_dilation()
    # and not by the regular min/max filters

    sizes, ftprnt, structure = _filters_core._check_size_footprint_structure(
        input.ndim, size, ftprnt, structure
    )
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")

    if sizes is not None:
        # Seperable filter, run as a series of 1D filters
        fltr = minimum_filter1d if func == "min" else maximum_filter1d
        return _filters_core._run_1d_filters(
            [fltr if size > 1 else None for size in sizes],
            input,
            sizes,
            output,
            mode,
            cval,
            origin,
        )

    origins, int_type = _filters_core._check_nd_args(
        input, ftprnt, mode, origin, "footprint"
    )
    if structure is not None and structure.ndim != input.ndim:
        raise RuntimeError("structure array has incorrect shape")

    if ftprnt.size == 0:
        return cupy.zeros_like(input)
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(
        mode,
        ftprnt.shape,
        func,
        offsets,
        float(cval),
        int_type,
        has_structure=structure is not None,
        has_central_value=bool(ftprnt[offsets]),
    )
    return _filters_core._call_kernel(
        kernel, input, ftprnt, output, structure, weights_dtype=bool
    )


def minimum_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Compute the minimum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the minimum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.minimum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, "min")


def maximum_filter1d(
    input, size, axis=-1, output=None, mode="reflect", cval=0.0, origin=0
):
    """Compute the maximum filter along a single axis.

    Args:
        input (cupy.ndarray): The input array.
        size (int): Length of the maximum filter.
        axis (int): The axis of input along which to calculate. Default is -1.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int): The origin parameter controls the placement of the
            filter, relative to the center of the current element of the
            input. Default is ``0``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.maximum_filter1d`
    """
    return _min_or_max_1d(input, size, axis, output, mode, cval, origin, "max")


def _min_or_max_1d(
    input,
    size,
    axis=-1,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    func="min",
):
    ftprnt = cupy.ones(size, dtype=bool)
    ftprnt, origin = _filters_core._convert_1d_args(
        input.ndim, ftprnt, origin, axis
    )
    origins, int_type = _filters_core._check_nd_args(
        input, ftprnt, mode, origin, "footprint"
    )
    offsets = _filters_core._origins_to_offsets(origins, ftprnt.shape)
    kernel = _get_min_or_max_kernel(
        mode,
        ftprnt.shape,
        func,
        offsets,
        float(cval),
        int_type,
        has_weights=False,
    )
    return _filters_core._call_kernel(
        kernel, input, None, output, weights_dtype=bool
    )


@cupy._util.memoize(for_each_device=True)
def _get_min_or_max_kernel(
    mode,
    w_shape,
    func,
    offsets,
    cval,
    int_type,
    has_weights=True,
    has_structure=False,
    has_central_value=True,
):
    # When there are no 'weights' (the footprint, for the 1D variants) then
    # we need to make sure intermediate results are stored as doubles for
    # consistent results with scipy.
    ctype = "X" if has_weights else "double"
    value = "{value}"
    if not has_weights:
        value = "cast<double>({})".format(value)

    # Having a non-flat structure biases the values
    if has_structure:
        value += ("-" if func == "min" else "+") + "cast<X>(sval)"

    if has_central_value:
        pre = "{} value = x[i];"
        found = "value = {func}({value}, value);"
    else:
        # If the central pixel is not included in the footprint we cannot
        # assume `x[i]` is not below the min or above the max and thus cannot
        # seed with that value. Instead we keep track of having set `value`.
        pre = "{} value; bool set = false;"
        found = "value = set ? {func}({value}, value) : {value}; set=true;"

    return _filters_core._generate_nd_kernel(
        func,
        pre.format(ctype),
        found.format(func=func, value=value),
        "y = cast<Y>(value);",
        mode,
        w_shape,
        int_type,
        offsets,
        cval,
        ctype=ctype,
        has_weights=has_weights,
        has_structure=has_structure,
    )


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
        rank (int): The rank of the element to get. Can be negative to count
            from the largest value, e.g. ``-1`` indicates the largest value.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.rank_filter`
    """
    rank = operator.index(rank)
    return _rank_filter(
        input,
        lambda fs: rank + fs if rank < 0 else rank,
        size,
        footprint,
        output,
        mode,
        cval,
        origin,
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
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.median_filter`
    """
    return _rank_filter(
        input, lambda fs: fs // 2, size, footprint, output, mode, cval, origin
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
        percentile (scalar): The percentile of the element to get (from ``0``
            to ``100``). Can be negative, thus ``-20`` equals ``80``.
        size (int or sequence of int): One of ``size`` or ``footprint`` must be
            provided. If ``footprint`` is given, ``size`` is ignored. Otherwise
            ``footprint = cupy.ones(size)`` with ``size`` automatically made to
            match the number of dimensions in ``input``.
        footprint (cupy.ndarray): a boolean array which specifies which of the
            elements within this shape will get passed to the filter function.
        output (cupy.ndarray, dtype or None): The array in which to place the
            output. Default is is same dtype as the input.
        mode (str): The array borders are handled according to the given mode
            (``'reflect'``, ``'constant'``, ``'nearest'``, ``'mirror'``,
            ``'wrap'``). Default is ``'reflect'``.
        cval (scalar): Value to fill past edges of input if mode is
            ``'constant'``. Default is ``0.0``.
        origin (int or sequence of int): The origin parameter controls the
            placement of the filter, relative to the center of the current
            element of the input. Default of 0 is equivalent to
            ``(0,)*input.ndim``.

    Returns:
        cupy.ndarray: The result of the filtering.

    .. seealso:: :func:`scipy.ndimage.percentile_filter`
    """
    percentile = float(percentile)
    if percentile < 0.0:
        percentile += 100.0
    if percentile < 0.0 or percentile > 100.0:
        raise RuntimeError("invalid percentile")
    if percentile == 100.0:

        def get_rank(fs):
            return fs - 1

    else:

        def get_rank(fs):
            return int(float(fs) * percentile / 100.0)

    return _rank_filter(
        input, get_rank, size, footprint, output, mode, cval, origin
    )


def _rank_filter(
    input,
    get_rank,
    size=None,
    footprint=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    _, footprint, _ = _filters_core._check_size_footprint_structure(
        input.ndim, size, footprint, None, force_footprint=True
    )
    if cval is cupy.nan:
        raise NotImplementedError("NaN cval is unsupported")
    origins, int_type = _filters_core._check_nd_args(
        input, footprint, mode, origin, "footprint"
    )
    if footprint.size == 0:
        return cupy.zeros_like(input)
    filter_size = int(cupy.count_nonzero(footprint))
    rank = get_rank(filter_size)
    if rank < 0 or rank >= filter_size:
        raise RuntimeError("rank not within filter footprint size")
    if rank == 0:
        return _min_or_max_filter(
            input, None, footprint, None, output, mode, cval, origins, "min"
        )
    if rank == filter_size - 1:
        return _min_or_max_filter(
            input, None, footprint, None, output, mode, cval, origins, "max"
        )
    offsets = _filters_core._origins_to_offsets(origins, footprint.shape)
    kernel = _get_rank_kernel(
        filter_size, rank, mode, footprint.shape, offsets, float(cval), int_type
    )
    return _filters_core._call_kernel(
        kernel, input, footprint, output, weights_dtype=bool
    )


__SHELL_SORT = """
__device__ void sort(X *array, int size) {{
    int gap = {gap};
    while (gap > 1) {{
        gap /= 3;
        for (int i = gap; i < size; ++i) {{
            X value = array[i];
            int j = i - gap;
            while (j >= 0 && value < array[j]) {{
                array[j + gap] = array[j];
                j -= gap;
            }}
            array[j + gap] = value;
        }}
    }}
}}"""


@cupy._util.memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3 * gap + 1
    return gap


@cupy._util.memoize(for_each_device=True)
def _get_rank_kernel(filter_size, rank, mode, w_shape, offsets, cval, int_type):
    is_median = rank == filter_size // 2
    if is_median:
        s_rank = rank
        sorter = median_preambles.get(filter_size, None)
    else:
        s_rank = min(rank, filter_size - rank - 1)
    if is_median and sorter is not None:
        # The sort() function defined in sorter returns only the median
        array_size = filter_size
        found_post = ""
        post = "y=cast<Y>(sort(values));"
    elif s_rank <= 80:
        # When s_rank is small and register usage is low, this partial
        # selection sort approach is faster than general sorting approach
        # using shell sort.
        if s_rank == rank:
            comp_op = "<"
        else:
            comp_op = ">"
        array_size = s_rank + 1
        found_post = """
            if (iv > {rank} + 1) {{{{
                int target_iv = 0;
                X target_val = values[0];
                for (int jv = 1; jv <= {rank} + 1; jv++) {{{{
                    if (target_val {comp_op} values[jv]) {{{{
                        target_val = values[jv];
                        target_iv = jv;
                    }}}}
                }}}}
                if (target_iv <= {rank}) {{{{
                    values[target_iv] = values[{rank} + 1];
                }}}}
                iv = {rank} + 1;
            }}}}""".format(
            rank=s_rank, comp_op=comp_op
        )
        post = """
            X target_val = values[0];
            for (int jv = 1; jv <= {rank}; jv++) {{
                if (target_val {comp_op} values[jv]) {{
                    target_val = values[jv];
                }}
            }}
            y=cast<Y>(target_val);""".format(
            rank=s_rank, comp_op=comp_op
        )
        sorter = ""
    else:
        array_size = filter_size
        found_post = ""
        post = "sort(values,{});\ny=cast<Y>(values[{}]);".format(
            filter_size, rank
        )
        sorter = __SHELL_SORT.format(gap=_get_shell_gap(filter_size))

    return _filters_core._generate_nd_kernel(
        "rank_{}_{}".format(filter_size, rank),
        "int iv = 0;\nX values[{}];".format(array_size),
        "values[iv++] = {value};" + found_post,
        post,
        mode,
        w_shape,
        int_type,
        offsets,
        cval,
        preamble=sorter,
    )
