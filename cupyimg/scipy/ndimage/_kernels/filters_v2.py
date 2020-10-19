import warnings

import cupy
from .support import _generate_boundary_condition_ops
from .filters import _get_correlate_kernel_masked
from cupyimg.scipy.ndimage import _util
from cupyimg import memoize
from cupyimg import _misc

# ######## Convolutions and Correlations ##########


def _correlate_or_convolve(
    input,
    weights,
    output,
    mode,
    cval,
    origin,
    convolution,
    dtype_mode,
    use_weights_mask=False,
):
    # if use_weights_mask:
    #     raise NotImplementedError("TODO")
    origins, int_type = _check_nd_args(input, weights, mode, origin)
    if weights.size == 0:
        return cupy.zeros_like(input)
    if convolution:
        weights = weights[tuple([slice(None, None, -1)] * weights.ndim)]
        origins = list(origins)
        for i, wsize in enumerate(weights.shape):
            origins[i] = -origins[i]
            if wsize % 2 == 0:
                origins[i] -= 1
        origins = tuple(origins)
    elif weights.dtype.kind == "c":
        # numpy.correlate conjugates weights rather than input.
        weights = weights.conj()

    if dtype_mode == "numpy":
        # This "numpy" mode is used by cupyimg.scipy.signal.signaltools
        # numpy.convolve and correlate do not always cast to floats
        dtype = cupy.promote_types(input.dtype, weights.dtype)
        output_dtype = dtype
        if dtype.char == "e":
            # promote internal float type to float32 for accuracy
            dtype = "f"
        if output is not None:
            raise ValueError(
                "dtype_mode == 'numpy' does not support the output " "argument"
            )
        weight_dtype = dtype
        if weights.dtype != dtype:
            weights = weights.astype(dtype)
        if input.dtype != dtype:
            input = input.astype(dtype)
        output = cupy.zeros(input.shape, output_dtype)
        weight_dtype = dtype
    else:
        if weights.dtype.kind == "c" or input.dtype.kind == "c":
            if dtype_mode == "ndimage":
                weight_dtype = cupy.complex128
            elif dtype_mode == "float":
                weight_dtype = cupy.promote_types(
                    input.real.dtype, cupy.complex64
                )
        else:
            if dtype_mode == "ndimage":
                weight_dtype = cupy.float64
            elif dtype_mode == "float":
                weight_dtype = cupy.promote_types(
                    input.real.dtype, cupy.float32
                )
        weight_dtype = cupy.dtype(weight_dtype)
        output = _util._get_output(output, input, None, weight_dtype)
    unsigned_output = output.dtype.kind in ["u", "b"]

    if use_weights_mask:
        input = cupy.ascontiguousarray(input)

        # The kernel needs only the non-zero kernel values and their coordinates.
        # This allows us to use a single for loop to compute the ndim convolution.
        # The loop will be over only the the non-zero entries of the filter.
        weights = cupy.ascontiguousarray(weights, weight_dtype)
        wlocs = cupy.nonzero(weights)
        wvals = weights[wlocs]  # (nnz,) array of non-zero values
        wlocs = cupy.stack(
            wlocs
        )  # (ndim, nnz) array of indices for these values

        return _get_correlate_kernel_masked(
            mode,
            cval,
            input.shape,
            weights.shape,
            wvals.size,
            tuple(origins),
            unsigned_output,
        )(input, wlocs, wvals, output)
    else:
        if mode == "constant":
            # TODO: negative strides gives incorrect result for constant mode
            #       so make sure input is C contiguous.
            input = cupy.ascontiguousarray(input)
        kernel = _get_correlate_kernel(
            mode, weights.shape, int_type, origins, cval, unsigned_output
        )
        return _call_kernel(kernel, input, weights, output, weight_dtype)


@memoize()
def _get_correlate_kernel(
    mode, wshape, int_type, origins, cval, unsigned_output
):
    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        # TODO: remove? only needed if dtype_mode == "numpy", but this is
        #       currently untested.
        out_op = "y = (sum > -1) ? (Y)sum : -(Y)(-sum);"
    else:
        out_op = "y = (Y)sum;"

    return _get_nd_kernel(
        "correlate",
        "W sum = (W)0;",
        "sum += (W){value} * wval;",
        out_op,
        mode,
        wshape,
        int_type,
        origins,
        cval,
    )


def _fix_sequence_arg(arg, ndim, name, conv=lambda x: x):
    if hasattr(arg, "__iter__") and not isinstance(arg, str):
        lst = [conv(x) for x in arg]
        if len(lst) != ndim:
            msg = "{} must have length equal to input rank".format(name)
            raise RuntimeError(msg)
    else:
        lst = [conv(arg)] * ndim
    return lst


def _check_mode(mode):
    if mode not in ("reflect", "constant", "nearest", "mirror", "wrap"):
        msg = "boundary mode not supported (actual: {}).".format(mode)
        raise RuntimeError(msg)
    return mode


def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError("incorrect filter size")
    axis = _misc._normalize_axis_index(axis, ndim)
    wshape = [1] * ndim
    wshape[axis] = weights.size
    weights = weights.reshape(wshape)
    origins = [0] * ndim
    origins[axis] = _util._check_origin(origin, weights.size)
    return weights, tuple(origins)


def _check_nd_args(input, weights, mode, origins, wghts_name="filter weights"):
    _check_mode(mode)
    # The integer type to use for positions in input
    # We will always assume that wsize is int32 however
    int_type = "size_t" if input.size > 1 << 31 else "int"
    weight_dims = [x for x in weights.shape if x != 0]
    if len(weight_dims) != input.ndim:
        raise RuntimeError("{} array has incorrect shape".format(wghts_name))
    origins = _fix_sequence_arg(origins, len(weight_dims), "origin", int)
    for origin, width in zip(origins, weight_dims):
        _util._check_origin(origin, width)
    return tuple(origins), int_type


def _call_kernel(kernel, input, weights, output, weight_dtype=cupy.float64):
    """
    Calls a constructed ElementwiseKernel. The kernel must take an input image,
    an array of weights, and an output array.

    The weights are the only optional part and can be passed as None and then
    one less argument is passed to the kernel. If the output is given as None
    then it will be allocated in this function.

    This function deals with making sure that the weights are contiguous and
    float64 or bool*, that the output is allocated and appriopate shaped. This
    also deals with the situation that the input and output arrays overlap in
    memory.

    * weights is always casted to float64 or bool in order to get an output
    compatible with SciPy, though float32 might be sufficient when input dtype
    is low precision.
    """
    if weights is not None:
        weights = cupy.ascontiguousarray(weights, weight_dtype)

    needs_temp = cupy.shares_memory(output, input, "MAY_SHARE_BOUNDS")
    if needs_temp:
        output, temp = (
            _util._get_output(output.dtype, input, None, weight_dtype),
            output,
        )
    if weights is None:
        kernel(input, output)
    else:
        kernel(input, weights, output)
    if needs_temp:
        temp[...] = output[...]
        output = temp
    return output


# ######## Generating Elementwise Kernels ##########


def _get_nd_kernel(
    name,
    pre,
    found,
    post,
    mode,
    wshape,
    int_type,
    origins,
    cval,
    preamble="",
    options=(),
    has_weights=True,
):
    ndim = len(wshape)
    in_params = "raw X x, raw W w"
    out_params = "Y y"

    inds = _generate_indices_ops(
        ndim,
        int_type,
        "xsize_{j}",
        [" - {}".format(wshape[j] // 2 + origins[j]) for j in range(ndim)],
    )
    sizes = [
        "{type} xsize_{j}=x.shape()[{j}], xstride_{j}=x.strides()[{j}];".format(
            j=j, type=int_type
        )
        for j in range(ndim)
    ]
    cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    if has_weights:
        weights_init = "const W* weights = (const W*)&w[0];\nint iw = 0;"
        weights_check = "W wval = weights[iw++];\nif (wval != (W)0)"
    else:
        in_params = "raw X x"
        weights_init = weights_check = ""

    loops = []
    for j in range(ndim):
        if wshape[j] == 1:
            loops.append(
                "{{ {type} ix_{j} = ind_{j} * xstride_{j};".format(
                    j=j, type=int_type
                )
            )
        else:
            boundary = _generate_boundary_condition_ops(
                mode, "ix_{}".format(j), "xsize_{}".format(j)
            )
            loops.append(
                """
    for (int iw_{j} = 0; iw_{j} < {wsize}; iw_{j}++)
    {{
        {type} ix_{j} = ind_{j} + iw_{j};
        {boundary}
        ix_{j} *= xstride_{j};
        """.format(
                    j=j, wsize=wshape[j], boundary=boundary, type=int_type
                )
            )

    value = "(*(X*)&data[{expr}])".format(expr=expr)
    if mode == "constant":
        value = "(({cond}) ? (X){cval} : {value})".format(
            cond=cond, cval=cval, value=value
        )
    found = found.format(value=value)

    operation = """
    {sizes}
    {inds}
    // don't use a CArray for indexing (faster to deal with indexing ourselves)
    const unsigned char* data = (const unsigned char*)&x[0];
    {weights_init}
    {pre}
    {loops}
        // inner-most loop
        {weights_check} {{
            {found}
        }}
    {end_loops}
    {post}
    """.format(
        sizes="\n".join(sizes),
        inds=inds,
        pre=pre,
        post=post,
        weights_init=weights_init,
        weights_check=weights_check,
        loops="\n".join(loops),
        found=found,
        end_loops="}" * ndim,
    )

    name = "cupy_ndimage_{}_{}d_{}_w{}".format(
        name, ndim, mode, "_".join(["{}".format(j) for j in wshape])
    )
    if int_type == "size_t":
        name += "_i64"
    return cupy.ElementwiseKernel(
        in_params,
        out_params,
        operation,
        name,
        reduce_dims=False,
        preamble=preamble,
        options=options,
    )


def _generate_indices_ops(ndim, int_type, xsize="x.shape()[{j}]", extras=None):
    if extras is None:
        extras = ("",) * ndim
    code = "{type} ind_{j} = _i % " + xsize + "{extra}; _i /= " + xsize + ";"
    body = [
        code.format(type=int_type, j=j, extra=extras[j])
        for j in range(ndim - 1, 0, -1)
    ]
    return "{type} _i = i;\n{body}\n{type} ind_0 = _i{extra};".format(
        type=int_type, body="\n".join(body), extra=extras[0]
    )


def _check_size_or_ftprnt(ndim, size, ftprnt, stacklevel, check_sep=False):
    if ftprnt is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _fix_sequence_arg(size, ndim, "size", int)
        if check_sep:
            return sizes, None
        ftprnt = cupy.ones(sizes, dtype=bool)
    else:
        if size is not None:
            warnings.warn(
                "ignoring size because footprint is set",
                UserWarning,
                stacklevel=stacklevel + 1,
            )
        ftprnt = cupy.array(ftprnt, bool, True, "C")
        if not ftprnt.any():
            raise ValueError("All-zero footprint is not supported.")
        if check_sep:
            if ftprnt.all():
                return ftprnt.shape, None
            return None, ftprnt
    return ftprnt
