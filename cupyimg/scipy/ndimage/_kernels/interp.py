
import cupy

from .support import _generate_boundary_condition_ops


def _get_map_coord(ndim):
    """Extract target coordinate from coords array (for map_coordinates).

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        coords (ndarray): array of shape (ncoords, ndim) containing the target
            coordinates.
        c_j: variables to hold the target coordinates

    computes::

        c_j = coords[i + j * ncoords];

    ncoords is determined by the size of the output array, y.
    y will be indexed by the CIndexer, _ind.
    Thus ncoords = _ind.size();

    """
    ops = []
    ops.append("ptrdiff_t ncoords = _ind.size();")
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = coords[i + {j} * ncoords];
            """.format(j=j))
    return ops


def _get_coord_zoom_and_shift(ndim):
    """Compute target coordinate based on a shift followed by a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the zoom for each axis
        c_j: values of the zoomed and shifted coordinates

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j])

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] - shift[{j}]);
            """.format(j=j))
    return ops


def _get_coord_zoom(ndim):
    """Compute target coordinate based on a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        c_j: values of the zoomed and shifted coordinates

    computes::

        c_j = zoom[j] * in_coord[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * (W)in_coord[{j}];
            """.format(j=j))
    return ops


def _get_coord_shift(ndim):
    """Compute target coordinate based on a shift.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        shift[ndim]: array containing the zoom for each axis
        c_j: values of the zoomed and shifted coordinates

    computes::

        c_j = in_coord[j] - shift[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = (W)in_coord[{j}] - shift[{j}];
            """.format(j=j))
    return ops


def _get_coord_affine(ndim):
    """Compute target coordinate based on a homogeneous tranfsormation matrix.

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(ndarray): array containing the (ndim, ndim + 1) transform matrix.
        c_j: values of the zoomed and shifted coordinates

    For example, in 2D:

        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

    """
    ops = []
    ncol = ndim + 1
    for j in range(ndim):
        ops.append("""
            W c_{j} = (W)0.0;
            """.format(j=j))
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            c_{j} += mat[{m_index}] * (W)in_coord[{k}];
                """.format(j=j, k=k, m_index=m_index))
        ops.append(
            """
            c_{j} += mat[{m_index}];
            """.format(j=j, m_index=ncol * j + ndim))
    return ops


def _generate_interp_custom(
    in_params,
    coord_func,
    xshape,
    yshape,
    mode,
    cval,
    order,
    name="",
    integer_output=False,
):
    """
    Args:
        in_params (str): Input arrays. The first must be raw X x. The rest will
            depend on the parameters needed by `coord_func`.
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_zoom_and_shift`.
        xshape (tuple): Shape of the array to be transformed.
        yshape (tuple): Shape of the output array.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        integer_output (bool): boolean indicating whether the output has an
            integer type.

    """
    out_params = "Y y"

    ndim = len(xshape)

    ops = []
    ops.append("double out = 0.0;")
    # ops.append("const ptrdiff_t *in_coord2 = _ind.get();".format(ndim=ndim))
    int_type = "unsigned int"  # TODO: finish converting to use inttype
    ops.append("{int_type} in_coord[{ndim}];".format(int_type=int_type, ndim=ndim))

    # determine strides for x along each axis
    ops.append("const int sx_{} = 1;".format(ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            "int sx_{jm} = sx_{j} * {xsize_j};".format(
                jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )

    # determine nd coordinate in x corresponding to a given raveled coordinate,
    # i, in y.
    ops.append(
        """
        {int_type} idx = i;
        {int_type} s, t;
    """.format(int_type=int_type))
    for j in range(ndim - 1, 0, -1):
        ops.append("""
            s = {zsize_j};
            t = idx / s;
            in_coord[{j}] = idx - t * s;
            idx = t;
        """.format(j=j, zsize_j=yshape[j]))
    ops.append("in_coord[0] = idx;")

    # compute the transformed (target) coordinates, c_j
    ops = ops + coord_func(ndim)

    def _init_coords(order, mode):
        ops = []
        if order == 0:
            for j in range(ndim):
                ops.append("""
                int cf_{j} = (int)lrint((double)c_{j});
                """.format(j=j))
            for j in range(ndim):
                ops.append("int cf_bounded_{j} = cf_{j};".format(j=j))

                # handle boundaries for extension modes.
                ixvar = "cf_bounded_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))
        else:
            for j in range(ndim):
                ops.append("""
                int cf_{j} = (int)floor((double)c_{j});
                int cc_{j} = cf_{j} + 1;
                int n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // 1 or 2 points needed?
                """.format(j=j))

            for j in range(ndim):
                ops.append("int cf_bounded_{j} = cf_{j};".format(j=j))
                ops.append("int cc_bounded_{j} = cc_{j};".format(j=j))
                # handle boundaries for extension modes.
                ixvar = "cf_bounded_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))
                ixvar = "cc_bounded_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, xshape[j]))
        return ops

    if mode == 'constant':
        _cond = " || ".join(["(c_{j} < 0) || (c_{j} > (int){cmax})".format(j=j, cmax=xshape[j] - 1) for j in range(ndim)])
        ops.append("""
            if ({cond})
            {{
                out = (double){cval};
            }}
            else
            {{""".format(cond=_cond, cval=cval))

    ops += _init_coords(order, mode)
    if order == 0:
        for j in range(ndim):
            ops.append("""
            int ic_{j} = cf_bounded_{j} * sx_{j};
            """.format(j=j))
        _coord_idx = " + ".join(["ic_{}".format(j) for j in range(ndim)])
        ops.append("""
            out = x[{coord_idx}];
            """.format(coord_idx=_coord_idx))

    elif order == 1:
        for j in range(ndim):
            ops.append("""
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    W w_{j};
                    int ic_{j};
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}""".format(j=j))

        _weight = " * ".join(["w_{j}".format(j=j) for j in range(ndim)])
        _coord_idx = " + ".join(["ic_{j}".format(j=j) for j in range(ndim)])
        ops.append("""
            X val = x[{coord_idx}];
            out += val * ({weight});
            """.format(coord_idx=_coord_idx, weight=_weight))
        ops.append("}" * ndim)

    if mode == "constant":
        ops.append("}")

    if integer_output:
        ops.append("y = (Y)rint((double)out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    name = "interpolate_{}_order{}_{}_x{}_y{}".format(
        name,
        order,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in yshape]),
    )
    return in_params, out_params, operation, name


@cupy.util.memoize()
def _get_interp_kernel(xshape, mode, cval=0.0, order=1, integer_output=False):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_interp_custom(
        in_params="raw X x, raw W coords",
        coord_func=_get_map_coord,
        xshape=xshape,
        yshape=xshape,
        mode=mode,
        cval=cval,
        order=order,
        name="shift",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_interp_shift_kernel(xshape, yshape, mode, cval=0.0, order=1, integer_output=False):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_interp_custom(
        in_params="raw X x, raw W shift",
        coord_func=_get_coord_shift,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="shift",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_interp_zoom_shift_kernel(xshape, yshape, mode, cval=0.0, order=1, integer_output=False):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_interp_custom(
        in_params="raw X x, raw W shift, raw W zoom",
        coord_func=_get_coord_zoom_and_shift,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_shift",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_interp_zoom_kernel(xshape, yshape, mode, cval=0.0, order=1, integer_output=False):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_interp_custom(
        in_params="raw X x, raw W zoom",
        coord_func=_get_coord_zoom,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize()
def _get_interp_affine_kernel(xshape, yshape, mode, cval=0.0, order=1, integer_output=False):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_interp_custom(
        in_params="raw X x, raw W mat",
        coord_func=_get_coord_affine,
        xshape=xshape,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="affine",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)
