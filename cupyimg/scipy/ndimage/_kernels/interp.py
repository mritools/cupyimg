import cupy
import cupy.core.internal

from .support import _generate_boundary_condition_ops, _raw_ptr_ops
from .spline import spline_weights_inline

const_legacy_mode = False


def _get_coord_map(ndim):
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
    W c_{j} = coords_data[i + {j} * ncoords];""".format(
                j=j
            )
        )
    return ops


def _get_coord_zoom_and_shift(ndim):
    """Compute target coordinate based on a shift followed by a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j])

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom_data[{j}] * ((W)in_coord[{j}] - shift_data[{j}]);""".format(
                j=j
            )
        )
    return ops


def _get_coord_zoom(ndim):
    """Compute target coordinate based on a zoom.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * in_coord[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom_data[{j}] * (W)in_coord[{j}];""".format(
                j=j
            )
        )
    return ops


def _get_coord_shift(ndim):
    """Compute target coordinate based on a shift.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = in_coord[j] - shift[j]

    """
    ops = []
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = (W)in_coord[{j}] - shift_data[{j}];""".format(
                j=j
            )
        )
    return ops


def _get_coord_affine(ndim):
    """Compute target coordinate based on a homogeneous transformation matrix.

    The homogeneous matrix has shape (ndim, ndim + 1). It corresponds to
    affine matrix where the last row of the affine is assumed to be:
    ``[0] * ndim + [1]``.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        mat(array): array containing the (ndim, ndim + 1) transform matrix.
        in_coords(array): coordinates of the input

    For example, in 2D:

        c_0 = mat[0] * in_coords[0] + mat[1] * in_coords[1] + aff[2];
        c_1 = mat[3] * in_coords[0] + mat[4] * in_coords[1] + aff[5];

    """
    ops = []
    ncol = ndim + 1
    for j in range(ndim):
        ops.append(
            """
            W c_{j} = (W)0.0;""".format(
                j=j
            )
        )
        for k in range(ndim):
            m_index = ncol * j + k
            ops.append(
                """
            c_{j} += mat_data[{m_index}] * (W)in_coord[{k}];""".format(
                    j=j, k=k, m_index=m_index
                )
            )
        ops.append(
            """
            c_{j} += mat_data[{m_index}];""".format(
                j=j, m_index=ncol * j + ndim
            )
        )
    return ops


def _unravel_loop_index(shape, uint_t="unsigned int"):
    """
    declare a multi-index array in_coord and unravel the 1D index, i into it.
    This code assumes that the array is a C-ordered array.
    """
    ndim = len(shape)
    code = [
        """
        {uint_t} in_coord[{ndim}];
        {uint_t} s, t, idx = i;
        """.format(
            uint_t=uint_t, ndim=ndim
        )
    ]
    for j in range(ndim - 1, 0, -1):
        code.append(
            """
        s = {size};
        t = idx / s;
        in_coord[{j}] = idx - t * s;
        idx = t;
        """.format(
                j=j, size=shape[j]
            )
        )
    code.append(
        """
        in_coord[0] = idx;"""
    )
    return "\n".join(code)


def _generate_interp_custom(
    in_params,
    coord_func,
    ndim,
    large_int,
    yshape,
    mode,
    cval,
    order,
    name="",
    integer_output=False,
):
    """
    Args:
        in_params (str): input parameters for the ElementwiseKernel
        coord_func (function): generates code to do the coordinate
            transformation. See for example, `_get_coord_shift`.
        ndim (int): The number of dimensions.
        large_int (bool): If true use Py_ssize_t instead of int for indexing.
        yshape (tuple): Shape of the output array.
        mode (str): Signal extension mode to use at the array boundaries
        cval (float): constant value used when `mode == 'constant'`.
        name (str): base name for the interpolation kernel
        integer_output (bool): boolean indicating whether the output has an
            integer type.

    Returns:
        operation (str): code body for the ElementwiseKernel
        name (str): name for the ElementwiseKernel
    """

    if mode == "constant" and not const_legacy_mode:
        mode = "constant2"  # only set constant2 if want bug-fixed constant mode

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    ops.append("double out = 0.0;")

    if large_int:
        uint_t = "size_t"
        int_t = "ptrdiff_t"
    else:
        uint_t = "unsigned int"
        int_t = "int"

    # determine strides of x (in elements, not bytes)
    for j in range(ndim):
        ops.append(
            "const {int_t} xsize_{j} = x.shape()[{j}];".format(int_t=int_t, j=j)
        )
    ops.append("const {uint_t} sx_{j} = 1;".format(uint_t=uint_t, j=ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            "const {uint_t} sx_{jm} = sx_{j} * xsize_{j};".format(
                uint_t=uint_t, jm=j - 1, j=j,
            )
        )

    # create out_coords array to store the unraveled indices into the output
    ops.append(_unravel_loop_index(yshape, uint_t))

    # compute the transformed (target) coordinates, c_j
    ops = ops + coord_func(ndim)

    if mode == "constant":
        # use cval if coordinate is outside the bounds of x
        _cond = " || ".join(
            [
                "(c_{j} < 0) || (c_{j} > xsize_{j} - 1)".format(j=j)
                for j in range(ndim)
            ]
        )
        ops.append(
            """
        if ({cond})
        {{
            out = (double){cval};
        }}
        else
        {{""".format(
                cond=_cond, cval=cval
            )
        )

    if order == 0:
        for j in range(ndim):
            # determine nearest neighbor
            ops.append(
                """
            {int_t} cf_{j} = ({int_t})lrint((double)c_{j});
            """.format(
                    int_t=int_t, j=j
                )
            )

            # handle boundary
            if mode != "constant":
                ixvar = "cf_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, "xsize_{}".format(j)
                    )
                )

            # sum over ic_j will give the raveled coordinate in the input
            ops.append(
                """
            {int_t} ic_{j} = cf_{j} * sx_{j};
            """.format(
                    int_t=int_t, j=j
                )
            )
        _coord_idx = " + ".join(["ic_{}".format(j) for j in range(ndim)])
        if mode == "constant2":
            _cond = " || ".join(["(ic_{0} < 0)".format(j) for j in range(ndim)])
            ops.append(
                """
            if ({cond}) {{
                out = (double){cval};
            }} else {{
                out = x_data[{coord_idx}];
            }}
            """.format(
                    cond=_cond, cval=cval, coord_idx=_coord_idx
                )
            )
        else:
            ops.append(
                """
                out = x_data[{coord_idx}];
                """.format(
                    coord_idx=_coord_idx
                )
            )

    elif order == 1:
        for j in range(ndim):
            # get coordinates for linear interpolation along axis j
            ops.append(
                """
            {int_t} cf_{j} = ({int_t})floor((double)c_{j});
            {int_t} cc_{j} = cf_{j} + 1;
            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed
            """.format(
                    int_t=int_t, j=j
                )
            )

            # handle boundaries for extension modes.
            ops.append(
                """
            {int_t} cf_bounded_{j} = cf_{j};
            {int_t} cc_bounded_{j} = cc_{j};
            """.format(
                    int_t=int_t, j=j
                )
            )
            if mode != "constant":
                ixvar = "cf_bounded_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, "xsize_{}".format(j)
                    )
                )
                ixvar = "cc_bounded_{j}".format(j=j)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, "xsize_{}".format(j)
                    )
                )

            ops.append(
                """
            W w_{j};
            {int_t} ic_{j};
            for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
                {{
                    if (s_{j} == 0)
                    {{
                        w_{j} = (W)cc_{j} - c_{j};
                        ic_{j} = cf_bounded_{j} * sx_{j};
                    }} else
                    {{
                        w_{j} = c_{j} - (W)cf_{j};
                        ic_{j} = cc_bounded_{j} * sx_{j};
                    }}""".format(
                    int_t=int_t, j=j
                )
            )

    elif order >= 1:
        # wx, wy are temporary variables used during spline weight computation
        if order == 1:
            ops.append(
                """
            W wx;"""
            )
        else:
            ops.append(
                """
            W wx, wy;"""
            )
        ops.append(
            """
        {int_t} start;""".format(
                int_t=int_t
            )
        )
        for j in range(ndim):
            # determine weights along the current axis
            ops.append(
                """
            W weights_{j}[{n}];""".format(
                    j=j, n=order + 1
                )
            )
            ops.append(spline_weights_inline[order].format(j=j, order=order))

            # get starting coordinates for spline interpolation along axis j
            if order & 1:
                op_str = """
                start = ({int_t})floor((double)c_{j}) - {order_2};"""
            else:
                op_str = """
                start = ({int_t})floor((double)c_{j} + 0.5) - {order_2};"""
            ops.append(op_str.format(int_t=int_t, j=j, order_2=order // 2))

            # set of coordinate values within spline footprint along axis j
            ops.append(
                """{int_t} ci_{j}[{n}];""".format(int_t=int_t, j=j, n=order + 1)
            )
            for k in range(order + 1):
                ops.append(
                    """
                ci_{j}[{k}] = start + {k};""".format(
                        j=j, k=k
                    )
                )
                ixvar = "ci_{j}[{k}]".format(j=j, k=k)
                ops.append(
                    _generate_boundary_condition_ops(
                        mode, ixvar, "xsize_{}".format(j)
                    )
                )

            # loop over the order + 1 values in the spline filter
            ops.append(
                """
            W w_{j};
            {int_t} ic_{j};
            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)
                {{
                    w_{j} = weights_{j}[k_{j}];
                    ic_{j} = ci_{j}[k_{j}] * sx_{j};
            """.format(
                    int_t=int_t, j=j, order=order
                )
            )

    if order > 0:

        _weight = " * ".join(["w_{j}".format(j=j) for j in range(ndim)])
        _coord_idx = " + ".join(["ic_{j}".format(j=j) for j in range(ndim)])
        if mode == "constant2" or (order > 1 and mode == "constant"):
            _cond = " || ".join(["(ic_{0} < 0)".format(j) for j in range(ndim)])
            ops.append(
                """
            if ({cond}) {{
                out += (double){cval} * ({weight});
            }} else {{
                X val = x_data[{coord_idx}];
                out += val * ({weight});
            }}
            """.format(
                    cond=_cond, cval=cval, coord_idx=_coord_idx, weight=_weight
                )
            )
        else:
            ops.append(
                """
            X val = x_data[{coord_idx}];
            out += val * ({weight});
            """.format(
                    coord_idx=_coord_idx, weight=_weight
                )
            )

        ops.append("}" * ndim)

    if mode == "constant":
        ops.append("}")

    if integer_output:
        ops.append("y = (Y)rint((double)out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    name = "interpolate_{}_order{}_{}_{}d_y{}".format(
        name, order, mode, ndim, "_".join(["{}".format(j) for j in yshape]),
    )
    if uint_t == "size_t":
        name += "_i64"
    return operation, name


@cupy.util.memoize(for_each_device=True)
def _get_map_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    in_params = "raw X x, raw W coords"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_map,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="map_coordinates",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize(for_each_device=True)
def _get_shift_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    in_params = "raw X x, raw W shift"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_shift,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="shift",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize(for_each_device=True)
def _get_zoom_shift_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    in_params = "raw X x, raw W shift, raw W zoom"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_zoom_and_shift,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_shift",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize(for_each_device=True)
def _get_zoom_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    in_params = "raw X x, raw W zoom"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_zoom,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@cupy.util.memoize(for_each_device=True)
def _get_affine_kernel(
    ndim, large_int, yshape, mode, cval=0.0, order=1, integer_output=False
):
    in_params = "raw X x, raw W mat"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_affine,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="affine",
        integer_output=integer_output,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)
