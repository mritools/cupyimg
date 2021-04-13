import cupy
import numpy

from cupyimg import memoize
from cupyimg.scipy.ndimage import _spline_prefilter_core
from cupyimg.scipy.ndimage import _spline_kernel_weights
from cupyimg.scipy.ndimage import _util

math_constants_preamble = r"""
// workaround for HIP: line begins with #include
#include <cupy/math_constants.h>
"""
spline_weights_inline = _spline_kernel_weights.spline_weights_inline
boundary_ops = _util._generate_boundary_condition_ops


def _get_coord_map(ndim, nprepad=0):
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
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
            W c_{j} = coords[i + {j} * ncoords]{pre};""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_zoom_and_shift(ndim, nprepad=0):
    """Compute target coordinate based on a shift followed by a zoom.

    This version zooms from the center of the edge pixels.

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
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
        W c_{j} = (zoom[{j}] *
                   ((W)in_coord[{j}] - shift[{j}]){pre});""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_zoom_and_shift_grid(ndim, nprepad=0):
    """Compute target coordinate based on a shift followed by a zoom.

    This version zooms from the outer edges of the grid pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] - shift[j] + 0.5) - 0.5

    """
    ops = []
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
        W c_{j} = (
            zoom[{j}] * ((W)in_coord[{j}] - shift[{j}] + 0.5) - 0.5{pre}
        );""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_zoom(ndim, nprepad=0):
    """Compute target coordinate based on a zoom.

    This version zooms from the center of the edge pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * in_coord[j]

    """
    ops = []
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * (W)in_coord[{j}]{pre};""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_zoom_grid(ndim, nprepad=0):
    """Compute target coordinate based on a zoom (grid_mode=True version).

    This version zooms from the outer edges of the grid pixels.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        zoom[ndim]: array containing the zoom for each axis

    computes::

        c_j = zoom[j] * (in_coord[j] + 0.5) - 0.5

    """
    ops = []
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = zoom[{j}] * ((W)in_coord[{j}] + 0.5) - 0.5{pre};""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_shift(ndim, nprepad=0):
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
    pre = " + (W){nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
    for j in range(ndim):
        ops.append(
            """
    W c_{j} = (W)in_coord[{j}] - shift[{j}]{pre};""".format(
                j=j, pre=pre
            )
        )
    return ops


def _get_coord_affine(ndim, nprepad=0):
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
    pre = " + {nprepad}".format(nprepad=nprepad) if nprepad > 0 else ""
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
            c_{j} += mat[{m_index}] * (W)in_coord[{k}];""".format(
                    j=j, k=k, m_index=m_index
                )
            )
        ops.append(
            """
            c_{j} += mat[{m_index}]{pre};""".format(
                j=j, m_index=ncol * j + ndim, pre=pre,
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
    nprepad=0,
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

    ops = []
    ops.append("double out = 0.0;")

    if large_int:
        uint_t = "size_t"
        int_t = "ptrdiff_t"
    else:
        uint_t = "unsigned int"
        int_t = "int"

    # determine strides of x (in elements, not bytes)
    for j in range(ndim):
        ops.append(f"const {int_t} xsize_{j} = x.shape()[{j}];")
    ops.append(f"const {uint_t} sx_{ndim - 1} = 1;")
    for j in range(ndim - 1, 0, -1):
        ops.append(f"const {uint_t} sx_{j - 1} = sx_{j} * xsize_{j};")

    # create out_coords array to store the unraveled indices into the output
    ops.append(_unravel_loop_index(yshape, uint_t))

    # compute the transformed (target) coordinates, c_j
    ops = ops + coord_func(ndim, nprepad)

    if cval is numpy.nan:
        cval = "CUDART_NAN"
    elif cval == numpy.inf:
        cval = "CUDART_INF"
    elif cval == -numpy.inf:
        cval = "-CUDART_INF"
    else:
        cval = "(double){cval}".format(cval=cval)
    if mode == "constant":
        # use cval if coordinate is outside the bounds of x
        _cond = " || ".join(
            [f"(c_{j} < 0) || (c_{j} > xsize_{j} - 1)" for j in range(ndim)]
        )
        ops.append(
            f"""
        if ({_cond})
        {{
            out = (double){cval};
        }}
        else
        {{"""
        )

    if order == 0:
        ops.append("double dcoord;")
        for j in range(ndim):
            # determine nearest neighbor
            if mode == "wrap":
                ops.append(f"dcoord = c_{j};")
            else:
                ops.append(
                    f"""
                {int_t} cf_{j} = ({int_t})lrint((double)c_{j});
                """
                )

            # handle boundary
            if mode != "constant":
                if mode == "wrap":
                    ixvar = "dcoord"
                    float_ix = True
                else:
                    ixvar = f"cf_{j}"
                    float_ix = False
                ops.append(
                    boundary_ops(mode, ixvar, f"xsize_{j}", int_t, float_ix)
                )
                if mode == "wrap":
                    ops.append(
                        f"{int_t} cf_{j} = ({int_t})floor(dcoord + 0.5);"
                    )

            # sum over ic_j will give the raveled coordinate in the input
            ops.append(
                f"""
            {int_t} ic_{j} = cf_{j} * sx_{j};
            """
            )
        _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
        if mode == "grid-constant":
            _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
            ops.append(
                f"""
            if ({_cond}) {{
                out = (double){cval};
            }} else {{
                out = x[{_coord_idx}];
            }}
            """
            )
        else:
            ops.append(
                f"""
                out = x[{_coord_idx}];
                """
            )

    elif order == 1:
        for j in range(ndim):
            # get coordinates for linear interpolation along axis j
            ops.append(
                f"""
            {int_t} cf_{j} = ({int_t})floor((double)c_{j});
            {int_t} cc_{j} = cf_{j} + 1;
            {int_t} n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // points needed
            """
            )

            if mode == "wrap":
                ops.append(f"double dcoordf = c_{j};")
                ops.append(f"double dcoordc = c_{j} + 1;")
            else:
                # handle boundaries for extension modes.
                ops.append(
                    f"""
                {int_t} cf_bounded_{j} = cf_{j};
                {int_t} cc_bounded_{j} = cc_{j};
                """
                )
            if mode != "constant":
                if mode == "wrap":
                    ixvar = "dcoordf"
                    float_ix = True
                else:
                    ixvar = f"cf_bounded_{j}"
                    float_ix = False
                ops.append(
                    boundary_ops(mode, ixvar, f"xsize_{j}", int_t, float_ix)
                )
                if mode == "wrap":
                    ixvar = "dcoordc"
                else:
                    ixvar = f"cc_bounded_{j}"
                    ops.append(
                        boundary_ops(mode, ixvar, f"xsize_{j}", int_t, float_ix)
                    )
                if mode == "wrap":
                    ops.append(
                        f"""
                    {int_t} cf_bounded_{j} = ({int_t})floor(dcoordf);;
                    {int_t} cc_bounded_{j} = ({int_t})floor(dcoordf + 1);;
                    """
                    )

            ops.append(
                f"""
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
                    }}"""
            )

    elif order > 1:
        if mode == "grid-constant":
            spline_mode = "constant"
        elif mode == "nearest":
            spline_mode = "nearest"
        else:
            spline_mode = _spline_prefilter_core._get_spline_mode(mode)

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
            f"""
        {int_t} start;"""
        )
        for j in range(ndim):
            # determine weights along the current axis
            ops.append(
                f"""
            W weights_{j}[{order + 1}];"""
            )
            ops.append(spline_weights_inline[order].format(j=j, order=order))

            # get starting coordinate for spline interpolation along axis j
            if mode in ["wrap"]:
                ops.append(f"double dcoord = c_{j};")
                ixvar = "dcoord"
                ops.append(boundary_ops(mode, ixvar, f"xsize_{j}", int_t, True))
                coord_var = "dcoord"
            else:
                coord_var = f"(double)c_{j}"

            if order & 1:
                op_str = """
                start = ({int_t})floor({coord_var}) - {order_2};"""
            else:
                op_str = """
                start = ({int_t})floor({coord_var} + 0.5) - {order_2};"""
            ops.append(
                op_str.format(
                    int_t=int_t, coord_var=coord_var, order_2=order // 2
                )
            )

            # set of coordinate values within spline footprint along axis j
            ops.append(f"""{int_t} ci_{j}[{order + 1}];""")
            for k in range(order + 1):
                ops.append(
                    f"""
                ci_{j}[{k}] = start + {k};"""
                )
                ixvar = f"ci_{j}[{k}]"
                ops.append(
                    boundary_ops(
                        spline_mode, ixvar, "xsize_{}".format(j), int_t
                    )
                )

            # loop over the order + 1 values in the spline filter
            ops.append(
                f"""
            W w_{j};
            {int_t} ic_{j};
            for (int k_{j} = 0; k_{j} <= {order}; k_{j}++)
                {{
                    w_{j} = weights_{j}[k_{j}];
                    ic_{j} = ci_{j}[k_{j}] * sx_{j};
            """
            )

    if order > 0:

        _weight = " * ".join([f"w_{j}" for j in range(ndim)])
        _coord_idx = " + ".join([f"ic_{j}" for j in range(ndim)])
        if mode == "grid-constant" or (order > 1 and mode == "constant"):
            _cond = " || ".join([f"(ic_{j} < 0)" for j in range(ndim)])
            ops.append(
                f"""
            if ({_cond}) {{
                out += (X){cval} * ({_weight});
            }} else {{
                X val = x[{_coord_idx}];
                out += val * ({_weight});
            }}
            """
            )
        else:
            ops.append(
                f"""
            X val = x[{_coord_idx}];
            out += val * ({_weight});
            """
            )

        ops.append("}" * ndim)

    if mode == "constant":
        ops.append("}")

    if integer_output:
        ops.append("y = (Y)rint((double)out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    modestr = mode.replace("-", "_")
    name = "interpolate_{}_order{}_{}_{}d_y{}".format(
        name, order, modestr, ndim, "_".join(["{}".format(j) for j in yshape]),
    )
    if uint_t == "size_t":
        name += "_i64"
    return operation, name


@memoize(for_each_device=True)
def _get_map_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
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
        nprepad=nprepad,
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@memoize(for_each_device=True)
def _get_shift_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
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
        nprepad=nprepad,
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )


@memoize(for_each_device=True)
def _get_zoom_shift_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
    grid_mode=False,
):
    in_params = "raw X x, raw W shift, raw W zoom"
    out_params = "Y y"
    if grid_mode:
        zoom_shift_func = _get_coord_zoom_and_shift_grid
    else:
        zoom_shift_func = _get_coord_zoom_and_shift
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=zoom_shift_func,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_shift_grid" if grid_mode else "zoom_shift",
        integer_output=integer_output,
        nprepad=nprepad,
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )


@memoize(for_each_device=True)
def _get_zoom_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
    grid_mode=False,
):
    in_params = "raw X x, raw W zoom"
    out_params = "Y y"
    operation, name = _generate_interp_custom(
        in_params=in_params,
        coord_func=_get_coord_zoom_grid if grid_mode else _get_coord_zoom,
        ndim=ndim,
        large_int=large_int,
        yshape=yshape,
        mode=mode,
        cval=cval,
        order=order,
        name="zoom_grid" if grid_mode else "zoom",
        integer_output=integer_output,
        nprepad=nprepad,
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )


@memoize(for_each_device=True)
def _get_affine_kernel(
    ndim,
    large_int,
    yshape,
    mode,
    cval=0.0,
    order=1,
    integer_output=False,
    nprepad=0,
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
        nprepad=nprepad,
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=math_constants_preamble
    )
