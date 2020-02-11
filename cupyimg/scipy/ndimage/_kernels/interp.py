import functools
import operator

import cupy


def _get_init_loop_vars(xshape):
    ops = []
    ndim = len(xshape)
    ops.append("const int sx_{} = 1;".format(ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            "int sx_{jm} = sx_{j} * {xsize_j};".format(
                jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )
    ops.append("int _i = i;")
    for j in range(ndim - 1, -1, -1):
        ops.append("int cx_{j} = _i % {xsize};".format(j=j, xsize=xshape[j]))
        if j > 0:
            ops.append("_i /= {xsize};".format(xsize=xshape[j]))
    return ops


def _get_lin_interp_loops(ndim, float_type):
    """
    These loops assume the following variables have been intialized
        sx_j are the shape of the input, x
        c_j are the coordinates at which to interpolation the value

    This loop sets:
        w_j factors that when multiplied give the final weight
        ic_j factors that when summed give the index into the input
    """
    ops = []

    # get the two points used to interpolate the value along each axis
    for j in range(ndim):
        ops.append(
            """
        I cf_{0} = floor(c_{0})
        I cc_{0} = cf_{0} + 1
        size_t n_{0} = (c_{0} == cf_{0}) ? 1 : 2;  // number of points used
        """.format(
                j
            )
        )

    for j in range(ndim):
        ops.append(
            """
        for (size_t s_{0} = 0; s_{0} < n_{0}; s_{0}++)
            {{
                {float_type} w_{0};
                int ic_{0};
                if (s_{0} == 0)
                {{
                    w_{0} = cc_{0} - c_{0};
                    ic_{0} = cf_{0} * sx_{0};
                }} else
                {{
                    w_{0} = c_{0} - cf_{0};
                    ic_{0} = cc_{0} * sx_{0};
                }}""".format(
                j, float_type=float_type
            )
        )
    return ops


def _generate_linear_interp_kernel(
    xshape, coords_shape, double_precision, unsigned_output
):
    """Generate a correlation kernel for dense filters.

    Looping is over the pixels of the output with interpolation based on the
    values in x at the coordinates in coords.

    Notes
    -----
    The kernel needs to loop over pixels in the output image, not the input, so
    the output is set as the first argument here.
    """

    in_params = "raw Y out, raw I coords, raw X x"
    out_params = "Y y"
    # Note: coords.shape = (x.ndim, ) + y.shape

    ndim = len(xshape)
    if coords_shape[0] != ndim:
        raise ValueError("invalid coordinate shape")
    if len(coords_shape) != ndim + 1:
        raise ValueError("invalid coordinate shape")
    ncoords = functools.reduce(operator.mul, coords_shape[1:])

    ops = []
    ops += _get_init_loop_vars(xshape)

    float_type = "double" if double_precision else "float"

    ops.append(
        "{float_type} out = ({float_type})0;".format(float_type=float_type)
    )

    # extract coordinates corresponding to the current output point
    for j in range(ndim):
        ops.append(
            """
        I c_{0} = coords[i + {0} * {ncoords}]
        """.format(
                j, ncoords=ncoords
            )
        )

    ops += _get_lin_interp_loops(ndim, float_type)

    _weight = " * ".join(["w_{0}".format(j) for j in range(ndim)])
    _coord_idx = " + ".join(["ic_{0}".format(j) for j in range(ndim)])
    ops.append(
        """
        {float_type} val = ({float_type})x[{coord_idx}];
        out += val * ({weight});""".format(
            float_type=float_type, coord_idx=_coord_idx, weight=_weight
        )
    )
    ops.append("}" * ndim)
    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        ops.append("y = (out > -1) ? (Y)out : -(Y)(-out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    name = "cupy_ndimage_linear_interp_{}d_x{}".format(
        ndim, "_".join(["{}".format(j) for j in xshape])
    )
    return in_params, out_params, operation, name


def _get_linear_interp_kernel(
    xshape, coords_shape, double_precision, unsigned_output
):
    in_params, out_params, operation, name = _generate_linear_interp_kernel(
        xshape, coords_shape, double_precision, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)
