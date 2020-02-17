
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




def _generate_linear_interp_kernel(xshape, integer_output=False):
    """

    Note:
        The coords array will have shape (ndim, ncoords) where
        ndim == len(xshape).

        ncoords is determined by the size of the output array, y.
        y will be indexed by the CIndexer, _ind.
        Thus ncoords = _ind.size();

    """
    in_params = 'raw X x, raw W coords'
    out_params = "Y y"

    ndim = len(xshape)

    ops = []
    ops.append("double out = 0.0;")

    ops.append("ptrdiff_t ncoords = _ind.size();")

    # determine stride along each axis
    ops.append("const int sx_{} = 1;".format(ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            "int sx_{jm} = sx_{j} * {xsize_j};".format(
                jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )

    # for each coordinate, determine its floor and ceil and whether 1 or 2
    # values are needed for linear interpolation
    for j in range(ndim):
        ops.append("""
        W c_{j} = coords[i + {j} * ncoords];
        W cf_{j} = (W)floor((double)c_{j});
        W cc_{j} = cf_{j} + 1;
        int n_{j} = (c_{j} == cf_{j}) ? 1 : 2;  // 1 or 2 points needed?
        """.format(j=j))

    for j in range(ndim):
        ops.append("""
        for (int s_{j} = 0; s_{j} < n_{j}; s_{j}++)
            {{
                W w_{j};
                int ic_{j};
                if (s_{j} == 0)
                {{
                    w_{j} = cc_{j} - c_{j};
                    ic_{j} = cf_{j} * sx_{j};
                }} else
                {{
                    w_{j} = c_{j} - cf_{j};
                    ic_{j} = cc_{j} * sx_{j};
                }}""".format(j=j))

    _weight = " * ".join(["w_{j}".format(j=j) for j in range(ndim)])
    _coord_idx = " + ".join(["ic_{j}".format(j=j) for j in range(ndim)])
    ops.append("""
        X val = x[{coord_idx}];
        out += val * ({weight});""".format(coord_idx=_coord_idx, weight=_weight))
    ops.append("}" * ndim)

    if integer_output:
        ops.append("y = (Y)rint((double)out);")
    else:
        ops.append("y = (Y)out;")
    operation = "\n".join(ops)

    name = "linear_interpolate_{}d_x{}".format(
        ndim,
        "_".join(["{}".format(j) for j in xshape]),
    )
    return in_params, out_params, operation, name


# @util.memoize.memoize()
def _get_linear_interp_kernel(xshape, integer_output):
    # weights is always casted to float64 in order to get an output compatible
    # with SciPy, thought float32 might be sufficient when input dtype is low
    # precision.
    in_params, out_params, operation, name = _generate_linear_interp_kernel(
        xshape, integer_output)
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)

