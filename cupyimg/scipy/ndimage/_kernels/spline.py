def get_poles(order):
    if order == 2:
        # sqrt(8.0) - 3.0
        return (-0.171572875253809902396622551580603843,)
    elif order == 3:
        # sqrt(3.0) - 2.0
        return (-0.267949192431122706472553658494127633,)
    elif order == 4:
        # sqrt(664.0 - sqrt(438976.0)) + sqrt(304.0) - 19.0
        # sqrt(664.0 + sqrt(438976.0)) - sqrt(304.0) - 19.0
        return (
            -0.361341225900220177092212841325675255,
            -0.013725429297339121360331226939128204,
        )
    elif order == 5:
        # sqrt(67.5 - sqrt(4436.25)) + sqrt(26.25) - 6.5
        # sqrt(67.5 + sqrt(4436.25)) - sqrt(26.25) - 6.5
        return (
            -0.430575347099973791851434783493520110,
            -0.043096288203264653822712376822550182,
        )
    else:
        raise ValueError("only order 2-5 supported")


def get_gain(poles):
    from cupyimg._misc import _prod

    return _prod([(1.0 - z) * (1.0 - 1.0 / z) for z in poles])


def causal_init_op(mode):
    """c is a 1d array of length n and z is a filter pole"""
    if mode in ["nearest", "constant"]:
        mode = "mirror"
    op = """
        // causal init for mode={mode}""".format(
        mode=mode
    )
    if mode == "mirror":
        op += """
        z_i = z;
        z_n_1 = pow(z, ({dtype_pole})(n - 1));

        c[0] = c[0] + z_n_1 * c[n - 1];
        for (i = 1; i < n - 1; ++i) {{
            c[0] += z_i * (c[i] + z_n_1 * c[n - 1 - i]);
            z_i *= z;
        }}
        c[0] /= 1 - z_n_1 * z_n_1;"""
    elif mode == "wrap":
        op += """
        z_i = z;

        for (i = 1; i < n; ++i) {{
            c[0] += z_i * c[n - i];
            z_i *= z;
        }}
        c[0] /= 1 - z_i; /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        op += """
        z_i = z;
        z_n = pow(z, ({dtype_pole})n);
        c0 = c[0];

        c[0] = c[0] + z_n * c[n - 1];
        for (i = 1; i < n; ++i) {{
            c[0] += z_i * (c[i] + z_n * c[n - 1 - i]);
            z_i *= z;
        }}
        c[0] *= z / (1 - z_n * z_n);
        c[0] += c0;"""
    else:
        raise ValueError("invalid mode: {}".format(mode))
    return op


def anticausal_init_op(mode):
    """c is a 1d array of length n and z is a filter pole"""
    if mode in ["nearest", "constant"]:
        mode = "mirror"
    op = """
        // anti-causal init for mode={mode}""".format(
        mode=mode
    )
    if mode == "mirror":
        op += """
        c[n - 1] = (z * c[n - 2] + c[n - 1]) * z / (z * z - 1);"""
    elif mode == "wrap":
        op += """
        z_i = z;

        for (i = 0; i < n - 1; ++i) {{
            c[n - 1] += z_i * c[i];
            z_i *= z;
        }}
        c[n - 1] *= z / (z_i - 1); /* z_i = pow(z, n) */"""
    elif mode == "reflect":
        op += """
        c[n - 1] *= z / (z - 1);"""
    else:
        raise ValueError("invalid mode: {}".format(mode))
    return op


def get_spline1d_code(mode, poles):

    ops = [
        """
    #include <cupy/complex.cuh>

    __device__ void spline_prefilter1d(
        {dtype_data}* c, {dtype_index} signal_length
    )
    {{"""
    ]
    ops += get_apply_filter_ops(mode, poles)
    ops += [
        """
    }}"""
    ]
    return "\n".join(ops)


# TODO: This is currently only for 1D `c`.
#       In n-d, apply separately along each axis (signal_length = shape[axis])
#       In that case, the indexing needs to use the appropriate stride.
def get_apply_filter_ops(mode, poles):
    ops = []
    ops.append(
        """
        {dtype_index} i, n = signal_length;"""
    )
    ops.append(
        """
        {dtype_pole} z, z_i;"""
    )
    if mode == "mirror":
        ops.append(
            """
        {dtype_pole} z_n_1;"""
        )
    elif mode == "reflect":
        ops.append(
            """
        {dtype_pole} z_n;
        {dtype_data} c0;"""
        )
    for pole in poles:
        ops.append(
            """
        z = {pole};""".format(
                pole=pole
            )
        )
        ops.append(causal_init_op(mode))
        ops.append(
            """
        // causal filter
        for (i = 1; i < n; ++i) {{
            c[i] += z * c[i - 1];
        }}"""
        )
        ops.append(anticausal_init_op(mode))
        ops.append(
            """
        // anti-causal filter
        for (i = n - 2; i >= 0; --i) {{
            c[i] = z * (c[i + 1] - c[i]);
        }}"""
        )
    return ops


batch_spline1d_template = """

    extern "C" {{
    __global__ void batch_spline_prefilter(
        {dtype_data} *x,
        {dtype_index} len_x,
        {dtype_index} n_batch)
    {{
        {dtype_index} unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;
        {dtype_index} batch_idx = unraveled_idx;
        if (batch_idx < n_batch)
        {{
            {dtype_index} offset_x = batch_idx * len_x;  // offset to the current line
            spline_prefilter1d(&x[offset_x], len_x);
        }}
    }}
    }}
"""


def get_raw_spline1d_code(
    mode, order=3, dtype_index="int", dtype_data="double", dtype_pole="double"
):
    poles = get_poles(order)
    code = get_spline1d_code(mode, poles)
    code += batch_spline1d_template
    code = code.format(
        dtype_index=dtype_index, dtype_data=dtype_data, dtype_pole=dtype_pole
    )
    return code
