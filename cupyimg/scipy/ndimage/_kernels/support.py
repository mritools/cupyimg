"""Building blocks used by multiple ndimage kernels.

"""


def _generate_boundary_condition_ops(mode, ix, xsize):
    if mode == "reflect":
        ops = """
        if ({ix} < 0) {{
            {ix} = -1 - {ix};
        }}
        {ix} %= {xsize} * 2;
        {ix} = min({ix}, 2 * {xsize} - 1 - {ix});""".format(
            ix=ix, xsize=xsize
        )
    elif mode == "mirror":
        ops = """
        if ({ix} < 0) {{
            {ix} = -{ix};
        }}
        if ({xsize} == 1) {{
            {ix} = 0;
        }} else {{
            {ix} = 1 + ({ix} - 1) % (({xsize} - 1) * 2);
            {ix} = min({ix}, 2 * {xsize} - 2 - {ix});
        }}""".format(
            ix=ix, xsize=xsize
        )
    elif mode == "nearest":
        ops = """
        {ix} = min(max({ix}, 0), {xsize} - 1);""".format(
            ix=ix, xsize=xsize
        )
    elif mode == "wrap":
        ops = """
        if ({ix} < 0) {{
            {ix} += (1 - ({ix} / {xsize})) * {xsize};
        }}
        {ix} %= {xsize};""".format(
            ix=ix, xsize=xsize
        )
    elif mode == "constant":
        ops = """
        if ({ix} >= {xsize}) {{
            {ix} = -1;
        }}""".format(
            ix=ix, xsize=xsize
        )
    elif mode == "constant2":
        ops = """
        if (({ix} < 0) || ({ix} > ({xsize} - 1))) {{
            {ix} = -1;
        }}""".format(
            ix=ix, xsize=xsize
        )
    else:
        raise ValueError("unrecognized mode: {}".format(mode))
    return ops


def _get_init_loop_vars(xshape, fshape, origin):
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
        ops.append(
            "int cx_{j} = _i % {xsize} - ({fsize} / 2) - ({origin});".format(
                j=j, xsize=xshape[j], fsize=fshape[j], origin=origin[j]
            )
        )
        if j > 0:
            ops.append("_i /= {xsize};".format(xsize=xshape[j]))
    return ops


def _nested_loops_init(
    mode,
    xshape,
    fshape,
    origin,
    # compute_center_index=False, background_val=0,
    extra_pre_loop_ops=[],
    trim_unit_dims=False,
):
    ndim = len(xshape)
    ops = []
    ops += _get_init_loop_vars(xshape, fshape, origin)
    # if compute_center_index:
    #     ops += _compute_center_index_and_val(
    #         mode, xshape, fshape, background_val)

    ops.append("int iw = 0;")
    if extra_pre_loop_ops:
        ops += extra_pre_loop_ops

    for j in range(ndim):
        if trim_unit_dims and fshape[j] == 1:
            # no offsets in case of size one filter so no need for loop or
            # checking of the boundary conditions on this axis
            ops.append(
                """
        int ix_{j} = cx_{j} * sx_{j};""".format(
                    j=j
                )
            )
        else:
            ops.append(
                """
    for (int iw_{j} = 0; iw_{j} < {fsize}; iw_{j}++)
    {{
        int ix_{j} = cx_{j} + iw_{j};""".format(
                    j=j, fsize=fshape[j]
                )
            )
            ixvar = "ix_{}".format(j)
            ops.append(_generate_boundary_condition_ops(mode, ixvar, xshape[j]))
            ops.append("        ix_{j} *= sx_{j};".format(j=j))
    return ops


def _masked_loop_init(mode, xshape, fshape, origin, nnz):
    ndim = len(xshape)
    ops = []
    ops.append("const int sx_{} = 1;".format(ndim - 1))
    for j in range(ndim - 1, 0, -1):
        ops.append(
            "int sx_{jm} = sx_{j} * {xsize_j};".format(
                jm=j - 1, j=j, xsize_j=xshape[j]
            )
        )
    ops.append("int _i = i;")
    for j in range(ndim - 1, -1, -1):
        ops.append(
            "int cx_{j} = _i % {xsize} - ({fsize} / 2) - ({origin});".format(
                j=j, xsize=xshape[j], fsize=fshape[j], origin=origin[j]
            )
        )
        if j > 0:
            ops.append("_i /= {xsize};".format(xsize=xshape[j]))

    ops.append(
        """
        for (int iw = 0; iw < {nnz}; iw++)
        {{
        """.format(
            nnz=nnz
        )
    )
    for j in range(ndim):
        ops.append(
            """
                int iw_{j} = wlocs_data[iw + {j} * {nnz}];
                int ix_{j} = cx_{j} + iw_{j};""".format(
                j=j, nnz=nnz
            )
        )
        ixvar = "ix_{}".format(j)
        ops.append(_generate_boundary_condition_ops(mode, ixvar, xshape[j]))
        ops.append("        ix_{j} *= sx_{j};".format(j=j))
    return ops


def _pixelmask_to_buffer(mode, cval, xshape, fshape, origin, nnz):
    """Declares and initializes the contents of an array called "selected".

    selected will contain the pixel values falling within a local footprint.

    """
    ndim = len(fshape)

    ops = []
    ops.append("X selected[{nnz}];".format(nnz=nnz))
    # ops.append("""
    #     for (size_t idx=0; i<{nnz}; i++){{
    #         selected[idx] = 0.0;
    #     }}
    #     """.format(nnz=nnz))

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _masked_loop_init(mode, xshape, fshape, origin, nnz)

    # GRL: end of different middle section here

    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    ops.append(
        """
        if ({cond}) {{
            selected[iw] = (X){cval};
        }} else {{
            int ix = {expr};
            selected[iw] = (X)x_data[ix];
        }}
        """.format(
            cond=_cond, expr=_expr, cval=cval
        )
    )

    ops.append("}")

    return ops


def _pixelregion_to_buffer(mode, cval, xshape, fshape, origin, nnz):
    """Declares and initializes the contents of an array called "selected".

    selected will contain the pixel values falling within a local footprint.

    """
    ndim = len(fshape)

    ops = []
    ops.append("X selected[{nnz}];".format(nnz=nnz))
    # ops.append("""
    #     for (size_t idx=0; i<{nnz}; i++){{
    #         selected[idx] = 0.0;
    #     }}
    #     """.format(nnz=nnz))

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _nested_loops_init(mode, xshape, fshape, origin)

    # GRL: end of different middle section here

    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    ops.append(
        """
        if ({cond}) {{
            selected[iw] = (X){cval};
        }} else {{
            int ix = {expr};
            selected[iw] = (X)x_data[ix];
        }}
        iw += 1;
        """.format(
            cond=_cond, expr=_expr, cval=cval
        )
    )

    ops.append("}" * ndim)

    return ops


def _raw_ptr_ops(in_params):
    """Generate pointers to an array to use in place of CArray indexing.

    The ptr will have a name matching the input variable, but will have
    _data appended to the name.

    As a concrete example, `_raw_ptr_ops('raw X x, raw W w')` would return:

        ['X x_data = (X*)&(x[0]);', 'W w_data = (W*)&(w[0]);']

    """
    ops = []
    for in_p in in_params.split(','):
        in_p = in_p.strip()
        if in_p.startswith('raw '):
            _, typ, name = in_p.split(' ')
            ops.append(f'{typ}* {name}_data = ({typ}*)&({name}[0]);')
    return ops
