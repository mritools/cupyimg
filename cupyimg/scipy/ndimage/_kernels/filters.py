import functools
import operator

import cupy

from .support import (
    _nested_loops_init,
    _masked_loop_init,
    _pixelregion_to_buffer,
    _pixelmask_to_buffer,
    _raw_ptr_ops,
)
from cupyimg import memoize
from cupyimg.scipy.ndimage._kernels.optimal_median_preambles import (
    opt_med_preambles,
)


def _generate_correlate_kernel(
    ndim, mode, cval, xshape, wshape, origin, unsigned_output
):
    """Generate a correlation kernel for dense filters.

    All positions within a filter of shape wshape are visited. This is done by
    nested loops over the filter axes.
    """

    in_params = "raw X x, raw W w"
    out_params = "Y y"

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    ops.append("W sum = (W)0;")
    trim_unit_dims = True  # timing seems the same with either True or False
    ops += _nested_loops_init(
        mode, xshape, wshape, origin, trim_unit_dims=trim_unit_dims
    )

    ops.append(
        """
        W wval = w_data[iw];
        if (wval == (W)0) {{
            iw += 1;
            continue;
        }}"""
    )
    # for cond: only need to check bounds on axes where filter shape is > 1
    _cond = " || ".join(
        ["(ix_{0} < 0)".format(j) for j in range(ndim) if wshape[j] > 1]
    )
    if _cond == "":
        _cond = "0"

    # _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])
    ops.append(
        """
        if ({cond}) {{
            sum += (W){cval} * wval;
        }} else {{
            int ix = {expr};
            sum += (W)x_data[ix] * wval;
        }}
        iw += 1;""".format(
            cond=_cond, expr=_expr, cval=cval
        )
    )

    if trim_unit_dims:
        ops.append(
            "} " * functools.reduce(operator.add, [s > 1 for s in wshape])
        )
    else:
        ops.append("} " * ndim)
    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        ops.append("y = (sum > -1) ? (Y)sum : -(Y)(-sum);")
    else:
        ops.append("y = (Y)sum;")
    operation = "\n".join(ops)

    name = "cupy_ndimage_correlate_{}d_{}_x{}_w{}".format(
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in wshape]),
    )
    return in_params, out_params, operation, name


def _generate_correlate_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, unsigned_output
):
    """Generate a correlation kernel for sparse filters.

    Only nonzero positions within a filter of shape fshape are visited. This is
    done via a single loop indexed by a counter, iw, on the nonzero position
    within the filter.
    """
    in_params = "raw X x, raw I wlocs, raw W wvals"
    out_params = "Y y"

    ndim = len(fshape)

    # any declarations outside the mask loop go here
    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    ops.append("W sum = (W)0;")

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _masked_loop_init(mode, xshape, fshape, origin, nnz)

    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    ops.append(
        """
        if ({cond}) {{
            sum += (W){cval} * wvals_data[iw];
        }} else {{
            int ix = {expr};
            sum += (W)x_data[ix] * wvals_data[iw];
        }}
        """.format(
            cond=_cond, expr=_expr, cval=cval
        )
    )

    ops.append("}")
    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        ops.append("y = (sum > -1) ? (Y)sum : -(Y)(-sum);")
    else:
        ops.append("y = (Y)sum;")

    operation = "\n".join(ops)

    name = "cupy_ndimage_correlate_{}d_{}_x{}_w{}_nnz{}".format(
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
        nnz,
    )
    return in_params, out_params, operation, name


def _generate_min_or_max_kernel(
    mode, cval, xshape, wshape, origin, minimum, unsigned_output
):
    in_params = "raw X x, raw W w"
    out_params = "Y y"

    ndim = len(wshape)

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    # any declarations outside the mask loop go here
    ops.append("double result = 0, val;")
    ops.append("size_t mask_count = 0;")

    # declare the loop and intialize image indices, ix_0, etc.
    trim_unit_dims = True  # timing seems equivalent either way, so could remove
    ops += _nested_loops_init(
        mode, xshape, wshape, origin, trim_unit_dims=trim_unit_dims
    )

    # Add code that is executed for each pixel in the footprint
    # _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    # for cond: only need to check bounds on axes where filter shape is > 1
    _cond = " || ".join(
        ["(ix_{0} < 0)".format(j) for j in range(ndim) if wshape[j] > 1]
    )
    if _cond == "":
        _cond = "0"
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    if minimum:
        comp_op = "<"
    else:
        comp_op = ">"

    ops.append(
        """
        if (w_data[iw]) {{
            if ({cond}) {{
                val = (X){cval};
            }} else {{
                int ix = {expr};
                val = (X)x_data[ix];
            }}
            if ((mask_count == 0) || (val {comp_op} result)) {{
                result = val;
            }}
            mask_count += 1;
        }}
        iw += 1;
        """.format(
            cond=_cond, expr=_expr, cval=cval, comp_op=comp_op
        )
    )
    if trim_unit_dims:
        ops.append(
            "} " * functools.reduce(operator.add, [s > 1 for s in wshape])
        )
    else:
        ops.append("}" * ndim)  # end of loop over footprint

    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        # TODO: fix this

        # ops.append("char *_po; _po = (char *)&y; *(Y*)_po = (result > -1) ? (Y)result : -(Y)(-result);")
        ops.append("y = (result > -1) ? (Y)result : -(Y)(-result);")

    else:
        ops.append("y = (Y)result;")
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}".format(
        "minimum" if minimum else "maximum",
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in wshape]),
    )
    return in_params, out_params, operation, name


def _generate_min_or_max_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
):
    in_params = "raw X x, raw I wlocs"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    # any declarations outside the mask loop go here
    ops.append("double result, val;")
    ops.append("Y _po;")

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _masked_loop_init(mode, xshape, fshape, origin, nnz)

    # Add code that is executed for each pixel in the footprint
    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    if minimum:
        comp_op = "<"
    else:
        comp_op = ">"

    ops.append(
        """
        if ({cond}) {{
            val = (X){cval};
        }} else {{
            int ix = {expr};
            val = (X)x_data[ix];
        }}
        if ((iw == 0) || (val {comp_op} result)) {{
            result = val;
        }}
        """.format(
            cond=_cond, expr=_expr, cval=cval, comp_op=comp_op
        )
    )

    ops.append("}")  # end of loop over footprint
    if unsigned_output:  # if unsigned_out
        # Avoid undefined behaviour of float -> unsigned conversions
        # ops.append("y = (result > -1) ? (Y)result : -(Y)(-result);")
        ops.append("_po = (result) > -1. ? (Y)(result) : -(Y)(-result);")
        ops.append(
            "y = (Y)_po;"
        )  # (result > -1) ? (Y)result : (Y)(-(Y)(-result));")

    else:
        ops.append("y = (Y)result;")
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}_nnz{}".format(
        "minimum" if minimum else "maximum",
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
        nnz,
    )
    return in_params, out_params, operation, name


# v2 version support values in structuring element
def _generate_min_or_max_kernel_masked_v2(
    mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
):
    in_params = "raw X x, raw I wlocs, raw W wvals"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    # any declarations outside the mask loop go here
    ops.append("double result;")
    ops.append("Y _po;")
    ops.append("X _cval = (X){cval};".format(cval=float(cval)))

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _masked_loop_init(mode, xshape, fshape, origin, nnz)

    # Add code that is executed for each pixel in the footprint
    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    if minimum:
        comp_op = "<"
    else:
        comp_op = ">"

    # see ndimage's ni_filters.c: CASE_MIN_OR_MAX_POINT
    ops.append(
        """
        if (iw == 0) {{
            if ({cond}) {{
                result = _cval;
            }} else {{
                int ix = {expr};
                result = (X)x_data[ix];
            }}
            result += wvals_data[0];
            continue;
        }} else {{
            X _tmp;
            if ({cond}) {{
                _tmp = _cval;
            }} else {{
                int ix = {expr};
                _tmp = (X)x_data[ix];
            }}
            _tmp += wvals_data[iw];

            if ((_tmp {comp_op} result)) {{
                result = _tmp;
            }}
        }}
        """.format(
            cond=_cond, expr=_expr, comp_op=comp_op
        )
    )

    ops.append("}")  # end of loop over footprint
    if unsigned_output:
        # Avoid undefined behaviour of float -> unsigned conversions
        ops.append("_po = (result) > -1. ? (Y)(result) : -(Y)(-result);")
        ops.append(
            "y = (Y)_po;"
        )  # (result > -1) ? (Y)result : (Y)(-(Y)(-result));")
    #         ops.append("y = (result > -1) ? (Y)result : -(Y)(-result);")
    else:
        ops.append("y = (Y)result;")
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_struct_{}d_{}_x{}_w{}_nnz{}".format(
        "minimum" if minimum else "maximum",
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
        nnz,
    )
    return in_params, out_params, operation, name


"""
shell sort and selection sort code for the rank filters was written by
Jeffrey Bush.

obtained from:
https://gist.github.com/coderforlife/d953303da4bb7d8d28e49a568cb107b2
"""

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


__SELECTION_SORT = """
__device__ void sort(X *array, int size) {
    for (int i = 0; i < size; ++i) {
        int min_val = array[i];
        int min_idx = i;
        for (int j = i+1; j < size; ++j) {
            int val_j = array[j];
            if (val_j < min_val) {
                min_idx = j;
                min_val = val_j;
            }
        }
        if (i != min_idx) {
            array[min_idx] = array[i];
            array[i] = min_val;
        }
    }
}"""


@memoize()
def _get_shell_gap(filter_size):
    gap = 1
    while gap < filter_size:
        gap = 3 * gap + 1
    return gap


def _generate_rank_kernel(mode, cval, xshape, fshape, origin, rank):
    in_params = "raw X x, raw W w"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    # declare the loop and intialize image indices, ix_0, etc.
    nnz = functools.reduce(operator.mul, fshape)
    ops += _pixelregion_to_buffer(mode, cval, xshape, fshape, origin, nnz)

    # the above ops initialized a buffer containing the values within the
    # footprint. Now we have to sort these and return the value of the
    # requested rank.
    is_median = rank == nnz // 2
    if is_median and nnz in opt_med_preambles:
        # special fast sorting cases for some common kernel sizes
        preamble = opt_med_preambles[nnz]
        ops.append(
            """
            y = (Y)sort(selected);
        """
        )
    else:
        preamble = (
            __SELECTION_SORT
            if nnz <= 225
            else __SHELL_SORT.format(gap=_get_shell_gap(nnz))
        )
        ops.append(
            """
            sort(selected, {});
            y = (Y)selected[{}];
        """.format(
                nnz, rank
            )
        )
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}".format(
        "median" if rank == nnz // 2 else "rank{}".format(rank),
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
    )
    return in_params, out_params, operation, name, preamble


def _generate_rank_kernel_masked(mode, cval, xshape, fshape, nnz, origin, rank):
    in_params = "raw X x, raw I wlocs"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    ops = ops + _raw_ptr_ops(in_params)
    # declare the loop and intialize image indices, ix_0, etc.
    ops += _pixelmask_to_buffer(mode, cval, xshape, fshape, origin, nnz)

    # the above ops initialized a buffer containing the values within the
    # footprint. Now we have to sort these and return the value of the
    # requested rank.
    is_median = rank == nnz // 2
    if is_median and nnz in opt_med_preambles:
        # special fast sorting cases for some common kernel sizes
        preamble = opt_med_preambles[nnz]
        ops.append(
            """
            y = (Y)sort(selected);
        """
        )
    else:
        preamble = (
            __SELECTION_SORT
            if nnz <= 225
            else __SHELL_SORT.format(gap=_get_shell_gap(nnz))
        )
        ops.append(
            """
            sort(selected, {});
            y = (Y)selected[{}];
        """.format(
                nnz, rank
            )
        )
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}_nnz{}".format(
        "median" if rank == nnz // 2 else "rank{}".format(rank),
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
        nnz,
    )
    return in_params, out_params, operation, name, preamble


@memoize()
def _get_correlate_kernel(
    ndim, mode, cval, xshape, fshape, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlate_kernel(
        ndim, mode, cval, xshape, fshape, origin, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @memoize()
def _get_correlate_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlate_kernel_masked(
        mode, cval, xshape, fshape, nnz, origin, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @memoize()
def _get_min_or_max_kernel(
    mode, cval, xshape, fshape, origin, minimum, unsigned_output
):
    in_params, out_params, operation, name = _generate_min_or_max_kernel(
        mode, cval, xshape, fshape, origin, minimum, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @memoize()
def _get_min_or_max_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
):
    in_params, out_params, operation, name = _generate_min_or_max_kernel_masked(
        mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# v2 kernel takes additional input argument containing the structuring element
# values.
def _get_min_or_max_kernel_masked_v2(
    mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
):
    (
        in_params,
        out_params,
        operation,
        name,
    ) = _generate_min_or_max_kernel_masked_v2(
        mode, cval, xshape, fshape, nnz, origin, minimum, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


@memoize()
def _get_rank_kernel(mode, cval, xshape, fshape, origin, rank):
    in_params, out_params, operation, name, preamble = _generate_rank_kernel(
        mode, cval, xshape, fshape, origin, rank
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=preamble
    )


def _get_rank_kernel_masked(mode, cval, xshape, fshape, nnz, origin, rank):
    (
        in_params,
        out_params,
        operation,
        name,
        preamble,
    ) = _generate_rank_kernel_masked(
        mode, cval, xshape, fshape, nnz, origin, rank
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=preamble
    )
