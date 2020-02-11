import functools
import operator

import cupy

from .support import (
    _nested_loops_init,
    _masked_loop_init,
    _pixelregion_to_buffer,
    _pixelmask_to_buffer,
)


def _generate_correlete_kernel(
    ndim, mode, cval, xshape, wshape, origin, unsigned_output
):
    """Generate a correlation kernel for dense filters.

    All positions within a filter of shape wshape are visited. This is done by
    nested loops over the filter axes.
    """

    in_params = "raw X x, raw W w"
    out_params = "Y y"

    ops = []
    ops.append("W sum = (W)0;")
    trim_unit_dims = False  # doesn't work correctly if set to True
    ops += _nested_loops_init(
        mode, xshape, wshape, origin, trim_unit_dims=trim_unit_dims
    )

    ops.append(
        """
        W wval = w[iw];
        if (wval == (W)0) {{
            iw += 1;
            continue;
        }}"""
    )
    # # for cond: only need to check bounds on axes where filter shape is > 1
    # _cond = " || ".join(
    #     ["(ix_{0} < 0)".format(j) for j in range(ndim) if wshape[j] > 1]
    # )
    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])
    ops.append(
        """
        if ({cond}) {{
            sum += (W){cval} * wval;
        }} else {{
            int ix = {expr};
            sum += (W)x[ix] * wval;
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


def _generate_correlete_kernel_masked(
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
    ops.append("W sum = (W)0;")

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _masked_loop_init(mode, xshape, fshape, origin, nnz)

    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    ops.append(
        """
        if ({cond}) {{
            sum += (W){cval} * wvals[iw];
        }} else {{
            int ix = {expr};
            sum += (W)x[ix] * wvals[iw];
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
    # any declarations outside the mask loop go here
    ops.append("double result, val;")
    ops.append("size_t mask_count = 0;")

    # declare the loop and intialize image indices, ix_0, etc.
    ops += _nested_loops_init(mode, xshape, wshape, origin)

    # Add code that is executed for each pixel in the footprint
    _cond = " || ".join(["(ix_{0} < 0)".format(j) for j in range(ndim)])
    _expr = " + ".join(["ix_{0}".format(j) for j in range(ndim)])

    if minimum:
        comp_op = "<"
    else:
        comp_op = ">"

    ops.append(
        """
        if (w[iw]) {{
            if ({cond}) {{
                val = (X){cval};
            }} else {{
                int ix = {expr};
                val = (X)x[ix];
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
            val = (X)x[ix];
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
                result = (X)x[ix];
            }}
            result += wvals[0];
            continue;
        }} else {{
            X _tmp;
            if ({cond}) {{
                _tmp = _cval;
            }} else {{
                int ix = {expr};
                _tmp = (X)x[ix];
            }}
            _tmp += wvals[iw];

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


rank_preamble = """
static __device__ X NI_Select(X *buffer, size_t min, size_t max, size_t rank)
{
    ptrdiff_t ii, jj;
    X x, t;

    if (min == max)
        return buffer[min];

    x = buffer[min];
    ii = min - 1;
    jj = max + 1;
    for(;;) {
        do
            jj--;
        while(buffer[jj] > x);
        do
            ii++;
        while(buffer[ii] < x);
        if (ii < jj) {
            t = buffer[ii];
            buffer[ii] = buffer[jj];
            buffer[jj] = t;
        } else {
            break;
        }
    }

    ii = jj - min + 1;
    if (rank < ii)
        return NI_Select(buffer, min, jj, rank);
    else
        return NI_Select(buffer, jj + 1, max, rank - ii);
}

/* Special cases below obtained from:
 * http://ndevilla.free.fr/median/median/index.html
 */

/*
 * The following routines have been built from knowledge gathered
 * around the Web. I am not aware of any copyright problem with
 * them, so use it as you want.
 * N. Devillard - 1998
 */

#define PIX_SORT(a,b) { if ((a)>(b)) PIX_SWAP((a),(b)); }
#define PIX_SWAP(a,b) { X temp=(a);(a)=(b);(b)=temp; }

/*----------------------------------------------------------------------------
   Function :   opt_med3()
   In       :   pointer to array of 3 pixel values
   Out      :   median value
   Job      :   optimized search of the median of 3 pixel values
   Notice   :   found on sci.image.processing
                cannot go faster unless assumptions are made
                on the nature of the input signal.
 ---------------------------------------------------------------------------*/

__device__ X opt_med3(X * p)
{
    PIX_SORT(p[0],p[1]) ; PIX_SORT(p[1],p[2]) ; PIX_SORT(p[0],p[1]) ;
    return(p[1]) ;
}

/*----------------------------------------------------------------------------
   Function :   opt_med5()
   In       :   pointer to array of 5 pixel values
   Out      :   median value
   Job      :   optimized search of the median of 5 pixel values
   Notice   :   found on sci.image.processing
                cannot go faster unless assumptions are made
                on the nature of the input signal.
 ---------------------------------------------------------------------------*/

__device__ X opt_med5(X * p)
{
    PIX_SORT(p[0],p[1]) ; PIX_SORT(p[3],p[4]) ; PIX_SORT(p[0],p[3]) ;
    PIX_SORT(p[1],p[4]) ; PIX_SORT(p[1],p[2]) ; PIX_SORT(p[2],p[3]) ;
    PIX_SORT(p[1],p[2]) ; return(p[2]) ;
}

/*----------------------------------------------------------------------------
   Function :   opt_med7()
   In       :   pointer to array of 7 pixel values
   Out      :   median value
   Job      :   optimized search of the median of 7 pixel values
   Notice   :   found on sci.image.processing
                cannot go faster unless assumptions are made
                on the nature of the input signal.
 ---------------------------------------------------------------------------*/

__device__ X opt_med7(X * p)
{
    PIX_SORT(p[0], p[5]) ; PIX_SORT(p[0], p[3]) ; PIX_SORT(p[1], p[6]) ;
    PIX_SORT(p[2], p[4]) ; PIX_SORT(p[0], p[1]) ; PIX_SORT(p[3], p[5]) ;
    PIX_SORT(p[2], p[6]) ; PIX_SORT(p[2], p[3]) ; PIX_SORT(p[3], p[6]) ;
    PIX_SORT(p[4], p[5]) ; PIX_SORT(p[1], p[4]) ; PIX_SORT(p[1], p[3]) ;
    PIX_SORT(p[3], p[4]) ; return (p[3]) ;
}

/*----------------------------------------------------------------------------
   Function :   opt_med9()
   In       :   pointer to an array of 9 Xs
   Out      :   median value
   Job      :   optimized search of the median of 9 Xs
   Notice   :   in theory, cannot go faster without assumptions on the
                signal.
                Formula from:
                XILINX XCELL magazine, vol. 23 by John L. Smith

                The input array is modified in the process
                The result array is guaranteed to contain the median
                value
                in middle position, but other elements are NOT sorted.
 ---------------------------------------------------------------------------*/

__device__ X opt_med9(X * p)
{
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ;
    PIX_SORT(p[0], p[1]) ; PIX_SORT(p[3], p[4]) ; PIX_SORT(p[6], p[7]) ;
    PIX_SORT(p[1], p[2]) ; PIX_SORT(p[4], p[5]) ; PIX_SORT(p[7], p[8]) ;
    PIX_SORT(p[0], p[3]) ; PIX_SORT(p[5], p[8]) ; PIX_SORT(p[4], p[7]) ;
    PIX_SORT(p[3], p[6]) ; PIX_SORT(p[1], p[4]) ; PIX_SORT(p[2], p[5]) ;
    PIX_SORT(p[4], p[7]) ; PIX_SORT(p[4], p[2]) ; PIX_SORT(p[6], p[4]) ;
    PIX_SORT(p[4], p[2]) ; return(p[4]) ;
}


/*----------------------------------------------------------------------------
   Function :   opt_med25()
   In       :   pointer to an array of 25 Xs
   Out      :   median value
   Job      :   optimized search of the median of 25 Xs
   Notice   :   in theory, cannot go faster without assumptions on the
                signal.
                Code taken from Graphic Gems.
 ---------------------------------------------------------------------------*/

__device__ X opt_med25(X * p)
{


    PIX_SORT(p[0], p[1]) ;   PIX_SORT(p[3], p[4]) ;   PIX_SORT(p[2], p[4]) ;
    PIX_SORT(p[2], p[3]) ;   PIX_SORT(p[6], p[7]) ;   PIX_SORT(p[5], p[7]) ;
    PIX_SORT(p[5], p[6]) ;   PIX_SORT(p[9], p[10]) ;  PIX_SORT(p[8], p[10]) ;
    PIX_SORT(p[8], p[9]) ;   PIX_SORT(p[12], p[13]) ; PIX_SORT(p[11], p[13]) ;
    PIX_SORT(p[11], p[12]) ; PIX_SORT(p[15], p[16]) ; PIX_SORT(p[14], p[16]) ;
    PIX_SORT(p[14], p[15]) ; PIX_SORT(p[18], p[19]) ; PIX_SORT(p[17], p[19]) ;
    PIX_SORT(p[17], p[18]) ; PIX_SORT(p[21], p[22]) ; PIX_SORT(p[20], p[22]) ;
    PIX_SORT(p[20], p[21]) ; PIX_SORT(p[23], p[24]) ; PIX_SORT(p[2], p[5]) ;
    PIX_SORT(p[3], p[6]) ;   PIX_SORT(p[0], p[6]) ;   PIX_SORT(p[0], p[3]) ;
    PIX_SORT(p[4], p[7]) ;   PIX_SORT(p[1], p[7]) ;   PIX_SORT(p[1], p[4]) ;
    PIX_SORT(p[11], p[14]) ; PIX_SORT(p[8], p[14]) ;  PIX_SORT(p[8], p[11]) ;
    PIX_SORT(p[12], p[15]) ; PIX_SORT(p[9], p[15]) ;  PIX_SORT(p[9], p[12]) ;
    PIX_SORT(p[13], p[16]) ; PIX_SORT(p[10], p[16]) ; PIX_SORT(p[10], p[13]) ;
    PIX_SORT(p[20], p[23]) ; PIX_SORT(p[17], p[23]) ; PIX_SORT(p[17], p[20]) ;
    PIX_SORT(p[21], p[24]) ; PIX_SORT(p[18], p[24]) ; PIX_SORT(p[18], p[21]) ;
    PIX_SORT(p[19], p[22]) ; PIX_SORT(p[8], p[17]) ;  PIX_SORT(p[9], p[18]) ;
    PIX_SORT(p[0], p[18]) ;  PIX_SORT(p[0], p[9]) ;   PIX_SORT(p[10], p[19]) ;
    PIX_SORT(p[1], p[19]) ;  PIX_SORT(p[1], p[10]) ;  PIX_SORT(p[11], p[20]) ;
    PIX_SORT(p[2], p[20]) ;  PIX_SORT(p[2], p[11]) ;  PIX_SORT(p[12], p[21]) ;
    PIX_SORT(p[3], p[21]) ;  PIX_SORT(p[3], p[12]) ;  PIX_SORT(p[13], p[22]) ;
    PIX_SORT(p[4], p[22]) ;  PIX_SORT(p[4], p[13]) ;  PIX_SORT(p[14], p[23]) ;
    PIX_SORT(p[5], p[23]) ;  PIX_SORT(p[5], p[14]) ;  PIX_SORT(p[15], p[24]) ;
    PIX_SORT(p[6], p[24]) ;  PIX_SORT(p[6], p[15]) ;  PIX_SORT(p[7], p[16]) ;
    PIX_SORT(p[7], p[19]) ;  PIX_SORT(p[13], p[21]) ; PIX_SORT(p[15], p[23]) ;
    PIX_SORT(p[7], p[13]) ;  PIX_SORT(p[7], p[15]) ;  PIX_SORT(p[1], p[9]) ;
    PIX_SORT(p[3], p[11]) ;  PIX_SORT(p[5], p[17]) ;  PIX_SORT(p[11], p[17]) ;
    PIX_SORT(p[9], p[17]) ;  PIX_SORT(p[4], p[10]) ;  PIX_SORT(p[6], p[12]) ;
    PIX_SORT(p[7], p[14]) ;  PIX_SORT(p[4], p[6]) ;   PIX_SORT(p[4], p[7]) ;
    PIX_SORT(p[12], p[14]) ; PIX_SORT(p[10], p[14]) ; PIX_SORT(p[6], p[7]) ;
    PIX_SORT(p[10], p[12]) ; PIX_SORT(p[6], p[10]) ;  PIX_SORT(p[6], p[17]) ;
    PIX_SORT(p[12], p[17]) ; PIX_SORT(p[7], p[17]) ;  PIX_SORT(p[7], p[10]) ;
    PIX_SORT(p[12], p[18]) ; PIX_SORT(p[7], p[12]) ;  PIX_SORT(p[10], p[18]) ;
    PIX_SORT(p[12], p[20]) ; PIX_SORT(p[10], p[20]) ; PIX_SORT(p[10], p[12]) ;

    return (p[12]);
}


"""


def _generate_rank_kernel(mode, cval, xshape, fshape, origin, rank):
    in_params = "raw X x, raw W w"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    # declare the loop and intialize image indices, ix_0, etc.
    nnz = functools.reduce(operator.mul, fshape)
    ops += _pixelregion_to_buffer(mode, cval, xshape, fshape, origin, nnz)

    # the above ops initialized a buffer containing the values within the
    # footprint. Now we have to sort these and return the value of the
    # requested rank.
    is_median = rank == nnz // 2
    if not (is_median and nnz in [3, 5, 7, 9, 25]):
        # Quickselect implementation from SciPy
        ops.append(
            "y = (Y)NI_Select(selected, 0, {nnz} - 1, {rank});".format(
                nnz=nnz, rank=rank
            )
        )
    else:
        if nnz == 3:
            ops.append("y = (Y)opt_med3(selected);")
        elif nnz == 5:
            ops.append("y = (Y)opt_med5(selected);")
        elif nnz == 7:
            ops.append("y = (Y)opt_med7(selected);")
        elif nnz == 9:
            ops.append("y = (Y)opt_med9(selected);")
        elif nnz == 25:
            ops.append("y = (Y)opt_med25(selected);")
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}".format(
        "median" if rank == nnz // 2 else "rank{}".format(rank),
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
    )
    return in_params, out_params, operation, name


def _generate_rank_kernel_masked(mode, cval, xshape, fshape, nnz, origin, rank):
    in_params = "raw X x, raw I wlocs"
    out_params = "Y y"

    ndim = len(fshape)

    ops = []
    # declare the loop and intialize image indices, ix_0, etc.
    ops += _pixelmask_to_buffer(mode, cval, xshape, fshape, origin, nnz)

    # the above ops initialized a buffer containing the values within the
    # footprint. Now we have to sort these and return the value of the
    # requested rank.
    is_median = rank == nnz // 2
    if not (is_median and nnz in [3, 5, 7, 9, 25]):
        # Quickselect implementation from SciPy
        ops.append(
            "y = (Y)NI_Select(selected, 0, {nnz} - 1, {rank});".format(
                nnz=nnz, rank=rank
            )
        )
    else:
        if nnz == 3:
            ops.append("y = (Y)opt_med3(selected);")
        elif nnz == 5:
            ops.append("y = (Y)opt_med5(selected);")
        elif nnz == 7:
            ops.append("y = (Y)opt_med7(selected);")
        elif nnz == 9:
            ops.append("y = (Y)opt_med9(selected);")
        elif nnz == 25:
            ops.append("y = (Y)opt_med25(selected);")
    operation = "\n".join(ops)

    name = "cupy_ndimage_{}_{}d_{}_x{}_w{}_nnz{}".format(
        "median" if rank == nnz // 2 else "rank{}".format(rank),
        ndim,
        mode,
        "_".join(["{}".format(j) for j in xshape]),
        "_".join(["{}".format(j) for j in fshape]),
        nnz,
    )
    return in_params, out_params, operation, name


@cupy.util.memoize()
def _get_correlete_kernel(
    ndim, mode, cval, xshape, fshape, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlete_kernel(
        ndim, mode, cval, xshape, fshape, origin, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @cupy.util.memoize()
def _get_correlete_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlete_kernel_masked(
        mode, cval, xshape, fshape, nnz, origin, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @cupy.util.memoize()
def _get_min_or_max_kernel(
    mode, cval, xshape, fshape, origin, minimum, unsigned_output
):
    in_params, out_params, operation, name = _generate_min_or_max_kernel(
        mode, cval, xshape, fshape, origin, minimum, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @cupy.util.memoize()
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


@cupy.util.memoize()
def _get_rank_kernel(mode, cval, xshape, fshape, origin, rank):
    in_params, out_params, operation, name = _generate_rank_kernel(
        mode, cval, xshape, fshape, origin, rank
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=rank_preamble
    )


def _get_rank_kernel_masked(mode, cval, xshape, fshape, nnz, origin, rank):
    in_params, out_params, operation, name = _generate_rank_kernel_masked(
        mode, cval, xshape, fshape, nnz, origin, rank
    )
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, preamble=rank_preamble
    )
