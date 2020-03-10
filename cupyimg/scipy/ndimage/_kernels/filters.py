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
    trim_unit_dims = False  # doesn't work correctly if set to True
    ops += _nested_loops_init(
        mode, xshape, wshape, origin, trim_unit_dims=trim_unit_dims
    )

    ops.append(
        """
        W wval = _w[iw];
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
            sum += (W)_x[ix] * wval;
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
            sum += (W){cval} * _wvals[iw];
        }} else {{
            int ix = {expr};
            sum += (W)_x[ix] * _wvals[iw];
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
        if (_w[iw]) {{
            if ({cond}) {{
                val = (X){cval};
            }} else {{
                int ix = {expr};
                val = (X)_x[ix];
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
            val = (X)_x[ix];
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
                result = (X)_x[ix];
            }}
            result += _wvals[0];
            continue;
        }} else {{
            X _tmp;
            if ({cond}) {{
                _tmp = _cval;
            }} else {{
                int ix = {expr};
                _tmp = (X)_x[ix];
            }}
            _tmp += _wvals[iw];

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
static __device__ X NI_Select(X *buffer, unsigned int min, unsigned int max, unsigned int rank)
{
    int ii, jj;
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

static __device__ X NI_Select_long(X *buffer, size_t min, size_t max, size_t rank)
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

/* Most pecial cases below obtained from:
 * http://ndevilla.free.fr/median/median/index.html
 *
 * Cases for length 27, 49, 81, 125 generated by Gregory R. Lee
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



/*----------------------------------------------------------------------------
   Function :   fast_med27()
   In       :   pointer to an array of 27 Xs
   Out      :   median value
   Job      :   (partial) sorting network for length 27
   Notice   :   This is not an "optimal" network. It was created by generating
                a full sorting network using Batcher's Merge-Exchange algorithm
                as implemented in the Perl Networksort library.
                Some nodes not needed for obtaining the median were
                removed. 127 pairwise comparisons are involved.
   Perl Script :
       use Algorithm::Networksort;
       my $inputs = 27;
       my $algorithm = "batcher";
       my $nw = nwsrt(inputs => $inputs, algorithm => $algorithm);
       print $nw->title(), "\n";
       print $nw, "\n\n",
    At each level nodes not involving the median or nodes leading to the median
    were trimmed.
 ---------------------------------------------------------------------------*/

__device__ X fast_med27(X * p)
{
    PIX_SORT(p[0], p[16]); PIX_SORT(p[1], p[17]); PIX_SORT(p[2], p[18]);
    PIX_SORT(p[3], p[19]); PIX_SORT(p[4], p[20]); PIX_SORT(p[5], p[21]);
    PIX_SORT(p[6], p[22]); PIX_SORT(p[7], p[23]); PIX_SORT(p[8], p[24]);
    PIX_SORT(p[9], p[25]); PIX_SORT(p[10], p[26]); PIX_SORT(p[0], p[8]);
    PIX_SORT(p[1], p[9]); PIX_SORT(p[2], p[10]); PIX_SORT(p[3], p[11]);
    PIX_SORT(p[4], p[12]); PIX_SORT(p[5], p[13]);PIX_SORT(p[6], p[14]);
    PIX_SORT(p[7], p[15]); PIX_SORT(p[16], p[24]); PIX_SORT(p[17], p[25]);
    PIX_SORT(p[18], p[26]); PIX_SORT(p[8], p[16]); PIX_SORT(p[9], p[17]);
    PIX_SORT(p[10], p[18]); PIX_SORT(p[11], p[19]); PIX_SORT(p[12], p[20]);
    PIX_SORT(p[13], p[21]); PIX_SORT(p[14], p[22]); PIX_SORT(p[15], p[23]);
    PIX_SORT(p[0], p[4]); PIX_SORT(p[1], p[5]); PIX_SORT(p[2], p[6]);
    PIX_SORT(p[3], p[7]); PIX_SORT(p[8], p[12]); PIX_SORT(p[9], p[13]);
    PIX_SORT(p[10], p[14]); PIX_SORT(p[11], p[15]); PIX_SORT(p[16], p[20]);
    PIX_SORT(p[17], p[21]); PIX_SORT(p[18], p[22]); PIX_SORT(p[19], p[23]);
    PIX_SORT(p[0], p[2]); PIX_SORT(p[1], p[3]); PIX_SORT(p[4], p[16]);
    PIX_SORT(p[5], p[17]); PIX_SORT(p[6], p[18]); PIX_SORT(p[7], p[19]);
    PIX_SORT(p[12], p[24]); PIX_SORT(p[13], p[25]); PIX_SORT(p[14], p[26]);
    PIX_SORT(p[0], p[1]); PIX_SORT(p[4], p[8]); PIX_SORT(p[5], p[9]);
    PIX_SORT(p[6], p[10]); PIX_SORT(p[7], p[11]); PIX_SORT(p[12], p[16]);
    PIX_SORT(p[13], p[17]); PIX_SORT(p[14], p[18]); PIX_SORT(p[15], p[19]);
    PIX_SORT(p[20], p[24]); PIX_SORT(p[21], p[25]); PIX_SORT(p[22], p[26]);
    PIX_SORT(p[4], p[6]); PIX_SORT(p[5], p[7]); PIX_SORT(p[8], p[10]);
    PIX_SORT(p[9], p[11]); PIX_SORT(p[12], p[14]); PIX_SORT(p[13], p[15]);
    PIX_SORT(p[16], p[18]); PIX_SORT(p[17], p[19]); PIX_SORT(p[20], p[22]);
    PIX_SORT(p[21], p[23]); PIX_SORT(p[24], p[26]); PIX_SORT(p[2], p[16]);
    PIX_SORT(p[3], p[17]); PIX_SORT(p[6], p[20]); PIX_SORT(p[7], p[21]);
    PIX_SORT(p[10], p[24]); PIX_SORT(p[11], p[25]); PIX_SORT(p[2], p[8]);
    PIX_SORT(p[3], p[9]); PIX_SORT(p[6], p[12]); PIX_SORT(p[7], p[13]);
    PIX_SORT(p[10], p[16]); PIX_SORT(p[11], p[17]); PIX_SORT(p[14], p[20]);
    PIX_SORT(p[15], p[21]); PIX_SORT(p[18], p[24]); PIX_SORT(p[19], p[25]);
    PIX_SORT(p[2], p[4]); PIX_SORT(p[3], p[5]); PIX_SORT(p[6], p[8]);
    PIX_SORT(p[7], p[9]); PIX_SORT(p[10], p[12]); PIX_SORT(p[11], p[13]);
    PIX_SORT(p[14], p[16]); PIX_SORT(p[15], p[17]); PIX_SORT(p[18], p[20]);
    PIX_SORT(p[19], p[21]); PIX_SORT(p[22], p[24]); PIX_SORT(p[23], p[25]);
    PIX_SORT(p[2], p[3]); PIX_SORT(p[4], p[5]); PIX_SORT(p[6], p[7]);
    PIX_SORT(p[8], p[9]); PIX_SORT(p[10], p[11]); PIX_SORT(p[12], p[13]);
    PIX_SORT(p[14], p[15]); PIX_SORT(p[16], p[17]); PIX_SORT(p[18], p[19]);
    PIX_SORT(p[20], p[21]); PIX_SORT(p[22], p[23]); PIX_SORT(p[24], p[25]);
    PIX_SORT(p[1], p[16]); PIX_SORT(p[3], p[18]); PIX_SORT(p[5], p[20]);
    PIX_SORT(p[7], p[22]); PIX_SORT(p[9], p[24]); PIX_SORT(p[11], p[26]);
    PIX_SORT(p[7], p[14]); PIX_SORT(p[9], p[16]); PIX_SORT(p[11], p[18]);
    PIX_SORT(p[13], p[20]); PIX_SORT(p[11], p[14]); PIX_SORT(p[13], p[16]);
    PIX_SORT(p[13], p[14]);
    return p[13];
}


/*----------------------------------------------------------------------------
   Function :   fast_med49()
   In       :   pointer to an array of 49 Xs
   Out      :   median value
   Job      :   (partial) sorting network for length 49
   Notice   :   This is not an "optimal" network. It was created by generating
                a full sorting network using Batcher's Merge-Exchange algorithm
                as implemented in the Perl Networksort library.
                Some nodes not needed for obtaining the median were
                removed. 313 pairwise comparisons are involved.
   Perl Script :
       use Algorithm::Networksort;
       my $inputs = 49;
       my $algorithm = "batcher";
       my $nw = nwsrt(inputs => $inputs, algorithm => $algorithm);
       print $nw->title(), "\n";
       print $nw, "\n\n",
    At each level nodes not involving the median or nodes leading to the median
    were trimmed.
---------------------------------------------------------------------------*/

__device__ X fast_med49(X * p)
{
    PIX_SORT(p[0], p[32]); PIX_SORT(p[1], p[33]); PIX_SORT(p[2], p[34]);
    PIX_SORT(p[3], p[35]); PIX_SORT(p[4], p[36]); PIX_SORT(p[5], p[37]);
    PIX_SORT(p[6], p[38]); PIX_SORT(p[7], p[39]); PIX_SORT(p[8], p[40]);
    PIX_SORT(p[9], p[41]); PIX_SORT(p[10], p[42]); PIX_SORT(p[11], p[43]);
    PIX_SORT(p[12], p[44]); PIX_SORT(p[13], p[45]); PIX_SORT(p[14], p[46]);
    PIX_SORT(p[15], p[47]); PIX_SORT(p[16], p[48]); PIX_SORT(p[0], p[16]);
    PIX_SORT(p[1], p[17]); PIX_SORT(p[2], p[18]); PIX_SORT(p[3], p[19]);
    PIX_SORT(p[4], p[20]); PIX_SORT(p[5], p[21]); PIX_SORT(p[6], p[22]);
    PIX_SORT(p[7], p[23]); PIX_SORT(p[8], p[24]); PIX_SORT(p[9], p[25]);
    PIX_SORT(p[10], p[26]); PIX_SORT(p[11], p[27]); PIX_SORT(p[12], p[28]);
    PIX_SORT(p[13], p[29]); PIX_SORT(p[14], p[30]); PIX_SORT(p[15], p[31]);
    PIX_SORT(p[32], p[48]); PIX_SORT(p[16], p[32]); PIX_SORT(p[17], p[33]);
    PIX_SORT(p[18], p[34]); PIX_SORT(p[19], p[35]); PIX_SORT(p[20], p[36]);
    PIX_SORT(p[21], p[37]); PIX_SORT(p[22], p[38]); PIX_SORT(p[23], p[39]);
    PIX_SORT(p[24], p[40]); PIX_SORT(p[25], p[41]); PIX_SORT(p[26], p[42]);
    PIX_SORT(p[27], p[43]); PIX_SORT(p[28], p[44]); PIX_SORT(p[29], p[45]);
    PIX_SORT(p[30], p[46]); PIX_SORT(p[31], p[47]); PIX_SORT(p[0], p[8]);
    PIX_SORT(p[1], p[9]); PIX_SORT(p[2], p[10]); PIX_SORT(p[3], p[11]);
    PIX_SORT(p[4], p[12]); PIX_SORT(p[5], p[13]); PIX_SORT(p[6], p[14]);
    PIX_SORT(p[7], p[15]); PIX_SORT(p[16], p[24]); PIX_SORT(p[17], p[25]);
    PIX_SORT(p[18], p[26]); PIX_SORT(p[19], p[27]); PIX_SORT(p[20], p[28]);
    PIX_SORT(p[21], p[29]); PIX_SORT(p[22], p[30]); PIX_SORT(p[23], p[31]);
    PIX_SORT(p[32], p[40]); PIX_SORT(p[33], p[41]); PIX_SORT(p[34], p[42]);
    PIX_SORT(p[35], p[43]); PIX_SORT(p[36], p[44]); PIX_SORT(p[37], p[45]);
    PIX_SORT(p[38], p[46]); PIX_SORT(p[39], p[47]); PIX_SORT(p[0], p[4]);
    PIX_SORT(p[1], p[5]); PIX_SORT(p[2], p[6]); PIX_SORT(p[3], p[7]);
    PIX_SORT(p[8], p[32]); PIX_SORT(p[9], p[33]); PIX_SORT(p[10], p[34]);
    PIX_SORT(p[11], p[35]); PIX_SORT(p[12], p[36]); PIX_SORT(p[13], p[37]);
    PIX_SORT(p[14], p[38]); PIX_SORT(p[15], p[39]); PIX_SORT(p[24], p[48]);
    PIX_SORT(p[41], p[45]); PIX_SORT(p[42], p[46]); PIX_SORT(p[43], p[47]);
    PIX_SORT(p[0], p[2]); PIX_SORT(p[1], p[3]); PIX_SORT(p[8], p[16]);
    PIX_SORT(p[9], p[17]); PIX_SORT(p[10], p[18]); PIX_SORT(p[11], p[19]);
    PIX_SORT(p[12], p[20]); PIX_SORT(p[13], p[21]); PIX_SORT(p[14], p[22]);
    PIX_SORT(p[15], p[23]); PIX_SORT(p[24], p[32]); PIX_SORT(p[25], p[33]);
    PIX_SORT(p[26], p[34]); PIX_SORT(p[27], p[35]); PIX_SORT(p[28], p[36]);
    PIX_SORT(p[29], p[37]); PIX_SORT(p[30], p[38]); PIX_SORT(p[31], p[39]);
    PIX_SORT(p[40], p[48]); PIX_SORT(p[45], p[47]); PIX_SORT(p[0], p[1]);
    PIX_SORT(p[8], p[12]); PIX_SORT(p[9], p[13]); PIX_SORT(p[10], p[14]);
    PIX_SORT(p[11], p[15]); PIX_SORT(p[16], p[20]); PIX_SORT(p[17], p[21]);
    PIX_SORT(p[18], p[22]); PIX_SORT(p[19], p[23]); PIX_SORT(p[24], p[28]);
    PIX_SORT(p[25], p[29]); PIX_SORT(p[26], p[30]); PIX_SORT(p[27], p[31]);
    PIX_SORT(p[32], p[36]); PIX_SORT(p[33], p[37]); PIX_SORT(p[34], p[38]);
    PIX_SORT(p[35], p[39]); PIX_SORT(p[40], p[44]); PIX_SORT(p[4], p[32]);
    PIX_SORT(p[5], p[33]); PIX_SORT(p[6], p[34]); PIX_SORT(p[7], p[35]);
    PIX_SORT(p[12], p[40]); PIX_SORT(p[13], p[41]); PIX_SORT(p[14], p[42]);
    PIX_SORT(p[15], p[43]); PIX_SORT(p[20], p[48]); PIX_SORT(p[4], p[16]);
    PIX_SORT(p[5], p[17]); PIX_SORT(p[6], p[18]); PIX_SORT(p[7], p[19]);
    PIX_SORT(p[12], p[24]); PIX_SORT(p[13], p[25]); PIX_SORT(p[14], p[26]);
    PIX_SORT(p[15], p[27]); PIX_SORT(p[20], p[32]); PIX_SORT(p[21], p[33]);
    PIX_SORT(p[22], p[34]); PIX_SORT(p[23], p[35]); PIX_SORT(p[28], p[40]);
    PIX_SORT(p[29], p[41]); PIX_SORT(p[30], p[42]); PIX_SORT(p[31], p[43]);
    PIX_SORT(p[36], p[48]); PIX_SORT(p[4], p[8]); PIX_SORT(p[5], p[9]);
    PIX_SORT(p[6], p[10]); PIX_SORT(p[7], p[11]); PIX_SORT(p[12], p[16]);
    PIX_SORT(p[13], p[17]); PIX_SORT(p[14], p[18]); PIX_SORT(p[15], p[19]);
    PIX_SORT(p[20], p[24]); PIX_SORT(p[21], p[25]); PIX_SORT(p[22], p[26]);
    PIX_SORT(p[23], p[27]); PIX_SORT(p[28], p[32]); PIX_SORT(p[29], p[33]);
    PIX_SORT(p[30], p[34]); PIX_SORT(p[31], p[35]); PIX_SORT(p[36], p[40]);
    PIX_SORT(p[37], p[41]); PIX_SORT(p[38], p[42]); PIX_SORT(p[39], p[43]);
    PIX_SORT(p[44], p[48]); PIX_SORT(p[4], p[6]); PIX_SORT(p[5], p[7]);
    PIX_SORT(p[8], p[10]); PIX_SORT(p[9], p[11]); PIX_SORT(p[12], p[14]);
    PIX_SORT(p[13], p[15]); PIX_SORT(p[16], p[18]); PIX_SORT(p[17], p[19]);
    PIX_SORT(p[20], p[22]); PIX_SORT(p[21], p[23]); PIX_SORT(p[24], p[26]);
    PIX_SORT(p[25], p[27]); PIX_SORT(p[28], p[30]); PIX_SORT(p[29], p[31]);
    PIX_SORT(p[32], p[34]); PIX_SORT(p[33], p[35]); PIX_SORT(p[36], p[38]);
    PIX_SORT(p[37], p[39]); PIX_SORT(p[40], p[42]); PIX_SORT(p[41], p[43]);
    PIX_SORT(p[44], p[46]); PIX_SORT(p[2], p[32]); PIX_SORT(p[3], p[33]);
    PIX_SORT(p[6], p[36]); PIX_SORT(p[7], p[37]); PIX_SORT(p[10], p[40]);
    PIX_SORT(p[11], p[41]); PIX_SORT(p[14], p[44]); PIX_SORT(p[15], p[45]);
    PIX_SORT(p[18], p[48]); PIX_SORT(p[2], p[16]); PIX_SORT(p[3], p[17]);
    PIX_SORT(p[6], p[20]); PIX_SORT(p[7], p[21]); PIX_SORT(p[10], p[24]);
    PIX_SORT(p[11], p[25]); PIX_SORT(p[14], p[28]); PIX_SORT(p[15], p[29]);
    PIX_SORT(p[18], p[32]); PIX_SORT(p[19], p[33]); PIX_SORT(p[22], p[36]);
    PIX_SORT(p[23], p[37]); PIX_SORT(p[26], p[40]); PIX_SORT(p[27], p[41]);
    PIX_SORT(p[30], p[44]); PIX_SORT(p[31], p[45]); PIX_SORT(p[34], p[48]);
    PIX_SORT(p[2], p[8]); PIX_SORT(p[3], p[9]); PIX_SORT(p[6], p[12]);
    PIX_SORT(p[7], p[13]); PIX_SORT(p[10], p[16]); PIX_SORT(p[11], p[17]);
    PIX_SORT(p[14], p[20]); PIX_SORT(p[15], p[21]); PIX_SORT(p[18], p[24]);
    PIX_SORT(p[19], p[25]); PIX_SORT(p[22], p[28]); PIX_SORT(p[23], p[29]);
    PIX_SORT(p[26], p[32]); PIX_SORT(p[27], p[33]); PIX_SORT(p[30], p[36]);
    PIX_SORT(p[31], p[37]); PIX_SORT(p[34], p[40]); PIX_SORT(p[35], p[41]);
    PIX_SORT(p[38], p[44]); PIX_SORT(p[39], p[45]); PIX_SORT(p[42], p[48]);
    PIX_SORT(p[2], p[4]); PIX_SORT(p[3], p[5]); PIX_SORT(p[6], p[8]);
    PIX_SORT(p[7], p[9]); PIX_SORT(p[10], p[12]); PIX_SORT(p[11], p[13]);
    PIX_SORT(p[14], p[16]); PIX_SORT(p[15], p[17]); PIX_SORT(p[18], p[20]);
    PIX_SORT(p[19], p[21]); PIX_SORT(p[22], p[24]); PIX_SORT(p[23], p[25]);
    PIX_SORT(p[26], p[28]); PIX_SORT(p[27], p[29]); PIX_SORT(p[30], p[32]);
    PIX_SORT(p[31], p[33]); PIX_SORT(p[34], p[36]); PIX_SORT(p[35], p[37]);
    PIX_SORT(p[38], p[40]); PIX_SORT(p[39], p[41]); PIX_SORT(p[42], p[44]);
    PIX_SORT(p[43], p[45]); PIX_SORT(p[46], p[48]); PIX_SORT(p[2], p[3]);
    PIX_SORT(p[4], p[5]); PIX_SORT(p[6], p[7]); PIX_SORT(p[8], p[9]);
    PIX_SORT(p[10], p[11]); PIX_SORT(p[12], p[13]); PIX_SORT(p[14], p[15]);
    PIX_SORT(p[16], p[17]); PIX_SORT(p[18], p[19]); PIX_SORT(p[20], p[21]);
    PIX_SORT(p[22], p[23]); PIX_SORT(p[24], p[25]); PIX_SORT(p[26], p[27]);
    PIX_SORT(p[28], p[29]); PIX_SORT(p[30], p[31]); PIX_SORT(p[32], p[33]);
    PIX_SORT(p[34], p[35]); PIX_SORT(p[36], p[37]); PIX_SORT(p[38], p[39]);
    PIX_SORT(p[40], p[41]); PIX_SORT(p[42], p[43]); PIX_SORT(p[44], p[45]);
    PIX_SORT(p[46], p[47]); PIX_SORT(p[1], p[32]); PIX_SORT(p[3], p[34]);
    PIX_SORT(p[5], p[36]); PIX_SORT(p[7], p[38]); PIX_SORT(p[9], p[40]);
    PIX_SORT(p[11], p[42]); PIX_SORT(p[13], p[44]); PIX_SORT(p[15], p[46]);
    PIX_SORT(p[17], p[48]); PIX_SORT(p[9], p[24]); PIX_SORT(p[11], p[26]);
    PIX_SORT(p[13], p[28]); PIX_SORT(p[15], p[30]); PIX_SORT(p[17], p[32]);
    PIX_SORT(p[19], p[34]); PIX_SORT(p[21], p[36]); PIX_SORT(p[23], p[38]);
    PIX_SORT(p[17], p[24]); PIX_SORT(p[19], p[26]); PIX_SORT(p[21], p[28]);
    PIX_SORT(p[23], p[30]); PIX_SORT(p[21], p[24]); PIX_SORT(p[23], p[26]);
    PIX_SORT(p[23], p[24]);
    return p[24];
}


/*----------------------------------------------------------------------------
   Function :   fast_med81()
   In       :   pointer to an array of 81 Xs
   Out      :   median value
   Job      :   (partial) sorting network for length 81
   Notice   :   This is not an "optimal" network. It was created by generating
                a full sorting network using Batcher's Merge-Exchange algorithm
                as implemented in the Perl Networksort library.
                Some nodes not needed for obtaining the median were
                removed. 661 pairwise comparisons are involved.
   Perl Script :
       use Algorithm::Networksort;
       my $inputs = 81;
       my $algorithm = "batcher";
       my $nw = nwsrt(inputs => $inputs, algorithm => $algorithm);
       print $nw->title(), "\n";
       print $nw, "\n\n",
    At each level nodes not involving the median or nodes leading to the median
    were trimmed.
---------------------------------------------------------------------------*/

__device__ X fast_med81(X * p)
{
    PIX_SORT(p[0], p[64]); PIX_SORT(p[1], p[65]); PIX_SORT(p[2], p[66]);
    PIX_SORT(p[3], p[67]); PIX_SORT(p[4], p[68]); PIX_SORT(p[5], p[69]);
    PIX_SORT(p[6], p[70]); PIX_SORT(p[7], p[71]); PIX_SORT(p[8], p[72]);
    PIX_SORT(p[9], p[73]); PIX_SORT(p[10], p[74]); PIX_SORT(p[11], p[75]);
    PIX_SORT(p[12], p[76]); PIX_SORT(p[13], p[77]); PIX_SORT(p[14], p[78]);
    PIX_SORT(p[15], p[79]); PIX_SORT(p[16], p[80]); PIX_SORT(p[17], p[49]);
    PIX_SORT(p[18], p[50]); PIX_SORT(p[19], p[51]); PIX_SORT(p[20], p[52]);
    PIX_SORT(p[21], p[53]); PIX_SORT(p[22], p[54]); PIX_SORT(p[23], p[55]);
    PIX_SORT(p[24], p[56]); PIX_SORT(p[25], p[57]); PIX_SORT(p[26], p[58]);
    PIX_SORT(p[27], p[59]); PIX_SORT(p[28], p[60]); PIX_SORT(p[29], p[61]);
    PIX_SORT(p[30], p[62]); PIX_SORT(p[31], p[63]); PIX_SORT(p[0], p[32]);
    PIX_SORT(p[1], p[33]); PIX_SORT(p[2], p[34]); PIX_SORT(p[3], p[35]);
    PIX_SORT(p[4], p[36]); PIX_SORT(p[5], p[37]); PIX_SORT(p[6], p[38]);
    PIX_SORT(p[7], p[39]); PIX_SORT(p[8], p[40]); PIX_SORT(p[9], p[41]);
    PIX_SORT(p[10], p[42]); PIX_SORT(p[11], p[43]); PIX_SORT(p[12], p[44]);
    PIX_SORT(p[13], p[45]); PIX_SORT(p[14], p[46]); PIX_SORT(p[15], p[47]);
    PIX_SORT(p[16], p[48]); PIX_SORT(p[32], p[64]); PIX_SORT(p[33], p[65]);
    PIX_SORT(p[34], p[66]); PIX_SORT(p[35], p[67]); PIX_SORT(p[36], p[68]);
    PIX_SORT(p[37], p[69]); PIX_SORT(p[38], p[70]); PIX_SORT(p[39], p[71]);
    PIX_SORT(p[40], p[72]); PIX_SORT(p[41], p[73]); PIX_SORT(p[42], p[74]);
    PIX_SORT(p[43], p[75]); PIX_SORT(p[44], p[76]); PIX_SORT(p[45], p[77]);
    PIX_SORT(p[46], p[78]); PIX_SORT(p[47], p[79]); PIX_SORT(p[48], p[80]);
    PIX_SORT(p[0], p[16]); PIX_SORT(p[1], p[17]); PIX_SORT(p[2], p[18]);
    PIX_SORT(p[3], p[19]); PIX_SORT(p[4], p[20]); PIX_SORT(p[5], p[21]);
    PIX_SORT(p[6], p[22]); PIX_SORT(p[7], p[23]); PIX_SORT(p[8], p[24]);
    PIX_SORT(p[9], p[25]); PIX_SORT(p[10], p[26]); PIX_SORT(p[11], p[27]);
    PIX_SORT(p[12], p[28]); PIX_SORT(p[13], p[29]); PIX_SORT(p[14], p[30]);
    PIX_SORT(p[15], p[31]); PIX_SORT(p[32], p[48]); PIX_SORT(p[33], p[49]);
    PIX_SORT(p[34], p[50]); PIX_SORT(p[35], p[51]); PIX_SORT(p[36], p[52]);
    PIX_SORT(p[37], p[53]); PIX_SORT(p[38], p[54]); PIX_SORT(p[39], p[55]);
    PIX_SORT(p[40], p[56]); PIX_SORT(p[41], p[57]); PIX_SORT(p[42], p[58]);
    PIX_SORT(p[43], p[59]); PIX_SORT(p[44], p[60]); PIX_SORT(p[45], p[61]);
    PIX_SORT(p[46], p[62]); PIX_SORT(p[47], p[63]); PIX_SORT(p[64], p[80]);
    PIX_SORT(p[17], p[65]); PIX_SORT(p[18], p[66]); PIX_SORT(p[19], p[67]);
    PIX_SORT(p[20], p[68]); PIX_SORT(p[21], p[69]); PIX_SORT(p[22], p[70]);
    PIX_SORT(p[23], p[71]); PIX_SORT(p[24], p[72]); PIX_SORT(p[25], p[73]);
    PIX_SORT(p[26], p[74]); PIX_SORT(p[27], p[75]); PIX_SORT(p[28], p[76]);
    PIX_SORT(p[29], p[77]); PIX_SORT(p[30], p[78]); PIX_SORT(p[31], p[79]);
    PIX_SORT(p[0], p[8]); PIX_SORT(p[1], p[9]); PIX_SORT(p[2], p[10]);
    PIX_SORT(p[3], p[11]); PIX_SORT(p[4], p[12]); PIX_SORT(p[5], p[13]);
    PIX_SORT(p[6], p[14]); PIX_SORT(p[7], p[15]); PIX_SORT(p[16], p[64]);
    PIX_SORT(p[17], p[33]); PIX_SORT(p[18], p[34]); PIX_SORT(p[19], p[35]);
    PIX_SORT(p[20], p[36]); PIX_SORT(p[21], p[37]); PIX_SORT(p[22], p[38]);
    PIX_SORT(p[23], p[39]); PIX_SORT(p[24], p[40]); PIX_SORT(p[25], p[41]);
    PIX_SORT(p[26], p[42]); PIX_SORT(p[27], p[43]); PIX_SORT(p[28], p[44]);
    PIX_SORT(p[29], p[45]); PIX_SORT(p[30], p[46]); PIX_SORT(p[31], p[47]);
    PIX_SORT(p[49], p[65]); PIX_SORT(p[50], p[66]); PIX_SORT(p[51], p[67]);
    PIX_SORT(p[52], p[68]); PIX_SORT(p[53], p[69]); PIX_SORT(p[54], p[70]);
    PIX_SORT(p[55], p[71]); PIX_SORT(p[56], p[72]); PIX_SORT(p[57], p[73]);
    PIX_SORT(p[58], p[74]); PIX_SORT(p[59], p[75]); PIX_SORT(p[60], p[76]);
    PIX_SORT(p[61], p[77]); PIX_SORT(p[62], p[78]); PIX_SORT(p[63], p[79]);
    PIX_SORT(p[0], p[4]); PIX_SORT(p[1], p[5]); PIX_SORT(p[2], p[6]);
    PIX_SORT(p[3], p[7]); PIX_SORT(p[16], p[32]); PIX_SORT(p[48], p[64]);
    PIX_SORT(p[17], p[25]); PIX_SORT(p[18], p[26]); PIX_SORT(p[19], p[27]);
    PIX_SORT(p[20], p[28]); PIX_SORT(p[21], p[29]); PIX_SORT(p[22], p[30]);
    PIX_SORT(p[23], p[31]); PIX_SORT(p[33], p[41]); PIX_SORT(p[34], p[42]);
    PIX_SORT(p[35], p[43]); PIX_SORT(p[36], p[44]); PIX_SORT(p[37], p[45]);
    PIX_SORT(p[38], p[46]); PIX_SORT(p[39], p[47]); PIX_SORT(p[49], p[57]);
    PIX_SORT(p[50], p[58]); PIX_SORT(p[51], p[59]); PIX_SORT(p[52], p[60]);
    PIX_SORT(p[53], p[61]); PIX_SORT(p[54], p[62]); PIX_SORT(p[55], p[63]);
    PIX_SORT(p[65], p[73]); PIX_SORT(p[66], p[74]); PIX_SORT(p[67], p[75]);
    PIX_SORT(p[68], p[76]); PIX_SORT(p[69], p[77]); PIX_SORT(p[70], p[78]);
    PIX_SORT(p[71], p[79]); PIX_SORT(p[0], p[2]); PIX_SORT(p[1], p[3]);
    PIX_SORT(p[16], p[24]); PIX_SORT(p[32], p[40]); PIX_SORT(p[48], p[56]);
    PIX_SORT(p[64], p[72]); PIX_SORT(p[9], p[65]); PIX_SORT(p[10], p[66]);
    PIX_SORT(p[11], p[67]); PIX_SORT(p[12], p[68]); PIX_SORT(p[13], p[69]);
    PIX_SORT(p[14], p[70]); PIX_SORT(p[15], p[71]); PIX_SORT(p[25], p[49]);
    PIX_SORT(p[26], p[50]); PIX_SORT(p[27], p[51]); PIX_SORT(p[28], p[52]);
    PIX_SORT(p[29], p[53]); PIX_SORT(p[30], p[54]); PIX_SORT(p[31], p[55]);
    PIX_SORT(p[73], p[77]); PIX_SORT(p[74], p[78]); PIX_SORT(p[75], p[79]);
    PIX_SORT(p[0], p[1]); PIX_SORT(p[8], p[64]); PIX_SORT(p[24], p[80]);
    PIX_SORT(p[9], p[33]); PIX_SORT(p[10], p[34]); PIX_SORT(p[11], p[35]);
    PIX_SORT(p[12], p[36]); PIX_SORT(p[13], p[37]); PIX_SORT(p[14], p[38]);
    PIX_SORT(p[15], p[39]); PIX_SORT(p[41], p[65]); PIX_SORT(p[42], p[66]);
    PIX_SORT(p[43], p[67]); PIX_SORT(p[44], p[68]); PIX_SORT(p[45], p[69]);
    PIX_SORT(p[46], p[70]); PIX_SORT(p[47], p[71]); PIX_SORT(p[77], p[79]);
    PIX_SORT(p[8], p[32]); PIX_SORT(p[24], p[48]); PIX_SORT(p[40], p[64]);
    PIX_SORT(p[56], p[80]); PIX_SORT(p[9], p[17]); PIX_SORT(p[10], p[18]);
    PIX_SORT(p[11], p[19]); PIX_SORT(p[12], p[20]); PIX_SORT(p[13], p[21]);
    PIX_SORT(p[14], p[22]); PIX_SORT(p[15], p[23]); PIX_SORT(p[25], p[33]);
    PIX_SORT(p[26], p[34]); PIX_SORT(p[27], p[35]); PIX_SORT(p[28], p[36]);
    PIX_SORT(p[29], p[37]); PIX_SORT(p[30], p[38]); PIX_SORT(p[31], p[39]);
    PIX_SORT(p[41], p[49]); PIX_SORT(p[42], p[50]); PIX_SORT(p[43], p[51]);
    PIX_SORT(p[44], p[52]); PIX_SORT(p[45], p[53]); PIX_SORT(p[46], p[54]);
    PIX_SORT(p[47], p[55]); PIX_SORT(p[57], p[65]); PIX_SORT(p[58], p[66]);
    PIX_SORT(p[59], p[67]); PIX_SORT(p[60], p[68]); PIX_SORT(p[61], p[69]);
    PIX_SORT(p[62], p[70]); PIX_SORT(p[63], p[71]); PIX_SORT(p[8], p[16]);
    PIX_SORT(p[24], p[32]); PIX_SORT(p[40], p[48]); PIX_SORT(p[56], p[64]);
    PIX_SORT(p[72], p[80]); PIX_SORT(p[9], p[13]); PIX_SORT(p[10], p[14]);
    PIX_SORT(p[11], p[15]); PIX_SORT(p[17], p[21]); PIX_SORT(p[18], p[22]);
    PIX_SORT(p[19], p[23]); PIX_SORT(p[25], p[29]); PIX_SORT(p[26], p[30]);
    PIX_SORT(p[27], p[31]); PIX_SORT(p[33], p[37]); PIX_SORT(p[34], p[38]);
    PIX_SORT(p[35], p[39]); PIX_SORT(p[41], p[45]); PIX_SORT(p[42], p[46]);
    PIX_SORT(p[43], p[47]); PIX_SORT(p[49], p[53]); PIX_SORT(p[50], p[54]);
    PIX_SORT(p[51], p[55]); PIX_SORT(p[57], p[61]); PIX_SORT(p[58], p[62]);
    PIX_SORT(p[59], p[63]); PIX_SORT(p[65], p[69]); PIX_SORT(p[66], p[70]);
    PIX_SORT(p[67], p[71]); PIX_SORT(p[8], p[12]); PIX_SORT(p[16], p[20]);
    PIX_SORT(p[24], p[28]); PIX_SORT(p[32], p[36]); PIX_SORT(p[40], p[44]);
    PIX_SORT(p[48], p[52]); PIX_SORT(p[56], p[60]); PIX_SORT(p[64], p[68]);
    PIX_SORT(p[72], p[76]); PIX_SORT(p[5], p[65]); PIX_SORT(p[6], p[66]);
    PIX_SORT(p[7], p[67]); PIX_SORT(p[13], p[73]); PIX_SORT(p[14], p[74]);
    PIX_SORT(p[15], p[75]); PIX_SORT(p[21], p[49]); PIX_SORT(p[22], p[50]);
    PIX_SORT(p[23], p[51]); PIX_SORT(p[29], p[57]); PIX_SORT(p[30], p[58]);
    PIX_SORT(p[31], p[59]); PIX_SORT(p[4], p[64]); PIX_SORT(p[12], p[72]);
    PIX_SORT(p[20], p[80]); PIX_SORT(p[5], p[33]); PIX_SORT(p[6], p[34]);
    PIX_SORT(p[7], p[35]); PIX_SORT(p[13], p[41]); PIX_SORT(p[14], p[42]);
    PIX_SORT(p[15], p[43]); PIX_SORT(p[28], p[56]); PIX_SORT(p[37], p[65]);
    PIX_SORT(p[38], p[66]); PIX_SORT(p[39], p[67]); PIX_SORT(p[45], p[73]);
    PIX_SORT(p[46], p[74]); PIX_SORT(p[47], p[75]); PIX_SORT(p[4], p[32]);
    PIX_SORT(p[12], p[40]); PIX_SORT(p[20], p[48]); PIX_SORT(p[36], p[64]);
    PIX_SORT(p[44], p[72]); PIX_SORT(p[52], p[80]); PIX_SORT(p[5], p[17]);
    PIX_SORT(p[6], p[18]); PIX_SORT(p[7], p[19]); PIX_SORT(p[13], p[25]);
    PIX_SORT(p[14], p[26]); PIX_SORT(p[15], p[27]); PIX_SORT(p[21], p[33]);
    PIX_SORT(p[22], p[34]); PIX_SORT(p[23], p[35]); PIX_SORT(p[29], p[41]);
    PIX_SORT(p[30], p[42]); PIX_SORT(p[31], p[43]); PIX_SORT(p[37], p[49]);
    PIX_SORT(p[38], p[50]); PIX_SORT(p[39], p[51]); PIX_SORT(p[45], p[57]);
    PIX_SORT(p[46], p[58]); PIX_SORT(p[47], p[59]); PIX_SORT(p[53], p[65]);
    PIX_SORT(p[54], p[66]); PIX_SORT(p[55], p[67]); PIX_SORT(p[61], p[73]);
    PIX_SORT(p[62], p[74]); PIX_SORT(p[63], p[75]); PIX_SORT(p[4], p[16]);
    PIX_SORT(p[12], p[24]); PIX_SORT(p[20], p[32]); PIX_SORT(p[28], p[40]);
    PIX_SORT(p[36], p[48]); PIX_SORT(p[44], p[56]); PIX_SORT(p[52], p[64]);
    PIX_SORT(p[60], p[72]); PIX_SORT(p[68], p[80]); PIX_SORT(p[5], p[9]);
    PIX_SORT(p[6], p[10]); PIX_SORT(p[7], p[11]); PIX_SORT(p[13], p[17]);
    PIX_SORT(p[14], p[18]); PIX_SORT(p[15], p[19]); PIX_SORT(p[21], p[25]);
    PIX_SORT(p[22], p[26]); PIX_SORT(p[23], p[27]); PIX_SORT(p[29], p[33]);
    PIX_SORT(p[30], p[34]); PIX_SORT(p[31], p[35]); PIX_SORT(p[37], p[41]);
    PIX_SORT(p[38], p[42]); PIX_SORT(p[39], p[43]); PIX_SORT(p[45], p[49]);
    PIX_SORT(p[46], p[50]); PIX_SORT(p[47], p[51]); PIX_SORT(p[53], p[57]);
    PIX_SORT(p[54], p[58]); PIX_SORT(p[55], p[59]); PIX_SORT(p[61], p[65]);
    PIX_SORT(p[62], p[66]); PIX_SORT(p[63], p[67]); PIX_SORT(p[69], p[73]);
    PIX_SORT(p[70], p[74]); PIX_SORT(p[71], p[75]); PIX_SORT(p[4], p[8]);
    PIX_SORT(p[12], p[16]); PIX_SORT(p[20], p[24]); PIX_SORT(p[28], p[32]);
    PIX_SORT(p[36], p[40]); PIX_SORT(p[44], p[48]); PIX_SORT(p[52], p[56]);
    PIX_SORT(p[60], p[64]); PIX_SORT(p[68], p[72]); PIX_SORT(p[76], p[80]);
    PIX_SORT(p[5], p[7]); PIX_SORT(p[9], p[11]); PIX_SORT(p[13], p[15]);
    PIX_SORT(p[17], p[19]); PIX_SORT(p[21], p[23]); PIX_SORT(p[25], p[27]);
    PIX_SORT(p[29], p[31]); PIX_SORT(p[33], p[35]); PIX_SORT(p[37], p[39]);
    PIX_SORT(p[41], p[43]); PIX_SORT(p[45], p[47]); PIX_SORT(p[49], p[51]);
    PIX_SORT(p[53], p[55]); PIX_SORT(p[57], p[59]); PIX_SORT(p[61], p[63]);
    PIX_SORT(p[65], p[67]); PIX_SORT(p[69], p[71]); PIX_SORT(p[73], p[75]);
    PIX_SORT(p[4], p[6]); PIX_SORT(p[8], p[10]); PIX_SORT(p[12], p[14]);
    PIX_SORT(p[16], p[18]); PIX_SORT(p[20], p[22]); PIX_SORT(p[24], p[26]);
    PIX_SORT(p[28], p[30]); PIX_SORT(p[32], p[34]); PIX_SORT(p[36], p[38]);
    PIX_SORT(p[40], p[42]); PIX_SORT(p[44], p[46]); PIX_SORT(p[48], p[50]);
    PIX_SORT(p[52], p[54]); PIX_SORT(p[56], p[58]); PIX_SORT(p[60], p[62]);
    PIX_SORT(p[64], p[66]); PIX_SORT(p[68], p[70]); PIX_SORT(p[72], p[74]);
    PIX_SORT(p[76], p[78]); PIX_SORT(p[3], p[65]); PIX_SORT(p[7], p[69]);
    PIX_SORT(p[11], p[73]); PIX_SORT(p[15], p[77]); PIX_SORT(p[19], p[49]);
    PIX_SORT(p[23], p[53]); PIX_SORT(p[27], p[57]); PIX_SORT(p[31], p[61]);
    PIX_SORT(p[2], p[64]); PIX_SORT(p[6], p[68]); PIX_SORT(p[10], p[72]);
    PIX_SORT(p[14], p[76]); PIX_SORT(p[18], p[80]); PIX_SORT(p[3], p[33]);
    PIX_SORT(p[7], p[37]); PIX_SORT(p[11], p[41]); PIX_SORT(p[15], p[45]);
    PIX_SORT(p[22], p[52]); PIX_SORT(p[26], p[56]); PIX_SORT(p[30], p[60]);
    PIX_SORT(p[35], p[65]); PIX_SORT(p[39], p[69]); PIX_SORT(p[43], p[73]);
    PIX_SORT(p[47], p[77]); PIX_SORT(p[2], p[32]); PIX_SORT(p[6], p[36]);
    PIX_SORT(p[10], p[40]); PIX_SORT(p[14], p[44]); PIX_SORT(p[18], p[48]);
    PIX_SORT(p[34], p[64]); PIX_SORT(p[38], p[68]); PIX_SORT(p[42], p[72]);
    PIX_SORT(p[46], p[76]); PIX_SORT(p[50], p[80]); PIX_SORT(p[3], p[17]);
    PIX_SORT(p[7], p[21]); PIX_SORT(p[11], p[25]); PIX_SORT(p[15], p[29]);
    PIX_SORT(p[19], p[33]); PIX_SORT(p[23], p[37]); PIX_SORT(p[27], p[41]);
    PIX_SORT(p[31], p[45]); PIX_SORT(p[35], p[49]); PIX_SORT(p[39], p[53]);
    PIX_SORT(p[43], p[57]); PIX_SORT(p[47], p[61]); PIX_SORT(p[51], p[65]);
    PIX_SORT(p[55], p[69]); PIX_SORT(p[59], p[73]); PIX_SORT(p[63], p[77]);
    PIX_SORT(p[2], p[16]); PIX_SORT(p[6], p[20]); PIX_SORT(p[10], p[24]);
    PIX_SORT(p[14], p[28]); PIX_SORT(p[18], p[32]); PIX_SORT(p[22], p[36]);
    PIX_SORT(p[26], p[40]); PIX_SORT(p[30], p[44]); PIX_SORT(p[34], p[48]);
    PIX_SORT(p[38], p[52]); PIX_SORT(p[42], p[56]); PIX_SORT(p[46], p[60]);
    PIX_SORT(p[50], p[64]); PIX_SORT(p[54], p[68]); PIX_SORT(p[58], p[72]);
    PIX_SORT(p[62], p[76]); PIX_SORT(p[66], p[80]); PIX_SORT(p[3], p[9]);
    PIX_SORT(p[7], p[13]); PIX_SORT(p[11], p[17]); PIX_SORT(p[15], p[21]);
    PIX_SORT(p[19], p[25]); PIX_SORT(p[23], p[29]); PIX_SORT(p[27], p[33]);
    PIX_SORT(p[31], p[37]); PIX_SORT(p[35], p[41]); PIX_SORT(p[39], p[45]);
    PIX_SORT(p[43], p[49]); PIX_SORT(p[47], p[53]); PIX_SORT(p[51], p[57]);
    PIX_SORT(p[55], p[61]); PIX_SORT(p[59], p[65]); PIX_SORT(p[63], p[69]);
    PIX_SORT(p[67], p[73]); PIX_SORT(p[71], p[77]); PIX_SORT(p[2], p[8]);
    PIX_SORT(p[6], p[12]); PIX_SORT(p[10], p[16]); PIX_SORT(p[14], p[20]);
    PIX_SORT(p[18], p[24]); PIX_SORT(p[22], p[28]); PIX_SORT(p[26], p[32]);
    PIX_SORT(p[30], p[36]); PIX_SORT(p[34], p[40]); PIX_SORT(p[38], p[44]);
    PIX_SORT(p[42], p[48]); PIX_SORT(p[46], p[52]); PIX_SORT(p[50], p[56]);
    PIX_SORT(p[54], p[60]); PIX_SORT(p[58], p[64]); PIX_SORT(p[62], p[68]);
    PIX_SORT(p[66], p[72]); PIX_SORT(p[70], p[76]); PIX_SORT(p[74], p[80]);
    PIX_SORT(p[3], p[5]); PIX_SORT(p[7], p[9]); PIX_SORT(p[11], p[13]);
    PIX_SORT(p[15], p[17]); PIX_SORT(p[19], p[21]); PIX_SORT(p[23], p[25]);
    PIX_SORT(p[27], p[29]); PIX_SORT(p[31], p[33]); PIX_SORT(p[35], p[37]);
    PIX_SORT(p[39], p[41]); PIX_SORT(p[43], p[45]); PIX_SORT(p[47], p[49]);
    PIX_SORT(p[51], p[53]); PIX_SORT(p[55], p[57]); PIX_SORT(p[59], p[61]);
    PIX_SORT(p[63], p[65]); PIX_SORT(p[67], p[69]); PIX_SORT(p[71], p[73]);
    PIX_SORT(p[75], p[77]); PIX_SORT(p[2], p[4]); PIX_SORT(p[6], p[8]);
    PIX_SORT(p[10], p[12]); PIX_SORT(p[14], p[16]); PIX_SORT(p[18], p[20]);
    PIX_SORT(p[22], p[24]); PIX_SORT(p[26], p[28]); PIX_SORT(p[30], p[32]);
    PIX_SORT(p[34], p[36]); PIX_SORT(p[38], p[40]); PIX_SORT(p[42], p[44]);
    PIX_SORT(p[46], p[48]); PIX_SORT(p[50], p[52]); PIX_SORT(p[54], p[56]);
    PIX_SORT(p[58], p[60]); PIX_SORT(p[62], p[64]); PIX_SORT(p[66], p[68]);
    PIX_SORT(p[70], p[72]); PIX_SORT(p[74], p[76]); PIX_SORT(p[78], p[80]);
    PIX_SORT(p[2], p[3]); PIX_SORT(p[4], p[5]); PIX_SORT(p[6], p[7]);
    PIX_SORT(p[8], p[9]); PIX_SORT(p[10], p[11]); PIX_SORT(p[12], p[13]);
    PIX_SORT(p[14], p[15]); PIX_SORT(p[16], p[17]); PIX_SORT(p[18], p[19]);
    PIX_SORT(p[20], p[21]); PIX_SORT(p[22], p[23]); PIX_SORT(p[24], p[25]);
    PIX_SORT(p[26], p[27]); PIX_SORT(p[28], p[29]); PIX_SORT(p[30], p[31]);
    PIX_SORT(p[32], p[33]); PIX_SORT(p[34], p[35]); PIX_SORT(p[36], p[37]);
    PIX_SORT(p[38], p[39]); PIX_SORT(p[40], p[41]); PIX_SORT(p[42], p[43]);
    PIX_SORT(p[44], p[45]); PIX_SORT(p[46], p[47]); PIX_SORT(p[48], p[49]);
    PIX_SORT(p[50], p[51]); PIX_SORT(p[52], p[53]); PIX_SORT(p[54], p[55]);
    PIX_SORT(p[56], p[57]); PIX_SORT(p[58], p[59]); PIX_SORT(p[60], p[61]);
    PIX_SORT(p[62], p[63]); PIX_SORT(p[64], p[65]); PIX_SORT(p[66], p[67]);
    PIX_SORT(p[68], p[69]); PIX_SORT(p[70], p[71]); PIX_SORT(p[72], p[73]);
    PIX_SORT(p[74], p[75]); PIX_SORT(p[76], p[77]); PIX_SORT(p[78], p[79]);
    PIX_SORT(p[1], p[64]); PIX_SORT(p[3], p[66]); PIX_SORT(p[5], p[68]);
    PIX_SORT(p[7], p[70]); PIX_SORT(p[9], p[72]); PIX_SORT(p[11], p[74]);
    PIX_SORT(p[13], p[76]); PIX_SORT(p[15], p[78]); PIX_SORT(p[17], p[80]);
    PIX_SORT(p[19], p[50]); PIX_SORT(p[21], p[52]); PIX_SORT(p[23], p[54]);
    PIX_SORT(p[25], p[56]); PIX_SORT(p[27], p[58]); PIX_SORT(p[29], p[60]);
    PIX_SORT(p[31], p[62]); PIX_SORT(p[9], p[40]); PIX_SORT(p[11], p[42]);
    PIX_SORT(p[13], p[44]); PIX_SORT(p[15], p[46]); PIX_SORT(p[17], p[48]);
    PIX_SORT(p[33], p[64]); PIX_SORT(p[35], p[66]); PIX_SORT(p[37], p[68]);
    PIX_SORT(p[39], p[70]); PIX_SORT(p[25], p[40]); PIX_SORT(p[27], p[42]);
    PIX_SORT(p[29], p[44]); PIX_SORT(p[31], p[46]); PIX_SORT(p[33], p[48]);
    PIX_SORT(p[35], p[50]); PIX_SORT(p[37], p[52]); PIX_SORT(p[39], p[54]);
    PIX_SORT(p[33], p[40]); PIX_SORT(p[35], p[42]); PIX_SORT(p[37], p[44]);
    PIX_SORT(p[39], p[46]); PIX_SORT(p[37], p[40]); PIX_SORT(p[39], p[42]);
    PIX_SORT(p[39], p[40]);
    return p[40];
}

/*----------------------------------------------------------------------------
   Function :   fast_med125()
   In       :   pointer to an array of 125 Xs
   Out      :   median value
   Job      :   (partial) sorting network for length 125
   Notice   :   This is not an "optimal" network. It was created by generating
                a full sorting network using Batcher's Merge-Exchange algorithm
                as implemented in the Perl Networksort library.
                Some nodes not needed for obtaining the median were
                removed. 1188 pairwise comparisons are involved.
   Perl Script :
       use Algorithm::Networksort;
       my $inputs = 125;
       my $algorithm = "batcher";
       my $nw = nwsrt(inputs => $inputs, algorithm => $algorithm);
       print $nw->title(), "\n";
       print $nw, "\n\n",
 ---------------------------------------------------------------------------*/

__device__ X fast_med125(X * p)
{
    PIX_SORT(p[0], p[64]); PIX_SORT(p[1], p[65]); PIX_SORT(p[2], p[66]);
    PIX_SORT(p[3], p[67]); PIX_SORT(p[4], p[68]); PIX_SORT(p[5], p[69]);
    PIX_SORT(p[6], p[70]); PIX_SORT(p[7], p[71]); PIX_SORT(p[8], p[72]);
    PIX_SORT(p[9], p[73]); PIX_SORT(p[10], p[74]); PIX_SORT(p[11], p[75]);
    PIX_SORT(p[12], p[76]); PIX_SORT(p[13], p[77]); PIX_SORT(p[14], p[78]);
    PIX_SORT(p[15], p[79]); PIX_SORT(p[16], p[80]); PIX_SORT(p[17], p[81]);
    PIX_SORT(p[18], p[82]); PIX_SORT(p[19], p[83]); PIX_SORT(p[20], p[84]);
    PIX_SORT(p[21], p[85]); PIX_SORT(p[22], p[86]); PIX_SORT(p[23], p[87]);
    PIX_SORT(p[24], p[88]); PIX_SORT(p[25], p[89]); PIX_SORT(p[26], p[90]);
    PIX_SORT(p[27], p[91]); PIX_SORT(p[28], p[92]); PIX_SORT(p[29], p[93]);
    PIX_SORT(p[30], p[94]); PIX_SORT(p[31], p[95]); PIX_SORT(p[32], p[96]);
    PIX_SORT(p[33], p[97]); PIX_SORT(p[34], p[98]); PIX_SORT(p[35], p[99]);
    PIX_SORT(p[36], p[100]); PIX_SORT(p[37], p[101]); PIX_SORT(p[38], p[102]);
    PIX_SORT(p[39], p[103]); PIX_SORT(p[40], p[104]); PIX_SORT(p[41], p[105]);
    PIX_SORT(p[42], p[106]); PIX_SORT(p[43], p[107]); PIX_SORT(p[44], p[108]);
    PIX_SORT(p[45], p[109]); PIX_SORT(p[46], p[110]); PIX_SORT(p[47], p[111]);
    PIX_SORT(p[48], p[112]); PIX_SORT(p[49], p[113]); PIX_SORT(p[50], p[114]);
    PIX_SORT(p[51], p[115]); PIX_SORT(p[52], p[116]); PIX_SORT(p[53], p[117]);
    PIX_SORT(p[54], p[118]); PIX_SORT(p[55], p[119]); PIX_SORT(p[56], p[120]);
    PIX_SORT(p[57], p[121]); PIX_SORT(p[58], p[122]); PIX_SORT(p[59], p[123]);
    PIX_SORT(p[60], p[124]); PIX_SORT(p[0], p[32]); PIX_SORT(p[1], p[33]);
    PIX_SORT(p[2], p[34]); PIX_SORT(p[3], p[35]); PIX_SORT(p[4], p[36]);
    PIX_SORT(p[5], p[37]); PIX_SORT(p[6], p[38]); PIX_SORT(p[7], p[39]);
    PIX_SORT(p[8], p[40]); PIX_SORT(p[9], p[41]); PIX_SORT(p[10], p[42]);
    PIX_SORT(p[11], p[43]); PIX_SORT(p[12], p[44]); PIX_SORT(p[13], p[45]);
    PIX_SORT(p[14], p[46]); PIX_SORT(p[15], p[47]); PIX_SORT(p[16], p[48]);
    PIX_SORT(p[17], p[49]); PIX_SORT(p[18], p[50]); PIX_SORT(p[19], p[51]);
    PIX_SORT(p[20], p[52]); PIX_SORT(p[21], p[53]); PIX_SORT(p[22], p[54]);
    PIX_SORT(p[23], p[55]); PIX_SORT(p[24], p[56]); PIX_SORT(p[25], p[57]);
    PIX_SORT(p[26], p[58]); PIX_SORT(p[27], p[59]); PIX_SORT(p[28], p[60]);
    PIX_SORT(p[29], p[61]); PIX_SORT(p[30], p[62]); PIX_SORT(p[31], p[63]);
    PIX_SORT(p[64], p[96]); PIX_SORT(p[65], p[97]); PIX_SORT(p[66], p[98]);
    PIX_SORT(p[67], p[99]); PIX_SORT(p[68], p[100]); PIX_SORT(p[69], p[101]);
    PIX_SORT(p[70], p[102]); PIX_SORT(p[71], p[103]); PIX_SORT(p[72], p[104]);
    PIX_SORT(p[73], p[105]); PIX_SORT(p[74], p[106]); PIX_SORT(p[75], p[107]);
    PIX_SORT(p[76], p[108]); PIX_SORT(p[77], p[109]); PIX_SORT(p[78], p[110]);
    PIX_SORT(p[79], p[111]); PIX_SORT(p[80], p[112]); PIX_SORT(p[81], p[113]);
    PIX_SORT(p[82], p[114]); PIX_SORT(p[83], p[115]); PIX_SORT(p[84], p[116]);
    PIX_SORT(p[85], p[117]); PIX_SORT(p[86], p[118]); PIX_SORT(p[87], p[119]);
    PIX_SORT(p[88], p[120]); PIX_SORT(p[89], p[121]); PIX_SORT(p[90], p[122]);
    PIX_SORT(p[91], p[123]); PIX_SORT(p[92], p[124]); PIX_SORT(p[32], p[64]);
    PIX_SORT(p[33], p[65]); PIX_SORT(p[34], p[66]); PIX_SORT(p[35], p[67]);
    PIX_SORT(p[36], p[68]); PIX_SORT(p[37], p[69]); PIX_SORT(p[38], p[70]);
    PIX_SORT(p[39], p[71]); PIX_SORT(p[40], p[72]); PIX_SORT(p[41], p[73]);
    PIX_SORT(p[42], p[74]); PIX_SORT(p[43], p[75]); PIX_SORT(p[44], p[76]);
    PIX_SORT(p[45], p[77]); PIX_SORT(p[46], p[78]); PIX_SORT(p[47], p[79]);
    PIX_SORT(p[48], p[80]); PIX_SORT(p[49], p[81]); PIX_SORT(p[50], p[82]);
    PIX_SORT(p[51], p[83]); PIX_SORT(p[52], p[84]); PIX_SORT(p[53], p[85]);
    PIX_SORT(p[54], p[86]); PIX_SORT(p[55], p[87]); PIX_SORT(p[56], p[88]);
    PIX_SORT(p[57], p[89]); PIX_SORT(p[58], p[90]); PIX_SORT(p[59], p[91]);
    PIX_SORT(p[60], p[92]); PIX_SORT(p[61], p[93]); PIX_SORT(p[62], p[94]);
    PIX_SORT(p[63], p[95]); PIX_SORT(p[0], p[16]); PIX_SORT(p[1], p[17]);
    PIX_SORT(p[2], p[18]); PIX_SORT(p[3], p[19]); PIX_SORT(p[4], p[20]);
    PIX_SORT(p[5], p[21]); PIX_SORT(p[6], p[22]); PIX_SORT(p[7], p[23]);
    PIX_SORT(p[8], p[24]); PIX_SORT(p[9], p[25]); PIX_SORT(p[10], p[26]);
    PIX_SORT(p[11], p[27]); PIX_SORT(p[12], p[28]); PIX_SORT(p[13], p[29]);
    PIX_SORT(p[14], p[30]); PIX_SORT(p[15], p[31]); PIX_SORT(p[96], p[112]);
    PIX_SORT(p[97], p[113]); PIX_SORT(p[98], p[114]); PIX_SORT(p[99], p[115]);
    PIX_SORT(p[100], p[116]); PIX_SORT(p[101], p[117]); PIX_SORT(p[102], p[118]);
    PIX_SORT(p[103], p[119]); PIX_SORT(p[104], p[120]); PIX_SORT(p[105], p[121]);
    PIX_SORT(p[106], p[122]); PIX_SORT(p[107], p[123]); PIX_SORT(p[108], p[124]);
    PIX_SORT(p[32], p[48]); PIX_SORT(p[33], p[49]); PIX_SORT(p[34], p[50]);
    PIX_SORT(p[35], p[51]); PIX_SORT(p[36], p[52]); PIX_SORT(p[37], p[53]);
    PIX_SORT(p[38], p[54]); PIX_SORT(p[39], p[55]); PIX_SORT(p[40], p[56]);
    PIX_SORT(p[41], p[57]); PIX_SORT(p[42], p[58]); PIX_SORT(p[43], p[59]);
    PIX_SORT(p[44], p[60]); PIX_SORT(p[45], p[61]); PIX_SORT(p[46], p[62]);
    PIX_SORT(p[47], p[63]); PIX_SORT(p[64], p[80]); PIX_SORT(p[65], p[81]);
    PIX_SORT(p[66], p[82]); PIX_SORT(p[67], p[83]); PIX_SORT(p[68], p[84]);
    PIX_SORT(p[69], p[85]); PIX_SORT(p[70], p[86]); PIX_SORT(p[71], p[87]);
    PIX_SORT(p[72], p[88]); PIX_SORT(p[73], p[89]); PIX_SORT(p[74], p[90]);
    PIX_SORT(p[75], p[91]); PIX_SORT(p[76], p[92]); PIX_SORT(p[77], p[93]);
    PIX_SORT(p[78], p[94]); PIX_SORT(p[79], p[95]); PIX_SORT(p[0], p[8]);
    PIX_SORT(p[1], p[9]); PIX_SORT(p[2], p[10]); PIX_SORT(p[3], p[11]);
    PIX_SORT(p[4], p[12]); PIX_SORT(p[5], p[13]); PIX_SORT(p[6], p[14]);
    PIX_SORT(p[7], p[15]); PIX_SORT(p[112], p[120]); PIX_SORT(p[113], p[121]);
    PIX_SORT(p[114], p[122]); PIX_SORT(p[115], p[123]); PIX_SORT(p[116], p[124]);
    PIX_SORT(p[16], p[64]); PIX_SORT(p[17], p[65]); PIX_SORT(p[18], p[66]);
    PIX_SORT(p[19], p[67]); PIX_SORT(p[20], p[68]); PIX_SORT(p[21], p[69]);
    PIX_SORT(p[22], p[70]); PIX_SORT(p[23], p[71]); PIX_SORT(p[24], p[72]);
    PIX_SORT(p[25], p[73]); PIX_SORT(p[26], p[74]); PIX_SORT(p[27], p[75]);
    PIX_SORT(p[28], p[76]); PIX_SORT(p[29], p[77]); PIX_SORT(p[30], p[78]);
    PIX_SORT(p[31], p[79]); PIX_SORT(p[48], p[96]); PIX_SORT(p[49], p[97]);
    PIX_SORT(p[50], p[98]); PIX_SORT(p[51], p[99]); PIX_SORT(p[52], p[100]);
    PIX_SORT(p[53], p[101]); PIX_SORT(p[54], p[102]); PIX_SORT(p[55], p[103]);
    PIX_SORT(p[56], p[104]); PIX_SORT(p[57], p[105]); PIX_SORT(p[58], p[106]);
    PIX_SORT(p[59], p[107]); PIX_SORT(p[60], p[108]); PIX_SORT(p[61], p[109]);
    PIX_SORT(p[62], p[110]); PIX_SORT(p[63], p[111]); PIX_SORT(p[0], p[4]);
    PIX_SORT(p[1], p[5]); PIX_SORT(p[2], p[6]); PIX_SORT(p[3], p[7]);
    PIX_SORT(p[120], p[124]); PIX_SORT(p[16], p[32]); PIX_SORT(p[17], p[33]);
    PIX_SORT(p[18], p[34]); PIX_SORT(p[19], p[35]); PIX_SORT(p[20], p[36]);
    PIX_SORT(p[21], p[37]); PIX_SORT(p[22], p[38]); PIX_SORT(p[23], p[39]);
    PIX_SORT(p[24], p[40]); PIX_SORT(p[25], p[41]); PIX_SORT(p[26], p[42]);
    PIX_SORT(p[27], p[43]); PIX_SORT(p[28], p[44]); PIX_SORT(p[29], p[45]);
    PIX_SORT(p[30], p[46]); PIX_SORT(p[31], p[47]); PIX_SORT(p[48], p[64]);
    PIX_SORT(p[49], p[65]); PIX_SORT(p[50], p[66]); PIX_SORT(p[51], p[67]);
    PIX_SORT(p[52], p[68]); PIX_SORT(p[53], p[69]); PIX_SORT(p[54], p[70]);
    PIX_SORT(p[55], p[71]); PIX_SORT(p[56], p[72]); PIX_SORT(p[57], p[73]);
    PIX_SORT(p[58], p[74]); PIX_SORT(p[59], p[75]); PIX_SORT(p[60], p[76]);
    PIX_SORT(p[61], p[77]); PIX_SORT(p[62], p[78]); PIX_SORT(p[63], p[79]);
    PIX_SORT(p[80], p[96]); PIX_SORT(p[81], p[97]); PIX_SORT(p[82], p[98]);
    PIX_SORT(p[83], p[99]); PIX_SORT(p[84], p[100]); PIX_SORT(p[85], p[101]);
    PIX_SORT(p[86], p[102]); PIX_SORT(p[87], p[103]); PIX_SORT(p[88], p[104]);
    PIX_SORT(p[89], p[105]); PIX_SORT(p[90], p[106]); PIX_SORT(p[91], p[107]);
    PIX_SORT(p[92], p[108]); PIX_SORT(p[93], p[109]); PIX_SORT(p[94], p[110]);
    PIX_SORT(p[95], p[111]); PIX_SORT(p[0], p[2]); PIX_SORT(p[1], p[3]);
    PIX_SORT(p[16], p[24]); PIX_SORT(p[17], p[25]); PIX_SORT(p[18], p[26]);
    PIX_SORT(p[19], p[27]); PIX_SORT(p[20], p[28]); PIX_SORT(p[21], p[29]);
    PIX_SORT(p[22], p[30]); PIX_SORT(p[23], p[31]); PIX_SORT(p[32], p[40]);
    PIX_SORT(p[33], p[41]); PIX_SORT(p[34], p[42]); PIX_SORT(p[35], p[43]);
    PIX_SORT(p[36], p[44]); PIX_SORT(p[37], p[45]); PIX_SORT(p[38], p[46]);
    PIX_SORT(p[39], p[47]); PIX_SORT(p[48], p[56]); PIX_SORT(p[49], p[57]);
    PIX_SORT(p[50], p[58]); PIX_SORT(p[51], p[59]); PIX_SORT(p[52], p[60]);
    PIX_SORT(p[53], p[61]); PIX_SORT(p[54], p[62]); PIX_SORT(p[55], p[63]);
    PIX_SORT(p[64], p[72]); PIX_SORT(p[65], p[73]); PIX_SORT(p[66], p[74]);
    PIX_SORT(p[67], p[75]); PIX_SORT(p[68], p[76]); PIX_SORT(p[69], p[77]);
    PIX_SORT(p[70], p[78]); PIX_SORT(p[71], p[79]); PIX_SORT(p[80], p[88]);
    PIX_SORT(p[81], p[89]); PIX_SORT(p[82], p[90]); PIX_SORT(p[83], p[91]);
    PIX_SORT(p[84], p[92]); PIX_SORT(p[85], p[93]); PIX_SORT(p[86], p[94]);
    PIX_SORT(p[87], p[95]); PIX_SORT(p[96], p[104]); PIX_SORT(p[97], p[105]);
    PIX_SORT(p[98], p[106]); PIX_SORT(p[99], p[107]); PIX_SORT(p[100], p[108]);
    PIX_SORT(p[101], p[109]); PIX_SORT(p[102], p[110]); PIX_SORT(p[103], p[111]);
    PIX_SORT(p[0], p[1]); PIX_SORT(p[8], p[64]); PIX_SORT(p[9], p[65]);
    PIX_SORT(p[10], p[66]); PIX_SORT(p[11], p[67]); PIX_SORT(p[12], p[68]);
    PIX_SORT(p[13], p[69]); PIX_SORT(p[14], p[70]); PIX_SORT(p[15], p[71]);
    PIX_SORT(p[24], p[80]); PIX_SORT(p[25], p[81]); PIX_SORT(p[26], p[82]);
    PIX_SORT(p[27], p[83]); PIX_SORT(p[28], p[84]); PIX_SORT(p[29], p[85]);
    PIX_SORT(p[30], p[86]); PIX_SORT(p[31], p[87]); PIX_SORT(p[40], p[96]);
    PIX_SORT(p[41], p[97]); PIX_SORT(p[42], p[98]); PIX_SORT(p[43], p[99]);
    PIX_SORT(p[44], p[100]); PIX_SORT(p[45], p[101]); PIX_SORT(p[46], p[102]);
    PIX_SORT(p[47], p[103]); PIX_SORT(p[56], p[112]); PIX_SORT(p[57], p[113]);
    PIX_SORT(p[58], p[114]); PIX_SORT(p[59], p[115]); PIX_SORT(p[60], p[116]);
    PIX_SORT(p[61], p[117]); PIX_SORT(p[62], p[118]); PIX_SORT(p[63], p[119]);
    PIX_SORT(p[8], p[32]); PIX_SORT(p[9], p[33]); PIX_SORT(p[10], p[34]);
    PIX_SORT(p[11], p[35]); PIX_SORT(p[12], p[36]); PIX_SORT(p[13], p[37]);
    PIX_SORT(p[14], p[38]); PIX_SORT(p[15], p[39]); PIX_SORT(p[24], p[48]);
    PIX_SORT(p[25], p[49]); PIX_SORT(p[26], p[50]); PIX_SORT(p[27], p[51]);
    PIX_SORT(p[28], p[52]); PIX_SORT(p[29], p[53]); PIX_SORT(p[30], p[54]);
    PIX_SORT(p[31], p[55]); PIX_SORT(p[40], p[64]); PIX_SORT(p[41], p[65]);
    PIX_SORT(p[42], p[66]); PIX_SORT(p[43], p[67]); PIX_SORT(p[44], p[68]);
    PIX_SORT(p[45], p[69]); PIX_SORT(p[46], p[70]); PIX_SORT(p[47], p[71]);
    PIX_SORT(p[56], p[80]); PIX_SORT(p[57], p[81]); PIX_SORT(p[58], p[82]);
    PIX_SORT(p[59], p[83]); PIX_SORT(p[60], p[84]); PIX_SORT(p[61], p[85]);
    PIX_SORT(p[62], p[86]); PIX_SORT(p[63], p[87]); PIX_SORT(p[72], p[96]);
    PIX_SORT(p[73], p[97]); PIX_SORT(p[74], p[98]); PIX_SORT(p[75], p[99]);
    PIX_SORT(p[76], p[100]); PIX_SORT(p[77], p[101]); PIX_SORT(p[78], p[102]);
    PIX_SORT(p[79], p[103]); PIX_SORT(p[88], p[112]); PIX_SORT(p[89], p[113]);
    PIX_SORT(p[90], p[114]); PIX_SORT(p[91], p[115]); PIX_SORT(p[92], p[116]);
    PIX_SORT(p[93], p[117]); PIX_SORT(p[94], p[118]); PIX_SORT(p[95], p[119]);
    PIX_SORT(p[8], p[16]); PIX_SORT(p[9], p[17]); PIX_SORT(p[10], p[18]);
    PIX_SORT(p[11], p[19]); PIX_SORT(p[12], p[20]); PIX_SORT(p[13], p[21]);
    PIX_SORT(p[14], p[22]); PIX_SORT(p[15], p[23]); PIX_SORT(p[24], p[32]);
    PIX_SORT(p[25], p[33]); PIX_SORT(p[26], p[34]); PIX_SORT(p[27], p[35]);
    PIX_SORT(p[28], p[36]); PIX_SORT(p[29], p[37]); PIX_SORT(p[30], p[38]);
    PIX_SORT(p[31], p[39]); PIX_SORT(p[40], p[48]); PIX_SORT(p[41], p[49]);
    PIX_SORT(p[42], p[50]); PIX_SORT(p[43], p[51]); PIX_SORT(p[44], p[52]);
    PIX_SORT(p[45], p[53]); PIX_SORT(p[46], p[54]); PIX_SORT(p[47], p[55]);
    PIX_SORT(p[56], p[64]); PIX_SORT(p[57], p[65]); PIX_SORT(p[58], p[66]);
    PIX_SORT(p[59], p[67]); PIX_SORT(p[60], p[68]); PIX_SORT(p[61], p[69]);
    PIX_SORT(p[62], p[70]); PIX_SORT(p[63], p[71]); PIX_SORT(p[72], p[80]);
    PIX_SORT(p[73], p[81]); PIX_SORT(p[74], p[82]); PIX_SORT(p[75], p[83]);
    PIX_SORT(p[76], p[84]); PIX_SORT(p[77], p[85]); PIX_SORT(p[78], p[86]);
    PIX_SORT(p[79], p[87]); PIX_SORT(p[88], p[96]); PIX_SORT(p[89], p[97]);
    PIX_SORT(p[90], p[98]); PIX_SORT(p[91], p[99]); PIX_SORT(p[92], p[100]);
    PIX_SORT(p[93], p[101]); PIX_SORT(p[94], p[102]); PIX_SORT(p[95], p[103]);
    PIX_SORT(p[104], p[112]); PIX_SORT(p[105], p[113]); PIX_SORT(p[106], p[114]);
    PIX_SORT(p[107], p[115]); PIX_SORT(p[108], p[116]); PIX_SORT(p[109], p[117]);
    PIX_SORT(p[110], p[118]); PIX_SORT(p[111], p[119]); PIX_SORT(p[8], p[12]);
    PIX_SORT(p[9], p[13]); PIX_SORT(p[10], p[14]); PIX_SORT(p[11], p[15]);
    PIX_SORT(p[16], p[20]); PIX_SORT(p[17], p[21]); PIX_SORT(p[18], p[22]);
    PIX_SORT(p[19], p[23]); PIX_SORT(p[24], p[28]); PIX_SORT(p[25], p[29]);
    PIX_SORT(p[26], p[30]); PIX_SORT(p[27], p[31]); PIX_SORT(p[32], p[36]);
    PIX_SORT(p[33], p[37]); PIX_SORT(p[34], p[38]); PIX_SORT(p[35], p[39]);
    PIX_SORT(p[40], p[44]); PIX_SORT(p[41], p[45]); PIX_SORT(p[42], p[46]);
    PIX_SORT(p[43], p[47]); PIX_SORT(p[48], p[52]); PIX_SORT(p[49], p[53]);
    PIX_SORT(p[50], p[54]); PIX_SORT(p[51], p[55]); PIX_SORT(p[56], p[60]);
    PIX_SORT(p[57], p[61]); PIX_SORT(p[58], p[62]); PIX_SORT(p[59], p[63]);
    PIX_SORT(p[64], p[68]); PIX_SORT(p[65], p[69]); PIX_SORT(p[66], p[70]);
    PIX_SORT(p[67], p[71]); PIX_SORT(p[72], p[76]); PIX_SORT(p[73], p[77]);
    PIX_SORT(p[74], p[78]); PIX_SORT(p[75], p[79]); PIX_SORT(p[80], p[84]);
    PIX_SORT(p[81], p[85]); PIX_SORT(p[82], p[86]); PIX_SORT(p[83], p[87]);
    PIX_SORT(p[88], p[92]); PIX_SORT(p[89], p[93]); PIX_SORT(p[90], p[94]);
    PIX_SORT(p[91], p[95]); PIX_SORT(p[96], p[100]); PIX_SORT(p[97], p[101]);
    PIX_SORT(p[98], p[102]); PIX_SORT(p[99], p[103]); PIX_SORT(p[104], p[108]);
    PIX_SORT(p[105], p[109]); PIX_SORT(p[106], p[110]); PIX_SORT(p[107], p[111]);
    PIX_SORT(p[112], p[116]); PIX_SORT(p[113], p[117]); PIX_SORT(p[114], p[118]);
    PIX_SORT(p[115], p[119]); PIX_SORT(p[4], p[64]); PIX_SORT(p[5], p[65]);
    PIX_SORT(p[6], p[66]); PIX_SORT(p[7], p[67]); PIX_SORT(p[12], p[72]);
    PIX_SORT(p[13], p[73]); PIX_SORT(p[14], p[74]); PIX_SORT(p[15], p[75]);
    PIX_SORT(p[20], p[80]); PIX_SORT(p[21], p[81]); PIX_SORT(p[22], p[82]);
    PIX_SORT(p[23], p[83]); PIX_SORT(p[28], p[88]); PIX_SORT(p[29], p[89]);
    PIX_SORT(p[30], p[90]); PIX_SORT(p[31], p[91]); PIX_SORT(p[36], p[96]);
    PIX_SORT(p[37], p[97]); PIX_SORT(p[38], p[98]); PIX_SORT(p[39], p[99]);
    PIX_SORT(p[44], p[104]); PIX_SORT(p[45], p[105]); PIX_SORT(p[46], p[106]);
    PIX_SORT(p[47], p[107]); PIX_SORT(p[52], p[112]); PIX_SORT(p[53], p[113]);
    PIX_SORT(p[54], p[114]); PIX_SORT(p[55], p[115]); PIX_SORT(p[60], p[120]);
    PIX_SORT(p[61], p[121]); PIX_SORT(p[62], p[122]); PIX_SORT(p[63], p[123]);
    PIX_SORT(p[4], p[32]); PIX_SORT(p[5], p[33]); PIX_SORT(p[6], p[34]);
    PIX_SORT(p[7], p[35]); PIX_SORT(p[12], p[40]); PIX_SORT(p[13], p[41]);
    PIX_SORT(p[14], p[42]); PIX_SORT(p[15], p[43]); PIX_SORT(p[20], p[48]);
    PIX_SORT(p[21], p[49]); PIX_SORT(p[22], p[50]); PIX_SORT(p[23], p[51]);
    PIX_SORT(p[28], p[56]); PIX_SORT(p[29], p[57]); PIX_SORT(p[30], p[58]);
    PIX_SORT(p[31], p[59]); PIX_SORT(p[36], p[64]); PIX_SORT(p[37], p[65]);
    PIX_SORT(p[38], p[66]); PIX_SORT(p[39], p[67]); PIX_SORT(p[44], p[72]);
    PIX_SORT(p[45], p[73]); PIX_SORT(p[46], p[74]); PIX_SORT(p[47], p[75]);
    PIX_SORT(p[52], p[80]); PIX_SORT(p[53], p[81]); PIX_SORT(p[54], p[82]);
    PIX_SORT(p[55], p[83]); PIX_SORT(p[60], p[88]); PIX_SORT(p[61], p[89]);
    PIX_SORT(p[62], p[90]); PIX_SORT(p[63], p[91]); PIX_SORT(p[68], p[96]);
    PIX_SORT(p[69], p[97]); PIX_SORT(p[70], p[98]); PIX_SORT(p[71], p[99]);
    PIX_SORT(p[76], p[104]); PIX_SORT(p[77], p[105]); PIX_SORT(p[78], p[106]);
    PIX_SORT(p[79], p[107]); PIX_SORT(p[84], p[112]); PIX_SORT(p[85], p[113]);
    PIX_SORT(p[86], p[114]); PIX_SORT(p[87], p[115]); PIX_SORT(p[92], p[120]);
    PIX_SORT(p[93], p[121]); PIX_SORT(p[94], p[122]); PIX_SORT(p[95], p[123]);
    PIX_SORT(p[4], p[16]); PIX_SORT(p[5], p[17]); PIX_SORT(p[6], p[18]);
    PIX_SORT(p[7], p[19]); PIX_SORT(p[12], p[24]); PIX_SORT(p[13], p[25]);
    PIX_SORT(p[14], p[26]); PIX_SORT(p[15], p[27]); PIX_SORT(p[20], p[32]);
    PIX_SORT(p[21], p[33]); PIX_SORT(p[22], p[34]); PIX_SORT(p[23], p[35]);
    PIX_SORT(p[28], p[40]); PIX_SORT(p[29], p[41]); PIX_SORT(p[30], p[42]);
    PIX_SORT(p[31], p[43]); PIX_SORT(p[36], p[48]); PIX_SORT(p[37], p[49]);
    PIX_SORT(p[38], p[50]); PIX_SORT(p[39], p[51]); PIX_SORT(p[44], p[56]);
    PIX_SORT(p[45], p[57]); PIX_SORT(p[46], p[58]); PIX_SORT(p[47], p[59]);
    PIX_SORT(p[52], p[64]); PIX_SORT(p[53], p[65]); PIX_SORT(p[54], p[66]);
    PIX_SORT(p[55], p[67]); PIX_SORT(p[60], p[72]); PIX_SORT(p[61], p[73]);
    PIX_SORT(p[62], p[74]); PIX_SORT(p[63], p[75]); PIX_SORT(p[68], p[80]);
    PIX_SORT(p[69], p[81]); PIX_SORT(p[70], p[82]); PIX_SORT(p[71], p[83]);
    PIX_SORT(p[76], p[88]); PIX_SORT(p[77], p[89]); PIX_SORT(p[78], p[90]);
    PIX_SORT(p[79], p[91]); PIX_SORT(p[84], p[96]); PIX_SORT(p[85], p[97]);
    PIX_SORT(p[86], p[98]); PIX_SORT(p[87], p[99]); PIX_SORT(p[92], p[104]);
    PIX_SORT(p[93], p[105]); PIX_SORT(p[94], p[106]); PIX_SORT(p[95], p[107]);
    PIX_SORT(p[100], p[112]); PIX_SORT(p[101], p[113]); PIX_SORT(p[102], p[114]);
    PIX_SORT(p[103], p[115]); PIX_SORT(p[108], p[120]); PIX_SORT(p[109], p[121]);
    PIX_SORT(p[110], p[122]); PIX_SORT(p[111], p[123]); PIX_SORT(p[4], p[8]);
    PIX_SORT(p[5], p[9]); PIX_SORT(p[6], p[10]); PIX_SORT(p[7], p[11]);
    PIX_SORT(p[12], p[16]); PIX_SORT(p[13], p[17]); PIX_SORT(p[14], p[18]);
    PIX_SORT(p[15], p[19]); PIX_SORT(p[20], p[24]); PIX_SORT(p[21], p[25]);
    PIX_SORT(p[22], p[26]); PIX_SORT(p[23], p[27]); PIX_SORT(p[28], p[32]);
    PIX_SORT(p[29], p[33]); PIX_SORT(p[30], p[34]); PIX_SORT(p[31], p[35]);
    PIX_SORT(p[36], p[40]); PIX_SORT(p[37], p[41]); PIX_SORT(p[38], p[42]);
    PIX_SORT(p[39], p[43]); PIX_SORT(p[44], p[48]); PIX_SORT(p[45], p[49]);
    PIX_SORT(p[46], p[50]); PIX_SORT(p[47], p[51]); PIX_SORT(p[52], p[56]);
    PIX_SORT(p[53], p[57]); PIX_SORT(p[54], p[58]); PIX_SORT(p[55], p[59]);
    PIX_SORT(p[60], p[64]); PIX_SORT(p[61], p[65]); PIX_SORT(p[62], p[66]);
    PIX_SORT(p[63], p[67]); PIX_SORT(p[68], p[72]); PIX_SORT(p[69], p[73]);
    PIX_SORT(p[70], p[74]); PIX_SORT(p[71], p[75]); PIX_SORT(p[76], p[80]);
    PIX_SORT(p[77], p[81]); PIX_SORT(p[78], p[82]); PIX_SORT(p[79], p[83]);
    PIX_SORT(p[84], p[88]); PIX_SORT(p[85], p[89]); PIX_SORT(p[86], p[90]);
    PIX_SORT(p[87], p[91]); PIX_SORT(p[92], p[96]); PIX_SORT(p[93], p[97]);
    PIX_SORT(p[94], p[98]); PIX_SORT(p[95], p[99]); PIX_SORT(p[100], p[104]);
    PIX_SORT(p[101], p[105]); PIX_SORT(p[102], p[106]); PIX_SORT(p[103], p[107]);
    PIX_SORT(p[108], p[112]); PIX_SORT(p[109], p[113]); PIX_SORT(p[110], p[114]);
    PIX_SORT(p[111], p[115]); PIX_SORT(p[116], p[120]); PIX_SORT(p[117], p[121]);
    PIX_SORT(p[118], p[122]); PIX_SORT(p[119], p[123]); PIX_SORT(p[4], p[6]);
    PIX_SORT(p[5], p[7]); PIX_SORT(p[8], p[10]); PIX_SORT(p[9], p[11]);
    PIX_SORT(p[12], p[14]); PIX_SORT(p[13], p[15]); PIX_SORT(p[16], p[18]);
    PIX_SORT(p[17], p[19]); PIX_SORT(p[20], p[22]); PIX_SORT(p[21], p[23]);
    PIX_SORT(p[24], p[26]); PIX_SORT(p[25], p[27]); PIX_SORT(p[28], p[30]);
    PIX_SORT(p[29], p[31]); PIX_SORT(p[32], p[34]); PIX_SORT(p[33], p[35]);
    PIX_SORT(p[36], p[38]); PIX_SORT(p[37], p[39]); PIX_SORT(p[40], p[42]);
    PIX_SORT(p[41], p[43]); PIX_SORT(p[44], p[46]); PIX_SORT(p[45], p[47]);
    PIX_SORT(p[48], p[50]); PIX_SORT(p[49], p[51]); PIX_SORT(p[52], p[54]);
    PIX_SORT(p[53], p[55]); PIX_SORT(p[56], p[58]); PIX_SORT(p[57], p[59]);
    PIX_SORT(p[60], p[62]); PIX_SORT(p[61], p[63]); PIX_SORT(p[64], p[66]);
    PIX_SORT(p[65], p[67]); PIX_SORT(p[68], p[70]); PIX_SORT(p[69], p[71]);
    PIX_SORT(p[72], p[74]); PIX_SORT(p[73], p[75]); PIX_SORT(p[76], p[78]);
    PIX_SORT(p[77], p[79]); PIX_SORT(p[80], p[82]); PIX_SORT(p[81], p[83]);
    PIX_SORT(p[84], p[86]); PIX_SORT(p[85], p[87]); PIX_SORT(p[88], p[90]);
    PIX_SORT(p[89], p[91]); PIX_SORT(p[92], p[94]); PIX_SORT(p[93], p[95]);
    PIX_SORT(p[96], p[98]); PIX_SORT(p[97], p[99]); PIX_SORT(p[100], p[102]);
    PIX_SORT(p[101], p[103]); PIX_SORT(p[104], p[106]); PIX_SORT(p[105], p[107]);
    PIX_SORT(p[108], p[110]); PIX_SORT(p[109], p[111]); PIX_SORT(p[112], p[114]);
    PIX_SORT(p[113], p[115]); PIX_SORT(p[116], p[118]); PIX_SORT(p[117], p[119]);
    PIX_SORT(p[120], p[122]); PIX_SORT(p[121], p[123]); PIX_SORT(p[2], p[64]);
    PIX_SORT(p[3], p[65]); PIX_SORT(p[6], p[68]); PIX_SORT(p[7], p[69]);
    PIX_SORT(p[10], p[72]); PIX_SORT(p[11], p[73]); PIX_SORT(p[14], p[76]);
    PIX_SORT(p[15], p[77]); PIX_SORT(p[18], p[80]); PIX_SORT(p[19], p[81]);
    PIX_SORT(p[22], p[84]); PIX_SORT(p[23], p[85]); PIX_SORT(p[26], p[88]);
    PIX_SORT(p[27], p[89]); PIX_SORT(p[30], p[92]); PIX_SORT(p[31], p[93]);
    PIX_SORT(p[34], p[96]); PIX_SORT(p[35], p[97]); PIX_SORT(p[38], p[100]);
    PIX_SORT(p[39], p[101]); PIX_SORT(p[42], p[104]); PIX_SORT(p[43], p[105]);
    PIX_SORT(p[46], p[108]); PIX_SORT(p[47], p[109]); PIX_SORT(p[50], p[112]);
    PIX_SORT(p[51], p[113]); PIX_SORT(p[54], p[116]); PIX_SORT(p[55], p[117]);
    PIX_SORT(p[58], p[120]); PIX_SORT(p[59], p[121]); PIX_SORT(p[62], p[124]);
    PIX_SORT(p[2], p[32]); PIX_SORT(p[3], p[33]); PIX_SORT(p[6], p[36]);
    PIX_SORT(p[7], p[37]); PIX_SORT(p[10], p[40]); PIX_SORT(p[11], p[41]);
    PIX_SORT(p[14], p[44]); PIX_SORT(p[15], p[45]); PIX_SORT(p[18], p[48]);
    PIX_SORT(p[19], p[49]); PIX_SORT(p[22], p[52]); PIX_SORT(p[23], p[53]);
    PIX_SORT(p[26], p[56]); PIX_SORT(p[27], p[57]); PIX_SORT(p[30], p[60]);
    PIX_SORT(p[31], p[61]); PIX_SORT(p[34], p[64]); PIX_SORT(p[35], p[65]);
    PIX_SORT(p[38], p[68]); PIX_SORT(p[39], p[69]); PIX_SORT(p[42], p[72]);
    PIX_SORT(p[43], p[73]); PIX_SORT(p[46], p[76]); PIX_SORT(p[47], p[77]);
    PIX_SORT(p[50], p[80]); PIX_SORT(p[51], p[81]); PIX_SORT(p[54], p[84]);
    PIX_SORT(p[55], p[85]); PIX_SORT(p[58], p[88]); PIX_SORT(p[59], p[89]);
    PIX_SORT(p[62], p[92]); PIX_SORT(p[63], p[93]); PIX_SORT(p[66], p[96]);
    PIX_SORT(p[67], p[97]); PIX_SORT(p[70], p[100]); PIX_SORT(p[71], p[101]);
    PIX_SORT(p[74], p[104]); PIX_SORT(p[75], p[105]); PIX_SORT(p[78], p[108]);
    PIX_SORT(p[79], p[109]); PIX_SORT(p[82], p[112]); PIX_SORT(p[83], p[113]);
    PIX_SORT(p[86], p[116]); PIX_SORT(p[87], p[117]); PIX_SORT(p[90], p[120]);
    PIX_SORT(p[91], p[121]); PIX_SORT(p[94], p[124]); PIX_SORT(p[2], p[16]);
    PIX_SORT(p[3], p[17]); PIX_SORT(p[6], p[20]); PIX_SORT(p[7], p[21]);
    PIX_SORT(p[10], p[24]); PIX_SORT(p[11], p[25]); PIX_SORT(p[14], p[28]);
    PIX_SORT(p[15], p[29]); PIX_SORT(p[18], p[32]); PIX_SORT(p[19], p[33]);
    PIX_SORT(p[22], p[36]); PIX_SORT(p[23], p[37]); PIX_SORT(p[26], p[40]);
    PIX_SORT(p[27], p[41]); PIX_SORT(p[30], p[44]); PIX_SORT(p[31], p[45]);
    PIX_SORT(p[34], p[48]); PIX_SORT(p[35], p[49]); PIX_SORT(p[38], p[52]);
    PIX_SORT(p[39], p[53]); PIX_SORT(p[42], p[56]); PIX_SORT(p[43], p[57]);
    PIX_SORT(p[46], p[60]); PIX_SORT(p[47], p[61]); PIX_SORT(p[50], p[64]);
    PIX_SORT(p[51], p[65]); PIX_SORT(p[54], p[68]); PIX_SORT(p[55], p[69]);
    PIX_SORT(p[58], p[72]); PIX_SORT(p[59], p[73]); PIX_SORT(p[62], p[76]);
    PIX_SORT(p[63], p[77]); PIX_SORT(p[66], p[80]); PIX_SORT(p[67], p[81]);
    PIX_SORT(p[70], p[84]); PIX_SORT(p[71], p[85]); PIX_SORT(p[74], p[88]);
    PIX_SORT(p[75], p[89]); PIX_SORT(p[78], p[92]); PIX_SORT(p[79], p[93]);
    PIX_SORT(p[82], p[96]); PIX_SORT(p[83], p[97]); PIX_SORT(p[86], p[100]);
    PIX_SORT(p[87], p[101]); PIX_SORT(p[90], p[104]); PIX_SORT(p[91], p[105]);
    PIX_SORT(p[94], p[108]); PIX_SORT(p[95], p[109]); PIX_SORT(p[98], p[112]);
    PIX_SORT(p[99], p[113]); PIX_SORT(p[102], p[116]); PIX_SORT(p[103], p[117]);
    PIX_SORT(p[106], p[120]); PIX_SORT(p[107], p[121]); PIX_SORT(p[110], p[124]);
    PIX_SORT(p[2], p[8]); PIX_SORT(p[3], p[9]); PIX_SORT(p[6], p[12]);
    PIX_SORT(p[7], p[13]); PIX_SORT(p[10], p[16]); PIX_SORT(p[11], p[17]);
    PIX_SORT(p[14], p[20]); PIX_SORT(p[15], p[21]); PIX_SORT(p[18], p[24]);
    PIX_SORT(p[19], p[25]); PIX_SORT(p[22], p[28]); PIX_SORT(p[23], p[29]);
    PIX_SORT(p[26], p[32]); PIX_SORT(p[27], p[33]); PIX_SORT(p[30], p[36]);
    PIX_SORT(p[31], p[37]); PIX_SORT(p[34], p[40]); PIX_SORT(p[35], p[41]);
    PIX_SORT(p[38], p[44]); PIX_SORT(p[39], p[45]); PIX_SORT(p[42], p[48]);
    PIX_SORT(p[43], p[49]); PIX_SORT(p[46], p[52]); PIX_SORT(p[47], p[53]);
    PIX_SORT(p[50], p[56]); PIX_SORT(p[51], p[57]); PIX_SORT(p[54], p[60]);
    PIX_SORT(p[55], p[61]); PIX_SORT(p[58], p[64]); PIX_SORT(p[59], p[65]);
    PIX_SORT(p[62], p[68]); PIX_SORT(p[63], p[69]); PIX_SORT(p[66], p[72]);
    PIX_SORT(p[67], p[73]); PIX_SORT(p[70], p[76]); PIX_SORT(p[71], p[77]);
    PIX_SORT(p[74], p[80]); PIX_SORT(p[75], p[81]); PIX_SORT(p[78], p[84]);
    PIX_SORT(p[79], p[85]); PIX_SORT(p[82], p[88]); PIX_SORT(p[83], p[89]);
    PIX_SORT(p[86], p[92]); PIX_SORT(p[87], p[93]); PIX_SORT(p[90], p[96]);
    PIX_SORT(p[91], p[97]); PIX_SORT(p[94], p[100]); PIX_SORT(p[95], p[101]);
    PIX_SORT(p[98], p[104]); PIX_SORT(p[99], p[105]); PIX_SORT(p[102], p[108]);
    PIX_SORT(p[103], p[109]); PIX_SORT(p[106], p[112]); PIX_SORT(p[107], p[113]);
    PIX_SORT(p[110], p[116]); PIX_SORT(p[111], p[117]); PIX_SORT(p[114], p[120]);
    PIX_SORT(p[115], p[121]); PIX_SORT(p[118], p[124]); PIX_SORT(p[2], p[4]);
    PIX_SORT(p[3], p[5]); PIX_SORT(p[6], p[8]); PIX_SORT(p[7], p[9]);
    PIX_SORT(p[10], p[12]); PIX_SORT(p[11], p[13]); PIX_SORT(p[14], p[16]);
    PIX_SORT(p[15], p[17]); PIX_SORT(p[18], p[20]); PIX_SORT(p[19], p[21]);
    PIX_SORT(p[22], p[24]); PIX_SORT(p[23], p[25]); PIX_SORT(p[26], p[28]);
    PIX_SORT(p[27], p[29]); PIX_SORT(p[30], p[32]); PIX_SORT(p[31], p[33]);
    PIX_SORT(p[34], p[36]); PIX_SORT(p[35], p[37]); PIX_SORT(p[38], p[40]);
    PIX_SORT(p[39], p[41]); PIX_SORT(p[42], p[44]); PIX_SORT(p[43], p[45]);
    PIX_SORT(p[46], p[48]); PIX_SORT(p[47], p[49]); PIX_SORT(p[50], p[52]);
    PIX_SORT(p[51], p[53]); PIX_SORT(p[54], p[56]); PIX_SORT(p[55], p[57]);
    PIX_SORT(p[58], p[60]); PIX_SORT(p[59], p[61]); PIX_SORT(p[62], p[64]);
    PIX_SORT(p[63], p[65]); PIX_SORT(p[66], p[68]); PIX_SORT(p[67], p[69]);
    PIX_SORT(p[70], p[72]); PIX_SORT(p[71], p[73]); PIX_SORT(p[74], p[76]);
    PIX_SORT(p[75], p[77]); PIX_SORT(p[78], p[80]); PIX_SORT(p[79], p[81]);
    PIX_SORT(p[82], p[84]); PIX_SORT(p[83], p[85]); PIX_SORT(p[86], p[88]);
    PIX_SORT(p[87], p[89]); PIX_SORT(p[90], p[92]); PIX_SORT(p[91], p[93]);
    PIX_SORT(p[94], p[96]); PIX_SORT(p[95], p[97]); PIX_SORT(p[98], p[100]);
    PIX_SORT(p[99], p[101]); PIX_SORT(p[102], p[104]); PIX_SORT(p[103], p[105]);
    PIX_SORT(p[106], p[108]); PIX_SORT(p[107], p[109]); PIX_SORT(p[110], p[112]);
    PIX_SORT(p[111], p[113]); PIX_SORT(p[114], p[116]); PIX_SORT(p[115], p[117]);
    PIX_SORT(p[118], p[120]); PIX_SORT(p[119], p[121]); PIX_SORT(p[122], p[124]);
    PIX_SORT(p[2], p[3]); PIX_SORT(p[4], p[5]); PIX_SORT(p[6], p[7]);
    PIX_SORT(p[8], p[9]); PIX_SORT(p[10], p[11]); PIX_SORT(p[12], p[13]);
    PIX_SORT(p[14], p[15]); PIX_SORT(p[16], p[17]); PIX_SORT(p[18], p[19]);
    PIX_SORT(p[20], p[21]); PIX_SORT(p[22], p[23]); PIX_SORT(p[24], p[25]);
    PIX_SORT(p[26], p[27]); PIX_SORT(p[28], p[29]); PIX_SORT(p[30], p[31]);
    PIX_SORT(p[32], p[33]); PIX_SORT(p[34], p[35]); PIX_SORT(p[36], p[37]);
    PIX_SORT(p[38], p[39]); PIX_SORT(p[40], p[41]); PIX_SORT(p[42], p[43]);
    PIX_SORT(p[44], p[45]); PIX_SORT(p[46], p[47]); PIX_SORT(p[48], p[49]);
    PIX_SORT(p[50], p[51]); PIX_SORT(p[52], p[53]); PIX_SORT(p[54], p[55]);
    PIX_SORT(p[56], p[57]); PIX_SORT(p[58], p[59]); PIX_SORT(p[60], p[61]);
    PIX_SORT(p[62], p[63]); PIX_SORT(p[64], p[65]); PIX_SORT(p[66], p[67]);
    PIX_SORT(p[68], p[69]); PIX_SORT(p[70], p[71]); PIX_SORT(p[72], p[73]);
    PIX_SORT(p[74], p[75]); PIX_SORT(p[76], p[77]); PIX_SORT(p[78], p[79]);
    PIX_SORT(p[80], p[81]); PIX_SORT(p[82], p[83]); PIX_SORT(p[84], p[85]);
    PIX_SORT(p[86], p[87]); PIX_SORT(p[88], p[89]); PIX_SORT(p[90], p[91]);
    PIX_SORT(p[92], p[93]); PIX_SORT(p[94], p[95]); PIX_SORT(p[96], p[97]);
    PIX_SORT(p[98], p[99]); PIX_SORT(p[100], p[101]); PIX_SORT(p[102], p[103]);
    PIX_SORT(p[104], p[105]); PIX_SORT(p[106], p[107]); PIX_SORT(p[108], p[109]);
    PIX_SORT(p[110], p[111]); PIX_SORT(p[112], p[113]); PIX_SORT(p[114], p[115]);
    PIX_SORT(p[116], p[117]); PIX_SORT(p[118], p[119]); PIX_SORT(p[120], p[121]);
    PIX_SORT(p[122], p[123]); PIX_SORT(p[1], p[64]); PIX_SORT(p[3], p[66]);
    PIX_SORT(p[5], p[68]); PIX_SORT(p[7], p[70]); PIX_SORT(p[9], p[72]);
    PIX_SORT(p[11], p[74]); PIX_SORT(p[13], p[76]); PIX_SORT(p[15], p[78]);
    PIX_SORT(p[17], p[80]); PIX_SORT(p[19], p[82]); PIX_SORT(p[21], p[84]);
    PIX_SORT(p[23], p[86]); PIX_SORT(p[25], p[88]); PIX_SORT(p[27], p[90]);
    PIX_SORT(p[29], p[92]); PIX_SORT(p[31], p[94]); PIX_SORT(p[33], p[96]);
    PIX_SORT(p[35], p[98]); PIX_SORT(p[37], p[100]); PIX_SORT(p[39], p[102]);
    PIX_SORT(p[41], p[104]); PIX_SORT(p[43], p[106]); PIX_SORT(p[45], p[108]);
    PIX_SORT(p[47], p[110]); PIX_SORT(p[49], p[112]); PIX_SORT(p[51], p[114]);
    PIX_SORT(p[53], p[116]); PIX_SORT(p[55], p[118]); PIX_SORT(p[57], p[120]);
    PIX_SORT(p[59], p[122]); PIX_SORT(p[61], p[124]); PIX_SORT(p[31], p[62]);
    PIX_SORT(p[33], p[64]); PIX_SORT(p[35], p[66]); PIX_SORT(p[37], p[68]);
    PIX_SORT(p[39], p[70]); PIX_SORT(p[41], p[72]); PIX_SORT(p[43], p[74]);
    PIX_SORT(p[45], p[76]); PIX_SORT(p[47], p[78]); PIX_SORT(p[49], p[80]);
    PIX_SORT(p[51], p[82]); PIX_SORT(p[53], p[84]); PIX_SORT(p[55], p[86]);
    PIX_SORT(p[57], p[88]); PIX_SORT(p[59], p[90]); PIX_SORT(p[61], p[92]);
    PIX_SORT(p[47], p[62]); PIX_SORT(p[49], p[64]); PIX_SORT(p[51], p[66]);
    PIX_SORT(p[53], p[68]); PIX_SORT(p[55], p[70]); PIX_SORT(p[57], p[72]);
    PIX_SORT(p[59], p[74]); PIX_SORT(p[61], p[76]); PIX_SORT(p[55], p[62]);
    PIX_SORT(p[57], p[64]); PIX_SORT(p[59], p[66]); PIX_SORT(p[61], p[68]);
    PIX_SORT(p[59], p[62]); PIX_SORT(p[61], p[64]); PIX_SORT(p[61], p[62]);

    return p[62];
}

"""


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
    if not (is_median and nnz in [3, 5, 7, 9, 25, 27, 49, 81, 125]):
        # Quickselect implementation from SciPy
        if nnz >= 1 << 31:
            ops.append(
                "y = (Y)NI_Select_long(selected, 0, {nnz} - 1, {rank});".format(
                    nnz=nnz, rank=rank
                )
            )
        else:
            # Quickselect implementation from SciPy
            ops.append(
                "y = (Y)NI_Select(selected, 0, {nnz} - 1, {rank});".format(
                    nnz=nnz, rank=rank
                )
            )
    else:
        # These specialized median implementations tend to be MUCH faster than
        # using NI_Select.
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
        elif nnz == 27:
            ops.append("y = (Y)fast_med27(selected);")
        elif nnz == 49:
            ops.append("y = (Y)fast_med49(selected);")
        elif nnz == 81:
            ops.append("y = (Y)fast_med81(selected);")
        elif nnz == 125:
            ops.append("y = (Y)fast_med125(selected);")
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
    ops = ops + _raw_ptr_ops(in_params)
    # declare the loop and intialize image indices, ix_0, etc.
    ops += _pixelmask_to_buffer(mode, cval, xshape, fshape, origin, nnz)

    # the above ops initialized a buffer containing the values within the
    # footprint. Now we have to sort these and return the value of the
    # requested rank.
    is_median = rank == nnz // 2
    if not (is_median and nnz in [3, 5, 7, 9, 25, 27, 49, 81, 125]):
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
        elif nnz == 27:
            ops.append("y = (Y)fast_med27(selected);")
        elif nnz == 49:
            ops.append("y = (Y)fast_med49(selected);")
        elif nnz == 81:
            ops.append("y = (Y)fast_med81(selected);")
        elif nnz == 125:
            ops.append("y = (Y)fast_med125(selected);")
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
def _get_correlate_kernel(
    ndim, mode, cval, xshape, fshape, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlate_kernel(
        ndim, mode, cval, xshape, fshape, origin, unsigned_output
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation, name)


# @cupy.util.memoize()
def _get_correlate_kernel_masked(
    mode, cval, xshape, fshape, nnz, origin, unsigned_output
):
    in_params, out_params, operation, name = _generate_correlate_kernel_masked(
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
