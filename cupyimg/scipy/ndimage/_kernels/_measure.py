# TODO: implement kernel corresponding to CASE_FIND_OBJECT_POINT

# Note: May not be well suited to ElementwiseKernel because multiple threads
# will end up accessing the same locations in _regions, requiring atomic
# operations.

import cupy
from .interp import _unravel_loop_index


@cupy.util.memoize(for_each_device=True)
def _get_find_objects_operation(shape, max_label, int_t="int"):
    in_params = "raw int32 labels, int32 max_label"
    out_params = "raw int32 regions"
    ops = []
    ndim = len(shape)
    if int_t != "int":
        raise ValueError("currently only int is supported for int_t")
    uint_t = "unsigned int" if int_t == "int" else "size_t"
    ops.append(
        """
        {int_t} s_, cc, kk, ndim = {ndim};""".format(
            int_t=int_t, ndim=ndim
        )
    )
    ops.append(_unravel_loop_index(shape, uint_t))

    ops.append(
        """
        s_ = labels[i] - 1;
        if ((s_ >= 0) && (s_ < max_label))
        {
            // ndim = 0 is handled on the Python side
            s_ *= 2 * ndim;
            if (regions[s_] < 0) {
                for (kk=0; kk < ndim; kk++)
                {
                    cc = in_coord[kk];
                    //regions[s_ + kk] = cc;
                    //regions[s_ + kk + ndim] = cc + 1;
                    atomicExch(&regions[s_ + kk], cc);
                    atomicExch(&regions[s_ + kk + ndim], cc + 1);
                }
            } else {
                for (kk=0; kk < ndim; kk++)
                {
                    cc = in_coord[kk];
                    // if (cc < regions[s_ + kk]){
                    //     regions[s_ + kk] = cc;
                    // }
                    atomicMin(&regions[s_ + kk], cc);
                    // if (cc + 1 > regions[s_ + kk + ndim]){
                    //     regions[s_ + kk + ndim] = cc + 1;
                    // }
                    atomicMax(&regions[s_ + kk + ndim], cc + 1);
                }
            }
        }"""
    )
    ops = "\n".join(ops)
    return in_params, out_params, ops


# # @cupy.util.memoize(for_each_device=True)
# def _get_find_objects_operation(shape, max_label, int_t='int'):
#     in_params = "raw int32 labels, int32 max_label"
#     out_params = "raw int32 regions"
#     ops = []
#     ndim = len(shape)
#     if int_t != 'int':
#         raise ValueError("currently only int is supported for int_t")
#     uint_t = 'unsigned int' if int_t == 'int' else 'size_t'
#     ops.append("""
#         {int_t} s_, cc, kk, ndim = {ndim};""".format(int_t=int_t,
#                                                           ndim=ndim))
#     ops.append(_unravel_loop_index(shape, uint_t))

#     ops.append("""
#         s_ = labels[i] - 1;
#         if ((s_ >= 0) && (s_ < max_label))
#         {
#             // ndim = 0 is handled on the Python side
#             s_ *= 2 * ndim;
#             if (regions[s_] < 0) {
#                 for (kk=0; kk < ndim; kk++)
#                 {
#                     cc = in_coord[kk];
#                     //regions[s_ + kk] = cc;
#                     //regions[s_ + kk + ndim] = cc + 1;
#                     atomicCAS(&regions[s_ + kk], regions[s_ + kk], cc);
#                     atomicCAS(&regions[s_ + kk + ndim], regions[s_ + kk + ndim], cc + 1);
#                 }
#             } else {
#                 for (kk=0; kk < ndim; kk++)
#                 {
#                     cc = in_coord[kk];
#                     if (cc < regions[s_ + kk]){
#                         atomicCAS(&regions[s_ + kk], regions[s_ + kk], cc);
#                     }

#                     if (cc + 1 > regions[s_ + kk + ndim]){
#                         atomicCAS(&regions[s_ + kk + ndim], regions[s_ + kk + ndim], cc + 1);
#                     }

#                 }
#             }
#         }""")
#     ops = '\n'.join(ops)
#     return in_params, out_params, ops


def _get_find_objects_kernel(shape, max_label, int_t="int"):
    in_params, out_params, operation = _get_find_objects_operation(
        shape, max_label, int_t
    )
    return cupy.ElementwiseKernel(in_params, out_params, operation)


if False:
    import os
    import cupy
    import numpy as np

    os.environ["CUPY_DUMP_CUDA_SOURCE_ON_ERROR"] = "1"
    a = np.zeros((6, 6), dtype=int)
    a[2:4, 2:4] = 1
    a[4, 4] = 1
    a[:2, :3] = 2
    a[0, 5] = 4
    a = cupy.asarray(a).astype(np.int32)
    max_label = int(a.max())
    regions = cupy.full((max_label, 2 * a.ndim), -1, dtype=np.int32)

    in_params, out_params, operation = _get_find_objects_operation(
        a.shape, max_label
    )
    kern = cupy.ElementwiseKernel(
        in_params, out_params, operation, name="find2"
    )
    kern(a, max_label, regions, size=np.prod(a.shape))

    regions1 = cupy.asnumpy(regions)
    result = []
    for ii in range(max_label):
        # idx = 2 * input.ndim * ii if ndim > 0 else ii
        if regions1[ii][0] >= 0:
            slices = tuple(
                [
                    slice(regions1[ii, jj], regions1[ii, jj + a.ndim])
                    for jj in range(a.ndim)
                ]
            )
        else:
            slices = None
        result.append(slices)
    result
