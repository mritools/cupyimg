# TODO: implement kernel corresponding to CASE_FIND_OBJECT_POINT

# Note: May not be well suited to ElementwiseKernel because multiple threads
# will end up accessing the same locations in _regions, requiring atomic
# operations.


# // CASE_FIND_OBJECT_POINT(NPY_BOOL, npy_bool,
# //                        pi, regions, input, max_label, ii);

# #define CASE_FIND_OBJECT_POINT(_TYPE, _type, _pi, _regions, _array,  \
#                                _max_label, _ii)                      \
# case _TYPE:                                                          \
# {                                                                    \
#     int _kk;                                                         \
#     npy_intp _rank = PyArray_NDIM(_array);                           \
#     npy_intp _sindex = *(_type *)_pi - 1;                            \
#     if (_sindex >= 0 && _sindex < _max_label) {                      \
#         if (_rank > 0) {                                             \
#             _sindex *= 2 * _rank;                                    \
#             if (_regions[_sindex] < 0) {                             \
#                 for (_kk = 0; _kk < _rank; _kk++) {                  \
#                     npy_intp _cc = _ii.coordinates[_kk];             \
#                     _regions[_sindex + _kk] = _cc;                   \
#                     _regions[_sindex + _kk + _rank] = _cc + 1;       \
#                 }                                                    \
#             }                                                        \
#             else {                                                   \
#                 for(_kk = 0; _kk < _rank; _kk++) {                   \
#                     npy_intp _cc = _ii.coordinates[_kk];             \
#                     if (_cc < _regions[_sindex + _kk]) {             \
#                         _regions[_sindex + _kk] = _cc;               \
#                     }                                                \
#                     if (_cc + 1 > _regions[_sindex + _kk + _rank]) { \
#                         _regions[_sindex + _kk + _rank] = _cc + 1;   \
#                     }                                                \
#                 }                                                    \
#             }                                                        \
#         }                                                            \
#         else {                                                       \
#             _regions[_sindex] = 1;                                   \
#         }                                                            \
#     }                                                                \
# }                                                                    \
# break
