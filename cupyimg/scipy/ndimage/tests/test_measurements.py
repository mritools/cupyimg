import os

import cupy as cp
import numpy as np
from numpy.testing import (
    assert_,
    assert_equal,
    assert_almost_equal,
    assert_raises,
)
from cupy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import suppress_warnings
import pytest

import cupyimg.scipy.ndimage as ndimage


types = [
    cp.int8,
    cp.uint8,
    cp.int16,
    cp.uint16,
    cp.int32,
    cp.uint32,
    cp.int64,
    cp.uint64,
    cp.float32,
    cp.float64,
]


np.mod(1.0, 1)  # Silence fmod bug on win-amd64. See #1408 and #1238.


# class Test_measurements_stats(object):
#     """ndimage.measurements._stats() is a utility function used by other functions."""

#     def test_a(self):
#         x = [0, 1, 2, 6]
#         labels = [0, 0, 1, 1]
#         index = [0, 1]
#         for shp in [(4,), (2, 2)]:
#             x = cp.asarray(x).reshape(shp)
#             labels = cp.asarray(labels).reshape(shp)
#             counts, sums = ndimage.measurements._stats(
#                 x, labels=labels, index=index
#             )
#             assert_array_equal(counts, [2, 2])
#             assert_array_equal(sums, [1.0, 8.0])

#     def test_b(self):
#         # Same data as test_a, but different labels.  The label 9 exceeds the
#         # length of 'labels', so this test will follow a different code path.
#         x = [0, 1, 2, 6]
#         labels = [0, 0, 9, 9]
#         index = [0, 9]
#         for shp in [(4,), (2, 2)]:
#             x = cp.asarray(x).reshape(shp)
#             labels = cp.asarray(labels).reshape(shp)
#             counts, sums = ndimage.measurements._stats(
#                 x, labels=labels, index=index
#             )
#             assert_array_equal(counts, [2, 2])
#             assert_array_equal(sums, [1.0, 8.0])

#     def test_a_centered(self):
#         x = [0, 1, 2, 6]
#         labels = [0, 0, 1, 1]
#         index = [0, 1]
#         for shp in [(4,), (2, 2)]:
#             x = cp.asarray(x).reshape(shp)
#             labels = cp.asarray(labels).reshape(shp)
#             counts, sums, centers = ndimage.measurements._stats(
#                 x, labels=labels, index=index, centered=True
#             )
#             assert_array_equal(counts, [2, 2])
#             assert_array_equal(sums, [1.0, 8.0])
#             assert_array_equal(centers, [0.5, 8.0])

#     def test_b_centered(self):
#         x = [0, 1, 2, 6]
#         labels = [0, 0, 9, 9]
#         index = [0, 9]
#         for shp in [(4,), (2, 2)]:
#             x = cp.asarray(x).reshape(shp)
#             labels = cp.asarray(labels).reshape(shp)
#             counts, sums, centers = ndimage.measurements._stats(
#                 x, labels=labels, index=index, centered=True
#             )
#             assert_array_equal(counts, [2, 2])
#             assert_array_equal(sums, [1.0, 8.0])
#             assert_array_equal(centers, [0.5, 8.0])

#     def test_nonint_labels(self):
#         x = [0, 1, 2, 6]
#         labels = [0.0, 0.0, 9.0, 9.0]
#         index = [0.0, 9.0]
#         for shp in [(4,), (2, 2)]:
#             x = cp.asarray(x).reshape(shp)
#             labels = cp.asarray(labels).reshape(shp)
#             counts, sums, centers = ndimage.measurements._stats(
#                 x, labels=labels, index=index, centered=True
#             )
#             assert_array_equal(counts, [2, 2])
#             assert_array_equal(sums, [1.0, 8.0])
#             assert_array_equal(centers, [0.5, 8.0])


class Test_measurements_select(object):
    """ndimage.measurements._select() is a utility function used by other functions."""

    def test_basic(self):
        x = cp.asarray([0, 1, 6, 2])
        cases = [
            ([0, 0, 1, 1], [0, 1]),  # "Small" integer labels
            ([0, 0, 9, 9], [0, 9]),  # A label larger than len(labels)
            ([0.0, 0.0, 7.0, 7.0], [0.0, 7.0]),  # Non-integer labels
        ]
        for labels, index in cases:
            labels = cp.asarray(labels)
            index = cp.asarray(index)
            result = ndimage.measurements._select(x, labels=labels, index=index)
            assert_(len(result) == 0)
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_max=True
            )
            assert_(len(result) == 1)
            assert_array_equal(result[0], [1, 6])
            result = ndimage.measurements._select(
                x, labels=labels, index=index, find_min=True
            )
            assert_(len(result) == 1)
            assert_array_equal(result[0], [0, 2])
            result = ndimage.measurements._select(
                x,
                labels=labels,
                index=index,
                find_min=True,
                find_min_positions=True,
            )
            assert_(len(result) == 2)
            assert_array_equal(result[0], [0, 2])
            assert_array_equal(result[1], [0, 3])
            assert_equal(result[1].dtype.kind, "i")
            result = ndimage.measurements._select(
                x,
                labels=labels,
                index=index,
                find_max=True,
                find_max_positions=True,
            )
            assert_(len(result) == 2)
            assert_array_equal(result[0], [1, 6])
            assert_array_equal(result[1], [1, 2])
            assert_equal(result[1].dtype.kind, "i")


def test_label01():
    data = cp.ones([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 1)
    assert_equal(n, 1)


def test_label02():
    data = cp.zeros([])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, 0)
    assert_equal(n, 0)


def test_label03():
    data = cp.ones([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1])
    assert_equal(n, 1)


def test_label04():
    data = cp.zeros([1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [0])
    assert_equal(n, 0)


def test_label05():
    data = cp.ones([5])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 1, 1, 1, 1])
    assert_equal(n, 1)


def test_label06():
    data = cp.asarray([1, 0, 1, 1, 0, 1])
    out, n = ndimage.label(data)
    assert_array_almost_equal(out, [1, 0, 2, 2, 0, 3])
    assert_equal(n, 3)


def test_label07():
    data = cp.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    out, n = ndimage.label(data)
    assert_array_almost_equal(
        out,
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    assert_equal(n, 0)


def test_label08():
    data = cp.asarray(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    out, n = ndimage.label(data)
    assert_array_almost_equal(
        out,
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 0],
            [3, 3, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 0],
        ],
    )
    assert_equal(n, 4)


def test_label09():
    data = cp.asarray(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
        ]
    )
    struct = ndimage.generate_binary_structure(2, 2)
    out, n = ndimage.label(data, struct)
    assert_array_almost_equal(
        out,
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 0],
            [2, 2, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0],
            [0, 0, 0, 3, 3, 0],
        ],
    )
    assert_equal(n, 3)


def test_label10():
    data = cp.asarray(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )
    struct = ndimage.generate_binary_structure(2, 2)
    out, n = ndimage.label(data, struct)
    assert_array_almost_equal(
        out,
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    )
    assert_equal(n, 1)


def test_label11():
    for type in types:
        data = cp.asarray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
            ],
            type,
        )
        out, n = ndimage.label(data)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 0],
            [3, 3, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 0],
        ]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 4)


def test_label11_inplace():
    for type in types:
        data = cp.asarray(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
            ],
            type,
        )
        n = ndimage.label(data, output=data)
        expected = [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 0],
            [3, 3, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [0, 0, 0, 4, 4, 0],
        ]
        assert_array_almost_equal(data, expected)
        assert_equal(n, 4)


def test_label12():
    for type in types:
        data = cp.asarray(
            [
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 1, 0, 1, 1],
                [0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 0],
            ],
            type,
        )
        out, n = ndimage.label(data)
        expected = [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 1)


def test_label13():
    for type in types:
        data = cp.asarray(
            [
                [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ],
            type,
        )
        out, n = ndimage.label(data)
        expected = [
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
        assert_array_almost_equal(out, expected)
        assert_equal(n, 1)


def test_label_output_typed():
    data = cp.ones([5])
    for t in types:
        output = cp.zeros([5], dtype=t)
        n = ndimage.label(data, output=output)
        assert_array_almost_equal(output, 1)
        assert_equal(n, 1)


def test_label_output_dtype():
    data = cp.ones([5])
    for t in types:
        output, n = ndimage.label(data, output=t)
        assert_array_almost_equal(output, 1)
        assert output.dtype == t


def test_label_output_wrong_size():
    data = cp.ones([5])
    for t in types:
        output = cp.zeros([10], t)
        assert_raises(
            (RuntimeError, ValueError), ndimage.label, data, output=output
        )


def test_label_structuring_elements():
    data = cp.asarray(
        np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data", "label_inputs.txt")
        )
    )
    strels = cp.asarray(
        np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data", "label_strels.txt")
        )
    )
    results = cp.asarray(
        np.loadtxt(
            os.path.join(os.path.dirname(__file__), "data", "label_results.txt")
        )
    )
    data = data.reshape((-1, 7, 7))
    strels = strels.reshape((-1, 3, 3))
    results = results.reshape((-1, 7, 7))
    r = 0
    for i in range(data.shape[0]):
        d = data[i, :, :]
        for j in range(strels.shape[0]):
            s = strels[j, :, :]
            assert_array_equal(ndimage.label(d, s)[0], results[r, :, :])
            r += 1


def test_label_default_dtype():
    test_array = cp.random.rand(10, 10)
    label, no_features = ndimage.label(test_array > 0.5)
    assert_(label.dtype in (cp.int32, cp.int64))
    # TODO: uncomment once find_objects is implemented
    # Shouldn't raise an exception
    # ndimage.find_objects(label)


# def test_find_objects01():
#     data = cp.ones([], dtype=int)
#     out = ndimage.find_objects(data)
#     assert_(out == [()])


# def test_find_objects02():
#     data = cp.zeros([], dtype=int)
#     out = ndimage.find_objects(data)
#     assert_(out == [])


# def test_find_objects03():
#     data = cp.ones([1], dtype=int)
#     out = ndimage.find_objects(data)
#     assert_equal(out, [(slice(0, 1, None),)])


# def test_find_objects04():
#     data = cp.zeros([1], dtype=int)
#     out = ndimage.find_objects(data)
#     assert_equal(out, [])


# def test_find_objects05():
#     data = cp.ones([5], dtype=int)
#     out = ndimage.find_objects(data)
#     assert_equal(out, [(slice(0, 5, None),)])


# def test_find_objects06():
#     data = cp.asarray([1, 0, 2, 2, 0, 3])
#     out = ndimage.find_objects(data)
#     assert_equal(out, [(slice(0, 1, None),),
#                        (slice(2, 4, None),),
#                        (slice(5, 6, None),)])


# def test_find_objects07():
#     data = cp.asarray([[0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0]])
#     out = ndimage.find_objects(data)
#     assert_equal(out, [])


# def test_find_objects08():
#     data = cp.asarray([[1, 0, 0, 0, 0, 0],
#                        [0, 0, 2, 2, 0, 0],
#                        [0, 0, 2, 2, 2, 0],
#                        [3, 3, 0, 0, 0, 0],
#                        [3, 3, 0, 0, 0, 0],
#                        [0, 0, 0, 4, 4, 0]])
#     out = ndimage.find_objects(data)
#     assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
#                        (slice(1, 3, None), slice(2, 5, None)),
#                        (slice(3, 5, None), slice(0, 2, None)),
#                        (slice(5, 6, None), slice(3, 5, None))])


# def test_find_objects09():
#     data = cp.asarray([[1, 0, 0, 0, 0, 0],
#                        [0, 0, 2, 2, 0, 0],
#                        [0, 0, 2, 2, 2, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 4, 4, 0]])
#     out = ndimage.find_objects(data)
#     assert_equal(out, [(slice(0, 1, None), slice(0, 1, None)),
#                        (slice(1, 3, None), slice(2, 5, None)),
#                        None,
#                        (slice(5, 6, None), slice(3, 5, None))])


def test_sum01():
    for type in types:
        input = cp.asarray([], type)
        output = ndimage.sum(input)
        assert_array_equal(output, 0.0)


def test_sum02():
    for type in types:
        input = cp.zeros([0, 4], type)
        output = ndimage.sum(input)
        assert_array_equal(output, 0.0)


def test_sum03():
    for type in types:
        input = cp.ones([], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 1.0)


def test_sum04():
    for type in types:
        input = cp.asarray([1, 2], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 3.0)


def test_sum05():
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.sum(input)
        assert_almost_equal(output, 10.0)


def test_sum06():
    labels = cp.asarray([], bool)
    for type in types:
        input = cp.asarray([], type)
        output = ndimage.sum(input, labels=labels)
        assert_array_equal(output, 0.0)


def test_sum07():
    labels = cp.ones([0, 4], bool)
    for type in types:
        input = cp.zeros([0, 4], type)
        output = ndimage.sum(input, labels=labels)
        assert_array_equal(output, 0.0)


def test_sum08():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([1, 2], type)
        output = ndimage.sum(input, labels=labels)
        assert_array_equal(output, 1.0)


def test_sum09():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels)
        assert_almost_equal(output, 4.0)


def test_sum10():
    labels = cp.asarray([1, 0], bool)
    input = cp.asarray([[1, 2], [3, 4]], bool)
    output = ndimage.sum(input, labels=labels)
    assert_almost_equal(output, 2.0)


def test_sum11():
    labels = cp.asarray([1, 2], cp.int8)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels, index=2)
        assert_almost_equal(output, 6.0)


def test_sum12():
    labels = cp.asarray([[1, 2], [2, 4]], cp.int8)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.sum(input, labels=labels, index=cp.asarray([4, 8, 2]))
        assert_array_almost_equal(output, [4.0, 0.0, 5.0])


@pytest.mark.parametrize("dtype", types)
def test_mean01(dtype):
    labels = cp.asarray([1, 0], bool)
    input = cp.asarray([[1, 2], [3, 4]], dtype)
    output = ndimage.mean(input, labels=labels)
    assert_almost_equal(output, 2.0)


def test_mean02():
    labels = cp.asarray([1, 0], bool)
    input = cp.asarray([[1, 2], [3, 4]], bool)
    output = ndimage.mean(input, labels=labels)
    assert_almost_equal(output, 1.0)


@pytest.mark.parametrize("dtype", types)
def test_mean03(dtype):
    labels = cp.asarray([1, 2])
    input = cp.asarray([[1, 2], [3, 4]], dtype)
    output = ndimage.mean(input, labels=labels, index=2)
    assert_almost_equal(output, 3.0)


@pytest.mark.parametrize("dtype", types)
def test_mean04(dtype):
    labels = cp.asarray([[1, 2], [2, 4]], cp.int8)
    olderr = np.seterr(all="ignore")
    try:
        input = cp.asarray([[1, 2], [3, 4]], dtype)
        output = ndimage.mean(input, labels=labels, index=cp.asarray([4, 8, 2]))
        assert_array_almost_equal(output[[0, 2]], [4.0, 2.5])
        assert_(cp.isnan(output[1]))
    finally:
        np.seterr(**olderr)


def test_minimum01():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels)
        assert_almost_equal(output, 1.0)


def test_minimum02():
    labels = cp.asarray([1, 0], bool)
    input = cp.asarray([[2, 2], [2, 4]], bool)
    output = ndimage.minimum(input, labels=labels)
    assert_almost_equal(output, 1.0)


def test_minimum03():
    labels = cp.asarray([1, 2])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels, index=2)
        assert_almost_equal(output, 2.0)


def test_minimum04():
    labels = cp.asarray([[1, 2], [2, 3]])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.minimum(input, labels=labels, index=[2, 3, 8])
        assert_array_almost_equal(output, [2.0, 4.0, 0.0])


def test_maximum01():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels)
        assert_almost_equal(output, 3.0)


def test_maximum02():
    labels = cp.asarray([1, 0], bool)
    input = cp.asarray([[2, 2], [2, 4]], bool)
    output = ndimage.maximum(input, labels=labels)
    assert_almost_equal(output, 1.0)


def test_maximum03():
    labels = cp.asarray([1, 2])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels, index=2)
        assert_almost_equal(output, 4.0)


def test_maximum04():
    labels = cp.asarray([[1, 2], [2, 3]])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.maximum(input, labels=labels, index=[2, 3, 8])
        assert_array_almost_equal(output, [3.0, 4.0, 0.0])


def test_maximum05():
    # Regression test for ticket #501 (Trac)
    x = cp.asarray([-3, -2, -1])
    assert_array_equal(ndimage.maximum(x), -1)


def test_median01():
    a = cp.asarray([[1, 2, 0, 1], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
    labels = cp.asarray(
        [[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 2], [3, 3, 0, 0]]
    )
    output = ndimage.median(a, labels=labels, index=cp.asarray([1, 2, 3]))
    assert_array_almost_equal(output, [2.5, 4.0, 6.0])


def test_median02():
    a = cp.asarray([[1, 2, 0, 1], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
    output = ndimage.median(a)
    assert_almost_equal(output, 1.0)


def test_median03():
    a = cp.asarray([[1, 2, 0, 1], [5, 3, 0, 4], [0, 0, 0, 7], [9, 3, 0, 0]])
    labels = cp.asarray(
        [[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 2], [3, 3, 0, 0]]
    )
    output = ndimage.median(a, labels=labels)
    assert_almost_equal(output, 3.0)


def test_variance01():
    olderr = np.seterr(all="ignore")
    try:
        for type in types:
            input = cp.asarray([], type)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "Mean of empty slice")
                output = ndimage.variance(input)
            assert_(cp.isnan(output))
    finally:
        np.seterr(**olderr)


def test_variance02():
    for type in types:
        input = cp.asarray([1], type)
        output = ndimage.variance(input)
        assert_almost_equal(output, 0.0)


def test_variance03():
    for type in types:
        input = cp.asarray([1, 3], type)
        output = ndimage.variance(input)
        assert_almost_equal(output, 1.0)


def test_variance04():
    input = cp.asarray([1, 0], bool)
    output = ndimage.variance(input)
    assert_almost_equal(output, 0.25)


def test_variance05():
    labels = cp.asarray([2, 2, 3])
    for type in types:
        input = cp.asarray([1, 3, 8], type)
        output = ndimage.variance(input, labels, 2)
        assert_almost_equal(output, 1.0)


def test_variance06():
    labels = cp.asarray([2, 2, 3, 3, 4])
    olderr = np.seterr(all="ignore")
    try:
        for type in types:
            input = cp.asarray([1, 3, 8, 10, 8], type)
            output = ndimage.variance(input, labels, cp.asarray([2, 3, 4]))
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])
    finally:
        np.seterr(**olderr)


def test_standard_deviation01():
    olderr = np.seterr(all="ignore")
    try:
        for type in types:
            input = cp.asarray([], type)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "Mean of empty slice")
                output = ndimage.standard_deviation(input)
            assert_(cp.isnan(output))
    finally:
        np.seterr(**olderr)


def test_standard_deviation02():
    for type in types:
        input = cp.asarray([1], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, 0.0)


def test_standard_deviation03():
    for type in types:
        input = cp.asarray([1, 3], type)
        output = ndimage.standard_deviation(input)
        assert_almost_equal(output, np.sqrt(1.0))


def test_standard_deviation04():
    input = cp.asarray([1, 0], bool)
    output = ndimage.standard_deviation(input)
    assert_almost_equal(output, 0.5)


def test_standard_deviation05():
    labels = cp.asarray([2, 2, 3])
    for type in types:
        input = cp.asarray([1, 3, 8], type)
        output = ndimage.standard_deviation(input, labels, 2)
        assert_almost_equal(output, 1.0)


def test_standard_deviation06():
    labels = cp.asarray([2, 2, 3, 3, 4])
    olderr = np.seterr(all="ignore")
    try:
        for type in types:
            input = cp.asarray([1, 3, 8, 10, 8], type)
            output = ndimage.standard_deviation(
                input, labels, cp.asarray([2, 3, 4])
            )
            assert_array_almost_equal(output, [1.0, 1.0, 0.0])
    finally:
        np.seterr(**olderr)


def test_standard_deviation07():
    labels = cp.asarray([1])
    olderr = np.seterr(all="ignore")
    try:
        for type in types:
            input = cp.asarray([-0.00619519], type)
            output = ndimage.standard_deviation(input, labels, cp.asarray([1]))
            assert_array_almost_equal(output, [0])
    finally:
        np.seterr(**olderr)


def test_minimum_position01():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.minimum_position(input, labels=labels)
        assert_equal(output, (0, 0))


def test_minimum_position02():
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input)
        assert_equal(output, (1, 2))


def test_minimum_position03():
    input = cp.asarray([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 1, 1]], bool)
    output = ndimage.minimum_position(input)
    assert_equal(output, (1, 2))


def test_minimum_position04():
    input = cp.asarray([[5, 4, 2, 5], [3, 7, 1, 2], [1, 5, 1, 1]], bool)
    output = ndimage.minimum_position(input)
    assert_equal(output, (0, 0))


def test_minimum_position05():
    labels = cp.asarray([1, 2, 0, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 2, 3]], type)
        output = ndimage.minimum_position(input, labels)
        assert_equal(output, (2, 0))


def test_minimum_position06():
    labels = cp.asarray([1, 2, 3, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input, labels, 2)
        assert_equal(output, (0, 1))


def test_minimum_position07():
    labels = cp.asarray([1, 2, 3, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 0, 2], [1, 5, 1, 1]], type)
        output = ndimage.minimum_position(input, labels, [2, 3])
        assert_equal(output[0], (0, 1))
        assert_equal(output[1], (1, 2))


def test_maximum_position01():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output = ndimage.maximum_position(input, labels=labels)
        assert_equal(output, (1, 0))


def test_maximum_position02():
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input)
        assert_equal(output, (1, 2))


def test_maximum_position03():
    input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], bool)
    output = ndimage.maximum_position(input)
    assert_equal(output, (0, 0))


def test_maximum_position04():
    labels = cp.asarray([1, 2, 0, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels)
        assert_equal(output, (1, 1))


def test_maximum_position05():
    labels = cp.asarray([1, 2, 0, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels, 1)
        assert_equal(output, (0, 0))


def test_maximum_position06():
    labels = cp.asarray([1, 2, 0, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels, [1, 2])
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (1, 1))


def test_maximum_position07():
    # Test float labels
    labels = cp.asarray([1.0, 2.5, 0.0, 4.5])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output = ndimage.maximum_position(input, labels, [1.0, 4.5])
        assert_equal(output[0], (0, 0))
        assert_equal(output[1], (0, 3))


def test_extrema01():
    labels = cp.asarray([1, 0], bool)
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        # TODO: grlee77: avoid use of item()?
        output1 = ndimage.extrema(input, labels=labels)
        output2 = ndimage.minimum(input, labels=labels)
        output3 = ndimage.maximum(input, labels=labels)
        output4 = ndimage.minimum_position(input, labels=labels)
        output5 = ndimage.maximum_position(input, labels=labels)
        assert_equal(output1, (output2, output3, output4, output5))


def test_extrema02():
    labels = cp.asarray([1, 2])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        # TODO: grlee77: avoid use of item()?
        output1 = ndimage.extrema(input, labels=labels, index=2)
        output2 = ndimage.minimum(input, labels=labels, index=2)
        output3 = ndimage.maximum(input, labels=labels, index=2)
        output4 = ndimage.minimum_position(input, labels=labels, index=2)
        output5 = ndimage.maximum_position(input, labels=labels, index=2)
        assert_equal(output1, (output2, output3, output4, output5))


def test_extrema03():
    labels = cp.asarray([[1, 2], [2, 3]])
    for type in types:
        input = cp.asarray([[1, 2], [3, 4]], type)
        output1 = ndimage.extrema(input, labels=labels, index=[2, 3, 8])
        output2 = ndimage.minimum(input, labels=labels, index=[2, 3, 8])
        output3 = ndimage.maximum(input, labels=labels, index=[2, 3, 8])
        output4 = ndimage.minimum_position(
            input, labels=labels, index=[2, 3, 8]
        )
        output5 = ndimage.maximum_position(
            input, labels=labels, index=[2, 3, 8]
        )
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


def test_extrema04():
    labels = cp.asarray([1, 2, 0, 4])
    for type in types:
        input = cp.asarray([[5, 4, 2, 5], [3, 7, 8, 2], [1, 5, 1, 1]], type)
        output1 = ndimage.extrema(input, labels, [1, 2])
        output2 = ndimage.minimum(input, labels, [1, 2])
        output3 = ndimage.maximum(input, labels, [1, 2])
        output4 = ndimage.minimum_position(input, labels, [1, 2])
        output5 = ndimage.maximum_position(input, labels, [1, 2])
        assert_array_almost_equal(output1[0], output2)
        assert_array_almost_equal(output1[1], output3)
        assert_array_almost_equal(output1[2], output4)
        assert_array_almost_equal(output1[3], output5)


def test_center_of_mass01():
    expected = [0.0, 0.0]
    for type in types:
        input = cp.asarray([[1, 0], [0, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass02():
    expected = [1, 0]
    for type in types:
        input = cp.asarray([[0, 0], [1, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass03():
    expected = [0, 1]
    for type in types:
        input = cp.asarray([[0, 1], [0, 0]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass04():
    expected = [1, 1]
    for type in types:
        input = cp.asarray([[0, 0], [0, 1]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass05():
    expected = [0.5, 0.5]
    for type in types:
        input = cp.asarray([[1, 1], [1, 1]], type)
        output = ndimage.center_of_mass(input)
        assert_array_almost_equal(output, expected)


def test_center_of_mass06():
    expected = [0.5, 0.5]
    input = cp.asarray([[1, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input)
    assert_array_almost_equal(output, expected)


def test_center_of_mass07():
    labels = cp.asarray([1, 0])
    expected = [0.5, 0.0]
    input = cp.asarray([[1, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input, labels)
    assert_array_almost_equal(output, expected)


def test_center_of_mass08():
    labels = cp.asarray([1, 2])
    expected = [0.5, 1.0]
    input = cp.asarray([[5, 2], [3, 1]], bool)
    output = ndimage.center_of_mass(input, labels, 2)
    assert_array_almost_equal(output, expected)


def test_center_of_mass09():
    labels = cp.asarray([1, 2])
    expected = [(0.5, 0.0), (0.5, 1.0)]
    input = cp.asarray([[1, 2], [1, 1]], bool)
    output = ndimage.center_of_mass(input, labels, cp.asarray([1, 2]))
    assert_array_almost_equal(output, expected)


def test_histogram01():
    expected = cp.ones(10)
    input = cp.arange(10)
    output = ndimage.histogram(input, 0, 10, 10)
    assert_array_almost_equal(output, expected)


def test_histogram02():
    labels = cp.asarray([1, 1, 1, 1, 2, 2, 2, 2])
    expected = [0, 2, 0, 1, 1]
    input = cp.asarray([1, 1, 3, 4, 3, 3, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, 1)
    assert_array_almost_equal(output, expected)


def test_histogram03():
    labels = cp.asarray([1, 0, 1, 1, 2, 2, 2, 2])
    expected1 = [0, 1, 0, 1, 1]
    expected2 = [0, 0, 0, 3, 0]
    input = cp.asarray([1, 1, 3, 4, 3, 5, 3, 3])
    output = ndimage.histogram(input, 0, 4, 5, labels, (1, 2))

    assert_array_almost_equal(output[0], expected1)
    assert_array_almost_equal(output[1], expected2)


def test_stat_funcs_2d():
    a = cp.asarray([[5, 6, 0, 0, 0], [8, 9, 0, 0, 0], [0, 0, 0, 3, 5]])
    lbl = cp.asarray([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 2, 2]])

    index = cp.asarray([1, 2])
    mean = ndimage.mean(a, labels=lbl, index=index)
    assert_array_equal(mean, [7.0, 4.0])

    var = ndimage.variance(a, labels=lbl, index=index)
    assert_array_equal(var, [2.5, 1.0])

    std = ndimage.standard_deviation(a, labels=lbl, index=index)
    assert_array_almost_equal(std, cp.sqrt(cp.asarray([2.5, 1.0])))

    med = ndimage.median(a, labels=lbl, index=index)
    assert_array_equal(med, [7.0, 4.0])

    min = ndimage.minimum(a, labels=lbl, index=index)
    assert_array_equal(min, [5, 3])

    max = ndimage.maximum(a, labels=lbl, index=index)
    assert_array_equal(max, [9, 5])
