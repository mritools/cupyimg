import cupy as cp
import numpy as np
from cupy.testing import assert_array_equal, assert_allclose

from cupyimg.skimage.segmentation import find_boundaries, mark_boundaries


white = (1, 1, 1)


def test_find_boundaries():
    image = cp.zeros((10, 10), dtype=cp.uint8)
    image[2:7, 2:7] = 1

    ref = cp.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_find_boundaries_bool():
    image = cp.zeros((5, 5), dtype=cp.bool)
    image[2:5, 2:5] = True

    ref = cp.asarray(
        [
            [False, False, False, False, False],
            [False, False, True, True, True],
            [False, True, True, True, True],
            [False, True, True, False, False],
            [False, True, True, False, False],
        ],
        dtype=cp.bool,
    )
    result = find_boundaries(image)
    assert_array_equal(result, ref)


def test_mark_boundaries():
    image = cp.zeros((10, 10))
    label_image = cp.zeros((10, 10), dtype=cp.uint8)
    label_image[2:7, 2:7] = 1

    ref = cp.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    marked = mark_boundaries(image, label_image, color=white, mode="thick")
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)

    ref = cp.asarray(
        [
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
            [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
            [2, 1, 1, 2, 0, 2, 1, 1, 2, 0],
            [2, 1, 1, 2, 2, 2, 1, 1, 2, 0],
            [2, 1, 1, 1, 1, 1, 1, 1, 2, 0],
            [2, 2, 1, 1, 1, 1, 1, 2, 2, 0],
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    marked = mark_boundaries(
        image, label_image, color=white, outline_color=(2, 2, 2), mode="thick"
    )
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_bool():
    image = cp.zeros((10, 10), dtype=cp.bool)
    label_image = cp.zeros((10, 10), dtype=cp.uint8)
    label_image[2:7, 2:7] = 1

    ref = cp.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    marked = mark_boundaries(image, label_image, color=white, mode="thick")
    result = cp.mean(marked, axis=-1)
    assert_array_equal(result, ref)


def test_mark_boundaries_subpixel():
    labels = cp.asarray(
        [[0, 0, 0, 0], [0, 0, 5, 0], [0, 1, 5, 0], [0, 0, 5, 0], [0, 0, 0, 0]],
        dtype=cp.uint8,
    )
    np.random.seed(0)
    # Note: use np.random to have same seed as NumPy
    # Note: use np.round until cp.round is implemented upstream
    image = cp.asarray(np.round(np.random.rand(*labels.shape), 2))
    marked = mark_boundaries(image, labels, color=white, mode="subpixel")
    marked_proj = cp.asarray(np.round(cp.mean(marked, axis=-1).get(), 2))

    # fmt: off
    ref_result = cp.asarray(
        [[0.55, 0.63, 0.72, 0.69, 0.6 , 0.55, 0.54],
         [0.45, 0.58, 0.72, 1.  , 1.  , 1.  , 0.69],
         [0.42, 0.54, 0.65, 1.  , 0.44, 1.  , 0.89],
         [0.69, 1.  , 1.  , 1.  , 0.69, 1.  , 0.83],
         [0.96, 1.  , 0.38, 1.  , 0.79, 1.  , 0.53],
         [0.89, 1.  , 1.  , 1.  , 0.38, 1.  , 0.16],
         [0.57, 0.78, 0.93, 1.  , 0.07, 1.  , 0.09],
         [0.2 , 0.52, 0.92, 1.  , 1.  , 1.  , 0.54],
         [0.02, 0.35, 0.83, 0.9 , 0.78, 0.81, 0.87]])
    # fmt: on

    # TODO: get fully equivalent interpolation/boundary as skimage

    # Note: grlee77: only test locations of ones, due to different default
    #                interpolation settings in CuPy version of mark_boundaries
    # assert_allclose(marked_proj, ref_result, atol=0.01)
    assert_allclose(marked_proj == 1, ref_result == 1, atol=0.01)
