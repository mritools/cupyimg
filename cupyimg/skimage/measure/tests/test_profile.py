import cupy as cp
import numpy as np
from cupy.testing import assert_array_equal, assert_array_almost_equal

from cupyimg.skimage.measure import profile_line


image = cp.arange(100).reshape((10, 10)).astype(cp.float)


def test_horizontal_rightward():
    prof = profile_line(image, (0, 2), (0, 8), order=0)
    expected_prof = cp.arange(2, 9)
    assert_array_equal(prof, expected_prof)


def test_horizontal_leftward():
    prof = profile_line(image, (0, 8), (0, 2), order=0)
    expected_prof = cp.arange(8, 1, -1)
    assert_array_equal(prof, expected_prof)


def test_vertical_downward():
    prof = profile_line(image, (2, 5), (8, 5), order=0)
    expected_prof = cp.arange(25, 95, 10)
    assert_array_equal(prof, expected_prof)


def test_vertical_upward():
    prof = profile_line(image, (8, 5), (2, 5), order=0)
    expected_prof = cp.arange(85, 15, -10)
    assert_array_equal(prof, expected_prof)


def test_45deg_right_downward():
    prof = profile_line(image, (2, 2), (8, 8), order=0)
    expected_prof = cp.array([22, 33, 33, 44, 55, 55, 66, 77, 77, 88])
    # repeats are due to aliasing using nearest neighbor interpolation.
    # to see this, imagine a diagonal line with markers every unit of
    # length traversing a checkerboard pattern of squares also of unit
    # length. Because the line is diagonal, sometimes more than one
    # marker will fall on the same checkerboard box.
    assert_array_almost_equal(prof, expected_prof)


def test_45deg_right_downward_interpolated():
    prof = profile_line(image, (2, 2), (8, 8), order=1)
    expected_prof = cp.linspace(22, 88, 10)
    assert_array_almost_equal(prof, expected_prof)


def test_45deg_right_upward():
    prof = profile_line(image, (8, 2), (2, 8), order=1)
    expected_prof = cp.arange(82, 27, -6)
    assert_array_almost_equal(prof, expected_prof)


def test_45deg_left_upward():
    prof = profile_line(image, (8, 8), (2, 2), order=1)
    expected_prof = cp.arange(88, 21, -22.0 / 3)
    assert_array_almost_equal(prof, expected_prof)


def test_45deg_left_downward():
    prof = profile_line(image, (2, 8), (8, 2), order=1)
    expected_prof = cp.arange(28, 83, 6)
    assert_array_almost_equal(prof, expected_prof)


def test_pythagorean_triangle_right_downward():
    prof = profile_line(image, (1, 1), (7, 9), order=0)
    expected_prof = cp.array([11, 22, 23, 33, 34, 45, 56, 57, 67, 68, 79])
    assert_array_equal(prof, expected_prof)


def test_pythagorean_triangle_right_downward_interpolated():
    prof = profile_line(image, (1, 1), (7, 9), order=1)
    expected_prof = cp.linspace(11, 79, 11)
    assert_array_almost_equal(prof, expected_prof)


pyth_image = np.zeros((6, 7), cp.float)
line = ((1, 2, 2, 3, 3, 4), (1, 2, 3, 3, 4, 5))
below = ((2, 2, 3, 4, 4, 5), (0, 1, 2, 3, 4, 4))
above = ((0, 1, 1, 2, 3, 3), (2, 2, 3, 4, 5, 6))
pyth_image[line] = 1.8
pyth_image[below] = 0.6
pyth_image[above] = 0.6
pyth_image = cp.asarray(pyth_image)


def test_pythagorean_triangle_right_downward_linewidth():
    prof = profile_line(pyth_image, (1, 1), (4, 5), linewidth=3, order=0)
    expected_prof = cp.ones(6)
    assert_array_almost_equal(prof, expected_prof)


def test_pythagorean_triangle_right_upward_linewidth():
    prof = profile_line(
        pyth_image[::-1, :], (4, 1), (1, 5), linewidth=3, order=0
    )
    expected_prof = cp.ones(6)
    assert_array_almost_equal(prof, expected_prof)


def test_pythagorean_triangle_transpose_left_down_linewidth():
    prof = profile_line(
        pyth_image.T[:, ::-1], (1, 4), (5, 1), linewidth=3, order=0
    )
    expected_prof = cp.ones(6)
    assert_array_almost_equal(prof, expected_prof)
