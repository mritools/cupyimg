"""Tests for spline filtering."""
import cupy as cp
import pytest

from cupy.testing import assert_array_almost_equal

from cupyimg.scipy import ndimage


def get_spline_knot_values(order):
    """Knot values to the right of a B-spline's center."""
    knot_values = {
        0: [1],
        1: [1],
        2: [6, 1],
        3: [4, 1],
        4: [230, 76, 1],
        5: [66, 26, 1],
    }

    return knot_values[order]


def make_spline_knot_matrix(n, order, mode="mirror"):
    """Matrix to invert to find the spline coefficients."""
    knot_values = get_spline_knot_values(order)

    matrix = cp.zeros((n, n))
    for diag, knot_value in enumerate(knot_values):
        indices = cp.arange(diag, n)
        if diag == 0:
            matrix[indices, indices] = knot_value
        else:
            matrix[indices, indices - diag] = knot_value
            matrix[indices - diag, indices] = knot_value

    knot_values_sum = knot_values[0] + 2 * sum(knot_values[1:])

    if mode == "mirror":
        start, step = 1, 1
    elif mode == "reflect":
        start, step = 0, 1
    elif mode == "wrap":
        start, step = -1, -1
    else:
        raise ValueError("unsupported mode {}".format(mode))

    for row in range(len(knot_values) - 1):
        for idx, knot_value in enumerate(knot_values[row + 1 :]):
            matrix[row, start + step * idx] += knot_value
            matrix[-row - 1, -start - 1 - step * idx] += knot_value

    return matrix / knot_values_sum


@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("mode", ["mirror", "wrap", "reflect"])
def test_spline_filter_vs_matrix_solution(order, mode):
    n = 100
    eye = cp.eye(n, dtype=float)
    spline_filter_axis_0 = ndimage.spline_filter1d(
        eye, axis=0, order=order, mode=mode
    )
    spline_filter_axis_1 = ndimage.spline_filter1d(
        eye, axis=1, order=order, mode=mode
    )
    matrix = make_spline_knot_matrix(n, order, mode=mode)
    assert_array_almost_equal(eye, cp.dot(spline_filter_axis_0, matrix))
    assert_array_almost_equal(eye, cp.dot(spline_filter_axis_1, matrix.T))
