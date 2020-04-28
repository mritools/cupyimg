import numpy as np
import pytest
from skimage._shared._warnings import expected_warnings

from cupyimg.skimage._shared.utils import (
    check_nD,
    _validate_interpolation_order,
)


def test_check_nD():
    z = np.random.random(200 ** 2).reshape((200, 200))
    x = z[10:30, 30:10]
    with pytest.raises(ValueError):
        check_nD(x, 2)


@pytest.mark.parametrize(
    "dtype", [bool, int, np.uint8, np.uint16, float, np.float32, np.float64]
)
@pytest.mark.parametrize("order", [None, -1, 0, 1, 2, 3, 4, 5, 6])
def test_validate_interpolation_order(dtype, order):
    if order is None:
        # Default order
        assert (
            _validate_interpolation_order(dtype, None) == 0
            if dtype == bool
            else 1
        )
    elif order < 0 or order > 5:
        # Order not in valid range
        with pytest.raises(ValueError):
            _validate_interpolation_order(dtype, order)
    elif dtype == bool and order != 0:
        # Deprecated order for bool array
        with expected_warnings(["Input image dtype is bool"]):
            assert _validate_interpolation_order(bool, order) == order
    else:
        # Valid use case
        assert _validate_interpolation_order(dtype, order) == order
