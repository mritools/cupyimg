import numpy as np
import pytest

from cupyimg.skimage._shared.utils import check_nD


def test_check_nD():
    z = np.random.random(200 ** 2).reshape((200, 200))
    x = z[10:30, 30:10]
    with pytest.raises(ValueError):
        check_nD(x, 2)
