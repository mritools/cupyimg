import pytest

import cupy as cp
import numpy as np
from cupy.testing import assert_allclose

from cupyimg.scipy import ndimage
from cupyimg.skimage.filters import median

# from cupyimg.skimage.filters import rank


@pytest.fixture
def image():
    return cp.asarray(
        [
            [1, 2, 3, 2, 1],
            [1, 1, 2, 2, 3],
            [3, 2, 1, 2, 1],
            [3, 2, 1, 1, 1],
            [1, 2, 1, 2, 3],
        ],
        dtype=np.uint8,
    )


# TODO: mode='rank' disabled until support is added
@pytest.mark.parametrize(
    "mask, shift_x, shift_y, mode, cval, behavior, n_warning, warning_type",
    [
        (True, None, None, "nearest", 0.0, "ndimage", 1, (UserWarning,)),
        (None, 1, None, "nearest", 0.0, "ndimage", 1, (UserWarning,)),
        (None, None, 1, "nearest", 0.0, "ndimage", 1, (UserWarning,)),
        (True, 1, 1, "nearest", 0.0, "ndimage", 1, (UserWarning,)),
        # (None, False, False, 'constant', 0.0, 'rank', 1, (UserWarning,)),
        # (None, False, False, 'nearest', 0.0, 'rank', 0, []),
        (None, False, False, "nearest", 0.0, "ndimage", 0, []),
    ],
)
def test_median_warning(
    image, mask, shift_x, shift_y, mode, cval, behavior, n_warning, warning_type
):
    if mask:
        mask = cp.ones((image.shape), dtype=np.bool_)

    with pytest.warns(None) as records:
        median(
            image,
            mask=mask,
            shift_x=shift_x,
            shift_y=shift_y,
            mode=mode,
            behavior=behavior,
        )

    assert len(records) == n_warning
    for rec in records:
        assert isinstance(rec.message, warning_type)


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    "behavior, func, params",
    [("ndimage", ndimage.median_filter, {"size": (3, 3)})]
    # ('rank', rank.median, {'selem': np.ones((3, 3), dtype=np.uint8)}),]
)
def test_median_behavior(image, behavior, func, params):
    assert_allclose(median(image, behavior=behavior), func(image, **params))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_median_preserve_dtype(image, dtype):
    median_image = median(image.astype(dtype), behavior="ndimage")
    assert median_image.dtype == dtype


# TODO: update if rank.median implemented
# def test_median_error_ndim():
#     img = cp.random.randint(0, 10, size=(5, 5, 5), dtype=np.uint8)
#     with pytest.raises(ValueError):
#         median(img, behavior='rank')


# TODO: update if rank.median implemented
@pytest.mark.parametrize(
    "img, behavior",
    # (np.random.randint(0, 10, size=(3, 3), dtype=np.uint8), 'rank'),
    [
        (cp.random.randint(0, 10, size=(3, 3), dtype=np.uint8), "ndimage"),
        (cp.random.randint(0, 10, size=(3, 3, 3), dtype=np.uint8), "ndimage"),
    ],
)
def test_median(img, behavior):
    median(img, behavior=behavior)
