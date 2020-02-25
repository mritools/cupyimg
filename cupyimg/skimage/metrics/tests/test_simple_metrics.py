import cupy
import numpy as np
import pytest

from cupyimg.skimage._shared._warnings import expected_warnings
from skimage import data  # TODO: remove need for skimage import
from cupyimg.skimage.metrics import (
    peak_signal_noise_ratio,
    normalized_root_mse,
    mean_squared_error,
    normalized_mutual_information,
)

np.random.seed(
    5
)  # need exact NumPy seed here. (Don't use CuPy as it won't be identical)
cam = cupy.asarray(data.camera())
sigma = 20.0
noise = cupy.asarray(sigma * np.random.randn(*cam.shape))
cam_noisy = cupy.clip(cam + noise, 0, 255)
cam_noisy = cam_noisy.astype(cam.dtype)

assert_equal = cupy.testing.assert_array_equal
assert_almost_equal = cupy.testing.assert_array_almost_equal


def test_PSNR_vs_IPOL():
    # Tests vs. imdiff result from the following IPOL article and code:
    # https://www.ipol.im/pub/art/2011/g_lmii/
    p_IPOL = 22.4497
    p = peak_signal_noise_ratio(cam, cam_noisy)
    assert_almost_equal(p, p_IPOL, decimal=4)


def test_PSNR_float():
    p_uint8 = peak_signal_noise_ratio(cam, cam_noisy)
    p_float64 = peak_signal_noise_ratio(
        cam / 255.0, cam_noisy / 255.0, data_range=1
    )
    assert_almost_equal(p_uint8, p_float64, decimal=5)

    # mixed precision inputs
    p_mixed = peak_signal_noise_ratio(
        cam / 255.0, cam_noisy.astype(np.float32) / 255.0, data_range=1
    )
    assert_almost_equal(p_mixed, p_float64, decimal=5)

    # mismatched dtype results in a warning if data_range is unspecified
    with expected_warnings(["Inputs have mismatched dtype"]):
        p_mixed = peak_signal_noise_ratio(
            cam / 255.0, cam_noisy.astype(np.float32) / 255.0
        )
    assert_almost_equal(p_mixed, p_float64, decimal=5)


def test_PSNR_errors():
    # shape mismatch
    with pytest.raises(ValueError):
        peak_signal_noise_ratio(cam, cam[:-1, :])


def test_NRMSE():
    x = cupy.ones(4)
    y = cupy.asarray([0.0, 2.0, 2.0, 2.0])
    assert_equal(
        normalized_root_mse(y, x, normalization="mean"), 1 / cupy.mean(y)
    )
    assert_equal(
        normalized_root_mse(y, x, normalization="euclidean"), 1 / cupy.sqrt(3)
    )
    assert_equal(
        normalized_root_mse(y, x, normalization="min-max"),
        1 / (y.max() - y.min()),
    )

    # mixed precision inputs are allowed
    assert_almost_equal(
        normalized_root_mse(y, x.astype(cupy.float32), normalization="min-max"),
        1 / (y.max() - y.min()),
    )


def test_NRMSE_no_int_overflow():
    camf = cam.astype(cupy.float32)
    cam_noisyf = cam_noisy.astype(cupy.float32)
    assert_almost_equal(
        mean_squared_error(cam, cam_noisy), mean_squared_error(camf, cam_noisyf)
    )
    assert_almost_equal(
        normalized_root_mse(cam, cam_noisy),
        normalized_root_mse(camf, cam_noisyf),
    )


def test_NRMSE_errors():
    x = cupy.ones(4)
    # shape mismatch
    with pytest.raises(ValueError):
        normalized_root_mse(x[:-1], x)
    # invalid normalization name
    with pytest.raises(ValueError):
        normalized_root_mse(x, x, normalization="foo")


def test_nmi():
    assert_almost_equal(normalized_mutual_information(cam, cam), 2)
    assert normalized_mutual_information(
        cam, cam_noisy
    ) < normalized_mutual_information(cam, cam)


def test_nmi_different_sizes():
    assert normalized_mutual_information(cam[:, :400], cam[:400, :]) > 1


def test_nmi_random():
    random1 = cupy.random.random((100, 100))
    random2 = cupy.random.random((100, 100))
    assert_almost_equal(
        normalized_mutual_information(random1, random2, bins=10), 1, decimal=2
    )
