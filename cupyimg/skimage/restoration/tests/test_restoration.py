from os.path import abspath, dirname, join as pjoin

import cupy as cp
import numpy as np
from scipy.signal import convolve2d
from cupyimg.scipy import ndimage as ndi

from cupyimg.skimage import restoration
from cupyimg.skimage.restoration import uft

try:
    from skimage._shared.testing import fetch

    have_fetch = True
except ImportError:
    have_fetch = False


def camera():
    import skimage
    import skimage.data

    return cp.asarray(skimage.img_as_float(skimage.data.camera()))


test_img = camera()


def test_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img.get(), psf, "same")
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)

    psf = cp.asarray(psf)
    data = cp.asarray(data)

    deconvolved = restoration.wiener(data, psf, 0.05)

    if have_fetch:
        path = fetch("restoration/tests/camera_wiener.npy")
    else:
        path = pjoin(dirname(abspath(__file__)), "camera_wiener.npy")
    cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)
    deconvolved = restoration.wiener(
        data, otf, 0.05, reg=laplacian, is_real=False
    )
    cp.testing.assert_allclose(cp.real(deconvolved), np.load(path), rtol=1e-3)


def test_unsupervised_wiener():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img.get(), psf, "same")
    cp.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)

    psf = cp.asarray(psf)
    data = cp.asarray(data)
    deconvolved, _ = restoration.unsupervised_wiener(data, psf)

    # grlee77: Note: skip comparisons based on a particular random seed
    # if have_fetch:
    #     path = fetch("restoration/tests/camera_unsup.npy")
    # else:
    #     path = pjoin(dirname(abspath(__file__)), 'camera_unsup.npy')
    # cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)

    _, laplacian = uft.laplacian(2, data.shape)
    otf = uft.ir2tf(psf, data.shape, is_real=False)

    cp.random.seed(0)
    restoration.unsupervised_wiener(
        data,
        otf,
        reg=laplacian,
        is_real=False,
        user_params={"callback": lambda x: None},
    )[0]
    # grlee77: Note: skip comparisons based on a particular random seed
    # if have_fetch:
    #     path = fetch("restoration/tests/camera_unsup2.npy")
    # else:
    #     path = pjoin(dirname(abspath(__file__)), 'camera_unsup2.npy')
    # cp.testing.assert_allclose(cp.real(deconvolved), np.load(path), rtol=1e-3)


def test_image_shape():
    """Test that shape of output image in deconvolution is same as input.

    This addresses issue #1172.
    """
    point = cp.zeros((5, 5), np.float)
    point[2, 2] = 1.0
    psf = ndi.gaussian_filter(point, sigma=1.0)
    # image shape: (45, 45), as reported in #1172
    image = cp.asarray(test_img[110:155, 225:270])  # just the face
    image_conv = ndi.convolve(image, psf)
    deconv_sup = restoration.wiener(image_conv, psf, 1)
    deconv_un = restoration.unsupervised_wiener(image_conv, psf)[0]
    # test the shape
    assert image.shape == deconv_sup.shape
    assert image.shape == deconv_un.shape
    # test the reconstruction error
    sup_relative_error = cp.abs(deconv_sup - image) / image
    un_relative_error = cp.abs(deconv_un - image) / image
    cp.testing.assert_array_less(np.median(sup_relative_error.get()), 0.1)
    cp.testing.assert_array_less(np.median(un_relative_error.get()), 0.1)


def test_richardson_lucy():
    psf = np.ones((5, 5)) / 25
    data = convolve2d(test_img.get(), psf, "same")
    np.random.seed(0)
    data += 0.1 * data.std() * np.random.standard_normal(data.shape)

    data = cp.asarray(data)
    psf = cp.asarray(psf)
    deconvolved = restoration.richardson_lucy(data, psf, 5)

    if have_fetch:
        path = fetch("restoration/tests/camera_rl.npy")
    else:
        path = pjoin(dirname(abspath(__file__)), "camera_rl.npy")
    cp.testing.assert_allclose(deconvolved, np.load(path), rtol=1e-3)
