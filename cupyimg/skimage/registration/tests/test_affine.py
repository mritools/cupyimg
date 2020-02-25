import cupy as cp
import numpy as np
from skimage import data

from cupyimg.skimage.registration import _affine
from cupyimg.scipy import ndimage as ndi
from cupy.testing import assert_array_almost_equal


def test_register_affine():
    reference = cp.asarray(data.camera()[::4, ::4])
    forward = cp.asarray([[1.1, 0, 0],
                          [0  , 1, 0],
                          [0  , 0, 1]])

    inverse = cp.linalg.inv(forward)

    target = ndi.affine_transform(reference, forward)
    powell_options = {'xtol': 0.001, 'ftol': 0.001}  # speed things up a little
    matrix = _affine.affine(reference, target, pyramid_minimum_size=16,
                            options=powell_options)
    assert_array_almost_equal(matrix, inverse, decimal=1)


def test_register_affine_multichannel():
    reference = cp.asarray(data.astronaut()[::4, ::4])
    forward = cp.asarray([[1.1, 0, 0],
                          [0  , 1, 0],
                          [0  , 0, 1]])
    inverse = cp.linalg.inv(forward)
    target = cp.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(reference[..., ch], forward,
                             output=target[..., ch])
    powell_options = {'xtol': 0.001, 'ftol': 0.001}  # speed things up a little
    matrix = _affine.affine(reference, target, multichannel=True,
                            pyramid_minimum_size=16, options=powell_options)
    assert_array_almost_equal(matrix, inverse, decimal=1)
