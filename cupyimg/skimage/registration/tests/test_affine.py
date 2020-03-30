import cupy as cp
import numpy as np
from skimage import data

from cupyimg.skimage.registration import _affine
from cupyimg.scipy import ndimage as ndi
from cupy.testing import assert_array_equal, assert_array_almost_equal


def test_register_affine():
    reference = cp.asarray(data.camera()[::4, ::4])
    # fmt: off
    forward = cp.asarray([[1.1, 0, 0],
                          [0  , 1, 0],
                          [0  , 0, 1]])
    # fmt: on
    inverse = cp.linalg.inv(forward)

    target = ndi.affine_transform(reference, forward)
    powell_options = {"xtol": 0.001, "ftol": 0.001}  # speed things up a little
    matrix = _affine.affine(
        reference, target, pyramid_minimum_size=16, options=powell_options
    )
    assert_array_almost_equal(matrix, inverse, decimal=1)


def test_register_affine_multichannel():
    reference = cp.asarray(data.astronaut()[::4, ::4])
    # fmt: off
    forward = cp.asarray([[1.1, 0, 0],
                          [0  , 1, 0],
                          [0  , 0, 1]])
    # fmt: on
    inverse = cp.linalg.inv(forward)
    target = cp.empty_like(reference)
    for ch in range(reference.shape[-1]):
        ndi.affine_transform(
            reference[..., ch], forward, output=target[..., ch]
        )
    powell_options = {"xtol": 0.001, "ftol": 0.001}  # speed things up a little
    matrix = _affine.affine(
        reference,
        target,
        multichannel=True,
        pyramid_minimum_size=16,
        options=powell_options,
    )
    assert_array_almost_equal(matrix, inverse, decimal=1)


def test_matrix_parameter_vector_conversion():
    for ndim in range(2, 5):
        p_v = cp.asarray(np.random.rand((ndim + 1) * ndim))
        matrix = _affine._parameter_vector_to_matrix(p_v)
        en = cp.zeros(ndim + 1)
        en[-1] = 1
        p_v_2 = cp.concatenate(
            (p_v.reshape((ndim, ndim + 1)), en[np.newaxis]), axis=0
        )
        assert_array_equal(matrix, p_v_2)
