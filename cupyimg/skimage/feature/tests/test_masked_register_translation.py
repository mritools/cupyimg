from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from cupyimg.scipy.ndimage import fourier_shift
from skimage.io import imread
from skimage.data import camera

from cupyimg.skimage.feature.register_translation import register_translation
from cupyimg.skimage.feature.masked_register_translation import (
    masked_register_translation,
    cross_correlate_masked,
)
from cupyimg.skimage._shared.fft import fftmodule as fft

# Location of test images
# These images are taken from Dirk Padfields' MATLAB package
# available on his website: www.dirkpadfield.com
IMAGES_DIR = Path(__file__).parent / "data"


def test_masked_registration_vs_register_translation():
    """masked_register_translation should give the same results as
    register_translation in the case of trivial masks."""
    reference_image = cp.asarray(camera())
    shift = (-7, 12)
    shifted = cp.real(
        fft.ifft2(fourier_shift(fft.fft2(reference_image), shift))
    )
    trivial_mask = cp.ones_like(reference_image)

    nonmasked_result, *_ = register_translation(reference_image, shifted)
    masked_result = masked_register_translation(
        reference_image, shifted, trivial_mask, overlap_ratio=1 / 10
    )

    cp.testing.assert_array_equal(nonmasked_result, masked_result)


def test_masked_registration_random_masks():
    """masked_register_translation should be able to register translations
    between images even with random masks."""
    # See random number generator for reproducible results
    np.random.seed(23)

    reference_image = cp.asarray(camera())
    shift = (-7, 12)
    shifted = np.real(
        fft.ifft2(fourier_shift(fft.fft2(reference_image), shift))
    )

    # Random masks with 75% of pixels being valid
    ref_mask = np.random.choice(
        [True, False], reference_image.shape, p=[3 / 4, 1 / 4]
    )
    shifted_mask = np.random.choice(
        [True, False], shifted.shape, p=[3 / 4, 1 / 4]
    )
    ref_mask = cp.asarray(ref_mask)
    shifted_mask = cp.asarray(shifted_mask)

    measured_shift = masked_register_translation(
        reference_image, shifted, ref_mask, shifted_mask
    )
    cp.testing.assert_array_equal(measured_shift, -cp.asarray(shift))


def test_masked_registration_random_masks_non_equal_sizes():
    """masked_register_translation should be able to register
    translations between images that are not the same size even
    with random masks."""
    # See random number generator for reproducible results
    np.random.seed(23)

    reference_image = cp.asarray(camera())
    shift = (-7, 12)
    shifted = cp.real(
        fft.ifft2(fourier_shift(fft.fft2(reference_image), shift))
    )

    # Crop the shifted image
    shifted = shifted[64:-64, 64:-64]

    # Random masks with 75% of pixels being valid
    ref_mask = np.random.choice(
        [True, False], reference_image.shape, p=[3 / 4, 1 / 4]
    )
    shifted_mask = np.random.choice(
        [True, False], shifted.shape, p=[3 / 4, 1 / 4]
    )

    reference_image = cp.asarray(reference_image)
    shifted = cp.asarray(shifted)
    measured_shift = masked_register_translation(
        reference_image,
        shifted,
        cp.ones(ref_mask.shape, dtype=ref_mask.dtype),
        cp.ones(shifted_mask.shape, dtype=shifted_mask.dtype),
    )
    cp.testing.assert_array_equal(measured_shift, -cp.asarray(shift))


def test_masked_registration_padfield_data():
    """ Masked translation registration should behave like in the original
    publication """
    # Test translated from MATLABimplementation `MaskedFFTRegistrationTest`
    # file. You can find the source code here:
    # http://www.dirkpadfield.com/Home/MaskedFFTRegistrationCode.zip

    shifts = [(75, 75), (-130, 130), (130, 130)]
    for xi, yi in shifts:

        fixed_image = cp.asarray(
            imread(IMAGES_DIR / "OriginalX{:d}Y{:d}.png".format(xi, yi))
        )
        moving_image = cp.asarray(
            imread(IMAGES_DIR / "TransformedX{:d}Y{:d}.png".format(xi, yi))
        )

        # Valid pixels are 1
        fixed_mask = fixed_image != 0
        moving_mask = moving_image != 0

        # Note that shifts in x and y and shifts in cols and rows
        shift_y, shift_x = masked_register_translation(
            fixed_image,
            moving_image,
            fixed_mask,
            moving_mask,
            overlap_ratio=1 / 10,
        )
        # Note: by looking at the test code from Padfield's
        # MaskedFFTRegistrationCode repository, the
        # shifts were not xi and yi, but xi and -yi
        cp.testing.assert_array_equal((shift_x, shift_y), (-xi, yi))


def test_cross_correlate_masked_output_shape():
    """Masked normalized cross-correlation should return a shape
    of N + M + 1 for each transform axis."""
    shape1 = (15, 4, 5)
    shape2 = (6, 12, 7)
    expected_full_shape = tuple(np.array(shape1) + np.array(shape2) - 1)
    expected_same_shape = shape1

    arr1 = cp.zeros(shape1)
    arr2 = cp.zeros(shape2)
    # Trivial masks
    m1 = cp.ones_like(arr1)
    m2 = cp.ones_like(arr2)

    full_xcorr = cross_correlate_masked(
        arr1, arr2, m1, m2, axes=(0, 1, 2), mode="full"
    )
    assert full_xcorr.dtype.kind != "c"  # grlee77: output should be real
    assert full_xcorr.shape == expected_full_shape

    same_xcorr = cross_correlate_masked(
        arr1, arr2, m1, m2, axes=(0, 1, 2), mode="same"
    )
    assert same_xcorr.shape == expected_same_shape


def test_cross_correlate_masked_test_against_mismatched_dimensions():
    """Masked normalized cross-correlation should raise an error if array
    dimensions along non-transformation axes are mismatched."""
    shape1 = (23, 1, 1)
    shape2 = (6, 2, 2)

    arr1 = cp.zeros(shape1)
    arr2 = cp.zeros(shape2)

    # Trivial masks
    m1 = cp.ones_like(arr1)
    m2 = cp.ones_like(arr2)

    with pytest.raises(ValueError):
        cross_correlate_masked(arr1, arr2, m1, m2, axes=(1, 2))


def test_cross_correlate_masked_output_range():
    """Masked normalized cross-correlation should return between 1 and -1."""
    # See random number generator for reproducible results
    np.random.seed(23)

    # Array dimensions must match along non-transformation axes, in
    # this case
    # axis 0
    shape1 = (15, 4, 5)
    shape2 = (15, 12, 7)

    # Initial array ranges between -5 and 5
    arr1 = 10 * np.random.random(shape1) - 5
    arr2 = 10 * np.random.random(shape2) - 5

    # random masks
    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)

    arr1 = cp.asarray(arr1)
    arr2 = cp.asarray(arr2)
    m1 = cp.asarray(m1)
    m2 = cp.asarray(m2)
    xcorr = cross_correlate_masked(arr1, arr2, m1, m2, axes=(1, 2))

    # No assert array less or equal, so we add an eps
    # Also could not find an `assert_array_greater`, Use (-xcorr) instead
    eps = np.finfo(np.float).eps
    cp.testing.assert_array_less(xcorr, 1 + eps)
    cp.testing.assert_array_less(-xcorr, 1 + eps)


def test_cross_correlate_masked_side_effects():
    """Masked normalized cross-correlation should not modify the inputs."""
    shape1 = (2, 2, 2)
    shape2 = (2, 2, 2)

    arr1 = cp.zeros(shape1)
    arr2 = cp.zeros(shape2)

    # Trivial masks
    m1 = cp.ones_like(arr1)
    m2 = cp.ones_like(arr2)

    # for arr in (arr1, arr2, m1, m2):
    #    arr.setflags(write=False)
    arr1c, arr2c, m1c, m2c = [a.copy() for a in (arr1, arr2, m1, m2)]

    cross_correlate_masked(arr1, arr2, m1, m2)

    cp.testing.assert_array_equal(arr1, arr1c)
    cp.testing.assert_array_equal(arr2, arr2c)
    cp.testing.assert_array_equal(m1, m1c)
    cp.testing.assert_array_equal(m2, m2c)


def test_cross_correlate_masked_over_axes():
    """Masked normalized cross-correlation over axes should be
    equivalent to a loop over non-transform axes."""
    # See random number generator for reproducible results
    np.random.seed(23)

    arr1 = np.random.random((8, 8, 5))
    arr2 = np.random.random((8, 8, 5))

    m1 = np.random.choice([True, False], arr1.shape)
    m2 = np.random.choice([True, False], arr2.shape)

    arr1 = cp.asarray(arr1)
    arr2 = cp.asarray(arr2)
    m1 = cp.asarray(m1)
    m2 = cp.asarray(m2)

    # Loop over last axis
    with_loop = cp.empty_like(arr1, dtype=np.complex)
    for index in range(arr1.shape[-1]):
        with_loop[:, :, index] = cross_correlate_masked(
            arr1[:, :, index],
            arr2[:, :, index],
            m1[:, :, index],
            m2[:, :, index],
            axes=(0, 1),
            mode="same",
        )

    over_axes = cross_correlate_masked(
        arr1, arr2, m1, m2, axes=(0, 1), mode="same"
    )

    cp.testing.assert_array_almost_equal(with_loop, over_axes)


def test_cross_correlate_masked_autocorrelation_trivial_masks():
    """Masked normalized cross-correlation between identical arrays
    should reduce to an autocorrelation even with random masks."""
    # See random number generator for reproducible results
    np.random.seed(23)

    arr1 = cp.asarray(camera())
    arr2 = arr1.copy()

    # Random masks with 75% of pixels being valid
    m1 = np.random.choice([True, False], arr1.shape, p=[3 / 4, 1 / 4])
    m2 = np.random.choice([True, False], arr2.shape, p=[3 / 4, 1 / 4])
    m1 = cp.asarray(m1)
    m2 = cp.asarray(m2)

    xcorr = cross_correlate_masked(
        arr1, arr1, m1, m1, axes=(0, 1), mode="same", overlap_ratio=0
    ).real
    max_index = cp.unravel_index(cp.argmax(xcorr), xcorr.shape)

    # Autocorrelation should have maximum in center of array
    assert_almost_equal(float(xcorr.max()), 1)
    cp.testing.assert_array_equal(max_index, cp.asarray(arr1.shape) / 2)
