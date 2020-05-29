import functools

# TODO: remove need for these skimage imports
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn, check_shape_equality

import cupy as cp


__all__ = [
    "mean_squared_error",
    "normalized_root_mse",
    "peak_signal_noise_ratio",
]


def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = functools.reduce(
        cp.promote_types, [image0.dtype, image1.dtype, cp.float32]
    )
    image0 = cp.asarray(image0, dtype=float_type)
    image1 = cp.asarray(image1, dtype=float_type)
    return image0, image1


def mean_squared_error(image0, image1):
    """
    Compute the mean-squared error between two images.

    Parameters
    ----------
    image0, image1 : ndarray
        Images.  Any dimensionality, must have same shape.

    Returns
    -------
    mse : float
        The mean-squared error (MSE) metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_mse`` to
        ``skimage.metrics.mean_squared_error``.

    """
    check_shape_equality(image0, image1)
    image0, image1 = _as_floats(image0, image1)
    diff = image0 - image1
    return cp.mean(diff * diff, dtype=cp.float64)


def normalized_root_mse(image_true, image_test, *, normalization="euclidean"):
    """
    Compute the normalized root mean-squared error (NRMSE) between two
    images.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    normalization : {'euclidean', 'min-max', 'mean'}, optional
        Controls the normalization method to use in the denominator of the
        NRMSE.  There is no standard method of normalization across the
        literature [1]_.  The methods available here are as follows:

        - 'euclidean' : normalize by the averaged Euclidean norm of
          ``im_true``::

              NRMSE = RMSE * sqrt(N) / || im_true ||

          where || . || denotes the Frobenius norm and ``N = im_true.size``.
          This result is equivalent to::

              NRMSE = || im_true - im_test || / || im_true ||.

        - 'min-max'   : normalize by the intensity range of ``im_true``.
        - 'mean'      : normalize by the mean of ``im_true``

    Returns
    -------
    nrmse : float
        The NRMSE metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_nrmse`` to
        ``skimage.metrics.normalized_root_mse``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Root-mean-square_deviation

    """
    check_shape_equality(image_true, image_test)
    image_true, image_test = _as_floats(image_true, image_test)

    # Ensure that both 'Euclidean' and 'euclidean' match
    normalization = normalization.lower()
    if normalization == "euclidean":
        denom = cp.sqrt(cp.mean((image_true * image_true), dtype=cp.float64))
    elif normalization == "min-max":
        denom = image_true.max() - image_true.min()
    elif normalization == "mean":
        denom = image_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return cp.sqrt(mean_squared_error(image_true, image_test)) / denom


def peak_signal_noise_ratio(image_true, image_test, *, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """
    check_shape_equality(image_true, image_test)

    if data_range is None:
        if image_true.dtype != image_test.dtype:
            warn(
                "Inputs have mismatched dtype.  Setting data_range based on "
                "im_true.",
                stacklevel=2,
            )
        dmin, dmax = dtype_range[image_true.dtype.type]
        true_min, true_max = cp.min(image_true), cp.max(image_true)
        if true_max > dmax or true_min < dmin:
            raise ValueError(
                "im_true has intensity values outside the range expected for "
                "its data type.  Please manually specify the data_range"
            )
        if true_min >= 0:
            # most common case (255 for uint8, 1 for float)
            data_range = dmax
        else:
            data_range = dmax - dmin

    image_true, image_test = _as_floats(image_true, image_test)

    err = mean_squared_error(image_true, image_test)
    return 10 * cp.log10((data_range * data_range) / err)


def _pad_to(arr, shape):
    """Pad an array with trailing zeros to a given target shape.

    Parameters
    ----------
    arr : ndarray
        The input array.
    shape : tuple
        The target shape.

    Returns
    -------
    padded : ndarray
        The padded array.

    Examples
    --------
    >>> _pad_to(np.ones((1, 1), dtype=int), (1, 3))
    array([[1, 0, 0]])
    """
    if not all(s >= i for s, i in zip(shape, arr.shape)):
        raise ValueError(
            "Target shape must be strictly greater " "than input shape."
        )
    padding = [(0, s - i) for s, i in zip(shape, arr.shape)]
    return cp.pad(arr, pad_width=padding, mode="constant", constant_values=0)
