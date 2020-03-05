import functools

# TODO: remove need for these skimage imports
from skimage.util.dtype import dtype_range
from skimage._shared.utils import warn, check_shape_equality

import cupy
from cupyimg.scipy.stats import entropy
from cupyimg import numpy as cnp


__all__ = [
    "mean_squared_error",
    "normalized_mutual_information",
    "normalized_root_mse",
    "peak_signal_noise_ratio",
]


def _as_floats(image0, image1):
    """
    Promote im1, im2 to nearest appropriate floating point precision.
    """
    float_type = functools.reduce(
        cupy.promote_types, [image0.dtype, image1.dtype, cupy.float32]
    )
    image0 = cupy.asarray(image0, dtype=float_type)
    image1 = cupy.asarray(image1, dtype=float_type)
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
    return cupy.mean(diff * diff, dtype=cupy.float64)


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
        denom = cupy.sqrt(
            cupy.mean((image_true * image_true), dtype=cupy.float64)
        )
    elif normalization == "min-max":
        denom = image_true.max() - image_true.min()
    elif normalization == "mean":
        denom = image_true.mean()
    else:
        raise ValueError("Unsupported norm_type")
    return cupy.sqrt(mean_squared_error(image_true, image_test)) / denom


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
        true_min, true_max = cupy.min(image_true), cupy.max(image_true)
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
    return 10 * cupy.log10((data_range * data_range) / err)


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
    return cupy.pad(arr, pad_width=padding, mode="constant", constant_values=0)


def normalized_mutual_information(im_true, im_test, *, bins=100):
    r"""Compute the normalized mutual information.

    The normalized mutual information is given by::

    ..math::
        Y(A, B) = \frac{H(A) + H(B)}{H(A, B)}

    where :math:`H(X)` is the entropy,
    :math:`- \sum_{x \in X}{x \log x}.`

    It was proposed to be useful in registering images by Colin Studholme and
    colleagues [1]_. It ranges from 1 (perfectly uncorrelated image values)
    to 2 (perfectly correlated image values, whether positively or negatively).

    Parameters
    ----------
    im_true, im_test : ndarray
        Images to be compared. The two input images must have the same number
        of dimensions.
    bins : int or sequence of int, optional
        The number of bins along each axis of the joint histogram.

    Returns
    -------
    nmi : float
        The normalized mutual information between the two arrays, computed at
        the granularity given by ``bins``. Higher NMI implies more similar
        input images.

    Raises
    ------
    ValueError
        If the images don't have the same number of dimensions.

    Notes
    -----
    If the two input images are not the same shape, the smaller image is padded
    with zeros.

    References
    ----------
    .. [1] C. Studholme, D.L.G. Hill, & D.J. Hawkes (1999). An overlap
           invariant entropy measure of 3D medical image alignment.
           Pattern Recognition 32(1):71-86
           :DOI:`10.1016/S0031-3203(98)00091-0`
    """
    if im_true.ndim != im_test.ndim:
        raise ValueError(
            "NMI requires images of same number of dimensions. "
            "Got {}D for `im_true` and {}D for `im_test`.".format(
                im_true.ndim, im_test.ndim
            )
        )
    if im_true.shape != im_test.shape:
        max_shape = tuple(
            [max(s1, s2) for s1, s2 in zip(im_true.shape, im_test.shape)]
        )
        padded_true = _pad_to(im_true, max_shape)
        padded_test = _pad_to(im_test, max_shape)
    else:
        padded_true, padded_test = im_true, im_test

    hist, bin_edges = cnp.histogramdd(
        [cupy.ravel(padded_true), cupy.ravel(padded_test)],
        bins=bins,
        density=True,
    )

    H_im_true = entropy(cupy.sum(hist, axis=0))
    H_im_test = entropy(cupy.sum(hist, axis=1))
    H_true_test = entropy(cupy.ravel(hist))

    return (H_im_true + H_im_test) / H_true_test