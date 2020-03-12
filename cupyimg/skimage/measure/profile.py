import math
from warnings import warn

import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as ndi
import cupyimg.numpy as cnp


def profile_line(
    image,
    src,
    dst,
    linewidth=1,
    order=None,
    mode="constant",
    cval=0.0,
    *,
    reduce_func=cp.mean,
):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    image : numeric array, shape (M, N[, C])
        The image, either grayscale (2D array) or multichannel
        (3D array, where the final axis contains the channel
        information).
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line. The destination point is *included*
        in the profile, in contrast to standard numpy indexing.
    linewidth : int, optional
        Width of the scan, perpendicular to the line
    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation, default is 0 if
        image.dtype is bool and 1 otherwise. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.
    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        How to compute any values falling outside of the image.
    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.

    Examples
    --------
    >>> x = cp.asarray([[1, 1, 1, 2, 2, 2]])
    >>> img = cp.vstack([cp.zeros_like(x), x, x, x, cp.zeros_like(x)])
    >>> img
    array([[0, 0, 0, 0, 0, 0],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [1, 1, 1, 2, 2, 2],
           [0, 0, 0, 0, 0, 0]])
    >>> profile_line(img, (2, 1), (2, 4))
    array([ 1.,  1.,  2.,  2.])
    >>> profile_line(img, (1, 0), (1, 6), cval=4)
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  4.])

    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    For example:

    >>> profile_line(img, (1, 0), (1, 6))  # The final point is out of bounds
    array([ 1.,  1.,  1.,  2.,  2.,  2.,  0.])
    >>> profile_line(img, (1, 0), (1, 5))  # This accesses the full first row
    array([ 1.,  1.,  1.,  2.,  2.,  2.])
    """
    if order is None:
        order = 0 if image.dtype == bool else 1

    if image.dtype == bool and order != 0:
        warn(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitely "
            "cast input image to another data type. Starting from version "
            "0.19 a ValueError will be raised instead of this warning.",
            FutureWarning,
            stacklevel=2,
        )

    perp_lines = _line_profile_coordinates(src, dst, linewidth=linewidth)
    if image.ndim == 3:
        pixels = [
            ndi.map_coordinates(
                image[..., i],
                perp_lines,
                prefilter=order > 1,
                order=order,
                mode=mode,
                cval=cval,
            )
            for i in range(image.shape[2])
        ]
        pixels = cp.transpose(cp.asarray(pixels), (1, 2, 0))
    else:
        pixels = ndi.map_coordinates(
            image,
            perp_lines,
            prefilter=order > 1,
            order=order,
            mode=mode,
            cval=cval,
        )
    # The outputted array with reduce_func=None gives an array where the
    # row values (axis=1) are flipped. Here, we make this consistent.
    pixels = np.flip(pixels, axis=1)

    if reduce_func is None:
        intensities = pixels
    else:
        try:
            intensities = reduce_func(pixels, axis=1)
        except TypeError:  # function doesn't allow axis kwarg
            intensities = cnp.apply_along_axis(reduce_func, arr=pixels, axis=1)

    return intensities


def _line_profile_coordinates(src, dst, linewidth=1):
    """Return the coordinates of the profile of an image along a scan line.

    Parameters
    ----------
    src : 2-tuple of numeric scalar (float or int)
        The start point of the scan line.
    dst : 2-tuple of numeric scalar (float or int)
        The end point of the scan line.
    linewidth : int, optional
        Width of the scan, perpendicular to the line

    Returns
    -------
    coords : array, shape (2, N, C), float
        The coordinates of the profile along the scan line. The length of the
        profile is the ceil of the computed length of the scan line.

    Notes
    -----
    This is a utility method meant to be used internally by skimage functions.
    The destination point is included in the profile, in contrast to
    standard numpy indexing.
    """
    src_row, src_col = src
    dst_row, dst_col = dst
    d_row, d_col = (d - s for d, s in zip(dst, src))
    theta = math.atan2(d_row, d_col)

    length = math.ceil(math.hypot(d_row, d_col) + 1)
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = cp.linspace(src_col, dst_col, length)
    line_row = cp.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * cp.sin(-theta) / 2
    row_width = (linewidth - 1) * cp.cos(theta) / 2
    perp_rows = cp.stack(
        [
            cp.linspace(row_i - row_width, row_i + row_width, linewidth)
            for row_i in line_row
        ]
    )
    perp_cols = cp.stack(
        [
            cp.linspace(col_i - col_width, col_i + col_width, linewidth)
            for col_i in line_col
        ]
    )
    return cp.stack([perp_rows, perp_cols])
