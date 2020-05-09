from itertools import combinations_with_replacement

import cupy as cp
import numpy as np

import cupyimg.numpy as cnp
from cupyimg.scipy import ndimage as ndi
from .. import img_as_float


def hessian_matrix(image, sigma=1, mode="constant", cval=0, order="rc"):
    """Compute Hessian matrix.

    The Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'rc' indicates the use of
        the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy)

    Returns
    -------
    Hrr : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hrc : ndarray
        Element of the Hessian matrix for each pixel in the input image.
    Hcc : ndarray
        Element of the Hessian matrix for each pixel in the input image.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.skimage.feature import hessian_matrix
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc')
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    """

    image = img_as_float(image)

    gaussian_filtered = ndi.gaussian_filter(
        image, sigma=sigma, mode=mode, cval=cval
    )

    gradients = cnp.gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == "rc":
        axes = reversed(axes)

    H_elems = [
        cnp.gradient(gradients[ax0], axis=ax1)
        for ax0, ax1 in combinations_with_replacement(axes, 2)
    ]

    return H_elems


def _hessian_matrix_image(H_elems):
    """Convert the upper-diagonal elements of the Hessian matrix to a matrix.

    Parameters
    ----------
    H_elems : list of array
        The upper-diagonal elements of the Hessian matrix, as returned by
        `hessian_matrix`.

    Returns
    -------
    hessian_image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the Hessian matrix corresponding to each coordinate.
    """
    image = H_elems[0]
    hessian_image = cp.zeros(image.shape + (image.ndim, image.ndim))
    for idx, (row, col) in enumerate(
        combinations_with_replacement(range(image.ndim), 2)
    ):
        hessian_image[..., row, col] = H_elems[idx]
        hessian_image[..., col, row] = H_elems[idx]
    return hessian_image


def _image_orthogonal_matrix22_eigvals(M00, M01, M11):
    l1 = (M00 + M11) / 2 + cp.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    l2 = (M00 + M11) / 2 - cp.sqrt(4 * M01 ** 2 + (M00 - M11) ** 2) / 2
    return l1, l2


def hessian_matrix_eigvals(H_elems):
    """Compute Eigenvalues of Hessian matrix.

    Parameters
    ----------
    H_elems : list of ndarray
        The upper-diagonal elements of the Hessian matrix, as returned
        by `hessian_matrix`.

    Returns
    -------
    eigs : ndarray
        The eigenvalues of the Hessian matrix, in decreasing order. The
        eigenvalues are the leading dimension. That is, ``eigs[i, j, k]``
        contains the ith-largest eigenvalue at position (j, k).

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = cp.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc')
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    if len(H_elems) == 3:  # Use fast Cython code for 2D
        eigvals = cp.stack(_image_orthogonal_matrix22_eigvals(*H_elems), axis=0)
    else:
        # TODO: grlee77: avoid host/device transfer.
        #                (currently cp.linalg.eigvalsh doesn't handle nd data)
        matrices = _hessian_matrix_image(H_elems).get()
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigvals = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigvals.ndim - 1))
        eigvals = np.transpose(eigvals, (eigvals.ndim - 1,) + leading_axes)
        eigvals = cp.asarray(eigvals)
    return eigvals
