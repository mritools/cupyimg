import cupy as cp
import numpy as np
from cupyimg.scipy import ndimage as ndi
from ..morphology import dilation, erosion, square
from ..util import img_as_float, view_as_windows
from ..color import gray2rgb


def _find_boundaries_subpixel(label_img):
    """See ``find_boundaries(..., mode='subpixel')``.

    Notes
    -----
    This function puts in an empty row and column between each *actual*
    row and column of the image, for a corresponding shape of ``2s - 1``
    for every image dimension of size ``s``. These "interstitial" rows
    and columns are filled as ``True`` if they separate two labels in
    `label_img`, ``False`` otherwise.

    I used ``view_as_windows`` to get the neighborhood of each pixel.
    Then I check whether there are two labels or more in that
    neighborhood.
    """
    ndim = label_img.ndim
    max_label = cp.iinfo(label_img.dtype).max

    label_img_expanded = cp.zeros(
        [(2 * s - 1) for s in label_img.shape], label_img.dtype
    )
    pixels = (slice(None, None, 2),) * ndim
    label_img_expanded[pixels] = label_img

    edges = cp.ones(label_img_expanded.shape, dtype=bool)
    edges[pixels] = False
    label_img_expanded[edges] = max_label
    windows = view_as_windows(
        cp.pad(label_img_expanded, 1, mode="constant", constant_values=0),
        (3,) * ndim,
    )

    boundaries = cp.zeros_like(edges)
    for index in np.ndindex(label_img_expanded.shape):
        if edges[index]:
            values = cp.unique(windows[index].ravel())
            if len(values) > 2:  # single value and max_label
                boundaries[index] = True
    return boundaries


def find_boundaries(label_img, connectivity=1, mode="thick", background=0):
    """Return bool array where boundaries between labeled regions are True.

    Parameters
    ----------
    label_img : array of int or bool
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:

        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.
    background : int, optional
        For modes 'inner' and 'outer', a definition of a background
        label is required. See `mode` for descriptions of these two.

    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    Examples
    --------
    >>> labels = cp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=cp.uint8)
    >>> find_boundaries(labels, mode='thick').astype(cp.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='inner').astype(cp.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='outer').astype(cp.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> labels_small = labels[::2, ::3]
    >>> labels_small
    array([[0, 0, 0, 0],
           [0, 0, 5, 0],
           [0, 1, 5, 0],
           [0, 0, 5, 0],
           [0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels_small, mode='subpixel').astype(cp.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1, 0],
           [0, 0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> bool_image = cp.array([[False, False, False, False, False],
    ...                        [False, False, False, False, False],
    ...                        [False, False,  True,  True,  True],
    ...                        [False, False,  True,  True,  True],
    ...                        [False, False,  True,  True,  True]], dtype=cp.bool_)
    >>> find_boundaries(bool_image)
    array([[False, False, False, False, False],
           [False, False,  True,  True,  True],
           [False,  True,  True,  True,  True],
           [False,  True,  True, False, False],
           [False,  True,  True, False, False]])
    """
    if label_img.dtype == "bool":
        label_img = label_img.astype(cp.uint8)
    ndim = label_img.ndim
    selem = ndi.generate_binary_structure(ndim, connectivity)
    if mode != "subpixel":
        boundaries = dilation(label_img, selem) != erosion(label_img, selem)
        if mode == "inner":
            foreground_image = label_img != background
            boundaries &= foreground_image
        elif mode == "outer":
            max_label = cp.iinfo(label_img.dtype).max
            background_image = label_img == background
            selem = ndi.generate_binary_structure(ndim, ndim)
            inverted_background = cp.array(label_img, copy=True)
            inverted_background[background_image] = max_label
            adjacent_objects = (
                dilation(label_img, selem)
                != erosion(inverted_background, selem)
            ) & ~background_image
            boundaries &= background_image | adjacent_objects
        return boundaries
    else:
        boundaries = _find_boundaries_subpixel(label_img)
        return boundaries


def mark_boundaries(
    image,
    label_img,
    color=(1, 1, 0),
    outline_color=None,
    mode="outer",
    background_label=0,
    *,
    order=3,
):
    """Return image with boundaries between labeled regions highlighted.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.
    label_img : (M, N) array of int
        Label array where regions are marked by different integer values.
    color : length-3 sequence, optional
        RGB color of boundaries in the output image.
    outline_color : length-3 sequence, optional
        RGB color surrounding boundaries in the output image. If None, no
        outline is drawn.
    mode : string in {'thick', 'inner', 'outer', 'subpixel'}, optional
        The mode for finding boundaries.
    background_label : int, optional
        Which label to consider background (this is only useful for
        modes ``inner`` and ``outer``).

    Additional Parameters
    ---------------------
    order : int
       The spline interpolation order to use when ``mode="subpixel"``.
       Unused by other modes.

    Returns
    -------
    marked : (M, N, 3) array of float
        An image in which the boundaries between labels are
        superimposed on the original image.

    See Also
    --------
    find_boundaries
    """
    marked = img_as_float(image, force_copy=True)
    if marked.ndim == 2:
        marked = gray2rgb(marked)
    if mode == "subpixel":
        # Here, we want to interpose an extra line of pixels between
        # each original line - except for the last axis which holds
        # the RGB information. ``ndi.zoom`` then performs the (cubic)
        # interpolation, filling in the values of the interposed pixels

        marked = ndi.zoom(
            marked,
            [2 - 1 / s for s in marked.shape[:-1]] + [1],
            mode="reflect",
            order=order,
        )
    boundaries = find_boundaries(
        label_img, mode=mode, background=background_label
    )
    if outline_color is not None:
        outlines = dilation(boundaries, square(3))
        marked[outlines] = outline_color
    marked[boundaries] = color
    return marked
