import operator

import cupy
import numpy

from . import _ni_support
from . import filters
from ._kernels.morphology import _get_erode_kernel


__all__ = [
    "binary_erosion",
    "binary_dilation",
    "binary_opening",
    "binary_closing",
    "binary_hit_or_miss",
    "binary_propagation",
    "binary_fill_holes",
    "grey_erosion",
    "grey_dilation",
    "grey_opening",
    "grey_closing",
    "morphological_gradient",
    "morphological_laplace",
    "white_tophat",
    "black_tophat",
    "generate_binary_structure",
    "iterate_structure",
]

# TODO: grlee77: 'distance_transform_bf', 'distance_transform_cdt',
#                'distance_transform_edt'
#
# see the following MIT-licensed code for euclidean distance transform
# (PBA algorithm):
#     https://github.com/orzzzjq/Parallel-Banding-Algorithm-plus
# There is also a Jump-Flooding Centroidal Voronoi Code (BSD 3-clause) here:


_brute_force_implemented = (
    False  # TODO: set to True if/when faster approach has been implemented
)


def _center_is_true(structure, origin):
    structure = cupy.asarray(structure)
    coor = tuple([oo + ss // 2 for ss, oo in zip(structure.shape, origin)])
    return bool(structure[coor])


def iterate_structure(structure, iterations, origin=None):
    """
    Iterate a structure by dilating it with itself.

    Parameters
    ----------
    structure : array_like
       Structuring element (an array of bools, for example), to be dilated with
       itself.
    iterations : int
       number of dilations performed on the structure with itself
    origin : optional
        If origin is None, only the iterated structure is returned. If
        not, a tuple of the iterated structure and the modified origin is
        returned.

    Returns
    -------
    iterate_structure : ndarray of bools
        A new structuring element obtained by dilating `structure`
        (`iterations` - 1) times with itself.

    See also
    --------
    generate_binary_structure

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct.astype(int)
    array([[0, 1, 0],
           [1, 1, 1],
           [0, 1, 0]])
    >>> ndimage.iterate_structure(struct, 2).astype(int)
    array([[0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0]])
    >>> ndimage.iterate_structure(struct, 3).astype(int)
    array([[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0]])

    """
    structure = cupy.asarray(structure)
    if iterations < 2:
        return structure.copy()
    ni = iterations - 1
    shape = [ii + ni * (ii - 1) for ii in structure.shape]
    pos = [ni * (structure.shape[ii] // 2) for ii in range(len(shape))]
    slc = tuple(
        slice(pos[ii], pos[ii] + structure.shape[ii], None)
        for ii in range(len(shape))
    )
    out = cupy.zeros(shape, bool)
    out[slc] = structure != 0
    out = binary_dilation(out, structure, iterations=ni)
    if origin is None:
        return out
    else:
        origin = _ni_support._normalize_sequence(origin, structure.ndim)
        origin = [iterations * o for o in origin]
        return out, origin


def generate_binary_structure(rank, connectivity):
    """
    Generate a binary structure for binary morphological operations.

    Parameters
    ----------
    rank : int
         Number of dimensions of the array to which the structuring element
         will be applied, as returned by `np.ndim`.
    connectivity : int
         `connectivity` determines which elements of the output array belong
         to the structure, i.e., are considered as neighbors of the central
         element. Elements up to a squared distance of `connectivity` from
         the center are considered neighbors. `connectivity` may range from 1
         (no diagonal elements are neighbors) to `rank` (all elements are
         neighbors).

    Returns
    -------
    output : ndarray of bools
         Structuring element which may be used for binary morphological
         operations, with `rank` dimensions and all dimensions equal to 3.

    See also
    --------
    iterate_structure, binary_dilation, binary_erosion

    Notes
    -----
    `generate_binary_structure` can only create structuring elements with
    dimensions equal to 3, i.e., minimal dimensions. For larger structuring
    elements, that are useful e.g., for eroding large objects, one may either
    use `iterate_structure`, or create directly custom arrays with
    numpy functions such as `cupy.ones`.

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> struct = ndimage.generate_binary_structure(2, 1)
    >>> struct
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> a = cp.zeros((5,5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> b = ndimage.binary_dilation(a, structure=struct).astype(a.dtype)
    >>> b
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(b, structure=struct).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])
    >>> struct = ndimage.generate_binary_structure(2, 2)
    >>> struct
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> struct = ndimage.generate_binary_structure(3, 1)
    >>> struct # no diagonal elements
    array([[[False, False, False],
            [False,  True, False],
            [False, False, False]],
           [[False,  True, False],
            [ True,  True,  True],
            [False,  True, False]],
           [[False, False, False],
            [False,  True, False],
            [False, False, False]]], dtype=bool)

    """
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return cupy.asarray(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    # small, so should be faster to leave the computations above in NumPy and
    # just transfer to the GPU at the end
    return cupy.asarray(output <= connectivity)


def _binary_erosion(
    input,
    structure,
    iterations,
    mask,
    output,
    border_value,
    origin,
    invert,
    brute_force=True,
):
    try:
        iterations = operator.index(iterations)
    except TypeError:
        raise TypeError("iterations parameter should be an integer")

    input = cupy.asarray(input)
    if cupy.iscomplexobj(input):
        raise TypeError("Complex type not supported")
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    else:
        structure = cupy.asarray(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError("structure and input must have same dimensionality")
    if not structure.flags.c_contiguous:
        structure = structure.copy()
    # if functools.reduce(operator.mul, structure.shape) < 1:
    if structure.size < 1:
        raise RuntimeError("structure must not be empty")

    if mask is not None:
        mask = cupy.asarray(mask)
        if mask.shape != input.shape:
            raise RuntimeError("mask and input must have equal sizes")
        masked = True
    else:
        masked = False
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    center_is_true = _center_is_true(structure, origin)
    if isinstance(output, cupy.ndarray):
        if cupy.iscomplexobj(output):
            raise TypeError("Complex output type not supported")
    else:
        output = bool
    output = _ni_support._get_output(output, input)
    if structure.ndim == 0:
        # kernel doesn't handle ndim=0, so special case it here
        if float(structure):
            output[...] = cupy.asarray(input, dtype=bool)
        else:
            output[...] = ~cupy.asarray(input, dtype=bool)
        return output
    origin = tuple(origin)
    erode_kernel = _get_erode_kernel(
        input.shape,
        structure.shape,
        origin,
        center_is_true,
        border_value,
        invert,
        masked,
    )

    # TODO: grlee77: current indexing via x_data, etc. requires
    #       C contiguous input arrays.
    input = cupy.ascontiguousarray(input)
    structure = cupy.ascontiguousarray(structure)
    if masked:
        mask = cupy.ascontiguousarray(mask)

    if iterations == 1:
        if masked:
            return erode_kernel(input, structure, mask, output)
        else:
            return erode_kernel(input, structure, output)
    elif center_is_true and not brute_force:
        raise NotImplementedError(
            "only brute_force iteration has been implemented"
        )
    else:
        tmp_in = cupy.empty_like(input, dtype=bool)
        tmp_out = output
        if iterations >= 1 and not iterations & 1:
            tmp_in, tmp_out = tmp_out, tmp_in
        if masked:
            tmp_out = erode_kernel(input, structure, mask, tmp_out)
        else:
            tmp_out = erode_kernel(input, structure, tmp_out)
        # TODO: the kernel doesn't return the changed status, so determine it
        #       here
        changed = not (tmp_in == tmp_out).all()
        ii = 1
        while ii < iterations or ((iterations < 1) and changed):
            tmp_in, tmp_out = tmp_out, tmp_in
            if masked:
                tmp_out = erode_kernel(tmp_in, structure, mask, tmp_out)
            else:
                tmp_out = erode_kernel(tmp_in, structure, tmp_out)
            changed = not (tmp_in == tmp_out).all()
            print(f"ii={ii}, changed={changed}, shape={structure.shape}")
            ii += 1
        return tmp_out


def binary_erosion(
    input,
    structure=None,
    iterations=1,
    mask=None,
    output=None,
    border_value=0,
    origin=0,
    brute_force=False,
):
    """
    Multidimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Parameters
    ----------
    input : array_like
        Binary image to be eroded. Non-zero (True) elements form
        the subset to be eroded.
    structure : array_like, optional
        Structuring element used for the erosion. Non-zero elements are
        considered True. If no structuring element is provided, an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The erosion is repeated `iterations` times (one, by default).
        If iterations is less than 1, the erosion is repeated until the
        result does not change anymore.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (eroded) in
        the current iteration; if True all pixels are considered as candidates
        for erosion, regardless of what happened in the previous iteration.
        False by default.

    Returns
    -------
    binary_erosion : ndarray of bools
        Erosion of the input by the structuring element.

    See also
    --------
    grey_erosion, binary_dilation, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Erosion [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for shrinking the shapes in an image. The binary
    erosion of an image by a structuring element is the locus of the points
    where a superimposition of the structuring element centered on the point
    is entirely contained in the set of non-zero elements of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[1:6, 2:5] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_erosion(a).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> #Erosion removes objects smaller than the structure
    >>> ndimage.binary_erosion(a, structure=cp.ones((5,5))).astype(a.dtype)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    return _binary_erosion(
        input,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
        0,
        brute_force,
    )


def binary_dilation(
    input,
    structure=None,
    iterations=1,
    mask=None,
    output=None,
    border_value=0,
    origin=0,
    brute_force=False,
):
    """
    Multidimensional binary dilation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be dilated. Non-zero (True) elements form
        the subset to be dilated.
    structure : array_like, optional
        Structuring element used for the dilation. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one.
    iterations : int, optional
        The dilation is repeated `iterations` times (one, by default).
        If iterations is less than 1, the dilation is repeated until the
        result does not change anymore. Only an integer of iterations is
        accepted.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated (dilated)
        in the current iteration; if True all pixels are considered as
        candidates for dilation, regardless of what happened in the previous
        iteration. False by default.

    Returns
    -------
    binary_dilation : ndarray of bools
        Dilation of the input by the structuring element.

    See also
    --------
    grey_dilation, binary_erosion, binary_closing, binary_opening,
    generate_binary_structure

    Notes
    -----
    Dilation [1]_ is a mathematical morphology operation [2]_ that uses a
    structuring element for expanding the shapes in an image. The binary
    dilation of an image by a structuring element is the locus of the points
    covered by the structuring element, when its center lies within the
    non-zero points of the image.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((5, 5))
    >>> a[2, 2] = 1
    >>> a
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a)
    array([[False, False, False, False, False],
           [False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False],
           [False, False, False, False, False]], dtype=bool)
    >>> ndimage.binary_dilation(a).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # 3x3 structuring element with connectivity 1, used by default
    >>> struct1 = ndimage.generate_binary_structure(2, 1)
    >>> struct1
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # 3x3 structuring element with connectivity 2
    >>> struct2 = ndimage.generate_binary_structure(2, 2)
    >>> struct2
    array([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)
    >>> ndimage.binary_dilation(a, structure=struct1).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct2).astype(a.dtype)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> ndimage.binary_dilation(a, structure=struct1,\\
    ... iterations=2).astype(a.dtype)
    array([[ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  1.,  1.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.]])

    """
    input = cupy.asarray(input)
    if structure is None:
        structure = generate_binary_structure(input.ndim, 1)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    structure = cupy.asarray(structure)
    structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if not structure.shape[ii] & 1:
            origin[ii] -= 1

    return _binary_erosion(
        input,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
        1,
        brute_force,
    )


def binary_opening(
    input,
    structure=None,
    iterations=1,
    output=None,
    origin=0,
    mask=None,
    border_value=0,
    brute_force=False,
):
    """
    Multidimensional binary opening with the given structuring element.

    The *opening* of an input image by a structuring element is the
    *dilation* of the *erosion* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be opened. Non-zero (True) elements form
        the subset to be opened.
    structure : array_like, optional
        Structuring element used for the opening. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The erosion step of the opening, then the dilation step are each
        repeated `iterations` times (one, by default). If `iterations` is
        less than 1, each operation is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.

        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.

        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true all pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.

        .. versionadded:: 1.1.0

    Returns
    -------
    binary_opening : ndarray of bools
        Opening of the input by the structuring element.

    See also
    --------
    grey_opening, binary_closing, binary_erosion, binary_dilation,
    generate_binary_structure

    Notes
    -----
    *Opening* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of an erosion and a dilation of the
    input with the same structuring element. Opening, therefore, removes
    objects smaller than the structuring element.

    Together with *closing* (`binary_closing`), opening can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Opening_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((5,5), dtype=int)
    >>> a[1:4, 1:4] = 1; a[4, 4] = 1
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 1]])
    >>> # Opening removes small objects
    >>> ndimage.binary_opening(a, structure=cp.ones((3,3))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening can also smooth corners
    >>> ndimage.binary_opening(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> # Opening is the dilation of the erosion of the input
    >>> ndimage.binary_erosion(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> ndimage.binary_dilation(ndimage.binary_erosion(a)).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])

    """
    input = cupy.asarray(input)
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)

    tmp = binary_erosion(
        input,
        structure,
        iterations,
        mask,
        None,
        border_value,
        origin,
        brute_force,
    )
    return binary_dilation(
        tmp,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
        brute_force,
    )


def binary_closing(
    input,
    structure=None,
    iterations=1,
    output=None,
    origin=0,
    mask=None,
    border_value=0,
    brute_force=False,
):
    """
    Multidimensional binary closing with the given structuring element.

    The *closing* of an input image by a structuring element is the
    *erosion* of the *dilation* of the image by the structuring element.

    Parameters
    ----------
    input : array_like
        Binary array_like to be closed. Non-zero (True) elements form
        the subset to be closed.
    structure : array_like, optional
        Structuring element used for the closing. Non-zero elements are
        considered True. If no structuring element is provided an element
        is generated with a square connectivity equal to one (i.e., only
        nearest neighbors are connected to the center, diagonally-connected
        elements are not considered neighbors).
    iterations : int, optional
        The dilation step of the closing, then the erosion step are each
        repeated `iterations` times (one, by default). If iterations is
        less than 1, each operations is repeated until the result does
        not change anymore. Only an integer of iterations is accepted.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.
    mask : array_like, optional
        If a mask is given, only those elements with a True value at
        the corresponding mask element are modified at each iteration.

        .. versionadded:: 1.1.0
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.

        .. versionadded:: 1.1.0
    brute_force : boolean, optional
        Memory condition: if False, only the pixels whose value was changed in
        the last iteration are tracked as candidates to be updated in the
        current iteration; if true al pixels are considered as candidates for
        update, regardless of what happened in the previous iteration.
        False by default.

        .. versionadded:: 1.1.0

    Returns
    -------
    binary_closing : ndarray of bools
        Closing of the input by the structuring element.

    See also
    --------
    grey_closing, binary_opening, binary_dilation, binary_erosion,
    generate_binary_structure

    Notes
    -----
    *Closing* [1]_ is a mathematical morphology operation [2]_ that
    consists in the succession of a dilation and an erosion of the
    input with the same structuring element. Closing therefore fills
    holes smaller than the structuring element.

    Together with *opening* (`binary_opening`), closing can be used for
    noise removal.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((5,5), dtype=int)
    >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Closing removes small holes
    >>> ndimage.binary_closing(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Closing is the erosion of the dilation of the input
    >>> ndimage.binary_dilation(a).astype(int)
    array([[0, 1, 1, 1, 0],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [0, 1, 1, 1, 0]])
    >>> ndimage.binary_erosion(ndimage.binary_dilation(a)).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])


    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # In addition to removing holes, closing can also
    >>> # coarsen boundaries with fine hollows.
    >>> ndimage.binary_closing(a).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_closing(a, structure=cp.ones((2,2))).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    input = cupy.asarray(input)
    if structure is None:
        rank = input.ndim
        structure = generate_binary_structure(rank, 1)

    tmp = binary_dilation(
        input,
        structure,
        iterations,
        mask,
        None,
        border_value,
        origin,
        brute_force,
    )
    return binary_erosion(
        tmp,
        structure,
        iterations,
        mask,
        output,
        border_value,
        origin,
        brute_force,
    )


def binary_hit_or_miss(
    input,
    structure1=None,
    structure2=None,
    output=None,
    origin1=0,
    origin2=None,
):
    """
    Multidimensional binary hit-or-miss transform.

    The hit-or-miss transform finds the locations of a given pattern
    inside the input image.

    Parameters
    ----------
    input : array_like (cast to booleans)
        Binary image where a pattern is to be detected.
    structure1 : array_like (cast to booleans), optional
        Part of the structuring element to be fitted to the foreground
        (non-zero elements) of `input`. If no value is provided, a
        structure of square connectivity 1 is chosen.
    structure2 : array_like (cast to booleans), optional
        Second part of the structuring element that has to miss completely
        the foreground. If no value is provided, the complementary of
        `structure1` is taken.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin1 : int or tuple of ints, optional
        Placement of the first part of the structuring element `structure1`,
        by default 0 for a centered structure.
    origin2 : int or tuple of ints, optional
        Placement of the second part of the structuring element `structure2`,
        by default 0 for a centered structure. If a value is provided for
        `origin1` and not for `origin2`, then `origin2` is set to `origin1`.

    Returns
    -------
    binary_hit_or_miss : ndarray
        Hit-or-miss transform of `input` with the given structuring
        element (`structure1`, `structure2`).

    See also
    --------
    binary_erosion

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Hit-or-miss_transform

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[1, 1] = 1; a[2:4, 2:4] = 1; a[4:6, 4:6] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> structure1 = cp.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])
    >>> structure1
    array([[1, 0, 0],
           [0, 1, 1],
           [0, 1, 1]])
    >>> # Find the matches of structure1 in the array a
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # Change the origin of the filter
    >>> # origin1=1 is equivalent to origin1=(1,1) here
    >>> ndimage.binary_hit_or_miss(a, structure1=structure1,\\
    ... origin1=1).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    input = cupy.asarray(input)
    if structure1 is None:
        structure1 = generate_binary_structure(input.ndim, 1)
    if structure2 is None:
        structure2 = cupy.logical_not(structure1)
    origin1 = _ni_support._normalize_sequence(origin1, input.ndim)
    if origin2 is None:
        origin2 = origin1
    else:
        origin2 = _ni_support._normalize_sequence(origin2, input.ndim)

    tmp1 = _binary_erosion(
        input, structure1, 1, None, None, 0, origin1, 0, False
    )
    inplace = isinstance(output, cupy.ndarray)
    result = _binary_erosion(
        input, structure2, 1, None, output, 0, origin2, 1, False
    )
    if inplace:
        cupy.logical_not(output, output)
        cupy.logical_and(tmp1, output, output)
    else:
        cupy.logical_not(result, result)
        return cupy.logical_and(tmp1, result)


def binary_propagation(
    input, structure=None, mask=None, output=None, border_value=0, origin=0
):
    """
    Multidimensional binary propagation with the given structuring element.

    Parameters
    ----------
    input : array_like
        Binary image to be propagated inside `mask`.
    structure : array_like, optional
        Structuring element used in the successive dilations. The output
        may depend on the structuring element, especially if `mask` has
        several connex components. If no structuring element is
        provided, an element is generated with a squared connectivity equal
        to one.
    mask : array_like, optional
        Binary mask defining the region into which `input` is allowed to
        propagate.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    border_value : int (cast to 0 or 1), optional
        Value at the border in the output array.
    origin : int or tuple of ints, optional
        Placement of the filter, by default 0.

    Returns
    -------
    binary_propagation : ndarray
        Binary propagation of `input` inside `mask`.

    Notes
    -----
    This function is functionally equivalent to calling binary_dilation
    with the number of iterations less than one: iterative dilation until
    the result does not change anymore.

    The succession of an erosion and propagation inside the original image
    can be used instead of an *opening* for deleting small objects while
    keeping the contours of larger objects untouched.

    References
    ----------
    .. [1] http://cmm.ensmp.fr/~serra/cours/pdf/en/ch6en.pdf, slide 15.
    .. [2] I.T. Young, J.J. Gerbrands, and L.J. van Vliet, "Fundamentals of
        image processing", 1998
        ftp://qiftp.tudelft.nl/DIPimage/docs/FIP2.3.pdf

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> input = cp.zeros((8, 8), dtype=int)
    >>> input[2, 2] = 1
    >>> mask = cp.zeros((8, 8), dtype=int)
    >>> mask[1:4, 1:4] = mask[4, 4]  = mask[6:8, 6:8] = 1
    >>> input
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> mask
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 0, 1, 1]])
    >>> ndimage.binary_propagation(input, mask=mask).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_propagation(input, mask=mask,\\
    ... structure=cp.ones((3,3))).astype(int)
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]])

    >>> # Comparison between opening and erosion+propagation
    >>> a = cp.zeros((6,6), dtype=int)
    >>> a[2:5, 2:5] = 1; a[0, 0] = 1; a[5, 5] = 1
    >>> a
    array([[1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1]])
    >>> ndimage.binary_opening(a).astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> b = ndimage.binary_erosion(a)
    >>> b.astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]])
    >>> ndimage.binary_propagation(b, mask=a).astype(int)
    array([[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0]])

    """
    return binary_dilation(
        input,
        structure,
        -1,
        mask,
        output,
        border_value,
        origin,
        brute_force=(not _brute_force_implemented),
    )


def binary_fill_holes(input, structure=None, output=None, origin=0):
    """
    Fill the holes in binary objects.


    Parameters
    ----------
    input : array_like
        N-D binary array with holes to be filled
    structure : array_like, optional
        Structuring element used in the computation; large-size elements
        make computations faster but may miss holes separated from the
        background by thin regions. The default element (with a square
        connectivity equal to one) yields the intuitive result where all
        holes in the input have been filled.
    output : ndarray, optional
        Array of the same shape as input, into which the output is placed.
        By default, a new array is created.
    origin : int, tuple of ints, optional
        Position of the structuring element.

    Returns
    -------
    out : ndarray
        Transformation of the initial image `input` where holes have been
        filled.

    See also
    --------
    binary_dilation, binary_propagation, label

    Notes
    -----
    The algorithm used in this function consists in invading the complementary
    of the shapes in `input` from the outer boundary of the image,
    using binary dilations. Holes are not connected to the boundary and are
    therefore not invaded. The result is the complementary subset of the
    invaded region.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology


    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((5, 5), dtype=int)
    >>> a[1:4, 1:4] = 1
    >>> a[2,2] = 0
    >>> a
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> ndimage.binary_fill_holes(a).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])
    >>> # Too big structuring element
    >>> ndimage.binary_fill_holes(a, structure=cp.ones((5,5))).astype(int)
    array([[0, 0, 0, 0, 0],
           [0, 1, 1, 1, 0],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 1, 0],
           [0, 0, 0, 0, 0]])

    """
    mask = cupy.logical_not(input)
    tmp = cupy.zeros(mask.shape, bool)
    inplace = isinstance(output, cupy.ndarray)
    if inplace:
        binary_dilation(
            tmp,
            structure,
            -1,
            mask,
            output,
            1,
            origin,
            brute_force=(not _brute_force_implemented),
        )
        cupy.logical_not(output, output)
    else:
        output = binary_dilation(
            tmp,
            structure,
            -1,
            mask,
            None,
            1,
            origin,
            brute_force=(not _brute_force_implemented),
        )
        cupy.logical_not(output, output)
        return output


def grey_erosion(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Calculate a greyscale erosion, using either a structuring element,
    or a footprint corresponding to a flat structuring element.

    Grayscale erosion is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a minimum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale erosion is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        erosion. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale erosion. Non-zero values give the set of
        neighbors of the center over which the minimum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale erosion. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the erosion may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    output : ndarray
        Grayscale erosion of `input`.

    See also
    --------
    binary_erosion, grey_dilation, grey_opening, grey_closing
    generate_binary_structure, minimum_filter

    Notes
    -----
    The grayscale erosion of an image input by a structuring element s defined
    over a domain E is given by:

    (input+s)(x) = min {input(y) - s(x-y), for y in E}

    In particular, for structuring elements defined as
    s(y) = 0 for y in E, the grayscale erosion computes the minimum of the
    input image inside a sliding window defined by E.

    Grayscale erosion [1]_ is a *mathematical morphology* operation [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Erosion_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[1:6, 1:6] = 3
    >>> a[4,4] = 2; a[2,3] = 1
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 1, 3, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 3, 3, 3, 2, 3, 0],
           [0, 3, 3, 3, 3, 3, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_erosion(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> footprint = ndimage.generate_binary_structure(2, 1)
    >>> footprint
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> # Diagonally-connected elements are not considered neighbors
    >>> ndimage.grey_erosion(a, size=(3,3), footprint=footprint)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 3, 1, 2, 0, 0],
           [0, 0, 3, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """
    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")

    return filters._min_or_max_filter(
        input,
        size,
        footprint,
        structure,
        output,
        mode,
        cval,
        origin,
        True,
        morphology_mode=True,
    )


def grey_dilation(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Calculate a greyscale dilation, using either a structuring element,
    or a footprint corresponding to a flat structuring element.

    Grayscale dilation is a mathematical morphology operation. For the
    simple case of a full and flat structuring element, it can be viewed
    as a maximum filter over a sliding window.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale dilation is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        dilation. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale dilation. Non-zero values give the set of
        neighbors of the center over which the maximum is chosen.
    structure : array of ints, optional
        Structuring element used for the grayscale dilation. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the dilation may be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_dilation : ndarray
        Grayscale dilation of `input`.

    See also
    --------
    binary_dilation, grey_erosion, grey_closing, grey_opening
    generate_binary_structure, maximum_filter

    Notes
    -----
    The grayscale dilation of an image input by a structuring element s defined
    over a domain E is given by:

    (input+s)(x) = max {input(y) + s(x-y), for y in E}

    In particular, for structuring elements defined as
    s(y) = 0 for y in E, the grayscale dilation computes the maximum of the
    input image inside a sliding window defined by E.

    Grayscale dilation [1]_ is a *mathematical morphology* operation [2]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> a[4,4] = 2; a[2,3] = 3
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, footprint=cp.ones((3,3)))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> s = ndimage.generate_binary_structure(2,1)
    >>> s
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]], dtype=bool)
    >>> ndimage.grey_dilation(a, footprint=s)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 1, 3, 2, 1, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.grey_dilation(a, size=(3,3), structure=cp.ones((3,3)))
    array([[1, 1, 1, 1, 1, 1, 1],
           [1, 2, 4, 4, 4, 2, 1],
           [1, 2, 4, 4, 4, 2, 1],
           [1, 2, 4, 4, 4, 3, 1],
           [1, 2, 2, 3, 3, 3, 1],
           [1, 2, 2, 3, 3, 3, 1],
           [1, 1, 1, 1, 1, 1, 1]])

    """

    if size is None and footprint is None and structure is None:
        raise ValueError("size, footprint, or structure must be specified")

    if structure is not None:
        structure = cupy.asarray(structure)
        structure = structure[tuple([slice(None, None, -1)] * structure.ndim)]
    if footprint is not None:
        footprint = cupy.asarray(footprint)
        footprint = footprint[tuple([slice(None, None, -1)] * footprint.ndim)]

    input = cupy.asarray(input)
    origin = _ni_support._normalize_sequence(origin, input.ndim)
    for ii in range(len(origin)):
        origin[ii] = -origin[ii]
        if footprint is not None:
            sz = footprint.shape[ii]
        elif structure is not None:
            sz = structure.shape[ii]
        elif cupy.isscalar(size):
            sz = size
        else:
            sz = size[ii]
        if not sz & 1:
            origin[ii] -= 1

    return filters._min_or_max_filter(
        input,
        size,
        footprint,
        structure,
        output,
        mode,
        cval,
        origin,
        False,
        morphology_mode=True,
    )


def grey_opening(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional grayscale opening.

    A grayscale opening consists in the succession of a grayscale erosion,
    and a grayscale dilation.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale opening is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        opening. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale opening.
    structure : array of ints, optional
        Structuring element used for the grayscale opening. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the opening may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_opening : ndarray
        Result of the grayscale opening of `input` with `structure`.

    See also
    --------
    binary_opening, grey_dilation, grey_erosion, grey_closing
    generate_binary_structure

    Notes
    -----
    The action of a grayscale opening with a flat structuring element amounts
    to smoothen high local maxima, whereas binary opening erases small objects.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.arange(36).reshape((6,6))
    >>> a[3, 3] = 50
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20, 50, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_opening(a, size=(3,3))
    array([[ 0,  1,  2,  3,  4,  4],
           [ 6,  7,  8,  9, 10, 10],
           [12, 13, 14, 15, 16, 16],
           [18, 19, 20, 22, 22, 22],
           [24, 25, 26, 27, 28, 28],
           [24, 25, 26, 27, 28, 28]])
    >>> # Note that the local maximum a[3,3] has disappeared

    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            "ignoring size because footprint is set", UserWarning, stacklevel=2
        )

    tmp = grey_erosion(
        input, size, footprint, structure, None, mode, cval, origin
    )
    return grey_dilation(
        tmp, size, footprint, structure, output, mode, cval, origin
    )


def grey_closing(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional grayscale closing.

    A grayscale closing consists in the succession of a grayscale dilation,
    and a grayscale erosion.

    Parameters
    ----------
    input : array_like
        Array over which the grayscale closing is to be computed.
    size : tuple of ints
        Shape of a flat and full structuring element used for the grayscale
        closing. Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the grayscale closing.
    structure : array of ints, optional
        Structuring element used for the grayscale closing. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the closing may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    grey_closing : ndarray
        Result of the grayscale closing of `input` with `structure`.

    See also
    --------
    binary_closing, grey_dilation, grey_erosion, grey_opening,
    generate_binary_structure

    Notes
    -----
    The action of a grayscale closing with a flat structuring element amounts
    to smoothen deep local minima, whereas binary closing fills small holes.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.arange(36).reshape((6,6))
    >>> a[3,3] = 0
    >>> a
    array([[ 0,  1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10, 11],
           [12, 13, 14, 15, 16, 17],
           [18, 19, 20,  0, 22, 23],
           [24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35]])
    >>> ndimage.grey_closing(a, size=(3,3))
    array([[ 7,  7,  8,  9, 10, 11],
           [ 7,  7,  8,  9, 10, 11],
           [13, 13, 14, 15, 16, 17],
           [19, 19, 20, 20, 22, 23],
           [25, 25, 26, 27, 28, 29],
           [31, 31, 32, 33, 34, 35]])
    >>> # Note that the local minimum a[3,3] has disappeared

    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            "ignoring size because footprint is set", UserWarning, stacklevel=2
        )

    tmp = grey_dilation(
        input, size, footprint, structure, None, mode, cval, origin
    )
    return grey_erosion(
        tmp, size, footprint, structure, output, mode, cval, origin
    )


def morphological_gradient(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional morphological gradient.

    The morphological gradient is calculated as the difference between a
    dilation and an erosion of the input with a given structuring element.

    Parameters
    ----------
    input : array_like
        Array over which to compute the morphlogical gradient.
    size : tuple of ints
        Shape of a flat and full structuring element used for the mathematical
        morphology operations. Optional if `footprint` or `structure` is
        provided. A larger `size` yields a more blurred gradient.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the morphology operations. Larger footprints
        give a more blurred morphological gradient.
    structure : array of ints, optional
        Structuring element used for the morphology operations.
        `structure` may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the morphological gradient
        may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    morphological_gradient : ndarray
        Morphological gradient of `input`.

    See also
    --------
    grey_dilation, grey_erosion, gaussian_gradient_magnitude

    Notes
    -----
    For a flat structuring element, the morphological gradient
    computed at a given point corresponds to the maximal difference
    between elements of the input among the elements covered by the
    structuring element centered on the point.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Mathematical_morphology

    Examples
    --------
    >>> import cupy as cp
    >>> from cupyimg.scipy import ndimage
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> ndimage.morphological_gradient(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> # The morphological gradient is computed as the difference
    >>> # between a dilation and an erosion
    >>> ndimage.grey_dilation(a, size=(3,3)) -\\
    ...  ndimage.grey_erosion(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> a = cp.zeros((7,7), dtype=int)
    >>> a[2:5, 2:5] = 1
    >>> a[4,4] = 2; a[2,3] = 3
    >>> a
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 3, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    >>> ndimage.morphological_gradient(a, size=(3,3))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 3, 3, 1, 0],
           [0, 1, 3, 2, 3, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 1, 1, 2, 2, 2, 0],
           [0, 0, 0, 0, 0, 0, 0]])

    """

    tmp = grey_dilation(
        input, size, footprint, structure, None, mode, cval, origin
    )
    if isinstance(output, cupy.ndarray):
        grey_erosion(
            input, size, footprint, structure, output, mode, cval, origin
        )
        return cupy.subtract(tmp, output, output)
    else:
        return tmp - grey_erosion(
            input, size, footprint, structure, None, mode, cval, origin
        )


def morphological_laplace(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional morphological laplace.

    Parameters
    ----------
    input : array_like
        Input.
    size : int or sequence of ints, optional
        See `structure`.
    footprint : bool or ndarray, optional
        See `structure`.
    structure : structure, optional
        Either `size`, `footprint`, or the `structure` must be provided.
    output : ndarray, optional
        An output array can optionally be provided.
    mode : {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The mode parameter determines how the array borders are handled.
        For 'constant' mode, values beyond borders are set to be `cval`.
        Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if mode is 'constant'.
        Default is 0.0
    origin : origin, optional
        The origin parameter controls the placement of the filter.

    Returns
    -------
    morphological_laplace : ndarray
        Output

    """

    tmp1 = grey_dilation(
        input, size, footprint, structure, None, mode, cval, origin
    )
    if isinstance(output, cupy.ndarray):
        grey_erosion(
            input, size, footprint, structure, output, mode, cval, origin
        )
        cupy.add(tmp1, output, output)
        cupy.subtract(output, input, output)
        return cupy.subtract(output, input, output)
    else:
        tmp2 = grey_erosion(
            input, size, footprint, structure, None, mode, cval, origin
        )
        cupy.add(tmp1, tmp2, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        cupy.subtract(tmp2, input, tmp2)
        return tmp2


def white_tophat(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional white tophat filter.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of elements of a flat structuring element
        used for the white tophat filter.
    structure : array of ints, optional
        Structuring element used for the filter. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'.
        Default is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default is 0.

    Returns
    -------
    output : ndarray
        Result of the filter of `input` with `structure`.

    See also
    --------
    black_tophat

    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            "ignoring size because footprint is set", UserWarning, stacklevel=2
        )

    tmp = grey_erosion(
        input, size, footprint, structure, None, mode, cval, origin
    )
    tmp = grey_dilation(
        tmp, size, footprint, structure, output, mode, cval, origin
    )

    if output is not None and tmp is not output:
        # TODO: should not need this if statement case
        output[...] = tmp
        tmp = output
    # if tmp is None:
    #    tmp = output

    if tmp is None:
        tmp = output

    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(input, tmp, out=tmp)
    else:
        cupy.subtract(input, tmp, out=tmp)
    return tmp


def black_tophat(
    input,
    size=None,
    footprint=None,
    structure=None,
    output=None,
    mode="reflect",
    cval=0.0,
    origin=0,
):
    """
    Multidimensional black tophat filter.

    Parameters
    ----------
    input : array_like
        Input.
    size : tuple of ints, optional
        Shape of a flat and full structuring element used for the filter.
        Optional if `footprint` or `structure` is provided.
    footprint : array of ints, optional
        Positions of non-infinite elements of a flat structuring element
        used for the black tophat filter.
    structure : array of ints, optional
        Structuring element used for the filter. `structure`
        may be a non-flat structuring element.
    output : array, optional
        An array used for storing the output of the filter may be provided.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0.
    origin : scalar, optional
        The `origin` parameter controls the placement of the filter.
        Default 0

    Returns
    -------
    black_tophat : ndarray
        Result of the filter of `input` with `structure`.

    See also
    --------
    white_tophat, grey_opening, grey_closing

    """
    if (size is not None) and (footprint is not None):
        warnings.warn(
            "ignoring size because footprint is set", UserWarning, stacklevel=2
        )

    tmp = grey_dilation(
        input, size, footprint, structure, None, mode, cval, origin
    )
    tmp = grey_erosion(
        tmp, size, footprint, structure, output, mode, cval, origin
    )

    if output is not None and tmp is not output:
        # TODO: should not need this if statement case
        output[...] = tmp
        tmp = output
    # if tmp is None:
    #     tmp = output

    if input.dtype == numpy.bool_ and tmp.dtype == numpy.bool_:
        cupy.bitwise_xor(tmp, input, out=tmp)
    else:
        cupy.subtract(tmp, input, out=tmp)
    return tmp
