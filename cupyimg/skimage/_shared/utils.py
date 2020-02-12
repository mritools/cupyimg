import numbers

import cupy
import numpy as np

from ..util import img_as_float
from ._warnings import all_warnings, warn

__all__ = [
    "get_bound_method_class",
    "all_warnings",
    "safe_as_int",
    "check_nD",
    "check_shape_equality",
    "warn",
]


def get_bound_method_class(m):
    """Return the class for a bound method.

    """
    return m.__self__.__class__


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `cupy.int64`
        Returns the input value(s) coerced to dtype `cupy.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
    53

    """
    # TODO: grlee77: reduce use of this function on cupy and provide a more
    #                appropriate GPU alternative with less overhead.
    xp = cupy.get_array_module(val)
    mod = xp.asarray(val) % 1  # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:  # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:  # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        xp.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError(
            "Integer argument required but received "
            "{0}, check inputs.".format(val)
        )

    if xp is cupy:
        # cupy.round does not exist
        val = np.round(val.get()).astype(np.int64)
        if isinstance(val, np.ndarray):
            val = cupy.asarray(val)
        return val
    else:
        return xp.round(val).astype(np.int64)


def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError("Input images must have the same dimensions.")
    return


def check_nD(array, ndim, arg_name="image"):
    """
    Verify an array meets the desired ndims and array isn't empty.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = cupy.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, "-or-".join([str(n) for n in ndim]))
        )


def check_random_state(seed):
    """Turn seed into a `cupy.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or cupy.random.RandomState
           If `seed` is None, return the RandomState singleton used by `cupy.random`.
           If `seed` is an int, return a new RandomState instance seeded with `seed`.
           If `seed` is already a RandomState instance, return it.

    Raises
    ------
    ValueError
        If `seed` is of the wrong type.

    """
    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is cupy.random:
        return cupy.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, cupy.integer)):
        return cupy.random.RandomState(seed)
    if isinstance(seed, cupy.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState"
        " instance" % seed
    )


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes:
    ------
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in "df":
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image
