import cupy
import numpy as np


def fourier_shift(input, shift, n=-1, axis=-1, output=None):
    """
    Multidimensional Fourier shift filter.

    The array is multiplied with the Fourier transform of a shift operation.

    Parameters
    ----------
    input : array_like
        The input array.
    shift : float or sequence
        The size of the box used for filtering.
        If a float, `shift` is the same for all axes. If a sequence, `shift`
        has to contain one value for each axis.
    n : int, optional
        If `n` is negative (default), then the input is assumed to be the
        result of a complex fft.
        If `n` is larger than or equal to zero, the input is assumed to be the
        result of a real fft, and `n` gives the length of the array before
        transformation along the real transform direction.
    axis : int, optional
        The axis of the real transform.
    output : ndarray, optional
        If given, the result of shifting the input is placed in this array.
        None is returned in this case.

    Returns
    -------
    fourier_shift : ndarray
        The shifted input.

    Examples
    --------
    >>> import cupy
    >>> from cupyimg.scipy import ndimage
    >>> from scipy import misc
    >>> import matplotlib.pyplot as plt
    >>> import cupy.fft
    >>> fig, (ax1, ax2) = plt.subplots(1, 2)
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ascent = cupy.asarray(misc.ascent())
    >>> input_ = cupy.fft.fft2(ascent)
    >>> result = ndimage.fourier_shift(input_, shift=200)
    >>> result = cupy.fft.ifft2(result)
    >>> ax1.imshow(ascent.get())
    >>> ax2.imshow(result.real.get())  # the imaginary part is an artifact
    >>> plt.show()
    """

    from cupyimg._misc import _reshape_nd

    iarr = input
    ndim = iarr.ndim
    if axis < -ndim or axis >= ndim:
        raise ValueError("invalid axis")
    axis = axis % ndim

    if np.isscalar(shift):
        shift = (shift,) * ndim
    elif len(shift) != ndim:
        raise ValueError("number of shifts must match input.ndim")

    for kk in range(ndim):
        ax_size = iarr.shape[kk]
        shiftk = shift[kk]
        if shiftk == 0:
            continue
        if kk == axis and n > 0:
            s = -2j * np.pi * shiftk / n
            arr = cupy.arange(ax_size, dtype=complex)
            arr *= s
            arr = _reshape_nd(arr, ndim=ndim, axis=kk)
        else:
            s = -2j * np.pi * shiftk / ax_size
            arr = cupy.concatenate(
                (
                    cupy.arange((ax_size + 1) // 2, dtype=complex),
                    cupy.arange(-(ax_size // 2), 0, dtype=complex),
                )
            )
            arr *= s
            arr = _reshape_nd(arr, ndim=ndim, axis=kk)
        cupy.exp(arr, out=arr)
        iarr = iarr * arr
    # TODO: could improve the efficiency by creating an ElementwiseKernel
    return iarr
