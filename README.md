
# cupyimg: CuPy-based GPU functions for image/signal processing

[cupyimg] extends [CuPy] with additional functions for image/signal processing.
This package implements a subset of functions from [NumPy], [SciPy] and
[scikit-image] with GPU support.

These implementations generally match the API and behavior of their
corresponding CPU equivalents, although there are some exceptions.
In some cases such as scipy.ndimage equivalents, complex-valued support is
available on the GPU even though it is not present as part of the upstream
library.

Ideally, the NumPy/Scipy function implemented here will be submitted upstream
to [CuPy] itself where they will benefit from a more comprehensive CI
architecture on real GPU hardware and a broader set of maintainers. Currently,
testing of this package on NVIDIA hardware has been done only on an
NVIDIA 1080 Ti GPU using CUDA versions 9.2-10.2. However, it should work for
all CUDA versions supported by the underlying CuPy library.

## Available Functions

**cupyimg.numpy**:

    - convolve
    - correlate
    - gradient
    - ndim

**cupyimg.scipy.interpolate**:

    - interpnd
    - RegularGridInterpolator

**cupyimg.scipy.ndimage**:

    - convolve1d
    - correlate1d
    - convolve
    - correlate
    - gaussian_filter1d
    - gaussian_filter
    - uniform_filter1d
    - uniform_filter
    - prewitt
    - sobel
    - generic_laplace
    - laplace
    - gaussian_laplace
    - generic_gradient_magnitude
    - gaussian_gradient_magnitude
    - maximum_filter1d
    - maximum_filter
    - minimum_filter1d
    - minimum_filter
    - rank_filter
    - median_filter
    - percentile_filter
    - generate_binary_structure
    - iterate_structure'
    - binary_erosion
    - binary_dilation
    - binary_opening
    - binary_closing
    - binary_hit_or_miss
    - binary_propagation
    - binary_fill_holes
    - morphological_gradient
    - morphological_laplace
    - white_tophat
    - black_tophat

**cupyimg.scipy.signal**:

    - upfirdn
    - choose_conv_method
    - convolve
    - convolve2d
    - correlate
    - correlate2d
    - fftconvolve
    - oaconvolve
    - hilbert
    - hilbert2
    - resample
    - resample_poly
    - wiener

**cupyimg.scipy.special**:

    - entr
    - kl_div
    - rel_entr
    - huber
    - pseudo_huber

**cupyimg.scipy.stats**:

    - entropy

**skimage.color**:

    - All functions in this module are supported

**skimage.exposure**:

    - adjust_gamma
    - adjust_log
    - adjust_sigmoid
    - cumulative_distribution
    - equalize_hist
    - equalize_adapthist
    - histogram
    - is_low_contrast
    - match_histograms
    - rescale_intensity

**skimage.filters**:

    - inverse
    - wiener
    - LPIFilter2D
    - gaussian
    - median (ndimage mode only)
    - farid
    - farid_h
    - farid_v
    - prewitt
    - prewitt_h
    - prewitt_v
    - roberts
    - roberts_pos_diag
    - roberts_neg_diag
    - scharr
    - scharr_h
    - scharr_v
    - sobel
    - sobel_h
    - sobel_v
    - laplace
    - rank_filter
    - gabor_kernel
    - gabor
    - meijering
    - sato
    - frangi
    - hessian
    - unsharp_mask
    - window

**skimage.measure**:

    - approximate_polygon
    - subdivide_polygon
    - block_reduce
    - profile_line
    - shannon_entropy

**skimage.metrics**:

    - mean_squared_error
    - normalized_root_mse
    - peak_signal_noise_ratio
    - structural_similarity

**skimage.registration**:

    - optical_flow_tvl1

**skimage.restoration**:

    - square
    - rectangle
    - diamond
    - disk
    - cube
    - octahedron
    - ball
    - octagon
    - star

**skimage.restoration**:

    - denoise_tv_chambolle

**skimage.transform**:

    - warp
    - warp_coords
    - warp_polar
    - swirl
    - resize
    - rotate
    - rescale
    - downscale_local_mean
    - estimate_transform
    - matrix_transform
    - EuclideanTransform
    - SimilarityTransform
    - AffineTransform
    - ProjectiveTransform
    - EssentialMatrixTransform
    - FundamentalMatrixTransform
    - PolynomialTransform
    - integral_image
    - integrate
    - pyramid_reduce
    - pyramid_expand
    - pyramid_gaussian
    - pyramid_laplacian

**skimage.util**:

    - img_as_float32
    - img_as_float64
    - img_as_float
    - img_as_int
    - img_as_uint
    - img_as_ubyte
    - img_as_bool
    - dtype_limits
    - view_as_blocks
    - view_as_windows
    - crop
    - invert


## Basic Usage

Functions tend to operate in the same manner as those from their upstream
counterparts. If there are differences in dtype handling, etc. these should be
noted within the corresponding function's docstring.

Aside from potential dtype differences, the primary difference with their CPU
counterparts tends to be a requirement for ``cupy.ndarray`` inputs rather than
allowing array-likes more generally. This behavior is consistent with [CuPy]
itself where support for such array-likes is generally disallowed due to
performance considerations. In ``cupyimg`` this ``cupy.ndarray`` rule is not
yet consistently enforced everywhere, so some functions still accept numpy
arrays as inputs and will transfer to the GPU automatically internally via
``cupy.asarray``.

An simple example demonstrating applying of a uniform_filter to an array is:

```Python
import cupy as cp
from cupyimg.scipy.ndimage import uniform_filter

x = np.random.randn(128, 128, 128)
y = uniform_filter(x, size=5)
```

## Similar Software

The [RAPIDS] project [cuSignal] provides an alternative implementation of
functions from ``scipy.signal``, including some not currently present here. Like
[cupyimg], it also depends on [CuPy], but has an additional dependency on
[Numba]. One other difference is at the time of writing, [cuSignal] does not
support all of the new ``upfirdn`` and ``resample_poly`` boundary handling
modes introduced in SciPy 1.4, while these are supported in [cupyimg].

## Documentation

cupyimg supports Python 3.6, 3.7 and 3.8.

**Requires:**

- NumPy (>=1.14)
- CuPy  (>=7.0)
- SciPy (>=1.2)
- scikit-image (>=0.16.2)
- fast_upfirdn (>=0.2.0)

To run the tests users will also need:

- pytest

Developers should see additional requirements for development in
``requirements-dev.txt``.

**Installation:**

This package can be obtained from PyPI via

```
pip install cupyimg
```

**Example**
```Python
import numpy as np
import cupy
import scipy.ndimage as ndi
from cupyimg.scipy.ndimage import uniform_filter
#from cupy.time import repeat

d = cupy.cuda.Device()
# separable 5x5x5 convolution kernel on the CPU
x = np.random.randn(256, 256, 256).astype(np.float32)
y = ndi.uniform_filter(x, size=5)
# %timeit y = ndi.uniform_filter(x, size=5)
#    -> 935 ms ± 69.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# separable 5x5x5 convolution kernel on the GPU
xg = cupy.asarray(x)
yg = uniform_filter(xg, size=5)
# %timeit yg = uniform_filter(xg, size=5); d.synchronize()
#    -> 26.6 ms ± 45 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

[conda]: https://docs.conda.io/en/latest/
[CuPy]: https://cupy.chainer.org
[cupyimg]: https://github.com/mritools/cupyimg
[cuSignal]: https://github.com/rapidsai/cusignal
[Numba]: numba.pydata.org
[NumPy]: https://numpy.org/
[PyPI]: https://pypi.org
[RAPIDS]: https://rapids.ai
[SciPy]: https://scipy.org
[scikit-image]: https://scikit-image.org
