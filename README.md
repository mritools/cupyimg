
# cupyimg: n-d signal and image processing on the GPU

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


## Available Functions

**cupyimg.numpy**:

- convolve
- correlate
- gradient
- histogram  (upstream PR: [3124](https://github.com/cupy/cupy/pull/3124))
- histogram2d
- histogramdd
- ndim (upstream PR: [3060](https://github.com/cupy/cupy/pull/3060))
- ravel_multi_index (upstream PR: [3104](https://github.com/cupy/cupy/pull/3104))

**cupyimg.scipy.interpolate**:

- interpnd
- RegularGridInterpolator

**cupyimg.scipy.ndimage**:

- binary_erosion
- binary_dilation
- binary_opening
- binary_closing
- binary_hit_or_miss
- binary_propagation
- binary_fill_holes- convolve
- black_tophat
- convolve1d
- correlate
- correlate1d
- gaussian_filter
- gaussian_filter1d
- gaussian_laplace
- gaussian_gradient_magnitude
- generate_binary_structure
- generic_laplace
- generic_gradient_magnitude
- iterate_structure
- laplace
- prewitt
- sobel
- uniform_filter
- uniform_filter1d
- maximum_filter
- maximum_filter1d
- median_filter
- minimum_filter
- minimum_filter1d
- morphological_gradient
- morphological_laplace
- percentile_filter
- rank_filter
- white_tophat

**cupyimg.scipy.signal**:

- choose_conv_method
- convolve
- convolve2d
- correlate
- correlate2d
- fftconvolve
- hilbert
- hilbert2
- oaconvolve
- resample
- resample_poly
- upfirdn
- wiener

**cupyimg.scipy.special**:

- entr  (upstream PR: [2861](https://github.com/cupy/cupy/pull/2861))
- kl_div  (upstream PR: [2861](https://github.com/cupy/cupy/pull/2861))
- rel_entr  (upstream PR: [2861](https://github.com/cupy/cupy/pull/2861))
- huber  (upstream PR: [2861](https://github.com/cupy/cupy/pull/2861))
- pseudo_huber  (upstream PR: [2861](https://github.com/cupy/cupy/pull/2861))

**cupyimg.scipy.stats**:

- entropy

**skimage.color**:

- All functions in this module are supported

**skimage.exposure**:

- adjust_gamma
- adjust_log
- adjust_sigmoid
- cumulative_distribution
- equalize_adapthist
- equalize_hist
- histogram
- is_low_contrast
- match_histograms
- rescale_intensity

**skimage.filters**:

- farid
- farid_h
- farid_v
- frangi
- gabor_kernel
- gabor
- gaussian
- hessian
- inverse
- laplace
- LPIFilter2D
- median (ndimage mode only)
- meijering
- prewitt
- prewitt_h
- prewitt_v
- rank_filter
- roberts
- roberts_pos_diag
- roberts_neg_diag
- sato
- scharr
- scharr_h
- scharr_v
- sobel
- sobel_h
- sobel_v
- unsharp_mask
- wiener
- window

**skimage.measure**:

- approximate_polygon
- subdivide_polygon
- block_reduce
- compare_mse
- compare_nrmse
- compare_psnr
- compare_ssim
- inertia_tensor
- inertia_tensor_eigenvals
- moments
- moments_central
- moments_coords
- moments_coords_central
- moments_hu
- moments_normalized
- perimeter
- profile_line
- regionprops
- regionprops_table
- shannon_entropy
- subdivide_polygon

**skimage.metrics**:

- mean_squared_error
- normalized_root_mse
- normalized_mutual_information
- peak_signal_noise_ratio
- structural_similarity

**skimage.morphology**:

- ball
- binary_erosion
- binary_dilation
- binary_opening
- binary_closing
- black_tophat
- closing
- cube
- diamond
- dilation
- disk
- erosion
- octagon
- octahedron
- opening
- rectangle
- remove_small_holes
- remove_small_objects
- square
- star
- white_tophat

**skimage.registration**:

- affine
- optical_flow_tvl1

**skimage.restoration**:

- denoise_tv_chambolle
- richardson_lucy
- unsupervised_wiener
- wiener

**skimage.segmentation**:

- checkerboard_level_set
- circle_level_set
- clear_border
- disk_level_set
- find_boundaries
- inverse_gaussian_gradient
- mark_boundaries
- morphological_chan_vese
- morphological_geodesic_active_contour

**skimage.transform**:

- AffineTransform
- downscale_local_mean
- EssentialMatrixTransform
- estimate_transform
- EuclideanTransform
- FundamentalMatrixTransform
- integral_image
- integrate
- matrix_transform
- PolynomialTransform
- ProjectiveTransform
- pyramid_expand
- pyramid_gaussian
- pyramid_laplacian
- pyramid_reduce
- rescale
- resize
- rotate
- SimilarityTransform
- swirl
- warp
- warp_coords
- warp_polar

**skimage.util**:

- crop
- dtype_limits
- img_as_bool
- img_as_float
- img_as_float32
- img_as_float64
- img_as_int
- img_as_ubyte
- img_as_uint
- invert
- view_as_blocks
- view_as_windows


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
