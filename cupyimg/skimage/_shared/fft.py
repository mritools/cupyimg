"""Prefer FFTs via the new scipy.fft module when available (SciPy 1.4+)

Otherwise fall back to numpy.fft.

Like numpy 1.15+ scipy 1.3+ is also using pocketfft, but a newer
C++/pybind11 version called pypocketfft
"""
try:
    import cupyx.scipy.fft

    fftmodule = cupyx.scipy.fft
except ImportError:
    import cupy.fft

    fftmodule = cupy.fft

# TODO: grlee77
# SciPy next_fast_len should work, but a CUDA-specific version would be better
try:
    from scipy.fft import next_fast_len
except ImportError:
    from scipy.fftpack import next_fast_len

__all__ = ["fftmodule", "next_fast_len"]
