from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ""  # use "" for first of series, number for 1 and above
_version_extra = "dev0"
# _version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "cupyimg: CuPy-based subset of the skimage, scipy, etc. APIs"
# Long description will go up on the pypi page
long_description = """

CuPy Extensions
===============
This project contains CuPy-based implementations of functions from NumPy,
SciPy and Scikit-image that are not currently available in CuPy itself.

Ideally, much of the NumPy and SciPy-based functionality in this package will
be submitted upstream to the core CuPy project. This will allow more regular
continuous integration on a wider range of hardware.

For now these functions are provided in a separate, standalone package to allow
for rapid implementation / revision.

To get started using cupyimg with your own software, please go to the
repository README_.

.. _README: https://github.com/mritools/cupyimg/blob/master/README.md

License
=======
``cupyimg`` is licensed under the terms of the BSD 3-clause license. See the
file "LICENSE" for information on the history of this software, terms &
conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019-2020,
Gregory R. Lee, Cincinnati Children's Hospital Medical Center.
"""

NAME = "cupyimg"
MAINTAINER = "Gregory R. Lee"
MAINTAINER_EMAIL = "grlee77@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/mritools/cupyimg"
DOWNLOAD_URL = ""
LICENSE = "BSD"
AUTHOR = "Gregory R. Lee"
AUTHOR_EMAIL = "grlee77@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {
    "cupyimg": [
        pjoin("numpy", "core", "tests"),
        pjoin("numpy", "lib", "tests"),
        pjoin("scipy", "interpolate", "tests"),
        pjoin("scipy", "ndimage", "tests"),
        pjoin("scipy", "signal", "tests"),
        pjoin("scipy", "special", "tests"),
        pjoin("scipy", "stats", "tests"),
        pjoin("skimage", "color", "tests"),
        pjoin("skimage", "exposure", "tests"),
        pjoin("skimage", "feature", "tests"),
        pjoin("skimage", "filters", "tests"),
        pjoin("skimage", "measure", "tests"),
        pjoin("skimage", "metrics", "tests"),
        pjoin("skimage", "morphology", "tests"),
        pjoin("skimage", "registration", "tests"),
        pjoin("skimage", "restoration", "tests"),
        pjoin("skimage", "_shared", "tests"),
        pjoin("skimage", "transform", "tests"),
        pjoin("skimage", "util", "tests"),
    ]
}
REQUIRES = ["numpy"]
PYTHON_REQUIRES = ">= 3.6"
