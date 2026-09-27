"""Explicit, optional CUDA image kernels.

Importing this module does not load CUDA. Use ``available()`` to probe the
runtime, then ``Context`` to upload, process and download images. The backend
is built only with VIAME_ENABLE_CUDA_KERNELS=ON. See CUDA.md beside the sources.
"""
from importlib import import_module


def _backend():
    return import_module("._cuda", __package__)


def availability_error():
    """Return an empty string when CUDA is usable, or a diagnostic otherwise."""
    try:
        return _backend().availability_error()
    except ImportError as error:
        return "CUDA image kernels unavailable (VIAME_ENABLE_CUDA_KERNELS=ON required): " + str(error)


def available():
    """Whether the optional extension and at least one CUDA device are usable."""
    return not availability_error()


def __getattr__(name):
    if name not in ("Context", "Image"):
        raise AttributeError(name)
    try:
        return getattr(_backend(), name)
    except ImportError as error:
        raise RuntimeError(availability_error()) from error
