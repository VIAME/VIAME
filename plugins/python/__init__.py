"""VIAME's Python interface."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from ._io import open, ImageSequence, VideoSequence, Pipeline

from ._embedded import EmbeddedPipeline

__all__ = ["open", "ImageSequence", "VideoSequence", "Pipeline", "EmbeddedPipeline"]
