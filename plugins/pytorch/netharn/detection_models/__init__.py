# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Detection Models

This module contains detection model implementations including:
- mm_models: MMDetection-based models (CascadeRCNN, etc.)
- efficientdet: EfficientDet models
- yolo2: YOLOv2 models
- new_models_v1: Newer model architectures
"""

# Import mm_models which is the primary detection model module
# This is always imported as it's needed for backwards compatibility
from . import mm_models

# Import other modules that don't have heavy optional dependencies
from . import yolo2

# Lazy import modules that depend on mmdet/mmcv to avoid import errors
# when those packages aren't installed or have version issues
__all__ = [
    'mm_models',
    'yolo2',
]

# Try to import optional modules
try:
    from . import efficientdet
    __all__.append('efficientdet')
except ImportError:
    efficientdet = None

try:
    from . import new_backbone
    from . import new_neck
    from . import new_head
    from . import new_detector
    from . import new_models_v1
    __all__.extend(['new_backbone', 'new_neck', 'new_head', 'new_detector', 'new_models_v1'])
except ImportError:
    new_backbone = None
    new_neck = None
    new_head = None
    new_detector = None
    new_models_v1 = None
