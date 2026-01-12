# -*- coding: utf-8 -*-
"""
DEPRECATED

NOTE:
    THIS IS DEPRECATED IN FAVOR OF COCO_DATASET IN KWCOCO
"""

__all__ = [
    'CocoDataset',
]

try:
    from kwcoco import CocoDataset
except ImportError:
    CocoDataset = None
