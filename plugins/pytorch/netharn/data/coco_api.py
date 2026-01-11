# -*- coding: utf-8 -*-
"""
DEPRECATED

NOTE:
    THIS IS DEPRECATED IN FAVOR OF COCO_DATASET IN KWCOCO
"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = [
    'CocoDataset',
]

try:
    from kwcoco import CocoDataset
except ImportError:
    CocoDataset = None
