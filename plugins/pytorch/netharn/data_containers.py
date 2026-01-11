"""
Proof-of-concept for porting mmcv DataContainer concept to netharn. Depending
on how well this works these features might be useful as a standalone module or
to contribute to torch proper.

References:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/scatter_gather.py

FIXME 0 dimension tensors
"""
from viame.arrows.pytorch.netharn.core.data.data_containers import *  # NOQA
