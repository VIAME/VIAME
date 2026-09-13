"""
mkinit ~/code/torch_liberator/torch_liberator/util/__init__.py -w
"""
from . import util_zip

from .util_zip import (split_archive, zopen,)

__all__ = ['split_archive', 'util_zip', 'zopen']
