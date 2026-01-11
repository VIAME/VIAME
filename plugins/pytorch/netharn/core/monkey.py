"""
Handles monkey patching for system compatability
"""
import sys
if sys.version_info[0:2] >= (3, 10):
    # Workaround for tensorboard_logger
    import collections
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
    collections.MutableSequence = abc.MutableSequence
