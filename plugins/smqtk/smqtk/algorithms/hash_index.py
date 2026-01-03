"""
SMQTK Hash Index - Index for bit-vector hash codes.
"""
import abc
import heapq
import threading

import numpy
import six
from six import BytesIO

from . import SmqtkAlgorithm
from ..representation import get_data_element_impls
from ..utils import merge_dict, check_empty_iterable
from ..utils.plugin import make_config, from_plugin_config, to_plugin_config
from ..utils.bit_utils import bit_vector_to_int_large, int_to_bit_vector_large
from ..utils.metrics import hamming_distance


@six.add_metaclass(abc.ABCMeta)
class HashIndex(SmqtkAlgorithm):
    """
    Specialized index for unique hash codes (bit-vectors) using hamming distance.
    """

    def __len__(self):
        return self.count()

    @staticmethod
    def _empty_iterable_exception():
        return ValueError("No hash vectors in provided iterable.")

    def build_index(self, hashes):
        """Build the index with the given hash codes."""
        check_empty_iterable(hashes, self._build_index,
                             self._empty_iterable_exception())

    def update_index(self, hashes):
        """Additively update the current index with hash vectors."""
        check_empty_iterable(hashes, self._update_index,
                             self._empty_iterable_exception())

    def remove_from_index(self, hashes):
        """Remove hash vectors from this index."""
        check_empty_iterable(hashes, self._remove_from_index,
                             self._empty_iterable_exception())

    def nn(self, h, n=1):
        """Return the nearest N neighbor hash codes to the given hash code."""
        if not self.count():
            raise ValueError("No index currently set to query from!")
        return self._nn(h, n)

    @abc.abstractmethod
    def count(self):
        """Return number of elements in this index."""

    @abc.abstractmethod
    def _build_index(self, hashes):
        """Internal method to build the index."""

    @abc.abstractmethod
    def _update_index(self, hashes):
        """Internal method to update the index."""

    @abc.abstractmethod
    def _remove_from_index(self, hashes):
        """Internal method to remove from the index."""

    @abc.abstractmethod
    def _nn(self, h, n=1):
        """Internal method to find nearest neighbors."""


class LinearHashIndex(HashIndex):
    """
    Basic linear index using heap sort (brute force).
    Hash codes are stored as large integer values.
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(LinearHashIndex, cls).get_default_config()
        c['cache_element'] = make_config(get_data_element_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        cache_element = None
        if config_dict['cache_element'] \
                and config_dict['cache_element']['type']:
            cache_element = \
                from_plugin_config(config_dict['cache_element'],
                                   get_data_element_impls())
        config_dict['cache_element'] = cache_element

        return super(LinearHashIndex, cls).from_config(config_dict, False)

    def __init__(self, cache_element=None):
        """Initialize linear, brute-force hash index."""
        super(LinearHashIndex, self).__init__()
        self.cache_element = cache_element
        self.index = set()
        self._model_lock = threading.RLock()
        self.load_cache()

    def get_config(self):
        c = self.get_default_config()
        if self.cache_element:
            c['cache_element'] = merge_dict(c['cache_element'],
                                            to_plugin_config(self.cache_element))
        return c

    def load_cache(self):
        """Load from file cache if available."""
        with self._model_lock:
            if self.cache_element and not self.cache_element.is_empty():
                buff = BytesIO(self.cache_element.get_bytes())
                self.index = set(numpy.load(buff, allow_pickle=True))

    def save_cache(self):
        """Save to file cache if configured."""
        with self._model_lock:
            if self.cache_element and self.index:
                if self.cache_element.is_read_only():
                    raise ValueError("Cache element (%s) is read-only."
                                     % self.cache_element)
                buff = BytesIO()
                numpy.save(buff, tuple(self.index))
                self.cache_element.set_bytes(buff.getvalue())

    def count(self):
        with self._model_lock:
            return len(self.index)

    def _build_index(self, hashes):
        with self._model_lock:
            new_index = set(map(bit_vector_to_int_large, hashes))
            self.index = new_index
            self.save_cache()

    def _update_index(self, hashes):
        with self._model_lock:
            self.index.update(set(map(bit_vector_to_int_large, hashes)))
            self.save_cache()

    def _remove_from_index(self, hashes):
        with self._model_lock:
            h_int_set = set(map(bit_vector_to_int_large, hashes))
            for h in h_int_set:
                if h not in self.index:
                    raise KeyError(h)
            self.index = self.index - h_int_set
            self.save_cache()

    def _nn(self, h, n=1):
        with self._model_lock:
            h_int = bit_vector_to_int_large(h)
            bits = len(h)
            near_codes = \
                heapq.nsmallest(n, self.index,
                                lambda e: hamming_distance(h_int, e))
            distances = [hamming_distance(c, h_int) for c in near_codes]
            return [int_to_bit_vector_large(c, bits) for c in near_codes], \
                   [d / float(bits) for d in distances]


HASH_INDEX_CLASS = LinearHashIndex


def get_hash_index_impls(reload_modules=False):
    """Return available HashIndex implementations."""
    return {
        'LinearHashIndex': LinearHashIndex,
    }


__all__ = [
    'HashIndex',
    'LinearHashIndex',
    'get_hash_index_impls',
]
