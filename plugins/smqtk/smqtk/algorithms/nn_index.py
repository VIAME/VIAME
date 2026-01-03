"""
SMQTK Nearest Neighbors Index - Abstract interface for NN algorithms.
"""
import abc

import six

from . import SmqtkAlgorithm
from ..utils import check_empty_iterable


@six.add_metaclass(abc.ABCMeta)
class NearestNeighborsIndex(SmqtkAlgorithm):
    """
    Common interface for descriptor-based nearest-neighbor indexing.
    """

    def __len__(self):
        return self.count()

    @staticmethod
    def _empty_iterable_exception():
        return ValueError("No descriptors in provided iterable.")

    def build_index(self, descriptors):
        """
        Build the index over the given descriptors.
        """
        check_empty_iterable(descriptors, self._build_index,
                             self._empty_iterable_exception())

    def update_index(self, descriptors):
        """
        Additively update the current index with given descriptors.
        """
        check_empty_iterable(descriptors, self._update_index,
                             self._empty_iterable_exception())

    def remove_from_index(self, uids):
        """
        Remove descriptors from this index by UID.
        """
        check_empty_iterable(uids, self._remove_from_index,
                             self._empty_iterable_exception())

    def nn(self, d, n=1):
        """
        Return the n nearest neighbors to the given descriptor.

        :param d: Descriptor element or vector to query.
        :param n: Number of nearest neighbors to return.
        :return: Tuple of (descriptors, distances).
        """
        if not self.count():
            raise ValueError("No index currently set to query from!")
        return self._nn(d, n)

    @abc.abstractmethod
    def count(self):
        """Return number of elements in this index."""

    @abc.abstractmethod
    def _build_index(self, descriptors):
        """Internal method to build the index."""

    @abc.abstractmethod
    def _update_index(self, descriptors):
        """Internal method to update the index."""

    @abc.abstractmethod
    def _remove_from_index(self, uids):
        """Internal method to remove from the index."""

    @abc.abstractmethod
    def _nn(self, d, n=1):
        """Internal method to find nearest neighbors."""


def get_nn_index_impls(reload_modules=False):
    """Return available NearestNeighborsIndex implementations."""
    from .lsh import LSHNearestNeighborIndex
    return {
        'LSHNearestNeighborIndex': LSHNearestNeighborIndex,
    }


__all__ = [
    'NearestNeighborsIndex',
    'get_nn_index_impls',
]
