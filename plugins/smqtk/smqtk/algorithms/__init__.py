"""
SMQTK Algorithms - Algorithm interfaces and implementations.
"""
import abc
import six

from ..utils import SmqtkObject, Configurable
from ..utils.plugin import Pluggable


@six.add_metaclass(abc.ABCMeta)
class SmqtkAlgorithm(SmqtkObject, Pluggable):
    """
    Base class for SMQTK algorithm interfaces.
    """
    pass


# Import and expose key classes
from .nn_index import NearestNeighborsIndex, get_nn_index_impls
from .hash_index import HashIndex, LinearHashIndex, get_hash_index_impls
from .lsh_functors import LshFunctor, ItqFunctor, get_lsh_functor_impls
from .lsh import LSHNearestNeighborIndex
from .relevancy_index import RelevancyIndex, LibSvmHikRelevancyIndex, get_relevancy_index_impls


__all__ = [
    'SmqtkAlgorithm',
    'NearestNeighborsIndex',
    'get_nn_index_impls',
    'HashIndex',
    'LinearHashIndex',
    'get_hash_index_impls',
    'LshFunctor',
    'ItqFunctor',
    'get_lsh_functor_impls',
    'LSHNearestNeighborIndex',
    'RelevancyIndex',
    'LibSvmHikRelevancyIndex',
    'get_relevancy_index_impls',
]
