"""
SMQTK - Minimal port for VIAME search and rapid model generation.

This package provides the core functionality needed for locality-sensitive
hashing (LSH) based search and interactive query refinement (IQR).
"""
from .exceptions import ReadOnlyError, NoUriResolutionError, InvalidUriError

from .utils import (
    SmqtkObject,
    Configurable,
    merge_dict,
    SimpleTimer,
    ncr,
)
from .utils.plugin import (
    Pluggable,
    make_config,
    from_plugin_config,
    to_plugin_config,
    get_plugins,
)

from .representation import (
    SmqtkRepresentation,
    DataElement,
    DataFileElement,
    get_data_element_impls,
    DescriptorElement,
    DescriptorMemoryElement,
    get_descriptor_element_impls,
    elements_to_matrix,
    DescriptorElementFactory,
    DescriptorIndex,
    MemoryDescriptorIndex,
    get_descriptor_index_impls,
    KeyValueStore,
    MemoryKeyValueStore,
    get_key_value_store_impls,
    NO_DEFAULT_VALUE,
)

from .algorithms import (
    SmqtkAlgorithm,
    NearestNeighborsIndex,
    get_nn_index_impls,
    HashIndex,
    LinearHashIndex,
    get_hash_index_impls,
    LshFunctor,
    ItqFunctor,
    get_lsh_functor_impls,
    LSHNearestNeighborIndex,
    RelevancyIndex,
    LibSvmHikRelevancyIndex,
    get_relevancy_index_impls,
)

from .iqr import IqrSession


__version__ = '0.1.0'

__all__ = [
    # Exceptions
    'ReadOnlyError',
    'NoUriResolutionError',
    'InvalidUriError',

    # Utils
    'SmqtkObject',
    'Configurable',
    'merge_dict',
    'SimpleTimer',
    'ncr',
    'Pluggable',
    'make_config',
    'from_plugin_config',
    'to_plugin_config',
    'get_plugins',

    # Representation
    'SmqtkRepresentation',
    'DataElement',
    'DataFileElement',
    'get_data_element_impls',
    'DescriptorElement',
    'DescriptorMemoryElement',
    'get_descriptor_element_impls',
    'elements_to_matrix',
    'DescriptorElementFactory',
    'DescriptorIndex',
    'MemoryDescriptorIndex',
    'get_descriptor_index_impls',
    'KeyValueStore',
    'MemoryKeyValueStore',
    'get_key_value_store_impls',
    'NO_DEFAULT_VALUE',

    # Algorithms
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

    # IQR
    'IqrSession',
]
