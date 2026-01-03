"""
SMQTK Representation - Data representation classes.
"""
import abc
import six

from ..utils import SmqtkObject, Configurable
from ..utils.plugin import Pluggable


@six.add_metaclass(abc.ABCMeta)
class SmqtkRepresentation(SmqtkObject, Configurable):
    """
    Base class for SMQTK representation types.
    """
    pass


# Import and expose key classes
from .data_element import DataElement, DataFileElement, get_data_element_impls
from .descriptor_element import (
    DescriptorElement,
    DescriptorMemoryElement,
    get_descriptor_element_impls,
    elements_to_matrix,
)
from .descriptor_element_factory import DescriptorElementFactory
from .descriptor_index import (
    DescriptorIndex,
    MemoryDescriptorIndex,
    get_descriptor_index_impls,
)
from .key_value import (
    KeyValueStore,
    MemoryKeyValueStore,
    get_key_value_store_impls,
    NO_DEFAULT_VALUE,
)


__all__ = [
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
]
