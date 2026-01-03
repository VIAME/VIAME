"""
SMQTK Descriptor Index - Index of descriptors keyed by UUID.
"""
import abc

import six
from six.moves import cPickle as pickle

from ..utils import merge_dict, SimpleTimer
from ..utils.plugin import Pluggable, make_config, from_plugin_config, to_plugin_config
from . import SmqtkRepresentation
from .descriptor_element import DescriptorElement
from .data_element import get_data_element_impls


@six.add_metaclass(abc.ABCMeta)
class DescriptorIndex(SmqtkRepresentation, Pluggable):
    """
    Index of descriptors, keyed and query-able by descriptor UUID.
    """

    def __delitem__(self, uuid):
        self.remove_descriptor(uuid)

    def __getitem__(self, uuid):
        return self.get_descriptor(uuid)

    def __iter__(self):
        return self.iterdescriptors()

    def __len__(self):
        return self.count()

    def __contains__(self, item):
        if isinstance(item, DescriptorElement):
            return self.has_descriptor(item.uuid())
        return False

    @abc.abstractmethod
    def count(self):
        """Return number of descriptor elements stored."""

    @abc.abstractmethod
    def clear(self):
        """Clear this descriptor index's entries."""

    @abc.abstractmethod
    def has_descriptor(self, uuid):
        """Check if a DescriptorElement with the given UUID exists."""

    @abc.abstractmethod
    def add_descriptor(self, descriptor):
        """Add a descriptor to this index."""

    @abc.abstractmethod
    def add_many_descriptors(self, descriptors):
        """Add multiple descriptors at one time."""

    @abc.abstractmethod
    def get_descriptor(self, uuid):
        """Get the descriptor associated with the given UUID."""

    @abc.abstractmethod
    def get_many_descriptors(self, uuids):
        """Get an iterator over descriptors associated to given UUIDs."""

    @abc.abstractmethod
    def remove_descriptor(self, uuid):
        """Remove a descriptor by UUID."""

    @abc.abstractmethod
    def remove_many_descriptors(self, uuids):
        """Remove descriptors associated to given UUIDs."""

    @abc.abstractmethod
    def iterkeys(self):
        """Return an iterator over indexed descriptor UUIDs."""

    @abc.abstractmethod
    def iterdescriptors(self):
        """Return an iterator over indexed descriptor elements."""

    @abc.abstractmethod
    def iteritems(self):
        """Return an iterator over indexed descriptor key and instance pairs."""

    def keys(self):
        """Alias for iterkeys."""
        return self.iterkeys()

    def items(self):
        """Alias for iteritems."""
        return self.iteritems()


class MemoryDescriptorIndex(DescriptorIndex):
    """
    In-memory descriptor index with optional file caching.
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(MemoryDescriptorIndex, cls).get_default_config()
        c['cache_element'] = make_config(get_data_element_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        if config_dict['cache_element'] \
                and config_dict['cache_element']['type']:
            e = from_plugin_config(config_dict['cache_element'],
                                   get_data_element_impls())
            config_dict['cache_element'] = e
        else:
            config_dict['cache_element'] = None

        return super(MemoryDescriptorIndex, cls).from_config(config_dict, False)

    def __init__(self, cache_element=None, pickle_protocol=-1):
        """Initialize a new in-memory descriptor index."""
        super(MemoryDescriptorIndex, self).__init__()

        self._table = {}
        self.cache_element = cache_element
        self.pickle_protocol = pickle_protocol

        if cache_element and not cache_element.is_empty():
            self._log.debug("Loading cached descriptor index table from %s "
                            "element.", cache_element.__class__.__name__)
            self._table = pickle.loads(cache_element.get_bytes())

    def get_config(self):
        c = merge_dict(self.get_default_config(), {
            "pickle_protocol": self.pickle_protocol,
        })
        if self.cache_element:
            merge_dict(c['cache_element'],
                       to_plugin_config(self.cache_element))
        return c

    def cache_table(self):
        if self.cache_element and self.cache_element.writable():
            with SimpleTimer("Caching descriptor table", self._log.debug):
                self.cache_element.set_bytes(pickle.dumps(self._table,
                                                          self.pickle_protocol))

    def count(self):
        return len(self._table)

    def clear(self):
        self._table = {}
        self.cache_table()

    def has_descriptor(self, uuid):
        return uuid in self._table

    def add_descriptor(self, descriptor, no_cache=False):
        self._table[descriptor.uuid()] = descriptor
        if not no_cache:
            self.cache_table()

    def add_many_descriptors(self, descriptors):
        added_something = False
        for d in descriptors:
            self.add_descriptor(d, no_cache=True)
            added_something = True
        if added_something:
            self.cache_table()

    def get_descriptor(self, uuid):
        return self._table[uuid]

    def get_many_descriptors(self, uuids):
        for uid in uuids:
            yield self._table[uid]

    def remove_descriptor(self, uuid, no_cache=False):
        del self._table[uuid]
        if not no_cache:
            self.cache_table()

    def remove_many_descriptors(self, uuids):
        for uid in uuids:
            self.remove_descriptor(uid, no_cache=True)
        self.cache_table()

    def iterkeys(self):
        return six.iterkeys(self._table)

    def iterdescriptors(self):
        return six.itervalues(self._table)

    def iteritems(self):
        return six.iteritems(self._table)


DESCRIPTOR_INDEX_CLASS = MemoryDescriptorIndex


def get_descriptor_index_impls(reload_modules=False):
    """Return available DescriptorIndex implementations."""
    return {
        'MemoryDescriptorIndex': MemoryDescriptorIndex,
    }


__all__ = [
    'DescriptorIndex',
    'MemoryDescriptorIndex',
    'get_descriptor_index_impls',
]
