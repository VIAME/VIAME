"""
SMQTK Key-Value Store - General key-value storage interface.
"""
import abc
import threading

import six
from six.moves import cPickle as pickle

from ..exceptions import ReadOnlyError
from ..utils.plugin import Pluggable, make_config, from_plugin_config, to_plugin_config
from . import SmqtkRepresentation
from .data_element import get_data_element_impls


NO_DEFAULT_VALUE = type("KeyValueStoreNoDefaultValueType", (object,), {})()


@six.add_metaclass(abc.ABCMeta)
class KeyValueStore(SmqtkRepresentation, Pluggable):
    """
    Interface for general key/value storage.
    """

    __hash__ = None

    def __len__(self):
        return self.count()

    def __contains__(self, item):
        return self.has(item)

    def __getitem__(self, item):
        return self.get(item)

    @abc.abstractmethod
    def __repr__(self):
        return '<' + self.__class__.__name__ + " %s>"

    @abc.abstractmethod
    def count(self):
        """Return the number of key-value relationships."""

    @abc.abstractmethod
    def keys(self):
        """Return iterator over keys."""

    def values(self):
        """Return iterator over values."""
        for k in self.keys():
            yield self.get(k)

    @abc.abstractmethod
    def is_read_only(self):
        """Return True if read-only."""

    @abc.abstractmethod
    def has(self, key):
        """Check if this store has a value for the given key."""

    @abc.abstractmethod
    def add(self, key, value):
        """Add a key-value pair."""
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def add_many(self, d):
        """Add multiple key-value pairs."""
        if self.is_read_only():
            raise ReadOnlyError("Cannot add to read-only instance %s." % self)

    @abc.abstractmethod
    def remove(self, key):
        """Remove a single key-value entry."""
        if self.is_read_only():
            raise ReadOnlyError("Cannot remove from read-only instance %s." % self)

    @abc.abstractmethod
    def remove_many(self, keys):
        """Remove multiple keys and associated values."""
        if self.is_read_only():
            raise ReadOnlyError("Cannot remove from read-only instance %s." % self)

    @abc.abstractmethod
    def get(self, key, default=NO_DEFAULT_VALUE):
        """Get the value for the given key."""

    @abc.abstractmethod
    def clear(self):
        """Clear this key-value store."""
        if self.is_read_only():
            raise ReadOnlyError("Cannot clear a read-only %s instance."
                                % self.__class__.__name__)


class MemoryKeyValueStore(KeyValueStore):
    """
    Thread-safe in-memory key-value store with optional caching.
    """

    PICKLE_PROTOCOL = 2

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        default = super(MemoryKeyValueStore, cls).get_default_config()
        default['cache_element'] = make_config(get_data_element_impls())
        return default

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        c = config_dict.copy()
        if 'cache_element' not in c or \
                c['cache_element'] is None or \
                c['cache_element']['type'] is None:
            c['cache_element'] = None
        else:
            c['cache_element'] = \
                from_plugin_config(config_dict['cache_element'],
                                   get_data_element_impls())
        return super(MemoryKeyValueStore, cls).from_config(c)

    def __init__(self, cache_element=None):
        """Create new in-memory key-value store."""
        super(MemoryKeyValueStore, self).__init__()
        self._cache_element = cache_element
        self._table = {}
        self._table_lock = threading.RLock()

        if self._cache_element:
            c_bytes = self._cache_element.get_bytes()
            if c_bytes:
                self._table = pickle.loads(c_bytes)

    def __repr__(self):
        return super(MemoryKeyValueStore, self).__repr__() \
            % ("cache_element: %s" % repr(self._cache_element))

    def cache_table(self):
        """Cache the current table to the cache element."""
        if self._cache_element is not None:
            self._cache_element.set_bytes(
                pickle.dumps(self._table, self.PICKLE_PROTOCOL))

    def count(self):
        with self._table_lock:
            return len(self._table)

    def get_config(self):
        if hasattr(self._cache_element, 'get_config'):
            elem_config = to_plugin_config(self._cache_element)
        else:
            elem_config = make_config(get_data_element_impls())
        return {
            'cache_element': elem_config
        }

    def keys(self):
        return six.iterkeys(self._table)

    def is_read_only(self):
        if self._cache_element and not self._cache_element.writable():
            return True
        return False

    def has(self, key):
        return key in self._table

    def add(self, key, value):
        super(MemoryKeyValueStore, self).add(key, value)
        with self._table_lock:
            self._table[key] = value
            self.cache_table()
        return self

    def add_many(self, d):
        super(MemoryKeyValueStore, self).add_many(d)
        with self._table_lock:
            for k, v in six.iteritems(d):
                self._table[k] = v
            self.cache_table()
        return self

    def remove(self, key):
        super(MemoryKeyValueStore, self).remove(key)
        with self._table_lock:
            del self._table[key]
            self.cache_table()
        return self

    def remove_many(self, keys):
        super(MemoryKeyValueStore, self).remove_many(keys)
        with self._table_lock:
            key_diff = set(keys) - set(self._table)
            if key_diff:
                if len(key_diff) == 1:
                    raise KeyError(list(key_diff)[0])
                else:
                    raise KeyError(key_diff)
            for k in keys:
                del self._table[k]
            self.cache_table()
        return self

    def get(self, key, default=NO_DEFAULT_VALUE):
        with self._table_lock:
            if default is NO_DEFAULT_VALUE:
                return self._table[key]
            else:
                return self._table.get(key, default)

    def clear(self):
        super(MemoryKeyValueStore, self).clear()
        with self._table_lock:
            self._table.clear()
        return self


KEY_VALUE_STORE_CLASS = MemoryKeyValueStore


def get_key_value_store_impls(reload_modules=False):
    """Return available KeyValueStore implementations."""
    return {
        'MemoryKeyValueStore': MemoryKeyValueStore,
    }


__all__ = [
    'KeyValueStore',
    'MemoryKeyValueStore',
    'get_key_value_store_impls',
    'NO_DEFAULT_VALUE',
]
