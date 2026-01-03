"""
SMQTK Descriptor Element - Descriptor vector container.
"""
import abc
import logging
import multiprocessing
import threading
import time

import numpy
import six
from six.moves import queue
from six import next

from ..utils import SmqtkObject, merge_dict
from ..utils.plugin import Pluggable
from . import SmqtkRepresentation


@six.add_metaclass(abc.ABCMeta)
class DescriptorElement(SmqtkRepresentation, Pluggable):
    """
    Abstract descriptor vector container.
    """

    def __init__(self, type_str, uuid):
        """Initialize a new descriptor element."""
        super(DescriptorElement, self).__init__()
        self._type_label = type_str
        self._uuid = uuid

    def __hash__(self):
        return hash(self.uuid())

    def __eq__(self, other):
        if isinstance(other, DescriptorElement):
            return numpy.array_equal(self.vector(), other.vector())
        return False

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s{type: %s, uuid: %s}" % (self.__class__.__name__, self.type(),
                                           self.uuid())

    def __getstate__(self):
        return {
            "_type_label": self._type_label,
            "_uuid": self._uuid,
        }

    def __setstate__(self, state):
        self._type_label = state['_type_label']
        self._uuid = state['_uuid']

    @classmethod
    def get_default_config(cls):
        """Generate and return a default configuration dictionary."""
        dc = super(DescriptorElement, cls).get_default_config()
        if 'type_str' in dc:
            del dc['type_str']
        if 'uuid' in dc:
            del dc['uuid']
        return dc

    @classmethod
    def from_config(cls, config_dict, type_str, uuid, merge_default=True):
        """Instantiate from configuration."""
        c = {}
        merge_dict(c, config_dict)
        c['type_str'] = type_str
        c['uuid'] = uuid
        return super(DescriptorElement, cls).from_config(c, merge_default)

    def uuid(self):
        """Return unique ID for this vector."""
        return self._uuid

    def type(self):
        """Return type label of the DescriptorGenerator."""
        return self._type_label

    @abc.abstractmethod
    def has_vector(self):
        """Return whether this container has a descriptor vector stored."""

    @abc.abstractmethod
    def vector(self):
        """Get the stored descriptor vector as a numpy array."""

    @abc.abstractmethod
    def set_vector(self, new_vec):
        """Set the contained vector."""


class DescriptorMemoryElement(DescriptorElement):
    """
    In-memory representation of descriptor elements.
    """

    @classmethod
    def is_usable(cls):
        return True

    def __init__(self, type_str, uuid):
        super(DescriptorMemoryElement, self).__init__(type_str, uuid)
        self.__v = None

    def __getstate__(self):
        state = super(DescriptorMemoryElement, self).__getstate__()
        from six import BytesIO
        b = BytesIO()
        numpy.save(b, self.vector())
        state['v'] = b.getvalue()
        return state

    def __setstate__(self, state):
        if isinstance(state, tuple):
            self._type_label = state[0]
            self._uuid = state[1]
            from six import BytesIO
            b = BytesIO(state[2])
        else:
            super(DescriptorMemoryElement, self).__setstate__(state)
            from six import BytesIO
            b = BytesIO(state['v'])
        self.__v = numpy.load(b)

    def get_config(self):
        return {}

    def has_vector(self):
        return self.__v is not None

    def vector(self):
        if self.__v is not None:
            return numpy.copy(self.__v)
        return None

    def set_vector(self, new_vec):
        if new_vec is not None:
            self.__v = numpy.copy(new_vec)
        else:
            self.__v = None
        return self


DESCRIPTOR_ELEMENT_CLASS = DescriptorMemoryElement


def get_descriptor_element_impls(reload_modules=False):
    """Return available DescriptorElement implementations."""
    return {
        'DescriptorMemoryElement': DescriptorMemoryElement,
    }


def elements_to_matrix(descr_elements, mat=None, procs=None, buffer_factor=2,
                       report_interval=None, use_multiprocessing=False,
                       thread_q_put_interval=0.001):
    """
    Add to or create a numpy matrix from DescriptorElement instances.
    """
    log = logging.getLogger(__name__)

    if mat is None:
        sample = next(iter(descr_elements))
        sample_v = sample.vector()
        shp = (len(descr_elements), sample_v.size)
        log.debug("Creating new matrix with shape: %s", shp)
        mat = numpy.ndarray(shp, sample_v.dtype)

    if procs is None:
        procs = multiprocessing.cpu_count()

    # Simple single-threaded implementation for reliability
    for r, d in enumerate(descr_elements):
        if r >= mat.shape[0]:
            break
        mat[r] = d.vector()

        if report_interval and r > 0 and r % int(1.0/report_interval) == 0:
            log.debug("Processed %d descriptors", r)

    return mat


__all__ = [
    'DescriptorElement',
    'DescriptorMemoryElement',
    'get_descriptor_element_impls',
    'elements_to_matrix',
]
