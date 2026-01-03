"""
SMQTK Descriptor Element Factory.
"""
from ..utils import merge_dict
from ..utils.plugin import make_config
from . import SmqtkRepresentation
from .descriptor_element import get_descriptor_element_impls


class DescriptorElementFactory(SmqtkRepresentation):
    """
    Factory class for producing DescriptorElement instances of a specified type.
    """

    def __init__(self, d_type, type_config):
        """Initialize the factory."""
        self._d_type = d_type
        self._d_type_config = type_config

    @classmethod
    def get_default_config(cls):
        """Generate and return a default configuration dictionary."""
        return make_config(get_descriptor_element_impls())

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        """Instantiate from configuration."""
        if merge_default:
            merged_config = cls.get_default_config()
            merge_dict(merged_config, config_dict)
            config_dict = merged_config

        return DescriptorElementFactory(
            get_descriptor_element_impls()[config_dict['type']],
            config_dict[config_dict['type']]
        )

    def get_config(self):
        d_type_name = self._d_type.__name__
        return {
            'type': d_type_name,
            d_type_name: self._d_type_config,
        }

    def new_descriptor(self, type_str, uuid):
        """Create a new DescriptorElement instance."""
        return self._d_type.from_config(self._d_type_config, type_str, uuid)

    def __call__(self, type_str, uuid):
        """Create a new DescriptorElement instance."""
        return self.new_descriptor(type_str, uuid)


__all__ = ['DescriptorElementFactory']
