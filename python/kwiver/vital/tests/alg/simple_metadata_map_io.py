# This file is part of KWIVER, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/kwiver/blob/master/LICENSE for details.
from __future__ import print_function

from kwiver.vital.algo import MetadataMapIO
from kwiver.vital.tests.py_helpers import CommonConfigurationMixin


class SimpleMetadataMapIO(CommonConfigurationMixin,
                            MetadataMapIO):
    """
    Implementation of MetadataMapIO to test it
    Examples:
    """
    def __init__(self):
        MetadataMapIO.__init__(self)


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name  = "MetadataMapIO"
    if algorithm_factory.has_algorithm_impl_name(
                            SimpleMetadataMapIO.static_type_name(),
                            implementation_name):
        return
    algorithm_factory.add_algorithm( implementation_name,
                                "Test kwiver.vital.algo.MetadataMapIO",
                                 SimpleMetadataMapIO )
    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
