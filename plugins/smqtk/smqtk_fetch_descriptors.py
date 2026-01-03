# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json

from kwiver.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process

from kwiver.vital.types import new_descriptor, DescriptorSet

# Use local smqtk package for VIAME
from .smqtk import representation as smqtk_representation
from .smqtk.utils import plugin as smqtk_plugin

class SmqtkFetchDescriptors (KwiverProcess):
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # register python config file
        self.add_config_trait(
            'config_file', 'config_file', '',
            'Path to the json configuration file for the descriptor index to fetch from.'
        )
        self.declare_config_using_trait('config_file')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('descriptor_set', required)
        self.declare_output_port_using_trait('string_vector', optional)

    def _configure(self):
        self.config_file = self.config_value('config_file')

        # parse json file
        with open(self.config_file) as data_file:
          self.json_config = json.load(data_file)

        # setup smqtk elements
        self.smqtk_descriptor_element_factory = \
            smqtk_representation.DescriptorElementFactory.from_config(
                self.json_config['descriptor_factory']
            )

        #: :type: smqtk_representation.DescriptorIndex
        self.smqtk_descriptor_index = smqtk_plugin.from_plugin_config(
            self.json_config['descriptor_index'],
            smqtk_representation.get_descriptor_index_impls()
        )

        self._base_configure()

    def _step(self):
        #
        # Grab input values from ports using traits.
        #
        # Vector of UIDs for vector of descriptors in descriptor_set.
        #
        #: :type: list[str]
        string_tuple = self.grab_input_using_trait('string_vector')

        descriptors = self.smqtk_descriptor_index.get_many_descriptors(string_tuple)
        vital_descriptors = []

        for desc in descriptors:
            vector = desc.vector()
            vital_desc = new_descriptor(len(vector), "d")
            vital_desc[:] = vector
            vital_descriptors.append(vital_desc)

        vital_descriptor_set = DescriptorSet(vital_descriptors)
        self.push_to_port_using_trait('descriptor_set', vital_descriptor_set)

        self._base_step()
