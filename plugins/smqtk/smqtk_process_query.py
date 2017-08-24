#ckwg +28
# Copyright 2017 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import print_function

import itertools
import json

from sprokit.pipeline import process

from kwiver.kwiver_process import KwiverProcess

import smqtk.representation
import smqtk.utils.plugin


class SmqtkProcessQuery (KwiverProcess):
    """
    Process for taking in a query descriptor set, alongside any known positive and negative UUIDs,
    converting them into SMQTK descriptor elements (variable backend) performing a query off of
    them.

    """

    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        # register python config file
        self.add_config_trait(
            'config_file', 'config_file', '',
            'Path to the json configuration file for the descriptor index.'
        )
        self.declare_config_using_trait('config_file')

        # register port traits
        self.add_port_trait("positive_uids", "string_vector",
                            "Positive sample UIDs")
        self.add_port_trait("negative_uids", "string_vector",
                            "Negative sample UIDs")

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        # declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('descriptor_set', required)
        self.declare_input_port_using_trait('positive_uids', optional)
        self.declare_input_port_using_trait('negative_uids', optional)

        self.declare_output_port_using_trait('string_vector', optional)

    def _configure(self):
        self.config_file = self.config_value('config_file')

        # parse json file
        with open(self.config_file) as data_file:
          self.json_config = json.load(data_file)

        # setup smqtk elements
        # TODO

        self._base_configure()

    def _step(self):
        #
        # Grab input values from ports using traits.
        #
        # Set/vector of descriptors to perform query off of
        #
        #: :type: vital.types.DescriptorSet
        vital_descriptor_set = self.grab_input_using_trait('descriptor_set')
        #
        # Vector of UIDs for vector of descriptors in descriptor_set.
        #
        #: :type: list[str]
        positive_tuple = self.grap_input_using_trait('positive_uids')
        negative_tuple = self.grap_input_using_trait('negative_uids')

        # Convert descriptors to SMQTK elements and add to configured index
        smqtk_descriptor_elements = []
        z = itertools.izip(vital_descriptor_set.descriptors(), string_tuple)
        for vital_descr, uid_str in z:
            smqtk_descr = self.smqtk_descriptor_element_factory.new_descriptor(
                'from_sprokit', uid_str
            )
            # A descriptor may already exist in the backend (if its persistant)
            # for the given UID. We currently always overwrite.
            smqtk_descr.set_vector(vital_descr)
            # Queue up element for adding to set.
            smqtk_descriptor_elements.append(smqtk_descr)

        # Perform query off of descriptor set
        # TODO

        # Pass on input descriptors and UIDs
        self.push_to_port_using_trait('descriptor_set', vital_descriptor_set)
        self.push_to_port_using_trait('string_vector', string_tuple)

        self._base_step()
