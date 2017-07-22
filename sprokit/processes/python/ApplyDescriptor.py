#ckwg +28
# Copyright 2015 by Kitware, Inc.
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

from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess

apply_descriptor_test_mode = False
try:
    import numpy as np
    from smqtk.algorithms.descriptor_generator import get_descriptor_generator_impls
    from smqtk.representation.data_element.file_element import DataFileElement
    from smqtk.representation.descriptor_element_factory import DescriptorElementFactory
    from smqtk.representation.descriptor_element.local_elements import DescriptorMemoryElement

    from smqtk.algorithms import get_descriptor_generator_impls
    from smqtk.utils.plugin import from_plugin_config
except:
    # By doing this we allow folks to test that their KWIVER environment is properly built, before
    # building and configuring SMQTK
    print "SMQTK not configured into this Python instance.  Entering ApplyDescriptor test mode"
    apply_descriptor_test_mode = True


class ApplyDescriptor(KwiverProcess):
    """
    This process gets ain image as input, does some stuff to it and
    sends the modified version to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

	# Add trait for our output port
        self.add_port_trait( 'vector', 'double_vector', 'Output descriptor vector' )

        # set up required flags
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our ports ( port-name, flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_output_port_using_trait('vector', required )

        # declare our configuration
        self.declare_configuration_key( "config_file", "",
                "Descriptor configuration file name" )


    # ----------------------------------------------
    def _configure(self):
	# Test extracting config as dictionary
        self.config_dict = {}
        cfg = self.available_config()
        for it in cfg:
            self.config_dict[it] = self.config_value(it)

        # If we're in test mode, don't do anything that requires smqtk.
        if not apply_descriptor_test_mode:
            # create descriptor factory
            self.factory = DescriptorElementFactory(DescriptorMemoryElement, {})

            # get config file name
            file_name = self.config_value( "config_file" )

            # open file
            cfg_file = open( file_name )

            from smqtk.utils.jsmin import jsmin
            import json

            self.descr_config = json.loads( jsmin( cfg_file.read() ) )

            #self.generator = CaffeDescriptorGenerator.from_config(self.descr_config)
            self.generator = from_plugin_config(self.descr_config, get_descriptor_generator_impls)

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # If we're in test mode, just grab the image and
        # push a fake descriptor without trying to use
        # smqtk.
        if not apply_descriptor_test_mode:
            # Get image from conatiner
            in_img = in_img_c.get_image()


            # convert generic image to PIL image
            pil_image = in_img.get_pil_image()
            pix = np.array(pil_image)

            # get image in acceptable format
            # TBD use in memory transfer
            pil_image.save( "file.png" )
            test_data = DataFileElement("file.png")

            result = self.generator.compute_descriptor(test_data, self.factory)
            desc_list = result.vector().tolist()

            # push list to output port
            self.push_to_port_using_trait( 'vector', desc_list )
        else:
            desc_list =  4096 * [0.223] # Create  fake descriptor in test mode
            self.push_to_port_using_trait('vector', desc_list)

        self._base_step()

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_factory

    module_name = 'python:kwiver.ApplyDescriptor'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('ApplyDescriptor', 'Apply descriptor to image', ApplyDescriptor)

    process_factory.mark_process_module_as_loaded(module_name)
