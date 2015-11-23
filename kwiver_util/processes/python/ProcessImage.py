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
from vital import Image
from vital import ImageContainer
import PIL


class ProcessImage(KwiverProcess):
    """
    This process gets ain image as input, does some stuff to it and
    sends the modified version to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.add_config_trait("output", "output", '.',
        'The path of the file to output to.')

        self.declare_config_using_trait( 'output' )

        # set up required flags
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        self.declare_input_port( 'input', 'integer',required,
            'Where numbers are read from.')


    # ----------------------------------------------
    def _configure(self):
        print "[DEBUG] ----- configure"
        path = self.config_value('output')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        num = self.grab_value_from_port('input')
        print "Number received:", num

        # expecting Image Container opaque tyle
        datum = self.grab_input_using_trait('image')

        # look up image container based on opaque handle (datum)
        in_img_c = ImageContainer.from_c_pointer( datum )

        in_img = in_img_c.get_image()
        pil_image = in_img.get_pil_image()

        import ImageDraw
        draw = ImageDraw.Draw(pil_image)
        draw.line((0, 0) + pil_image.size, fill=128, width=5)
        draw.line((0, pil_image.size[1], pil_image.size[0], 0), fill=128, width=5)
        draw.rectangle( [num*10, num*10, num+100, num+100], outline=125 )
        del draw

        pil_image.show()
        time.sleep(3)

        self._base_step()

# ==================================================================
def __sprokit_register__():
    from sprokit.pipeline import process_registry

    module_name = 'python:kwiver.ProcessImage'

    reg = process_registry.ProcessRegistry.self()

    if reg.is_module_loaded(module_name):
        return

    reg.register_process('ProcessImage', 'Process image test', ProcessImage)

    reg.mark_module_as_loaded(module_name)
