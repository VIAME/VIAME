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
from __future__ import print_function
import kwiver.sprokit.pipeline.process
import kwiver.vital.config.config

import os.path

class kw_print_number_process(kwiver.sprokit.pipeline.process.PythonProcess):
    def __init__(self, conf ):
        kwiver.sprokit.pipeline.process.PythonProcess.__init__(self, conf)

        # declare our configuration items
        self.declare_configuration_key(
            'output',
            '.',
            'The path for the output file.')

        # create port flags
        flags = kwiver.sprokit.pipeline.process.PortFlags()
        flags.add(self.flag_required)

        # create input ports
        self.declare_input_port( # example
            'input',
            'integer',
            flags,
            'Where numbers are read from.')

    # ----------------------------------------------------------------
    def _configure(self):
        path = self.config_value('output')

        if not path:
            raise RuntimeError('The path given was empty')

        self.fout = open(path, 'w+')
        self.fout.flush()

        self._base_configure()

    # ----------------------------------------------------------------
    def _reset(self):
        self.fout.close()

        self._base_reset()

    # ----------------------------------------------------------------
    def _step(self):
        num = self.grab_value_from_port('input')

        print("Number received:", num)
        self.fout.write('%d\n' % num)

        self._base_step()

# ==================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.print_number'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('kw_print_number_process',
                                'A Simple Kwiver Test Process',
                                kw_print_number_process)

    process_factory.mark_process_module_as_loaded(module_name)
