# ckwg +29
# Copyright 2020 by Kitware, Inc.
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

import numpy

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import datum
from kwiver.sprokit.pipeline import process
import kwiver.vital.types as kvt

class HomographyReaderProcess(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        self.add_config_trait('input', 'input', '', 'Path to input file')
        self.declare_config_using_trait('input')

        optional = process.PortFlags()
        self.declare_output_port_using_trait('homography_src_to_ref', optional)

    def _configure(self):
        self.fin = open(self.config_value('input'))
        self._base_configure()

    def _step(self):
        line = self.fin.readline()
        if not line:
            self.mark_process_as_complete()
            d = datum.complete()
            self.push_datum_to_port_using_trait('homography_src_to_ref', d)
            return
        line = line.split()
        array, (from_id, to_id) = line[:9], line[9:]
        array = numpy.array(list(map(float, array))).reshape((3, 3))
        homog = kvt.F2FHomography.from_doubles(array, int(from_id), int(to_id))
        self.push_to_port_using_trait('homography_src_to_ref', homog)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:kwiver.read_homography'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process('kw_read_homography',
                                'A Simple Kwiver homography reader',
                                HomographyReaderProcess)
    process_factory.mark_process_module_as_loaded(module_name)

