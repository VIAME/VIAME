# -*- coding: utf-8 -*-
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
"""
CommandLine:
    workon_py2
    source ~/code/VIAME/build/install/setup_viame.sh
    cd ~/code/VIAME/plugins/camtrawl

    feat1 = Feature(loc=(10, 1))
    feat2 = Feature(loc=(2, 3))

    np.sum((feat1.location - feat2.location) ** 2)

CommandLine:
    workon_py2
    cd ~/code/VIAME/plugins/camtrawl/python

    source ~/code/VIAME/build/install/setup_viame.sh
    export KWIVER_DEFAULT_LOG_LEVEL=info
    export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
    export PYTHONPATH=$(pwd):$PYTHONPATH

    python ~/code/VIAME/plugins/camtrawl/python/run_camtrawl.py
    python run_camtrawl.py

    ~/code/VIAME/build/install/bin/pipeline_runner -p camtrawl.pipe -S pythread_per_process

SeeAlso
    ~/code/VIAME/packages/kwiver/vital/bindings/python/vital/types
"""
from __future__ import absolute_import, print_function, division
from sprokit.pipeline import process
from sprokit.pipeline import datum
from kwiver.kwiver_process import KwiverProcess
from vital.types import DetectedObjectSet
import logging
import os
import define_pipeline

try:
    import ubelt as ub
except ImportError:
    print('Warning: pip install ubelt')
    ub = None

logging.basicConfig(level=getattr(logging, os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'INFO').upper(), logging.DEBUG))
log = logging.getLogger(__name__)
print = log.info


# TODO: add something similar in sprokit proper
TMP_SPROKIT_PROCESS_REGISTRY = []


def tmp_sprokit_register_process(name=None, doc=''):
    def _wrp(cls):
        name_ = name
        if name is None:
            name_ = cls.__name__
        TMP_SPROKIT_PROCESS_REGISTRY.append((name_, doc, cls))
        return cls
    return _wrp


N_SOURCE_NODES = 1


@tmp_sprokit_register_process(name='make_dos')
class MakeDOSProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.

    """
    # ----------------------------------------------
    def __init__(self, conf):
        print('conf = {!r}'.format(conf))
        log.debug(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        required = process.PortFlags()
        required.add(self.flag_required)

        self.counter = 0
        self.limit = 100000000

        self.declare_output_port_using_trait('detected_object_set', required )

    # ----------------------------------------------
    def _step(self):
        log.debug(' ----- step ' + self.__class__.__name__)

        detection_set = DetectedObjectSet()
        self.counter += 1

        if self.counter > self.limit:
            self.mark_process_as_complete()
            dat = datum.complete()
            self.push_datum_to_port('detected_object_set', dat)
        else:
            self.push_to_port_using_trait('detected_object_set', detection_set)

        self._base_step()


@tmp_sprokit_register_process(name='measure_dos')
class MeasureDOSProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        log.debug(' ----- init ' + self.__class__.__name__)

        KwiverProcess.__init__(self, conf)

        required = process.PortFlags()
        required.add(self.flag_required)

        for i in range(N_SOURCE_NODES):
            self.add_port_trait('detected_object_set' + str(i), 'detected_object_set', 'Detections from camera1')
            self.declare_input_port_using_trait('detected_object_set' + str(i), required)

        if ub is not None:
            self.prog = ub.ProgIter(verbose=3)
            self.prog.begin()

    # ----------------------------------------------
    def _step(self):
        log.debug(' ----- step ' + self.__class__.__name__)
        self.prog.step()

        detection_sets = {}
        for i in range(N_SOURCE_NODES):
            detection_sets[i] = self.grab_input_using_trait('detected_object_set' + str(i))

        self._base_step()


def main():
    """
    Processing_with_species_id.m is their main file

    CommandLine:
        # SPROKIT_PIPELINE_RUNNER=pipeline_runner
        cd ~/code/VIAME/plugins/camtrawl/python

        workon_py2
        source ~/code/VIAME/build/install/setup_viame.sh

        we-py2debug
        source ~/code/VIAME/build-relwithdeb/install/setup_viame.sh

        cd ~/code/VIAME/plugins/camtrawl/python
        # Ensure python and sprokit knows about our module
        export PYTHONPATH=$(pwd):$PYTHONPATH
        export KWIVER_DEFAULT_LOG_LEVEL=info
        export SPROKIT_PYTHON_MODULES=kwiver.processes:viame.processes:segfault_pipeline

        python ~/code/VIAME/plugins/camtrawl/python/segfault_pipeline.py

        /home/joncrall/code/VIAME/build/install/bin/pipeline_runner -p /home/joncrall/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe

        gdb -ex=r --args /home/joncrall/code/VIAME/build/install/bin/pipeline_runner -p /home/joncrall/.cache/sprokit/temp_pipelines/temp_pipeline_file.pipe
    """

    def add_stereo_camera_branch(pipe, prefix):
        """
        Helper that defines a single branch, so it can easilly be duplicated.
        """
        cam = {}
        cam['detect'] = detect = pipe.add_process(
            name=prefix + 'make_dos', type='make_dos', config={})
        detect.iports.connect({})
        return cam

    pipe = define_pipeline.Pipeline()

    # Make an arbitrary number of source nodes
    camera_branches = {
        i: add_stereo_camera_branch(pipe, 'node{}_'.format(i))
        for i in range(N_SOURCE_NODES)
    }

    # ------
    pipe.add_process(name='measure', type='measure_dos', config={})
    pipe['measure'].iports.connect({
        'detected_object_set{}'.format(i): cam['detect'].oports['detected_object_set']
        for i, cam in camera_branches.items()
    })
    # ------

    pipe.config['_pipeline:_edge']['capacity'] = 1
    pipe.config['_scheduler']['type'] = 'pythread_per_process'

    print('  --- RUN PIPELINE ---')
    print(pipe.make_pipeline_text())
    # pipe.draw_graph('mwe_segfault.png')
    # ub.startfile('mwe_segfault.png')
    pipe.run()
    return pipe


def __sprokit_register__():

    from sprokit.pipeline import process_factory

    module_name = 'python_' + __name__
    print("REGISTER THIS MODULE: {}, {}".format(module_name, __file__))
    if process_factory.is_process_module_loaded(module_name):
        return

    if ub is not None:
        print('TMP_SPROKIT_PROCESS_REGISTRY = {}'.format(ub.repr2(TMP_SPROKIT_PROCESS_REGISTRY)))

    for name, doc, cls in TMP_SPROKIT_PROCESS_REGISTRY:
        print("REGISTER PROCESS:")
        print(' * name = {!r}'.format(name))
        print(' * cls = {!r}'.format(cls))
        process_factory.add_process(name, doc, cls)

    process_factory.mark_process_module_as_loaded(module_name)


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/VIAME/plugins/camtrawl/python
        python ~/code/VIAME/plugins/camtrawl/python/segfault_pipeline.py
    """
    main()
