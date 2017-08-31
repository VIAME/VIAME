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

SeeAlso
    ~/code/VIAME/packages/kwiver/vital/bindings/python/vital/types
"""
from __future__ import print_function
import numpy as np
from sprokit.pipeline import process
from kwiver.kwiver_process import KwiverProcess
from vital.types import (  # NOQA
    Image,
    BoundingBox,
    DetectedObjectSet,
    DetectedObjectType,
    DetectedObject,
    Feature,
)
import logging
import os

logging.basicConfig(level=getattr(logging, os.environ.get('KWIVER_DEFAULT_LOG_LEVEL', 'INFO').upper(), logging.DEBUG))
log = logging.getLogger(__name__)
print = log.info


class CamtrawlDetectFishProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        print(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        self.add_config_trait('text', 'text', 'some default value',
                              'A description of this param')
        self.declare_config_using_trait('text')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)

        self.declare_output_port_using_trait('detected_object_set', optional )

    # ----------------------------------------------
    def _configure(self):
        print(' ----- configure ' + self.__class__.__name__)
        self.text = self.config_value('text')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print(' ----- step ' + self.__class__.__name__)
        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')

        # Get python image from conatiner (just for show)
        in_img = in_img_c.get_image()
        print('in_img = {!r}'.format(in_img))

        # Print out text to screen
        print("Text: " + str( self.text ))

        bbox = BoundingBox.from_coords(0, 0, 10, 10)
        obj1 = DetectedObject(bbox, 1.0)
        obj2 = DetectedObject([1, 2, 3, 5], 0.5)

        objset = DetectedObjectSet()
        objset.add(obj1)
        objset.add(obj2)

        # feat = Feature(loc=(0, 0), mag=0, scale=1, angle=0, rgb_color=None)
        # featset = [feat]
        # # push dummy image object (same as input) to output port
        self.push_to_port_using_trait('detected_object_set', objset)

        self._base_step()


class CamtrawlDetectBiomarkerProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        print(' ----- init ' + self.__class__.__name__)
        KwiverProcess.__init__(self, conf)

        self.add_config_trait('text', 'text', 'some default value',
                              'A description of this param')
        self.declare_config_using_trait('text')

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)

        self.declare_output_port_using_trait('feature_set', optional )

    # ----------------------------------------------
    def _configure(self):
        print(' ----- configure ' + self.__class__.__name__)
        self.text = self.config_value('text')

        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print(' ----- step ' + self.__class__.__name__)

        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        detected_object_set_c = self.grab_input_using_trait('detected_object_set')

        feat = Feature((0, 1))
        featset = {feat}

        print('in_img_c = {!r}'.format(in_img_c))
        print('detected_object_set_c = {!r}'.format(detected_object_set_c))
        print('featset = {!r}'.format(featset))

        # Get python image from conatiner (just for show)
        # in_img = in_img_c.get_image()
        # print('in_img = {!r}'.format(in_img))

        # # Print out text to screen
        # print("Text: " + str( self.text ))

        # feat = Feature(loc=(0, 0), mag=0, scale=1, angle=0, rgb_color=None)

        # featset = [feat]

        # # push dummy image object (same as input) to output port
        self.push_to_port_using_trait('feature_set', featset)

        self._base_step()


class CamtrawlMeasureProcess(KwiverProcess):
    """
    This process gets an image and detection_set as input, extracts each chip,
    does postprocessing and then sends the extracted chip to the output port.
    """
    # ----------------------------------------------
    def __init__(self, conf):
        print(' ----- init ' + self.__class__.__name__)

        KwiverProcess.__init__(self, conf)

        self.add_config_trait('text', 'text', 'some default value',
                              'A description of this param')
        self.declare_config_using_trait('text')

        # set up required flags
        # optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.add_port_trait('feature_set' + '1', 'feature_set', 'Feature from camera1')
        self.add_port_trait('feature_set' + '2', 'feature_set', 'Feature from camera2')

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('feature_set' + '1', required)
        self.declare_input_port_using_trait('feature_set' + '2', required)

        # self.declare_output_port_using_trait('feature_set', optional )

    # ----------------------------------------------
    def _configure(self):
        print(' ----- configure ' + self.__class__.__name__)

        self.text = self.config_value('text')
        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print(' ----- step ' + self.__class__.__name__)

        # grab image container from port using traits
        feat1 = self.grab_input_using_trait('feature_set' + '1')
        feat2 = self.grab_input_using_trait('feature_set' + '2')

        distance = np.sum((feat1.location - feat2.location) ** 2)
        print('distance = {!r}'.format(distance))

        # Get python image from conatiner (just for show)
        # Print out text to screen
        print("Measure Text: " + str( self.text ))

        # push dummy image object (same as input) to output port
        # self.push_to_port_using_trait('out_image', ImageContainer(in_img))

        self._base_step()


def __sprokit_register__():
    """
    workon_py2
    source ~/code/VIAME/build/install/setup_viame.sh
    export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
    export PYTHONPATH=$(pwd):$PYTHONPATH

    cd ~/code/VIAME/plugins/camtrawl
    # python camtrawl_pipeline_def.py
    ~/code/VIAME/build/install/bin/pipeline_runner -p camtrawl.pipe -S pythread_per_process
    """

    from sprokit.pipeline import process_factory

    # module_name = 'python:camtrawl.camtrawl_processes'
    module_name = 'python:camtrawl_processes'
    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('camtrawl_detect_fish',
                                'preliminatry detection / feature extraction',
                                CamtrawlDetectFishProcess)

    process_factory.add_process('camtrawl_detect_biomarker',
                                'preliminatry feature extraction',
                                CamtrawlDetectBiomarkerProcess)

    process_factory.add_process('camtrawl_measure',
                                'preliminatry measurement',
                                CamtrawlMeasureProcess)

    process_factory.mark_process_module_as_loaded(module_name)
