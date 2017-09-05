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
    export SPROKIT_PYTHON_MODULES=camtrawl_processes:kwiver.processes:viame.processes
    export PYTHONPATH=$(pwd):$PYTHONPATH

    python run_camtrawl.py

    ~/code/VIAME/build/install/bin/pipeline_runner -p camtrawl.pipe -S pythread_per_process

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

import camtrawl_algos as ctalgo

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


@tmp_sprokit_register_process(name='camtrawl_detect_fish',
                              doc='preliminatry fish detection')
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

        # gmm_params = {
        #     'n_training_frames': 9999,
        #     # 'gmm_thresh': 20,
        #     'gmm_thresh': 30,
        #     'min_size': 800,
        #     'edge_trim': [40, 40],
        #     'n_startup_frames': 3,
        #     'factor': 2,
        #     'smooth_ksize': None,
        #     # 'smooth_ksize': (3, 3),
        #     # 'smooth_ksize': (10, 10),  # wrt original image size
        # }

        gmm_params = {}
        filter_config = {}
        self.detector = ctalgo.GMMForegroundObjectDetector(config=gmm_params)
        self.dfilter = ctalgo.FishDetectionFilter(config=filter_config)
        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print(' ----- step ' + self.__class__.__name__)
        # grab image container from port using traits
        img_container = self.grab_input_using_trait('image')
        img = img_container.asarray()

        img_dsize = tuple(img.shape[0:2][::-1])

        _detect_gen = self.detector.detect(img)
        detect_gen = self.dfilter.filter_detections(_detect_gen, img_dsize)

        objset = DetectedObjectSet()
        for detection in detect_gen:
            bbox = BoundingBox.from_coords(*detection.bbox.coords)
            mask = detection.mask
            obj = DetectedObject(bbox, 1.0, mask=mask)
            objset.add(obj)
        print('objset = {!r}'.format(objset))

        # # push dummy image object (same as input) to output port
        self.push_to_port_using_trait('detected_object_set', objset)

        self._base_step()


@tmp_sprokit_register_process(name='camtrawl_measure',
                              doc='preliminatry fish length measurement')
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

        self.add_port_trait('detected_object_set' + '1', 'detected_object_set', 'Detections from camera1')
        self.add_port_trait('detected_object_set' + '2', 'detected_object_set', 'Detections from camera2')

        #  declare our input port ( port-name,flags)
        self.declare_input_port_using_trait('detected_object_set' + '1', required)
        self.declare_input_port_using_trait('detected_object_set' + '2', required)

    # ----------------------------------------------
    def _configure(self):
        print(' ----- configure ' + self.__class__.__name__)

        self.text = self.config_value('text')

        triangulate_params = {}
        self.triangulator = ctalgo.FishStereoTriangulationAssignment(**triangulate_params)
        cal_fpath = ''

        # NEED TO LOAD CALIBRATION DATA
        # IDEALLY THIS IS A PIPELINE INPUT NOT A CONFIG
        self.cal = ctalgo.StereoCalibration.from_file(cal_fpath)
        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        print(' ----- step ' + self.__class__.__name__)

        # grab image container from port using traits
        detections1 = self.grab_input_using_trait('detected_object_set' + '1')
        detections2 = self.grab_input_using_trait('detected_object_set' + '2')

        assignment, assign_data, cand_errors = self.triangulator.find_matches(
            self.cal, detections1, detections2)

        # distance = np.sum((feat1.location - feat2.location) ** 2)
        # print('distance = {!r}'.format(distance))

        # Get python image from conatiner (just for show)
        # Print out text to screen
        print("Measure Text: " + str( self.text ))

        # push dummy image object (same as input) to output port
        # self.push_to_port_using_trait('out_image', ImageContainer(in_img))
        self._base_step()


def __sprokit_register__():

    from sprokit.pipeline import process_factory

    # module_name = 'python:camtrawl.camtrawl_processes'
    module_name = 'python' + __name__
    if process_factory.is_process_module_loaded(module_name):
        return

    for name, doc, cls in TMP_SPROKIT_PROCESS_REGISTRY:
        process_factory.add_process(name, doc, cls)

    # process_factory.add_process('camtrawl_detect_fish',
    #                             'preliminatry detection / feature extraction',
    #                             CamtrawlDetectFishProcess)

    # process_factory.add_process('camtrawl_measure',
    #                             'preliminatry measurement',
    #                             CamtrawlMeasureProcess)

    process_factory.mark_process_module_as_loaded(module_name)
