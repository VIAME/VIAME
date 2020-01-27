# ckwg +29
# Copyright 2019 by Kitware, Inc.
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

from kwiver.vital.algo import ImageObjectDetector
from kwiver.vital.types import DetectedObjectSet, DetectedObject, BoundingBox
from kwiver.vital.types import ImageContainer

class SimpleImageObjectDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector that creates a bounding box on the
    coordinates specified by the user using configuration

    Examples:
        With default value of center(200, 200) and bbox dimension (100, 200)

        >>> from kwiver.vital.modules import modules
        >>> modules.load_known_modules()
        >>> from kwiver.vital.algo import ImageObjectDetector
        >>> image_detector = ImageObjectDetector.create("SimpleImageObjectDetector")
        >>> from kwiver.vital.types import Image, ImageContainer
        >>> image = ImageContainer(Image())
        >>> detection = image_detector.detect(image)
        >>> print(str(detection[0].bounding_box()))
        <BoundingBox(150.0, 100.0, 250.0, 300.0)>

        With configuration that changes the center value

        >>> modules.load_known_modules()
        >>> from kwiver.vital.algo import ImageObjectDetector
        >>> image_detector = ImageObjectDetector.create("SimpleImageObjectDetector")
        >>> from kwiver.vital.types import Image, ImageContainer
        >>> image = ImageContainer(Image())
        >>> from kwiver.vital.config import config
        >>> tc = config.empty_config()
        >>> tc.set_value("center_x", "200")
        >>> tc.set_value("center_y", "100")
        >>> image_detector.check_configuration(tc)
        False
        >>> image_detector.set_configuration(tc)
        >>> detection = image_detector.detect(image)
        >>> print(detection[0].bounding_box())
        <BoundingBox(150.0, 0.0, 250.0, 200.0)>

        Using kwiver runner from build/install directory
        $ kwiver runner examples/pipelines/example_pydetector_on_image.pipe
    """
    def __init__(self):
        ImageObjectDetector.__init__(self)
        self.m_center_x = 200.0
        self.m_center_y = 200.0
        self.m_height = 200.0
        self.m_width = 100.0
        self.m_dx = 0
        self.m_dy = 0
        self.frame_ct = 0

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
        cfg.set_value( "center_x", str(self.m_center_x) )
        cfg.set_value( "center_y", str(self.m_center_y) )
        cfg.set_value( "height", str(self.m_height) )
        cfg.set_value( "width", str(self.m_width) )
        cfg.set_value( "dx", str(self.m_dx) )
        cfg.set_value( "dy", str(self.m_dy) )
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self.m_center_x     = float(cfg.get_value( "center_x" ))
        self.m_center_y     = float(cfg.get_value( "center_y" ))
        self.m_height       = float(cfg.get_value( "height" ))
        self.m_width        = float(cfg.get_value( "width" ))
        self.m_dx           = int(float(cfg.get_value( "dx" )))
        self.m_dy           = int(float(cfg.get_value( "dy" )))

    def check_configuration( self, cfg):
        if cfg.has_value("center_x") and not float(cfg.get_value( "center_x" ))==self.m_center_x:
            return False
        if cfg.has_value("center_y") and not float(cfg.get_value( "center_y" ))==self.m_center_y:
            return False
        if cfg.has_value("height") and not float(cfg.get_value( "height" ))==self.m_height:
            return False
        if cfg.has_value("width") and not float(cfg.get_value( "width" ))==self.m_width:
            return False
        if cfg.has_value("dx") and not int(float(cfg.get_value( "dx" )))==self.m_dx:
            return False
        if cfg.has_value("dy") and not int(float(cfg.get_value( "dy" )))==self.m_dy:
            return False
        return True

    def detect(self, image_data):
        dot = DetectedObjectSet([DetectedObject(
            BoundingBox(self.m_center_x + self.frame_ct*self.m_dx - self.m_width/2.0,
                        self.m_center_y + self.frame_ct*self.m_dy - self.m_height/2.0,
                        self.m_center_x + self.frame_ct*self.m_dx + self.m_width/2.0,
                        self.m_center_y + self.frame_ct*self.m_dy + self.m_height/2.0))])
        self.frame_ct+=1
        return dot

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory
    # Register Algorithm
    implementation_name  = "SimpleImageObjectDetector"
    if algorithm_factory.has_algorithm_impl_name(
                                SimpleImageObjectDetector.static_type_name(),
                                implementation_name):
        return
    algorithm_factory.add_algorithm( implementation_name,
                                "test image object detector arrow",
                                 SimpleImageObjectDetector )
    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
