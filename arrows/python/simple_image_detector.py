from __future__ import print_function

from vital.algo import ImageObjectDetector
from vital.types import DetectedObjectSet, DetectedObject, BoundingBox
from sprokit.pipeline import config

class SimpleImageObjectDetector(ImageObjectDetector):
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
        self.m_center_x     = float(cfg.get_value( "center_x" ));
        self.m_center_y     = float(cfg.get_value( "center_y" ));
        self.m_height       = float(cfg.get_value( "height" ));
        self.m_width        = float(cfg.get_value( "width" ));
        self.m_dx           = int(cfg.get_value( "dx" ));
        self.m_dy           = int(cfg.get_value( "dy" ));

    def check_configuration( self, cfg):
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
    from vital.algo import algorithm_factory
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
