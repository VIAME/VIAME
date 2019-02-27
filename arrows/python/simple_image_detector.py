from __future__ import print_function

from vital.algo.image_object_detector import ImageObjectDetector
from vital.types import DetectedObjectSet, DetectedObject, BoundingBox


class SimpleImageObjectDetector(ImageObjectDetector):
    def __init__(self):
        ImageObjectDetector.__init__(self)

    def detect(self, image_data):
        print("Simple Image Detector")
        return DetectedObjectSet([DetectedObject(BoundingBox(0, 4, 3, 1))])

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
    

