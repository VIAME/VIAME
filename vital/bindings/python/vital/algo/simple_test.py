from __future__ import print_function

from vital.algo.image_object_detector import ImageObjectDetector
from vital.algo import algorithm_factory
from vital.types import Image, ImageContainer
from vital.types import DetectedObjectSet, DetectedObject, BoundingBox
from sprokit.pipeline import modules
import inspect


if __name__ == "__main__":
    # Register Algorithm
    implementation_name  = "SimpleImageObjectDetector"
    """
    if not algorithm_factory.has_algorithm_impl_name(
                                SimpleImageObjectDetector.static_type_name(), 
                                implementation_name):
        algorithm_factory.add_algorithm( implementation_name, 
                                    "test image object detector arrow",
                                     SimpleImageObjectDetector )
        algorithm_factory.mark_algorithm_as_loaded(implementation_name)
    # Create Algorithm
    print(SimpleImageObjectDetector.registered_names())
    """
    modules.load_known_modules()    
    example_detector = ImageObjectDetector.create(implementation_name)
    # Dummy Detect
    image = Image()
    image_container = ImageContainer(image)
    print(example_detector.detect(image_container)[0])
    

