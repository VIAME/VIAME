# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import sys

import viame
from viame.algo import ImageObjectDetector
from viame.config import read_config_file

image = viame.open(sys.argv[1])

detector = ImageObjectDetector.create("hough_circle")

config = detector.get_configuration()
if len(sys.argv) > 2:
    config.merge_config(read_config_file(sys.argv[2]))
detector.set_configuration(config)

detections = detector.detect(image)

print("There were", len(detections), "detections in the image.")
