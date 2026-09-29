# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import sys

import viame
from viame.algo import ImageObjectDetector
from viame.config import read_config_file

image = viame.open(sys.argv[1])
config = read_config_file(sys.argv[2])

if not ImageObjectDetector.check_nested_algo_configuration("detector", config):
    sys.exit("Configuration check failed.")

detector = ImageObjectDetector.set_nested_algo_configuration("detector", config)

if detector is None:
    sys.exit("Unable to create detector")

detections = detector.detect(image)

print("There were", len(detections), "detections in the image.")
