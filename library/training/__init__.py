# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The backend-independent training framework, in python.

Only the `export` subpackage: the `convert_to_onnx` process, which hands each
model to its backend's exporter where that backend lives. Each backend's
trainer is beside its inference routine since P2-T07 -- the detector
trainers in `viame.object_detectors`, the tracker trainers and their training
data handling in `viame.object_trackers`, SAM3 in `viame.segmentation`, SLEAP
in `viame.classifiers`. The adaptive and windowed trainers are C++, in this
library.
"""

__vital_algorithm_declarations__ = []
__sprokit_process_declarations__ = []
