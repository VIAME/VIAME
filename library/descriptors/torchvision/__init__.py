# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Torchvision feature extraction and augmentation.

The ResNet, AlexNet and EfficientNet feature extractors, the augmenter, and
the two processes that run them. From `viame.pytorch` in P2-T06; installed
only with `VIAME_ENABLE_PYTORCH-VISION`, so these declarations exist only in a
build that has it.
"""


# ----------------------------------------------------------------------------
# The processes this package provides, without importing any of them.
#
# Each entry is ( name, description, "module:Class" ).
__sprokit_process_declarations__ = [
    ( "desc_augmentation",
      "Pytorch-Based Augmentation",
      "viame.descriptors.torchvision.torchvision_augment_process:DataAugmentation" ),
    ( "pytorch_descriptors",
      "pytorch feature extraction",
      "viame.descriptors.torchvision.torchvision_descriptors:ResNetDescriptors" ),
]
