# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The SiamMask tracker, and the vendored SiamMask tree it wraps as `siammask/`.

From `viame.pytorch` in P2-T06. Installed only with
`VIAME_ENABLE_PYTORCH-SIAMMASK`, so its declarations exist only in a build
that has it.
"""

# ----------------------------------------------------------------------------
# What this package provides, without importing any of it.
#
# Each entry is ( interface, name, description, "module:Class" ).
__vital_algorithm_declarations__ = [
    ( "track_objects", "siammask",
      "SiamMask visual object tracker",
      "viame.object_trackers.siammask.siammask_tracker:SiamMaskTracker" ),
    ( "train_tracker", "siammask",
      "PyTorch SiamMask tracker training routine",
      "viame.object_trackers.siammask.siammask_trainer:SiamMaskTrainer" ),
    ( "train_tracker", "siamrpn",
      "PyTorch SiamRPN++ tracker training routine",
      "viame.object_trackers.siammask.siammask_trainer:SiamRPNTrainer" ),
]
