# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from viame.pytorch.siammask.core.config import cfg
from viame.pytorch.siammask.tracker.siamrpn_tracker import SiamRPNTracker
from viame.pytorch.siammask.tracker.siammask_tracker import SiamMaskTracker
from viame.pytorch.siammask.tracker.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
