# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SRNN (Structural Recurrent Neural Network) Tracker

This module provides components for the SRNN tracker including:
- models: Neural network architectures (Siamese, LSTMs, TargetLSTM)
- track: Track state and track set management
- srnn_matching: LSTM-based track-to-detection matching
- siamese_feature_extractor: Appearance feature extraction
- iou_tracker: IoU-based pre-filtering
- gt_bbox: Ground truth bounding box loading
- training: Training infrastructure (submodule)
"""
