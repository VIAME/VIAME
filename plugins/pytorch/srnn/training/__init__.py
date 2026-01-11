# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SRNN Tracker Training Module

This module provides training infrastructure for the SRNN (Structural RNN) tracker,
including:
- Siamese network training for appearance features
- Individual LSTM training for motion, appearance, interaction, and bbox features
- Combined TargetLSTM training for final SRNN model
- Data preparation from KW18 format
"""
