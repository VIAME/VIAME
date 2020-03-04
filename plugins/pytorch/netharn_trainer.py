# ckwg +29
# Copyright 2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#    * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
from __future__ import division

from vital.algo import TrainDetector
from vital.algo import DetectedObjectSetOutput

from vital.types import BoundingBox
from vital.types import CategoryHierarchy
from vital.types import DetectedObjectSet
from vital.types import DetectedObject

from PIL import Image

from distutils.util import strtobool
from shutil import copyfile

import argparse
import numpy as np
import torch
import pickle
import os
import signal
import sys
import subprocess

class NetHarnTrainer( TrainDetector ):
    """
    Implementation of TrainDetector class
    """
    def __init__( self ):
        TrainDetector.__init__( self )

        self._config_file = "bioharn-det-v14-cascade"
        self._seed_weights = ""
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "custom_cfrnn"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._tmp_training_file = "training_truth.json"
        self._tmp_validation_file = "validation_truth.json"
        self._gt_frames_only = False
        self._backbone = ""
        self._categories = []

    def get_configuration( self ):
        # Inherit from the base class
        cfg = super( TrainDetector, self ).get_configuration()

        cfg.set_value( "config_file", self._config_file )
        cfg.set_value( "seed_weights", self._seed_weights )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "output_directory", self._output_directory )
        cfg.set_value( "output_prefix", self._output_prefix )
        cfg.set_value( "pipeline_template", self._pipeline_template )
        cfg.set_value( "gpu_count", str( self._gpu_count ) )
        cfg.set_value( "gt_frames_only", str( self._gt_frames_only ) )
        cfg.set_value( "backbone", self._backbone )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        # Read configs from file
        self._config_file = str( cfg.get_value( "config_file" ) )
        self._seed_weights = str( cfg.get_value( "seed_weights" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )
        self._output_prefix = str( cfg.get_value( "output_prefix" ) )
        self._pipeline_template = str( cfg.get_value( "pipeline_template" ) )
        self._gpu_count = int( cfg.get_value( "gpu_count" ) )
        self._gt_frames_only = strtobool( cfg.get_value( "gt_frames_only" ) )
        self._backbone = str( cfg.get_value( "backbone" ) )

        # Check variables
        if torch.cuda.is_available() and self._gpu_count < 0:
            self._gpu_count = torch.cuda.device_count()

        # Make required directories and file streams
        if self._train_directory is not None:
            if not os.path.exists( self._train_directory ):
                os.mkdir( self._train_directory )
            self._training_file = os.path.join(
                self._train_directory, self._tmp_training_file )
            self._validation_file = os.path.join(
                self._train_directory, self._tmp_validation_file )
        else:
            self._training_file = self._tmp_training_file
            self._validation_file = self._tmp_validation_file

        from vital.modules.modules import load_known_modules
        load_known_modules()

        self._training_writer = \
          DetectedObjectSetOutput.create( "DetectedObjectSetOutputCoco" )
        self._validation_writer = \
          DetectedObjectSetOutput.create( "DetectedObjectSetOutputCoco" )

        self._training_writer.open( self._training_file )
        self._validation_writer.open( self._validation_file )

        # Initialize persistent variables
        self._training_data = []
        self._validation_data = []
        self._sample_count = 0

    def check_configuration( self, cfg ):
        if not cfg.has_value( "config_file" ) or \
          len( cfg.get_value( "config_file") ) == 0:
            print( "A config file must be specified!" )
            return False
        return True

    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        if len( train_files ) != len( train_dets ):
            print( "Error: train file and groundtruth count mismatch" )
            return
        if categories is not None:
            self._categories = categories.all_class_names()
        for i in range( len( train_files ) + len( test_files ) ):
            if i < len( train_files ):
                filename = train_files[ i ]
                groundtruth = train_dets[ i ]
                self._training_writer.write_set( groundtruth, os.path.abspath( filename ) )
            else:
                filename = test_files[ i-len( train_files ) ]
                groundtruth = test_dets[ i-len( train_files ) ]
                self._validation_writer.write_set( groundtruth, os.path.abspath( filename ) )

    def update_model( self ):
        self._training_writer.complete()
        self._validation_writer.complete()

        gpu_string = ','.join([ str(i) for i in range(0,self._gpu_count) ])

        cmd = [ "python.exe" if os.name == 'nt' else "python",
                "-m",
                "bioharn.detect_fit",
                "--nice=" + self._config_file,
                "--train_dataset=" + self._training_file,
                "--vali_dataset=" + self._validation_file,
                "--workdir=" + self._train_directory,
                "--schedule=ReduceLROnPlateau-p2-c2",
                "--augment=complex",
                "--init=noop",
                "--arch=cascade",
                "--optim=sgd",
                "--lr=1e-3",
                "--max_epoch=50",
                "--input_dims=window",
                "--window_dims=576,576",
                "--window_overlap=0.20",
                "--multiscale=True",
                "--normalize_inputs=True",
                "--workers=4",
                "--sampler_backend=none",
                "--xpu=" + gpu_string,
                "--batch_size=4",
                "--bstep=4",
                "--timeout=1209600",
                "--channels=rgb" ]

        if len( self._backbone ) > 0:
            cmd.append( "--backbone_init=" + self._backbone )

        subprocess.call( cmd )

        print( "\nModel training complete!\n" )

def __vital_algorithm_register__():
    from vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "netharn"

    if algorithm_factory.has_algorithm_impl_name(
      NetHarnTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "PyTorch NetHarn detection training routine", NetHarnTrainer )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
