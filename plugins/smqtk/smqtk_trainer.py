# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function
from __future__ import division

from kwiver.vital.algo import TrainDetector
from kwiver.vital.algo import DetectedObjectSetOutput

from distutils.util import strtobool
from shutil import copyfile

import os
import sys
import signal
import sys
import subprocess
import threading
import time

# Use local module
from . import smqtk_train_svm_models as trainer

class SMQTKTrainer( TrainDetector ):
    """
    Implementation of TrainDetector class
    """
    def __init__( self ):
        TrainDetector.__init__( self )

        self._root_dir = ""

        self._mode = "detector"
        self._gt_frames_only = False
        self._ingest_pipeline = ""
        self._pipeline_template = ""
        self._output_directory = "category_models"

        self._tmp_directory = "svm_training"

        self._detection_file = os.path.join( self._tmp_directory, "train_dets.csv" )
        self._list_file = os.path.join( self._tmp_directory, "train_list.txt" )
        self._label_file = os.path.join( self._tmp_directory, "labels.txt" )

    def get_configuration( self ):
        cfg = super( TrainDetector, self ).get_configuration()
        cfg.set_value( "mode", self._mode )
        cfg.set_value( "gt_frames_only", str( self._gt_frames_only ) )
        cfg.set_value( "ingest_pipeline", self._ingest_pipeline )
        cfg.set_value( "pipeline_template", self._pipeline_template )
        cfg.set_value( "output_directory", self._output_directory )
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._mode = str( cfg.get_value( "mode" ) )
        self._gt_frames_only = strtobool( cfg.get_value( "gt_frames_only" ) )
        self._ingest_pipeline = str( cfg.get_value( "ingest_pipeline" ) )
        self._pipeline_template = str( cfg.get_value( "pipeline_template" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )

        if not self._ingest_pipeline:
            if self._mode == "detector":
                pipe_file = "index_default.svm.pipe"
            elif self._gt_frames_only:
                pipe_file = "index_frame.svm.annot_only.pipe"
            else:
                pipe_file = "index_frame.svm.pipe"
            self._ingest_pipeline = os.path.join( "pipelines", pipe_file )

        self._viame_install = os.environ['VIAME_INSTALL']

        if not os.path.exists( self._tmp_directory ):
            os.mkdir( self._tmp_directory )

        self._detection_writer = DetectedObjectSetOutput.create( "viame_csv" )
        writer_conf = self._detection_writer.get_configuration()
        self._detection_writer.set_configuration( writer_conf )

        self._detection_writer.open( self._detection_file )
        self._image_list_writer = open( self._list_file, "w" )
        self._label_writer = open( self._label_file, "w" )

        if len( self._viame_install ) < 0:
            print( "ERROR: VIAME_INSTALL OS variable required" )
            return False
        return True

    def check_configuration( self, cfg ):
        return True

    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        if categories is not None:
            self._categories = categories.all_class_names()
        for i in range( len( train_files ) + len( test_files ) ):
            if i < len( train_files ):
                filename = train_files[ i ]
                groundtruth = train_dets[ i ]
                self._image_list_writer.write( filename + "\n" )
                self._detection_writer.write_set( groundtruth, os.path.split( filename )[1] )
            else:
                filename = test_files[ i-len( train_files ) ]
                groundtruth = test_dets[ i-len( train_files ) ]
                self._image_list_writer.write( filename + "\n" )
                self._detection_writer.write_set( groundtruth, os.path.split( filename )[1] )

    def update_model( self ):

        for cat in self._categories:
            self._label_writer.write( cat + "\n" )

        self._detection_writer.complete()
        self._image_list_writer.close()
        self._label_writer.close()

        cmd = [ "python.exe" if os.name == 'nt' else "python" ]

        script = os.path.join( self._viame_install, "configs", "process_video.py" )

        cmd += [ script,
                 "--init",
                 "-l", self._list_file,
                 "-p", self._ingest_pipeline,
                 "-logs", "PIPE",
                 "-o", "database",
                 "--build-index",
                 "-gt-file", self._detection_file,
                 "-lbl-file", self._label_file,
                 "--no-reset-prompt",
                 "-install", self._viame_install ]

        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal( signal.SIGINT, lambda signal, frame: self.interupt_handler() )
            signal.signal( signal.SIGTERM, lambda signal, frame: self.interupt_handler() )

        self.proc = subprocess.Popen( cmd )
        self.proc.wait()

        trainer.generate_svm_models()

        fin = open( self._pipeline_template )
        fout = open( os.path.join( self._output_directory, "detector.pipe" ), 'w' )
        for s in list( fin ):
            fout.write( s )
        fout.close()
        fin.close()

    def interupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        sys.exit( 0 )

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "smqtk"

    if algorithm_factory.has_algorithm_impl_name(
      SMQTKTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "PyTorch SMQTK detection training routine", SMQTKTrainer )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
