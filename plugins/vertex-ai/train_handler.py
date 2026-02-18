#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import logging
import subprocess
import threading

logger = logging.getLogger( "viame.vertex-ai.train" )


class TrainHandler:

  def __init__( self, viame_dir, work_dir, model_storage_uri="" ):
    self.viame_dir = viame_dir
    self.work_dir = work_dir
    self.model_storage_uri = model_storage_uri
    self.setup_script = os.path.join( viame_dir, "setup_viame.sh" )

    self._status = "idle"
    self._process = None
    self._log_lines = []
    self._lock = threading.Lock()

  def get_status( self ):
    with self._lock:
      return {
        "state": self._status,
        "log_tail": self._log_lines[ -50: ]
      }

  def run( self, payload ):
    """Run 'viame train' with the given configuration.

    payload fields:
      input_dir  - local path or GCS URI to training data
      output_dir - local path or GCS URI for output model
      config     - path to .conf file (relative to pipelines dir or absolute)
      settings   - dict of key=value config overrides
    """
    with self._lock:
      if self._status == "running":
        return { "error": "Training job already in progress" }
      self._status = "running"
      self._log_lines = []

    input_dir = payload.get( "input_dir", "" )
    output_dir = payload.get( "output_dir",
                              os.path.join( self.work_dir, "training_output" ) )
    config = payload.get( "config", "" )
    settings = payload.get( "settings", {} )

    local_input = self._resolve_gcs_path( input_dir, "training_input" )
    local_output = os.path.join( self.work_dir, "training_output" )
    os.makedirs( local_output, exist_ok=True )

    # Build viame train command
    cmd = "source {} && viame train".format( self.setup_script )

    if config:
      cmd += " -c {}".format( config )

    cmd += " -i {}".format( local_input )
    cmd += " -o {}".format( local_output )

    for key, value in settings.items():
      cmd += " -s {}={}".format( key, value )

    logger.info( "Running: %s", cmd )

    try:
      proc = subprocess.Popen(
        [ "bash", "-c", cmd ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=self.work_dir
      )

      self._process = proc

      for line in proc.stdout:
        line = line.rstrip()
        logger.info( line )
        with self._lock:
          self._log_lines.append( line )

      proc.wait()

      if proc.returncode != 0:
        with self._lock:
          self._status = "failed"
        return {
          "status": "failed",
          "return_code": proc.returncode,
          "log_tail": self._log_lines[ -20: ]
        }

      # Upload output model to GCS if requested
      gcs_output = payload.get( "output_dir", "" )
      if gcs_output.startswith( "gs://" ):
        self._upload_to_gcs( local_output, gcs_output )

      with self._lock:
        self._status = "completed"

      return {
        "status": "completed",
        "output_dir": local_output,
        "model_files": os.listdir( local_output )
      }

    except Exception as e:
      with self._lock:
        self._status = "failed"
      raise

  def _resolve_gcs_path( self, path, local_name ):
    """Download from GCS if path starts with gs://, otherwise return as-is."""
    if not path.startswith( "gs://" ):
      return path

    local_path = os.path.join( self.work_dir, local_name )
    os.makedirs( local_path, exist_ok=True )

    logger.info( "Downloading %s to %s", path, local_path )
    subprocess.check_call( [
      "gsutil", "-m", "cp", "-r", path + "/*", local_path
    ] )
    return local_path

  def _upload_to_gcs( self, local_dir, gcs_uri ):
    """Upload local directory contents to GCS."""
    logger.info( "Uploading %s to %s", local_dir, gcs_uri )
    subprocess.check_call( [
      "gsutil", "-m", "cp", "-r", local_dir + "/*", gcs_uri
    ] )
