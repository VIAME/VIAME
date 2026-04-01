#!/usr/bin/env python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import json
import logging
import subprocess
import threading
import glob

logger = logging.getLogger( "viame.vertex-ai.process" )

# Map VIAME writer type names to the file extension they produce
WRITER_TYPE_EXTENSIONS = {
  "viame_csv": ".csv",
  "coco":      ".json",
  "kw18":      ".kw18",
}


class ProcessHandler:

  def __init__( self, viame_dir, work_dir, default_pipeline="",
                model_storage_uri="", output_type="viame_csv" ):
    self.viame_dir = viame_dir
    self.work_dir = work_dir
    self.default_pipeline = default_pipeline
    self.model_storage_uri = model_storage_uri
    self.output_type = output_type
    self.setup_script = os.path.join( viame_dir, "setup_viame.sh" )
    self.process_video = os.path.join( viame_dir, "configs", "process_video.py" )

    self._status = "idle"
    self._lock = threading.Lock()

  def get_status( self ):
    with self._lock:
      return { "state": self._status }

  def download_model_artifacts( self, storage_uri ):
    """Download model artifacts from AIP_STORAGE_URI at container startup."""
    model_dir = os.path.join( self.work_dir, "category_models" )
    os.makedirs( model_dir, exist_ok=True )

    logger.info( "Downloading model artifacts from %s", storage_uri )
    subprocess.check_call( [
      "gsutil", "-m", "cp", "-r", storage_uri + "/*", model_dir
    ] )

  def run( self, instances, parameters ):
    """Process one or more inputs through a VIAME pipeline.

    Each instance can specify:
      input_path - local path or GCS URI to video/image folder
      pipeline   - pipeline .pipe file to use (optional)
      settings   - dict of pipeline setting overrides (optional)

    Parameters (global):
      frame_rate    - processing frame rate
      output_format - output format (default viame_csv)
    """
    with self._lock:
      self._status = "running"

    results = []

    for instance in instances:
      try:
        result = self._process_single( instance, parameters )
        results.append( result )
      except Exception as e:
        logger.error( "Failed to process instance: %s", str( e ) )
        results.append( { "error": str( e ) } )

    with self._lock:
      self._status = "idle"

    return results

  def _process_single( self, instance, parameters ):
    """Run process_video.py on a single input."""
    input_path = instance.get( "input_path", "" )
    pipeline = instance.get( "pipeline", self.default_pipeline )
    settings = instance.get( "settings", {} )
    frame_rate = parameters.get( "frame_rate", "5" )

    if not input_path:
      return { "error": "No input_path specified" }

    local_input = self._resolve_gcs_path( input_path )

    output_dir = os.path.join( self.work_dir, "output" )
    os.makedirs( output_dir, exist_ok=True )

    output_type = self.output_type
    output_ext = WRITER_TYPE_EXTENSIONS.get( output_type, ".csv" )

    # Build process_video.py command
    cmd = "source {} && python {}".format(
      self.setup_script, self.process_video
    )
    cmd += " -i {}".format( local_input )
    cmd += " -o {}".format( output_dir )
    cmd += " -p {}".format( pipeline )
    cmd += " -frate {}".format( frame_rate )
    cmd += " -output-ext {}".format( output_ext )
    cmd += " --no-reset-prompt"

    # Override the writer type in both detector and track writers
    cmd += " -s detector_writer:writer:type={}".format( output_type )
    cmd += " -s track_writer:writer:type={}".format( output_type )

    for key, value in settings.items():
      cmd += " -s {}={}".format( key, value )

    logger.info( "Running: %s", cmd )

    proc = subprocess.run(
      [ "bash", "-c", cmd ],
      capture_output=True,
      text=True,
      cwd=self.work_dir
    )

    if proc.returncode != 0:
      logger.error( "process_video.py stderr: %s", proc.stderr )
      return {
        "error": "Processing failed",
        "return_code": proc.returncode,
        "stderr": proc.stderr[ -500: ]
      }

    # Collect output detection and track files
    outputs = []
    for suffix in ( "_detections", "_tracks" ):
      pattern = os.path.join(
        output_dir, "**", "*" + suffix + output_ext
      )
      for out_file in glob.glob( pattern, recursive=True ):
        with open( out_file, "r" ) as f:
          content = f.read()
        if output_ext == ".json":
          content = json.loads( content )
        outputs.append( {
          "file": os.path.basename( out_file ),
          "content": content
        } )

    return {
      "status": "completed",
      "input": input_path,
      "outputs": outputs
    }

  def _resolve_gcs_path( self, path ):
    """Download from GCS if needed, return local path."""
    if not path.startswith( "gs://" ):
      return path

    basename = os.path.basename( path.rstrip( "/" ) )
    local_path = os.path.join( self.work_dir, "input", basename )
    os.makedirs( os.path.dirname( local_path ), exist_ok=True )

    logger.info( "Downloading %s to %s", path, local_path )

    if "." in basename:
      subprocess.check_call( [ "gsutil", "cp", path, local_path ] )
    else:
      os.makedirs( local_path, exist_ok=True )
      subprocess.check_call( [
        "gsutil", "-m", "cp", "-r", path + "/*", local_path
      ] )

    return local_path
