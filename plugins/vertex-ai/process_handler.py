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
import tempfile

logger = logging.getLogger( "viame.vertex-ai.process" )

# Map VIAME writer type names to the file extension they produce
WRITER_TYPE_EXTENSIONS = {
  "viame_csv": ".csv",
  "coco":      ".json",
  "kw18":      ".kw18",
}

VIDEO_EXTENSIONS = set( (
  "3qp", "3g2", "amv", "asf", "avi", "drc", "gif", "gifv",
  "f4v", "f4p", "f4a", "f4b", "flv", "m4v", "mkv", "mp4",
  "m4p", "mpg", "mpg2", "mp2", "mpeg", "mpe", "mpv", "mng",
  "mts", "m2ts", "mov", "mxf", "nsv", "ogg", "ogv", "qt",
  "roq", "rm", "rmvb", "svi", "webm", "wmv", "vob", "yuv",
) )

IMAGE_EXTENSIONS = set( (
  "bmp", "dds", "gif", "heic", "jpg", "jpeg", "png", "psd",
  "psp", "pspimage", "tga", "thm", "tif", "tiff", "yuv",
) )


def _is_video( path ):
  """Return True if the path looks like a video file."""
  ext = os.path.splitext( path )[1].lstrip( "." ).lower()
  return ext in VIDEO_EXTENSIONS


def _make_image_list( folder, list_path ):
  """Create a newline-delimited image list file from a folder of images."""
  images = sorted(
    f for f in os.listdir( folder )
    if os.path.splitext( f )[1].lstrip( "." ).lower() in IMAGE_EXTENSIONS
  )
  with open( list_path, "w" ) as fh:
    for img in images:
      fh.write( os.path.join( folder, img ) + "\n" )
  return list_path


class ProcessHandler:

  def __init__( self, viame_dir, work_dir, default_pipeline="",
                model_storage_uri="", output_type="coco",
                frame_rate="5" ):
    self.viame_dir = viame_dir
    self.work_dir = work_dir
    self.default_pipeline = default_pipeline
    self.model_storage_uri = model_storage_uri
    self.output_type = output_type
    self.frame_rate = frame_rate
    self.setup_script = os.path.join( viame_dir, "setup_viame.sh" )
    self.pipeline_dir = os.path.join( viame_dir, "configs" )

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
      input_path  - local path or GCS URI to a video or image folder
      input_paths - list of paths for stereo (e.g. [left, right])
      pipeline    - pipeline .pipe file to use (optional)
      settings    - dict of pipeline setting overrides (optional)

    Parameters (global):
      frame_rate  - target processing frame rate
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

  # -------------------------------------------------------------------
  # Core runner
  # -------------------------------------------------------------------

  def _process_single( self, instance, parameters ):
    """Build and execute a viame_runner command for a single request."""

    pipeline = instance.get( "pipeline", self.default_pipeline )
    settings = instance.get( "settings", {} )
    frame_rate = parameters.get( "frame_rate", self.frame_rate )
    output_type = self.output_type
    output_ext = WRITER_TYPE_EXTENSIONS.get( output_type, ".csv" )

    output_dir = os.path.join( self.work_dir, "output" )
    os.makedirs( output_dir, exist_ok=True )

    # Resolve input(s) — single or stereo
    input_paths = instance.get( "input_paths", [] )
    if not input_paths:
      single = instance.get( "input_path", "" )
      if not single:
        return { "error": "No input_path or input_paths specified" }
      input_paths = [ single ]

    local_inputs = [ self._resolve_gcs_path( p ) for p in input_paths ]
    num_cameras = len( local_inputs )

    # Resolve the pipeline file
    pipeline_file = pipeline
    if not os.path.isabs( pipeline_file ):
      pipeline_file = os.path.join( self.pipeline_dir, pipeline )

    # Start building the kwiver runner command
    cmd = [ "viame", "runner", pipeline_file ]

    # Wire each input stream
    for idx, local_input in enumerate( local_inputs ):
      prefix = "input" + ( str( idx + 1 ) if num_cameras > 1 else "" ) + ":"
      is_vid = _is_video( local_input )

      if is_vid:
        cmd += [ "-s", prefix + "video_filename=" + local_input ]
        cmd += [ "-s", prefix + "video_reader:type=vidl_ffmpeg" ]
      elif os.path.isdir( local_input ):
        list_path = os.path.join(
          self.work_dir, "input_list_{}.txt".format( idx ) )
        _make_image_list( local_input, list_path )
        cmd += [ "-s", prefix + "video_filename=" + list_path ]
        cmd += [ "-s", prefix + "video_reader:type=image_list" ]
      else:
        # Assume it is already an image list file
        cmd += [ "-s", prefix + "video_filename=" + local_input ]
        cmd += [ "-s", prefix + "video_reader:type=image_list" ]

    # Frame rate / downsampler
    cmd += [ "-s", "downsampler:target_frame_rate=" + str( frame_rate ) ]

    # Output writers — one set per camera
    basename = os.path.splitext(
      os.path.basename( local_inputs[0] ) )[0]

    if num_cameras == 1:
      cmd += self._writer_settings(
        output_dir, basename, output_type, output_ext )
    else:
      camera_names = [ "left", "right", "center" ]
      for cid in range( num_cameras ):
        cam_name = camera_names[cid] if cid < len( camera_names ) \
          else "cam" + str( cid + 1 )
        cmd += self._writer_settings(
          output_dir, cam_name, output_type, output_ext, cid + 1 )
        # Also set unnumbered writers for the first camera since some
        # pipelines use detector_writer/track_writer without a suffix
        if cid == 0:
          cmd += self._writer_settings(
            output_dir, cam_name, output_type, output_ext )

    # Extra per-request setting overrides
    for key, value in settings.items():
      cmd += [ "-s", "{}={}".format( key, value ) ]

    # Run via bash so we can source the VIAME environment first
    shell_cmd = "source {} && {}".format(
      self.setup_script,
      " ".join( _quote( t ) for t in cmd )
    )

    logger.info( "Running: %s", shell_cmd )

    proc = subprocess.run(
      [ "bash", "-c", shell_cmd ],
      capture_output=True,
      text=True,
      cwd=self.pipeline_dir
    )

    if proc.returncode != 0:
      logger.error( "viame runner stderr: %s", proc.stderr )
      return {
        "error": "Processing failed",
        "return_code": proc.returncode,
        "stderr": proc.stderr[ -500: ]
      }

    # Collect output detection and track files
    outputs = []
    for suffix in ( "_detections", "_tracks" ):
      pattern = os.path.join(
        output_dir, "**", "*" + suffix + output_ext )
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
      "input": input_paths[0] if num_cameras == 1 else input_paths,
      "outputs": outputs
    }

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  @staticmethod
  def _writer_settings( output_dir, basename, output_type, output_ext,
                        cid=None ):
    """Return -s flags for detection and track writers."""
    det_file = os.path.join( output_dir, basename + "_detections" + output_ext )
    trk_file = os.path.join( output_dir, basename + "_tracks" + output_ext )

    det_prefix = "detector_writer" + ( str( cid ) if cid else "" ) + ":"
    trk_prefix = "track_writer" + ( str( cid ) if cid else "" ) + ":"

    out = []
    out += [ "-s", det_prefix + "file_name=" + det_file ]
    out += [ "-s", det_prefix + "writer:type=" + output_type ]
    out += [ "-s", trk_prefix + "file_name=" + trk_file ]
    out += [ "-s", trk_prefix + "writer:type=" + output_type ]
    return out

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


def _quote( s ):
  """Shell-quote a string if it contains spaces or special characters."""
  if any( c in s for c in " \t'\"\\$" ):
    return "'" + s.replace( "'", "'\\''" ) + "'"
  return s
