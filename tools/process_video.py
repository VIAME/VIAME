#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
import argparse
import contextlib
import itertools
import signal
import subprocess
import tempfile
import threading

try:
  import queue # Python 3
except ImportError:
  import Queue as queue # Python 2

sys.dont_write_bytecode = True

import database_tool

# Character short-cuts and global constants
div = os.path.sep
lb  = '\n'

lb1 = lb
lb2 = lb * 2
lb3 = lb * 3

default_gt_ext = ".csv"
default_pipe_ext = ".pipe"

detection_ext = "_detections" + default_gt_ext
track_ext = "_tracks" + default_gt_ext

default_pipeline = "pipelines" + div + "index_default" + default_pipe_ext
no_pipeline = "none"

# Global flag to see if any video has successfully completed processing
any_video_complete = False

def list_elems_in_dir( folder ):
  if not os.path.exists( folder ) and os.path.exists( folder + ".lnk" ):
    folder = folder + ".lnk"
  folder = folder if not os.path.islink( folder ) else os.readlink( folder )
  if not os.path.isdir( folder ):
    exit_with_error( "Input folder \"" + folder + "\" does not exist" )
  return [
    os.path.join( folder, f ) for f in sorted( os.listdir( folder ) )
    if not f.startswith('.')
  ]

def list_files_in_dir_w_ext( folder, extension ):
  return [ f for f in list_elems_in_dir( folder ) if f.endswith( extension ) ]

def has_valid_ext( f, ext_list ):
  for ext in ext_list:
    if f.lower().endswith( ext ):
      return True
  return False

def has_file_with_extension( folder, extension ):
  for filename in list_files_in_dir_w_ext( folder, extension ):
    if filename.endswith( extension ):
      return True
  return False

def list_files_in_dir_w_exts( folder, extensions ):
  ext_list = extensions.split(";")
  return [ f for f in list_elems_in_dir( folder ) if has_valid_ext( f, ext_list ) ]

def ordered_return( retvar, refvar, cats ):
  output = []
  for cat in cats:
    output.append( retvar[ refvar.index( cat ) ] )
  return output

def check_for_multicam_folder( folder, subfolders = None ):
  if not os.path.isdir( folder ):
    return False, []
  if subfolders is None:
    subfolders = [ f for f in list_elems_in_dir( folder ) if os.path.isdir( f ) ]
  lowercase = [ os.path.basename( f ).lower().replace( "/", "" ) for f in subfolders ]
  if len( subfolders ) == 3:
    if "left" in lowercase and "center" in lowercase and "right" in lowercase:
      return True, ordered_return( subfolders, lowercase, [ "left", "center", "right" ] )
    if "star" in lowercase and "center" in lowercase and "port" in lowercase:
      return True, ordered_return( subfolders, lowercase, [ "star", "center", "port" ] )
  elif len( subfolders ) == 2:
    if "left" in lowercase and "right" in lowercase:
      return True, ordered_return( subfolders, lowercase, [ "left", "right" ] )
  return False, []

def auto_folder_recurse( folder, video_exts, image_exts ):
  files = list_files_in_dir_w_exts( folder, video_exts )
  has_video_files = len( files ) > 0
  image_files = list_files_in_dir_w_exts( folder, image_exts )
  has_image_files = len( image_files ) > 0
  has_several_images = len( image_files ) > 2
  subfolders = [ f for f in list_elems_in_dir( folder ) if os.path.isdir( f ) ]
  has_subfolders = len( subfolders ) > 0
  is_multi_cam_folder, _ = check_for_multicam_folder( folder, subfolders )
  if is_multi_cam_folder or has_several_images or \
     ( has_image_files and not has_video_files ):
    files.append( folder )
  if is_multi_cam_folder:
    return files
  elif has_subfolders:
    for f in subfolders:
      files.extend( auto_folder_recurse( f, video_exts, image_exts ) )
  elif not has_subfolders and not has_image_files and not has_video_files:
    if len( list_elems_in_dir( folder ) ) > 2:
      files.extend( folder )
  return files

def auto_identify_data( folder, video_exts, image_exts ):
  entries = auto_folder_recurse( folder, video_exts, image_exts )
  print( "\nFound " + str( len( entries ) ) + " items for possible processing\n" )
  for i in entries:
    print( i )
  return entries

# Default message logging
def log_info( msg ):
  sys.stdout.write( msg )
  sys.stdout.flush()

# Create a directory if it doesn't exist
def create_dir( dirname, logging=True, recreate=False, prompt=True ):
  if dirname == '.' or dirname == "":
    return
  if recreate:
    if os.path.exists( dirname ):
      if not prompt or \
         database_tool.query_yes_no( lb1 + "Reset output folder: " + dirname + "?" ):
        if logging:
          log_info( "Removing " + dirname + lb )
        shutil.rmtree( dirname )
      elif prompt:
        sys.exit(0)
    else:
      log_info( lb )
  if not os.path.exists( dirname ):
    if logging:
      log_info( "Creating " + dirname + lb )
    os.makedirs( dirname )

CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"

def get_real_gpu_index(n):
  """Return the real index for the nth GPU as a string.  This respects
  CUDA_VISIBLE_DEVICES

  """
  cvd = os.environ.get( CUDA_VISIBLE_DEVICES )
  if not cvd:  # Treat empty string and None the same
    return str(n)
  # This is an attempt to respect the fact that an invalid index hides
  # the GPUs listed after it
  cvd_parsed = list( itertools.takewhile( lambda i: not i.startswith('-'),
                                          cvd.split(',') ) )
  if 0 <= n < len( cvd_parsed ):
    return cvd_parsed[n]
  else:
    raise IndexError('Only {} visible GPUs; you asked for number {}!'
                     .format( len( cvd_parsed ), n) )

def execute_command( cmd, stdout=None, stderr=None, gpu=None ):
  if gpu is None:
    env = None
  else:
    env = dict( os.environ )
    env[ CUDA_VISIBLE_DEVICES ] = get_real_gpu_index( gpu )
  return subprocess.call( cmd, stdout=stdout, stderr=stderr, env=env )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def get_pipeline_cmd( debug=False ):
  if os.name == 'nt':
    if debug:
      return [ 'kwiver.exe', 'runner' ]
    else:
      return [ 'kwiver.exe', 'runner' ]
  else:
    if debug:
      return [ 'gdb', '--args', 'kwiver', 'runner' ]
    else:
      return [ 'kwiver', 'runner' ]

def get_python_cmd():
  if os.name == 'nt':
    return [ 'python.exe' ]
  else:
    return [ 'python' ]

def exit_with_error( error_str, force=False ):
  log_info( lb1 + 'ERROR: ' + error_str + lb2 )
  # Kill this process to end all threads
  if not isinstance( threading.current_thread(), threading._MainThread ):
    if os.name == 'nt':
      os.kill( os.getpid(), signal.SIGTERM )
    else:
      os.kill( os.getpid(), signal.SIGKILL )
  # Default exit case, if main thread
  sys.exit(0)

def check_file( filename ):
  if not os.path.exists( filename ):
    exit_with_error( "Unable to find: " + filename )
  return filename

@contextlib.contextmanager
def get_log_output_files( output_prefix ):
  if os.name == 'nt':
    with open( output_prefix + '.out.txt', 'w' ) as fo, \
         open( output_prefix + '.err.txt', 'w' ) as fe:
      yield dict( stdout=fo, stderr=fe)
  else:
    with open( output_prefix + '.txt', 'w' ) as fo:
      yield dict( stdout=fo, stderr=fo )

def find_file( filename ):
  if( os.path.exists( filename ) ):
    return filename
  elif os.path.exists( get_script_path() + div + filename ):
    return get_script_path() + div + filename
  else:
    exit_with_error( "Unable to find " + filename )

def rate_from_gt( filename ):
  if not os.path.exists( filename ):
    return ""
  with open( filename ) as fin:
    head = [ next( fin ) for x in range( 2 ) ]
    for line in head:
      if "fps:" in line:
        fps = line.split( "fps:", 1 )[1].split( "," )[0]
        fps = fps[1:] if len( fps ) > 1 and fps[0] == " " else fps
        log_info( "Using FPS " + fps + "... " )
        return fps
  return ""

def make_filelist_for_dir( input_dir, output_dir, output_name ):
  # The most common extension in the folder is most likely images.
  # Sometimes people have small text files alongside the images
  # so just choose the most common filetype.
  exts = dict()
  files = dict()
  for f in sorted( os.listdir( input_dir ) ):
    f_fp = os.path.join( input_dir, f )
    if os.path.isfile( f_fp ):
      _, ext = os.path.splitext( f )
      if ext in exts:
        exts[ext] += 1
        files[ext].append( f_fp )
      else:
        exts[ext] = 1
        files[ext] = [ f_fp ]
  if len(exts) == 0:
    return ""

  top_ext = sorted( exts, key=exts.get, reverse=True )[0]

  # Write out list to file
  output_file = os.path.join( output_dir, output_name + ".txt" )
  fout = open( output_file, "w" )
  for f in files[top_ext]:
    fout.write( os.path.abspath( f + lb1 ) )
  fout.close()
  return output_file

# Other helpers
def signal_handler( signal, frame ):
  log_info( lb1 )
  exit_with_error( 'Processing aborted, see you next time' )

def file_length( filename ):
  if not os.path.exists( filename ):
    exit_with_error( filename + " does not exist" )
  with open( filename, 'r' ) as f:
    for i, l in enumerate( f ):
      pass
  return i + 1

def split_image_list( image_list_file, n, dir ):
  """Create and return the paths to n temp files that when appended
  reproduce the original file.  The names are created
  deterministically like "orig_name_part0.ext", "orig_name_part1.ext",
  etc., but with the original name used as is when n == 1.

  Existing files with the same names are overwritten without question.
  Deleting the files is the responsibility of the caller.

  """
  input_basename = os.path.basename( image_list_file )
  if n == 1:
    new_file_names = [ input_basename ]
  else:
    prefix, suffix = os.path.splitext( input_basename )
    num_width = len( str( n - 1 ) )
    new_file_names = [
      prefix + '_part{:0{}}'.format( i, num_width ) + suffix
      for i in range( n )
    ]
  new_file_names = [ os.path.join( dir, fn ) for fn in new_file_names ]

  try:
    # Build manually to have the intermediate state in case of error
    temp_files = []
    divisor = math.floor( file_length( image_list_file ) / n ) + 1
    for fn in new_file_names:
      temp_files.append( open( fn, 'w' ) )
    with open( image_list_file ) as f:
      for i, line in enumerate( f ):
        temp_index = int( math.floor( i / divisor ) )
        temp_files[ temp_index ].write( line )
  finally:
    for f in temp_files:
      f.close()
  return new_file_names

def fset( setting_str ):
  return ['-s', setting_str]

def pipe_starts_with( filename, substr ):
  return os.path.basename( filename ).startswith( substr )

def detection_output_settings_list( options, basename, cid = None ):
  output_dir = options.output_directory

  det_writer_str = 'detector_writer' + ( str( cid ) + ':' if cid else ':' )
  trk_writer_str = 'track_writer' + ( str( cid ) + ':' if cid else ':' )

  if pipe_starts_with( options.pipeline, "filter_" ) or \
     pipe_starts_with( options.pipeline, "transcode_" ):
    detection_file = output_dir + div + basename + default_gt_ext
    track_file = output_dir + div + basename + default_gt_ext
  else:
    detection_file = output_dir + div + basename + detection_ext
    track_file = output_dir + div + basename + track_ext

  return list(itertools.chain(
    fset( det_writer_str + 'file_name=' + detection_file ),
    fset( det_writer_str + 'stream_identifier=' + basename ),
    fset( trk_writer_str + 'file_name=' + track_file ),
    fset( trk_writer_str + 'stream_identifier=' + basename ),
  ))

def search_output_settings_list( options, basename ):
  output_dir = options.output_directory
  return list(itertools.chain(
    fset( 'track_writer_db:writer:db:video_name=' + basename ),
    fset( 'track_writer_kw18:file_name=' + output_dir + div + basename + '.kw18' ),
    fset( 'descriptor_writer_db:writer:db:video_name=' + basename ),
    fset( 'track_descriptor:uid_basename=' + basename ),
    fset( 'kwa_writer:output_directory=' + output_dir ),
    fset( 'kwa_writer:base_filename=' + basename ),
    fset( 'kwa_writer:stream_id=' + basename ),
  ))

def plot_settings_list( options, basename ):
  output_dir = options.output_directory

  return list(itertools.chain(
    fset( 'detector_writer:file_name=' + output_dir + div + basename + detection_ext ),
    fset( 'kwa_writer:output_directory=' + output_dir ),
    fset( 'kwa_writer:base_filename=' + basename ),
    fset( 'kwa_writer:stream_id=' + basename ),
  ))

def archive_dimension_settings_list( options ):
  if options.archive_width:
    return list(itertools.chain(
      fset( 'kwa_writer:fixed_col_count=' + options.archive_width ),
      fset( 'kwa_writer:fixed_row_count=' + options.archive_height ),
    ))
  return []

def object_detector_settings_list( options ):
  if options.detection_threshold:
    return list( itertools.chain(
      fset( 'detector:detector:darknet:thresh=' + options.detection_threshold ),
      fset( 'detector1:detector:darknet:thresh=' + options.detection_threshold ),
      fset( 'detector2:detector:darknet:thresh=' + options.detection_threshold ),
      fset( 'detector_filter:filter:class_probablity_filter:threshold=' + options.detection_threshold ),
    ))
  return []

def object_tracker_settings_list( options ):
  if options.tracker_threshold:
    return list( itertools.chain(
      fset( 'track_initializer:track_initializer:threshold:'
            'filter:class_probablity_filter:threshold=' + options.tracker_threshold ),
      fset( 'tracker:detection_select_threshold=' + options.tracker_threshold ),
    ))
  return []

def video_frame_rate_settings_list( options, frame_rate = None, source_rate = None ):
  output = []
  if source_rate:
    output += fset( 'input:frame_time=' + str( 1.0 / float( source_rate ) ) )
  elif options.input_frame_rate:
    output += fset( 'input:frame_time=' + str( 1.0 / float( options.input_frame_rate ) ) )
  if frame_rate:
    output += fset( 'downsampler:target_frame_rate=' + frame_rate )
  elif options.frame_rate:
    output += fset( 'downsampler:target_frame_rate=' + options.frame_rate )

  if options.batch_size:
    output += fset( 'downsampler:burst_frame_count=' + options.batch_size )
  if options.batch_skip:
    output += fset( 'downsampler:burst_frame_break=' + options.batch_skip )
  if options.start_time:
    output += fset( 'downsampler:start_time=' + options.start_time )
  if options.duration:
    output += fset( 'downsampler:duration=' + options.duration )
  return output

def groundtruth_reader_settings_list( options, gt_files, basename, gpu_id, gt_type ):
  output = []
  if len( gt_files ) == 0:
    exit_with_error( "Directory " + basename + " contains no GT files" )
  elif len( gt_files ) > 1:
    exit_with_error( "Directory " + basename + " contains ambiguous annotation files" )
  else:
    if gpu_id > 0:
      output_extension = str( gpu_id ) + '.lbl'
    else:
      output_extension = 'lbl'

    if options.label_file:
      lbl_file = options.label_file
    else:
      lbl_file = options.input_dir + "/labels.txt"
      if not os.path.exists( lbl_file ):
        lbl_file = "labels.txt"

    output += fset( 'detection_reader:file_name=' + gt_files[0] )
    output += fset( 'detection_reader:reader:type=' + gt_type )
    output += fset( 'track_reader:file_name=' + gt_files[0] )
    output += fset( 'track_reader:reader:type=' + gt_type )
    output += fset( 'write_descriptor_ids:category_file=' + lbl_file )
    output += fset( 'write_descriptor_ids:output_directory=' + options.output_directory )
    output += fset( 'write_descriptor_ids:output_extension=' + output_extension )
  return output

def remove_quotes( input_str ):
  return input_str.replace( "\"", "" )

def add_final_list_csv( args, video_list ):
  if len( video_list ) == 0:
    return
  for video in video_list:
    if video.endswith( "_part0.txt" ):
      output_file = video_list[0].replace( "_part0.txt", detection_ext )
      output_stream = open( output_file, "w" )
      id_adjustment = 0
      is_first = True
      used_ids = set()
      last_id = 0
    input_stream = open( video.replace( ".txt", detection_ext ), "r" )
    id_mappings = dict()
    for line in input_stream:
      if len( line ) > 0 and ( line[0] == '#' or line[0:9] == 'target_id' ):
        if is_first:
          output_stream.write( line )
        continue
      parsed_line = line.rstrip().split(',')
      if len( parsed_line ) < 2:
        continue
      orig_id = int( parsed_line[0] )
      if orig_id in id_mappings:
        final_id = id_mappings[ orig_id ]
      elif orig_id in used_ids:
        last_id = last_id + 1
        final_id = last_id
        id_mappings[ orig_id ] = final_id
        used_ids.add( final_id )
      else:
        final_id = orig_id
        id_mappings[ orig_id ] = orig_id
        used_ids.add( orig_id )
      last_id = max( last_id, final_id )
      parsed_line[0] = str( final_id )
      parsed_line[2] = str( int( parsed_line[2] ) + id_adjustment )
      output_stream.write( ','.join( parsed_line ) + '\n' )
    id_adjustment = id_adjustment + file_length( video )
    input_stream.close()
    is_first = False

# Process a single data item (image list, folder, or video)
def process_using_kwiver( input_path, options, is_image_list=False,
                          base_name_override='', cpu=0, gpu=None ):

  # Generic settings shared across function
  multi_threaded = ( options.gpu_count * options.pipes > 1 )
  auto_detect_gt = ( options.auto_detect_gt )
  use_gt = ( options.gt_file or auto_detect_gt )
  output_dir = options.output_directory
  is_multi_cam, camera_folders = check_for_multicam_folder( input_path )
  input_paths = []
  camera_names = []

  # Output naming and directory formation if necessary
  if os.path.isdir( input_path ):
    input_dir = os.path.abspath( options.input if options.input else options.input_dir )
    input_basename = os.path.relpath( os.path.abspath( input_path ), input_dir )
    if options.build_index:
      input_basename = input_basename.replace( div, "_" )
      input_basename = input_basename.replace( " ", "_" )
  else:
    input_basename = os.path.basename( input_path )
  output_subdir = output_dir + div + input_basename
  input_ext = os.path.splitext( input_path )[1]

  if not os.path.exists( output_subdir ) and \
     not options.build_index and \
     ( os.path.isdir( input_path ) or \
       "filter_" in options.pipeline or \
       "transcode_" in options.pipeline ):
    os.makedirs( output_subdir )

  # GPU checks for logging statements
  try:
    import torch
    has_gpu = torch.cuda.is_available()
  except Exception as e:
    has_gpu = False
  if gpu is None and has_gpu:
    gpu = 0

  # Main logging statements
  if has_gpu:
    if multi_threaded:
      log_info( 'Processing: {} on GPU {}'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Processing: {} on GPU... '.format( input_basename ) )
  else:
    if multi_threaded:
      log_info( 'Processing: {} on CPU {}'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Processing: {} on CPU... '.format( input_basename ) )

  # Get video name without extension and full path
  if base_name_override:
    basename_no_ext = base_name_override
  else:
    basename_no_ext = os.path.splitext( input_basename )[0]

  # Formulate input setting string
  if auto_detect_gt:
    if options.auto_detect_gt == 'habcam' or 'csv' in options.auto_detect_gt:
      gt_ext = '.csv'
    elif options.auto_detect_gt[0] != '.':
      gt_ext = '.' + options.auto_detect_gt
    else:
      gt_ext = options.auto_detect_gt

  if not is_image_list and \
      ( input_ext == '.csv' or input_ext == '.txt' or input_path == "__pycache__" ):
    if multi_threaded:
      log_info( 'Skipped {} on GPU {}'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Skipped' + lb1 )
    return
  elif not os.path.exists( input_path ):
    if multi_threaded:
      log_info( 'Skipped {} on GPU {}'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Skipped' + lb1 )
    return
  elif os.path.isdir( input_path ):
    if auto_detect_gt:
      gt_files = list_files_in_dir_w_ext( input_path, gt_ext )
    if is_multi_cam:
      for camera_folder in camera_folders:
        camera_name = os.path.basename( camera_folder )
        camera_names.append( camera_name )
        input_paths.append( make_filelist_for_dir( camera_folder, output_dir, camera_name ) )
      input_path = input_paths[0]
    else:
      input_path = make_filelist_for_dir( input_path, output_dir, basename_no_ext )
    if not input_path:
      if multi_threaded:
        log_info( 'Skipped {} on GPU {}'.format( input_basename, gpu ) + lb1 )
      else:
        log_info( 'Skipped' + lb1 )
      return
    is_image_list = True
  elif auto_detect_gt:
    input_path = os.path.dirname( os.path.abspath( input_path ) )
    all_gt_files = list_files_in_dir_w_ext( input_path, gt_ext )
    better_fit = [ i for i in all_gt_files if basename_no_ext in i ]
    best_fit = [ i for i in better_fit if basename_no_ext + ".csv" in i ]
    if len( best_fit ) > 0:
      gt_files = best_fit
    elif len( better_fit ) > 0:
      gt_files = better_fit
    else:
      gt_files = all_gt_files

  # Begin to formulate external CLI call

  # For single camera case
  input_settings = fset( 'input:video_filename=' + input_path )
  if not is_image_list:
    input_settings += fset( 'input:video_reader:type=vidl_ffmpeg' )
  elif options.ts_from_file:
    input_settings += fset( 'input:video_reader:type=add_timestamp_from_filename' )

  # For multi camera case
  for idx, path in enumerate( input_paths ):
    input_str = 'input' + str( idx ) + ':'
    input_settings = fset( input_str + 'video_filename=' + input_path )
    if not is_image_list:
      input_settings += fset( input_str + 'video_reader:type=vidl_ffmpeg' )

  # Base command
  command = ( get_pipeline_cmd( options.debug ) +
              [ find_file( options.pipeline ) ] +
              input_settings )

  if use_gt and len( gt_files ) > 0:
    gt_rate = rate_from_gt( gt_files[0] )
    if not args.input_frame_rate and is_image_list:
      source_rate = gt_rate if gt_rate else args.frame_rate
    else:
      source_rate = None
    command += video_frame_rate_settings_list( options, gt_rate, source_rate )
  else:
    if not args.input_frame_rate and is_image_list:
      source_rate = args.frame_rate
    else:
      source_rate = None
    command += video_frame_rate_settings_list( options, None, source_rate )

  # Additional options
  command += detection_output_settings_list( options, basename_no_ext )
  command += search_output_settings_list( options, basename_no_ext )
  command += archive_dimension_settings_list( options )
  command += object_detector_settings_list( options )
  command += object_tracker_settings_list( options )

  for camera_id, camera_name in enumerate( camera_names ):
    command += detection_output_settings_list( options, camera_name, camera_id)

  if options.write_svm_info and not use_gt:
    if len( options.input_detections ) == 0:
      exit_with_error( "Input detections must be specified to write out svm header info" )
    if not os.path.exists( options.input_detections ):
      exit_with_error( "Unable to find input detections" )
    gt_files = [ options.input_detections ]
  if use_gt or options.write_svm_info:
    gt_type = options.auto_detect_gt if auto_detect_gt else "viame_csv"
    gt_files = [ options.gt_file ] if not auto_detect_gt else gt_files
    command += groundtruth_reader_settings_list( options, gt_files, basename_no_ext, gpu, gt_type )

  if ( not is_image_list and not pipe_starts_with( options.pipeline, "filter_" ) ) or \
       pipe_starts_with( options.pipeline, "transcode_" ):
    command += fset( 'track_writer:writer:viame_csv:write_time_as_uid=true' )
    command += fset( 'detector_writer:writer:viame_csv:write_time_as_uid=true' )
  else:
    command += fset( 'track_writer:writer:viame_csv:stream_identifier=' + input_basename )
    command += fset( 'detector_writer:writer:viame_csv:stream_identifier=' + input_basename )

  if options.input_detections:
    command += fset( "detection_reader:file_name=" + options.input_detections )
    command += fset( "track_reader:file_name=" + options.input_detections )

  if options.pattern:
    full_pattern = output_subdir + div + options.pattern
    command += fset( "image_writer:file_name_template=" + full_pattern )

  if pipe_starts_with( options.pipeline, "transcode_" ):
    full_pattern = output_subdir + div + input_basename
    command += fset( "video_writer:video_filename=" + full_pattern )

  try:
    if len( options.extra_settings ) > 0:
      for extra_option in options.extra_settings:
        command += fset( " ".join( extra_option ) )
  except:
    pass

  # Process command, possibly with logging
  log_base = ""
  if len( options.log_directory ) > 0 and not options.debug and options.log_directory != "PIPE":
    log_base = output_dir + div + options.log_directory + div + basename_no_ext
    if os.path.sep in basename_no_ext and not os.path.exists( os.path.dirname( log_base ) ):
      os.makedirs( os.path.dirname( log_base ) )
    with get_log_output_files( log_base ) as kwargs:
      res = execute_command( command, gpu=gpu, **kwargs )
  else:
    res = execute_command( command, gpu=gpu )

  global any_video_complete

  # Generate optional mosaic for sequence
  if options.mosaic:
    log_info( "Building mosaic... " )
    cmd2 = []

  if res == 0:
    if multi_threaded:
      log_info( 'Completed: {} on GPU {}'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Success' + lb1 )
    any_video_complete = True
  else:
    if multi_threaded:
      log_info( 'Failure: {} on GPU {} Failed'.format( input_basename, gpu ) + lb1 )
    else:
      log_info( 'Failure' + lb1 )

    if res == -11:
      s = os.statvfs( output_dir )

      if s.f_bavail * s.f_frsize < 100000000:
        exit_with_error( lb1 + 'Out of disk space. Clean up space and then re-run.' )

      log_info( lb1 + 'Pipeline failed with code 11. This is typically indicative of an '
        'issue with system resources, e.g. low disk space or running out of '
        'memory, but could be indicative of a pipeline issue. It\'s also possible '
        'the pipeline you are running just had a shutdown issue. Attempting to '
        'continue processing.' + lb1 )

      any_video_complete = True

    if not any_video_complete:
      if len( log_base ) > 0:
        exit_with_error( 'Processing failed, check ' + log_base + '.txt, terminating.' )
      else:
        exit_with_error( 'Processing failed, terminating.' )
    elif len( log_base ) > 0:
      log_info( lb1 + 'Check ' + log_base + '.txt for error messages' + lb2 )
    

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Process new videos",
     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument( "-i", dest="input", default="",
    help="Input folder, video, or input list (autodetect)" )

  parser.add_argument( "-v", dest="input_video", default="",
    help="Input single video to process" )

  parser.add_argument( "-d", dest="input_dir", default="",
    help="Input directory of videos or image folders to process" )

  parser.add_argument( "-l", dest="input_list", default="",
    help="Input list of image files to process" )

  parser.add_argument( "-p", dest="pipeline", default=default_pipeline,
    help="Input pipeline for processing video or image data" )

  parser.add_argument( "-s", dest="extra_settings", action='append', nargs='*',
    help="Extra command line arguments for the pipeline runner" )

  parser.add_argument( "-id", dest="input_detections", default="",
    help="Input detections around which to create descriptors" )

  parser.add_argument( "-o", dest="output_directory", default=".",
    help="Output directory to store files in" )

  parser.add_argument( "-logs", dest="log_directory", default="logs",
    help="Output sub-directory for log files, if empty will not use files" )

  parser.add_argument( "-video-exts", dest="video_exts", default="3qp;3g2;amv;"
         "asf;avi;drc;gif;gifv;f4v;f4p;f4a;f4bflv;m4v;mkv;mp4;m4p;m4v;mpg;mpg2;"
         "mp2;mpeg;mpe;mpv;mng;mts;m2ts;mov;mxf;nsv;ogg;ogv;qt;roq;rm;rmvb;svi;"
         "webm;wmv;vob;yuv",
    help="Allowable video extensions" )

  parser.add_argument( "-image-exts", dest="image_exts", default="bmp;dds;gif;"
         "heic;jpg;jpeg;png;psd;psp;pspimage;tga;thm;tif;tiff;yuv",
    help="Expected image extensions, used first before fallback." )

  parser.add_argument( "-frate", dest="frame_rate", default="",
    help="Processing frame rate over-ride to process videos at, specified "
         "in hertz (frames per second)" )

  parser.add_argument( "-ifrate", dest="input_frame_rate", default="",
    help="Input frame rate over-ride to process videos at. This is useful "
         "for specifying the frame rate of input image lists, which typically "
         "don't have frame rates" )

  parser.add_argument( "-fbatch", dest="batch_size", default="",
    help="Optional number of frames to process in batches" )

  parser.add_argument( "-fskip", dest="batch_skip", default="",
    help="If batching frames, number of frames to skip between batches" )

  parser.add_argument( "-detection-threshold", dest="detection_threshold", default="",
    help="Optional detection threshold over-ride parameter" )

  parser.add_argument( "-tracker-threshold", dest="tracker_threshold", default="",
    help="Optional tracking threshold over-ride parameter" )

  parser.add_argument( "-archive-height", dest="archive_height", default="",
    help="Advanced: Optional video archive height over-ride" )

  parser.add_argument( "-archive-width", dest="archive_width", default="",
    help="Advanced: Optional video archive width over-ride" )

  parser.add_argument( "-gpus", "--gpu-count", default=1, type=int, metavar='N',
    help="Parallelize the ingest by using the first N GPUs in parallel" )

  parser.add_argument( "-pipes-per-gpu", "--pipes", default=1, type=int, metavar='N',
    help="Parallelize the ingest by using the first N GPUs in parallel" )

  parser.add_argument( "-pattern", dest="pattern", default="frame%06d.png",
    help="Pattern to use for output names for pipes outputting frames" )

  parser.add_argument( "-start-time", dest="start_time", default="",
    help="Optional video start time for processing or conversion" )

  parser.add_argument( "-duration", dest="duration", default="",
    help="Optional video duration for processing or conversion" )

  parser.add_argument( "--detection-plots", dest="detection_plots", action="store_true",
    help="Produce per-video detection plot summaries" )

  parser.add_argument( "--track-plots", dest="track_plots", action="store_true",
    help="Produce per-video track plot summaries" )

  parser.add_argument( "--mosaic", dest="mosaic", action="store_true",
    help="Generate mosaics for the supplied sequences where applicable" )

  parser.add_argument( "-plot-objects", dest="objects", default="fish",
    help="Objects to generate plots for" )

  parser.add_argument( "-plot-threshold", dest="plot_threshold", default=0.25, type=float,
    help="Threshold to generate plots for" )

  parser.add_argument( "-plot-smooth", dest="smooth", default=1, type=int,
    help="Smoothing factor for plots" )

  parser.add_argument( "-gt-file", dest="gt_file", default="",
    help="Pass this groundtruth files to pipes" )

  parser.add_argument( "-auto-detect-gt", dest="auto_detect_gt", default="",
    help="Automatically pass to pipes GT of this type if present" )

  parser.add_argument( "-lbl-file", dest="label_file", default="",
    help="Pass this label file to pipes" )

  parser.add_argument( "--init-db", dest="init_db", action="store_true",
    help="Re-initialize database" )

  parser.add_argument( "--build-index", dest="build_index", action="store_true",
    help="Build searchable index on completion" )

  parser.add_argument( "--ball-tree", dest="ball_tree", action="store_true",
    help="Use a ball tree for the searchable index" )

  parser.add_argument( "--no-reset-prompt", dest="no_reset_prompt", action="store_true",
    help="Don't prompt if the output folder should be reset" )

  parser.add_argument( "--ts-from-file", dest="ts_from_file", action="store_true",
    help="Attempt to retrieve timestamps from image filenames." )

  parser.add_argument( "--write-svm-info", dest="write_svm_info", action="store_true",
    help="Write out header information used for training SVMs" )

  parser.add_argument( "--debug", dest="debug", action="store_true",
    help="Run with debugger attached to process" )

  parser.add_argument( "-install", dest="install_dir", default="",
    help="Optional install dir over-ride for all application "
         "binaries. If this is not specified, it is expected that all "
         "viame binaries are already in our path." )

  args = parser.parse_args()

  # Assorted error checking up front
  process_data = True

  number_input_args = sum( len( inp_x ) > 0 for inp_x in \
    [ args.input, args.input_video, args.input_dir, args.input_list ] )

  if number_input_args == 0 or args.pipeline == no_pipeline:
    if not args.build_index and not args.detection_plots and not args.track_plots:
      exit_with_error( "Either input video or input directory must be specified" )
    else:
      process_data = False
  elif number_input_args > 1:
    exit_with_error( "Only one of input video, directory, or list should be specified" )

  if ( args.detection_plots or args.track_plots ) and len( args.frame_rate ) == 0:
    exit_with_error( "Must specify frame rate if generating detection or track plots" )

  signal.signal( signal.SIGINT, signal_handler )

  # Initialize database
  if args.init_db:
    if len( args.log_directory ) > 0:
      init_log_file = args.output_directory + div + args.log_directory + div + "database_log.txt"
    else:
      init_log_file = ""
    db_is_init, user_select = database_tool.init( log_file=init_log_file, prompt=(not args.no_reset_prompt) )
    if not db_is_init:
      if user_select:
        log_info( "User decided to not initialize new database, shutting down." + lb2 )
        sys.exit( 0 )
      elif len( args.log_directory ) > 0:
        exit_with_error( "Unable to initialize database, check " + init_log_file + lb2 +
         "You may have another database running on your system, or ran "
         "a failed operation in the past and need to re-log or restart." )
      else:
        exit_with_error( "Unable to initialize database" )
    log_info( lb1 )

  # Call processing pipelines on all input data
  if process_data:

    # Handle output directory creation if necessary
    if len( args.output_directory ) > 0:
      recreate_dir = ( not args.init_db and not args.no_reset_prompt )
      prompt_user = ( not args.no_reset_prompt )
      create_dir( args.output_directory, logging=False, recreate=recreate_dir, prompt=prompt_user )

    if len( args.log_directory ) > 0:
      create_dir( args.output_directory + div + args.log_directory, logging=False )

    # Identify all videos to process
    if len( args.input ) > 0:
      # Auto-identify input source
      if not os.path.exists( args.input ):
        exit_with_error( "Input folder \"" + args.input + "\" does not exist" )
      if os.path.isfile( args.input ):
        textchars = bytearray( {7,8,9,10,12,13,27} | set( range(0x20, 0x100) ) - {0x7f} )
        is_binary_string = lambda bytes: bool( bytes.translate( None, textchars ) )
        if is_binary_string( open( args.input, 'rb' ).read( 1024 ) ):
          args.input_video = args.input
        else:
          args.input_list = args.input
      else:
        args.input_dir = args.input

    if len( args.input_list ) > 0:
      if args.gpu_count > 1:
        video_list = split_image_list( args.input_list, args.gpu_count, args.output_directory )
      else:
        video_list = [ args.input_list ]
      is_image_list = True
    elif len( args.input_dir ) > 0:
      video_list = auto_identify_data( args.input_dir, args.video_exts, args.image_exts )
      is_image_list = False
    else:
      video_list = [ args.input_video ]
      is_image_list = False

    if len( video_list ) == 0:
      exit_with_error( "No videos found for ingest in given folder, exiting." )
    elif not is_image_list:
      if not args.init_db:
        log_info( lb1 )
      video_str = " video" if len( video_list ) == 1 else " videos"
      log_info( "Processing " + str( len( video_list ) ) + video_str + lb2 )
    elif not args.build_index:
      log_info( lb1 )

    # Check for local pipelines and pre-reqs present
    if "_project_folder.pipe" in args.pipeline:
      if not os.path.exists( "category_models/detector.pipe" ):
        if has_file_with_extension( "category_models", "svm" ):
          if args.pipeline.endswith( "detector_project_folder.pipe" ):
            args.pipeline = os.path.join( "pipelines", "detector_svm_models.pipe" )
          elif args.pipeline.endswith( "frame_classifier_project_folder.pipe" ):
            args.pipeline = os.path.join( "pipelines", "frame_classifier_svm.pipe" )
          elif args.pipeline.endswith( "tracker_project_folder.pipe" ):
            args.pipeline = os.path.join( "pipelines", "tracker_svm_models.pipe" )
          else:
            exit_with_error( "Use of this script requires training a detector first" )
        else:
          exit_with_error( "Use of this script requires training a detector first" )

    # Process videos in parallel, one per GPU
    video_queue = queue.Queue()
    for video_name in video_list:
      if os.path.isfile( video_name ) or os.path.isdir( video_name ):
        video_queue.put( video_name )
      else:
        log_info( "Skipping unknown input: " + video_name + lb )

    def process_on_thread( gpu, cpu ):
      while True:
        try:
          video_name = video_queue.get_nowait()
        except queue.Empty:
          break
        process_using_kwiver( video_name, args, is_image_list, cpu=cpu, gpu=gpu )

    gpu_thread_list = [ i for i in range( args.gpu_count ) for _ in range( args.pipes ) ]
    cpu_thread_list = list( range( args.pipes ) ) * args.gpu_count

    threads = [ threading.Thread( target = process_on_thread, args = (gpu,cpu,) )
                for gpu, cpu in zip( gpu_thread_list, cpu_thread_list ) ]

    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    if is_image_list:
      if args.gpu_count > 1: # Each thread outputs 1 list, add multiple
        add_final_list_csv( args, video_list )
        for image_list in video_list: # Clean up after split_image_list
          os.unlink( image_list )

    if not video_queue.empty():
      exit_with_error( "Some videos were not processed!" )

  # Build out detection vs time plots for both detections and tracks
  if args.detection_plots:
    import generate_detection_plots
    log_info( lb1 + "Generating data plots for detections" )
    detection_plot_dir = os.path.join( args.output_directory, "detection_plots" )
    create_dir( detection_plot_dir, logging=False, recreate=True, prompt=False )
    generate_detection_plots.detection_plot( args.output_directory,
      detection_plot_dir, args.objects.split( "," ), float( args.plot_threshold ),
      float( args.frame_rate ), int( args.smooth ),
      ext = detection_ext, top_category_only = False )

  if args.track_plots:
    import generate_detection_plots
    log_info( lb1 + "Generating data plots for tracks" )
    track_plot_dir = os.path.join( args.output_directory, "track_plots" )
    create_dir( track_plot_dir, logging=False, recreate=True, prompt=False )
    generate_detection_plots.detection_plot( args.output_directory,
      track_plot_dir, args.objects.split( "," ), float( args.plot_threshold ),
      float( args.frame_rate ), int( args.smooth ),
      ext = track_ext, top_category_only = True )

  if args.detection_plots or args.track_plots:
    log_info( lb1 )

  # Build searchable index
  if args.build_index:
    log_info( lb1 + "Building searchable index" + lb2 )

    if len( args.log_directory ) > 0 and args.log_directory != "PIPE":
      index_log_file = args.output_directory + div + args.log_directory + div + "smqtk_indexer.txt"
    else:
      index_log_file = ""

    if args.ball_tree:
      print( "Warning: building a ball tree is deprecated" )

    if not database_tool.build_standard_index( remove_quotes( args.install_dir ),
                                               log_file = index_log_file ):
      exit_with_error( "Unable to build index" )

  # Output complete message
  if os.name == 'nt':
    log_info( lb1 + "Processing complete, close this window before launching any GUI." + lb2 )
  else:
    log_info( lb1 + "Processing complete" + lb2 )
