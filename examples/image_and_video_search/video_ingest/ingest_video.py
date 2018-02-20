#!python

import sys
import os
import glob
import argparse
import signal

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  return glob.glob( folder + '/*' )

def list_files_in_dir_w_ext( folder, extension ):
  return glob.glob( folder + '/*' + extension )

# Create a directory if it doesn't exist
def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

# Get correct OS-specific calls
def execute_command( cmd ):
  if os.name == 'nt':
    return os.system( cmd )
  else:
    return os.system( '/bin/bash -c \"' + cmd + '\"'  )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def get_pipeline_cmd( debug=False ):
  if os.name == 'nt':
    if debug:
      return 'pipeline_runner.exe '
    else:
      return 'pipeline_runner.exe '
  else:
    if debug:
      return 'gdb --args pipeline_runner '
    else:
      return 'pipeline_runner '

def get_path_prefix():
  if os.name == 'nt':
    return 'CALL ' + get_script_path() + '\..\..\..\setup_viame.bat '
  else:
    return 'source ' + get_script_path() + '/../../../setup_viame.sh '

def get_path_prefix_with_and():
  if os.name == 'nt':
    return get_path_prefix() + '&& '
  else:
    return get_path_prefix() + '&& '

def get_log_postfix( output_prefix ):
  if os.name == 'nt':
    return '1> ' + output_prefix + '.out.txt 2> ' + output_prefix + '.err.txt'
  else:
    return '> ' + output_prefix + '.txt 2>&1'

# Other helpers
def signal_handler( signal, frame ):
  print( 'Ingest aborted, see you next time' )
  sys.exit(0)

# Process a single video
def process_video_kwiver( pipeline_location, video_name, extra_options, logging_dir, debug ):

  print( 'Processing: ' + video_name + "... " )

  # Get video name without extension and full path
  basename = os.path.splitext( os.path.basename( video_name ) )[0]

  # Formulate command
  command = get_path_prefix_with_and() + \
            get_pipeline_cmd( debug ) + \
            '-p ' + pipeline_location + ' ' + \
            '-s input:video_filename=' + video_name + ' ' + \
            '-s detector_writer:file_name=database/' + basename + '_detections.kw18 ' + \
            '-s track_writer:file_name=database/' + basename + '_tracks.kw18 ' + \
            '-s descriptor_writer:file_name=database/' + basename + '_descriptors.csv ' + \
            '-s track_descriptor:uid_basename=' + basename + ' ' + \
            '-s kwa_writer:output_directory=database ' + \
            '-s kwa_writer:base_filename=' + basename + ' ' + \
            '-s kwa_writer:stream_id=' + basename

  if len( extra_options ) > 0:
    command = command + '-s ' + extra_options

  if len( logging_dir ) > 0 and not debug:
    command = command + get_log_postfix( logging_dir + '/' + basename )

  # Process command
  res = execute_command( command )

  if res == 0:
    print( 'Success' )
  else:
    print( 'Failure' )

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Ingest new videos",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-v", dest="input_video", default="",
                      help="Input single video to ingest")

  parser.add_argument("-d", dest="input_dir", default="",
                      help="Input directory to ingest")

  parser.add_argument("-p", dest="pipeline", default="configs/ingest_video.pipe",
                      help="Input pipeline for ingestation")

  parser.add_argument("-l", dest="log_dir", default="database/Logs",
                      help="Directory for log files, if empty will not use files")

  parser.add_argument("-e", dest="extra_options", default="",
                      help="Extra command line arguments for the pipeline runner")

  parser.add_argument("--rel-to-script", dest="rel_to_script", action="store_true",
                      help="Pipeline file is relative to script location")

  parser.add_argument("--init-db", dest="init_db", action="store_true",
                      help="Re-initialize database")

  parser.add_argument("--build-index", dest="build_index", action="store_true",
                      help="Build searchable index on completion")

  parser.add_argument("--debug", dest="debug", action="store_true",
                      help="Run with debugger attached to process")

  parser.set_defaults( init_db=False )
  parser.set_defaults( rel_to_script=False )
  parser.set_defaults( build_index=False )
  parser.set_defaults( debug=False )

  args = parser.parse_args()

  # Error checking
  if len( args.input_video ) == 0 and len( args.input_dir ) == 0:
    print( "Either input video or input directory must be specified" )
    sys.exit( 0 ) 

  if len( args.input_video ) > 0 and len( args.input_dir ) > 0:
    print( "Only an input video or directory should be specified, not both" )
    sys.exit( 0 ) 

  signal.signal( signal.SIGINT, signal_handler )

  # Identify all videos to process
  video_list = []

  if len( args.input_dir ) > 0:
    video_list = list_files_in_dir( args.input_dir ) 
  else:
    video_list.append( args.input_video )

  # Get required paths
  pipeline_loc = args.pipeline

  if args.rel_to_script:
    pipeline_loc = get_script_path() + '/' + args.pipeline

  if len( args.log_dir ) > 0:
    create_dir( args.log_dir )

  # Initialize database
  if args.init_db:
    # TODO
    do_something = 1

  # Process videos 
  for video_name in video_list:
    process_video_kwiver( pipeline_loc, video_name, args.extra_options, args.log_dir, args.debug )

  # Build index
  if args.build_index:
    # TODO
    do_something = 1
