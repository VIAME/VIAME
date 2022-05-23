#!/usr/bin/env python

import sys
import os
import argparse
import signal
import subprocess

sys.dont_write_bytecode = True

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

invalid_time = "99:99:99.99"

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  if not os.path.isdir( folder ):
    exit_with_error( "Input folder \"" + folder + "\" does not exist" )
  return [
    os.path.join( folder, f ) for f in sorted( os.listdir( folder ) )
    if not f.startswith( '.' )
  ]

# Create a directory if it doesn't exist
def create_dir( dirname, logging=True ):
  if not os.path.exists( dirname ):
    if logging:
      print( "Creating " + dirname )
    os.makedirs( dirname )

def get_ffmpeg_cmd():
  if os.name == 'nt':
    return [ 'ffmpeg.exe' ]
  else:
    return [ 'ffmpeg' ]

def get_python_cmd():
  if os.name == 'nt':
    return [ 'python.exe' ]
  else:
    return [ 'python' ]

def exit_with_error( error_str ):
  sys.stdout.write( '\n\nERROR: ' + error_str + '\n\n' )
  sys.stdout.flush()
  sys.exit( 0 )

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser( description="Ingest new videos",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter )

  parser.add_argument( "-d", dest="input_dir", default="videos",
    help="Input directory containing videos" )

  parser.add_argument( "-o", dest="output_dir", default="frames",
    help="Output directory to put frames into" )

  parser.add_argument( "-r", dest="frame_rate", default="",
    help="Video frame rate in hz to extract at" )

  parser.add_argument( "-s", dest="start_time", default=invalid_time,
    help="Start time of frames to extract" )

  parser.add_argument( "-t", dest="duration", default=invalid_time,
    help="Duration of sequence to extract" )

  parser.add_argument( "-p", dest="pattern", default="frame%06d.png",
    help="Frame pattern to dump frames into" )

  parser.add_argument( "-m", dest="method", default="kwiver",
    help="Can either be kwiver or ffmpeg" )

  args = parser.parse_args()
  create_dir( args.output_dir )

  if args.method == "ffmpeg":
    files = list_files_in_dir( args.input_dir ) 
    for file_with_path in files:
      file_no_path = os.path.basename( file_with_path )
      output_folder = args.output_dir + div + file_no_path
      create_dir( output_folder )
      cmd = get_ffmpeg_cmd() + [ "-i", file_with_path ]
      if len( args.frame_rate ) > 0:
        cmd += [ "-r", args.frame_rate ]
      if len( args.start_time ) > 0 and args.start_time != invalid_time:
        cmd += [ "-ss", args.start_time ]
      if len( args.duration ) > 0 and args.duration != invalid_time:
        cmd += [ "-t", args.duration ]
      cmd += [ output_folder + div + args.pattern ]
      subprocess.call( cmd )
  else:
    cmd = get_python_cmd()
    cmd += [ os.path.dirname( os.path.abspath( __file__ ) ) + div + "process_video.py" ]
    cmd += [ "-d", args.input_dir ]
    cmd += [ "-o", args.output_dir ]
    cmd += [ "-p", "pipelines/filter_default.pipe" ]
    cmd += [ "-pattern", args.pattern ]
    if len( args.start_time ) > 0 and args.start_time != invalid_time:
      cmd += [ "-start-time", args.start_time ]
    if len( args.duration ) > 0 and args.duration != invalid_time:
      cmd += [ "-duration", args.duration ]
    subprocess.call( cmd )

  print( "\n\nFrame extraction complete, exiting.\n\n" )
