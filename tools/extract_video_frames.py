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
    os.path.join(folder, f) for f in sorted(os.listdir(folder))
    if not f.startswith('.')
  ]

# Create a directory if it doesn't exist
def create_dir( dirname, logging=True ):
  if not os.path.exists( dirname ):
    if logging:
      print( "Creating " + dirname )
    os.makedirs( dirname )

def get_ffmpeg_cmd():
  if os.name == 'nt':
    return ['ffmpeg.exe']
  else:
    return ['ffmpeg']

def exit_with_error( error_str ):
  sys.stdout.write( '\n\nERROR: ' + error_str + '\n\n' )
  sys.stdout.flush()
  sys.exit(0)

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Ingest new videos",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="videos",
                      help="Input directory containing videos")

  parser.add_argument("-o", dest="output_dir", default="frames",
                      help="Output directory to put frames into")

  parser.add_argument("-s", dest="start_time", default="00:00:00.00",
                      help="Start time of frames to extract")

  parser.add_argument("-r", dest="frame_rate", default="",
                      help="Video frame rate in hz to extract at")

  parser.add_argument("-t", dest="duration", default=invalid_time,
                      help="Duration of sequence to extract")

  parser.add_argument("-p", dest="pattern", default="frame%06d.png",
                      help="Frame pattern to dump frames into")

  args = parser.parse_args()

  files = list_files_in_dir( args.input_dir )
  create_dir( args.output_dir )

  for file_with_path in files:
    file_no_path = os.path.basename( file_with_path )
    output_folder = args.output_dir + div + file_no_path
    create_dir( output_folder )
    cmd = get_ffmpeg_cmd() + [ "-i", file_with_path ]
    if len( args.frame_rate ) > 0:
      cmd += [ "-r", args.frame_rate ]
    if len( args.start_time ) > 0:
      cmd += [ "-ss", args.start_time ]
    if len( args.duration ) > 0 and args.duration != invalid_time:
      cmd += [ "-t", args.duration ]
    cmd += [ output_folder + div + args.pattern ]
    subprocess.call( cmd )

  print( "\n\nFrame extraction complete, exiting.\n\n" )
