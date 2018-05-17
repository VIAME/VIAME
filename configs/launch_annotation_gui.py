#!python

import sys
import os
import glob
import argparse

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  return glob.glob( folder + '/*' )

def list_files_in_dir( folder, extension ):
  return glob.glob( folder + '/*' + extension )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    print( "Creating " + dirname )
    os.makedirs( dirname )
  if not os.path.exists( dirname ):
    print( "Unable to create " + dirname )
    sys.exit( 0 )

def get_gui_cmd():
  if os.name == 'nt':
    return 'vpView.exe '
  else:
    return 'vpView '

def get_writer_cmd():
  if os.name == 'nt':
    return 'kwa_tool.exe '
  else:
    return 'kwa_tool '

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch annotation GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="database",
                      help="Input directory containing results")

  args = parser.parse_args()

  files = list_files_in_dir( args.input_dir, "index" )

  if len( files ) == 0:
    print( "No computed results in input directory" )
    sys.exit(0)

  files.sort()

  print( "" )
  counter = 1
  for filen in files:
    print( "(" + str(counter) + ") " + filen )
    counter = counter + 1

  sys.stdout.write( "\nSelect File: " )
  sys.stdout.flush()

  if sys.version_info[0] < 3:
    choice = raw_input().lower()
  else:
    choice = input().lower()

  if int(choice) < 0 or int(choice) > len( files ):
    print( "Invalid selection, must be a number" )
    sys.exit(0)

  sys.stdout.write( "\n" )

  filename = files[int(choice)-1]
  base, ext = os.path.splitext( filename )
  basename = os.path.splitext( os.path.basename( filename ) )[0] 

  image_dir = args.input_dir + "/Raw/" + basename
  project_file = image_dir + "/view_detections.prj"

  print( "Dumping out frames to directory" )
  if not os.path.exists( image_dir ) or not os.path.exists( image_dir + "/frame000010.png" ):
    create_dir( image_dir )
    os.system( get_writer_cmd() + " --input " + filename +
               " --output-pattern " + image_dir + "/frame%06d.png" +
               " --mode extract" )

  fout = open( project_file, 'w' )
  fout.write( "DataSetSpecifier=*.png\n" )
  fout.write( "TracksFile=../../" + base + "_detections.kw18" )
  fout.close()

  cmd = get_gui_cmd() + " -p " + project_file
  os.system( cmd )
