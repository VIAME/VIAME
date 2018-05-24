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

def get_gui_cmd():
  if os.name == 'nt':
    return 'vsPlay.exe '
  else:
    return 'vsPlay '

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

  cmd = get_gui_cmd() + " -tf " + base + ".kw18 -vf " + filename

  os.system( cmd )
