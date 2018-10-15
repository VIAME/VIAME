#!python

import sys
import os
import glob
import argparse

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

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

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def find_file( filename ):
  if( os.path.exists( filename ) ):
    return os.path.abspath( filename )
  elif os.path.exists( get_script_path() + div + filename ):
    return get_script_path() + div + filename
  else:
    print( "Unable to find " + filename )
    sys.exit( 0 )

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch track viewer GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="database",
                      help="Input directory containing results")

  parser.add_argument("-theme", dest="gui_theme",
                      default="gui-params" + div + "dark_gui_settings.ini",
                      help="Predefined query directory, if present")

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

  if int(choice) < 1 or int(choice) > len( files ):
    print( "Invalid selection, must be a number" )
    sys.exit(0)

  sys.stdout.write( "\n" )

  filename = files[int(choice)-1]
  base, ext = os.path.splitext( filename )

  cmd = get_gui_cmd() + " -tf " + base + ".kw18 -vf " + filename

  if len( args.gui_theme ) > 0:
    cmd = cmd + " --theme \"" + find_file( args.gui_theme ) + "\" "

  os.system( cmd )
