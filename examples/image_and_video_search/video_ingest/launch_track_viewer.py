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

# Get correct OS-specific calls
def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def get_gui_cmd():
  if os.name == 'nt':
    return 'vsPlay.exe '
  else:
    return 'vsPlay '

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

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch annotation GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", "database",
                      help="Input directory containing results")

  args = parser.parse_args()

  # Formulate command
  command = get_path_prefix_with_and() + \
            get_gui_cmd()

  # Process command
  res = os.system( command )
