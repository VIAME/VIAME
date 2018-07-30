#!python

import sys
import os
import glob
import argparse
import atexit
import shutil
import tempfile

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

temp_dir = tempfile.mkdtemp(prefix='vpview-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

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

def execute_command( cmd ):
  if os.name == 'nt':
    return os.system( cmd )
  else:
    return os.system( '/bin/bash -c \"' + cmd + '\"'  )

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

def get_writer_cmd():
  if os.name == 'nt':
    return 'kwa_tool.exe '
  else:
    return 'kwa_tool '

def process_video( args ):
  print( "Function not yet implemented" )
  sys.exit(0)

def process_list( args ):
  print( "Function not yet implemented" )
  sys.exit(0)

def process_dir( args ):
  print( "Function not yet implemented" )
  sys.exit(0)

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

def create_pipelines_list( glob_str ):
  (fd, name) = tempfile.mkstemp(prefix='vpview-pipelines-',
                                suffix='.ini',
                                text=True, dir=temp_dir)

  search_str = os.path.join( get_script_path(), glob_str )
  pipeline_files = sorted( glob.glob( search_str ) )
  total_entries = len( pipeline_files )

  f = os.fdopen(fd, 'w')

  f.write("[EmbeddedPipelines]\n")
  f.write("size=" + str( total_entries ) + "\n")

  for ind, full_path in enumerate( pipeline_files ):
    name_id = os.path.splitext( os.path.basename( full_path ) )[0]
    f.write("%s\\Name=\"%s\"\n" % (ind+1, name_id) )
    f.write("%s\\Path=\"%s\"\n" % (ind+1, full_path) )

  f.close()
  return name

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch annotation GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="",
                      help="Input directory to run annotator on")

  parser.add_argument("-v", dest="input_video", default="",
                      help="Input video file to run annotator on")

  parser.add_argument("-l", dest="input_list", default="",
                      help="Input image list file to run annotator on")

  parser.add_argument("-theme", dest="gui_theme",
                      default="gui-params" + div + "dark_gui_settings.ini",
                      help="Predefined query directory, if present")

  parser.add_argument("-pipelines", dest="pipelines",
                      default="pipelines/embedded*.pipe",
                      help="Glob pattern for runable processing pipelines")

  parser.add_argument("--debug", dest="debug", action="store_true",
                      help="Run with debugger attached to process")
                      
  parser.set_defaults( debug=False )

  args = parser.parse_args()

  if len( args.input_dir ) > 0:
    process_dir( args )
  elif len( args.input_video ) > 0:
    process_video( args )
  elif len( args.input_list ) > 0:
    process_list( args )
  else:
    command = get_gui_cmd()
    if len( args.gui_theme ) > 0:
      command = command + " --theme \"" + find_file( args.gui_theme ) + "\" "
    if len( args.pipelines ) > 0:
      command = command + " --import-config \"" + create_pipelines_list( args.pipelines ) + "\" "
    execute_command( command )
