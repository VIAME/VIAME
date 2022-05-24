#!python

import sys
import os
import glob
import argparse
import atexit
import shutil
import tempfile
import subprocess

temp_dir = tempfile.mkdtemp(prefix='vpview-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  if os.path.exists( folder ):
    return [
      os.path.join( folder, f ) for f in sorted( os.listdir( folder ) )
      if not f.startswith('.')
    ]
  return []

def list_files_in_dir_w_ext( folder, extension ):
  return [f for f in list_files_in_dir(folder) if f.endswith(extension)]

def glob_files_in_folder( folder, prefix, extension ):
  return glob.glob( os.path.join( folder, prefix ) + "*" + extension )

def multi_glob_files_in_folder( folder, prefixes, extensions ):
  output = []
  for prefix in prefixes:
    for extension in extensions:
      output.extend( glob.glob( os.path.join( folder, prefix ) + "*" + extension ) )
  return output

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    print( "Creating " + dirname )
    os.makedirs( dirname )
  if not os.path.exists( dirname ):
    print( "Unable to create " + dirname )
    sys.exit( 0 )

def get_gui_cmd( debug=False ):
  if os.name == 'nt':
    return ['vpView.exe']
  else:
    if debug:
      return [ 'gdb', '--args', 'vpView' ]
    else:
      return ['vpView']

def execute_command( cmd, stdout=None, stderr=None ):
  if os.name == 'nt' and stdout == None:
    fnull = open( os.devnull, "w" )
    return subprocess.call( cmd, stdout=fnull, stderr=subprocess.STDOUT )

  return subprocess.call( cmd, stdout=stdout, stderr=stderr )

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

def create_pipelines_list( glob_str ):
  (fd, name) = tempfile.mkstemp(prefix='vpview-pipelines-',
                                suffix='.ini',
                                text=True, dir=temp_dir)

  search_str = os.path.join( get_script_path(), glob_str )
  pipeline_files = sorted( glob.glob( search_str ) )
  total_entries = len( pipeline_files )

  f = os.fdopen(fd, 'w')

  f.write( "[EmbeddedPipelines]\n" )
  f.write( "size=" + str( total_entries ) + "\n" )

  for ind, full_path in enumerate( pipeline_files ):
    name_id = os.path.splitext( os.path.basename( full_path ) )[0]
    f.write( "%s\\Name=\"%s\"\n" % (ind+1, name_id) )
    f.write( "%s\\Path=\"%s\"\n" % (ind+1, full_path.replace( "\\","\\\\" ) ) )

  f.close()
  return name

def default_annotator_args( args ):
  command_args = []
  if len( args.gui_theme ) > 0:
    command_args += [ "--theme", find_file( args.gui_theme ) ]
  if len( args.pipelines ) > 0:
    command_args += [ "--import-config", create_pipelines_list( args.pipelines ) ]
  return command_args

def get_pipeline_cmd( debug=False ):
  if os.name == 'nt':
    if debug:
      return ['kwiver.exe', 'runner']
    else:
      return ['kwiver.exe', 'runner']
  else:
    if debug:
      return ['gdb', '--args', 'kwiver', 'runner ']
    else:
      return ['kwiver', 'runner']

def generate_index_for_video( args, file_path, basename ):

  if not os.path.isfile( file_path ):
    print( "Unable to find file: " + file_path )
    sys.exit( 0 )

  cmd = get_pipeline_cmd() +                                   \
    ["-p", find_file( args.cache_pipeline ) ] +                \
    ["-s", 'input:video_filename=' + file_path ] +             \
    ["-s", 'input:video_reader:type=vidl_ffmpeg' ] +           \
    ["-s", 'kwa_writer:output_directory=' + args.cache_dir ] + \
    ["-s", 'kwa_writer:base_filename=' + basename ] +          \
    ["-s", 'kwa_writer:stream_id=' + basename ]

  if len( args.frame_rate ) > 0:
    cmd += ["-s", 'downsampler:target_frame_rate=' + args.frame_rate ]

  execute_command( cmd )

  return args.cache_dir + div + basename + ".index"

def select_option( option_list, display_str="Select Option:" ):
  sys.stdout.write( "\n" )

  counter = 1
  for option in option_list:
    print( "(" + str(counter) + ") " + option )
    counter = counter + 1

  sys.stdout.write( "\n" + display_str + " " )
  sys.stdout.flush()

  if sys.version_info[0] < 3:
    choice = raw_input().lower()
  else:
    choice = input().lower()

  if int( choice ) < 1 or int( choice ) > len( option_list ):
    print( "Invalid selection, must be a valid number" )
    sys.exit(0)

  return int( choice ) - 1

def process_video( args ):
  print( "Function not yet implemented" )
  sys.exit(0)

def process_list( args ):
  print( "Function not yet implemented" )
  sys.exit(0)

def process_video_dir( args ):
  
  video_files = list_files_in_dir( args.video_dir )
  index_files = list_files_in_dir_w_ext( args.cache_dir, "index" )

  video_files.sort()
  index_files.sort()

  video_files_no_ext_no_path = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
  index_files_no_ext_no_path = [os.path.splitext(os.path.basename(f))[0] for f in index_files]

  net_files = video_files_no_ext_no_path
  net_full_paths = video_files

  total_video_count = len( video_files_no_ext_no_path )
  total_index_count = len( index_files_no_ext_no_path )

  has_index = [False] * total_video_count

  for fpath, fname in zip( index_files, index_files_no_ext_no_path ):
    if fname in net_files:
      index = net_files.index( fname )
      has_index[ index ] = True
      net_full_paths[ index ] = fpath
    else:
      net_files.append( fname )
      has_index.append( True )
      net_full_paths.append( fpath )

  if len( net_files ) == 0:
    print( "\nError: No videos found in input directory: " + args.video_dir + "\n" )
    print( "If you want to load videos, not just images, make sure it is non-empty" )

  # Have user select video file
  file_list = []
  for fname, is_cached in zip( net_files, has_index ):
    file_list.append( fname + ( " (cached in: " + args.cache_dir + ")" if is_cached else "" ) )

  if len( file_list ) > 0 and file_list[0].islower():
    no_file = "with_no_imagery_loaded"
  else:
    no_file = "With No Imagery Loaded"

  file_list = [no_file] + sorted( file_list )

  special_list_option = "input_list.txt"
  has_special_list_option = False

  if os.path.exists( special_list_option ):
    file_list = file_list + [ special_list_option ]
    has_special_list_option = True

  file_id = select_option( file_list )

  if file_id == 0:
    execute_command( get_gui_cmd( args.debug ) + default_annotator_args( args ) )
    sys.exit(0)
  elif has_special_list_option and file_id == len( file_list ) - 1:
    file_no_ext = special_list_option
    file_has_index = True
    file_path = special_list_option
  else:
    file_id = file_id - 1
    file_no_ext = net_files[ file_id ]
    file_has_index = has_index[ file_id ]
    file_path = net_full_paths[ file_id ]

  # Scan for possible detection file
  detection_list = []
  detection_file = ""

  search_prefix = [ file_no_ext + ".", file_no_ext + "_detections", file_no_ext + "_tracks" ]

  detection_search = multi_glob_files_in_folder( '.', file_no_ext, ["csv"] )
  if len( detection_search ) > 0:
    detection_list.extend( detection_search )
  if len( args.video_dir ) > 0 and args.video_dir != '.':
    detection_search = glob_files_in_folder( args.video_dir, file_no_ext, "csv" )
    detection_list.extend( detection_search )
  if len( args.cache_dir ) > 0 and args.cache_dir != '.' and args.cache_dir != args.video_dir:
    detection_search = glob_files_in_folder( args.cache_dir, file_no_ext, "csv" )
    detection_list.extend( detection_search )
  detection_list = sorted( detection_list )

  if len( detection_list ) > 0:
    if len( detection_list ) > 0 and detection_list[0].islower():
      no_file = "with_no_detections"
    else:
      no_file = "Launch Without Loading Detections"
    detection_list = [ no_file ] + detection_list
    detection_id = select_option( detection_list )
    if detection_id != 0:
      detection_file = detection_list[ detection_id ]

  # Launch GUI with required options
  if not file_has_index:
    create_dir( args.cache_dir )
    if not os.path.isdir( file_path ):
      print( "Generating cache for video file, this may take up to a few minutes.\n" )
      file_path = generate_index_for_video( args, file_path, file_no_ext )
    else:
      from process_video import make_filelist_for_dir
      file_path = make_filelist_for_dir( file_path, args.cache_dir,
                                         file_no_ext )

  (fd, name) = tempfile.mkstemp(prefix='vpview-project-',
                                suffix='.prj',
                                text=True, dir=temp_dir)

  ftmp = os.fdopen(fd, 'w')
  ftmp.write( "DataSetSpecifier=" + os.path.abspath( file_path ).replace( "\\","\\\\" ) + "\n" )
  if len( detection_file ) > 0:
    ftmp.write( "TracksFile=" + os.path.abspath( detection_file ).replace( "\\","\\\\" ) + "\n" )
  ftmp.close()

  execute_command( get_gui_cmd( args.debug ) + [ "-p", name ] + default_annotator_args( args ) )

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch annotation GUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument( "-d", dest="video_dir", default="",
    help="Input directory containing videos to run annotator on" )

  parser.add_argument( "-c", dest="cache_dir", default="",
    help="Input directory containing cached video .index files" )

  parser.add_argument( "-o", dest="output_directory", default="database",
    help="Output directory to store files in" )

  parser.add_argument( "-v", dest="input_video", default="",
    help="Input video file to run annotator on" )

  parser.add_argument( "-l", dest="input_list", default="",
    help="Input image list file to run annotator on" )

  parser.add_argument( "-theme", dest="gui_theme",
    default="gui-params" + div + "dark_gui_settings.ini",
    help="Predefined query directory, if present" )

  parser.add_argument( "-pipelines", dest="pipelines",
    default="pipelines" + div + "embedded_single_stream" + div + "*.pipe",
    help="Glob pattern for runable processing pipelines" )

  parser.add_argument( "-cache-pipe", dest="cache_pipeline",
    default="pipelines" + div + "filter_to_kwa.pipe",
    help="Pipeline used for generative video .index files" )

  parser.add_argument( "-frate", dest="frame_rate", default="",
    help="Frame rate over-ride to process videos at" )

  parser.add_argument( "--debug", dest="debug", action="store_true",
    help="Run with debugger attached to process" )

  parser.set_defaults( debug=False )

  args = parser.parse_args()

  if len( args.video_dir ) > 0 or len( args.cache_dir ) > 0:
    process_video_dir( args )
  elif len( args.input_video ) > 0:
    process_video( args )
  elif len( args.input_list ) > 0:
    process_list( args )
  else:
    execute_command( get_gui_cmd( args.debug ) + default_annotator_args( args ) )
