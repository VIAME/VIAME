#!python

import sys
import os
import glob
import argparse
import atexit
import errno
import shutil
import string
import tempfile

if sys.version_info.major < 3:
  import urlparse
else:
  import urllib.parse as urlparse

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

sys.dont_write_bytecode = True
debug_mode = False

import database_tool

temp_dir = tempfile.mkdtemp(prefix='viqui-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

#------------------------------------------------------------------------------
# Small helper functions

def format_fp( input ):
  if os.name == 'nt':
    tmp1 = input.replace('\\','/')
    tmp2 = tmp1.replace(' ', '%20')
    return tmp2
  else:
    return input

def list_files_in_dir( folder ):
  return glob.glob( folder + div + '*' )

def list_files_in_dir( folder, extension ):
  return glob.glob( folder +  div + '*' + extension )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

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

def get_gui_cmd():
  if os.name == 'nt':
    return 'viqui.exe '
  else:
    if debug_mode:
      return 'gdb --args viqui '
    else:
      return 'viqui '

def get_path_prefix_with_and():
  if os.name == 'nt':
    return get_path_prefix() + '&& '
  else:
    return get_path_prefix() + '&& '

def make_uri( scheme='file', authority='', path='', query='' ):
  return urlparse.urlunsplit([ scheme, authority, '' + format_fp( path ) + '', query, '' ])

def full_path( dirname ):
  return os.path.abspath( dirname )

#------------------------------------------------------------------------------
def is_valid_database( options ):
  if not os.path.exists( options.input_dir ):
    print( "\nERROR: \"" + options.input_dir + "\" directory does not exist, "
           "was your create_index call successful?\n" )
    return False
  if len( glob.glob( os.path.join( options.input_dir, "*.index" ) ) ) == 0:
    print( "\nERROR: \"" + options.input_dir + "\" is empty, "
           "was your create_index call successful?\n" )
    return False
  return True

#------------------------------------------------------------------------------
def get_import_config_args( config_files ):
  args = ''
  command = '--import-config '
  for config_file in config_files:
    args = args + command + config_file + ' '
  return args

#------------------------------------------------------------------------------
def get_query_server_uri( options ):
  query_pipeline = full_path( find_file( options.query_pipe ) )
  query = 'Pipeline=%s' % (format_fp( query_pipeline ))
  return make_uri(scheme='kip', query=query)

#------------------------------------------------------------------------------
def create_archive_file( options ):
  (fd, name) = tempfile.mkstemp(prefix='viqui-archive-',
                                suffix='.archive',
                                text=True, dir=temp_dir)
  f = os.fdopen(fd, 'w')

  f.write('archive1\n')
  for index_file in list_files_in_dir( options.input_dir, 'index' ):
      f.write( os.path.abspath( index_file ) + "\n" )
  f.close()
  return name

#------------------------------------------------------------------------------
def get_predefined_query_dir( options ):

  if options.predefined_dir:
    query_dir = options.predefined_dir

  if os.path.exists( query_dir ):
      if not os.path.isdir( query_dir ):
          print( '%s is not a directory.' % (query_dir) )
          sys.exit(1)

  return query_dir

#------------------------------------------------------------------------------
config_template= string.Template('''
[General]
QueryVideoUri = $query_video_uri
VideoProviderUris=$video_provider_uri
QueryCacheUri=$query_cache_uri
QueryServerUri=$query_server_uri
PredefinedQueryUri=$predefined_query_uri
''')

config_template_no_cache= string.Template('''
[General]
QueryVideoUri = $query_video_uri
VideoProviderUris=$video_provider_uri
QueryServerUri=$query_server_uri
PredefinedQueryUri=$predefined_query_uri
''')

def create_constructed_config( options ):
  query_server_uri = get_query_server_uri( options )
  query_video_uri = make_uri( path=full_path( options.query_dir ) )

  archive_file = create_archive_file( options )
  video_provider_uri = make_uri( path=archive_file )

  predefined_query_dir = get_predefined_query_dir( options )
  predefined_query_uri = make_uri( path=full_path( predefined_query_dir ) )

  (fd, name) = tempfile.mkstemp( prefix='viqui-config-', suffix='.conf',
                                 text=True, dir=temp_dir )
  f = os.fdopen( fd, 'w' )

  if options.cache_dir != "disabled":
    query_cach_uri = make_uri( path=full_path( options.cache_dir ) )

    f.write( config_template.substitute(
               query_video_uri = query_video_uri,
               video_provider_uri = video_provider_uri,
               query_cache_uri = query_cach_uri,
               query_server_uri = query_server_uri,
               predefined_query_uri = predefined_query_uri ) )
  else:
    f.write( config_template_no_cache.substitute(
               query_video_uri = query_video_uri,
               video_provider_uri = video_provider_uri,
               query_server_uri = query_server_uri,
               predefined_query_uri = predefined_query_uri ) )

  f.close()
  return name

#------------------------------------------------------------------------------
# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch Query GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="database",
                      help="Input directory containing videos and results")

  parser.add_argument("-c", dest="context_file",
                      default="gui-params" + div + "context_view_bluemarble_low_res.kst",
                      help="GUI context file for display on left panel")

  parser.add_argument("-s", dest="style", default="",
                      help="Optional GUI style option, blank for default")

  parser.add_argument("-e", "--engineer", action="store_true",
                      dest="engineer_mode", default=False,
                      help="Turn on the engineer UI (add developer options).")

  parser.add_argument("-qp", dest="query_pipe",
                      default="pipelines" + div + "query_retrieval_and_iqr.pipe",
                      help="Pipeline for performing new system queries")

  parser.add_argument("-qd", dest="query_dir",
                      default="database" + div + "Queries",
                      help="Directory for writing new queries and configs to")

  parser.add_argument("-cd", dest="cache_dir",
                      default="disabled",
                      help="Directory for caching repeated queries")

  parser.add_argument("-pd", dest="predefined_dir",
                      default="pipelines" + div + "predefined_queries",
                      help="Predefined query directory, if present")

  parser.add_argument("-theme", dest="gui_theme",
                      default="gui-params" + div + "dark_gui_settings.ini",
                      help="Predefined query directory, if present")

  parser.add_argument("--no-reconfig", dest="no_reconfig", action="store_true",
                      help="Do not run any reconfiguration of the GUI")

  parser.add_argument("--debug", dest="debug", action="store_true",
                      help="Run with debugger attached to process")

  parser.set_defaults( no_reconfig=False )
  parser.set_defaults( debug=False )

  args = parser.parse_args()

  # Basic checking
  if not is_valid_database( args ):
    sys.exit(0)

  # Create required directories
  create_dir( args.query_dir )

  if args.cache_dir != "disabled":
    create_dir( args.cache_dir )

  # Check if in debug mode
  if args.debug:
    debug_mode = True

  # Build command line
  command = get_gui_cmd() + \
            "--add-layer \"" + get_script_path() + div + args.context_file + "\" "

  if not args.no_reconfig:
    if len( args.gui_theme ) > 0:
      command = command + "--theme \"" + find_file( args.gui_theme ) + "\" "

    if args.engineer_mode:
      command = command + "--ui engineering "
    else:
      command = command + "--ui analyst "
    
    config_args = get_import_config_args([create_constructed_config( args )])

    command = command + config_args

  # Make sure database online
  database_tool.start( quiet=True )
  database_tool.status()

  # Process command
  print( "\nLaunching search GUI. When finished, make sure this console is closed.\n" )

  res = execute_command( command )
