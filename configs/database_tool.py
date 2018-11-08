#!python

import sys
import os
import os.path
import shutil
import subprocess

database_dir = "database"
pipelines_dir = "pipelines"

sql_dir = os.path.join(database_dir, "SQL")
init_file = os.path.join(pipelines_dir, "sql_init_table.sql")
log_file = os.path.join(database_dir, "SQL_Log_File")

SMQTK_ITQ_TRAIN_CONFIG = os.path.join(pipelines_dir, "smqtk_train_itq.json")
SMQTK_HCODE_CONFIG = os.path.join(pipelines_dir, "smqtk_compute_hashes.json")
SMQTK_HCODE_PICKLE = os.path.join(database_dir, "ITQ", "alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle")
SMQTK_BTREE_CONFIG = os.path.join(pipelines_dir, "smqtk_make_balltree.json")

def query_yes_no(question, default="yes"):
  valid = {"yes": True, "y": True, "ye": True,
           "no": False, "n": False}
  if default is None:
    prompt = " [y/n] "
  elif default == "yes":
    prompt = " [Y/n] "
  elif default == "no":
    prompt = " [y/N] "
  else:
    raise ValueError("invalid default answer: '%s'" % default)

  while True:
    sys.stdout.write(question + prompt)
    if sys.version_info[0] < 3:
      choice = raw_input().lower()
    else:
      choice = input().lower()
    sys.stdout.write( "\n" )
    if default is not None and choice == '':
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write("Please respond with 'yes' or 'no' "
                       "(or 'y' or 'n').\n")

def remove_dir( dirname ):
  if os.path.exists( dirname ):
    if query_yes_no( "\nYou are about to reset \"" + dirname + "\", continue?" ):
      shutil.rmtree( dirname )
    else:
      print( "Exiting without initializing database" )
      sys.exit(0)

def is_windows():
  return ( os.name == 'nt' )
      
def format_cmd( cmd ):
  if is_windows():
    return cmd + ".exe"
  else:
    return cmd

def format_pycmd( install_dir, cmd ): # special use case for SMQTK tools
  if len( install_dir ) > 0:
    python_prefix = "Python" if is_windows() else "python"
    output = [ sys.executable, os.path.join(
        install_dir,
        python_prefix + str( sys.version_info[0] ) + str( sys.version_info[1] ),
        "site-packages",
        "smqtk",
        "bin",
        cmd + ".py" )
    ]
    return output
  elif is_windows():
    return [ cmd + ".exe" ]
  else:
    return [ cmd ]

def execute_cmd( cmd, args ):
  all_args = [ format_cmd( cmd ) ]
  all_args.extend( args )
  subprocess.check_call( all_args )

def execute_pycmd( install_dir, cmd, args ):
  all_args = format_pycmd( install_dir, cmd )
  all_args.extend( args )
  subprocess.check_call( all_args )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def find_config( filename ):
  if os.path.exists( filename ):
    return filename
  elif os.path.exists( os.path.join( get_script_path(), filename ) ):
    return os.path.join( get_script_path(), filename )
  else:
    print( "Unable to find " + filename )
    sys.exit( 0 )
  
def init():
  # Kill and remove existing database
  stop()

  remove_dir( database_dir )

  # Generate new database
  execute_cmd( "initdb", [ "-D", sql_dir ] )
  start()
  status()
  execute_cmd( "createuser", [ "-e", "-E", "-s", "-i", "-r", "-d", "postgres" ] )
  execute_cmd( "psql", [ "-f", find_config( init_file ), "postgres" ] )

def status():
  execute_cmd( "pg_ctl", [ "-D", sql_dir, "status" ] )

def start():
  execute_cmd( "pg_ctl", [ "-D", sql_dir, "-w", "-t", "20", "-l", log_file, "start" ] )

def stop():
  try:
    execute_cmd( "pg_ctl", [ "-D", sql_dir, "stop" ] )
    if is_windows():
      execute_cmd( "net", [ "stop", "postgresql-x64-9.5 (64-bit windows)" ] )
    else:
      execute_cmd( "pkill", [ "postgres" ] )
  except subprocess.CalledProcessError:
    # Most likely happened because the database wasn't started in the first place.
    # No problem - just ignore the error.
    pass

def build_balltree_index( install_dir="" ):
  build_standard_index( install_dir )
  print( "3. Generating Ball Tree" )
  execute_pycmd( install_dir, "make_balltree",
    [ "-vc", find_config( SMQTK_BTREE_CONFIG ) ] )

def build_standard_index( install_dir="" ):
  print( "1. Training ITQ Model" )
  execute_pycmd( install_dir, "train_itq",
    [ "-vc", find_config( SMQTK_ITQ_TRAIN_CONFIG ) ] )
  print( "2. Computing Hash Codes" )
  execute_pycmd( install_dir, "compute_hash_codes",
    [ "-vc", find_config( SMQTK_HCODE_CONFIG ) ] )

def output_usage():
  print( "Usage: database_tool.py [initialize | status | start | stop | index]" )
  sys.exit( 0 )
  
# Main Function
if __name__ == "__main__" :

  if len( sys.argv ) != 2:
    output_usage()

  if sys.argv[1] == "init" or sys.argv[1] == "initialize":
    init()
  elif sys.argv[1] == "status":
    status()
  elif sys.argv[1] == "start":
    start()
  elif sys.argv[1] == "stop":
    stop()
  elif sys.argv[1] == "index" or sys.argv[1] == "build_index":
    build_standard_index()
  elif sys.argv[1] == "build_balltree":
    build_balltree_index()
  else:
    output_usage()
