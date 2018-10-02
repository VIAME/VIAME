#!python

import sys
import os
import shutil
import subprocess

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

database_dir="database"
sql_dir=database_dir + div + "SQL"
init_file="pipelines" + div + "sql_init_table.sql"
log_file="database" + div + "SQL_Log_File"

SMQTK_ITQ_TRAIN_CONFIG="pipelines" + div + "smqtk_train_itq.json"
SMQTK_HCODE_CONFIG="pipelines" + div + "smqtk_compute_hashes.json"
SMQTK_HCODE_PICKLE="database" + div + "ITQ" + div + "alexnet_fc7.itq_b256_i50_n2_r0.lsh_hash2uuids.pickle"
SMQTK_BTREE_CONFIG="pipelines" + div + "smqtk_make_balltree.json"

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
  if is_windows():
    if len( install_dir ) > 0:
      output = "\"" + sys.executable + "\" \"" + install_dir + div + "Python" + str( sys.version_info[0] ) + str( sys.version_info[1] )
      output = output + div + "site-packages" + div + "smqtk" + div + "bin" + div + cmd + ".py\""
      return output
    else:
      return cmd + ".exe"
  else:
    if len( install_dir ) > 0:
      output = sys.executable + " " + install_dir + div + "lib" + div + "python" + str( sys.version_info[0] ) + "." + str( sys.version_info[1] )
      output = output + div + "site-packages" + div + "smqtk" + div + "bin" + div + cmd + ".py"
      return output
    else:
      return cmd

def sequence_cmd( prefix, cmd, args  ):
  return prefix + " && " + format_cmd( cmd ) + " " + args

def execute_cmd( cmd, args  ):
  subprocess.check_output( format_cmd( cmd ) + " " + args, shell=True )

def execute_pycmd( install_dir, cmd, args  ):
  os.system( format_pycmd( install_dir, cmd ) + " " + args )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def find_config( filename ):
  if( os.path.exists( filename ) ):
    return filename
  elif os.path.exists( get_script_path() + div + filename ):
    return get_script_path() + div + filename
  else:
    print( "Unable to find " + filename )
    sys.exit( 0 )
  
def init():
  # Kill and remove existing database
  execute_cmd( "pg_ctl", "stop -D " + sql_dir )

  if not is_windows():
    execute_cmd( "pkill", "postgres" )
  else:
    os.system( "net stop \"postgresql-x64-9.5 (64-bit windows)\" 2> nul" )

  remove_dir( database_dir )

  # Generate new database
  cmd = format_cmd( "initdb" ) + " -D " + sql_dir

  cmd = sequence_cmd( cmd, "pg_ctl", "-w -t 20 -D " + sql_dir + " -l " + log_file + " start" )
  cmd = sequence_cmd( cmd, "pg_ctl", "status -D " + sql_dir )
  cmd = sequence_cmd( cmd, "createuser", "-e -E -s -i -r -d postgres" )
  cmd = sequence_cmd( cmd, "psql", "-f \"" + find_config( init_file ) + "\" postgres" )

  os.system( cmd )

def status():
  execute_cmd( "pg_ctl", "status -D " + sql_dir )

def start():
  execute_cmd( "pg_ctl", "-w -t 20 -D " + sql_dir + " -l " + log_file + " start" )

def stop():
  execute_cmd( "pg_ctl", "stop -D " + sql_dir )
  if is_windows():
    os.system( "net stop \"postgresql-x64-9.5 (64-bit windows)\" 2> nul" )
  else:
     execute_cmd( "pkill", "postgres" )

def build_balltree_index( install_dir="" ):
  build_standard_index( install_dir )
  print( "3. Generating Ball Tree" )
  execute_pycmd( install_dir, "make_balltree",
    "-vc \"" + find_config( SMQTK_BTREE_CONFIG ) + "\"" )

def build_standard_index( install_dir="" ):
  print( "1. Training ITQ Model" )
  execute_pycmd( install_dir, "train_itq",
    "-v -c \"" + find_config( SMQTK_ITQ_TRAIN_CONFIG ) + "\"" )
  print( "2. Computing Hash Codes" )
  execute_pycmd( install_dir, "compute_hash_codes",
    "-v -c \"" + find_config( SMQTK_HCODE_CONFIG ) + "\"" )

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
