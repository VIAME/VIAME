#!python

import sys
import os
import os.path
import shutil
import subprocess
import threading

database_dir = "database"
pipelines_dir = "pipelines"
status_log_file = ""

sql_dir = os.path.join(database_dir, "SQL")
sql_init_file = os.path.join(pipelines_dir, "sql_init_table.sql")
sql_log_file = os.path.join(database_dir, "SQL_Log_File")

smqtk_itq_train_config = os.path.join(pipelines_dir, "smqtk_train_itq.json")
smqtk_hcode_config = os.path.join(pipelines_dir, "smqtk_compute_hashes.json")
smqtk_hcode_pickle = os.path.join(database_dir, "ITQ", "lsh_hash_to_descriptor_ids.pickle")
smqtk_btree_config = os.path.join(pipelines_dir, "smqtk_make_balltree.json")

lb1 = '\n'
lb2 = lb1 + lb1
lb3 = lb2 + lb1

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

def animated_loading():
  chars = "/-\\|" 
  for char in chars:
    sys.stdout.write( '\r' + char )
    time.sleep( 0.1 )
    sys.stdout.flush()

def run_with_animation( fun, *args ):
  process = threading.Thread( name='process', target=fun, kwargs=args )
  process.start()
  while process.isAlive():
    animated_loading();

def remove_file( filename ):
  if os.path.exists( filename ):
    os.remove( filename )

def is_windows():
  return ( os.name == 'nt' )
      
def format_cmd( cmd ):
  if is_windows():
    return cmd + ".exe"
  else:
    return cmd

def format_pycmd( install_dir, cmd ): # special use case for SMQTK tools
  if len( install_dir ) > 0:
    python_postfix = str( sys.version_info[0] ) + "." + str( sys.version_info[1] )

    script_path = os.path.join(
      install_dir, "lib/python" + python_postfix,
      "site-packages", "smqtk", "bin", cmd + ".py" )

    if not os.path.exists( script_path ):
      python_postfix = str( sys.version_info[0] ) + str( sys.version_info[1] )

      script_path = os.path.join(
        install_dir, "Python" + python_postfix,
        "site-packages", "smqtk", "bin", cmd + ".py" )

    return [ sys.executable, script_path ]
  elif is_windows():
    return [ cmd + ".exe" ]
  else:
    return [ cmd ]

def setup_stream( log_file ):
  log = None
  if len( log_file ) > 0:
    if log_file == "NULL":
      log=open( os.devnull, 'w' )
    else:
      log_dir = os.path.dirname( log_file )
      if not os.path.exists( log_dir ):
        os.makedirs( log_dir )
      log = open( log_file, 'a' )
  return log

def execute_cmd( cmd, args ):
  all_args = [ format_cmd( cmd ) ]
  all_args.extend( args )
  log = setup_stream( status_log_file )
  subprocess.check_call( all_args, stdout=log, stderr=log )

def execute_pycmd( install_dir, cmd, args ):
  all_args = format_pycmd( install_dir, cmd )
  all_args.extend( args )
  log = setup_stream( status_log_file )
  subprocess.check_call( all_args, stdout=log, stderr=log )

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

def log_info( msg ):
  sys.stdout.write( msg )
  sys.stdout.flush()

def init( log_file="", prompt=True ):

  # Set new log file, flush file since this is a new database
  global status_log_file
  status_log_file = log_file

  if len( log_file ) > 0:
    remove_file( log_file )

  try:
    # Kill and remove existing database, call may fail (if no existing db) and that's okay
    stop( quiet=True )

    # Remove directory, will be remade in next step, query user in case this was a mistake
    if os.path.exists( database_dir ):
      if not prompt or query_yes_no( lb1 + "You are about to reset \"" + database_dir + "\", continue?" ):
        shutil.rmtree( database_dir )
      else:
        return [ False, True ]
    else:
      log_info( lb1 )

    # Generate new database
    log_info( "Initializing database... " )
    execute_cmd( "initdb", [ "-D", sql_dir ] )
    execute_cmd( "pg_ctl", [ "-D", sql_dir, "-w", "-t", "20", "-l", sql_log_file, "start" ] )
    execute_cmd( "pg_ctl", [ "-D", sql_dir, "status" ] )
    execute_cmd( "createuser", [ "-e", "-E", "-s", "-i", "-r", "-d", "postgres" ] )
    execute_cmd( "psql", [ "-f", find_config( sql_init_file ), "postgres" ] )
    log_info( "Success" + lb1 )
    return [ True, True ]

  except:
    log_info( "Failure" + lb1 )
    return [ False, False ]

def status():
  execute_cmd( "pg_ctl", [ "-D", sql_dir, "status" ] )

def start( quiet=False ):
  global status_log_file
  original_log_file = status_log_file
  status_log_file= "NULL" if quiet else status_log_file
  try:
    execute_cmd( "pg_ctl", [ "-D", sql_dir, "-w", "-t", "20", "-l", sql_log_file, "start" ] )
    status_log_file = original_log_file
    return True
  except:
    status_log_file = original_log_file
    return False

def stop( quiet=False ):
  global status_log_file
  original_log_file = status_log_file
  status_log_file= "NULL" if quiet else status_log_file
  try:
    execute_cmd( "pg_ctl", [ "-D", sql_dir, "stop" ] )
  except subprocess.CalledProcessError:
    # Most likely happened because the database wasn't started in the first place.
    # No problem - just ignore the error.
    pass
  try:
    if is_windows():
      execute_cmd( "net", [ "stop", "postgresql-x64-9.5 (64-bit windows)" ] )
    else:
      execute_cmd( "pkill", [ "postgres" ] )
  except subprocess.CalledProcessError:
    pass
  status_log_file = original_log_file

def build_balltree_index( install_dir="", log_file="" ):
  if not build_standard_index( install_dir, log_file ):
    return False
  try:
    global status_log_file
    status_log_file = log_file
    log_info( "  (3/3) Generating Ball Tree... " )
    execute_pycmd( install_dir, "make_balltree",
      [ "-vc", find_config( smqtk_btree_config ) ] )
    log_info( "Success" + lb1 )
    return True
  except:
    if len( log_file ) > 0:
      log_info( "Failure" + lb1 + "  Check log: " + log_file + lb2 )
    return False

def build_standard_index( install_dir="", log_file="" ):
  try:
    global status_log_file
    status_log_file = log_file
    log_info( "  (1/2) Training ITQ Model... " )
    execute_pycmd( install_dir, "train_itq",
      [ "-vc", find_config( smqtk_itq_train_config ) ] )
    log_info( "Success" + lb1 + "  (2/2) Computing Hash Codes... " )
    execute_pycmd( install_dir, "compute_hash_codes",
      [ "-vc", find_config( smqtk_hcode_config ) ] )
    log_info( "Success" + lb1 )
    return True
  except:
    if len( log_file ) > 0:
      log_info( "Failure" + lb1 + "  Check log: " + log_file + lb2 )
    return False

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
