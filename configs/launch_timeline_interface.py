#!python

import sys
import os
import glob
import argparse
import atexit
import shutil
import tempfile
import subprocess

if os.name == 'nt':
  div = '\\'
else:
  div = '/'

temp_dir = tempfile.mkdtemp(prefix='viqui-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

# Helper class to list files with a given extension in a directory
def list_files_in_dir( folder ):
  return glob.glob( folder + '/*' )

def list_files_in_dir( folder, extension ):
  return glob.glob( folder + '/*' + extension )

def get_script_path():
  return os.path.dirname( os.path.realpath( sys.argv[0] ) )

def get_gui_cmd():
  if os.name == 'nt':
    return ['vsPlay.exe']
  else:
    return ['vsPlay']

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

def select_option( option_list, display_str="Select File:" ):
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

  return option_list[ int(choice)-1 ]

# Main Function
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Launch track viewer GUI",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-d", dest="input_dir", default="database",
                      help="Input directory containing results")

  parser.add_argument("-t", dest="threshold",
                      default="0.0",
                      help="Optional detection threshold to apply")

  parser.add_argument("-theme", dest="gui_theme_file",
                      default="gui-params" + div + "dark_gui_settings.ini",
                      help="GUI default theme settings")

  parser.add_argument("-filter", dest="filter_file",
                      default="gui-params" + div + "default_timeline_filter.vpefs",
                      help="GUI default filter settings")

  args = parser.parse_args()

  files = list_files_in_dir( args.input_dir, "index" )

  if len( files ) == 0:
    print( "Error: No computed results in input directory" )
    sys.exit(0)

  files.sort()
  filename = select_option( files, "Select File:" )

  base, ext = os.path.splitext( filename )

  # Find detection file and confirm it exits
  detection_file = os.path.join( base + "_tracks.csv" )
  if not os.path.isfile( detection_file ):
    detection_file = os.path.join( base + "_detections.csv" )
    if not os.path.isfile( detection_file ):
      print( "Error: Detection file does not exist" )
      sys.exit(0)

  # Look for object category instances in file
  unique_ids = set()
  with open( detection_file ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 10:
        continue
      if len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      idx = 9
      while idx < len( lis ):
        unique_ids.add( lis[idx] )
        idx = idx + 2

  if len( unique_ids ) == 0:
    print( "Error: Detection file contains no categories" )
    sys.exit(0)

  all_category = "all_categories" if list(unique_ids)[0][0].islower() else "All Categories"

  category_list = [all_category] + list(unique_ids)
  category = select_option( category_list, "Select Category:" )

  # Open index file and get timestamp vector
  ts_vec = []
  counter = 0
  with open( filename ) as f:
    for line in f:
      counter = counter + 1
      if counter < 6:
        continue
      lis = line.strip().split()
      if len( lis ) != 2:
        continue
      ts_vec.append( str( float( lis[0] ) / 1e6 ) )

  if len( ts_vec ) == 0:
    print( "Error: Selected video file is empty" )
    sys.exit(0)

  # Perform conversion and thresholding
  (fd1, track_file) = tempfile.mkstemp( prefix='vsplay-tmp-tracks-',
                                        suffix='.kw18',
                                        text=True, dir=temp_dir )
  (fd2, class_file) = tempfile.mkstemp( prefix='vsplay-tmp-tracks-',
                                        suffix='.fso.txt',
                                        text=True, dir=temp_dir )

  ftrk = os.fdopen( fd1, 'w' )
  fcls = os.fdopen( fd2, 'w' )

  with open( detection_file ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 10:
        continue
      if len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      idx = 9
      use_detection = False
      confidence = 0.0
      while idx < len( lis ):
        if ( category == all_category or lis[idx] == category ) and \
             float( lis[idx+1] ) >= float( args.threshold ):
          use_detection = True
          confidence = float( lis[idx+1] )
          break
        idx = idx + 2

      # Column(s) 1: Track-id
      # Column(s) 2: Track-length (# of detections)
      # Column(s) 3: Frame-number (-1 if not available)
      # Column(s) 4-5: Tracking-plane-loc(x,y) (Could be same as World-loc)
      # Column(s) 6-7: Velocity(x,y)
      # Column(s) 8-9: Image-loc(x,y)
      # Column(s) 10-13: Img-bbox(TL_x,TL_y,BR_x,BR_y) (location of top-left & bottom-right vertices)
      # Column(s) 14: Area
      # Column(s) 15-17: World-loc(x,y,z) (longitude, latitude, 0 - when available)
      # Column(s) 18: Timesetamp(-1 if not available)
      # Column(s) 19: Track-confidence(-1_when_not_available)
      #    TO
      # 0: Detection or Track Unique ID
      # 1: Video or Image String Identifier
      # 2: Unique Frame Integer Identifier
      # 3: TL-x (top left of the image is the origin: 0,0)
      # 4: TL-y
      # 5: BR-x
      # 6: BR-y
      # 7: Detection Confidence (how likely is this actually an object)
      # 8: Target Length
      # 9,10+ : class-name score (this pair may be omitted or repeated)

      c_x = int( ( float( lis[3] ) + float( lis[5] ) ) / 2 )
      c_y = int( ( float( lis[4] ) + float( lis[6] ) ) / 2 )

      ftrk.write( lis[0] + ' 1 ' + lis[2] + ' 0 0 0 0 ' + str( c_x ) + ' ' + str( c_y ) )
      ftrk.write( ' ' + lis[3] + ' ' + lis[4] + ' ' + lis[5] + ' ' + lis[6] + ' 0 0 0 0 ' )
      ftrk.write( ts_vec[ int( lis[2] ) ]  + ' ' + str( confidence ) + '\n' )

      fcls.write( lis[0] + ' ' + str( confidence ) + ' 0 ' + str( 1.0 - confidence ) + '\n')

  ftrk.close()
  fcls.close()

  sys.stdout.write( "\n" )

  cmd = get_gui_cmd() + [ "-tf", track_file, "-vf", filename, "-df", class_file ]

  if len( args.gui_theme_file ) > 0:
    cmd = cmd + [ "--theme", find_file( args.gui_theme_file ) ]
  if len( args.filter_file ) > 0:
    cmd = cmd + [ "-ff", find_file( args.filter_file ) ]

  subprocess.call( cmd )
