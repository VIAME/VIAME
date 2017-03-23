#!/usr/bin/env python

import sys
import os
import glob
import numpy as np
import cv2
import argparse
import math

from shutil import copyfile

# Parsing Functions
def parse_habcam( input_file ):
  output = dict()
  for line in open( input_file ):
    parsed_line = filter( None, line.split() )

    if len( parsed_line ) < 1:
      continue
    if parsed_line[0] == '#':
      continue

    image_name = parsed_line[0]

    if not image_name in output:
      output[ image_name ] = []
    if len( parsed_line ) < 2:
      continue

    sid = int( parsed_line[1] )
    if sid in ( 524, 533, 1003, 1001 ): # Fish
      sid = 0
    elif sid in ( 185, 197, 207, 211, 515, 912, 919, 920 ): # Live Scallop
      sid = 1
    elif sid in ( 188, 99999 ): # Dead Scallop
      sid = 2
      continue # Ignore for now
    elif sid in ( 158, 258 ): # Crab
      sid = 3
      continue # Ignore for now
    else:
      continue

    if len( parsed_line ) < 8:
      continue

    entry = [ sid, 0.0, 0.0, 0.0, 0.0 ]

    p1x = float( parsed_line[4] )
    p1y = float( parsed_line[5] )
    p2x = float( parsed_line[6] )
    p2y = float( parsed_line[7] )

    if parsed_line[3] == 'boundingBox':
      entry[1] = max( min( p1x, p2x ), 0.0 )
      entry[2] = max( min( p1y, p2y ), 0.0 )
      entry[3] = abs( p2x - p1x )
      entry[4] = abs( p2y - p1y )
    elif parsed_line[3] == 'line':
      cx = ( p1x + p2x ) / 2.0
      cy = ( p1y + p2y ) / 2.0
      r = math.sqrt( math.pow( p2x - p1x, 2 ) + math.pow( p2y - p1y, 2 ) ) / 2.0
      entry[1] = max( cx - r, 0.0 )
      entry[2] = max( cy - r, 0.0 )
      entry[3] = 2.0 * r
      entry[4] = 2.0 * r
    else:
      continue

    output[ image_name ].append( entry )

  return output

def parse_clef( input_file ):
  import xmltodict
  output = dict()

  with open( input_file ) as fd:
    doc = xmltodict.parse( fd.read() )
  video_name = doc['video']['@id']
  fids = doc['video']['frame']
  for frame in fids:
    frame_id = int( frame['@id'] )
    frame_str = "frames" + str( frame_id ).zfill( 6 ) + ".png"
    for obj in frame['object']:
      if isinstance( obj, dict ):
        if not frame_str in output:
          output[ frame_str ] = []
        entry = [ 0, 0.0, 0.0, 0.0, 0.0 ]
        entry[1] = float( obj['@x'] )
        entry[2] = float( obj['@y'] )
        entry[3] = float( obj['@w'] )
        entry[4] = float( obj['@h'] )
        output[ frame_str ].append( entry )
  return output

def parse_wild( input_file ):
  output = dict()
  for line in open( input_file ):
    parsed_line = filter( None, line.split() )

    if len( parsed_line ) < 3:
      continue

    image_name = parsed_line[0]

    if not image_name in output:
      output[ image_name ] = []

    for i in range( 0, int( parsed_line[1] ) ):
      entry = [ 0, 0.0, 0.0, 0.0, 0.0 ]
      entry[1] = float( parsed_line[4*i+2] )
      entry[2] = float( parsed_line[4*i+3] )
      entry[3] = float( parsed_line[4*i+4] )
      entry[4] = float( parsed_line[4*i+5] )
      output[ image_name ].append( entry )

  return output

def parse_camtrawl( input_file ):
  output = dict()
  return output

def parse_file( input_file, format_id ):

  if format_id == "habcam":
    return parse_habcam( input_file )
  elif format_id == "clef":
    return parse_clef( input_file )
  elif format_id == "wild":
    return parse_wild( input_file )
  elif format_id == "camtrawl":
    return parse_camtrawl( input_file )

  print( "Invalid file format specified" )
  sys.exit()

# Helper Utilities
def list_files_in_dir( folder, extension ):
  return glob.glob( folder + '/*' + extension )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

def does_file_exist( filename ):
  return os.path.isfile( filename )

# Main Utility
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Format data for object detection training",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-i", dest="input_file", default=None, required=True,
                      help="Input text file containing annotations.")

  parser.add_argument("-o", dest="output_folder", default="formatted_samples",
                      help="Output folder where training samples will be placed.")

  parser.add_argument("-v", dest="validation_folder", default="",
                      help="Folder to store validation imagery")

  parser.add_argument("-f", dest="format_type", default="",
                      help="Groundtruth file format: habcam, clef, wild, camtrawl.")

  parser.add_argument("-e", dest="exclude_list", default="",
                      help="A txt file containing a list of images to exclude in training.")

  parser.add_argument("-s", dest="string_adj", default="",
                      help="A string prefix for output files to prevent duplicates")

  parser.add_argument("-ni", dest="ni", default="-1", type=int,
                      help="Image number of columns to resize imagery to.")

  parser.add_argument("-nj", dest="nj", default="-1", type=int,
                      help="Image number of rows to resize imagery to.")

  parser.add_argument("--no-empty", dest="no_empty", action="store_true",
                      help="Only use frames containing objects of interest")

  parser.add_argument("--filter", dest="filter", action="store_true",
                      help="Filter input boxes to make sure they lie in the image.")

  parser.add_argument("--norm", dest="norm", action="store_true",
                      help="Normalize output coordinates")

  parser.add_argument("--gray", dest="gray", action="store_true",
                      help="Produce grayscale image outputs")

  parser.add_argument("--clip-right", dest="clip_right", action="store_true",
                      help="Produce grayscale image outputs")

  # Check arguments
  args = parser.parse_args()

  ni = args.ni
  nj = args.nj

  if ( ni > 0 and nj < 0 ) or ( ni < 0 and nj > 0 ):
    print( "Must specify both image width and height" )
    sys.exit() 

  resize_images = False

  if ni > 0 or nj > 0:
    resize_images = True

  input_file = args.input_file
  input_folder = os.path.dirname( input_file )
  output_folder = args.output_folder

  # Read exclude list
  exclude_list = []

  if len( args.exclude_list ) > 0:
    if not does_file_exist( args.exclude_list ):
      print( "Exlude file doesn't exist" )
      sys.exit( 0 )
    fin = open( args.exclude_list, 'r' )
    for line in fin.readlines():
      exclude_list.append( line.rstrip() )
    fin.close()

  # Make output directories
  create_dir( output_folder )

  if len( args.validation_folder ) > 0:
    create_dir( args.validation_folder )

  if not does_file_exist( input_file ):
    print( "Input groundtruth file does not exist" )
    sys.exit()

  # Parse input groundtruth file
  parsed_file = parse_file( input_file, args.format_type )

  # For every image file in the groundtruth file
  for image_name in parsed_file:

    # Get and check annotations for this image
    annotations = parsed_file[ image_name ]

    if len( annotations ) < 1 and args.no_empty:
      continue

    if image_name in exclude_list:
      continue

    image_fn = input_folder + '/' + image_name
    output_image_fn = os.path.splitext( output_folder + '/' + args.string_adj + image_name )[0] + '.png'
    output_txt_fn = os.path.splitext( output_folder + '/' + args.string_adj + image_name )[0] + ".txt"

    if not does_file_exist( image_fn ):
      print( "Missing file " + image_fn )
      continue

    # Compute scaled image if necessary
    im = cv2.imread( image_fn )
    height, width = im.shape[:2]
    scale = 1.0

    if args.clip_right:
      im = im[ 0:height, 0:int(width/2) ]
      height, width = im.shape[:2]

    if args.gray:
      im = cv2.cvtColor( im, cv2.COLOR_RGB2GRAY )

    if resize_images:
      if height > nj:
        scale = float( nj ) / height
      if width > ni:
        scale = min( scale, float( ni ) / width )
      res = cv2.resize( im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC )
      final = np.zeros( (nj, ni, res.shape[2]), res.dtype )
      final[ 0:res.shape[0], 0:res.shape[1] ] = res
      height, width = final.shape[:2]
    else:
      final = im

    # Write out image
    cv2.imwrite( output_image_fn, final )

    # Write out annotation file for image
    fout = open( output_txt_fn, "w" )
    for annotation in annotations:

      if args.filter:
        br_x = min( max( 2, int( annotation[1] ) ), width-2 )
        br_y = min( max( 2, int( annotation[2] ) ), height-2 )
        tr_x = min( max( 2, int( annotation[1] + annotation[3] ) ), width-2 )
        tr_y = min( max( 2, int( annotation[2] + annotation[4] ) ), height-2 )

        if tr_x-br_x <= 1 or tr_y-br_y <= 1:
          continue
  
        annotation = [ annotation[1], br_x, br_y, tr_x-br_x, tr_y-br_y ]
  
      if args.norm:
        fout.write( str( annotation[0] )
            + " " + str( ( annotation[1] + annotation[3] / 2.0 ) * scale / width )
            + " " + str( ( annotation[2] + annotation[4] / 2.0 ) * scale / height )
            + " " + str( annotation[3] * scale / width )
            + " " + str( annotation[4] * scale / height ) )
      elif resize_images:
        fout.write( str( annotation[0] )
            + " " + str( annotation[1] * scale )
            + " " + str( annotation[2] * scale )
            + " " + str( annotation[3] * scale )
            + " " + str( annotation[4] * scale ) )
      else:
        fout.write( str( annotation[0] )
            + " " + str( annotation[1] )
            + " " + str( annotation[2] )
            + " " + str( annotation[3] )
            + " " + str( annotation[4] ) )

      fout.write( '\n' )
    fout.close()

    # Write out validation image if enabled
    if len( args.validation_folder ) > 0:
      val_image_fn = args.validation_folder + '/' + args.string_adj + image_name
      for annotation in annotations:
        cv2.rectangle( final, ( int( annotation[1] * scale ),
                             int( annotation[2] * scale) ),
          ( int( ( annotation[1] + annotation[3] ) * scale ),
            int( ( annotation[2] + annotation[4] ) * scale ) ), 255, 2 )
      cv2.imwrite( val_image_fn, final )
