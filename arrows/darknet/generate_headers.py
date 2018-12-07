#!/usr/bin/env python

import sys
import os
import glob
import numpy as np
import cv2
import argparse
import math
import random

# Helper Utilities
if os.name == 'nt':
  div = '\\'
else:
  div = '/'

def list_files_in_dir( folder, extension ):
  return glob.glob( folder + '/*' + extension )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

def does_file_exist( filename ):
  return os.path.isfile( filename )

def replace_str_in_file( input_fn, output_fn, repl_array ):
  inputf = open( input_fn )
  outputf = open( output_fn, 'w' )
  all_lines = []
  for s in list( inputf ):
    all_lines.append( s )
  for repl in repl_array:
    for i, s in enumerate( all_lines ):
      all_lines[i] = s.replace( repl[0], repl[1] )
  for s in all_lines:
    outputf.write( s )
  outputf.close()
  inputf.close()

# Main Utility
def generate_yolo_v3_headers( working_dir, labels, width, height, input_model, \
  output_str="yolo_v3", image_ext=".png", test_per=0.05 ):

  # Check arguments
  if len( labels ) < 0:
    print( "Must specify labels vector" )
    sys.exit(0)

  # Hard coded configs
  label_file = output_str + ".lbl"
  conf_file = output_str + ".cfg"
  train_file = output_str + ".data"

  # Dump out adjusted network file
  repl_strs = [ ["[-HEIGHT_INSERT-]",str(height)], \
                ["[-WIDTH_INSERT-]",str(width)], \
                ["[-FILTER_COUNT_INSERT-]",str((len(labels)+5)*3)], \
                ["[-CLASS_COUNT_INSERT-]",str(len(labels))] ]

  replace_str_in_file( input_model, working_dir + div + conf_file, repl_strs )

  # Dump out labels file
  with open( working_dir + div + label_file, "w" ) as f:
    for item in labels:
      f.write( item + "\n" )

  # Dump out special files for varients
  with open( working_dir + div + train_file, "w" ) as f:
    f.write( "train = " + working_dir + div + "train_files.txt\n" )
    f.write( "valid = " + working_dir + div + "test_files.txt\n" )
    f.write( "names = " + label_file + "\n" )
    f.write( "backup = " + working_dir + div + "models\n" )

  # Dump out list files
  create_dir( working_dir + div + "models" )

  image_list = list_files_in_dir( working_dir + div + "train_images", image_ext )
  shuffled_list = image_list
  random.shuffle( shuffled_list )

  pivot = int( ( 1.0 - test_per ) * len( shuffled_list ) )

  train_list = shuffled_list[:pivot]
  test_list = shuffled_list[pivot:]

  with open( working_dir + div + "train_files.txt", "w" ) as f:
    for item in train_list:
      f.write( item + "\n" )

  with open( working_dir + div +"test_files.txt", "w" ) as f:
    for item in test_list:
      f.write( item + "\n" )
