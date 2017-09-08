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
def list_files_in_dir( folder, extension ):
  return glob.glob( folder + '/*' + extension )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

def does_file_exist( filename ):
  return os.path.isfile( filename )

# Main Utility
def generate_yolo_headers( input_dir, output_dir, type_str, image_ext, test_per ):
arg
  # Check arguments
  args = parser.parse_args()

  if not does_file_exist( input_dir + "/labels.txt" ):
    print( "labels.txt must exist in the input directory" )
    sys.exit(0)

  with open( input_dir + "/labels.txt", "r" ) as f:
    content = f.readlines()
  content = [x.strip() for x in content]

  parsed_lbls = []
  for line in content:
    parsed_lbls.append( line.split( None, 1 )[0] )

  # Dump out labels file
  with open( output_dir + "/" + type_str + ".lbl", "w" ) as f:
    for item in parsed_lbls:
      f.write( item + "\n" )

  # Dump out special files for varients
  if type_str == "YOLO" or type_str == "YOLOv2":
    with open( output_dir + "/" + type_str + ".data", "w" ) as f:
      f.write( "train = " + output_dir + "/train_files.txt\n" )
      f.write( "valid = " + output_dir + "/test_files.txt\n" )
      f.write( "names = " + type_str + ".lbl\n" )
      f.write( "backup = " + output_dir + "/models\n" )

  # Dump out list files
  create_dir( output_dir + "/models" )

  image_list = list_files_in_dir( output_dir + "/formatted_samples", image_ext )
  shuffled_list = image_list
  random.shuffle( shuffled_list )

  pivot = int( ( 1.0 - test_per ) * len( shuffled_list ) )

  train_list = shuffled_list[:pivot]
  test_list = shuffled_list[pivot:]

  with open( output_dir + "/train_files.txt", "w" ) as f:
    for item in train_list:
      f.write( item + "\n" )

  with open( output_dir + "/test_files.txt", "w" ) as f:
    for item in test_list:
      f.write( item + "\n" )
