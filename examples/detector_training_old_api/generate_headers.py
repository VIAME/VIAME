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
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Format data for object detection training",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("-i", dest="input_dir", default=None, required=True,
                      help="Input directory.")

  parser.add_argument("-o", dest="output_dir", default=".",
                      help="Output directory.")

  parser.add_argument("-t", dest="type_str", default="",
                      help="Type of data to format for.")

  parser.add_argument("-e", dest="image_ext", default=".png",
                      help="Type of data to format for.")

  parser.add_argument("-p", dest="train_per", default="0.90", type=float,
                      help="Percentage of samples to retain in training set.")

  # Check arguments
  args = parser.parse_args()

  if not does_file_exist( args.input_dir + "/labels.txt" ):
    print( "labels.txt must exist in the input directory" )
    sys.exit(0)

  with open( args.input_dir + "/labels.txt", "r" ) as f:
    content = f.readlines()
  content = [x.strip() for x in content]

  parsed_lbls = []
  for line in content:
    parsed_lbls.append( line.split( None, 1 )[0] )

  # Dump out labels file
  with open( args.output_dir + "/" + args.type_str + ".lbl", "w" ) as f:
    for item in parsed_lbls:
      f.write( item + "\n" )

  # Dump out special files for varients
  if args.type_str == "YOLO" or args.type_str == "YOLOv2":
    with open( args.output_dir + "/" + args.type_str + ".data", "w" ) as f:
      f.write( "train = " + args.output_dir + "/train_files.txt\n" )
      f.write( "valid = " + args.output_dir + "/test_files.txt\n" )
      f.write( "names = " + args.type_str + ".lbl\n" )
      f.write( "backup = models\n" )

  # Dump out list files
  create_dir( args.output_dir + "/models" )

  image_list = list_files_in_dir( args.output_dir + "/formatted_samples", args.image_ext )
  shuffled_list = image_list
  random.shuffle( shuffled_list )

  pivot = int( args.train_per * len( shuffled_list ) )
  train_list = shuffled_list[:pivot]
  test_list = shuffled_list[pivot:]

  with open( args.output_dir + "/train_files.txt", "w" ) as f:
    for item in train_list:
      f.write( item + "\n" )

  with open( args.output_dir + "/test_files.txt", "w" ) as f:
    for item in test_list:
      f.write( item + "\n" )
  
