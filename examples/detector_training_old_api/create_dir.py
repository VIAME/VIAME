#!/usr/bin/env python

import sys
import os
import glob
import numpy as np
import cv2
import argparse
import math

# Helper Utilities
def remove_contents( top ):
  for root, dirs, files in os.walk( top, topdown=False ):
    for name in files:
        os.remove( os.path.join( root, name ) )
    for name in dirs:
        os.rmdir( os.path.join( root, name ) )

def create_dir( dirname ):
  if not os.path.exists( dirname ):
    os.makedirs( dirname )

def does_folder_exist( dirname ):
  return os.path.isdir( dirname )

def query_yes_no( question, default="no" ):
  """Ask a yes/no question via raw_input() and return their answer.

  "question" is a string that is presented to the user.
  "default" is the presumed answer if the user just hits <Enter>.
    It must be "yes" (the default), "no" or None (meaning
    an answer is required of the user).

  The "answer" return value is True for "yes" or False for "no".
  """
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
    choice = raw_input().lower()
    if default is not None and choice == '':
      return valid[default]
    elif choice in valid:
      return valid[choice]
    else:
      sys.stdout.write("Please respond with 'yes' or 'no' "
                       "(or 'y' or 'n').\n")

# Main Utility
if __name__ == "__main__" :

  parser = argparse.ArgumentParser(description="Create directory",
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument( "-d", dest="dir_name", default=None, required=True,
                       help="Input directory." )

  # Check arguments
  args = parser.parse_args()

  if not does_folder_exist( args.dir_name ):
    create_dir( args.dir_name )
    sys.exit( 0 )

  if query_yes_no( "Before training, do you want to remove the contents of folder: " + args.dir_name + "?" ):
    remove_contents( args.dir_name )
    create_dir( args.dir_name )
  
