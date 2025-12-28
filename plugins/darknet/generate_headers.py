#!/usr/bin/env python

import sys
import os
import glob
import numpy as np
import argparse
import math
import random

# Helper Utilities
if os.name == 'nt':
  div = '\\'
else:
  div = '/'

def list_files_in_dir( folder, extension ):
  output = glob.glob( folder + '/*' + extension )

  index = 1
  while True:
    ending_pf = "_" + str( index ) + extension
    initial_size = len( output )
    output = [v for v in output if not v.endswith( ending_pf )]
    if initial_size == len( output ):
      break;
    index = index + 1

  return output

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

def generate_kwiver_pipeline(
    input_pipe, output_pipe, net_config, wgt_file, lbl_file ):

  repl_strs = [ [ "[-NETWORK-CONFIG-]", net_config ],
                [ "[-NETWORK-WEIGHTS-]", wgt_file ],
                [ "[-NETWORK-CLASSES-]", lbl_file ] ]

  replace_str_in_file( input_pipe, output_pipe, repl_strs )

def generate_yolo_headers(
    working_dir, labels, width, height, channels, filter_count,
    batch_size, batch_subdivisions, input_model, samp_count, gt_count,
    output_str="yolo", image_ext=".png", test_per=0.05 ):

  # Check arguments
  if len( labels ) < 0:
    print( "Must specify labels vector" )
    sys.exit(0)

  # Hard coded configs
  label_file = output_str + ".lbl"
  train_conf_file = output_str + ".cfg"
  test_conf_file = output_str + "_test.cfg"
  train_file = output_str + ".data"

  # Approximate
  max_batches = 7500
  if int( samp_count ) > max_batches:
    max_batches = int( samp_count )
  if max_batches > 100000:
    max_batches = 100000

  step1 = int( max_batches * 2 / 3 )
  step2 = int( max_batches * 5 / 6 )

  # Dump out adjusted network file
  train_repl_strs = [ ["[-HEIGHT_INSERT-]",str(height)],
                      ["[-WIDTH_INSERT-]",str(width)],
                      ["[-CHANNEL_INSERT-]",str(channels)],
                      ["[-FILTER_COUNT_INSERT-]",str(filter_count)],
                      ["[-BATCH_SIZE_INSERT-]",str(batch_size)],
                      ["[-BATCH_SUBDIVISIONS_INSERT-]",str(batch_subdivisions)],
                      ["[-MAX_BATCHES-]",str(max_batches)],
                      ["[-STEP1-]",str(step1)],
                      ["[-STEP2-]",str(step2)],
                      ["[-CLASS_COUNT_INSERT-]",str(len(labels))] ]

  test_repl_strs = [ ["[-HEIGHT_INSERT-]",str(height)],
                     ["[-WIDTH_INSERT-]",str(width)],
                     ["[-CHANNEL_INSERT-]",str(channels)],
                     ["[-FILTER_COUNT_INSERT-]",str(filter_count)],
                     ["[-BATCH_SIZE_INSERT-]","1"],
                     ["[-BATCH_SUBDIVISIONS_INSERT-]","1"],
                     ["[-MAX_BATCHES-]",str(max_batches)],
                     ["[-STEP1-]",str(step1)],
                     ["[-STEP2-]",str(step2)],
                     ["[-CLASS_COUNT_INSERT-]",str(len(labels))] ]

  output_train_cfg = working_dir + div + train_conf_file
  output_test_cfg = working_dir + div + test_conf_file

  replace_str_in_file( input_model, output_train_cfg, train_repl_strs )
  replace_str_in_file( input_model, output_test_cfg, test_repl_strs )

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
