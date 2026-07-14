#!/usr/bin/python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #


import os
import sys
import re
import json
import csv
import math
import numpy as np
import argparse
import atexit
import tempfile
import subprocess
import shutil
import logging

from glob import glob
from pathlib import Path

from kwiver.vital.algo import (
    DetectedObjectSetInput,
    DetectedObjectSetOutput
)

from kwiver.vital.types import (
    BoundingBoxD, CategoryHierarchy,
    DetectedObjectSet, DetectedObject, DetectedObjectType
)

# ----------------- GLOBAL VARIABLES AND PROPERTIES --------------------

temp_dir = tempfile.mkdtemp( prefix='viame-score-tmp' )
atexit.register( lambda: shutil.rmtree( temp_dir ) )

hierarchy=None
default_label="fish"
min_conf = float( -10000 )
max_conf = -min_conf

# -------------------- GENERIC UTILITY FUNCTIONS -----------------------

def print_and_exit( msg, code=1 ):
  print( msg )
  sys.exit( code )

def log_with_spaces( msg ):
  logging.info( '' )
  logging.info( msg )
  logging.info( '' )

def log_and_write( fout, msg ):
  fout.write( msg + os.linesep ) 
  logging.info( msg )

def log_and_write_with_spaces( fout, msg ):
  fout.write( os.linesep + msg + os.linesep + os.linesep ) 
  log_with_spaces( msg )

def remove_if_exists( item ):
  if os.path.exists( item ):
    if os.path.isdir( item ):
      shutil.rmtree( item )
    else:
      os.remove( item )

def remake_dir( dirname ):
  remove_if_exists( dirname )
  os.mkdir( dirname )

def make_dir_if_not_exist( dirname ):
  if not os.path.exists( dirname ):
    os.mkdir( dirname )

def format_class_fn( fn ):
  return fn.replace( "/", "-" )

def float_to_str_safe( value ):
  try:
    return '{:.3f}'.format( value )
  except ValueError:
    print( "ValueError converting float value: " + value )
    return str( value )
  except Exception:
    print( "Unknown exception converting float value: " + value )
    return str( value )

def load_hierarchy( filename ):
  global hierarchy
  hierarchy = None
  if filename:
    hierarchy = CategoryHierarchy()
    try:
      hierarchy.load_from_file( filename )
    except Exception:
      print_and_exit( "Unable to parse classes file: " + filename )
    return True
  else:
    return False

def set_default_label( user_input=None ):
  global default_label
  if user_input:
    default_label = user_input
  elif hierarchy and len( hierarchy.all_class_names() ) == 1:
    default_label = hierarchy.all_class_names()[0]

def list_files_w_ext_rec( folder, ext ):
  result = [ y for x in os.walk( folder )
               for y in glob( os.path.join( x[0], '*' + ext ) ) ]
  return result

def safe_val( input_str ):
  try:
    out = float( input_str )
    if math.isnan( out ):
      return min_conf
    return out
  except Exception:
    return min_conf

def neg_safe_val( input_str ):
  return -safe_val( input_str )

# Given a text file with 1 filename per line, return list of filenames
def get_file_list_from_txt_list( filename ):
  out = []
  with open( filename ) as f:
    for line in f:
      line = line.rstrip()
      if len( line ) > 0:
        out.append( line )
  return out

# Dump csv of arbitrary scores given by a dict with category name
def create_net_csv( filename, scores, header ):
  with open( filename, 'w' ) as fout:
    fout.write( header + os.linesep )
    for key, value in scores.items():
      value_str = [ float_to_str_safe( x ) for x in value ]
      fout.write( key + "," + ','.join( value_str ) + os.linesep )
  return True

# Compute filename alignment of 2 seperate folders containing detections
def compute_alignment( computed_dir, truth_dir, ext = '.csv',
                       remove_postfix = '_detections',
                       skip_postfix = '_tracks' ):

  out = dict()

  computed_files = list_files_w_ext_rec( computed_dir, ext )
  truth_files = list_files_w_ext_rec( truth_dir, ext )

  for comp_file in computed_files:
    if skip_postfix and skip_postfix + ext in comp_file:
      continue
    if remove_postfix and remove_postfix + ext in comp_file:
      comp_no_postfix = comp_file.replace( remove_postfix + ext, ext )
    else:
      comp_no_postfix = comp_file
    comp_base = os.path.basename( comp_no_postfix )
    match = False
    for truth_file in truth_files:
      if comp_base == os.path.basename( truth_file ):
        out[ comp_file ] = truth_file
        match = True
        break
    if not match:
      print_and_exit( "Could not find corresponding truth for: " + comp_base )
  return out

# -------------- VIAME CSV-SPECIFIC UTILITY FUNCTIONS ------------------

def is_valid_viame_csv_entry( parsed_entry ):
  if len( parsed_entry ) < 9:
    return False
  if len( parsed_entry[0] ) > 0 and parsed_entry[0][0] == '#':
    return False
  return True

def list_entry_classes( parsed_entry, threshold=min_conf ):
  if len( parsed_entry ) < 9:
    return []
  cls_list = {}
  idx = 9
  top_score = min_conf
  top_category = None
  while idx < len( parsed_entry ):
    cls = parsed_entry[idx]
    if not cls or cls[0] == '(':
      break
    score = float( parsed_entry[idx+1] )
    if score >= threshold:
      if score > top_score:
        top_category = cls
        top_score = score
      cls_list[ cls ] = score
    idx = idx + 2
  return cls_list, top_category

def list_classes_viame_csv( input_fn, ext='.csv', top_only=True ):
  unique_ids = set()
  if os.path.isdir( input_fn ):
    for fn in list_files_w_ext_rec( input_fn, ext ):
      unique_ids = unique_ids.union( list_classes_viame_csv( fn, ext, top_only ) )
    return list( unique_ids )
  with open( input_fn ) as f:
    for line in f:
      parsed_entry = line.strip().split( ',' )
      if not is_valid_viame_csv_entry( parsed_entry ):
        continue
      classes, top = list_entry_classes( parsed_entry )
      if classes:
        if top_only:
          unique_ids.add( top )
        else:
          unique_ids = unique_ids.union( classes )
  return list( unique_ids )

def filter_viame_csv( fn, fout, target_cls=None,
                      threshold=min_conf, top_only=False ):
  with open( fn ) as fin:
    for line in fin:
      parsed_entry = line.strip().split(',')
      if not is_valid_viame_csv_entry( parsed_entry ):
        continue
      classes, top_cls = list_entry_classes( parsed_entry, threshold )
      if hierarchy:
        if not hierarchy.has_class_name( top_cls ):
          top_cls = None
        else:
          top_cls = hierarchy.get_class_name( top_cls )
        adj_classes = dict()
        for cls, score in classes.items():
          if not hierarchy.has_class_name( cls ):
            continue
          cls = hierarchy.get_class_name( cls )
          if cls in adj_classes and score < adj_classes[cls]:
            continue
          adj_classes[cls] = score
        classes = adj_classes
      parsed_entry = parsed_entry[0:9]
      if target_cls:
        if top_only and target_cls != top_cls:
          continue
        if target_cls not in classes:
          continue
        parsed_entry.extend( [ target_cls, str( classes[target_cls] ) ] )
      elif top_only:
        if not top_cls:
          continue
        parsed_entry.extend( [ top_cls, str( classes[top_cls] ) ] )
      else:
        for c in classes:
          parsed_entry.extend( [ c, str( classes[c] ) ] )
      fout.write( ','.join( parsed_entry ) + os.linesep )

def filter_viame_csv_tmp( fn, target_cls=None,
                          threshold=min_conf, top_only=False ):
  (fd, handle) = tempfile.mkstemp( prefix='viame-score-',
                                   suffix='.csv',
                                   text=True,
                                   dir=temp_dir )

  fout = os.fdopen( fd, 'w' )
  filter_viame_csv( fn, fout, target_cls, threshold, top_only )
  fout.close()
  return fd, handle

def filter_viame_csv_fixed( fn_in, fn_out, target_cls=None,
                            threshold=min_conf, top_only=False ):
  with open( fn_out, 'w' ) as fout:
    filter_viame_csv( fn_in, fout, target_cls, threshold, top_only )

def filter_viame_csv_auto( fn_in, fn_out, ext='.csv', target_class=None,
                           threshold=min_conf, top_only=False ):
  if os.path.isdir( fn_in ):
    fns = list_files_w_ext_rec( fn_in, ext )
    for subfile_in in fns:
      subfile_out = os.path.join( fn_out, os.path.basename( subfile_in ) )
      with open( subfile_out, 'w' ) as fout:
        filter_viame_csv( subfile_in, fout, target_class, threshold, top_only )
  else:
    with open( fn_out, 'w' ) as fout:
      filter_viame_csv( fn_in, fout, target_class, threshold, top_only )

# Returns joint filename list, if mismatched names found, max-frame-id
def get_file_list_from_viame_csvs( computed, truth ):
  (fd, handle) = tempfile.mkstemp( prefix='viame-list-',
                                   suffix='.txt',
                                   text=True,
                                   dir=temp_dir )
  fns = dict()
  dup = dict()
  mismatch = False
  with open( computed ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 6 or len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      fid = int( lis[2] )
      if lis[1] or not fid in fns:
        fns[ fid ] = lis[1]
      if lis[1] in dup:
        if dup[ lis[1] ] != fid:
          mismatch = True
      else:
        dup[ lis[1] ] = fid
  with open( truth ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 6 or len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      fid = int( lis[2] )
      if not fid in fns:
        fns[ fid ] = lis[1]
      elif lis[1]:
        if fns[ fid ] != lis[1]:
          mismatch = True
        fns[ fid ] = lis[1]
      if lis[1] in dup:
        if dup[ lis[1] ] != fid:
          mismatch = True
      else:
        dup[ lis[1] ] = fid
  out = []
  last_id = 0
  max_id = 0
  for fid, fn in fns.items():
    max_id = max( fid, max_id )
    if fid > last_id:
      for i in range( last_id, fid ):
        out.append( "unnamed_frame" + str(i) )
    out.append( fn )
    last_id = fid + 1
  return out, mismatch, max_id

# -------------------- KWIVER UTILITY FUNCTIONS ------------------------

def filter_kwiver_detections( dets,
                              target_cls = None,
                              ignore_all_cls = None,
                              top_cls_only = False,
                              thresh = 0.0 ):

  output = DetectedObjectSet()
  for i, item in enumerate( dets ):
    if item.type is None:
      if ignore_all_cls:
        # Ignore classes option with no real classes
        score = item.confidence
        item.type = DetectedObjectType( default_label, score )
      else:
        continue
    elif ignore_all_cls:
      # Ignore all classes option, relabel top class to default
      cls = item.type.get_most_likely_class()
      if hierarchy and not hierarchy.has_class_name( cls ):
        continue
      score = item.type.score( cls )
      item.type = DetectedObjectType( default_label, score )
    elif top_cls_only:
      # Top class option, only use highest scoring class
      cls = item.type.get_most_likely_class()
      score = item.type.score( cls )
      item.type = DetectedObjectType( cls, score )

    # Applies threshold parameter
    all_classes = item.type.class_names( thresh )

    # Apply hierarchy if present, also remakes type after threshold
    output_type = DetectedObjectType()
    for cls in all_classes:
      score = item.type.score( cls )
      if args.aux_confidence:
        score = item.confidence
      if hierarchy and not ignore_all_cls:
        if not hierarchy.has_class_name( cls ):
          continue
        else:
          cls = hierarchy.get_class_name( cls )
      if target_cls and cls != target_cls:
        continue
      if not output_type.has_class_name( cls ) or output_type.score( cls ) < score:
        output_type.set_score( cls, score )

    # If no classes left after filters, skip detection
    if len( output_type ) == 0:
      continue
    else:
      item.type = output_type
      output.add( item )
  return output

def filter_detections( args, dets,
                       target_class=None,
                       ignore_classes=None,
                       top_class=None,
                       threshold=None ):
  if not ignore_classes:
    ignore_classes = args.ignore_classes
  if not top_class:
    top_class = args.top_class
  if not threshold:
    threshold = args.threshold
  return filter_kwiver_detections( dets,
    target_class, ignore_classes, top_class, threshold )

def read_sets_by_image( csv_file ):
  """Map each image name in a VIAME CSV to its DetectedObjectSet.

  The kwiver python bindings do not expose a read-by-path call, and read_set()
  reports no image name: it simply yields one set per frame index, in order,
  including empty sets for frame indices that carry no annotations. So read the
  sets in order and pair them with the frame index to image name mapping taken
  from the file itself.
  """
  frame_to_image = dict()

  with open( csv_file, 'r' ) as fin:
    for row in csv.reader( fin ):
      if not row or row[0].startswith( '#' ) or len( row ) < 3:
        continue
      try:
        frame_to_image.setdefault( int( row[2] ), row[1] )
      except ValueError:
        continue

  reader = DetectedObjectSetInput.create( "viame_csv" )
  reader.set_configuration( reader.get_configuration() )
  reader.open( csv_file )

  sets = dict()
  frame_id = 0

  while True:
    output = reader.read_set()
    if output is None or output[0] is None:
      break

    image = frame_to_image.get( frame_id )
    if image is not None:
      sets[ image ] = output[0]

    frame_id = frame_id + 1

  return sets

def get_set_for_image( sets, image ):
  """Look up an image's detections, defaulting to an empty set."""
  dets = sets.get( image )
  if dets is None:
    dets = DetectedObjectSet()
  return dets

# ---------------- KWCOCO-SPECIFIC UTILITY FUNCTIONS -------------------

def convert_to_kwcoco( args, csv_file, image_list,
                       target_class=None, top_class=False,
                       retain_labels=False ):

  (fd, handle) = tempfile.mkstemp( prefix='viame-coco-',
                                   suffix='.json',
                                   text=True,
                                   dir=temp_dir )

  input_sets = read_sets_by_image( csv_file )

  coco_writer =  DetectedObjectSetOutput.create( "coco" )
  writer_conf = coco_writer.get_configuration()
  writer_conf.set_value( "global_categories", str( retain_labels ) )
  coco_writer.set_configuration( writer_conf )
  coco_writer.open( handle )

  for img in image_list:
    dets = get_set_for_image( input_sets, img )
    dets = filter_detections( args, dets, target_class, top_class=top_class )
    coco_writer.write_set( dets, img )
  coco_writer.complete()
  return fd, handle

def find_kwcoco_metrics_json( output_dir ):
  """Locate the metrics.json kwcoco eval wrote.

  Newer kwcoco versions nest their outputs in a subdirectory named after the
  evaluation settings instead of writing metrics.json at the top of the output
  directory, so search for it rather than assuming a fixed location.
  """
  direct = os.path.join( output_dir, 'metrics.json' )
  if os.path.exists( direct ):
    return direct

  matches = sorted( glob( os.path.join( output_dir, '**', 'metrics.json' ),
                          recursive=True ) )
  return matches[0] if matches else None

def generate_metrics_csv_kwcoco( output_dir, output_file ):
  input_file = find_kwcoco_metrics_json( output_dir )

  if not input_file:
    print( os.linesep + "Warning: kwcoco eval wrote no metrics.json under "
           + output_dir + ", skipping " + output_file )
    return None

  print( os.linesep + "write " + output_file  )

  fin = open( input_file )

  if not fin:
    return None

  output = []
  metrics = json.load( fin )
  for i in metrics:
    for j in metrics[i]:
       if j != "ovr_measures":
         continue
       for k in metrics[i][j]:
         row = []
         row.append( k )
         row.append( metrics[i][j][k]['ap'] )
         row.append( metrics[i][j][k]['auc'] )
         row.append( int( metrics[i][j][k]['realpos_total'] ) )

         if int( row[3] ):
           output.append( row )

  output.sort( key = lambda x: neg_safe_val( x[1] ) )
  output = [[ "#category", "ap", "auc", "samples" ]] + output

  with open( output_file, 'w', newline='' ) as fout:
    writer = csv.writer( fout )
    writer.writerows( output )

  return [ output[-1][1], output[-1][2], output[-1][3] ]

# ---------------- PRECISION-RECALL AND CONF MAT -----------------------

def get_prc_conf_cmd():
  if os.name == 'nt':
    return ['python.exe', '-m', 'kwcoco', 'eval' ]
  else:
    return ['kwcoco', 'eval' ]

def generate_det_prc_conf_directory( args, target_class=None ):
  is_multi_class = False if target_class else True

  if args.track_detections:
    aligned_files = compute_alignment( args.computed, args.truth, \
      args.input_ext, remove_postfix = '_tracks', skip_postfix = '_detections' )
  else:
    aligned_files = compute_alignment( args.computed, args.truth, \
      args.input_ext, remove_postfix = '_detections', skip_postfix = '_tracks' )

  (fd1, handle1) = tempfile.mkstemp( prefix='viame-coco-',
                                     suffix='.json',
                                     text=True,
                                     dir=temp_dir )
  (fd2, handle2) = tempfile.mkstemp( prefix='viame-coco-',
                                     suffix='.json',
                                     text=True,
                                     dir=temp_dir )

  truth_writer =  DetectedObjectSetOutput.create( "coco" )
  computed_writer =  DetectedObjectSetOutput.create( "coco" )

  writer_conf = truth_writer.get_configuration()
  writer_conf.set_value( "global_categories", str( is_multi_class ) )

  truth_writer.set_configuration( writer_conf )
  truth_writer.open( handle1 )

  computed_writer.set_configuration( writer_conf )
  computed_writer.open( handle2 )

  print( "Processing identified truth/computed matches" )

  for computed, truth in aligned_files.items():
    joint_images, mismatch, fc = get_file_list_from_viame_csvs( computed, truth )

    if mismatch:
      # Image names do not line up between the two files, so fall back to
      # pairing them positionally under a synthetic name
      truth_reader =  DetectedObjectSetInput.create( "viame_csv" )
      truth_reader.set_configuration( truth_reader.get_configuration() )
      truth_reader.open( truth )

      computed_reader =  DetectedObjectSetInput.create( "viame_csv" )
      computed_reader.set_configuration( computed_reader.get_configuration() )
      computed_reader.open( computed )

      syn_base_name = os.path.splitext( os.path.basename( computed ) )[0]
      for i in range( 0, fc+1 ):
        syn_file_name = syn_base_name + "-" + str( i ).zfill( 6 )
        truth_dets = truth_reader.read_set()
        computed_dets = computed_reader.read_set()
        if truth_dets is None or computed_dets is None:
          continue
        truth_dets = truth_dets[0]
        computed_dets = computed_dets[0]
        truth_dets = filter_detections( args, truth_dets,
          target_class=target_class )
        computed_dets = filter_detections( args, computed_dets,
          target_class=target_class, top_class=True )
        truth_writer.write_set( truth_dets, syn_file_name )
        computed_writer.write_set( computed_dets, syn_file_name )
    else:
      truth_sets = read_sets_by_image( truth )
      computed_sets = read_sets_by_image( computed )

      for i in joint_images:
        truth_dets = filter_detections( args,
          get_set_for_image( truth_sets, i ), target_class=target_class )
        computed_dets = filter_detections( args,
          get_set_for_image( computed_sets, i ),
          target_class=target_class, top_class=True )
        truth_writer.write_set( truth_dets, i )
        computed_writer.write_set( computed_dets, i )

  print( "Writing compiled detections to json" )

  truth_writer.complete()
  computed_writer.complete()

  print( "Running scoring scripts" )

  if target_class:
    output_dir = os.path.join( args.det_prc_conf, target_class )
  else:
    output_dir = args.det_prc_conf

  cmd = get_prc_conf_cmd() + [ '--true_dataset', handle1 ]
  cmd = cmd + [  '--pred_dataset', handle2 ]
  cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
  cmd = cmd + [  '--out_dpath', output_dir ]

  subprocess.call( cmd )

  output_csv_file = os.path.join( output_dir, "metrics.csv" )
  return generate_metrics_csv_kwcoco( output_dir, output_csv_file )

def generate_det_prc_conf_single( args, target_class=None ):

  if args.list:
    image_list = get_file_list_from_txt_list( args.list )
  else:
    image_list, _, _ = get_file_list_from_viame_csvs( args.computed, args.truth )

  if target_class:
    output_dir = os.path.join( args.det_prc_conf, target_class )
    retain_labels = False
  else:
    output_dir = args.det_prc_conf
    retain_labels = True

  _, filtered_computed_json = convert_to_kwcoco( args, args.computed,
    image_list, target_class, retain_labels=retain_labels )
  _, filtered_truth_json = convert_to_kwcoco( args, args.truth,
    image_list, target_class, True, retain_labels=retain_labels )

  cmd = get_prc_conf_cmd() + [ '--true_dataset', filtered_truth_json ]
  cmd = cmd + [  '--pred_dataset', filtered_computed_json ]
  cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
  cmd = cmd + [  '--out_dpath', output_dir ]
  subprocess.call( cmd )

  output_csv_file = os.path.join( output_dir, "metrics.csv" )
  return generate_metrics_csv_kwcoco( output_dir, output_csv_file )

def generate_det_prc_conf( args, classes ):

  if not classes:
    classes = [ None ]

  is_dir_input = os.path.isdir( args.computed )

  scores = dict()

  for target_class in classes:
    if is_dir_input:
      score = generate_det_prc_conf_directory( args, target_class )
    else:
      score = generate_det_prc_conf_single( args, target_class )

    # Absent when kwcoco eval writes no metrics.json for the class
    if score is not None:
      scores[ target_class ] = score

  print( os.linesep + "Conf matrix and PRC plot generation is complete" + os.linesep )

  if len( classes ) > 1 and scores:
    net_score_file = os.path.join( args.det_prc_conf, "class_metrics.csv" )
    create_net_csv( net_score_file, scores,
      "# class,ap,auc,samples" )

  if os.name == "nt":
    print( "On windows, ignore the following temp file error" + os.linesep )

# ---------------------- MOT TRACK STATISTICS --------------------------

def generate_trk_mot_stats_single( args, target_class=None ):

  import motmetrics as mm
  from collections import OrderedDict

  is_folder_input = os.path.isdir( args.computed )

  if os.path.isdir( args.computed ) != os.path.isdir( args.truth ):
    print_and_exit( "Inputs must be either both folders or both csvs" )

  input_computed = args.computed
  input_truth = args.truth

  if hierarchy or target_class:
    tmp_computed = os.path.join( temp_dir, "computed" )
    tmp_truth = os.path.join( temp_dir, "truth" )
    if is_folder_input:
      remake_dir( tmp_computed )
      remake_dir( tmp_truth )
    else:
      remove_if_exists( tmp_computed )
      remove_if_exists( tmp_truth )
    filter_viame_csv_auto( input_computed, tmp_computed, args.input_ext,
      target_class=target_class, top_only=args.top_class )
    filter_viame_csv_auto( input_truth, tmp_truth, args.input_ext,
      target_class=target_class, top_only=True )
    input_computed = tmp_computed
    input_truth = tmp_truth

  if target_class:
    output_file = os.path.join( args.trk_mot_stats, target_class + ".txt" )
  else:
    output_file = args.trk_mot_stats

  if is_folder_input:
    aligned_files = compute_alignment( input_computed, input_truth, \
      args.input_ext, remove_postfix = '_tracks', skip_postfix = '_detections' )
  else:
    aligned_files = { input_computed : input_truth }

  loglevel = getattr( logging, 'INFO', None )
  logging.basicConfig( level=loglevel,
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='%I:%M:%S' )

  use_class_id = not args.ignore_classes
  use_class_confidences = not args.aux_confidence

  thresholds = [ args.threshold ]

  if args.sweep_thresholds:
    thresholds = [ x / args.sweep_interval for x in range( 0, args.sweep_interval ) ]
  else:
    thresholds = [ args.threshold ]

  def compare_dataframes( gts, ts ):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
      if k in gts:
        logging.info( 'Comparing %s...', k )
        accs.append( mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5 ) )
        names.append( k )
      else:
        logging.warning( 'No ground truth for %s, skipping.', k )
    return accs, names

  max_mota = min_conf
  max_mota_thresh = 0.0
  max_idf1 = min_conf
  max_idf1_thresh = 0.0

  metrics = [
    "idf1",
    "mota",
    "motp",
    "idp",
    "idr",
    "recall",
    "precision",
    "num_unique_objects",
    "mostly_tracked",
    "partially_tracked",
    "mostly_lost",
    "num_false_positives",
    "num_misses",
    "num_switches",
    "num_fragmentations",
    "num_transfer",
    "num_ascend",
    "num_migrate",
  ]

  fout = open( output_file, 'w' )

  for threshold in thresholds:

    if target_class:
      log_with_spaces( "Loading Class: " + target_class + " at Threshold " + str( threshold ) )
    else:
      log_with_spaces( "Loading Data at Threshold " + str( threshold ) )

    cf = OrderedDict( [ ( os.path.splitext( Path( f.replace( "_tracks", "" ) ).parts[-1])[0], \
      mm.io.loadtxt( f, fmt='viame-csv', min_confidence=threshold, \
                     use_class_ids=use_class_id, \
                     use_class_confidence=use_class_confidences ) ) \
      for f in aligned_files ] )

    gt = OrderedDict( [ ( os.path.splitext( Path( aligned_files[f] ).parts[-1])[0], \
      mm.io.loadtxt( aligned_files[f], fmt='viame-csv', min_confidence=0, \
                     force_conf_to_one=True, use_class_ids=use_class_id, \
                     use_class_confidence=use_class_confidences ) ) \
      for f in aligned_files ] )

    # In the case of input files instead of folder, don't need to worry about alignment
    if not is_folder_input:
      gt = OrderedDict( [ ( list( cf.keys() )[0], gt[ list( gt.keys() )[0] ] ) ] )

    mh = mm.metrics.create()

    accs, names = compare_dataframes( gt, cf )

    log_and_write_with_spaces( fout, 'Running MOT Metrics at Threshold ' + str( threshold ) )

    summary = mh.compute_many( accs, names=names, metrics=metrics, generate_overall=True )

    log_and_write( fout, os.linesep + mm.io.render_summary( summary, formatters=mh.formatters, \
      namemap=mm.io.motchallenge_metric_names ) )

    mota = float( summary.loc["OVERALL"].at['mota'] )
    idf1 = float( summary.loc["OVERALL"].at['idf1'] )

    if mota > max_mota:
      max_mota = mota
      max_mota_thresh = threshold
    if idf1 > max_idf1:
      max_idf1 = idf1
      max_idf1_thresh = threshold

  if len( thresholds ) > 1:
    log_and_write( fout, os.linesep +
      'Top IDF1 value: ' + float_to_str_safe( max_idf1 ) +
      ' at threshold ' + float_to_str_safe( max_idf1_thresh ) + os.linesep +
      'Top MOTA value: ' + float_to_str_safe( max_mota ) +
      ' at threshold ' + float_to_str_safe( max_mota_thresh ) )

  fout.close()

  return [ max_idf1, max_idf1_thresh, max_mota, max_mota_thresh ]

def create_mot_filter_json( filename, scores, method ):
  filters = dict()
  if method == "min":
    for key, value in scores.items():
      filters[ key ] = min( value[1], value[3] )
  elif method == "avg" or method == "avg_minus_1p":
    adj = -0.01 if method == "avg_minus_1p" else 0.00
    for key, value in scores.items():
      score = 0.5 * ( value[1] + value[3] ) + adj
      filters[ key ] = max( score, 0.0 )
  elif method == "idf1":
    for key, value in scores.items():
      filters[ key ] = value[1]
  elif method == "mota":
    for key, value in scores.items():
      filters[ key ] = value[3]
  else:
    print( "Unknown filter method: " + method )
    return False

  if "default" not in scores:
    key_min = min( filters.keys(), key=( lambda k: filters[k] ) )
    min_filter = filters[ key_min ]
    if min_filter > 0:
      filters[ "default" ] = min_filter

  with open( filename, 'w' ) as fout:
    data = { "confidenceFilters" : filters }
    json.dump( data, fout, ensure_ascii=True, indent=4 )
  return True

def generate_trk_mot_stats( args, classes ):
  if classes:
    remake_dir( args.trk_mot_stats )
  else:
    classes = [ None ]

  top_scores = dict()

  for target in classes:
    top_scores[ target ] = generate_trk_mot_stats_single( args, target )

  if len( classes ) > 1:
    # Net score file contains top metrics and thresholds for each class
    net_score_file = os.path.join( args.trk_mot_stats, "class_metrics.csv" )
    create_net_csv( net_score_file, top_scores,
      "# class,idf1,idf1_thresh,mota,mota_thresh" )

    # Filter file contains optimal thresholds per class in DIVE format
    if args.sweep_thresholds and args.filter_estimator != "none":
      filter_file = os.path.join( args.trk_mot_stats, "dive.config.json" )
      create_mot_filter_json( filter_file, top_scores, args.filter_estimator )

# -------------------------- MAIN FUNCTION -----------------------------

if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = 'Evaluate Detections' )

  # Inputs
  parser.add_argument( '-computed', default=None,
    help='Input filename or folder for computed files.' )
  parser.add_argument( '-truth', default=None,
    help='Input filename or folder for groundtruth files.' )
  parser.add_argument( '-threshold', type=float, default=0.0,
    help='Optional input detection confidence threshold for statistics.' )
  parser.add_argument( '-labels', dest="labels", default=None,
    help='Optional input label synonym file to use during evaluation.' )
  parser.add_argument( '-list', default=None,
    help='Optional input image list file for downselecting scoring.' )
  parser.add_argument( '-input-ext', dest="input_ext", default=".csv",
    help='Optional input file extension, used if inputs are folders.' )
  parser.add_argument( '-input-format', dest="input_format", default="viame_csv",
    help='Optional input file format (e.g. viame_csv, coco, ...).' )

  # Core output type options
  parser.add_argument( '-det-prc-conf', dest="det_prc_conf", default=None,
    help='Folder for PRC curves, conf matrix, and related stats.' )
  parser.add_argument( '-trk-mot-stats', dest="trk_mot_stats", default=None,
    help='File or folder name for output MOT statistics (IDF1, MOTA, etc...).' )

  # Scoring settings
  parser.add_argument( "-iou-thresh", dest="iou_thresh", type=float, default=0.5,
    help="IOU threshold for detection and track scoring methods" )
  parser.add_argument( "--ignore-classes", dest="ignore_classes", action="store_true",
    help="Ignore classes in the file and score all detection types as the same" )
  parser.add_argument( "--top-class", dest="top_class", action="store_true",
    help="Only use the highest scoring class on each detection in scoring" )
  parser.add_argument( "--per-class", dest="per_class", action="store_true",
    help="Run scoring routines on the scores for each class independently" )
  parser.add_argument( "--sweep-thresholds", dest="sweep_thresholds", action="store_true",
    help="For operations where thresholds are used, run with multiple thresholds" )
  parser.add_argument( "--aux-confidence", dest="aux_confidence", action="store_true",
    help="Use the auxiliary confidence as opposed to type confidence in operations" )
  parser.add_argument( "-sweep-interval", dest="sweep_interval", type=int, default=100,
    help="Number of different thresholds to use when sweeping potential values" )
  parser.add_argument( "--track-detections", dest="track_detections", action="store_true",
    help="For VIAME-generated folders with both track and detection files, use the "
    "detections stored within the track files for scoring instead of the detection "
    "files. This is currently just for the PRC/Conf Mat generation scripts." )
  parser.add_argument( "-filter-estimator", dest="filter_estimator", default="min",
    help="Method to use for generating output confidence filter estimate for use "
    "within the DIVE interface. Can be: none, min, avg, avg_minus_1p, idf1, mota." )

  parser.add_argument( '-defaultlabel', dest="default_label", default='',
    help='if ignoring labels an optional class to display' )

  args = parser.parse_args()

  if not args.computed or not args.truth:
    print_and_exit( "Error: both computed and truth files must be specified" )

  if not args.det_prc_conf and not args.trk_mot_stats:
    print_and_exit( "Error: either 'trk-mot-stats' or 'det-prc-conf' must be "
                    "specified. For ROC curves, KWANT-style track statistics, "
                    "or HOTA, use the viame_score_results tool." )

  from kwiver.vital.modules import load_known_modules
  load_known_modules()

  # Data formatting and checking based on select options  
  classes = []

  if args.labels:
    if not load_hierarchy( args.labels ):
      print_and_exit( "Unable to load labels from file: " + args.labels )
    if not hierarchy.all_class_names():
      print_and_exit( "Label file is empty" )

  set_default_label( args.default_label )

  if args.per_class:
    if args.labels:
      classes = hierarchy.all_class_names()
    elif args.input_format != "viame_csv":
      print_and_exit( "--per-class option only supported for viame_csv" )
    elif os.path.exists( args.truth ):
      classes = list_classes_viame_csv( args.truth, ext=args.input_ext )
    if len( classes ) == 0:
      print_and_exit( "--per-class option enabled but no classes present" )

  # Generate specified outputs
  if args.det_prc_conf:
    generate_det_prc_conf( args, classes )

  if args.trk_mot_stats:
    generate_trk_mot_stats( args, classes )
