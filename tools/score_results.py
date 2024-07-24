#!/usr/bin/python

import os
import sys
import re
import numpy as np
import argparse
import atexit
import tempfile
import subprocess
import shutil
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

temp_dir = tempfile.mkdtemp(prefix='score-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

linestyles = ['-', '--', '-.', ':']
linecolors = ['#25233d', '#161891', '#316f6a', '#662e43']

hierarchy=None
default_label="fish"
filtered_input=False

def get_kwant_cmd():
  if os.name == 'nt':
    return ['score_tracks.exe','--hadwav']
  else:
    return ['score_tracks','--hadwav']

def get_prc_conf_cmd():
  if os.name == 'nt':
    return ['python.exe', '-m', 'kwcoco', 'eval' ]
  else:
    return ['kwcoco', 'eval' ]

def get_roc_cmd():
  if os.name == 'nt':
    return ['score_events.exe']
  else:
    return ['score_events']

def print_and_exit( msg, code=0 ):
  print( msg )
  sys.exit( code )

def log_with_spaces( msg ):
  logging.info( '' )
  logging.info( msg )
  logging.info( '' )

def format_class_fn( fn ):
  return fn.replace( "/", "-" )

def load_roc( fn ):
  x_fa = np.array( [] )
  y_pd = np.array( [] )

  with open(fn) as f:
    while (1):
      raw_line = f.readline()
      if not raw_line:
        break
      fields = raw_line.split()
      x_fa = np.append( x_fa, float( fields[47] ) )
      y_pd = np.append( y_pd, float( fields[7] ) )
  return ( x_fa, y_pd )

def list_classes_viame_csv( filename ):
  unique_ids = set()
  with open( filename ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 10:
        continue
      if len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      idx = 9
      while idx < len( lis ):
        if lis[idx][0] == '(':
          break
        unique_ids.add( lis[idx] )
        idx = idx + 2
  return list( unique_ids )

def load_hierarchy( filename ):
  global hierarchy
  hierarchy = None
  if filename:
    hierarchy = CategoryHierarchy()
    if not hierarchy.load_from_file( filename ):
      print( "Unable to parse classes file: " + filename )
      sys.exit( 0 )
    return True
  else:
    return False

def set_default_label( user_input=None ):
  global default_label
  if user_input:
    default_label = user_input
  elif hierarchy and len( hierarchy.all_class_names() ) == 1:
    default_label = hierarchy.all_class_names()[0]

def list_files_rec_w_ext( folder, ext ):
  result = [ y for x in os.walk( folder )
               for y in glob( os.path.join( x[0], '*' + ext ) ) ]
  return result

def compute_alignment( computed_dir, truth_dir, ext = '.csv',
                       remove_postfix = '_detections',
                       skip_postfix = '_tracks' ):

  out = dict()

  computed_files = list_files_rec_w_ext( computed_dir, ext )
  truth_files = list_files_rec_w_ext( truth_dir, ext )

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
      print( "Could not find corresponding truth for: " + comp_base )
      sys.exit( 0 )
  return out

def filter_viame_csv_by_class( filename, cls, threshold=0.0 ):
  (fd, handle) = tempfile.mkstemp( prefix='viame-score-',
                                   suffix='.csv',
                                   text=True,
                                   dir=temp_dir )

  fout = os.fdopen( fd, 'w' )

  with open( filename ) as f:
    for line in f:
      lis = line.strip().split(',')
      if len( lis ) < 10:
        continue
      if len( lis[0] ) > 0 and lis[0][0] == '#':
        continue
      idx = 9
      use_detection = False
      confidence = 0.0
      object_label = ""
      while idx < len( lis ):
        if lis[idx][0] == '(':
          break
        if lis[idx] == cls and \
           ( args.threshold == 0.0 or float( lis[idx+1] ) >= float( args.threshold ) ):
          use_detection = True
          confidence = float( lis[idx+1] )
          object_label = lis[idx]
          break
        idx = idx + 2

      if use_detection:
        fout.write( lis[0] + ',' + lis[1] + ',' + lis[2] + ',' + lis[3] + ',' )
        fout.write( lis[4] + ',' + lis[5] + ',' + lis[6] + ',' + str( confidence ) + ',' )
        if len( object_label ) > 0:
          fout.write( lis[8] + ',' + object_label + ',' + str( confidence ) + '\n' )
        else:
          fout.write( lis[8] + '\n' )
  fout.close()
  return fd, handle

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

def read_list_from_file_list( filename ):
  out = []
  with open( filename ) as f:
    for line in f:
      line = line.rstrip()
      if len( line ) > 0:
        out.append( line )
  return out

def filter_detections( args, dets ):
  output = DetectedObjectSet()
  for i, item in enumerate( dets ):
    if item.type is None:
      if args.ignore_classes:
        # Ignore classes option with no real classes
        score = item.confidence
        item.type = DetectedObjectType( default_label, score )
      else:
        continue
    elif args.ignore_classes:
      # Ignore classes option, relabel top class to default
      cls = item.type.get_most_likely_class()
      score = item.type.score( cls )
      item.type = DetectedObjectType( default_label, score )
    elif args.top_class:
      # Top class option, only use highest scoring class
      cls = item.type.get_most_likely_class()
      score = item.type.score( cls )
      item.type = DetectedObjectType( cls, score )

    # Applies threshold parameter
    all_classes = item.type.class_names( args.threshold )

    # Apply hierarchy if present, also remakes type after threshold
    output_type = DetectedObjectType()
    for cls in all_classes:
      score = item.type.score( cls )
      if hierarchy:
        if not hierarchy.has_class_name( cls ):
          continue
        else:
          cls = hierarchy.get_class_name( cls )
      if not output_type.has_class_name( cls ) or output_type.score( cls ) < score:
        output_type.set_score( cls, score )

    # If no classes left after filters, skip detection
    if len( output_type ) == 0:
      continue
    else:
      item.type = output_type
      output.add( item )
  return output

def standardize_single( args, input_file, output_file ):

  input_reader =  DetectedObjectSetInput.create( args.input_format )
  reader_conf = input_reader.get_configuration()
  input_reader.set_configuration( reader_conf )
  input_reader.open( input_file )

  csv_reader =  DetectedObjectSetOutput.create( "viame_csv" )
  writer_conf = csv_reader.get_configuration()
  csv_reader.set_configuration( writer_conf )
  csv_reader.open( handle )

  for img in image_list:
    dets = input_reader.read_set_by_path( img )
    filter_detections( args, dets )
    csv_reader.write_set( dets, img )

  csv_reader.complete()

def standardize_input( args, folder, output ):
  return False


def convert_to_kwcoco( csv_file, image_list, retain_labels=False ):
  (fd, handle) = tempfile.mkstemp( prefix='viame-coco-',
                                   suffix='.json',
                                   text=True,
                                   dir=temp_dir )

  csv_reader =  DetectedObjectSetInput.create( "viame_csv" )
  reader_conf = csv_reader.get_configuration()
  csv_reader.set_configuration( reader_conf )
  csv_reader.open( csv_file )

  coco_writer =  DetectedObjectSetOutput.create( "coco" )
  writer_conf = coco_writer.get_configuration()
  writer_conf.set_value( "global_categories", str( retain_labels ) )
  coco_writer.set_configuration( writer_conf )
  coco_writer.open( handle )

  for img in image_list:
    truth = csv_reader.read_set_by_path( img )
    if not filtered_input:
      truth = filter_detections( args, truth )
    coco_writer.write_set( truth, img )
  coco_writer.complete()
  return fd, handle

def generate_det_prc_conf_directory( args, classes ):
  aligned_truth = compute_alignment( args.computed, args.truth )

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
  writer_conf.set_value( "global_categories", "true" )

  truth_writer.set_configuration( writer_conf )
  truth_writer.open( handle1 )

  computed_writer.set_configuration( writer_conf )
  computed_writer.open( handle2 )

  print( "Processing identified truth/computed matches" )

  for computed, truth in aligned_truth.items():
    truth_reader =  DetectedObjectSetInput.create( "viame_csv" )
    reader_conf = truth_reader.get_configuration()
    truth_reader.set_configuration( reader_conf )
    truth_reader.open( truth )

    computed_reader =  DetectedObjectSetInput.create( "viame_csv" )
    reader_conf = computed_reader.get_configuration()
    computed_reader.set_configuration( reader_conf )
    computed_reader.open( computed )

    joint_images, mismatch, fc = get_file_list_from_viame_csvs( computed, truth )

    if mismatch:
      syn_base_name = os.path.splitext( os.path.basename( computed ) )[0]
      for i in range( 0, fc+1 ):
        syn_file_name = syn_base_name + "-" + str( i ).zfill( 6 )
        truth_dets = truth_reader.read_set()
        computed_dets = computed_reader.read_set()
        if truth_dets is None or computed_dets is None:
          continue
        truth_dets = truth_dets[0]
        computed_dets = computed_dets[0]
        if not filtered_input:
          truth_dets = filter_detections( args, truth_dets )
          computed_dets = filter_detections( args, computed_dets )
        truth_writer.write_set( truth_dets, syn_file_name )
        computed_writer.write_set( computed_dets, syn_file_name )
    else:
      for i in joint_images:
        truth_dets = truth_reader.read_set_by_path( i )
        computed_dets = computed_reader.read_set_by_path( i )
        if not filtered_input:
          truth_dets = filter_detections( args, truth_dets )
          computed_dets = filter_detections( args, computed_dets )
        truth_writer.write_set( truth_dets, i )
        computed_writer.write_set( computed_dets, i )

  print( "Writing compiled detections to json" )

  truth_writer.complete()
  computed_writer.complete()

  print( "Running scoring scripts" )

  cmd = get_prc_conf_cmd() + [ '--true_dataset', handle1 ]
  cmd = cmd + [  '--pred_dataset', handle2 ]
  cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
  cmd = cmd + [  '--out_dpath', args.det_prc_conf ]
  subprocess.call( cmd )
  
def generate_det_prc_conf_single( args, classes ):

  if args.list:
    image_list = read_list_from_file_list( args.list )
  else:
    image_list, _, _ = get_file_list_from_viame_csvs( args.computed, args.truth )

  for cls in classes:
    _, filtered_computed_csv = filter_viame_csv_by_class( args.computed, cls, args.threshold )
    _, filtered_truth_csv = filter_viame_csv_by_class( args.truth, cls, args.threshold )
    _, filtered_computed_json = convert_to_kwcoco( filtered_computed_csv, image_list )
    _, filtered_truth_json = convert_to_kwcoco( filtered_truth_csv, image_list )

    cmd = get_prc_conf_cmd() + [ '--true_dataset', filtered_truth_json ]
    cmd = cmd + [  '--pred_dataset', filtered_computed_json ]
    cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
    cmd = cmd + [  '--out_dpath', "conf-" + format_class_fn( cls ) ]
    subprocess.call( cmd )

  if not classes:
    _, filtered_computed_json = convert_to_kwcoco( args.computed, image_list, True )
    _, filtered_truth_json = convert_to_kwcoco( args.truth, image_list, True )

    cmd = get_prc_conf_cmd() + [ '--true_dataset', filtered_truth_json ]
    cmd = cmd + [  '--pred_dataset', filtered_computed_json ]
    cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
    cmd = cmd + [  '--out_dpath', "conf-joint-output" ]
    subprocess.call( cmd )

def generate_det_prc_conf( args, classes ):

  if os.path.isdir( args.computed ):
    generate_det_prc_conf_directory( args, classes )
  else:
    generate_det_prc_conf_single( args, classes )

  print( "\nConf matrix and PRC plot generation is complete\n" )

  if os.name == "nt":
    print( "On windows, ignore the following temp file error\n" )

def generate_trk_kwant_stats( args, classes ):

  # Generate roc files
  base, ext = os.path.splitext( args.trk_kwant_stats )
  input_format = args.input_format if args.input_format != "viame_csv" else "noaa-csv"

  base_cmd = get_kwant_cmd()
  base_cmd += [ '--computed-format', input_format, '--truth-format', input_format ]
  base_cmd += [ '--fn2ts' ]

  for cls in classes:
    stat_file = base + "." + format_class_fn( cls ) + ext
    _, filtered_computed = filter_viame_csv_by_class( args.computed, cls, args.threshold )
    _, filtered_truth = filter_viame_csv_by_class( args.truth, cls, args.threshold )
    cmd = base_cmd + [ '--computed-tracks', filtered_computed, '--truth-tracks', filtered_truth ]
    with open( stat_file, 'w' ) as fout:
      if not args.use_cache:
        subprocess.call( cmd, stdout=fout, stderr=fout )

  if len( classes ) != 1:
    cmd = base_cmd + [ '--computed-tracks', args.computed, '--truth-tracks', args.truth ]
    with open( args.trk_kwant_stats, 'w' ) as fout:
      if not args.use_cache:
        subprocess.call( cmd, stdout=fout, stderr=fout )

def generate_det_rocs( args, classes ):

  # Generate roc files
  base, ext = os.path.splitext( args.det_roc )

  roc_files = []

  input_format = args.input_format if args.input_format != "viame_csv" else "noaa-csv"

  base_cmd = get_roc_cmd()
  base_cmd += [ '--computed-format', input_format, '--truth-format', input_format ]
  base_cmd += [ '--fn2ts', '--gt-prefiltered', '--ct-prefiltered' ]

  if ',' in args.computed:
    input_files = [ i.lstrip() for i in args.computed.split(',') ]
  else:
    input_files = [ args.computed ]

  for filename in input_files:
    for cls in classes:
      roc_file = base + "." + format_class_fn( cls ) + ".txt"
      if len( input_files ) > 1:
        roc_file = filename + '.' + roc_file
      if not args.use_cache:
        _, filtered_computed = filter_viame_csv_by_class( filename, cls )
        _, filtered_truth = filter_viame_csv_by_class( args.truth, cls )
        cmd = base_cmd + [ '--roc-dump', roc_file ]
        cmd += [ '--computed-tracks', filtered_computed, '--truth-tracks', filtered_truth ]
        subprocess.call( cmd )
      roc_files.append( roc_file )

    if len( classes ) != 1:
      net_roc_file = base + ".txt"
      if len( input_files ) > 1:
        net_roc_file = filename + '.' + net_roc_file
      if not args.use_cache:
        cmd = base_cmd + [ '--roc-dump', net_roc_file ]
        cmd += [ '--computed-tracks', filename, '--truth-tracks', args.truth ]
        subprocess.call( cmd )
      roc_files.append( net_roc_file )

  # Generate plot
  fig = plt.figure()

  xscale_arg = 'log' if args.logx else 'linear'

  rocplot = plt.subplot( 1, 1, 1, xscale=xscale_arg )
  rocplot.set_title( args.title ) if args.title else None

  plt.xlabel( args.xlabel )
  plt.ylabel( args.ylabel )

  plt.xticks()
  plt.yticks()

  user_titles = args.key.split(',') if args.key else None
  i = 0
  for i, fn in enumerate( roc_files ):
    (x,y) = load_roc( fn )
    display_label = ""
    if user_titles and i < len( user_titles ):
      display_label = user_titles[i]
    else:
      display_label = fn.replace( ".txt", "" )
      display_label = display_label.replace( ".csv", "" )
      display_label = display_label.replace( base + ".", "" )
      display_label = display_label.replace( base, "" )
      if len( display_label ) == 0:
        display_label = "aggregate"
      if display_label.endswith( "." ):
        display_label = display_label[:-1]
    sys.stderr.write( "Info: %d: loading %s as '%s'...\n" % ( i, fn, display_label ) )
    stl = linestyles[ i % len(linestyles) ]
    if i < len( linecolors ):
      cl = linecolors[ i ]
    else:
      cl = np.random.rand( 3 )
    if len( display_label ) > 0:
      rocplot.plot( x, y, linestyle=stl, color=cl, linewidth=args.lw, label=display_label )
    else:
      rocplot.plot( x, y, linestyle=stl, color=cl, linewidth=args.lw )
    rocplot.set_xlim( xmin=0 )
    i += 1

  if args.autoscale:
    rocplot.autoscale()
  else:
    tmp = args.rangey.split( ':' )
    if len( tmp ) != 2:
      sys.stderr.write( 'Error: rangey option must be two floats ' )
      sys.stderr.write( 'separated by a colon, e.g. 0.2:0.7\n' )
      sys.exit( 1 )
    ( ymin, ymax ) = ( float( tmp[0] ), float( tmp[1]) )
    rocplot.set_ylim( ymin, ymax )

    if args.rangex:
      tmp = args.rangex.split( ':' )
      if len( tmp ) != 2:
        sys.stderr.write( 'Error: rangex option must be two floats ' )
        sys.stderr.write( 'separated by a colon, e.g. 0.2:0.7\n' )
        sys.exit( 1 )
      ( xmin, xmax ) = ( float(tmp[0]), float(tmp[1]) )
      rocplot.set_xlim( xmin,xmax )

  if not args.nokey:
    legend_loc = args.keyloc
    if legend_loc == "auto":
      if len( classes ) < 15:
        plt.legend( loc="best" )
      else:
        colcount = int( 1 + len( classes ) / 15 )
        plt.legend( loc='center right', bbox_to_anchor = ( 1.75, 0.6 ), ncol = colcount )
    else:
      plt.legend( loc=args.keyloc )

  plt.savefig( args.det_roc, bbox_inches='tight' )

def generate_trk_mot_stats( args, classes ):

  import motmetrics as mm

  from collections import OrderedDict

  if os.path.isdir( args.computed ):
    aligned_files = compute_alignment( args.computed, args.truth, \
      remove_postfix = '_tracks', skip_postfix = '_detections' )
  else:
    aligned_files = { args.computed : args.truth }

  loglevel = getattr( logging, 'INFO', None )
  logging.basicConfig( level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S' )

  use_class_id = not args.ignore_classes
  use_class_confidences = not args.aux_confidence

  thresholds = [ args.threshold ]

  if args.sweep_thresholds:
    thresholds = [ x / 100 for x in range( 0, 101 ) ]
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

  max_mota = -9999.9999
  max_mota_thresh = 0.0
  max_idf1 = -9999.9999
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

  for threshold in thresholds:

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

    mh = mm.metrics.create()

    accs, names = compare_dataframes( gt, cf )

    log_with_spaces( 'Running MOT Metrics at Threshold ' + str( threshold ) )

    summary = mh.compute_many( accs, names=names, metrics=metrics, generate_overall=True )

    logging.info( mm.io.render_summary( summary, formatters=mh.formatters, \
      namemap=mm.io.motchallenge_metric_names ) )

    mota = float( summary.loc["OVERALL"].at['mota'] )
    idf1 = float( summary.loc["OVERALL"].at['idf1'] )
    hota = 0

    if mota > max_mota:
      max_mota = mota
      max_mota_thresh = threshold
    if idf1 > max_idf1:
      max_idf1 = idf1
      max_idf1_thresh = threshold

  if len( thresholds ) > 1:
    logging.info( '' )
    logging.info( 'Top IDF1 value: ' + str( max_idf1 ) + ' at threshold ' + str( max_idf1_thresh ) )
    logging.info( 'Top MOTA value: ' + str( max_mota ) + ' at threshold ' + str( max_mota_thresh ) )

def generate_trk_hota_stats( args, classes ):

  print_and_exit( "Implementation for HOTA not yet finished" )

  import motmetrics as mm
  import logging

  from collections import OrderedDict

  if os.path.isdir( args.computed ):
    aligned_files = compute_alignment( args.computed, args.truth, \
      remove_postfix = '_tracks', skip_postfix = '_detections' )
  else:
    aligned_files = { args.computed : args.truth }

  loglevel = getattr( logging, 'INFO', None )
  logging.basicConfig( level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S' )

  use_class_id = not args.ignore_classes
  use_class_confidences = not args.aux_confidence

  thresholds = [ args.threshold ]

  if args.sweep_thresholds:
    thresholds = [ x / 100 for x in range( 99, 101 ) ]
  else:
    thresholds = [ args.threshold ]

  max_hota = -9999.9999
  max_hota_thresh = 0.0

  metrics = [
    "hota"
  ]

  for threshold in thresholds:

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

    mh = mm.metrics.create()

    accs, names = compare_dataframes( gt, cf )

    log_with_spaces( 'Running MOT Metrics at Threshold ' + str( threshold ) )

    summary = mh.compute_many( accs, names=names, metrics=metrics, generate_overall=True )

    logging.info( mm.io.render_summary( summary, formatters=mh.formatters, \
      namemap=mm.io.motchallenge_metric_names ) )

    hota = float( summary.loc["OVERALL"].at['idf1'] )

    if hota > max_hota:
      max_hota = hota
      max_hota_thresh = threshold

  if len( thresholds ) > 1:
    logging.info( '' )
    logging.info( 'Top HOTA value: ' + str( max_hota ) + ' at threshold ' + str( max_hota_thresh ) )

if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = 'Evaluate Detections' )

  # Inputs
  parser.add_argument( '-computed', default=None,
    help='Input filename or folder for computed files.' )
  parser.add_argument( '-truth', default=None,
    help='Input filename or folder for groundtruth files.' )
  parser.add_argument( '-threshold', type=float, default=0.001,
    help='Input detection confidence threshold for statistics.' )
  parser.add_argument( '-labels', dest="labels", default=None,
    help='Input label synonym file to use during evaluation.' )
  parser.add_argument( '-list', default=None,
    help='Input filename for optional image list file.' )
  parser.add_argument( '-input-format', dest="input_format", default="viame_csv",
    help='Input file format.' )

  # Output options
  parser.add_argument( '-det-prc-conf', dest="det_prc_conf", default=None,
    help='Folder for PRC curves, conf matrix, and related stats.' )
  parser.add_argument( '-det-roc', dest="det_roc", default=None,
    help='Filename for output ROC curves.' )
  parser.add_argument( '-trk-kwant-stats', dest="trk_kwant_stats", default=None,
    help='Filename for output KWANT track statistics.' )
  parser.add_argument( '-trk-mot-stats', dest="trk_mot_stats", default=None,
    help='Filename for output MOT track statistics (IDF1, MOTA, etc...).' )
  parser.add_argument( '-trk-hota-stats', dest="trk_hota_stats", default=None,
    help='Filename for output HOTA track statistics.' )

  # Scoring settings
  parser.add_argument( "-iou-thresh", dest="iou_thresh", default=0.5,
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

  # Plot settings
  parser.add_argument( '-rangey', metavar='rangey', nargs='?', default='0:1',
    help='ymin:ymax (quote w/ spc for negative, i.e. " -0.1:5")' )
  parser.add_argument( '-rangex', metavar='rangex', nargs='?',
    help='xmin:xmax (quote w/ spc for negative, i.e. " -0.1:5")' )
  parser.add_argument( '-autoscale', action='store_true',
    help='Ignore -rangex -rangey and autoscale both axes of the plot.' )
  parser.add_argument( '-logx', action='store_true',
    help='Use logscale for x' )
  parser.add_argument( '-xlabel', nargs='?', default='Detection FA count',
    help='title for x axis' )
  parser.add_argument( '-ylabel', nargs='?', default='Detection PD',
    help='title for y axis' )
  parser.add_argument( '-defaultlabel', dest="default_label", default='',
    help='if ignoring labels an optional class to display' )
  parser.add_argument( '-title', nargs='?',
    help='title for plot' )
  parser.add_argument( '-lw', nargs='?', type=float, default=2,
    help='line width' )
  parser.add_argument( '-key', nargs='?', default=None,
    help='comma-separated set of strings labeling each line in order read' )
  parser.add_argument( '-keyloc', nargs='?', default='auto',
    help='Key location ("upper left", "lower right", etc; help for list)' )
  parser.add_argument( '--nokey', action='store_true',
    help='Set to suppress plot legend' )
  parser.add_argument( '--use-cache', dest="use_cache", action='store_true',
    help='Do not recompute roc or conf intermediate files' )

  args = parser.parse_args()

  if not args.computed or not args.truth:
    print_and_exit( "Error: both computed and truth files must be specified" )

  if not args.det_roc and not args.det_prc_conf and \
     not args.trk_kwant_stats and not args.trk_mot_stats:
    print_and_exit( "Error: either 'trk-kwant-stats', 'trk-mot-stats', " \
                    "'det-roc', or 'det-prc-conf' must be specified" )

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

  #if args.input_format != "viame_csv" or args.labels \
  #   or args.ignore_classes or args.top_class:
  #
  #  args.truth = standardize_input( args, args.truth, "truth" )
  #  args.computed = standardize_input( args, args.computed, "computed" )
  #  args.input_format = "viame_csv"

  if args.per_class:
    if args.labels:
      classes = hierarchy.all_class_names()
    elif args.input_format != "viame_csv":
      print_and_exit( "--per-class option only supported for viame_csv" )
    elif os.path.exists( args.truth ) and not os.path.isdir( args.truth ):
      classes = list_classes_viame_csv( args.truth )

  # Generate specified outputs
  if args.det_roc:
    generate_det_rocs( args, classes )

  if args.det_prc_conf:
    generate_det_prc_conf( args, classes )

  if args.trk_kwant_stats:
    generate_trk_kwant_stats( args, classes )

  if args.trk_mot_stats:
    generate_trk_mot_stats( args, classes )

  if args.trk_hota_stats:
    generate_trk_hota_stats( args, classes )
