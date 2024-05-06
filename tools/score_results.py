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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from glob import glob

from kwiver.vital.algo import (
    DetectedObjectSetInput,
    DetectedObjectSetOutput
)

from kwiver.vital.types import (
    Image, ImageContainer,
    BoundingBoxD, CategoryHierarchy,
    DetectedObjectSet, DetectedObject, DetectedObjectType
)

temp_dir = tempfile.mkdtemp(prefix='score-tmp')
atexit.register(lambda: shutil.rmtree(temp_dir))

linestyles = ['-', '--', '-.', ':']
linecolors = ['#25233d', '#161891', '#316f6a', '#662e43']

def get_stat_cmd():
  if os.name == 'nt':
    return ['score_tracks.exe','--hadwav']
  else:
    return ['score_tracks','--hadwav']

def get_conf_cmd():
  if os.name == 'nt':
    return ['python.exe', '-m', 'kwcoco', 'eval' ]
  else:
    return ['kwcoco', 'eval' ]

def get_roc_cmd():
  if os.name == 'nt':
    return ['score_events.exe']
  else:
    return ['score_events']

def format_cat_fn( fn ):
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

def list_categories_viame_csv( filename ):
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

def filter_by_category( filename, category, threshold=0.0 ):
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
        if lis[idx] == category and \
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

def filter_dets( dets, categories=None ):
  #if categories is None:
  #  return dets
  output = DetectedObjectSet()
  for i, item in enumerate( dets ):
    if item.type is None:
      continue
    #class_lbl = item.type.get_most_likely_class()
    #class_lbl = categories.get_class_name( class_lbl )
    item.type = DetectedObjectType( "fish", 1.0 )
    output.add( item )
  return output

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
    coco_writer.write_set( truth, img )
  coco_writer.complete()
  return fd, handle

def generate_det_conf_directory( args, categories ):
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
        filter_dets( truth_dets )
        truth_writer.write_set( truth_dets, syn_file_name )
        computed_writer.write_set( computed_dets, syn_file_name )
    else:
      for i in joint_images:
        truth_dets = truth_reader.read_set_by_path( i )
        computed_dets = computed_reader.read_set_by_path( i )
        filter_dets( truth_dets )
        truth_writer.write_set( truth_dets, i )
        computed_writer.write_set( computed_dets, i )

  print( "Writing compiled detections to json" )

  truth_writer.complete()
  computed_writer.complete()

  print( "Running scoring scripts" )

  cmd = get_conf_cmd() + [ '--true_dataset', handle1 ]
  cmd = cmd + [  '--pred_dataset', handle2 ]
  cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
  cmd = cmd + [  '--out_dpath', "conf-joint-output" ]
  subprocess.call( cmd )
  
def generate_det_conf_single( args, categories ):

  if args.list:
    image_list = read_list_from_file_list( args.list )
  else:
    image_list, _, _ = get_file_list_from_viame_csvs( args.computed, args.truth )

  for cat in categories:
    _, filtered_computed_csv = filter_by_category( args.computed, cat, args.threshold )
    _, filtered_truth_csv = filter_by_category( args.truth, cat, args.threshold )
    _, filtered_computed_json = convert_to_kwcoco( filtered_computed_csv, image_list )
    _, filtered_truth_json = convert_to_kwcoco( filtered_truth_csv, image_list )

    cmd = get_conf_cmd() + [ '--true_dataset', filtered_truth_json ]
    cmd = cmd + [  '--pred_dataset', filtered_computed_json ]
    cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
    cmd = cmd + [  '--out_dpath', "conf-" + format_cat_fn( cat ) ]
    subprocess.call( cmd )

  if not categories:
    _, filtered_computed_json = convert_to_kwcoco( args.computed, image_list, True )
    _, filtered_truth_json = convert_to_kwcoco( args.truth, image_list, True )

    cmd = get_conf_cmd() + [ '--true_dataset', filtered_truth_json ]
    cmd = cmd + [  '--pred_dataset', filtered_computed_json ]
    cmd = cmd + [  '--iou_thresh', str( args.iou_thresh ) ]
    cmd = cmd + [  '--out_dpath', "conf-joint-output" ]
    subprocess.call( cmd )

def generate_det_conf( args, categories ):

  from kwiver.vital.modules import load_known_modules
  load_known_modules()

  if os.path.isdir( args.computed ):
    generate_det_conf_directory( args, categories )
  else:
    generate_det_conf_single( args, categories )
  
  print( "\nConf matrix and PRC plot generation is complete\n" )

  if os.name == "nt":
    print( "On windows, ignore the following temp file error\n" )

def generate_trk_simp_stats( args, categories ):

  # Generate roc files
  base, ext = os.path.splitext( args.trk_simple )

  base_cmd = get_stat_cmd()
  base_cmd += [ '--computed-format', args.input_format, '--truth-format', args.input_format ]
  base_cmd += [ '--fn2ts' ]

  for cat in categories:
    stat_file = base + "." + format_cat_fn( cat ) + ext
    _, filtered_computed = filter_by_category( args.computed, cat, args.threshold )
    _, filtered_truth = filter_by_category( args.truth, cat, args.threshold )
    cmd = base_cmd + [ '--computed-tracks', filtered_computed, '--truth-tracks', filtered_truth ]
    with open( stat_file, 'w' ) as fout:
      if not args.use_cache:
        subprocess.call( cmd, stdout=fout, stderr=fout )

  if len( categories ) != 1:
    cmd = base_cmd + [ '--computed-tracks', args.computed, '--truth-tracks', args.truth ]
    with open( args.trk_simple, 'w' ) as fout:
      if not args.use_cache:
        subprocess.call( cmd, stdout=fout, stderr=fout )

def generate_det_rocs( args, categories ):

  # Generate roc files
  base, ext = os.path.splitext( args.det_roc )

  roc_files = []

  base_cmd = get_roc_cmd()
  base_cmd += [ '--computed-format', args.input_format, '--truth-format', args.input_format ]
  base_cmd += [ '--fn2ts', '--gt-prefiltered', '--ct-prefiltered' ]

  if ',' in args.computed:
    input_files = [ i.lstrip() for i in args.computed.split(',') ]
  else:
    input_files = [ args.computed ]

  for filename in input_files:
    for cat in categories:
      roc_file = base + "." + format_cat_fn( cat ) + ".txt"
      if len( input_files ) > 1:
        roc_file = filename + '.' + roc_file
      if not args.use_cache:
        _, filtered_computed = filter_by_category( filename, cat )
        _, filtered_truth = filter_by_category( args.truth, cat )
        cmd = base_cmd + [ '--roc-dump', roc_file ]
        cmd += [ '--computed-tracks', filtered_computed, '--truth-tracks', filtered_truth ]
        subprocess.call( cmd )
      roc_files.append( roc_file )

    if len( categories ) != 1:
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
      if len( categories ) < 15:
        plt.legend( loc="best" )
      else:
        colcount = int( 1 + len( categories ) / 15 )
        plt.legend( loc='center right', bbox_to_anchor = ( 1.75, 0.6 ), ncol = colcount )
    else:
      plt.legend( loc=args.keyloc )

  plt.savefig( args.det_roc, bbox_inches='tight' )


if __name__ == "__main__":
  parser = argparse.ArgumentParser( description = 'Generate detection scores and ROCs' )

  # Inputs
  parser.add_argument( '-computed', default=None,
             help='Input filename or folder for computed files.' )
  parser.add_argument( '-truth', default=None,
             help='Input filename or folder for groundtruth files.' )
  parser.add_argument( '-list', default=None,
             help='Input filename for optional image list file.' )
  parser.add_argument( '-threshold', type=float, default=0.05,
             help='Input threshold for statistics.' )
  parser.add_argument( '-labels', dest="input_labels", default=None,
             help='Input label synonym file.' )
  parser.add_argument( '-input-format', dest="input_format", default="viame-csv",
             help='Input file format.' )

  # Output options
  parser.add_argument( '-det-conf', dest="det_conf", default=None,
             help='Folder for conf matrix, prc curves, and related stats.' )
  parser.add_argument( '-det-roc', dest="det_roc", default=None,
             help='Filename for output roc curves.' )
  parser.add_argument( '-trk-simple', dest="trk_simple", default=None,
             help='Filename for output track statistics.' )
  parser.add_argument( '-trk-mot', dest="trk_mot", default=None,
             help='Filename for output track statistics.' )

  # Scoring settings
  parser.add_argument( "-iou-thresh", dest="iou_thresh", default=0.5,
             help="IOU threshold for detection conf matrices and stats option" )
  parser.add_argument( "--per-category", dest="per_category", action="store_true",
             help="For options where it matters, run scoring individually per category" )

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
    print( "Error: both computed and truth file must be specified" )
    sys.exit( 0 )

  if not args.det_roc and not args.trk_simple and not args.det_conf and not args.trk_mot:
    print( "Error: either 'trk-simple', 'trk-mot', 'det-roc', or 'det-conf' must be specified" )
    sys.exit( 0 )

  categories = []

  if args.input_format != "viame-csv":
    print( "Error: only viame-csv format is widely supported by this tool" )
    sys.exit( 0 )
  else:
    args.input_format = "noaa-csv"

  if args.per_category:
    categories = list_categories_viame_csv( args.truth )

  if args.det_roc:
    generate_det_rocs( args, categories )

  if args.det_conf:
    generate_det_conf( args, categories )

  if args.trk_simple:
    generate_trk_simp_stats( args, categories )

  if args.trk_mot:
    generate_trk_mot_stats( args, categories )

