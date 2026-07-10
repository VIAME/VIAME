#!/usr/bin/python

# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Learn detection fusion parameters for the 'nms_fusion' merger.

Given groundtruth and the scored detection CSVs from two or more detectors
run over the same imagery, search for the fusion settings (fusion method,
per-detector weights, IoU threshold) which maximize mean average precision,
then write out a pipeline configuration block for use with the
merge_detection_sets process.

Supported fusion methods are weighted boxes fusion ('wbf'), non-maximum
weighted ('nmw'), soft NMS ('soft_nms'), plain NMS ('nms'), and
probabilistic ensembling of full class distributions ('proben').

Optionally, per-detector per-class isotonic score calibration can be
learned before fusion (--calibrate), and a logistic re-scorer over fusion
agreement features can be learned on top of the fused output (--stack).
Both write side files consumed by the runtime process.

Example:

  python train_detection_fusion.py                    \\
    -truth seq1_truth.csv seq2_truth.csv              \\
    -computed seq1_det1.csv seq2_det1.csv             \\
    -computed seq1_det2.csv seq2_det2.csv             \\
    --calibrate --stack                               \\
    -output-config fusion_params.pipe

Each -computed flag lists a single detector's outputs, with files in the
same sequence order as the files given to -truth. Frames are aligned
across files using the frame number column of each CSV.

Detectors trained with minor class differences can be reconciled with the
-labels flag (a file where each line contains a canonical class name
followed by any synonyms for it), and detectors with hierarchical class
differences can additionally use the same label_dic / pseudo_dic /
pseudo_ind options supported by the runtime fusion process.
"""

import argparse
import ast
import contextlib
import io
import json
import os
import random
import sys

import numpy as np

with contextlib.redirect_stdout( io.StringIO() ):
  from map_boxes import mean_average_precision_for_boxes

from ensemble_boxes import nms
from ensemble_boxes import soft_nms
from ensemble_boxes import non_maximum_weighted
from ensemble_boxes import weighted_boxes_fusion

try:
  from viame.core import detection_fusion_core as dfc
except ImportError:
  sys.path.insert( 0, os.path.join(
    os.path.dirname( os.path.abspath( __file__ ) ),
    '..', 'plugins', 'core' ) )
  import detection_fusion_core as dfc

fusion_methods = [ 'wbf', 'nmw', 'soft_nms', 'nms', 'proben' ]

# -------------------------- INPUT PARSING -----------------------------

def print_and_exit( msg, code=1 ):
  print( msg )
  sys.exit( code )

def parse_synonym_file( filename ):
  output = {}
  with open( filename ) as fin:
    for line in fin:
      if line.startswith( '#' ):
        continue
      tokens = [ t for t in line.strip().split() if not t.startswith( ':' ) ]
      if len( tokens ) < 2:
        continue
      for synonym in tokens[1:]:
        output[ synonym ] = tokens[0]
  return output

def parse_viame_csv( filename, class_map ):
  # Returns dict of frame number ->
  #   list of [ x1, y1, x2, y2, score, label, { label: score } ]
  output = {}
  with open( filename ) as fin:
    for line in fin:
      line = line.strip()
      if not line or line.startswith( '#' ):
        continue
      parts = line.split( ',' )
      if len( parts ) < 11:
        continue
      frame = int( parts[2] )
      x1, y1, x2, y2 = map( float, parts[3:7] )
      dist = {}
      idx = 9
      while idx + 1 < len( parts ):
        name = parts[idx].strip()
        if not name or name.startswith( '(' ):
          break
        try:
          score = float( parts[idx+1] )
        except ValueError:
          break
        name = class_map.get( name, name )
        dist[ name ] = max( dist.get( name, 0.0 ), score )
        idx += 2
      if not dist:
        continue
      label = max( dist, key=dist.get )
      output.setdefault( frame, [] ).append(
        [ x1, y1, x2, y2, dist[ label ], label, dist ] )
  return output

# ------------------------- SCORE CALIBRATION --------------------------

def collect_calibration_tables( seq_truth, seq_computed, n_models, args ):
  """Learn per-model isotonic calibration tables mapping raw scores to
  the empirical probability of being a correct detection. Per-class
  tables are fit for classes with enough samples; a pooled '*' table is
  always fit as the fallback.
  """
  tables = []
  for m in range( n_models ):
    samples = {}
    for s in range( len( seq_truth ) ):
      truth = seq_truth[s]
      computed = seq_computed[s][m]
      for fid, dets in computed.items():
        truth_dets = truth.get( fid, [] )
        by_class = {}
        for d in dets:
          by_class.setdefault( d[5], [] ).append( d )
        for cls, cls_dets in by_class.items():
          gt = [ t for t in truth_dets if t[5] == cls ]
          used = [ False ] * len( gt )
          for d in sorted( cls_dets, key=lambda d: -d[4] ):
            best_ind = -1
            best_iou = args.map_iou
            for gi, t in enumerate( gt ):
              if used[ gi ]:
                continue
              iou = dfc.bb_intersection_over_union( d[:4], t[:4] )
              if iou > best_iou:
                best_ind = gi
                best_iou = iou
            if best_ind >= 0:
              used[ best_ind ] = True
            scores, targets = samples.setdefault( cls, ( [], [] ) )
            scores.append( d[4] )
            targets.append( 1 if best_ind >= 0 else 0 )
    model_table = {}
    all_scores = []
    all_targets = []
    for cls, ( scores, targets ) in samples.items():
      all_scores.extend( scores )
      all_targets.extend( targets )
      if len( scores ) >= args.calibration_min_samples:
        table = dfc.fit_isotonic( scores, targets )
        if table:
          model_table[ cls ] = table
    table = dfc.fit_isotonic( all_scores, all_targets )
    if table:
      model_table[ '*' ] = table
    tables.append( model_table )
  return tables

def apply_calibration_to_dets( seq_computed, tables ):
  for computed_models in seq_computed:
    for m, computed in enumerate( computed_models ):
      for dets in computed.values():
        for d in dets:
          d[6] = { name: dfc.calibrate_model_score( tables[m], name, score )
                   for name, score in d[6].items() }
          d[5] = max( d[6], key=d[6].get )
          d[4] = d[6][ d[5] ]

# --------------------------- FRAME BUILDING ---------------------------

def apply_pseudonyms( models, pseudo_dic, pseudo_ind, match_iou ):
  # Mirrors the category upsampling step of merge_detections_nms_fusion
  for chk_ind in pseudo_ind:
    boxes, _, labels = models[ chk_ind ][:3]
    for k in range( len( boxes ) ):
      orig_label = labels[k]
      if orig_label not in pseudo_dic:
        continue
      for other_ind in pseudo_ind[ chk_ind ]:
        best_idx, _ = dfc.find_matching_box(
          models[ other_ind ][0], boxes[k], match_iou )
        if best_idx >= 0:
          new_label = models[ other_ind ][2][ best_idx ]
          if new_label in pseudo_dic[ orig_label ]:
            labels[k] = new_label
            break

def build_frames( seq_truth, seq_computed, args, label_dic,
                  pseudo_dic, pseudo_ind, need_dists ):
  n_models = len( seq_computed[0] ) if seq_computed else 0

  if label_dic is None:
    names = set()
    for per_frame in seq_truth + [ c for s in seq_computed for c in s ]:
      for dets in per_frame.values():
        names.update( d[5] for d in dets )
    label_dic = { 'background': 0 }
    for i, name in enumerate( sorted( names ) ):
      if name not in label_dic:
        label_dic[ name ] = i + 1

  id_to_label = {}
  for name, label_id in label_dic.items():
    if label_id not in id_to_label:
      id_to_label[ label_id ] = name
  class_count = max( label_dic.values() ) + 1

  frames = []
  ann_rows = []
  for s, truth in enumerate( seq_truth ):
    computed = seq_computed[s]
    frame_ids = sorted( set( truth ) | { f for c in computed for f in c } )
    for fid in frame_ids:
      img_id = str( s ) + '_' + str( fid )
      frame_truth = []
      for d in truth.get( fid, [] ):
        if d[5] not in label_dic:
          continue
        ann_rows.append(
          [ img_id, id_to_label[ label_dic[ d[5] ] ], d[0], d[2], d[1], d[3] ] )
        frame_truth.append( ( d[:4], label_dic[ d[5] ] ) )
      model_dets = [ [ d for d in c.get( fid, [] ) if d[5] in label_dic ]
                     for c in computed ]
      if not any( len( m ) for m in model_dets ):
        continue
      all_dets = [ d for m in model_dets for d in m ]
      norm_w = args.width if args.width > 0 else max( d[2] for d in all_dets ) + 1
      norm_h = args.height if args.height > 0 else max( d[3] for d in all_dets ) + 1
      models = []
      for m in model_dets:
        boxes = np.array( [ [ d[0] / norm_w, d[1] / norm_h,
                              d[2] / norm_w, d[3] / norm_h ]
                            for d in m ], dtype=float ).reshape( -1, 4 )
        boxes = np.clip( boxes, 0.0, 1.0 )
        scores = np.array( [ d[4] for d in m ], dtype=float )
        labels = np.array( [ label_dic[ d[5] ] for d in m ], dtype=int )
        dists = None
        if need_dists:
          dists = dfc.dists_to_array(
            [ { label_dic[ name ]: score for name, score in d[6].items()
                if name in label_dic } for d in m ], class_count )
        models.append( [ boxes, scores, labels, dists ] )
      if pseudo_ind:
        apply_pseudonyms( models, pseudo_dic, pseudo_ind, args.match_iou )
      frames.append( { 'img': img_id, 'nw': norm_w, 'nh': norm_h,
                       'models': models, 'truth': frame_truth } )

  ann = np.array( ann_rows, dtype=object )
  return frames, ann, label_dic, id_to_label, class_count

# ------------------------- FUSION + SCORING ---------------------------

def run_fusion( frame, method, weights, iou_thr, skip_box_thr, sigma,
                want_contribs=False ):
  """Fuse one frame. Returns (boxes, scores, labels, contributors),
  where contributors is None unless requested (or free), or None when
  the frame has no detections.
  """
  boxes_list = []
  scores_list = []
  labels_list = []
  dists_list = []
  used_weights = []
  for m in range( len( frame['models'] ) ):
    boxes, scores, labels, dists = frame['models'][m]
    if len( boxes ) == 0:
      continue
    boxes_list.append( boxes )
    scores_list.append( scores )
    labels_list.append( labels )
    dists_list.append( dists )
    used_weights.append( weights[m] )
  if not boxes_list:
    return None
  if method == 'proben':
    boxes, scores, labels, _, contribs = dfc.proben_fusion(
      boxes_list, scores_list, labels_list, dists_list,
      used_weights, iou_thr, skip_box_thr )
    return boxes, scores, labels, contribs
  if method == 'wbf':
    boxes, scores, labels = weighted_boxes_fusion(
      boxes_list, scores_list, labels_list,
      weights=used_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr )
  elif method == 'nmw':
    boxes, scores, labels = non_maximum_weighted(
      boxes_list, scores_list, labels_list,
      weights=used_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr )
  elif method == 'soft_nms':
    boxes, scores, labels = soft_nms(
      boxes_list, scores_list, labels_list,
      weights=used_weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr )
  elif method == 'nms':
    boxes, scores, labels = nms(
      boxes_list, scores_list, labels_list,
      weights=used_weights, iou_thr=iou_thr )
  else:
    print_and_exit( "Unknown fusion method: " + method )
  contribs = None
  if want_contribs:
    contribs = dfc.find_contributors(
      boxes, labels, boxes_list, labels_list, iou_thr )
  return boxes, scores, labels, contribs

def compute_map( ann, pred_rows, map_iou ):
  if len( pred_rows ) == 0 or len( ann ) == 0:
    return 0.0, {}
  pred = np.array( pred_rows, dtype=object )
  with contextlib.redirect_stdout( io.StringIO() ):
    mean_ap, per_class = mean_average_precision_for_boxes(
      ann, pred, iou_threshold=map_iou, verbose=False )
  return mean_ap, per_class

def evaluate( frames, ann, id_to_label, map_iou, method, weights,
              iou_thr, skip_box_thr, sigma ):
  pred_rows = []
  for frame in frames:
    result = run_fusion( frame, method, weights, iou_thr, skip_box_thr, sigma )
    if result is None:
      continue
    boxes, scores, labels, _ = result
    for i in range( len( boxes ) ):
      pred_rows.append( [ frame['img'], id_to_label[ int( labels[i] ) ],
                          scores[i],
                          boxes[i][0] * frame['nw'], boxes[i][2] * frame['nw'],
                          boxes[i][1] * frame['nh'], boxes[i][3] * frame['nh'] ] )
  return compute_map( ann, pred_rows, map_iou )

def evaluate_baseline( frames, ann, id_to_label, map_iou, model_inds ):
  pred_rows = []
  for frame in frames:
    for m in model_inds:
      boxes, scores, labels = frame['models'][m][:3]
      for i in range( len( boxes ) ):
        pred_rows.append( [ frame['img'], id_to_label[ int( labels[i] ) ],
                            scores[i],
                            boxes[i][0] * frame['nw'], boxes[i][2] * frame['nw'],
                            boxes[i][1] * frame['nh'], boxes[i][3] * frame['nh'] ] )
  return compute_map( ann, pred_rows, map_iou )

# --------------------------- OPTIMIZATION -----------------------------

def optimize_method( method, n_models, args, cached_eval ):
  iou_grid = [ round( float( x ), 4 ) for x in
               np.arange( args.iou_min, args.iou_max + 1e-6, args.iou_step ) ]
  sigma_grid = [ 0.05, 0.1, 0.25, 0.5 ] if method == 'soft_nms' \
               else [ args.sigma ]

  best = [ -1.0, [ 1.0 ] * n_models, iou_grid[0], sigma_grid[0] ]

  def consider( weights, iou_thr, sigma ):
    mean_ap = cached_eval( method, weights, iou_thr, sigma )
    if mean_ap > best[0]:
      best[:] = [ mean_ap, list( weights ), iou_thr, sigma ]
      return True
    return False

  # Stage 1: sweep iou threshold (and sigma) with equal weights
  equal_weights = [ 1.0 ] * n_models
  for iou_thr in iou_grid:
    for sigma in sigma_grid:
      consider( equal_weights, iou_thr, sigma )
  print( "  {:>9}: stage 1 equal-weight sweep, mAP {:.4f} at iou_thr {}".format(
    method, best[0], best[2] ) )

  # Stage 2: random search over per-detector weights
  rng = random.Random( args.seed )
  for _ in range( args.trials ):
    weights = [ 1.0 ] + [ round( rng.uniform( 0.1, 3.0 ), 2 )
                          for _ in range( n_models - 1 ) ]
    consider( weights, rng.choice( iou_grid ), rng.choice( sigma_grid ) )
  print( "  {:>9}: stage 2 random search, mAP {:.4f} with weights {}".format(
    method, best[0], best[1] ) )

  # Stage 3: coordinate ascent refinement
  for _ in range( args.refine_iters ):
    improved = False
    for ind in range( n_models ):
      for factor in ( 0.5, 0.75, 0.9, 1.1, 1.33, 2.0 ):
        weights = list( best[1] )
        weights[ind] = round( weights[ind] * factor, 4 )
        improved |= consider( weights, best[2], best[3] )
    for delta in ( -args.iou_step, -args.iou_step / 2,
                   args.iou_step / 2, args.iou_step ):
      iou_thr = min( 0.95, max( 0.05, round( best[2] + delta, 4 ) ) )
      improved |= consider( best[1], iou_thr, best[3] )
    if method == 'soft_nms':
      for factor in ( 0.5, 0.8, 1.25, 2.0 ):
        improved |= consider( best[1], best[2], round( best[3] * factor, 4 ) )
    if not improved:
      break
  print( "  {:>9}: stage 3 refinement, mAP {:.4f} with weights {}, "
    "iou_thr {}".format( method, best[0], best[1], best[2] ) )
  return best

# ------------------------- STACKED RE-SCORING -------------------------

def train_stacker( frames, ann, id_to_label, args, method, weights,
                   sigma, iou_thr, n_models ):
  """Learn a logistic re-scorer over fusion agreement features, and
  return (model_dict, rescored_map, sample_count).
  """
  features = []
  targets = []
  pred_meta = []

  for frame in frames:
    result = run_fusion( frame, method, weights, iou_thr,
                         args.skip_box_thr, sigma, want_contribs=True )
    if result is None:
      continue
    boxes, scores, labels, contribs = result
    scores_lists = [ frame['models'][m][1] for m in range( n_models ) ]
    # Map model indices in contributors back to global model indices in
    # case some models had no detections this frame
    active = [ m for m in range( n_models )
               if len( frame['models'][m][0] ) ]
    used = [ False ] * len( frame['truth'] )
    order = np.argsort( -np.asarray( scores ) )
    correct = [ 0 ] * len( scores )
    for i in order:
      box_px = [ boxes[i][0] * frame['nw'], boxes[i][1] * frame['nh'],
                 boxes[i][2] * frame['nw'], boxes[i][3] * frame['nh'] ]
      best_ind = -1
      best_iou = args.map_iou
      for gi, ( gt_box, gt_label ) in enumerate( frame['truth'] ):
        if used[ gi ] or gt_label != int( labels[i] ):
          continue
        iou = dfc.bb_intersection_over_union( box_px, gt_box )
        if iou > best_iou:
          best_ind = gi
          best_iou = iou
      if best_ind >= 0:
        used[ best_ind ] = True
        correct[i] = 1
    for i in range( len( scores ) ):
      remapped = [ ( active[m], j, iou ) for m, j, iou in contribs[i] ]
      feats = dfc.compute_stack_features(
        scores[i], remapped, scores_lists, n_models )
      features.append( feats )
      targets.append( correct[i] )
      pred_meta.append( ( frame, boxes[i], int( labels[i] ), feats ) )

  if len( features ) < 20 or sum( targets ) in ( 0, len( targets ) ):
    return None, 0.0, len( features )

  model = dfc.logistic_fit( features, targets )

  pred_rows = []
  for frame, box, label_id, feats in pred_meta:
    pred_rows.append( [ frame['img'], id_to_label[ label_id ],
                        dfc.logistic_predict( feats, model ),
                        box[0] * frame['nw'], box[2] * frame['nw'],
                        box[1] * frame['nh'], box[3] * frame['nh'] ] )
  rescored_map = compute_map( ann, pred_rows, args.map_iou )[0]
  return model, rescored_map, len( features )

# ----------------------------- OUTPUT ---------------------------------

def format_dict( dic ):
  entries = sorted( dic.items(), key=lambda kv: ( kv[1], kv[0] ) ) \
            if all( isinstance( k, str ) for k in dic ) else sorted( dic.items() )
  formatted = []
  for key, value in entries:
    key_str = "'" + key + "'" if isinstance( key, str ) else str( key )
    formatted.append( key_str + ": " + str( value ) )
  return "{ " + ", ".join( formatted ) + " }"

def write_pipe_config( args, best, label_dic, pseudo_dic, pseudo_ind,
                       calibration_file=None, rescore_file=None ):
  mean_ap, weights, iou_thr, sigma, method = \
    best[0], best[1], best[2], best[3], best[4]
  entries = [
    ( 'fusion_type', method ),
    ( 'fusion_weights', '[ ' + ', '.join( '{:g}'.format( w )
                                          for w in weights ) + ' ]' ),
    ( 'iou_thr', '{:g}'.format( iou_thr ) ),
    ( 'skip_box_thr', '{:g}'.format( args.skip_box_thr ) ) ]
  if method == 'soft_nms':
    entries.append( ( 'sigma', '{:g}'.format( sigma ) ) )
  if calibration_file:
    entries.append( ( 'calibration_file', calibration_file ) )
  if rescore_file:
    entries.append( ( 'rescore_model_file', rescore_file ) )
  entries.append( ( 'label_dic', format_dict( label_dic ) ) )
  if pseudo_dic:
    entries.append( ( 'match_iou', '{:g}'.format( args.match_iou ) ) )
    entries.append( ( 'pseudo_dic', format_dict( pseudo_dic ) ) )
    entries.append( ( 'pseudo_ind', format_dict( pseudo_ind ) ) )

  with open( args.output_config, 'w' ) as fout:
    fout.write( "# Detection fusion parameters generated by "
      "train_detection_fusion.py\n" )
    fout.write( "# Achieved mAP@{:g} of {:.4f} on the training set\n".format(
      args.map_iou, mean_ap ) )
    fout.write( "#\n" )
    fout.write( "# Connect each detector to the detected_object_set1..N "
      "ports of this\n" )
    fout.write( "# process in the same order as the -computed flags given "
      "during training.\n\n" )
    fout.write( "process merger\n" )
    fout.write( "  :: merge_detection_sets\n" )
    fout.write( "  :merger:type                       nms_fusion\n" )
    for key, value in entries:
      fout.write( "  :merger:nms_fusion:{:<18} {}\n".format( key, value ) )

# ------------------------------ MAIN ----------------------------------

def main():
  parser = argparse.ArgumentParser(
    description='Learn detection fusion parameters for the nms_fusion merger',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter )

  parser.add_argument( '-truth', nargs='+', required=True,
    help='Groundtruth VIAME CSV file(s), one per sequence' )
  parser.add_argument( '-computed', action='append', nargs='+', required=True,
    help='Computed detection CSVs for one detector, one file per sequence '
         'in the same order as -truth; repeat this flag for each detector' )
  parser.add_argument( '-labels', default=None,
    help='Optional class synonym file; each line lists a canonical class '
         'name followed by synonyms which should be mapped onto it' )
  parser.add_argument( '-methods', default='wbf,nmw,proben',
    help='Comma-separated fusion methods to consider out of: '
         + ','.join( fusion_methods ) )
  parser.add_argument( '--calibrate', action='store_true',
    help='Learn per-detector per-class isotonic score calibration before '
         'fusion and write it alongside the fusion parameters' )
  parser.add_argument( '--stack', action='store_true',
    help='Learn a logistic re-scorer over fusion agreement features on '
         'top of the best fusion configuration' )
  parser.add_argument( '-map-iou', dest='map_iou', type=float, default=0.5,
    help='IoU threshold used when scoring mean average precision' )
  parser.add_argument( '-trials', type=int, default=100,
    help='Number of random search trials per fusion method' )
  parser.add_argument( '-refine-iters', dest='refine_iters', type=int,
    default=20, help='Maximum coordinate ascent refinement iterations' )
  parser.add_argument( '-seed', type=int, default=0,
    help='Random seed used during random weight search' )
  parser.add_argument( '-iou-min', dest='iou_min', type=float, default=0.30,
    help='Minimum fusion iou_thr considered' )
  parser.add_argument( '-iou-max', dest='iou_max', type=float, default=0.85,
    help='Maximum fusion iou_thr considered' )
  parser.add_argument( '-iou-step', dest='iou_step', type=float, default=0.05,
    help='Step between considered fusion iou_thr values' )
  parser.add_argument( '-skip-box-thr', dest='skip_box_thr', type=float,
    default=0.0001, help='Minimum box confidence used during fusion' )
  parser.add_argument( '-sigma', type=float, default=0.1,
    help='Default sigma for soft_nms (a small grid is also searched)' )
  parser.add_argument( '-width', type=int, default=0,
    help='Image width for coordinate normalization; when 0 a per-frame '
         'normalization factor is computed, matching runtime behavior' )
  parser.add_argument( '-height', type=int, default=0,
    help='Image height for coordinate normalization; when 0 a per-frame '
         'normalization factor is computed, matching runtime behavior' )
  parser.add_argument( '-label-dic', dest='label_dic', default='',
    help='Optional class name to integer id mapping, using the same syntax '
         'as the runtime process; derived from the data when unset' )
  parser.add_argument( '-pseudo-dic', dest='pseudo_dic', default='',
    help='Optional class id pseudonym mapping applied before fusion, using '
         'the same syntax as the runtime process (requires -label-dic)' )
  parser.add_argument( '-pseudo-ind', dest='pseudo_ind', default='',
    help='Optional detector index pseudonym mapping applied before fusion, '
         'using the same syntax as the runtime process' )
  parser.add_argument( '-match-iou', dest='match_iou', type=float, default=0.5,
    help='IoU used when matching boxes during pseudonym upsampling' )
  parser.add_argument( '-calibration-min-samples',
    dest='calibration_min_samples', type=int, default=50,
    help='Minimum samples for a per-class calibration table; classes with '
         'fewer fall back to the pooled per-detector table' )
  parser.add_argument( '-output-config', dest='output_config',
    default='fusion_params.pipe',
    help='Output pipeline configuration file for the learned parameters' )
  parser.add_argument( '-output-calibration', dest='output_calibration',
    default='fusion_calibration.json',
    help='Output file for learned score calibration (with --calibrate)' )
  parser.add_argument( '-output-rescore', dest='output_rescore',
    default='fusion_rescore.json',
    help='Output file for the learned re-scorer (with --stack)' )
  parser.add_argument( '-output-json', dest='output_json', default='',
    help='Optional output json file with full optimization results' )

  args = parser.parse_args()

  if len( args.computed ) < 2:
    print_and_exit( "At least two -computed detector inputs are required" )
  for grp in args.computed:
    if len( grp ) != len( args.truth ):
      print_and_exit( "Each -computed flag must list one file per -truth "
        "sequence ({} expected, {} received)".format(
          len( args.truth ), len( grp ) ) )

  methods = [ m.strip() for m in args.methods.split( ',' ) if m.strip() ]
  for method in methods:
    if method not in fusion_methods:
      print_and_exit( "Unknown fusion method: " + method )

  class_map = parse_synonym_file( args.labels ) if args.labels else {}
  label_dic = ast.literal_eval( args.label_dic ) if args.label_dic else None
  pseudo_dic = ast.literal_eval( args.pseudo_dic ) if args.pseudo_dic else {}
  pseudo_ind = ast.literal_eval( args.pseudo_ind ) if args.pseudo_ind else {}
  if pseudo_ind and label_dic is None:
    print_and_exit( "-pseudo-dic and -pseudo-ind require -label-dic so that "
      "class ids are unambiguous" )

  n_models = len( args.computed )
  seq_truth = [ parse_viame_csv( f, class_map ) for f in args.truth ]
  seq_computed = [ [ parse_viame_csv( grp[s], class_map )
                     for grp in args.computed ]
                   for s in range( len( args.truth ) ) ]

  calibration_file = None
  if args.calibrate:
    tables = collect_calibration_tables(
      seq_truth, seq_computed, n_models, args )
    apply_calibration_to_dets( seq_computed, tables )
    dfc.save_calibration( args.output_calibration, tables )
    calibration_file = args.output_calibration
    print( "Learned score calibration:" )
    for m, table in enumerate( tables ):
      per_class = sorted( k for k in table if k != '*' )
      print( "  detector {}: pooled table{}".format( m + 1,
        ", per-class tables for " + ", ".join( per_class )
        if per_class else "" ) )
    print( "Wrote " + args.output_calibration + "\n" )

  need_dists = 'proben' in methods
  frames, ann, label_dic, id_to_label, class_count = build_frames(
    seq_truth, seq_computed, args, label_dic,
    pseudo_dic, pseudo_ind, need_dists )

  if len( ann ) == 0:
    print_and_exit( "No groundtruth annotations were loaded" )
  if not frames:
    print_and_exit( "No computed detections were loaded" )

  print( "Loaded {} frame(s) over {} sequence(s), {} groundtruth "
    "annotations".format( len( frames ), len( args.truth ), len( ann ) ) )
  for m in range( n_models ):
    count = sum( len( f['models'][m][0] ) for f in frames )
    print( "  detector {}: {} detections".format( m + 1, count ) )

  print( "\nBaseline mAP@{:g}{}:".format( args.map_iou,
    " (calibrated scores)" if args.calibrate else "" ) )
  baselines = {}
  for m in range( n_models ):
    baselines[ 'detector_' + str( m + 1 ) ] = evaluate_baseline(
      frames, ann, id_to_label, args.map_iou, [ m ] )[0]
    print( "  detector {} alone     : {:.4f}".format(
      m + 1, baselines[ 'detector_' + str( m + 1 ) ] ) )
  baselines[ 'concatenation' ] = evaluate_baseline(
    frames, ann, id_to_label, args.map_iou, list( range( n_models ) ) )[0]
  print( "  naive concatenation  : {:.4f}".format(
    baselines[ 'concatenation' ] ) )

  cache = {}
  def cached_eval( method, weights, iou_thr, sigma ):
    key = ( method, tuple( round( w, 4 ) for w in weights ),
            round( iou_thr, 4 ), round( sigma, 4 ) )
    if key not in cache:
      cache[ key ] = evaluate( frames, ann, id_to_label, args.map_iou,
        method, weights, iou_thr, args.skip_box_thr, sigma )[0]
    return cache[ key ]

  print( "\nOptimizing fusion parameters:" )
  results = {}
  for method in methods:
    results[ method ] = optimize_method( method, n_models, args, cached_eval )

  best_method = max( results, key=lambda m: results[m][0] )
  best = results[ best_method ] + [ best_method ]

  mean_ap, per_class = evaluate( frames, ann, id_to_label, args.map_iou,
    best_method, best[1], best[2], args.skip_box_thr, best[3] )

  rescore_file = None
  stack_map = None
  if args.stack:
    print( "\nTraining stacked re-scorer on best configuration..." )
    stack_model, stack_map, n_samples = train_stacker(
      frames, ann, id_to_label, args, best_method,
      best[1], best[3], best[2], n_models )
    if stack_model is None:
      print( "  insufficient or degenerate samples "
        "({}); skipping".format( n_samples ) )
    else:
      print( "  re-scored mAP {:.4f} (fusion alone {:.4f}, {} samples)".format(
        stack_map, best[0], n_samples ) )
      dfc.save_rescore_model( args.output_rescore, stack_model,
        dfc.stack_feature_names( n_models ) )
      print( "  Wrote " + args.output_rescore )
      if stack_map > best[0]:
        rescore_file = args.output_rescore
      else:
        print( "  (no improvement over fusion alone; not referenced in "
          "the output pipe config)" )

  print( "\nBest configuration:" )
  print( "  fusion_type    : " + best_method )
  print( "  fusion_weights : " + str( best[1] ) )
  print( "  iou_thr        : {:g}".format( best[2] ) )
  if best_method == 'soft_nms':
    print( "  sigma          : {:g}".format( best[3] ) )
  print( "  mAP@{:g}        : {:.4f}".format( args.map_iou, mean_ap ) )
  for name in sorted( per_class ):
    print( "    {:<20} AP {:.4f} ({} gt)".format(
      name, per_class[ name ][0], per_class[ name ][1] ) )

  write_pipe_config( args, best, label_dic, pseudo_dic, pseudo_ind,
                     calibration_file, rescore_file )
  print( "\nWrote " + args.output_config )

  if args.output_json:
    output = {
      'baselines': baselines,
      'methods': { m: { 'mean_ap': r[0], 'fusion_weights': r[1],
                        'iou_thr': r[2], 'sigma': r[3] }
                   for m, r in results.items() },
      'best': { 'fusion_type': best_method, 'mean_ap': mean_ap,
                'fusion_weights': best[1], 'iou_thr': best[2],
                'sigma': best[3], 'skip_box_thr': args.skip_box_thr,
                'map_iou': args.map_iou,
                'per_class_ap': { k: v[0] for k, v in per_class.items() } },
      'calibration_file': calibration_file,
      'rescore_file': args.output_rescore if args.stack else None,
      'rescored_map': stack_map,
      'label_dic': label_dic }
    with open( args.output_json, 'w' ) as fout:
      json.dump( output, fout, indent=2 )
    print( "Wrote " + args.output_json )

if __name__ == "__main__":
  main()
