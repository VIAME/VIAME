# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Shared, KWIVER-independent helpers for detection fusion.

This module contains the pure-numpy math used both by the runtime
'nms_fusion' merger (merge_detections_nms_fusion.py) and by the offline
parameter learning tool (tools/train_detection_fusion.py):

  - IoU / box matching helpers
  - ProbEn-style probabilistic ensembling of full class distributions
  - Recovery of the input detections contributing to each fused box
  - Per-model per-class isotonic score calibration (fit and apply)
  - A small logistic-regression re-scorer ("stacking") over fusion
    agreement features (fit and apply)
  - Weighted fusion of instance masks belonging to a fused box

Everything here operates on plain python / numpy values so it can be
used without KWIVER present.
"""

import json
import math

import numpy as np

BACKGROUND_ID = 0

# --------------------------- IOU HELPERS -------------------------------

def bb_intersection_over_union( A, B ):
    xA = max( A[0], B[0] )
    yA = max( A[1], B[1] )
    xB = min( A[2], B[2] )
    yB = min( A[3], B[3] )

    inter_area = max( 0, xB - xA ) * max( 0, yB - yA )

    if inter_area == 0:
        return 0.0

    area_a = ( A[2] - A[0] ) * ( A[3] - A[1] )
    area_b = ( B[2] - B[0] ) * ( B[3] - B[1] )

    return inter_area / float( area_a + area_b - inter_area )

def find_matching_box( boxes_list, new_box, match_iou ):
    best_iou = match_iou
    best_index = -1
    for i in range( len( boxes_list ) ):
        iou = bb_intersection_over_union( boxes_list[i], new_box )
        if iou > best_iou:
            best_index = i
            best_iou = iou
    return best_index, best_iou

class BoxSpatialIndex( object ):
    """Uniform-grid index over a fixed list of boxes for fast IoU-match
    queries. Boxes are ( x1, y1, x2, y2 ) in any consistent coordinate scale;
    cell_size must be in that same scale (~0.05 for [0,1]-normalized boxes,
    the scale the pseudonym matching runs in). find_matching_box() returns the
    same ( index, iou ) as the module-level linear scan but only tests boxes
    sharing a grid cell with the query, turning the O(N*M) pseudonym pass into
    roughly O(N+M) on the typical case of small, spatially-separated boxes."""

    def __init__( self, boxes_list, cell_size=0.05 ):
        self._boxes = [ np.asarray( b, dtype=float ).reshape( -1 )[:4]
                        for b in boxes_list ]
        self._cell = float( cell_size ) if cell_size and cell_size > 0 else 0.05
        self._grid = {}
        for i, b in enumerate( self._boxes ):
            for key in self._cells( b ):
                self._grid.setdefault( key, [] ).append( i )

    def _cells( self, b ):
        c = self._cell
        cx0 = int( math.floor( min( b[0], b[2] ) / c ) )
        cx1 = int( math.floor( max( b[0], b[2] ) / c ) )
        cy0 = int( math.floor( min( b[1], b[3] ) / c ) )
        cy1 = int( math.floor( max( b[1], b[3] ) / c ) )
        for cx in range( cx0, cx1 + 1 ):
            for cy in range( cy0, cy1 + 1 ):
                yield ( cx, cy )

    def find_matching_box( self, new_box, match_iou ):
        new_box = np.asarray( new_box, dtype=float ).reshape( -1 )[:4]
        best_iou = match_iou
        best_index = -1
        seen = set()
        for key in self._cells( new_box ):
            for i in self._grid.get( key, () ):
                if i in seen:
                    continue
                seen.add( i )
                iou = bb_intersection_over_union( self._boxes[i], new_box )
                if iou > best_iou:
                    best_index = i
                    best_iou = iou
        return best_index, best_iou

# ---------------------- PROBABILISTIC ENSEMBLING ------------------------

def dists_to_array( dist_dicts, class_count ):
    """Convert a list of { class_id: probability } dicts into a dense
    (N, class_count) array, assigning any leftover probability mass to
    the background column.
    """
    output = np.zeros( ( len( dist_dicts ), class_count ), dtype=float )
    for i, dist in enumerate( dist_dicts ):
        for class_id, prob in dist.items():
            if 0 <= class_id < class_count:
                output[ i, class_id ] += prob
        leftover = 1.0 - output[i].sum()
        output[ i, BACKGROUND_ID ] += max( leftover, 0.0 )
    return output

def proben_fusion( boxes_list, scores_list, labels_list, dists_list,
                   weights, iou_thr, skip_box_thr ):
    """Probabilistic ensembling (ProbEn-style) of detections from
    multiple models.

    Detections are greedily clustered across models by same-label IoU,
    then each cluster's class distributions are fused with a weighted
    (log-linear) independent-evidence product and its boxes with a
    score-weighted average. Models without a detection in a cluster
    simply contribute no evidence.

    Inputs are per-model lists: boxes (N,4 arrays, any consistent
    coordinate scale), top-class scores (N,), top-class labels (N,),
    and full class distributions (N,K arrays whose column BACKGROUND_ID
    is the background probability). Weights are per-model exponents.

    Returns fused boxes (M,4), scores (M,), labels (M,), distributions
    (M,K), and per-fused-box contributor lists of (model_index,
    detection_index, iou_with_fused_box) tuples.
    """
    entries = []
    for m in range( len( boxes_list ) ):
        for i in range( len( boxes_list[m] ) ):
            if scores_list[m][i] >= skip_box_thr:
                entries.append( ( float( scores_list[m][i] ), m, i ) )
    entries.sort( key=lambda e: -e[0] )

    cluster_members = []   # list of lists of ( model, index )
    cluster_boxes = []     # running score-weighted mean box per cluster
    cluster_labels = []
    cluster_weights = []   # running sum of score * model weight

    for score, m, i in entries:
        box = boxes_list[m][i]
        label = labels_list[m][i]
        weight = score * weights[m]
        best_iou = iou_thr
        best_ind = -1
        for c in range( len( cluster_members ) ):
            if cluster_labels[c] != label:
                continue
            iou = bb_intersection_over_union( cluster_boxes[c], box )
            if iou > best_iou:
                best_ind = c
                best_iou = iou
        if best_ind < 0:
            cluster_members.append( [ ( m, i ) ] )
            cluster_boxes.append( np.array( box, dtype=float ) )
            cluster_labels.append( label )
            cluster_weights.append( weight )
        else:
            total = cluster_weights[ best_ind ] + weight
            cluster_boxes[ best_ind ] = (
              cluster_boxes[ best_ind ] * cluster_weights[ best_ind ] +
              np.asarray( box, dtype=float ) * weight ) / total
            cluster_members[ best_ind ].append( ( m, i ) )
            cluster_weights[ best_ind ] = total

    class_count = dists_list[0].shape[1] if len( dists_list ) and \
                  hasattr( dists_list[0], 'shape' ) else 0
    eps = 1e-8

    fused_boxes = []
    fused_scores = []
    fused_labels = []
    fused_dists = []
    contributors = []

    for c in range( len( cluster_members ) ):
        log_p = np.zeros( class_count, dtype=float )
        for m, i in cluster_members[c]:
            log_p += weights[m] * np.log(
              np.clip( dists_list[m][i], eps, 1.0 ) )
        log_p -= log_p.max()
        dist = np.exp( log_p )
        dist /= dist.sum()

        foreground = np.array( dist )
        foreground[ BACKGROUND_ID ] = -1.0
        label = int( np.argmax( foreground ) )
        score = float( dist[ label ] )

        fused_boxes.append( cluster_boxes[c] )
        fused_scores.append( score )
        fused_labels.append( label )
        fused_dists.append( dist )
        contributors.append(
          [ ( m, i, bb_intersection_over_union(
                cluster_boxes[c], boxes_list[m][i] ) )
            for m, i in cluster_members[c] ] )

    order = np.argsort( -np.array( fused_scores ) ) if fused_scores else []
    return ( np.array( [ fused_boxes[i] for i in order ] ).reshape( -1, 4 ),
             np.array( [ fused_scores[i] for i in order ] ),
             np.array( [ fused_labels[i] for i in order ], dtype=int ),
             np.array( [ fused_dists[i] for i in order ] ).reshape(
               -1, class_count ),
             [ contributors[i] for i in order ] )

# ----------------------- CONTRIBUTOR RECOVERY --------------------------

def find_contributors( fused_boxes, fused_labels, boxes_list, labels_list,
                       iou_thr ):
    """For each fused box, find the best-matching input detection with
    the same label from each model. Returns one list per fused box of
    (model_index, detection_index, iou) tuples. Used to recover cluster
    membership for fusion methods which do not report it.
    """
    output = []
    for f in range( len( fused_boxes ) ):
        contribs = []
        for m in range( len( boxes_list ) ):
            best_iou = iou_thr
            best_ind = -1
            for i in range( len( boxes_list[m] ) ):
                if labels_list[m][i] != fused_labels[f]:
                    continue
                iou = bb_intersection_over_union(
                  fused_boxes[f], boxes_list[m][i] )
                if iou > best_iou:
                    best_ind = i
                    best_iou = iou
            if best_ind >= 0:
                contribs.append( ( m, best_ind, best_iou ) )
        output.append( contribs )
    return output

# ------------------------- SCORE CALIBRATION ---------------------------

def fit_isotonic( scores, targets, max_knots=32 ):
    """Fit an isotonic (monotone non-decreasing) regression of binary
    targets onto scores using pool-adjacent-violators, returning a
    piecewise-linear table { 'x': [...], 'y': [...] } usable with
    apply_calibration(). Returns None when there are no samples or
    the targets are constant.
    """
    scores = np.asarray( scores, dtype=float )
    targets = np.asarray( targets, dtype=float )
    if len( scores ) == 0 or targets.min() == targets.max():
        return None

    order = np.argsort( scores )
    x = scores[ order ]
    y = targets[ order ]

    # Pool adjacent violators
    block_val = []
    block_wt = []
    block_x0 = []
    block_x1 = []
    for xi, yi in zip( x, y ):
        block_val.append( float( yi ) )
        block_wt.append( 1.0 )
        block_x0.append( float( xi ) )
        block_x1.append( float( xi ) )
        while len( block_val ) > 1 and block_val[-2] >= block_val[-1]:
            wt = block_wt[-2] + block_wt[-1]
            val = ( block_val[-2] * block_wt[-2] +
                    block_val[-1] * block_wt[-1] ) / wt
            x1 = block_x1[-1]
            for lst in ( block_val, block_wt, block_x0, block_x1 ):
                lst.pop()
            block_val[-1] = val
            block_wt[-1] = wt
            block_x1[-1] = x1

    # Convert blocks to interpolation knots at block midpoints
    knot_x = []
    knot_y = []
    for x0, x1, val in zip( block_x0, block_x1, block_val ):
        mid = 0.5 * ( x0 + x1 )
        if knot_x and mid <= knot_x[-1]:
            mid = knot_x[-1] + 1e-6
        knot_x.append( mid )
        knot_y.append( val )

    # Downsample to a bounded number of knots
    if len( knot_x ) > max_knots:
        inds = np.unique( np.linspace(
          0, len( knot_x ) - 1, max_knots ).astype( int ) )
        knot_x = [ knot_x[i] for i in inds ]
        knot_y = [ knot_y[i] for i in inds ]

    return { 'x': [ round( v, 6 ) for v in knot_x ],
             'y': [ round( v, 6 ) for v in knot_y ] }

def apply_calibration( score, table ):
    """Map a raw score through a piecewise-linear calibration table."""
    return float( np.interp( score, table['x'], table['y'] ) )

def calibrate_model_score( model_table, class_name, score ):
    """Calibrate one score using a per-model table dict which maps class
    names (or '*' for the model-wide fallback) to calibration tables.
    Returns the input unchanged when no applicable table exists.
    """
    if not model_table:
        return score
    table = model_table.get( class_name, model_table.get( '*' ) )
    if not table:
        return score
    return apply_calibration( score, table )

def load_calibration( filename ):
    """Load a calibration file, returning a list (one entry per model)
    of dicts mapping class name (or '*') to calibration tables.
    """
    with open( filename ) as fin:
        data = json.load( fin )
    return data[ 'models' ]

def save_calibration( filename, model_tables ):
    with open( filename, 'w' ) as fout:
        json.dump( { 'models': model_tables }, fout, indent=2 )

# ------------------------ STACKED RE-SCORING ---------------------------

def stack_feature_names( model_count ):
    return [ 'fused_score', 'agreement', 'mean_iou', 'max_model_score' ] + \
           [ 'model_score_' + str( m ) for m in range( model_count ) ]

def compute_stack_features( fused_score, contribs, scores_list, model_count ):
    """Compute the agreement feature vector for one fused box given its
    contributor list of (model_index, detection_index, iou) tuples.
    """
    per_model = [ 0.0 ] * model_count
    ious = []
    for m, i, iou in contribs:
        per_model[m] = max( per_model[m], float( scores_list[m][i] ) )
        ious.append( iou )
    agreement = len( set( m for m, _, _ in contribs ) ) / float( model_count )
    mean_iou = float( np.mean( ious ) ) if ious else 0.0
    max_model = max( per_model ) if per_model else 0.0
    return [ float( fused_score ), agreement, mean_iou, max_model ] + per_model

def logistic_fit( features, targets, l2=1e-3, iters=100 ):
    """Fit an L2-regularized logistic regression with Newton-Raphson on
    standardized features. Returns a model dict usable with
    logistic_predict(); feature order must match stack_feature_names().
    """
    X = np.asarray( features, dtype=float )
    y = np.asarray( targets, dtype=float )
    mean = X.mean( axis=0 )
    std = X.std( axis=0 )
    std[ std < 1e-9 ] = 1.0
    Xs = ( X - mean ) / std
    Xs = np.hstack( [ Xs, np.ones( ( len( Xs ), 1 ) ) ] )

    w = np.zeros( Xs.shape[1] )
    for _ in range( iters ):
        z = np.clip( Xs.dot( w ), -30, 30 )
        p = 1.0 / ( 1.0 + np.exp( -z ) )
        grad = Xs.T.dot( p - y ) + l2 * w
        r = np.maximum( p * ( 1.0 - p ), 1e-6 )
        hess = ( Xs * r[ :, None ] ).T.dot( Xs ) + l2 * np.eye( Xs.shape[1] )
        step = np.linalg.solve( hess, grad )
        w -= step
        if np.abs( step ).max() < 1e-8:
            break

    return { 'mean': [ float( v ) for v in mean ],
             'std': [ float( v ) for v in std ],
             'coef': [ float( v ) for v in w[:-1] ],
             'intercept': float( w[-1] ) }

def logistic_predict( feature_row, model ):
    x = ( np.asarray( feature_row, dtype=float ) -
          np.asarray( model['mean'] ) ) / np.asarray( model['std'] )
    z = float( np.dot( x, model['coef'] ) ) + model['intercept']
    z = min( max( z, -30.0 ), 30.0 )
    return 1.0 / ( 1.0 + math.exp( -z ) )

def load_rescore_model( filename ):
    with open( filename ) as fin:
        return json.load( fin )

def save_rescore_model( filename, model, feature_names ):
    output = dict( model )
    output[ 'features' ] = feature_names
    with open( filename, 'w' ) as fout:
        json.dump( output, fout, indent=2 )

# --------------------------- MASK FUSION -------------------------------

def fuse_mask_arrays( fused_box, contrib_boxes, contrib_masks,
                      contrib_weights, method='union' ):
    """Fuse instance masks belonging to one fused box.

    fused_box is [x1, y1, x2, y2] in pixels; each contributor has its
    own pixel-space box, a 2D mask array aligned to that box's top-left
    corner, and a scalar weight. Returns a uint8 mask sized to the fused
    box, or None when no valid contributor masks exist.

    method selects how overlapping masks combine (use it to favor a detector):
      'union'        - foreground if ANY contributor sets the pixel. On the sea
                       lion seg models this scored highest (the models slightly
                       under-segment, so the union recovers more true extent).
      'intersection' - foreground only where EVERY contributor sets the pixel
                       (conservative; keeps just the agreed core).
      'max_conf'     - use only the single highest-weight contributor's mask, so
                       the mask from the higher-weighted / higher-confidence
                       detector wins outright.
      'vote'         - per-pixel weighted majority: foreground where the summed
                       contributor weight reaches half the total covering weight.
    """
    fx1, fy1 = int( round( fused_box[0] ) ), int( round( fused_box[1] ) )
    fx2, fy2 = int( round( fused_box[2] ) ), int( round( fused_box[3] ) )
    width, height = fx2 - fx1, fy2 - fy1
    if width <= 0 or height <= 0:
        return None

    if method == 'max_conf':
        best = None
        for box, mask, weight in zip( contrib_boxes, contrib_masks,
                                      contrib_weights ):
            if mask is not None and ( best is None or weight > best[0] ):
                best = ( weight, box, mask )
        if best is None:
            return None
        contrib_boxes, contrib_masks, contrib_weights = \
            [ best[1] ], [ best[2] ], [ best[0] ]

    acc = np.zeros( ( height, width ), dtype=float )   # weighted set
    wsum = np.zeros( ( height, width ), dtype=float )  # weighted covering
    hit = np.zeros( ( height, width ), dtype=int )     # count of contributors set
    n_valid = 0

    for box, mask, weight in zip( contrib_boxes, contrib_masks,
                                  contrib_weights ):
        if mask is None:
            continue
        mask = np.squeeze( np.asarray( mask ) )
        if mask.ndim != 2:
            continue
        n_valid += 1
        bx1, by1 = int( round( box[0] ) ), int( round( box[1] ) )
        # Region of the contributor mask inside the fused box
        x1 = max( bx1, fx1 )
        y1 = max( by1, fy1 )
        x2 = min( bx1 + mask.shape[1], fx2 )
        y2 = min( by1 + mask.shape[0], fy2 )
        if x2 <= x1 or y2 <= y1:
            continue
        sub = ( mask[ y1 - by1 : y2 - by1, x1 - bx1 : x2 - bx1 ] > 0 )
        acc[ y1 - fy1 : y2 - fy1, x1 - fx1 : x2 - fx1 ] += weight * sub
        wsum[ y1 - fy1 : y2 - fy1, x1 - fx1 : x2 - fx1 ] += weight
        hit[ y1 - fy1 : y2 - fy1, x1 - fx1 : x2 - fx1 ] += sub

    if n_valid == 0:
        return None
    if method == 'intersection':
        return ( hit >= n_valid ).astype( np.uint8 )
    if method == 'union' or method == 'max_conf':
        return ( hit > 0 ).astype( np.uint8 )
    return ( ( acc >= 0.5 * wsum ) & ( wsum > 0 ) ).astype( np.uint8 )
