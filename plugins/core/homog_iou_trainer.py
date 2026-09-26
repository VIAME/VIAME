# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Homography IoU tracker parameter estimation from track groundtruth.

The frame-to-frame camera motion is approximated from the groundtruth itself:
for each track, the other tracks' box centers in two consecutive frames give
a leave-one-out transform, which stands in for image registration. The IoU
gate is then chosen to separate a track's mapped box overlapping its own next
box from it overlapping another target's.
"""

from kwiver.vital.algo import TrainTracker

import os
import json
import numpy as np


def _iou( a, b ):
    x0 = max( a[0], b[0] )
    y0 = max( a[1], b[1] )
    x1 = min( a[2], b[2] )
    y1 = min( a[3], b[3] )
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = ( x1 - x0 ) * ( y1 - y0 )
    area_a = ( a[2] - a[0] ) * ( a[3] - a[1] )
    area_b = ( b[2] - b[0] ) * ( b[3] - b[1] )
    return inter / ( area_a + area_b - inter )


def _center( box ):
    return np.array( [ ( box[0] + box[2] ) / 2.0, ( box[1] + box[3] ) / 2.0 ] )


def _fit_transform( src, dst ):
    """Linear part and translation mapping src points onto dst, or None."""
    if len( src ) >= 4:
        centered = src - src.mean( axis=0 )
        sv = np.linalg.svd( centered, compute_uv=False )
        if sv[-1] > 1e-3 * max( sv[0], 1e-9 ):
            design = np.hstack( [ src, np.ones( ( len( src ), 1 ) ) ] )
            sol, _, _, _ = np.linalg.lstsq( design, dst, rcond=None )
            linear = sol[:2].T
            if abs( np.linalg.det( linear ) ) > 1e-6:
                return linear, sol[2]
    if len( src ) >= 2 and np.ptp( src, axis=0 ).max() > 1e-3:
        # Similarity: a translation alone misses the rotation, which far from
        # the image center moves a small box by more than its own size
        n = len( src )
        design = np.zeros( ( 2 * n, 4 ) )
        design[0::2] = np.column_stack( [ src[:, 0], -src[:, 1], np.ones( n ), np.zeros( n ) ] )
        design[1::2] = np.column_stack( [ src[:, 1], src[:, 0], np.zeros( n ), np.ones( n ) ] )
        ( a, b, tx, ty ), _, _, _ = np.linalg.lstsq( design, dst.reshape( -1 ), rcond=None )
        if a * a + b * b > 1e-6:
            return np.array( [ [ a, -b ], [ b, a ] ] ), np.array( [ tx, ty ] )
    if len( src ) >= 1:
        return np.eye( 2 ), np.median( dst - src, axis=0 )
    return None


def _map_box( box, linear, offset ):
    center = linear @ _center( box ) + offset
    scale = np.sqrt( abs( np.linalg.det( linear ) ) )
    hw = ( box[2] - box[0] ) * scale / 2.0
    hh = ( box[3] - box[1] ) * scale / 2.0
    return ( center[0] - hw, center[1] - hh, center[0] + hw, center[1] + hh )


class HomogIOUTrainer( TrainTracker ):
    """
    Estimates the homog_iou tracker's IoU gate and lost-track buffer from
    groundtruth tracks alone.
    """
    def __init__( self ):
        TrainTracker.__init__( self )

        self._identifier = "viame-homog-iou-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "homog_iou_tracker"
        self._threshold = "0.00"
        self._min_iou_floor = 0.05
        self._min_iou_ceiling = 0.6

        self._train_tracks = []
        self._test_tracks = []

    def get_configuration( self ):
        cfg = super( TrainTracker, self ).get_configuration()

        cfg.set_value( "identifier", self._identifier )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "output_directory", self._output_directory )
        cfg.set_value( "output_prefix", self._output_prefix )
        cfg.set_value( "threshold", self._threshold )
        cfg.set_value( "min_iou_floor", str( self._min_iou_floor ) )
        cfg.set_value( "min_iou_ceiling", str( self._min_iou_ceiling ) )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._identifier = str( cfg.get_value( "identifier" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )
        self._output_prefix = str( cfg.get_value( "output_prefix" ) )
        self._threshold = str( cfg.get_value( "threshold" ) )
        self._min_iou_floor = float( cfg.get_value( "min_iou_floor" ) )
        self._min_iou_ceiling = float( cfg.get_value( "min_iou_ceiling" ) )

        for directory in ( self._train_directory, self._output_directory ):
            if directory and not os.path.exists( directory ):
                os.makedirs( directory )

        return True

    def check_configuration( self, cfg ):
        if not cfg.has_value( "identifier" ) or \
          len( cfg.get_value( "identifier" ) ) == 0:
            print( "A model identifier must be specified!" )
            return False
        return True

    def add_data_from_disk( self, categories, train_files, train_tracks,
                            test_files, test_tracks ):
        print( "Adding training data from disk..." )
        print( "  Training tracks: ", len( train_tracks ) )
        print( "  Test tracks: ", len( test_tracks ) )

        self._train_tracks = list( train_tracks )
        self._test_tracks = list( test_tracks )

    def _collect( self ):
        """True/false IoU samples and within-track frame gaps."""
        true_ious = []
        false_ious = []
        gaps = []

        for track_set in self._train_tracks + self._test_tracks:
            if track_set is None:
                continue

            frames = {}
            for track in track_set.tracks():
                prev_frame = None
                for state in track:
                    det = state.detection()
                    if det is None:
                        continue
                    bbox = det.bounding_box
                    frames.setdefault( state.frame_id, {} )[ track.id ] = (
                        bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y() )
                    if prev_frame is not None and state.frame_id > prev_frame:
                        gaps.append( state.frame_id - prev_frame )
                    prev_frame = state.frame_id

            ordered = sorted( frames )
            for f0, f1 in zip( ordered[:-1], ordered[1:] ):
                prev, cur = frames[ f0 ], frames[ f1 ]
                common = [ tid for tid in prev if tid in cur ]

                for tid in common:
                    others = [ o for o in common if o != tid ]
                    src = np.array( [ _center( prev[o] ) for o in others ] ).reshape( -1, 2 )
                    dst = np.array( [ _center( cur[o] ) for o in others ] ).reshape( -1, 2 )
                    fit = _fit_transform( src, dst )
                    if fit is None:
                        continue

                    mapped = _map_box( prev[ tid ], *fit )
                    true_ious.append( _iou( mapped, cur[ tid ] ) )

                    wrong = [ _iou( mapped, box ) for o, box in cur.items() if o != tid ]
                    if wrong:
                        false_ious.append( max( wrong ) )

        return np.array( true_ious ), np.array( false_ious ), np.array( gaps )

    def _choose_min_iou( self, true_ious, false_ious ):
        if len( true_ious ) == 0:
            return 0.2

        grid = np.round( np.arange( self._min_iou_floor,
                                    self._min_iou_ceiling + 1e-9, 0.01 ), 4 )
        scores = np.array( [ np.mean( true_ious >= t ) -
                             ( np.mean( false_ious >= t ) if len( false_ious ) else 0.0 )
                             for t in grid ] )

        # Sparse targets tie across a wide band; stay near the stock 0.2
        # rather than taking the most permissive end of it.
        tied = grid[ scores >= scores.max() - 1e-9 ]
        return float( tied[ np.argmin( np.abs( tied - 0.2 ) ) ] )

    @staticmethod
    def _choose_max_lost( gaps ):
        if len( gaps ) == 0:
            return 2
        return int( np.clip( np.ceil( np.percentile( gaps, 95 ) ) - 1, 1, 10 ) )

    def update_model( self ):
        print( "\nEstimating homog_iou tracker parameters from groundtruth..." )

        true_ious, false_ious, gaps = self._collect()
        min_iou = self._choose_min_iou( true_ious, false_ious )
        max_lost = self._choose_max_lost( gaps )

        def summary( values ):
            if len( values ) == 0:
                return { 'count': 0 }
            return {
                'count': int( len( values ) ),
                'mean': float( np.mean( values ) ),
                'p05': float( np.percentile( values, 5 ) ),
                'median': float( np.median( values ) ),
                'p95': float( np.percentile( values, 95 ) ),
                'frac_ge_min_iou': float( np.mean( values >= min_iou ) ),
            }

        params = {
            'min_iou': min_iou,
            'max_lost': max_lost,
            'new_track_thresh': float( self._threshold ),
            'true_iou': summary( true_ious ),
            'false_iou': summary( false_ious ),
        }

        print( "  frame pairs: {} true, {} false".format(
            len( true_ious ), len( false_ious ) ) )
        if len( true_ious ):
            print( "  true IoU median {:.3f}, 5th percentile {:.3f}".format(
                params['true_iou']['median'], params['true_iou']['p05'] ) )
        if len( false_ious ):
            print( "  false IoU median {:.3f}, 95th percentile {:.3f}".format(
                params['false_iou']['median'], params['false_iou']['p95'] ) )
        print( "  min_iou: {:.3f}".format( min_iou ) )
        print( "  max_lost: {}".format( max_lost ) )

        params_file = os.path.join( self._train_directory, "homog_iou_params.json" )
        with open( params_file, 'w' ) as f:
            json.dump( params, f, indent=2 )
        print( "Saved parameters to {}".format( params_file ) )

        algo = "homog_iou"
        return {
            "type": algo,
            algo + ":min_iou": "{:.3f}".format( min_iou ),
            algo + ":max_lost": str( max_lost ),
            algo + ":new_track_thresh": self._threshold,
            "homog_iou_params.json": params_file,
        }


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        HomogIOUTrainer,
        "homog_iou",
        "Homography IoU tracker parameter estimation from track groundtruth",
    )
