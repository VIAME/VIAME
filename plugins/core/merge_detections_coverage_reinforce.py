# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import MergeDetections

from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

##############################################################################
# Asymmetric fusion of a well-localizing primary detector with weakly-localizing
# evidence sources (frame differencing, saliency, blob proposals, coarse
# segmentation). Box fusion averages geometry across inputs, which degrades the
# primary when the evidence finds objects but outlines them badly. Here the
# evidence never contributes geometry, only score:
#
#   reinforce  s' = s + alpha * m * ( 1 - s ), m the fraction of the primary
#              box covered by evidence. Monotone in s, saturates at 1.
#   recover    evidence boxes touching no primary box are emitted directly,
#              ranked by relative area into [ 0, recover_score ]. Ranking rather
#              than a constant matters: average precision is computed over a
#              ranked list, so a constant collapses the recovered tail to one
#              operating point. recover_score is meant to sit below the
#              primary's own output threshold.
##############################################################################

def _area( box ):
    return max( 0.0, box[2] - box[0] ) * max( 0.0, box[3] - box[1] )

def _intersection( a, b ):
    ix1, iy1 = max( a[0], b[0] ), max( a[1], b[1] )
    ix2, iy2 = min( a[2], b[2] ), min( a[3], b[3] )
    return max( 0.0, ix2 - ix1 ) * max( 0.0, iy2 - iy1 )

def _to_box( det ):
    bbox = det.bounding_box
    return [ bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y() ]

class MergeDetectionsCoverageReinforce( MergeDetections ):
    """
    Implementation of MergeDetections class
    """
    def __init__( self ):
        MergeDetections.__init__( self )

        self._alpha = 0.3
        self._recover_score = 0.05

        # Ignore evidence boxes larger than this, in px^2. 0 disables the gate.
        self._max_evidence_area = 0.0

        # 1-based input port carrying the primary. All others are evidence.
        self._primary_index = 1
        self._recover = True

    def get_configuration( self ):
        cfg = super( MergeDetections, self ).get_configuration()

        cfg.set_value( "alpha", str( self._alpha ) )
        cfg.set_value( "recover_score", str( self._recover_score ) )
        cfg.set_value( "max_evidence_area", str( self._max_evidence_area ) )
        cfg.set_value( "primary_index", str( self._primary_index ) )
        cfg.set_value( "recover", str( self._recover ) )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._alpha = float( cfg.get_value( "alpha" ) )
        self._recover_score = float( cfg.get_value( "recover_score" ) )
        self._max_evidence_area = float( cfg.get_value( "max_evidence_area" ) )
        self._primary_index = int( cfg.get_value( "primary_index" ) )
        self._recover = str( cfg.get_value( "recover" ) ).lower() in \
          ( 'true', '1', 'yes', 'on' )

        return True

    def check_configuration( self, cfg ):
        alpha = float( cfg.get_value( "alpha", str( self._alpha ) ) )
        if alpha < 0.0 or alpha > 1.0:
            return False
        if int( cfg.get_value( "primary_index",
                               str( self._primary_index ) ) ) < 1:
            return False
        return True

    def _rescored( self, det, new_score ):
        dot = None
        if det.type is not None:
            dot = DetectedObjectType()
            for class_name in det.type.class_names():
                dot.set_score( class_name, new_score )

        output = DetectedObject( det.bounding_box, new_score, dot )

        if det.mask is not None:
            output.mask = det.mask

        return output

    def merge( self, det_sets ):

        output = DetectedObjectSet()

        if not det_sets:
            return output

        primary_ind = self._primary_index - 1

        if primary_ind >= len( det_sets ):
            for det in det_sets[0]:
                output.add( det )
            return output

        primary = [ det for det in det_sets[ primary_ind ] ]

        evidence = []
        for ind, det_set in enumerate( det_sets ):
            if ind == primary_ind:
                continue
            for det in det_set:
                box = _to_box( det )
                if self._max_evidence_area > 0.0 and \
                   _area( box ) > self._max_evidence_area:
                    continue
                evidence.append( ( det, box ) )

        claimed = [ False ] * len( evidence )

        for det in primary:
            box = _to_box( det )
            box_area = _area( box )

            covered = 0.0
            for ind, ( _, evidence_box ) in enumerate( evidence ):
                overlap = _intersection( box, evidence_box )
                if overlap > 0.0:
                    claimed[ ind ] = True
                    covered += overlap

            coverage = min( 1.0, covered / box_area ) if box_area > 0.0 else 0.0
            score = det.confidence

            output.add( self._rescored(
              det, score + self._alpha * coverage * ( 1.0 - score ) ) )

        if self._recover:
            loose = [ ( det, box ) for ind, ( det, box ) in enumerate( evidence )
                      if not claimed[ ind ] ]

            if loose:
                largest = max( _area( box ) for _, box in loose ) or 1.0

                for det, box in loose:
                    output.add( self._rescored( det, self._recover_score *
                      min( 1.0, _area( box ) / largest ) ) )

        return output

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name  = "coverage_reinforce"

    if algorithm_factory.has_algorithm_impl_name(
      MergeDetectionsCoverageReinforce.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "Reinforce detections using a weakly-localizing evidence source",
      MergeDetectionsCoverageReinforce )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
