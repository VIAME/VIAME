# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# Copyright (c) Microsoft Corporation. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

from viame.processes.base import ViameProcess
from viame.pipeline import process, datum

from viame.types import Image
from viame.types import ImageContainer
from viame.types import F2FHomography

from PIL import Image as pil_image
from viame.util.pil import get_pil_image, from_pil

from viame import image_kernels
from viame.image_processing import features, matching
from viame.utilities import geometry

import csv
import logging
import numpy as np
import scipy.spatial

logger = logging.getLogger(__name__)

# What `cv2.MOTION_HOMOGRAPHY`, `MOTION_AFFINE` and `MOTION_EUCLIDEAN` named.
# The values are cv2's, so a caller that passed the cv2 constant still gets what
# it asked for.
MOTION_TRANSLATION = 0
MOTION_EUCLIDEAN = 1
MOTION_AFFINE = 2
MOTION_HOMOGRAPHY = 3


def compute_transform( optical, thermal, warp_mode = MOTION_HOMOGRAPHY,
  match_low_res=True, good_match_percent = 0.15, ratio_test = .85,
  match_height = 512, min_matches = 4, min_inliers = 4 ):
 
    # Convert images to grayscale    
    if len( thermal.shape ) == 3 and thermal.shape[2] == 3:
        thermal_gray = image_kernels.to_gray(thermal)
    else:
        thermal_gray = thermal
 
    if len( optical.shape ) == 3 and optical.shape[2] == 3: 
        optical_gray = image_kernels.to_gray(optical)
    else:
        optical_gray = optical

    # resize if requested
    if match_low_res:
        aspect = optical_gray.shape[1] / optical_gray.shape[0]
        optical_gray = image_kernels.resize(optical_gray, int( match_height*aspect ), match_height)
    
    # Detect SIFT features and compute descriptors. `features.sift` is VIAME's
    # own, and gives keypoints as an (n, 6) array rather than a list of
    # objects -- so scaling the locations below is a slice rather than a loop.
    keypoints1, descriptors1 = features.sift( thermal_gray )
    keypoints2, descriptors2 = features.sift( optical_gray )

    if len( keypoints1 ) < 2:
        logger.warning("Not enough keypoints in thermal image")
        return False, np.identity( 3 ), 0

    if len( keypoints2 ) < 2:
        logger.warning("Not enough keypoints in optical image")
        return False, np.identity( 3 ), 0

    # scale feature points back to original size
    if match_low_res:
        scale = optical.shape[0] / optical_gray.shape[0]
        keypoints2 = keypoints2.copy()
        keypoints2[:, :2] *= scale

    # Pick good features
    if ratio_test < 1:
        matches = matching.ratio_match( descriptors1, descriptors2,
                                        ratio_test )
    else:
        # The top percentage by distance, over the mutually nearest pairs.
        pairs = matching.match( descriptors1, descriptors2, cross_check=True )

        if pairs:
            query = np.array( [ q for q, _ in pairs ] )
            train = np.array( [ t for _, t in pairs ] )
            distance = np.sqrt( ( ( descriptors1[ query ].astype( np.float64 )
                                    - descriptors2[ train ] ) ** 2
                                  ).sum( axis=1 ) )
            order = np.argsort( distance, kind="stable" )
            keep = int( len( pairs ) * good_match_percent )
            matches = [ pairs[ i ] for i in order[ :keep ] ]
        else:
            matches = []

    logger.debug( "%d matches found", len(matches) )

    if len( matches ) < min_matches:
        logger.warning( "Not enough matches: %d < %d", len(matches), min_matches )
        return False, np.identity( 3 ), 0

    # Extract location of good matches
    points1 = np.array( [ keypoints1[ q, :2 ] for q, _ in matches ],
                        dtype=np.float32 )
    points2 = np.array( [ keypoints2[ t, :2 ] for _, t in matches ],
                        dtype=np.float32 )

    # Find homography
    h, mask = geometry.find_homography( points1, points2 )

    if h is None:
        logger.warning( "Homography estimation found no consensus" )
        return False, np.identity( 3 ), 0

    mask = mask.astype( np.uint8 ).reshape( -1, 1 )

    logger.debug( "%d inliers found", sum( mask ) )

    if sum( mask ) < min_inliers:
        logger.warning( "Not enough inliers: %d < %d", sum(mask), min_inliers )
        return False, np.identity( 3 ), 0

    # Check if we have a robust set of inliers by computing the area of the convex hull
    
    # Good area is 11392
    try:
        inlier_area = scipy.spatial.ConvexHull( points2[ np.isclose( mask.ravel(), 1 ) ] ).area
        logger.debug( "Inlier area: %f", inlier_area )

        if inlier_area < 1000:
            logger.warning("Inliers seem colinear or too close, skipping")
            return False, np.identity(3), 0
    except (scipy.spatial.QhullError, ValueError) as e:
        logger.warning( "Inliers seem colinear or too close, skipping: %s", e )
        return False, np.identity(3), 0

    # if non homography requested, compute from inliers
    if warp_mode != MOTION_HOMOGRAPHY:
        inliers = np.isclose( mask.ravel(), 1 )
        points1_inliers = points1[ inliers ]
        points2_inliers = points2[ inliers ]

        # `geometry.estimate_affine_2d` where this called
        # `cv2.estimateRigidTransform`, which **OpenCV removed in 4.x**: this
        # branch had not run under a modern cv2 at all, it raised
        # AttributeError. Its `fullAffine` flag is this one's `full`, which is
        # the migration OpenCV's own deprecation notice names.
        a, _ = geometry.estimate_affine_2d(
            points1_inliers, points2_inliers,
            full=( warp_mode == MOTION_AFFINE ) )

        if a is None:
            return False, np.identity(3), 0

        h = np.identity(3)

        # turn in 3x3 transform
        h[0,:] = a[0,:]
        h[1,:] = a[1,:]

    return True, h, sum( mask )

# normlize thermal image
def normalize_thermal( thermal_image, percent=0.01 ):
    if thermal_image is None or not thermal_image.size:
        return None
    if thermal_image.dtype == np.uint8:
        return thermal_image
    low, high = np.percentile(thermal_image, [percent, 100 - percent])
    if high <= low:
        return np.zeros(thermal_image.shape, dtype=np.uint8)
    scaled = np.floor((thermal_image.astype(np.float64) - low) *
                      (256.0 / (high - low)))
    return np.clip(scaled, 0, 255).astype(np.uint8)


class register_frames_process( ViameProcess ):
    """
    Register optical and thermal frames.
    """
    # -------------------------------------------------------------------------
    def __init__( self, conf ):
        ViameProcess.__init__( self, conf )

        # set up configs
        self.add_config_trait( "good_match_percent", "good_match_percent",
                               '0.15', 'Good match percent [0.0,1.0].' )
        self.add_config_trait( "ratio_test", "ratio_test",
                               '0.85', 'Feature point test ratio' )
        self.add_config_trait( "match_height", "match_height",
                               '512', 'Match height.' )
        self.add_config_trait( "min_matches", "min_matches",
                               '4', 'Minimum number of feature matches' )
        self.add_config_trait( "min_inliers", "min_inliers",
                               '4', 'Minimum number of inliers' )

        self.declare_config_using_trait( 'good_match_percent' )
        self.declare_config_using_trait( 'ratio_test' )
        self.declare_config_using_trait( 'match_height' )
        self.declare_config_using_trait( 'min_matches' )
        self.declare_config_using_trait( 'min_inliers' )

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        # declare our ports (port-name, flags)
        self.add_port_trait( "optical_image", "image", "Input image" )
        self.add_port_trait( "thermal_image", "image", "Input image" )

        self.add_port_trait( "warped_optical_image", "image", "Output image" )
        self.add_port_trait( "warped_thermal_image", "image", "Output image" )
        self.add_port_trait( "optical_to_thermal_homog", "homography_src_to_ref", "Output homog" )
        self.add_port_trait( "thermal_to_optical_homog", "homography_src_to_ref", "Output homog" )

        self.declare_input_port_using_trait( 'optical_image', required )
        self.declare_input_port_using_trait( 'thermal_image', required )

        self.declare_output_port_using_trait( 'warped_optical_image', optional )
        self.declare_output_port_using_trait( 'warped_thermal_image', optional )
        self.declare_output_port_using_trait( 'optical_to_thermal_homog', optional )
        self.declare_output_port_using_trait( 'thermal_to_optical_homog', optional )

    # -------------------------------------------------------------------------
    def _configure( self ):
        self._base_configure()

        self._good_match_percent = float( self.config_value( 'good_match_percent' ) )
        self._ratio_test = float( self.config_value( 'ratio_test' ) )
        self._match_height = int( self.config_value( 'match_height' ) )
        self._min_matches = int( self.config_value( 'min_matches' ) )
        self._min_inliers = int( self.config_value( 'min_inliers' ) )

    # -------------------------------------------------------------------------
    def _step( self ):
        # grab image container from port using traits
        optical_c = self.grab_input_using_trait( 'optical_image' )
        thermal_c = self.grab_input_using_trait( 'thermal_image' )

        # Get python image from conatiner (just for show)
        optical_npy = optical_c.image().asarray().astype('uint8')
        thermal_npy = thermal_c.image().asarray().astype('uint16')

        thermal_norm = normalize_thermal( thermal_npy )

        if thermal_norm is not None and optical_npy is not None:
            # compute transform
            ret, transform, _ = compute_transform(
                optical_npy,
                thermal_norm,
                warp_mode = MOTION_HOMOGRAPHY,
                match_low_res = True,
                good_match_percent = self._good_match_percent,
                ratio_test = self._ratio_test,
                match_height = self._match_height,
                min_matches = self._min_matches,
                min_inliers = self._min_inliers )
        else:
           ret = False

        if ret:
            # TODO: Make all of these computations conditional on port connection
            inv_transform = np.linalg.inv( transform )

            thermal_warped = image_kernels.warp_perspective( thermal_npy,
              transform, optical_npy.shape[1], optical_npy.shape[0] )
            optical_warped = image_kernels.warp_perspective( optical_npy,
              inv_transform, thermal_npy.shape[1], thermal_npy.shape[0] )

            self.push_to_port_using_trait( 'thermal_to_optical_homog',
                F2FHomography.from_doubles( transform, 0, 0 ) )
            self.push_to_port_using_trait( 'optical_to_thermal_homog',
                F2FHomography.from_doubles( inv_transform, 0, 0 ) )

            self.push_to_port_using_trait( 'warped_thermal_image',
              ImageContainer.fromarray( thermal_warped ) )
            self.push_to_port_using_trait( 'warped_optical_image',
              ImageContainer.fromarray( optical_warped ) )
        else:
            logger.warning( "Frame alignment failed" )

            self.push_datum_to_port( 'thermal_to_optical_homog', datum.empty() )
            self.push_datum_to_port( 'optical_to_thermal_homog', datum.empty() )

            self.push_datum_to_port( 'warped_optical_image', datum.empty() )
            self.push_datum_to_port( 'warped_thermal_image', datum.empty() )

        self._base_step()


def __sprokit_register__():
    """Register when the module loader discovers this file directly."""
    from viame.pipeline import process_factory
    module_name = 'python:opencv.multimodal_registration'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process('ocv_multimodal_registration',
                                'Register optical and thermal frames',
                                register_frames_process)
    process_factory.mark_process_module_as_loaded(module_name)
