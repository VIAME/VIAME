# ckwg +29
# Copyright 2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import itertools

import kwiver.vital.types as kvt

# XXX This class should be defined somewhere else
from .simple_homog_tracker import Transformer

@Transformer.decorate
def stabilize_many_images(
        compute_features_and_descriptors,
        estimate_single_homography,
):
    """Create a Transformer that performs stabilization on multiple
    images captured by a multi-camera system.  Arguments:
    - compute_features_and_descriptors should be a
      FeatureAndDescriptorComputer or similar callable
    - estimate_single_homography should be a SingleHomographyEstimator
      or similar callable

    The .step call expects one argument:
    - a list of kvt.BaseImageContainer objects
    and returns:
    - a list of homographies, one for each input image, that maps them
      and other processed images to a common coordinate system (XXX
      what homography type?)

    """
    ris = register_image_set(estimate_single_homography)
    eh = estimate_homography()
    output = None
    while True:
        images, = yield output
        fds = list(map(compute_features_and_descriptors, images))
        spatial_homogs = ris.step(fds)
        all_features, all_descs = merge_feature_and_descriptor_sets(
            warp_features_and_descriptors(h, *fd)
            for h, fd in zip(spatial_homogs, fds)
            # XXX Maybe we want to support something like this to
            # handle questionable matches:
            #   if h is not None
        )
        temporal_homog = eh.step(all_features, all_descs)
        output = [compose_homographies(temporal_homog, sh)
                  for sh in spatial_homogs]

@Transformer.decorate
def register_image_set(estimate_single_homography):
    """Create a Transformer that registers sets of images (described
    by feature and descriptor sets) from a camera system together,
    maintaining information on previously estimated relative
    positions.  Arguments:
    - estimate_single_homography should be a SingleHomographyEstimator
      or similar callable

    The .step call expects one argument:
    - a list of kvt.FeatureSet--kvt.DescriptorSet pairs (adjacent
      images should overlap)
    and returns:
    - a list of homographies, one for each input image, that map them
      to some common coordinate space (XXX what homography type?)

    """
    # XXX At the moment it's stateless
    output = None
    while True:
        # The returned homographies will be relative to the "middle" one
        fds, = yield output
        hl = len(fds) // 2
        # Compute homographies between adjacent images
        l2c = [estimate_single_homography(*sfd, *tfd)
               for sfd, tfd in zip(fds[:hl], fds[1:hl + 1])]
        r2c = [estimate_single_homography(*sfd, *tfd)
               for tfd, sfd in zip(fds[hl:-1], fds[hl + 1:])]
        # Compute homographies to center
        l2c = list(itertools.accumulate(reversed(l2c), compose_homographies))[::-1]
        r2c = itertools.accumulate(r2c, compose_homographies)
        output = [*l2c, get_identity_homography(), *r2c]

@Transformer.decorate
def estimate_homography():
    """Create a Transformer that estimates homographies using features and
    descriptors.  The .step call expects two arguments:
    - a kvt.FeatureSet
    - the corresponding kvt.DescriptorSet
    and returns:
    - a homography mapping the current coordinates to those in a
      reference frame

    """
    output = None
    while True:
        features, descriptors = yield output
        raise NotImplementedError

def get_identity_homography():
    """Return a homography representing the identity transformation"""
    return kvt.HomographyD()

class SingleHomographyEstimator:
    __slots__ = '_feature_matcher', '_homography_estimator'
    def __init__(self, feature_matcher, homography_estimator):
        self._feature_matcher = feature_matcher
        self._homography_estimator = homography_estimator

    def __call__(self,
                 source_features, source_descriptors,
                 target_features, target_descriptors):
        """Return a homography that converts coordinates of the source to
        those of the target, estimated using the provided feature and
        descriptor sets.

        (XXX What happens when estimation fails?)

        """
        matches = self._feature_matcher.match(
            source_features, source_descriptors,
            target_features, target_descriptors,
        )
        return self._homography_estimator.estimate(
            source_features, target_features, matches,
        )[0]  # Only return the homography, not the inliers

class FeatureAndDescriptorComputer:
    __slots__ = '_feature_detector', '_descriptor_extractor'
    def __init__(self, feature_detector, descriptor_extractor):
        self._feature_detector = feature_detector
        self._descriptor_extractor = descriptor_extractor

    def __call__(self, image):
        """Return a pair containing the kvt.FeatureSet and corresponding
        kvt.DescriptorSet for the given kvt.BaseImageContainer

        """
        features = self._feature_detector.detect(image)
        return self._descriptor_extractor.extract(image, features)[::-1]

def warp_features_and_descriptors(homog, features, descriptors):
    """Return a pair of the given feature set and descriptor set, warped
    according to the provided homography.

    """
    raise NotImplementedError

def merge_feature_and_descriptor_sets(fd_pairs):
    """Merge an iterable of pairs of feature sets and corresponding
    descriptor sets into a single pair of one feature set and the
    corresponding descriptor set.

    Near-duplicates (e.g. those nearly sharing a position) are removed
    except for a single copy.

    (XXX estimate_single_homography will compute a matching between
    feature points, which could be used to very good effect here)

    """
    raise NotImplementedError

def compose_homographies(second, first):
    """Return a homography corresponding to applying the first, then the
    second homography.

    The arguments may be either kvt.BaseHomography or
    kvt.F2FHomography objects.  The result is a kvt.BaseHomography if
    both arguments are, and a kvt.F2FHomography if at least one
    argument is.  In the latter case, if only one argument is a
    kvt.F2FHomography, its frame numbers are preserved in the output,
    while if both are, their frame numbers are combined in the
    expected manner.

    """
    # XXX This is broken or at least incredibly fragile because
    # multiplication of kvt.BaseHomography objects only works if they
    # have the same scalar data type.  (The C++ f2f_homography class
    # works around this...)
    if isinstance(second, kvt.F2FHomography):
        if isinstance(first, kvt.F2FHomography):
            return second * first
        else:
            return kvt.F2FHomography(second.homography * first,
                                     second.from_id, second.to_id)
    else:
        if isinstance(first, kvt.F2FHomography):
            return kvt.F2FHomography(second * first.homography,
                                     first.from_id, first.to_id)
        else:
            return second * first
