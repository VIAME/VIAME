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
        match_features, estimate_single_homography,
        compute_ref_homography,
):
    """Create a Transformer that performs stabilization on multiple
    images captured by a multi-camera system.  Arguments:
    - compute_features_and_descriptors should be a
      FeatureAndDescriptorComputer or similar callable
    - match_features should be kwiver.vital.algo.MatchFeatures.match
      (bound) or a similar callable
    - estimate_single_homography should be
      kwiver.vital.algo.EstimateHomography.estimate (bound) or a
      similar callable
    - compute_ref_homography should be
      kwiver.vital.algo.ComputeRefHomography.estimate (bound) or a
      similar callable

    The .step call expects one argument:
    - a list of kvt.BaseImageContainer objects
    and returns:
    - a list of kvt.F2FHomography objects, one for each input image,
      that maps them and other processed images to a common coordinate
      system

    """
    ris = register_image_set(SingleHomographyEstimator(
        match_features, estimate_single_homography,
    ))
    eh = estimate_homography(match_features, compute_ref_homography)
    output = None
    while True:
        images, = yield output
        fds = list(map(compute_features_and_descriptors, images))
        spatial_homogs = ris.step(fds)[0]
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
    and returns a tuple of:
    - a list of kvt.BaseHomography objects, one for each input image,
      that map them to some common coordinate space
    - a list of lists, each with one index for each input image,
      indicating corresponding features across images.  (A value of
      None indicates a lack of a matching feature in the corresponding
      image)

    """
    # XXX At the moment it's stateless
    output = None
    while True:
        # The returned homographies will be relative to the "middle" one
        fds, = yield output
        hl = len(fds) // 2
        # Compute homographies between adjacent images
        l2c, l2c_matches = [], []
        for sfd, tfd in zip(fds[:hl], fds[1:hl + 1]):
            h, m = estimate_single_homography(*sfd, *tfd)
            l2c.append(h)
            l2c_matches.append(m)
        r2c, r2c_matches = [], []
        for tfd, sfd in zip(fds[hl:-1], fds[hl + 1:]):
            h, m = estimate_single_homography(*sfd, *tfd)
            r2c.append(h)
            r2c_matches.append(m)
        # Compute homographies to center
        l2c = list(itertools.accumulate(reversed(l2c), compose_homographies))[::-1]
        r2c = itertools.accumulate(r2c, compose_homographies)
        # Merge matches
        matches = combine_matches(itertools.chain(
            (m.matches() for m in l2c_matches),
            ((p[::-1] for p in m.matches()) for m in r2c_matches),
        ))
        output = [*l2c, get_identity_homography(), *r2c], matches

def combine_matches(match_sets):
    """Given an iterable of iterables of pairs, return a list of lists
    corresponding to "match chains".  (XXX improve wording)

    """
    DEFAULT = None
    curr, result = {}, []
    for i, matches in enumerate(itertools.chain(match_sets, [[]])):
        curr, old = {}, curr
        for x, y in matches:
            try:
                chain = old.pop(x)
            except KeyError:
                chain = i * [DEFAULT]
                chain.append(x)
            chain.append(y)
            curr[y] = chain
        for chain in result:
            chain.append(DEFAULT)
        result += old.values()
    return result

@Transformer.decorate
def estimate_homography(match_features, compute_ref_homography):
    """Create a Transformer that estimates homographies using features and
    descriptors.  Arguments:
    - match_features should be kwiver.vital.algo.MatchFeatures.match
      (bound) or a similar callable
    - compute_ref_homography should be
      kwiver.vital.algo.ComputeRefHomography.estimate (bound) or a
      similar callable

    The .step call expects two arguments:
    - a kvt.FeatureSet
    - the corresponding kvt.DescriptorSet
    and returns:
    - a kvt.F2FHomography mapping the current coordinates to those in
      a reference frame

    """
    frame_id = track_id = 0
    fts = kvt.FeatureTrackSet()

    def step_fts(features, descriptors):
        """Update fts with the provided features and corresponding
        descriptors

        """
        nonlocal track_id
        atl = [] if frame_id == 0 else fts.active_tracks(frame_id - 1)
        atsl = [t[frame_id - 1] for t in atl]
        afs = kvt.SimpleFeatureSet([ts.feature for ts in atsl])
        ads = kvt.DescriptorSet([ts.descriptor for ts in atsl])
        m = match_features(afs, ads, features, descriptors)
        if m is None:
            m = kvt.MatchSet()
        ftsl = [kvt.FeatureTrackState(frame_id, *fd) for fd in zip(
            features.features(), descriptors.descriptors(),
        )]
        matched = set()
        for ai, i in m.matches():
            matched.add(i)
            atl[ai].append(ftsl[i])
        for i, ts in enumerate(ftsl):
            if i not in matched:
                t = kvt.Track(track_id)
                track_id += 1
                t.append(ts)
                fts.insert(t)

    output = None
    while True:
        features, descriptors = yield output
        step_fts(features, descriptors)
        output = compute_ref_homography(frame_id, fts)
        frame_id += 1

def get_identity_homography():
    """Return a homography representing the identity transformation"""
    return kvt.HomographyD()

class SingleHomographyEstimator:
    __slots__ = '_match_features', '_estimate_homography'
    def __init__(self, match_features, estimate_homography):
        """Initialize an instance from callables with signatures comparable to
        the (bound) methods kwiver.vital.algo.MatchFeatures.match and
        kwiver.vital.algo.EstimateHomography.estimate, respectively.

        """
        self._match_features = match_features
        self._estimate_homography = estimate_homography

    def __call__(self,
                 source_features, source_descriptors,
                 target_features, target_descriptors):
        """Return a kvt.BaseHomography that converts coordinates of the source
        to those of the target and a kvt.BaseMatchSet of the
        corresponding feature points, estimated using the provided
        feature and descriptor sets.

        (XXX What happens when estimation fails?)

        """
        matches = self._match_features(
            source_features, source_descriptors,
            target_features, target_descriptors,
        )
        homog, inliers = self._estimate_homography(
            source_features, target_features, matches,
        )
        inlier_matches = kvt.MatchSet([
            m for i, m in zip(inliers, matches.matches()) if i
        ])
        return homog, inlier_matches

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
    """Return a pair of the given kvt.FeatureSet and kvt.DescriptorSet, warped
    according to the provided kv.BaseHomography.

    (Note that in fact only the feature locations are different.)

    """
    warped_features = []
    for f in features.features():
        wf = f.clone()
        wf.location = homog.map(wf.location)
        warped_features.append(wf)
    return kvt.SimpleFeatureSet(warped_features), descriptors

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
