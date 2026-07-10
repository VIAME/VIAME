# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import itertools

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
import kwiver.vital.algo as kva
import kwiver.vital.types as kvt

# XXX These items should be defined somewhere else
from .simple_homog_tracker import add_declare_config, Transformer

@Transformer.decorate
def stabilize_many_images(
        compute_features_and_descriptors,
        match_features, estimate_single_homography,
        compute_ref_homography, close_loops=None,
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
    - close_loops should be kwiver.vital.algo.CloseLoops.stitch
      (bound) or a similar callable if supplied

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
    eh = estimate_homography(
        match_features, compute_ref_homography, close_loops,
    )
    output = None
    while True:
        images, = yield output
        fds = list(map(compute_features_and_descriptors, images))
        spatial_homogs, spatial_matches = ris.step(fds)
        valid = [i for i, h in enumerate(spatial_homogs) if h is not None]
        val_features, val_descs = merge_feature_and_descriptor_sets([
            warp_features_and_descriptors(h, *fd)
            for h, fd in zip(spatial_homogs, fds)
            if h is not None
        ], [[m[i] for i in valid] for m in spatial_matches])
        val_image = next((img for img, h in zip(images, spatial_homogs)
                          if h is not None), None)
        temporal_homog = eh.step(val_features, val_descs, val_image)
        # Default unknown cameras to be the same as their neighbors
        output = list(fill_nones(h and compose_homographies(temporal_homog, h)
                                 for h in spatial_homogs))

@Transformer.decorate
def register_image_set(estimate_single_homography):
    """Create a Transformer that registers sets of images (described
    by feature and descriptor sets) from a camera system together,
    maintaining information on previously estimated relative
    positions.  Arguments:
    - estimate_single_homography should be a SingleHomographyEstimator
      or similar callable

    The .step call expects one argument:
    - a non-empty list of kvt.FeatureSet--kvt.DescriptorSet pairs
      (adjacent images should overlap)
    and returns a tuple of:
    - a list of kvt.BaseHomography objects, one for each input image,
      that map them to some common coordinate space, or None for an
      unmappable image.  At least one element will not be None.
    - a list of lists, each with one index for each input image,
      indicating corresponding features across images.  (A value of
      None indicates a lack of a matching feature in the corresponding
      image)

    """
    def comp(x, y):
        """compose_homographies but returning None if either argument is"""
        return None if x is None or y is None else compose_homographies(x, y)
    prev_homogs = None
    output = None
    while True:
        fds, = yield output
        if prev_homogs is None:
            prev_homogs = len(fds) * [None]
        # The returned homographies will be relative to the "middle" one
        hl = len(fds) // 2
        # Compute homographies between adjacent images
        homogs, matches = [], []
        for i, lrfd in enumerate(zip(fds[:-1], fds[1:])):
            # Targets towards the middle
            sfd, tfd = lrfd if i < hl else lrfd[::-1]
            hm = estimate_single_homography(*sfd, *tfd)
            if hm is None:
                # Reuse the estimate from last frame, if any
                hm = prev_homogs[i], kvt.MatchSet()
            h, m = hm
            homogs.append(h)
            matches.append(m)
        # Compute homographies to center
        l2c, r2c = homogs[:hl], homogs[hl:]
        l2c = list(itertools.accumulate(reversed(l2c), comp))[::-1]
        r2c = itertools.accumulate(r2c, comp)
        # Merge matches
        matches = combine_matches(itertools.chain(
            (m.matches() for m in matches[:hl]),
            ((p[::-1] for p in m.matches()) for m in matches[hl:]),
        ))
        prev_homogs = homogs
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
def estimate_homography(
        match_features, compute_ref_homography, close_loops=None,
):
    """Create a Transformer that estimates homographies using features and
    descriptors.  Arguments:
    - match_features should be kwiver.vital.algo.MatchFeatures.match
      (bound) or a similar callable
    - compute_ref_homography should be
      kwiver.vital.algo.ComputeRefHomography.estimate (bound) or a
      similar callable
    - close_loops should be kwiver.vital.algo.CloseLoops.stitch
      (bound) or a similar callable if supplied

    The .step call expects two arguments:
    - a kvt.FeatureSet
    - the corresponding kvt.DescriptorSet
    and returns:
    - a kvt.F2FHomography mapping the current coordinates to those in
      a reference frame

    """
    frame_id = track_id = 0
    fts = kvt.FeatureTrackSet()

    def step_fts(features, descriptors, image=None):
        """Update fts with the provided features and corresponding
        descriptors

        """
        nonlocal fts, track_id
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
        if close_loops is not None:
            # Some CloseLoops implementations (e.g.
            # vxl_homography_guided) use the image argument.  Pass it
            # through when available; if None, implementations should
            # handle it gracefully.  The fourth argument, a mask, is
            # supposed to be optional but the wrapping is imperfect.
            #
            # Also, at least some implementations mutate the
            # FeatureTrackSet, though fortunately this is irrelevant
            # to us since fts doesn't escape and is used linearly.
            fts = close_loops(frame_id, fts, image, None)

    output = None
    while True:
        print(frame_id)
        features, descriptors, image = yield output
        step_fts(features, descriptors, image)
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

        If estimation fails, return None instead of the above pair.

        """
        matches = self._match_features(
            source_features, source_descriptors,
            target_features, target_descriptors,
        )
        hi = self._estimate_homography(
            source_features, target_features, matches,
        )
        if hi is None:
            return None
        homog, inliers = hi
        inlier_matches = kvt.MatchSet([
            m for i, m in zip(inliers, matches.matches()) if i
        ])
        vet_homography(homog, inlier_matches, source_features, target_features)
        # XXX MatchSet doesn't implemented __len__
        # XXX Hard-coded value
        if inlier_matches.size() < 10:
            return None
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

def merge_feature_and_descriptor_sets(fd_pairs, matches):
    """Merge a sequence of pairs of feature sets and corresponding
    descriptor sets into a single pair of one feature set and the
    corresponding descriptor set.

    Matches (given in an iterable as lists of indices (and None)) are
    removed except for a single copy.

    """
    fd_pairs = [(f.features(), d.descriptors()) for f, d in fd_pairs]
    features, descriptors = [], []
    consumed = set()
    for m in matches:
        it = ((i, j) for i, j in enumerate(m) if j is not None)
        i, j = next(it)
        # Just take the feature from the first available set
        fs, ds = fd_pairs[i]
        features.append(fs[j])
        descriptors.append(ds[j])
        consumed.update([(i, j)], it)
    for i, (fs, ds) in enumerate(fd_pairs):
        for j, (f, d) in enumerate(zip(fs, ds)):
            if (i, j) not in consumed:
                features.append(f)
                descriptors.append(d)
    return kvt.SimpleFeatureSet(features), kvt.DescriptorSet(descriptors)

def fill_nones(it):
    """Flood-fill Nones in the provided iterable"""
    it = iter(it)
    for i, val in enumerate(it, 1):
        if val is not None:
            break
    yield from itertools.repeat(val, i)
    for x in it:
        if x is not None:
            val = x
        yield val

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

def vet_homography(homog, inlier_matches, source_features, target_features):
    sf, tf = source_features.features(), target_features.features()
    sp, tp = [], []
    for si, ti in inlier_matches.matches():
        sp.append(sf[si].location)
        tp.append(tf[ti].location)
    return _vet_homography(homog.matrix(), sp, tp)

def _vet_homography(homog, source_points, target_points):
    import numpy as np
    def norm(h): return h / h[2, 2]
    def conform(ps):
        return np.concatenate((ps, np.ones((len(ps), 1))), 1).T
    def message(count, total, name):
        print(f'{count}/{total} {name} points transformed out-of-camera!')
    def vet(homog, ps, name):
        tps = norm(homog) @ conform(ps)
        c = np.count_nonzero(tps[2] <= 0)
        # XXX matches hard-coded value above
        if c or len(ps) < 10: message(c, len(ps), name)
    vet(homog, source_points, 'source')
    vet(np.linalg.inv(homog), target_points, 'target')

def add_declare_input_port(process, name, type, flag, desc):
    process.add_port_trait(name, type, desc)
    process.declare_input_port_using_trait(name, flag)

def add_declare_output_port(process, name, type, flag, desc):
    process.add_port_trait(name, type, desc)
    process.declare_output_port_using_trait(name, flag)

class ManyImageStabilizer(KwiverProcess):
    # Required algos.  There's also an optional
    # loop_closer=kva.CloseLoops that's handled specially.
    _REQUIRED_ALGOS = dict(
        feature_detector=kva.DetectFeatures,
        descriptor_extractor=kva.ExtractDescriptors,
        feature_matcher=kva.MatchFeatures,
        homography_estimator=kva.EstimateHomography,
        ref_homography_computer=kva.ComputeRefHomography,
    )

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')
        for k, v in self._REQUIRED_ALGOS.items():
            add_declare_config(self, k, '',
                               'Configuration for a nested ' + v.static_type_name())
        add_declare_config(self, 'loop_closer', '',
                           'Configuration for a nested close_loops (optional)')

        # XXX work around insufficient wrapping
        self._n_input = int(self.config_value('n_input'))
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)
        for i in range(1, self._n_input + 1):
            add_declare_input_port(self, 'image' + str(i), 'image',
                                   required, 'Input image #' + str(i))
            add_declare_output_port(self, 'homog' + str(i), 'homography_src_to_ref',
                                    optional, 'Output homography (source-to-ref) #' + str(i))

    def _configure(self):
        config = self.get_config()
        def snac(a, k): return a.set_nested_algo_configuration(k, config)
        algos = {k: snac(v, k) for k, v in self._REQUIRED_ALGOS.items()}
        assert None not in algos.values()
        loop_closer = snac(kva.CloseLoops, 'loop_closer')

        self._n_input = int(self.config_value('n_input'))
        self._stabilizer = stabilize_many_images(
            FeatureAndDescriptorComputer(
                algos['feature_detector'], algos['descriptor_extractor'],
            ),
            algos['feature_matcher'].match,
            algos['homography_estimator'].estimate,
            algos['ref_homography_computer'].estimate,
            None if loop_closer is None else loop_closer.stitch,
        )

        self._base_configure()

    def _step(self):
        homogs = self._stabilizer.step([
            self.grab_input_using_trait('image' + str(i))
            for i in range(1, self._n_input + 1)
        ])
        for i, h in enumerate(homogs, 1):
            self.push_to_port_using_trait('homog' + str(i), h)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.ManyImageStabilizer'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'many_image_stabilizer',
        'Simultaneous multi-image stabilization',
        ManyImageStabilizer,
    )
    process_factory.mark_process_module_as_loaded(module_name)
