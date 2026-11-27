# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
align_cameras: multi-image-pair camera registration process.

Consumes N synchronized image streams (via their ``file_name{i}`` ports)
and computes one homography per camera pair, pooled over many image pairs,
using the MINIMA-LoFTR cross-spectral matcher from
viame.core.alignment_core. The result is written as a DIVE camera
registration JSON (format version 2, ``observations`` carrying per-frame
provenance) at pipeline completion.

Frame choice belongs to the caller: every row received is matched. The
streaming phase (_step) only records file paths; at end of stream the
matcher runs once per frame per configured pair, a pooled MAGSAC fit is
computed per pair over every observation that passes the per-frame gates
(min_matches, min_inliers, min_inlier_ratio), and for a fully-connected
triplet a loop-closure residual (H13 vs H23.H12) is reported. The gates
are what reject a featureless frame, and they do so per pair: a frame
that is blank in one camera costs only the pairs that include it. Rejected
frames are recorded, disabled, with the matcher's reason -- one
blank-ocean frame must not kill the job.

The images are read with cv2 from the file_name ports rather than from
the decoded ``image{i}`` ports (declared optional, left unconnected in
the shipped pipes). The common input includes configure
``image_reader:vxl:force_byte true`` with ``auto_stretch`` left at its
default false, which narrows 16-bit thermal to 8 bits by a plain
truncating cast, whereas the alignment core applies a percentile
stretch. Because nothing consumes the decoded port, the shipped pipes
override the shared input with ``image_list:disable_image_load true``,
so the reader does not decode every frame only for the result to be
dropped at the unconnected port.
"""

import json
import os
import sys
import time

import numpy as np

from kwiver.sprokit.pipeline import datum
from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

from .simple_homog_tracker import add_declare_config
from .stabilize_many_images import add_declare_input_port


def _log(message):
    print("align_cameras: %s" % message, flush=True)


class AlignCamerasProcess(KwiverProcess):
    """Pooled multi-frame camera-pair registration (MINIMA-LoFTR)."""

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of camera inputs')
        add_declare_config(
            self, 'pairs', '',
            'Which camera pairs to match, as "1-2,1-3,2-3" (1-based input '
            'indices). Default: all pairs.')
        add_declare_config(
            self, 'camera_names', '',
            'Comma-separated camera names aligned with input indices, used '
            'in the output JSON. Default: cam1..camN.')
        add_declare_config(self, 'output_directory', './',
                           'Directory for the output JSON')
        add_declare_config(self, 'output_json_file', 'registration.json',
                           'Output registration JSON filename')
        add_declare_config(
            self, 'weights_path', '',
            'Path to minima_loftr.ckpt; empty locates it in the install')
        add_declare_config(self, 'device', 'auto',
                           'Matcher device (auto, cuda, cuda:N, mps, cpu)')
        add_declare_config(self, 'transform_type', 'homography',
                           'Transform type recorded in the output')
        add_declare_config(self, 'ransac_threshold', '2.0',
                           'RANSAC reprojection threshold in '
                           'matcher-resolution pixels')
        add_declare_config(self, 'min_matches', '100',
                           'Per-frame gate: minimum raw matches')
        add_declare_config(self, 'min_inliers', '30',
                           'Per-frame gate: minimum RANSAC inliers')
        add_declare_config(self, 'min_inlier_ratio', '0.15',
                           'Per-frame gate: minimum inlier ratio')
        add_declare_config(self, 'top_k', '24',
                           'Spatially-spread inliers kept per frame')
        add_declare_config(self, 'match_threshold', '0.2',
                           'LoFTR coarse match confidence threshold')
        add_declare_config(
            self, 'lower_percentile', '2.0',
            'Percentile stretch lower bound. Consulted ONLY for '
            'high-bit-depth inputs; ignored outright for 8-bit.')
        add_declare_config(
            self, 'upper_percentile', '98.0',
            'Percentile stretch upper bound. Consulted ONLY for '
            'high-bit-depth inputs; ignored outright for 8-bit.')

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('timestamp', optional)

        # XXX work around insufficient wrapping: config_value is read here,
        # before _configure, which is what makes dynamic port counts possible.
        self._n_input = int(self.config_value('n_input'))
        for i in range(1, self._n_input + 1):
            add_declare_input_port(
                self, 'file_name%d' % i, 'file_name', required,
                'Input image file name #%d' % i)
            add_declare_input_port(
                self, 'image%d' % i, 'image', optional,
                'Unused; the process reads image files itself to preserve '
                'native bit depth (#%d)' % i)
            # Optional per-camera stretch override for a rig with two
            # dissimilar high-bit-depth sensors; falls back to the shared
            # lower/upper_percentile pair. Pipe-level config, not surfaced
            # in the run dialog.
            add_declare_config(
                self, 'cam%d_lower_percentile' % i, '',
                'Per-camera override of lower_percentile for input #%d' % i)
            add_declare_config(
                self, 'cam%d_upper_percentile' % i, '',
                'Per-camera override of upper_percentile for input #%d' % i)

    # ------------------------------------------------------------ configure
    def _configure(self):
        self._n_input = int(self.config_value('n_input'))
        self._output_directory = self.config_value('output_directory')
        self._output_json_file = self.config_value('output_json_file')
        self._weights_path = self.config_value('weights_path') or None
        self._device = self.config_value('device') or 'auto'
        self._transform_type = self.config_value('transform_type')

        names = [
            n.strip() for n in self.config_value('camera_names').split(',')
            if n.strip()
        ]
        if names and len(names) != self._n_input:
            raise ValueError(
                'camera_names has %d entries for %d inputs'
                % (len(names), self._n_input))
        self._camera_names = names or [
            'cam%d' % i for i in range(1, self._n_input + 1)]

        pairs_cfg = self.config_value('pairs').strip()
        if pairs_cfg:
            pairs = []
            for token in pairs_cfg.split(','):
                a, _, b = token.strip().partition('-')
                i, j = int(a), int(b)
                if not (1 <= i <= self._n_input and 1 <= j <= self._n_input) \
                        or i == j:
                    raise ValueError('Bad camera pair: %r' % token)
                pairs.append((min(i, j) - 1, max(i, j) - 1))
            self._pairs = sorted(set(pairs))
        else:
            self._pairs = [
                (i, j)
                for i in range(self._n_input)
                for j in range(i + 1, self._n_input)
            ]

        shared_lo = float(self.config_value('lower_percentile'))
        shared_hi = float(self.config_value('upper_percentile'))
        self._percentiles = []
        for i in range(1, self._n_input + 1):
            lo = self.config_value('cam%d_lower_percentile' % i).strip()
            hi = self.config_value('cam%d_upper_percentile' % i).strip()
            self._percentiles.append((
                float(lo) if lo else shared_lo,
                float(hi) if hi else shared_hi,
            ))

        self._match_options = {
            'ransac_threshold': float(self.config_value('ransac_threshold')),
            'min_matches': int(self.config_value('min_matches')),
            'min_inliers': int(self.config_value('min_inliers')),
            'min_inlier_ratio': float(self.config_value('min_inlier_ratio')),
            'top_k': int(self.config_value('top_k')),
            'match_threshold': float(self.config_value('match_threshold')),
        }

        # One entry per input row: the file path from each camera.
        self._frames = []
        self._finalized = False
        self._warned_bare_name = False

        self._base_configure()

    # ----------------------------------------------------------------- step
    def _step(self):
        paths = [
            str(self.grab_input_using_trait('file_name%d' % i))
            for i in range(1, self._n_input + 1)
        ]
        if self.has_input_port_edge_using_trait('timestamp'):
            self.grab_input_using_trait('timestamp')

        self._frames.append(paths)

        # The upstream readers queue their complete datum immediately after
        # the last frame, so peeking after processing this row tells us
        # whether it was the final one -- the same trick
        # accumulate_image_statistics uses. Intercepting it here (rather
        # than letting the framework auto-complete us) is what gives the
        # process its finalize step.
        if self.peek_at_datum_on_port('file_name1').type() \
                == datum.DatumType.complete:
            try:
                self._finalize()
            finally:
                self.mark_process_as_complete()
            return

        self._base_step()

    def _warn_if_bare_names(self, paths):
        """Name the likely cause when every path arrives as a bare basename.

        The process opens these paths itself, but the file_name port is
        stripped of its directory unless the input sets no_path_in_name
        false, and DIVE runs pipelines from the job output directory
        rather than the image directory. The resulting failure is a wall
        of 'unreadable' skips that says nothing about why. Warn once.
        """
        if self._warned_bare_name:
            return
        bare = [
            path for path in paths
            if not os.path.dirname(path) and not os.path.exists(path)
        ]
        if not bare:
            return
        self._warned_bare_name = True
        _log('%s has no directory component and does not resolve against the '
             'working directory (%s). align_cameras opens these paths itself, '
             'so the input needs ":no_path_in_name false".'
             % (bare[0], os.getcwd()))

    # ------------------------------------------------------------- finalize
    def _finalize(self):
        from viame.core import alignment_core

        if self._finalized:
            return
        self._finalized = True

        _log('matching %d frame(s) across %d pair(s)' % (
            len(self._frames), len(self._pairs)))

        matcher = alignment_core.LoftrMatcher(
            weights_path=self._weights_path,
            device=self._device,
            log=lambda message: _log(message),
        )

        pairs_json = []
        homographies = {}
        sizes_a = {}
        try:
            for p, (ci, cj) in enumerate(self._pairs):
                pairs_json.append(self._register_pair(
                    alignment_core, matcher, p, ci, cj,
                    homographies, sizes_a))
        finally:
            matcher.unload()

        output = {
            'type': 'dive-camera-registration',
            'version': 2,
            'source': {
                'model': 'minima_loftr',
                'generated': time.strftime(
                    '%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'frames': len(self._frames),
            },
            'pairs': pairs_json,
        }

        # Loop closure: with all three pairs of a triplet solved, compare
        # the direct H13 against the H23 . H12 route. This is the number
        # that says the three-camera solution is mutually consistent
        # rather than three independently plausible fits.
        loop_pairs = [(0, 1), (1, 2), (0, 2)]
        if self._n_input >= 3 \
                and all(pair in homographies for pair in loop_pairs):
            residual = alignment_core.loop_closure_residual(
                homographies[(0, 1)], homographies[(1, 2)],
                homographies[(0, 2)], sizes_a[(0, 1)])
            output['loopClosure'] = {
                'meanPx': residual['mean_px'],
                'maxPx': residual['max_px'],
            }
            _log('loop closure: mean %.2f px, max %.2f px' % (
                residual['mean_px'], residual['max_px']))

        # Write atomically: a canceled job must yield no file rather than
        # a plausible-looking half-written calibration.
        out_path = os.path.join(
            self._output_directory, self._output_json_file)
        os.makedirs(self._output_directory or '.', exist_ok=True)
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(output, f, indent=2)
        os.replace(tmp_path, out_path)
        _log('wrote %s' % out_path)

    def _register_pair(self, alignment_core, matcher, p, ci, cj,
                       homographies, sizes_a):
        left, right = self._camera_names[ci], self._camera_names[cj]
        lo_a, hi_a = self._percentiles[ci]
        lo_b, hi_b = self._percentiles[cj]
        options = dict(self._match_options)
        options.update({
            'lower_percentile': lo_a,
            'upper_percentile': hi_a,
            'cam2_lower_percentile': lo_b,
            'cam2_upper_percentile': hi_b,
        })

        observations = []
        fit_inputs = []
        total = len(self._frames)
        for k, paths in enumerate(self._frames):
            path_a, path_b = paths[ci], paths[cj]
            _log('matching frame %d/%d, pair %d/%d (%s <-> %s)' % (
                k + 1, total, p + 1, len(self._pairs), left, right))
            try:
                result = alignment_core.register_image_pair(
                    matcher, path_a, path_b, options)
            except Exception as e:
                _log('frame %d/%d: matcher error: %s' % (k + 1, total, e))
                self._warn_if_bare_names([path_a, path_b])
                result = {'success': False, 'code': 'error', 'error': str(e)}
            observation = {
                'imageLeft': os.path.basename(path_a),
                'imageRight': os.path.basename(path_b),
                'source': 'minima_loftr',
            }
            if result.get('success'):
                observation.update({
                    'enabled': True,
                    'points': result['inliers'],
                    'stats': {
                        'numMatches': result['num_matches'],
                        'numInliers': result['num_inliers'],
                        'inlierRatio': result['inlier_ratio'],
                        'coverage': result['coverage'],
                    },
                })
                fit_inputs.append((observation, {
                    'points': result['inliers'],
                    'size_a': tuple(result['image_size_a']),
                    'size_b': tuple(result['image_size_b']),
                }))
            else:
                observation.update({
                    'enabled': False,
                    'points': [],
                    'stats': {
                        'skipped': result.get('code', 'error'),
                    },
                })
                _log('frame %d/%d: %s <-> %s rejected (%s)' % (
                    k + 1, total, left, right, result.get('code', 'error')))
            observations.append(observation)

        pair_json = {
            'left': left,
            'right': right,
            'transformType': self._transform_type,
            'observations': observations,
        }

        fit = alignment_core.fit_pooled_homography(
            [pooled for _, pooled in fit_inputs], self._match_options)
        if fit.get('success'):
            H = np.asarray(fit['homography'], np.float64)
            homographies[(ci, cj)] = H
            sizes_a[(ci, cj)] = fit_inputs[0][1]['size_a']
            pair_json['leftToRight'] = fit['homography']
            H_inv = np.linalg.inv(H)
            pair_json['rightToLeft'] = (H_inv / H_inv[2, 2]).tolist()
            pair_json['stats'] = {
                'numPoints': fit['num_points'],
                'numInliers': fit['num_inliers'],
                'rmsPx': fit['rms_px'],
            }
            for (observation, _), obs_stats \
                    in zip(fit_inputs, fit['observations']):
                observation['stats']['rmsPx'] = obs_stats['rms_px']
            _log('pair %s <-> %s: %d points, %d inliers, rms %.2f px' % (
                left, right, fit['num_points'], fit['num_inliers'],
                fit['rms_px']))
        else:
            pair_json['stats'] = {
                'error': fit.get('code', 'error'),
                'numPoints': fit.get('num_points', 0),
            }
            _log('pair %s <-> %s: pooled fit failed (%s)' % (
                left, right, fit.get('code', 'error')))
        return pair_json


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.AlignCamerasProcess'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'align_cameras',
        'Multi-image-pair camera-to-camera registration (MINIMA-LoFTR)',
        AlignCamerasProcess,
    )
    process_factory.mark_process_module_as_loaded(module_name)
