# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""colmap_registration - in-pipeline multi-camera registration node.

Drop-in replacement for the ``many_image_stabilizer`` process on sea-lion
surveys: identical ports (``image1..N`` in, ``homog1..N`` out, plus a
``file_name1..N`` key), so the multicam suppressor / tracker / homography
writer downstream are unchanged. The homographies are computed by the newer
affine-chain + rig cross-camera + GPS geo-anchoring registration
(viame.opencv.prior_coverage_opencv), which optionally uses an external
flight-log metadata file when one is available and falls back to pure
image registration when it is not.

Because that registration is a whole-survey batch operation (it needs every
frame to build the chains and the shared reference), the node registers once on
the first step, caches the per-frame homographies to disk, and then simply
emits the precomputed source-to-reference homography for whichever frame is
flowing through on each step. Frames dropped by an upstream downsampler are
handled naturally: the emitted homography is looked up by image filename, not
by counting steps.

What gets registered is controlled by register_scope: "list" (default)
registers only the frames the pipeline is given (the union of the per-camera
image_list<i> files, each defaulting to the pipeline input list), so a subset
drawn from a larger or mixed folder is registered on its own; "folder" registers
the whole survey folder for the full-survey geometry. The survey folder itself
is resolved from site_folder or image_list1 (the streamed file_name port carries
only basenames, so the folder cannot be recovered from it).

The reference frame is local ENU metres (shared across the rig cameras, so
their relative mapping is exact) when GPS/flight-log metadata is available,
else pseudo-ENU from the registration chains. Either way the absolute
reference cancels in the cross-camera differences the suppressor/tracker take,
and image sizes come from the images, so the units do not matter downstream.
"""

import hashlib
import os
import sys

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
import kwiver.vital.types as kvt

_IDENTITY = np.eye(3, dtype=np.float64)

# Bump whenever the registration algorithm changes the emitted homographies
# (feeds the disk-cache key so stale caches self-invalidate).
_REG_ALGO_VERSION = 2


def _log(msg):
    """Progress to stderr so it shows in the pipeline log (the vital logger's
    INFO level is not surfaced by kwiver runner)."""
    print('[colmap_registration] ' + msg, file=sys.stderr, flush=True)


def _add_declare_config(proc, name, default, desc):
    proc.add_config_trait(name, name, default, desc)
    proc.declare_config_using_trait(name)


def _add_declare_input_port(proc, name, type_trait, flag, desc):
    proc.add_port_trait(name, type_trait, desc)
    proc.declare_input_port_using_trait(name, flag)


def _add_declare_output_port(proc, name, type_trait, flag, desc):
    proc.add_port_trait(name, type_trait, desc)
    proc.declare_output_port_using_trait(name, flag)


def _f2f(matrix, from_id, to_id):
    """Wrap a 3x3 numpy matrix as a Kwiver source-to-ref F2FHomography."""
    return kvt.F2FHomography(kvt.HomographyD(np.ascontiguousarray(
        matrix, dtype=np.float64)), int(from_id), int(to_id))


class ColmapRegistration(KwiverProcess):
    """Multi-camera registration node backed by prior_coverage_opencv."""

    # to_id of every emitted homography: a single shared reference frame, so
    # all cameras (and all timesteps) live in one coordinate system and the
    # suppressor/tracker differences are valid.
    _REF_ID = 0

    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        _add_declare_config(self, 'n_input', '2', 'Number of camera inputs')
        _add_declare_config(
            self, 'flight_log', '',
            'Optional flight-log CSV or directory of daily FMCLOG CSVs. Empty '
            '= no external metadata (registration falls back to image-only '
            'pseudo-georeferencing).')
        _add_declare_config(
            self, 'method', 'hybrid',
            'Registration method: "hybrid" (affine chains + rig cross-camera '
            'consensus + GPS geo-anchoring) or "metadata" (GPS footprints '
            'only, no image registration).')
        _add_declare_config(
            self, 'water_method', 'auto',
            'Water/land classifier for the hybrid method (auto|svm|sift).')
        _add_declare_config(
            self, 'chain_anchored_placement', 'true',
            'Feature-primary placement (hybrid method): anchor each camera '
            'chain to ENU with one similarity fitted over its GPS fixes, so '
            'frame-to-frame geometry comes from image features instead of '
            'per-frame GPS+heading dead reckoning. Falls back to per-frame '
            'GPS placement when the fit is untrustworthy (few chained frames, '
            'low GPS spread, or high fit residual). Set false for the prior '
            'per-frame GPS placement.')
        _add_declare_config(
            self, 'gps_chain_reconcile', 'true',
            'Correct per-frame GPS positions that disagree with the image '
            'registration chain before placing frames (hybrid method). Fixes '
            'the periodic misregistration from sub-second trigger / GPS-sample '
            'aliasing (a frame whose logged GPS step is short/long while the '
            'imagery shows steady motion). Every correction must be confirmed '
            'by an independent high-quality match of the two frames, so frames '
            'are never moved without direct image evidence. Set false to place '
            'frames at raw GPS as before.')
        _add_declare_config(
            self, 'export_homogs', '',
            'Optional path to write the COMPLETE per-frame homography map '
            '(basename -> 3x3, npz) as soon as registration finishes. Because '
            'registration is a whole-survey batch, this gives downstream '
            'streaming processes (e.g. the suppressor boundary-cutoff check) '
            'access to FUTURE frames\' homographies. Empty = disabled.')
        _add_declare_config(
            self, 'cache', '',
            'Explicit path to the full-folder registration cache file. Empty = '
            '<site>/VIAME/registration_<method>.npz (see use_cache). The cache '
            'stores an algorithm signature (method, flight log, full folder '
            'listing) and is reused only while that still matches.')
        _add_declare_config(
            self, 'site_folder', '',
            'Survey folder containing the per-camera image subdirectories. '
            'Empty = derive it from the first entry of "image_list1" (the '
            'streamed file_name port only carries basenames, so the folder '
            'cannot be recovered from it).')
        _add_declare_config(
            self, 'register_scope', 'list',
            'What to register: "list" (default) registers only the frames the '
            'pipeline is given (the image_list<i> files) - so a subset drawn '
            'from a larger or mixed folder is registered on its own. "folder" '
            'registers the whole survey folder - best when you want the '
            'full-survey geometry (more frames, all cross-camera/loop overlap). '
            'Chains still need a reasonably contiguous per-camera run to '
            'register well.')
        _add_declare_config(
            self, 'use_cache', 'true',
            'Persist/reuse a full-folder registration in <site>/VIAME/ '
            '(alongside the camera folders) when the data folder is writable. '
            'A whole-folder run writes registration_<method>.npz there; any '
            'run - including a list-scope one - reuses it when its stored '
            'algorithm signature still matches, since the full-survey geometry '
            'covers a subset at higher quality. Set false to skip the cache.')
        _add_declare_config(
            self, 'require_metadata', 'false',
            'Fail the pipeline if no metadata file (flight_log) was provided to '
            'the registration. Default false: image-only registration is allowed.')
        _add_declare_config(
            self, 'require_metadata_match', 'false',
            'Fail the pipeline if the provided metadata does not correspond to '
            'the current image sequence - fewer than min_metadata_coverage of '
            'its frames get a pose from the metadata file. Default false.')
        _add_declare_config(
            self, 'min_metadata_coverage', '1.0',
            'When require_metadata_match is set, the minimum fraction (0..1) of '
            'the image sequence that the metadata must cover. Default 1.0 (every '
            'frame).')

        self._n_input = int(self.config_value('n_input'))
        # One image list per camera: each a single file of line-separated image
        # paths (never a comma-separated list). Camera 1's list also locates the
        # survey folder when site_folder is empty. Defaults keep the pipeline
        # input lists working with no override.
        for i in range(1, self._n_input + 1):
            # Single-camera pipelines feed input_list.txt; multi-camera ones feed
            # input_list_1.txt, input_list_2.txt, ... Match those names so the
            # default works with no override (and DIVE's -s can still override,
            # since nothing is set in the .pipe file).
            default_list = ('input_list.txt' if self._n_input == 1
                            else 'input_list_' + str(i) + '.txt')
            _add_declare_config(
                self, 'image_list' + str(i), default_list,
                'Image-list file for camera ' + str(i) + ': a single file of '
                'line-separated image paths (not comma-separated).'
                + (' Its first entry also locates the survey folder when '
                   'site_folder is empty.' if i == 1 else ''))
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)
        for i in range(1, self._n_input + 1):
            _add_declare_input_port(
                self, 'image' + str(i), 'image', required,
                'Input image #' + str(i) + ' (drives one step)')
            _add_declare_input_port(
                self, 'file_name' + str(i), 'file_name', required,
                'Input image path #' + str(i) + ' (homography lookup key)')
            _add_declare_output_port(
                self, 'homog' + str(i), 'homography_src_to_ref', optional,
                'Output homography (source-to-ref) #' + str(i))

    def _configure(self):
        self._n_input = int(self.config_value('n_input'))
        self._flight_log = self.config_value('flight_log') or None
        self._method = self.config_value('method') or 'hybrid'
        self._water_method = self.config_value('water_method') or 'auto'
        self._gps_chain_reconcile = (self.config_value('gps_chain_reconcile')
                                     or 'true').lower() in (
                                         'true', '1', 'yes', 'on')
        self._chain_anchored = (self.config_value('chain_anchored_placement')
                                or 'true').lower() in (
                                    'true', '1', 'yes', 'on')
        self._cache = self.config_value('cache') or None
        self._export_homogs = self.config_value('export_homogs') or None
        self._site_folder = self.config_value('site_folder') or None
        # One image-list file per camera (single file each, line-separated).
        self._image_lists = [
            self.config_value('image_list' + str(i)) or None
            for i in range(1, self._n_input + 1)
        ]
        self._scope = (self.config_value('register_scope') or 'list').lower()
        self._use_cache = (self.config_value('use_cache') or 'true').lower() \
            not in ('false', '0', 'no', 'off')
        _truthy = ('true', '1', 'yes', 'on')
        self._require_metadata = (self.config_value('require_metadata')
                                  or 'false').lower() in _truthy
        self._require_metadata_match = (self.config_value('require_metadata_match')
                                        or 'false').lower() in _truthy
        try:
            self._min_metadata_coverage = float(
                self.config_value('min_metadata_coverage') or '1.0')
        except ValueError:
            self._min_metadata_coverage = 1.0
        self._by_name = None                 # basename -> 3x3 or None
        self._frame = 0
        self._last = [None] * self._n_input  # last good homog matrix per cam
        self._warned_missing = 0
        self._base_configure()

    def _resolve_site(self, hint_name=None):
        """Survey folder, in precedence order: explicit site_folder config;
        the first full image path in the image_list file; or the streamed
        file_name itself when it is a full path (set the input reader's
        no_path_in_name=false to feed full paths over the port, so the folder
        need not be named again for the node)."""
        if self._site_folder:
            return self._site_folder
        # Camera 1's image list locates the survey folder from its first full
        # image path (see _survey_folder_of for the rig-vs-single-camera rule).
        lst = self._image_lists[0] if self._image_lists else None
        if lst and os.path.exists(lst):
            with open(lst) as f:
                for line in f:
                    p = line.strip()
                    if p:
                        return self._survey_folder_of(p)
        if hint_name and os.path.dirname(str(hint_name)):
            return self._survey_folder_of(str(hint_name))
        return None

    @staticmethod
    def _survey_folder_of(image_path):
        """Survey folder that holds an image. A multi-camera rig lays images out
        as <survey>/<CENTER|PORT|STAR>/<image> (survey is two dirs up); a single-
        camera UAS survey puts images directly in <survey>/ (one dir up, and the
        imagelog.json sits in that same folder). Only strip the camera level when
        the image's parent really is a rig camera folder, so single-camera
        surveys resolve to the folder that actually holds their metadata."""
        parent = os.path.dirname(os.path.abspath(image_path))
        if os.path.basename(parent).upper() in ('CENTER', 'PORT', 'STAR'):
            return os.path.dirname(parent)
        return parent

    def _resolve_images(self):
        """Explicit image paths to register (register_scope=list), or None for
        whole-folder registration. Frames are the union of the per-camera
        image_list<i> files (one single-file, line-separated list per camera)."""
        if self._scope != 'list':
            return None
        images = []
        for lf in self._image_lists:
            if not lf:
                continue
            if not os.path.exists(lf):
                _log('image list file not found: %s' % lf)
                continue
            with open(lf) as f:
                images += [ln.strip() for ln in f if ln.strip()]
        return images or None

    def _metadata_coverage(self, site_folder, images):
        """(matched, total): frames that get a pose FROM THE PROVIDED metadata
        file - a flight-log CSV row matched by frame number, or an imagelog JSON
        record matched to the image by GPS position. EXIF-only GPS is not
        counted: this measures whether the file itself was applied, so
        require_metadata can fail when a file was passed but is the wrong log,
        unreadable, or left unlinked by a survey-folder mismatch."""
        from viame.core import survey_metadata as smd
        fl = self._flight_log
        cams = smd.list_site_images(site_folder, image_list=images)
        total = sum(len(v) for v in cams.values())
        if not fl or not os.path.exists(fl):
            return 0, total
        if fl.lower().endswith('.json'):
            # Single-camera imagelog: only count GPS-position links; an order
            # fallback means the file did not actually correspond to the frames.
            rels = cams.get(None, [])
            recs = smd.load_imagelog(fl)
            if not recs or not rels:
                return 0, total
            _links, stats = smd.link_imagelog(
                site_folder, rels, recs, read_exif=True)
            return stats.get('by_position', 0), total
        # Flight-log CSV: count images whose per-day frame number has a log row.
        date = smd.folder_date(site_folder)
        log_path = smd.find_flight_log(fl, date)
        log = smd.load_flight_log(log_path) if log_path else {}
        if not log:
            return 0, total
        matched = 0
        for rels in cams.values():
            for r in rels:
                fr = smd.parse_image_filename(r).get('frame')
                if fr is not None and fr in log:
                    matched += 1
        return matched, total

    def _enforce_metadata_requirements(self, site_folder, images):
        """Raise if require_metadata / require_metadata_match are set and the
        metadata is missing, unreadable, or does not actually apply to the
        frames. Unlike a plain "a path was passed" check, require_metadata now
        verifies the file is USED: at least one frame must link to it (a
        flight-log row by frame number, or a GPS-position imagelog match). A
        wrong, unreadable, or unresolved metadata file therefore fails loudly
        here instead of silently registering nothing (all-identity homographies
        that read downstream as "every frame already observed")."""
        if not (self._require_metadata or self._require_metadata_match):
            return
        if self._require_metadata and not self._flight_log:
            raise RuntimeError(
                'colmap_registration: require_metadata is set but no metadata '
                'file was provided (set stabilizer:flight_log to a .csv flight '
                'log or .json imagelog).')
        if self._flight_log and not os.path.exists(self._flight_log):
            raise RuntimeError(
                'colmap_registration: metadata file not found: %s'
                % self._flight_log)
        matched, total = self._metadata_coverage(site_folder, images)
        if total == 0:
            return
        if matched == 0:
            raise RuntimeError(
                'colmap_registration: the provided metadata file (%s) could not '
                'be applied to any of the %d frames. Check that it is the '
                'correct flight log / imagelog for this sequence and that the '
                'survey folder resolved correctly (resolved site_folder=%r).'
                % (self._flight_log, total, site_folder))
        # require_metadata alone only needs the file to be used at all; the
        # stricter fractional coverage threshold applies under require_metadata_match.
        min_cov = (self._min_metadata_coverage
                   if self._require_metadata_match else 0.0)
        if matched < total * min_cov:
            raise RuntimeError(
                'colmap_registration: the metadata covers only %d/%d frames of '
                'the image sequence (require >= %.0f%%). It likely does not '
                'correspond to this sequence.'
                % (matched, total, 100.0 * self._min_metadata_coverage))

    # ---- one-time batch registration -------------------------------------

    def _cache_key(self, site_folder, images):
        # Key on the canonical camera image listing (via list_site_images,
        # which sees only the CENTER/PORT/STAR frames or root images) - never
        # the whole site tree, so the VIAME cache dir does not perturb its own
        # key. The folder cache is always keyed on the FULL folder (images=None)
        # so a list-scope run validates against it too.
        from viame.core import survey_metadata as smd
        cams = smd.list_site_images(site_folder, image_list=images)
        names = sorted(os.path.basename(r)
                       for rels in cams.values() for r in rels)
        fl = self._flight_log or ''
        have_fl = fl and os.path.exists(fl)
        fl_stamp = str(os.path.getmtime(fl)) if have_fl else ''
        # _REG_ALGO_VERSION invalidates cached registrations whenever the
        # placement algorithm changes (a cache written by an older algorithm
        # would otherwise be served silently forever). Bump on algorithm
        # changes that affect the emitted homographies.
        blob = '\n'.join(names) + (
            f'|{self._method}|{fl}|{fl_stamp}|{_REG_ALGO_VERSION}'
            f'|reconcile={self._gps_chain_reconcile}'
            f'|anchored={self._chain_anchored}')
        return hashlib.sha1(blob.encode('utf-8', 'replace')).hexdigest()

    @staticmethod
    def _writable(path):
        return os.path.isdir(path) and os.access(path, os.W_OK)

    def _folder_cache_path(self, site_folder, for_write=False):
        """Full-folder registration cache, kept next to the imagery in
        <site>/VIAME/registration_<method>.npz so it is easy to find and
        share. An explicit `cache` config overrides the location. Returns None
        if nothing suitable is writable (for_write) / present (read)."""
        if self._cache:
            return self._cache
        viame = os.path.join(site_folder, 'VIAME')
        if os.path.isdir(viame) or (for_write and self._writable(site_folder)):
            if for_write and not os.path.isdir(viame):
                try:
                    os.makedirs(viame, exist_ok=True)
                except OSError:
                    return None
            if not for_write or self._writable(viame):
                return os.path.join(viame,
                                    'registration_%s.npz' % self._method)
        return None

    def _load_cache(self, path, key):
        if not path or not os.path.exists(path):
            return None
        try:
            z = np.load(path, allow_pickle=True)
        except (OSError, ValueError):
            return None
        if str(z['key']) != key:       # algorithm signature changed -> stale
            return None
        mats = z['mats']
        return {str(n): (None if bool(m) else mats[i])
                for i, (n, m) in enumerate(zip(z['names'], z['is_none']))}

    def _save_cache(self, path, key, by_name):
        names = list(by_name)
        is_none = [by_name[n] is None for n in names]
        mats = np.stack([by_name[n] if by_name[n] is not None else _IDENTITY
                         for n in names]) if names else np.empty((0, 3, 3))
        try:
            np.savez(path, key=key, names=np.array(names),
                     is_none=np.array(is_none), mats=mats)
            _log('cached full-folder registration to %s' % path)
        except OSError as e:
            _log('could not write cache %s: %s' % (path, e))

    def _register(self, site_folder, images):
        from viame.opencv import prior_coverage_opencv as pcc
        # The full-folder cache is keyed and validated on the WHOLE folder
        # (images=None), so a list-scope run reuses it too when present.
        folder_key = self._cache_key(site_folder, None)
        if self._use_cache:
            cached = self._load_cache(
                self._folder_cache_path(site_folder), folder_key)
            if cached is not None:
                _log('reusing full-folder registration from %s (%d frames)'
                     % (self._folder_cache_path(site_folder), len(cached)))
                return cached
        scope = ('%d listed frames' % len(images) if images is not None
                 else 'whole folder')
        _log('registering %s [%s] (method=%s, flight_log=%s) - one-time, '
             'blocks the first frame and may be slow...'
             % (site_folder, scope, self._method, self._flight_log or 'none'))
        homogs = pcc.compute_frame_homographies(
            site_folder, flight_logs=self._flight_log, method=self._method,
            water_method=self._water_method, images=images,
            reg_overrides={'gps_chain_reconcile': self._gps_chain_reconcile,
                           'chain_anchored_placement': self._chain_anchored})
        by_name = {}
        n_geo = 0
        for rel, info in homogs.items():
            H = info['H']
            by_name[os.path.basename(rel)] = None if H is None else \
                np.asarray(H, dtype=np.float64)
            n_geo += H is not None
        _log('%d images, %d geo-referenced' % (len(by_name), n_geo))
        # Persist only whole-folder results (the reusable, higher-quality
        # geometry), and only where the data folder is writable.
        if self._use_cache and images is None:
            wpath = self._folder_cache_path(site_folder, for_write=True)
            if wpath:
                self._save_cache(wpath, folder_key, by_name)
        return by_name

    # ---- streaming --------------------------------------------------------

    def _homog_for(self, cam_idx, file_name):
        """Source-to-ref matrix for the frame arriving on camera cam_idx."""
        H = None
        if file_name:
            H = self._by_name.get(os.path.basename(str(file_name)))
        if H is None:
            # No transform for this frame: hold the camera's last good one so
            # it stays in the shared metric frame (identity only until the
            # first placement, i.e. an unregistered lead-in).
            if not self._warned_missing:
                _log('warning: no registered homography for %r (and possibly '
                     'others); holding the previous frame. Frames unplaced by '
                     'the registration fall back to identity/hold.'
                     % (os.path.basename(str(file_name)) if file_name else ''))
            self._warned_missing += 1
            H = self._last[cam_idx]
            if H is None:
                return _IDENTITY
        else:
            self._last[cam_idx] = H
        return H

    def _step(self):
        # The image ports are consumed only to advance in lockstep with the
        # stream (drop-in port compatibility with many_image_stabilizer); the
        # geometry comes from the precomputed registration, keyed by file_name.
        for i in range(1, self._n_input + 1):
            self.grab_input_using_trait('image' + str(i))
        names = [self.grab_input_using_trait('file_name' + str(i))
                 for i in range(1, self._n_input + 1)]
        if self._by_name is None:
            site = self._resolve_site(names[0] if names else None)
            if site is None or not os.path.isdir(site):
                # A required metadata file must fail loudly even when the survey
                # folder cannot be resolved (there is nothing to register against).
                if self._require_metadata and not self._flight_log:
                    raise RuntimeError(
                        'colmap_registration: require_metadata is set but no '
                        'metadata file was provided (set stabilizer:flight_log '
                        'to a .csv/.json flight log).')
                _log('could not resolve survey folder (site_folder=%r '
                     'image_list1=%r); emitting identity homographies'
                     % (self._site_folder,
                        self._image_lists[0] if self._image_lists else None))
                self._by_name = {}
            else:
                images = self._resolve_images()
                self._enforce_metadata_requirements(site, images)
                self._by_name = self._register(site, images)
            if self._export_homogs and self._by_name:
                try:
                    np.savez(self._export_homogs,
                             **{n: H for n, H in self._by_name.items()
                                if H is not None})
                    _log('exported %d frame homographies to %s' % (
                        sum(1 for H in self._by_name.values()
                            if H is not None), self._export_homogs))
                except OSError as e:
                    _log('could not export homographies: %s' % e)
        for i in range(self._n_input):
            H = self._homog_for(i, names[i])
            self.push_to_port_using_trait(
                'homog' + str(i + 1), _f2f(H, self._frame, self._REF_ID))
        self._frame += 1
        self._base_step()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.colmap.ColmapRegistration'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'colmap_registration',
        'Multi-camera survey registration (affine chains + rig cross-camera '
        'consensus + optional GPS metadata); drop-in for many_image_stabilizer',
        ColmapRegistration,
    )
    process_factory.mark_process_module_as_loaded(module_name)
