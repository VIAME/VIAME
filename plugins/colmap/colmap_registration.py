# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""colmap_registration - in-pipeline multi-camera registration node.

Drop-in replacement for the ``many_image_stabilizer`` process on sea-lion
surveys: identical ports (``image1..N`` in, ``homog1..N`` out, plus a
``file_name1..N`` key), so the multicam suppressor / tracker / homography
writer downstream are unchanged. The homographies are computed by the newer
affine-chain + rig cross-camera + GPS geo-anchoring registration
(viame.opencv.prior_coverage_core), which optionally uses an external
flight-log metadata file when one is available and falls back to pure
image registration when it is not.

Because that registration is a whole-survey batch operation (it needs every
frame to build the chains and the shared reference), the node registers once on
the first step, caches the per-frame homographies to disk, and then simply
emits the precomputed source-to-reference homography for whichever frame is
flowing through on each step. Frames dropped by an upstream downsampler are
handled naturally: the emitted homography is looked up by image filename, not
by counting steps.

What gets registered is controlled by register_scope: "folder" (default)
registers the whole survey folder - best when the streamed frames are a
contiguous slice of one coherent survey; "list" registers only an explicit set
of frames (frame_list, defaulting to the pipeline input lists) - for a subset
drawn from a larger or mixed folder. The survey folder itself is resolved from
site_folder or the image_list (the streamed file_name port carries only
basenames, so the folder cannot be recovered from it).

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
    """Multi-camera registration node backed by prior_coverage_core."""

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
            self, 'cache', '',
            'Path to a homography cache file. Empty = '
            '<site>/.sea_lion_reg_cache_<key>.npz (the key covers the image '
            'set, scope, method and flight log, so distinct runs cache '
            'independently and are recomputed only when their inputs change).')
        _add_declare_config(
            self, 'site_folder', '',
            'Survey folder containing the per-camera image subdirectories. '
            'Empty = derive it from the first entry of "image_list" (the '
            'streamed file_name port only carries basenames, so the folder '
            'cannot be recovered from it).')
        _add_declare_config(
            self, 'image_list', 'input_list_1.txt',
            'Image-list file used only to locate the survey folder when '
            'site_folder is empty; its first entry must be a full image path. '
            'Defaults to the pipeline input list.')
        _add_declare_config(
            self, 'register_scope', 'folder',
            'What to register: "folder" (default) registers the whole survey '
            'folder - best when the streamed frames are a contiguous slice of '
            'one coherent survey. "list" registers only an explicit set of '
            'frames (see frame_list) - use when the frames are a subset drawn '
            'from a larger or mixed folder. Chains still need a reasonably '
            'contiguous per-camera run to register well.')
        _add_declare_config(
            self, 'frame_list', '',
            'For register_scope=list: comma-separated image-list file(s) whose '
            'lines are the frames to register. Empty = the pipeline input '
            'lists input_list_1.txt..input_list_<n_input>.txt.')

        self._n_input = int(self.config_value('n_input'))
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
        self._cache = self.config_value('cache') or None
        self._site_folder = self.config_value('site_folder') or None
        self._image_list = self.config_value('image_list') or None
        self._scope = (self.config_value('register_scope') or 'folder').lower()
        self._frame_list = self.config_value('frame_list') or None
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
        lst = self._image_list
        if lst and os.path.exists(lst):
            with open(lst) as f:
                for line in f:
                    p = line.strip()
                    if p:
                        return os.path.dirname(os.path.dirname(
                            os.path.abspath(p)))
        if hint_name and os.path.dirname(str(hint_name)):
            return os.path.dirname(os.path.dirname(
                os.path.abspath(str(hint_name))))
        return None

    def _resolve_images(self):
        """Explicit image paths to register (register_scope=list), or None for
        whole-folder registration. Reads frame_list files, or the pipeline
        input lists input_list_1.txt..input_list_<n_input>.txt by default."""
        if self._scope != 'list':
            return None
        if self._frame_list:
            files = [p.strip() for p in self._frame_list.split(',')
                     if p.strip()]
        else:
            files = ['input_list_%d.txt' % i
                     for i in range(1, self._n_input + 1)]
        images = []
        for lf in files:
            if not os.path.exists(lf):
                _log('frame_list file not found: %s' % lf)
                continue
            with open(lf) as f:
                images += [ln.strip() for ln in f if ln.strip()]
        return images or None

    # ---- one-time batch registration -------------------------------------

    def _cache_key(self, site_folder, images):
        if images is not None:
            names = sorted(os.path.basename(p) for p in images)
            scope = 'list'
        else:
            names = []
            for cam in sorted(os.listdir(site_folder)):
                d = os.path.join(site_folder, cam)
                if os.path.isdir(d):
                    names += sorted(os.listdir(d))
            scope = 'folder'
        fl = self._flight_log or ''
        have_fl = fl and os.path.exists(fl)
        fl_stamp = str(os.path.getmtime(fl)) if have_fl else ''
        blob = '\n'.join(names) + f'|{scope}|{self._method}|{fl}|{fl_stamp}'
        return hashlib.sha1(blob.encode('utf-8', 'replace')).hexdigest()

    def _cache_path(self, site_folder, key):
        # Key the filename so folder-mode and each distinct frame subset cache
        # independently (no thrash when alternating scopes/subsets).
        return self._cache or os.path.join(
            site_folder, '.sea_lion_reg_cache_%s.npz' % key[:12])

    def _load_cache(self, path, key):
        try:
            z = np.load(path, allow_pickle=True)
        except (OSError, ValueError):
            return None
        if str(z['key']) != key:
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
        except OSError as e:
            _log('could not write cache %s: %s' % (path, e))

    def _register(self, site_folder, images):
        from viame.opencv import prior_coverage_core as pcc
        key = self._cache_key(site_folder, images)
        path = self._cache_path(site_folder, key)
        cached = self._load_cache(path, key)
        if cached is not None:
            _log('loaded %d cached homographies from %s' % (len(cached), path))
            return cached
        scope = ('%d listed frames' % len(images) if images is not None
                 else 'whole folder')
        _log('registering %s [%s] (method=%s, flight_log=%s) - one-time, '
             'blocks the first frame and may be slow...'
             % (site_folder, scope, self._method, self._flight_log or 'none'))
        homogs = pcc.compute_frame_homographies(
            site_folder, flight_logs=self._flight_log, method=self._method,
            water_method=self._water_method, images=images)
        by_name = {}
        n_geo = 0
        for rel, info in homogs.items():
            H = info['H']
            by_name[os.path.basename(rel)] = None if H is None else \
                np.asarray(H, dtype=np.float64)
            n_geo += H is not None
        _log('%d images, %d geo-referenced' % (len(by_name), n_geo))
        self._save_cache(path, key, by_name)
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
                _log('could not resolve survey folder (site_folder=%r '
                     'image_list=%r); emitting identity homographies'
                     % (self._site_folder, self._image_list))
                self._by_name = {}
            else:
                self._by_name = self._register(site, self._resolve_images())
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
