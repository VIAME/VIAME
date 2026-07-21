# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""multicam_homog_blackout - black out previously-observed image regions.

For each camera of a registered multi-camera survey, computes the regions of
the current image that were already observed by other frames -- the same
regions the multicam_homog_det_suppressor emits as suppression polygons
(past frames per `past_frames`, plus the directional same-timestep camera
overlap) -- and outputs a copy of the image with those regions filled black.
Downstream detectors then cannot re-detect animals that were already seen,
and reviewers see at a glance which parts of a frame are new.

Ports per camera i (1-based): image<i> + homog<i> (source-to-ref, e.g. from
colmap_registration or many_image_stabilizer) in, image<i> out.
"""

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import Image, ImageContainer

from .multicam_homog_tracker import MultiHomographyF2F
from .multicam_homog_det_suppressor import (
    concat_suppression_homogs_and_sizes,
    find_all_suppression_homogs_and_sizes,
    find_prev_suppression_homogs_and_sizes,
    get_self_suppression_homogs_and_sizes,
    suppression_polys,
)
from .simple_homog_tracker import add_declare_config, wrap_F2FHomography
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)


def blackout_image(img, polys):
    """Return a copy of `img` (ndarray) with the given polygons filled black"""
    import cv2
    out = np.ascontiguousarray(img)
    pts = [np.round(np.asarray(p)).astype(np.int32) for p, _name in polys]
    pts = [p for p in pts if len(p) >= 3]
    if pts:
        out = out.copy()
        cv2.fillPoly(out, pts, 0)
    return out


class MulticamHomogBlackout(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')
        add_declare_config(self, 'enabled', 'true', (
            'When false, images pass through unchanged (no blackout)'
        ))
        add_declare_config(self, 'past_frames', 'prev_neighbors', (
            'Which past frames count as prior observations.  Valid values'
            ' are "prev_neighbors" (previous frame and same and neighboring'
            ' cameras only; this is the default) and "all" (all past frames'
            ' and cameras)'
        ))

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self._n_input = int(self.config_value('n_input'))
        for i in range(1, self._n_input + 1):
            add_declare_input_port(self, 'image' + str(i), 'image', required,
                                   'Input image #' + str(i))
            add_declare_input_port(self, 'homog' + str(i), 'homography_src_to_ref',
                                   required, 'Input homography (source-to-ref) #' + str(i))
            add_declare_output_port(self, 'image' + str(i), 'image',
                                    optional, 'Output image #' + str(i))

    def _configure(self):
        self._n_input = int(self.config_value('n_input'))
        self._enabled = (self.config_value('enabled').lower()
                         in ('true', '1', 'yes', 'on'))
        pf = self.config_value('past_frames')
        if pf == 'prev_neighbors':
            self._fshs = find_prev_suppression_homogs_and_sizes()
        elif pf == 'all':
            self._fshs = find_all_suppression_homogs_and_sizes()
        else:
            raise ValueError("Invalid value for past_frames")
        self._base_configure()

    def _step(self):
        images = [self.grab_input_using_trait('image' + str(i))
                  for i in range(1, self._n_input + 1)]
        homogs = [self.grab_input_using_trait('homog' + str(i))
                  for i in range(1, self._n_input + 1)]
        arrays = [im.image().asarray() for im in images]
        if not self._enabled:
            for i, im in enumerate(images, 1):
                self.push_to_port_using_trait('image' + str(i), im)
            self._base_step()
            return
        sizes = np.array([[a.shape[1], a.shape[0]] for a in arrays])
        names = np.array([''] * self._n_input, dtype=object)
        multihomog = MultiHomographyF2F.from_homographyf2fs(
            map(wrap_F2FHomography, homogs))
        prev_shs = self._fshs.step(multihomog, sizes, names)
        curr_shs = get_self_suppression_homogs_and_sizes(
            multihomog, sizes, names)
        shs = concat_suppression_homogs_and_sizes(prev_shs, curr_shs)
        polys = suppression_polys(shs, sizes)
        for i, (im, arr, ps) in enumerate(zip(images, arrays, polys), 1):
            if ps:
                out = blackout_image(arr, ps)
                self.push_to_port_using_trait(
                    'image' + str(i), ImageContainer(Image(out)))
            else:
                self.push_to_port_using_trait('image' + str(i), im)
        self._base_step()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MulticamHomogBlackout'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_blackout',
        'Black out previously-observed regions of registered multi-camera '
        'imagery',
        MulticamHomogBlackout,
    )
    process_factory.mark_process_module_as_loaded(module_name)
