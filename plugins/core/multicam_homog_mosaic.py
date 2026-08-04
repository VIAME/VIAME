# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
"""multicam_homog_mosaic - per-timestep tile mosaics of a registered rig.

For each time slice of a registered multi-camera survey, warps every
camera's image into the first camera's pixel frame (through the shared
source-to-ref homographies, e.g. from colmap_registration or
many_image_stabilizer) and composites them into one mosaic image. This is
the in-pipeline, DIVE-runnable counterpart of the older offline mosaic
scripts (tools/create_mosaic.py and the process-data flows).

Ports per camera i (1-based): image<i> + homog<i> in, plus an optional
file_name<i>.  Outputs one `image` (the mosaic) and a `file_name` for it
(derived from camera 1's input file name) per timestep, suitable for
connecting straight to an image_writer.
"""

import os

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import Image, ImageContainer

from .simple_homog_tracker import add_declare_config, wrap_F2FHomography
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)


def compute_mosaic(arrays, homogs, max_dimension=20000):
    """Composite images into the first image's pixel frame.

    Arguments:
    - arrays: list of HxW[xC] ndarrays
    - homogs: list of 3x3 source-to-ref matrices, one per image
    - max_dimension: cap on either mosaic dimension; the mosaic is scaled
      down uniformly to fit when the warped extent exceeds it

    Returns the mosaic ndarray. The first camera is drawn last, so in
    overlap areas the reference camera's pixels win.
    """
    import cv2
    try:
        ref_inv = np.linalg.inv(homogs[0])
    except np.linalg.LinAlgError:
        ref_inv = np.eye(3)
    rels = []
    corners = []
    for arr, H in zip(arrays, homogs):
        h, w = arr.shape[:2]
        try:
            R = ref_inv @ H
        except (ValueError, TypeError):
            R = np.eye(3)
        if not np.all(np.isfinite(R)) or abs(R[2, 2]) < 1e-12:
            R = np.eye(3)
        rels.append(R)
        pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float64)
        q = np.column_stack([pts, np.ones(4)]) @ R.T
        corners.append(q[:, :2] / q[:, 2:3])
    corners = np.concatenate(corners)
    lo = np.floor(corners.min(axis=0))
    hi = np.ceil(corners.max(axis=0))
    size = hi - lo + 1
    scale = 1.0
    if size.max() > max_dimension:
        scale = max_dimension / size.max()
    W = max(1, int(round(size[0] * scale)))
    Ht = max(1, int(round(size[1] * scale)))
    S = np.array([[scale, 0, -lo[0] * scale],
                  [0, scale, -lo[1] * scale],
                  [0, 0, 1.0]])
    mosaic = None
    # Reverse order: camera 1 drawn last so the reference camera wins overlaps
    for arr, R in list(zip(arrays, rels))[::-1]:
        arr = np.ascontiguousarray(arr)
        warped = cv2.warpPerspective(arr, S @ R, (W, Ht))
        mask = cv2.warpPerspective(
            np.full(arr.shape[:2], 255, dtype=np.uint8), S @ R, (W, Ht))
        if mosaic is None:
            mosaic = np.zeros_like(warped)
        m = mask > 127
        mosaic[m] = warped[m]
    return mosaic


class MulticamHomogMosaic(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, 'n_input', '2', 'Number of inputs')
        add_declare_config(self, 'max_dimension', '20000', (
            'Cap on either mosaic dimension in pixels; larger warped extents'
            ' are scaled down uniformly to fit'
        ))
        add_declare_config(self, 'file_name_suffix', '_mosaic.png', (
            'Suffix (including extension) appended to camera 1\'s file name'
            ' stem to name the output mosaic; a downstream image_writer'
            ' file_name_template extension still overrides the extension'
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
            add_declare_input_port(self, 'file_name' + str(i), 'file_name',
                                   optional, 'Input image path #' + str(i))
        add_declare_output_port(self, 'image', 'image', optional,
                                'Output mosaic image')
        add_declare_output_port(self, 'file_name', 'file_name', optional,
                                'Output mosaic file name')

    def _configure(self):
        self._n_input = int(self.config_value('n_input'))
        self._max_dimension = int(float(self.config_value('max_dimension')))
        self._suffix = self.config_value('file_name_suffix')
        self._fn_connected = None
        self._frame = 0
        self._base_configure()

    def _step(self):
        if self._fn_connected is None:
            self._fn_connected = [
                self.has_input_port_edge_using_trait('file_name' + str(i))
                for i in range(1, self._n_input + 1)]
        images = [self.grab_input_using_trait('image' + str(i))
                  for i in range(1, self._n_input + 1)]
        homogs = [self.grab_input_using_trait('homog' + str(i))
                  for i in range(1, self._n_input + 1)]
        names = [(self.grab_input_using_trait('file_name' + str(i))
                  if self._fn_connected[i - 1] else '')
                 for i in range(1, self._n_input + 1)]
        arrays = [im.image().asarray() for im in images]
        mats = [wrap_F2FHomography(h).homog.matrix for h in homogs]
        mosaic = compute_mosaic(arrays, mats, self._max_dimension)
        self.push_to_port_using_trait('image', ImageContainer(Image(mosaic)))
        if names[0]:
            stem = os.path.splitext(os.path.basename(str(names[0])))[0]
        else:
            stem = 'frame%06d' % self._frame
        self.push_to_port_using_trait('file_name', stem + self._suffix)
        self._frame += 1
        self._base_step()


def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MulticamHomogMosaic'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_mosaic',
        'Per-timestep tile mosaic of registered multi-camera imagery',
        MulticamHomogMosaic,
    )
    process_factory.mark_process_module_as_loaded(module_name)
