# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import itertools
import sys
import numpy as np
import scipy as sp
import scipy.optimize

import torch

# Initialize cuDNN early at module import time to prevent
# CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED when running with other CUDA processes
if torch.cuda.is_available():
    try:
        with torch.no_grad():
            _dummy_conv = torch.nn.Conv2d(3, 3, 3, padding=1).cuda()
            _dummy_input = torch.randn(1, 3, 8, 8, device='cuda')
            _ = _dummy_conv(_dummy_input)
            del _dummy_conv, _dummy_input
            torch.cuda.synchronize()
    except Exception:
        pass  # If this fails, we'll handle it later during _configure

from timeit import default_timer as timer
from torchvision import models, transforms
from PIL import Image as pilImage

from kwiver.sprokit.pipeline import process
from kwiver.sprokit.processes.kwiver_process import KwiverProcess

from kwiver.vital.types import Image
from kwiver.vital.types import DetectedObject, DetectedObjectSet
from kwiver.vital.types import ObjectTrackState, Track, ObjectTrackSet
from kwiver.vital.types import new_descriptor
from kwiver.vital.util.VitalPIL import get_pil_image

from viame.pytorch.utilities import Grid, gpu_list_desc, parse_gpu_list

from viame.pytorch.srnn.track import track_state, track, track_set
from viame.pytorch.srnn.models import Siamese
from viame.pytorch.srnn.srnn_matching import SRNNMatching, RnnType
from viame.pytorch.srnn.siamese_feature_extractor import SiameseFeatureExtractor
from viame.pytorch.srnn.iou_tracker import IOUTracker
from viame.pytorch.srnn.gt_bbox import GTBBox, GTFileType
from viame.pytorch.srnn.models import get_config

g_config = get_config()

def timing(desc, f):
    """Return f(), printing a message about how long it took"""
    start = timer()
    result = f()
    end = timer()
    print('%%%', desc, ' elapsed time: ', end - start, sep='')
    return result

def groupby(it, key):
    result = {}
    for x in it:
        result.setdefault(key(x), []).append(x)
    return result

def ts2ots(track_set):
    ot_list = [Track(id=t.track_id) for t in track_set]

    for idx, t in enumerate(track_set):
        ot = ot_list[idx]
        for ti in t.full_history:
            ot_state = ObjectTrackState(ti.sys_frame_id, ti.sys_frame_time,
                                        ti.detected_object)
            if not ot.append(ot_state):
                print('Error: Cannot add ObjectTrackState')
    return ObjectTrackSet(ot_list)

def from_homog_f2f(homog_f2f):
    """Take a F2FHomography and return a triple of a 3x3 numpy.ndarray and
    two integers corresponding to the contained homography and the
    from and to IDs, respectively.

    """
    arr = np.array([
        [homog_f2f.get(r, c) for c in range(3)] for r in range(3)
    ])
    return arr, homog_f2f.from_id, homog_f2f.to_id

def transform_homog(homog, point):
    """Transform point (a length-2 array-like) using homog (a 3x3 ndarray)"""
    # We actually write this generically so it has signature (m+1, n+1), (n) -> (m)
    point = np.asarray(point)
    ones = np.ones(point.shape[:-1] + (1,), dtype=point.dtype)
    point = np.concatenate((point, ones), axis=-1)
    result = np.matmul(homog, point[..., np.newaxis])[..., 0]
    return result[..., :-1] / result[..., -1:]

def transform_homog_bbox(homog, bbox):
    """Given a bbox as [x_min, y_min, width, height], transform it
    according to homog and return the smallest enclosing bbox in the
    same format.

    """
    x_min, y_min, width, height = bbox
    points = [
        [x_min, y_min],
        [x_min, y_min + height],
        [x_min + width, y_min],
        [x_min + width, y_min + height],
    ]
    tpoints = transform_homog(homog, points)
    tx_min, ty_min = tpoints.min(0)
    tx_max, ty_max = tpoints.max(0)
    twidth, theight = tx_max - tx_min, ty_max - ty_min
    return [tx_min, ty_min, twidth, theight]

class SRNNTracker(KwiverProcess):
    # ----------------------------------------------
    def __init__(self, conf):
        KwiverProcess.__init__(self, conf)

        self.__declare_config_traits()

        # AFRL start id : 0
        # MOT start id : 1
        self._step_id = 0

        # Homography state
        #
        # We maintain transformations to a "base" coordinate system.
        # Since homographies come in as mappings from the current
        # frame to an infrequently changing reference frame, we
        # separately store the mapping from the current reference
        # frame to "base" coordinates and the mapping from the current
        # frame to the reference.
        #
        # We will handle breaks (changes in the reference frame) by
        # assuming that the mapping from the new reference frame to
        # the last frame is identity.
        #
        # Missing input is treated as a break with an anonymous
        # reference.

        # 3x3 ndarray from current reference frame to base
        self._homog_ref_to_base = np.identity(3)
        # Current reference frame (or None for anonymous)
        self._homog_ref_id = None
        # Mapping from current frame to reference (or None for identity)
        self._homog_src_to_ref = None

        # Old state maintained to allow (limited) use of
        # initializations from the previous frame
        #
        # Other variables of the form self._prev_* are used as well,
        # but they don't need to be set ahead of time as here.
        self._prev_frame = None  # Previous frame ID or None

        # set up required flags
        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        #  input port ( port-name,flags)
        # self.declare_input_port_using_trait('framestamp', optional)
        self.declare_input_port_using_trait('image', required)
        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)
        # Initializations
        self.declare_input_port_using_trait('object_track_set', optional)
        self.declare_input_port_using_trait('homography_src_to_ref', optional)

        #  output port ( port-name,flags)
        self.declare_output_port_using_trait('object_track_set', optional)
        self.declare_output_port_using_trait('detected_object_set', optional)

    def __declare_config_traits(self):
        def add_declare_config_trait(name, default, desc):
            self.add_config_trait(name, name, default, desc)
            self.declare_config_using_trait(name)

        #GPU list
        add_declare_config_trait('gpu_list', 'all',
                                 gpu_list_desc(use_for='SRNN tracking'))

        # siamese
        #----------------------------------------------------------------------------------
        add_declare_config_trait('siamese_model_path',
                                 'siamese/snapshot_epoch_6.pt',
                                 'Trained PyTorch model.')

        add_declare_config_trait('siamese_model_input_size', '224',
                                 'Model input image size')

        add_declare_config_trait('siamese_batch_size', '128',
                                 'siamese model processing batch size')
        #----------------------------------------------------------------------------------

        # detection select threshold
        add_declare_config_trait('detection_select_threshold', '0.0',
                                 'detection select threshold')
        add_declare_config_trait('track_initialization_threshold', '0.0',
                                 'track initialization threshold')

        # SRNN
        #----------------------------------------------------------------------------------
        # target RNN full model
        add_declare_config_trait("targetRNN_AIM_model_path",
                                 'targetRNN_snapshot/App_LSTM_epoch_51.pt',
                                 'Trained targetRNN PyTorch model.')

        # target RNN AI model
        add_declare_config_trait("targetRNN_AIM_V_model_path",
                                 'targetRNN_AI/App_LSTM_epoch_51.pt',
                                 'Trained targetRNN AIM with variable input size PyTorch model.')

        # target RNN batch size
        add_declare_config_trait("targetRNN_batch_size", '256',
                                 'targetRNN model processing batch size')

        # target RNN normalization
        add_declare_config_trait("targetRNN_normalized_models", 'False',
                                 "If the provided models have a normalization layer, "
                                 "this should be exactly the string 'True' (no quotes)")

        # matching similarity threshold
        add_declare_config_trait("similarity_threshold", '0.5',
                                 'similarity threshold.')
        #----------------------------------------------------------------------------------

        # IOU
        #----------------------------------------------------------------------------------
        # IOU tracker flag
        add_declare_config_trait("IOU_tracker_flag", 'True', 'IOU tracker flag.')

        # IOU accept threshold
        add_declare_config_trait("IOU_accept_threshold", '0.5',
                                 'IOU accept threshold.')

        # IOU reject threshold
        add_declare_config_trait("IOU_reject_threshold", '0.1',
                                 'IOU reject threshold.')
        #----------------------------------------------------------------------------------

        # search threshold
        add_declare_config_trait("track_search_threshold", '0.1',
                                 'track search threshold.')

        # matching active track threshold
        add_declare_config_trait("terminate_track_threshold", '15',
                                 'terminate the tracking if the target has been lost for more than '
                                 'terminate_track_threshold read-in frames.')

        # matching active track threshold
        add_declare_config_trait("sys_terminate_track_threshold", '50',
                                 'terminate the tracking if the target has been lost for more than '
                                 'terminate_track_threshold system (original) frames.')

        # MOT gt detection
        #-------------------------------------------------------------------
        add_declare_config_trait("MOT_GTbbox_flag", 'False', 'MOT GT bbox flag')
        #-------------------------------------------------------------------

        # AFRL gt detection
        #-------------------------------------------------------------------
        add_declare_config_trait("AFRL_GTbbox_flag", 'False', 'AFRL GT bbox flag')
        #-------------------------------------------------------------------

        # GT bbox file
        #-------------------------------------------------------------------
        add_declare_config_trait("GT_bbox_file_path", '',
                                 'ground truth detection file for testing')
        #-------------------------------------------------------------------

        # Add features to detections
        #-------------------------------------------------------------------
        add_declare_config_trait("add_features_to_detections", 'True',
                                 'Should we add internally computed features to detections?')
        #-------------------------------------------------------------------

        # Track initialization
        # -------------------------------------------------------------------
        # XXX Otherwise keep IDs consistent input-to-output and
        # prevent overlapping tracks.
        add_declare_config_trait('explicit_initialization', 'False',
                                 'The string "True" (no quotes, same capitalization) '
                                 'if only tracks derived from the most recently provided '
                                 'nonempty object track set should be output')

        add_declare_config_trait('initialization_overlap_threshold', '0.7',
                                 'When initializations are present, any additional '
                                 'incoming detection is only considered when its IOU '
                                 'with each of the initializations is at most this value')
        #-------------------------------------------------------------------

    # ----------------------------------------------
    def _configure(self):
        self._select_threshold = float(self.config_value('detection_select_threshold'))
        self._track_initialization_threshold = float(self.config_value('track_initialization_threshold'))

        #GPU_list
        self._gpu_list = parse_gpu_list(self.config_value('gpu_list'))

        # Siamese model config
        siamese_img_size = int(self.config_value('siamese_model_input_size'))
        siamese_batch_size = int(self.config_value('siamese_batch_size'))
        siamese_model_path = self.config_value('siamese_model_path')
        self._app_feature_extractor = SiameseFeatureExtractor(siamese_model_path,
                siamese_img_size, siamese_batch_size, self._gpu_list)

        # targetRNN_full model config
        targetRNN_batch_size = int(self.config_value('targetRNN_batch_size'))
        targetRNN_AIM_model_path = self.config_value('targetRNN_AIM_model_path')
        targetRNN_AIM_V_model_path = self.config_value('targetRNN_AIM_V_model_path')
        targetRNN_normalized_models = 'True' == self.config_value('targetRNN_normalized_models')
        self._srnn_matching = SRNNMatching(
            targetRNN_AIM_model_path, targetRNN_AIM_V_model_path,
            targetRNN_normalized_models, targetRNN_batch_size, self._gpu_list,
        )

        self._gtbbox_flag = False
        # use MOT gt detection
        MOT_GTbbox_flag = self.config_value('MOT_GTbbox_flag')
        MOT_GT_flag = (MOT_GTbbox_flag == 'True')
        if MOT_GT_flag:
            file_format = GTFileType.MOT

        # use AFRL gt detection
        AFRL_GTbbox_flag = self.config_value('AFRL_GTbbox_flag')
        AFRL_GT_flag = (AFRL_GTbbox_flag == 'True')
        if AFRL_GT_flag:
            file_format = GTFileType.AFRL

        # IOU tracker flag
        self._IOU_flag = True
        IOU_flag = self.config_value('IOU_tracker_flag')
        self._IOU_flag = (IOU_flag == 'True')

        self._gtbbox_flag = MOT_GT_flag or AFRL_GT_flag

        # read GT bbox related
        if self._gtbbox_flag:
            gtbbox_file_path = self.config_value('GT_bbox_file_path')
            self._m_bbox = GTBBox(gtbbox_file_path, file_format)

        self._similarity_threshold = float(self.config_value('similarity_threshold'))

        # IOU tracker
        iou_accept_threshold = float(self.config_value('IOU_accept_threshold'))
        iou_reject_threshold = float(self.config_value('IOU_reject_threshold'))
        self._iou_tracker = IOUTracker(iou_accept_threshold, iou_reject_threshold)

        # track search threshold
        self._ts_threshold = float(self.config_value('track_search_threshold'))
        self._grid = Grid()
        # generated track_set
        self._track_set = track_set()
        self._terminate_track_threshold = int(self.config_value('terminate_track_threshold'))
        self._sys_terminate_track_threshold = int(self.config_value('sys_terminate_track_threshold'))
        # add features to detections?
        self._add_features_to_detections = \
                (self.config_value('add_features_to_detections') == 'True')
        self._explicit_initialization = \
            self.config_value('explicit_initialization') == 'True'
        self._init_max_iou = float(self.config_value('initialization_overlap_threshold'))
        self._base_configure()

    # ----------------------------------------------
    def _step(self):
        try:
            det_obj_set = self._step_unwrapped()
        except BaseException as e:
            print( repr( e ) )
            import traceback
            print( traceback.format_exc() )
            sys.stdout.flush()
            raise

        # push track set to output port
        ots = ts2ots(self._track_set)
        self.push_to_port_using_trait('object_track_set', ots)
        self.push_to_port_using_trait('detected_object_set', det_obj_set)

        self._step_id += 1
        self._base_step()


    def _step_unwrapped(self):
        """Perform _step, but don't handle errors or increment self._step_id
        and return the output DetectedObjectSet.  Mutates this object,
        in particular self._track_set.

        """
        print('step', self._step_id)

        # grab image container from port using traits
        in_img_c = self.grab_input_using_trait('image')
        timestamp = self.grab_input_using_trait('timestamp')
        dos_ptr = self.grab_input_using_trait('detected_object_set')
        if self.has_input_port_edge('object_track_set'):
            # Initializations
            inits = self.grab_input_using_trait('object_track_set')
            inits = [] if inits is None else inits.tracks()
        else:
            # An empty value is treated the same as no value
            inits = []
        if self.has_input_port_edge('homography_src_to_ref'):
            homog_f2f = self.grab_input_using_trait('homography_src_to_ref')
        else:
            homog_f2f = None
        print('timestamp =', repr(timestamp))

        # Get current frame
        im = get_pil_image(in_img_c.image()).convert('RGB')

        # Get detection bbox
        if self._gtbbox_flag:
            dos = [DetectedObject(bbox=bbox, confidence=1.)
                   for bbox in self._m_bbox[self._step_id]]
        else:
            dos = dos_ptr.select(self._select_threshold)
        #print('bbox list len is', dos.size())

        homog_src_to_base = self._step_homog_state(homog_f2f)

        inits = {
            lf: {t.id: t[lf].detection() for t in tracks} for lf, tracks
            in groupby(inits, lambda t: t.last_frame).items()
        }

        def max_iou_filter(det_dict, max_iou):
            """Return a function that takes a DetectedObject and returns true when
            the overlap of its bounding box with each of the provided
            detections is at most the provided maximum IOU.

            """
            # XXX ious should be defined in a more generic place
            from kwiver.sprokit.processes.simple_homog_tracker import ious
            bboxes = []
            for det in det_dict.values():
                bb = det.bounding_box
                bboxes.append((bb.min_x(), bb.max_x(), bb.min_y(), bb.max_y()))
            # dims: Nx{x,y}x{min,max}
            bboxes = np.stack(bboxes).reshape((-1, 2, 2))
            def run(det):
                bb = det.bounding_box
                bb = np.array(((bb.min_x(), bb.max_x()), (bb.min_y(), bb.max_y())))
                return (ious(bb, bboxes) <= max_iou).all()
            return run

        prev_inits = inits.get(self._prev_frame)
        if prev_inits:
            # Ignore tracks we saw as same-frame initializations on
            # the previous frame.  Do this on the expectation that
            # they are equivalent to (though perhaps not sharing
            # object identity with) the initializations on the
            # previous frame.
            prev_inits = {
                tid: det for tid, det in prev_inits.items()
                if tid not in self._prev_inits
            }
        if prev_inits:
            if not self._explicit_initialization:
                is_overlap_free = max_iou_filter(prev_inits, self._init_max_iou)
                for track in list(self._track_set.iter_active()):
                    if not is_overlap_free(track[-1].detected_object):
                        self._track_set.deactivate_track(track)

            _, prev_track_state_list = self._convert_detected_objects(
                list(prev_inits.values()),
                self._step_id - 1, self._prev_fid, self._prev_ts,
                self._prev_im, self._prev_homog_src_to_base,
                extra_dos=self._prev_all_dos,
            )

            if self._explicit_initialization:
                # XXX This has a delayed effect compared to with
                # normal inits
                self._track_set.deactivate_all_tracks()

            # This is the only relevant part of _step_track_set.
            # Unlike there, we permit overwriting the last frame's
            # state if necessary (on_duplicate='replace').

            # Directly add explicit init tracks
            for tid, ts in zip(prev_inits, prev_track_state_list):
                # XXX This doesn't check for unintended overlap with an automatic ID
                track = self._track_set.make_track(tid, on_exist='resume')
                track.append(ts, on_duplicate='replace')

        inits = inits.get(timestamp.get_frame(), {})
        if not self._explicit_initialization and inits:
            is_overlap_free = max_iou_filter(inits, self._init_max_iou)
            dos = list(filter(is_overlap_free, dos))

        all_dos = list(itertools.chain(dos, inits.values()))

        if self._gtbbox_flag:
            fid = ts = self._step_id
        else:
            fid = timestamp.get_frame()
            ts = timestamp.get_time_usec()

        det_obj_set, all_track_state_list = self._convert_detected_objects(
            all_dos, self._step_id, fid, ts, im, homog_src_to_base,
        )
        track_state_list = all_track_state_list[:len(dos)]
        init_track_state_list = all_track_state_list[len(dos):]

        if self._explicit_initialization and inits:
            self._track_set.deactivate_all_tracks()

        self._step_track_set(fid, track_state_list, zip(inits, init_track_state_list))

        self._prev_inits = inits
        self._prev_frame = timestamp.get_frame()
        self._prev_fid, self._prev_ts = fid, ts
        self._prev_im = im
        self._prev_homog_src_to_base = homog_src_to_base
        self._prev_all_dos = all_dos

        return det_obj_set


    def _convert_detected_objects(
            self, dos, frame_id, sys_frame_id, sys_frame_time,
            image, homog_src_to_base, extra_dos=None,
    ):
        """Turn a list of DetectedObjects into a feature-enhanced
        DetectedObjectSet and list of track_states.

        Parameters:
        - dos: The list of DetectedObjects
        - frame_id: The current frame ID
        - sys_frame_id: The externally provided frame ID
        - sys_frame_time: The externally provided time
        - image: PIL image for the current frame
        - homog_src_to_base: 3x3 ndarray transforming current to
          "base" coordinates
        - extra_dos: (optional) A list of DetectedObjects not
          represented in the output

        """
        bboxes = [d_obj.bounding_box for d_obj in dos]
        extra_bboxes = None if extra_dos is None else [
            d_obj.bounding_box for d_obj in extra_dos
        ]

        # interaction features
        grid_feature_list = timing('grid feature', lambda: (
            self._grid(image.size, bboxes, extra_bboxes)))

        # appearance features (format: pytorch tensor)
        pt_app_features = timing('app feature', lambda: (
            self._app_feature_extractor(image, bboxes)))

        det_obj_set = DetectedObjectSet()
        track_state_list = []

        # get new track state from new frame and detections
        for bbox, d_obj, grid_feature, app_feature in zip(
                bboxes, dos, grid_feature_list, pt_app_features,
        ):
            if self._add_features_to_detections:
                # store app feature to detected_object
                app_f = new_descriptor(g_config.A_F_num)
                app_f[:] = app_feature.numpy()
                d_obj.set_descriptor(app_f)
            det_obj_set.add(d_obj)

            # build track state for current bbox for matching
            bbox_as_list = [bbox.min_x(), bbox.min_y(), bbox.width(), bbox.height()]
            cur_ts = track_state(
                frame_id=frame_id,
                bbox_center=bbox.center(),
                ref_point=transform_homog(homog_src_to_base, bbox.center()),
                interaction_feature=grid_feature,
                app_feature=app_feature,
                bbox=[int(x) for x in bbox_as_list],
                ref_bbox=transform_homog_bbox(homog_src_to_base, bbox_as_list),
                detected_object=d_obj,
                sys_frame_id=sys_frame_id,
                sys_frame_time=sys_frame_time,
            )
            track_state_list.append(cur_ts)

        return det_obj_set, track_state_list


    def _step_track_set(self, frame_id, track_state_list, init_track_states):
        """Step self._track_set using the current frame id, the list of track
        states, and an iterable of (track_id, track_state) pairs to
        directly initialize.

        This deactivates old tracks, extends existing ones, and
        creates new ones according to this object's configuration.

        """
        # check whether we need to terminate a track
        for track in list(self._track_set.iter_active()):
            # terminating a track based on readin_frame_id or original_frame_id gap
            if (self._step_id - track[-1].frame_id > self._terminate_track_threshold
                or frame_id - track[-1].sys_frame_id > self._sys_terminate_track_threshold):
                self._track_set.deactivate_track(track)

        # Get a list of the active tracks before directly adding the
        # explicitly initialized ones.
        tracks = list(self._track_set.iter_active())

        # Directly add explicit init tracks
        for tid, ts in init_track_states:
            # XXX This doesn't check for unintended overlap with an automatic ID
            self._track_set.make_track(tid, on_exist='restart').append(ts)

        next_track_id = int(self._track_set.get_max_track_id()) + 1

        # call IOU tracker
        if self._IOU_flag:
            tracks, track_state_list = timing('IOU tracking', lambda: (
                self._iou_tracker(tracks, track_state_list)
            ))

        #print('***track_set len', len(self._track_set))
        #print('***track_state_list len', len(track_state_list))

        # estimate similarity matrix
        similarity_mat = timing('SRNN association', lambda: (
            self._srnn_matching(tracks, track_state_list, self._ts_threshold)
        ))

        # Hungarian algorithm
        row_idx_list, col_idx_list = timing('Hungarian algorithm', lambda: (
            sp.optimize.linear_sum_assignment(similarity_mat)
        ))

        # Contains the row associated with each column, or None
        hung_idx_list = [None] * len(track_state_list)
        for r, c in zip(row_idx_list, col_idx_list):
            hung_idx_list[c] = r

        for c, r in enumerate(hung_idx_list):
            if r is None or -similarity_mat[r, c] < self._similarity_threshold:
                # Conditionally initialize a new track
                if not self._explicit_initialization and (
                        track_state_list[c].detected_object.confidence
                        >= self._track_initialization_threshold
                ):
                    track = self._track_set.make_track(next_track_id)
                    track.append(track_state_list[c])
                    next_track_id += 1
            else:
                # add to existing track
                tracks[r].append(track_state_list[c])

        print('total tracks', len(self._track_set))


    def _step_homog_state(self, homog_f2f):
        """Step stabilization state (self._homog_* instance variables) using
        the provided HomographyF2F (or None), returning a
        transformation from current coordinates to "base" coordinates.

        """
        # Update homography
        if homog_f2f is not None:
            homog_f2f_arr, homog_f2f_from, homog_f2f_to = from_homog_f2f(homog_f2f)
        if homog_f2f is None or homog_f2f_to != self._homog_ref_id:
            # We have a new reference frame
            # Update self._homog_ref_to_base (assume curr->prev is identity)
            if self._homog_src_to_ref is not None:
                self._homog_ref_to_base = np.matmul(self._homog_ref_to_base, self._homog_src_to_ref)
                self._homog_src_to_ref = None
            # Update self._homog_ref_id
            if homog_f2f is None:
                self._homog_ref_id = None
            else:
                assert homog_f2f_from == homog_f2f_to, "After break homog should map to self"
                self._homog_ref_id = homog_f2f_to
            # This is a reference frame, so src->base is just ref->base
            homog_src_to_base = self._homog_ref_to_base
        else:
            # We use the same reference frame
            self._homog_src_to_ref = homog_f2f_arr
            homog_src_to_base = np.matmul(self._homog_ref_to_base, self._homog_src_to_ref)
        return homog_src_to_base


# ==================================================================
def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory

    module_name = 'python:kwiver.pytorch.SRNNTracker'

    if process_factory.is_process_module_loaded(module_name):
        return

    process_factory.add_process('srnn_tracker',
                                'Structural RNN based tracking',
                                SRNNTracker)

    process_factory.mark_process_module_as_loaded(module_name)
