# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
from collections import defaultdict
import functools
import itertools
import os
from pathlib import Path
import pickle
import random

import cv2
import numpy as np
import scipy.optimize
import scipy.special
from tqdm import tqdm
tqdm.monitor_interval = 0

from viame.pytorch.utilities import Grid

from ..storage import DataStorage, SequenceList
from ..utilities import load_track_feature_file


# Homography state
#
# We maintain transformations to a "base" coordinate system.  Since
# homographies come in as mappings from the current frame to an
# infrequently changing reference frame, we separately store the
# mapping from the current reference frame to "base" coordinates and
# the mapping from the current frame to the reference.
#
# We will handle breaks (changes in the reference frame) by assuming
# that the mapping from the new reference frame to the last frame is
# identity.
#
# Missing input is treated as a break with an anonymous reference.
class HomogState:
    __slots__ = '_ref_to_base', '_ref_id', '_src_to_ref'

    def __init__(self):
        # 3x3 ndarray from current reference frame to base
        self._ref_to_base = np.identity(3)
        # Current reference frame (or None for anonymous)
        self._ref_id = None
        # Mapping from current frame to reference (or None for identity)
        self._src_to_ref = None

    def step_raw(self, homog, from_id, to_id):
        """Perform .step using an explicit homography (3x3 ndarray) and frame
        IDs.  (Pass all arguments as None to indicate a missing
        homography.)

        """
        if to_id is None or to_id != self._ref_id:
            # We have a new reference frame
            assert (
                homog is None or (homog == np.identity(3) * homog[2, 2]).all()
            ), "After break homog should be identity"
            assert from_id == to_id, "After break homog should map to self"
            # Update self._ref_to_base (assume curr->prev is identity)
            if self._src_to_ref is not None:
                self._ref_to_base = np.matmul(self._ref_to_base, self._src_to_ref)
                self._src_to_ref = None
            # Update self._ref_id
            self._ref_id = to_id
            # This is a reference frame, so src->base is just ref->base
            src_to_base = self._ref_to_base
        else:
            # We use the same reference frame
            self._src_to_ref = homog
            src_to_base = np.matmul(self._ref_to_base, self._src_to_ref)
        return src_to_base


class concat:
    """Lightweight combination of two sequences into one"""
    __slots__ = 'left', 'right'

    def __init__(self, left, right):
        self.left, self.right = left, right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, x):
        """(Does not support slicing)"""
        if x < 0:
            x += len(self)
        len_left = len(self.left)
        return self.left[x] if x < len_left else self.right[x - len_left]


def choice_reject(seq, exclude):
    """Return a random element from the sequence seq that isn't in exclude"""
    MAX_ITER = 20
    for _ in range(MAX_ITER):
        x = random.choice(seq)
        if x not in exclude:
            return x
    # Do it the slow way
    return random.choice([x for x in seq if x not in exclude])


class AugOcclusion:
    __slots__ = 'mean_gap', 'mean_length', 'stddev_length', 'distribution',

    def __init__(self, mean_gap, mean_length, stddev_length, distribution):
        self.mean_gap = mean_gap
        self.distribution = distribution
        if distribution == 'normal':
            self.mean_length = mean_length
            self.stddev_length = stddev_length
        elif distribution == 'log_normal':
            v = np.log(1 + (stddev_length / mean_length) ** 2)
            self.mean_length = np.log(mean_length) - v / 2
            self.stddev_length = v ** .5
        else:
            raise ValueError

    class History:
        __slots__ = 'start_occluded', 'rev_state_changes',

        def __init__(self, start_occluded, state_changes):
            self.start_occluded = start_occluded
            self.rev_state_changes = state_changes[::-1]

        def is_occluded_forward(self, t):
            """Return true if the given moment in time is occluded, mutating this
            object so that all moments before the given one will
            return the same result.

            """
            rsc = self.rev_state_changes
            i = next((i for i, s in enumerate(reversed(rsc)) if s > t), len(rsc))
            so = self.start_occluded = self.start_occluded ^ bool(i & 1)
            if i:
                del rsc[-i:]
            return so

    def end_of_current_occlusion_cdf(self, t):
        """Return the probability that all occlusions that started before an
        arbitrary moment have ended by time t after that arbitrary
        moment.

        """
        from scipy.special import erfc
        # (Normal mean and stddev)
        m, s = self.mean_length, self.stddev_length
        # Antiderivate of the complementary CDF, with offset
        # chosen to approach 0 as t approaches infinity
        if self.distribution == 'normal':
            tn = (t - m) / s  # T Normalized
            int_ccdf = .5 * s * (
                tn * erfc(tn * .5 ** .5)
                - np.exp(- tn ** 2 / 2) * (2 / np.pi) ** .5
            )
        elif self.distribution == 'log_normal':
            logtn = (np.log(t) - m) / s  # Log T normalized
            int_ccdf_p = .5 * (
                t * erfc(logtn * .5 ** .5)
                - np.exp(m + s ** 2 / 2) * erfc((logtn - s) * .5 ** .5)
            )
            int_ccdf_np = t - np.exp(m + s ** 2 / 2)
            int_ccdf = np.where(t > 0, int_ccdf_p, int_ccdf_np)
        return np.exp(int_ccdf / self.mean_gap)

    def get_gap(self):
        """Generate a single gap length"""
        return np.random.exponential(self.mean_gap)

    def get_occlusion_length(self):
        """Generate a single occlusion length"""
        dist = {
            'normal': np.random.normal,
            'log_normal': np.random.lognormal,
        }[self.distribution]
        return dist(self.mean_length, self.stddev_length)

    def make_history(self, length):
        """Make an AugOcclusion.History of at least the specified length"""
        stop_cdf = np.random.random()
        stop_result = scipy.optimize.root_scalar(
            lambda t: self.end_of_current_occlusion_cdf(t) - stop_cdf,
            x0=0, x1=1,
        )
        assert stop_result.converged
        stop = max(0, stop_result.root)
        start_occluded = stop > 0
        state_changes = [stop] if start_occluded else []
        last = stop + self.get_gap()
        state_changes.append(last)
        while max(last, stop) < length:
            stop = max(stop, last + self.get_occlusion_length())
            last += self.get_gap()
            if last > stop:
                state_changes += stop, last
        return self.History(start_occluded, state_changes)


def generate_sequences(
        input_file, out_file, time_seq_len, fixSize_flag, aug_occlusion=None,
):
    """Generate all possible subsequences of each track of the desired
    length (and up to it if fixSize_flag is false).  The output file
    will hold multiple pickled tuples of DetectionIDs, where each
    tuple holds first
    seq_len+1 entries from the sequence and then a single negative
    sample.  For a given sequence and track, all chosen negative
    samples are unique.

    """
    # read all features (with track information)
    tracks, f_dids, ranges = load_track_feature_file(input_file)

    # track_sampling_factor is used below to approximately ensure that
    # the same number of sequences are generated.  It is the rounded
    # reciprocal of the probability that an arbitrary detection is not
    # occluded.
    #
    # The call to float works around a Numpy issue:
    # https://github.com/numpy/numpy/issues/15297
    track_sampling_factor = 1 if aug_occlusion is None else round(float(
        1 / aug_occlusion.end_of_current_occlusion_cdf(0),
    ))
    initial_seq_len = time_seq_len if fixSize_flag else 1

    seqs = []
    for track_states, (start, stop) in zip(tracks.values(), tqdm(ranges)):
        neg_set = set()
        for _ in range(track_sampling_factor):
            if aug_occlusion is None:
                track_states = [p for _, p in track_states]
            else:
                t0 = track_states[0][0]
                history = aug_occlusion.make_history(track_states[-1][0] - t0)
                track_states = [p for t, p in track_states
                                if not history.is_occluded_forward(t - t0)]
            # for generating sequence with different length
            for seq_len in range(initial_seq_len, time_seq_len + 1):
                for i in range(len(track_states) - seq_len):
                    pos_f_list = track_states[i:i + seq_len + 1]

                    # generate negative sample
                    p_neg_idx = choice_reject(concat(
                        range(0, start),
                        range(stop, len(f_dids)),
                    ), neg_set)

                    neg_set.add(p_neg_idx)
                    neg_f_did = f_dids[p_neg_idx]
                    seqs.append((*pos_f_list, neg_f_did))
    with open(out_file, 'wb') as f:
        pickle.dump(SequenceList.from_iterable(seqs), f)


class BoundingBox:
    __slots__ = '_min_x', '_min_y', '_width', '_height'

    def __init__(self, min_x, min_y, width, height):
        if not (width > 0 and height > 0):
            raise ValueError("Width and height must be positive")
        self._min_x = min_x
        self._min_y = min_y
        self._width = width
        self._height = height

    @classmethod
    def from_corners(cls, min_x, min_y, max_x, max_y):
        return cls(min_x, min_y, max_x - min_x, max_y - min_y)

    # Define these as methods for compatibility with the KWIVER type
    def min_x(self):
        return self._min_x

    def min_y(self):
        return self._min_y

    def width(self):
        return self._width

    def height(self):
        return self._height


def restrict_bounding_box(image_width, image_height, bounding_box):
    """Return a copy of the provided BoundingBox, clipped if needed to a
    frame of [0..image_width)x[0..image_height), or None if there is
    too little overlap.

    """
    iw, ih, bb = image_width, image_height, bounding_box
    x, y, w, h = bb.min_x(), bb.min_y(), bb.width(), bb.height()
    nx, ny = max(x, 0), max(y, 0)
    nw, nh = min(x + w, iw) - nx, min(y + h, ih) - ny
    # Have "too little overlap" mean <50%
    if nw > 0 and nh > 0 and nw * nh >= .5 * w * h:
        return BoundingBox(nx, ny, nw, nh)
    else:
        return None


def process_gt_file(gt_file_path):
    r"""Process MOT gt file
        The output of the function is a list with following format,
        indexed by frame number: [[(id_num, bbox)]].
        All quantities (include bbox coordinates) are ints.
    """
    tracks = defaultdict(list)
    with open(gt_file_path, 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            cur_line_list = line.rstrip('\n').split(' ')

            frame_id = int(cur_line_list[2])
            track_id = int(cur_line_list[0])
            bbox = [int(float(c)) for c in cur_line_list[9:13]]

            tup = (track_id, BoundingBox.from_corners(*bbox))
            tracks[frame_id].append(tup)

    return [tracks[i] for i in range(max(tracks) + 1)] if tracks else []


class Homography:
    __slots__ = 'matrix',

    def __init__(self, matrix):
        """Create a Homography from an array-like"""
        self.matrix = np.asarray(matrix)
        if self.matrix.ndim != 2:
            raise ValueError("Expected 2D array")

    def transform(self, point):
        """Transform point with the given homography"""
        point = np.asarray(point)
        # Add ones
        ones = np.ones(point.shape[:-1] + (1,), dtype=point.dtype)
        point = np.concatenate((point, ones), axis=-1)
        result = np.matmul(self.matrix, point[..., np.newaxis])[..., 0]
        # Remove extra dimension
        return result[..., :-1] / result[..., -1:]


def load_homographies(path):
    """Load homographies from a file in the format produced by
    kw_write_homography, returning a list of "Homography"s.

    The results map points to a constant reference frame.  Breaks are
    currently handled by assuming no change across the break.

    """
    result = []
    homog_state = HomogState()
    with open(path) as f:
        for line in f:
            *array, from_, to = line.split()
            homog = np.array(list(map(float, array))).reshape((3, 3))
            from_, to = int(from_), int(to)
            homog = homog_state.step_raw(homog, from_, to)
            result.append(Homography(homog / homog[2, 2]))
    return result


def get_images(path):
    """Get all the images in the given training data folder (passed as a
    Path object), returned as (unordered) Path objects in an iterable.

    """
    return (p for p in path.iterdir() if (
        p.is_file() and p.suffix.lower() not in ('.csv', '.kw18')
    ))


def generate_pairs_forSiamese(
        input_file, out_file, img_sample_rate, pos_sample_rate,
):

    tracks, dets, ranges = load_track_feature_file(input_file)

    neg_dict = defaultdict(set)

    pairs = []
    for track_states, (start, stop) in zip(tracks.values(), tqdm(ranges)):
        enum_track_states = list(enumerate(track_states))
        for i, (_, p_first_det) in enum_track_states[::img_sample_rate]:
            # generate positive pairs
            for j, (_, p_pos_det) in enum_track_states[i + 1::pos_sample_rate]:
                p_pos_idx = j + start
                p_neg_idx = choice_reject(concat(
                    range(0, start),
                    range(stop, len(dets)),
                ), neg_dict[p_pos_idx])

                neg_dict[p_pos_idx].add(p_neg_idx)
                neg_dict[p_neg_idx].add(p_pos_idx)

                p_neg_det = dets[p_neg_idx]

                pairs.append((p_first_det, p_pos_det, p_neg_det))
    with open(out_file, 'wb') as f:
        pickle.dump(SequenceList.from_iterable(pairs), f)


def create_bbox_files(
        bbox, cur_frame, get_blob, n_grid, homography=None,
):
    """Create feature files for the given bounding box.  Arguments:
    - bbox: BoundingBox
    - cur_frame: ndarray of the current frame's image data
    - get_blob: A function taking a feature name and returning a
      DataStorage.Blob
    - n_grid: ndarray of flattened neighborhood grid
    - homography: Homography object that transforms to a reference
      coordinate frame for stabilization (optional)

    """
    x, y, w, h = bbox.min_x(), bbox.min_y(), bbox.width(), bbox.height()
    img_h, img_w = cur_frame.shape[:2]

    # Check that box is valid
    if x < 0 or x + w > img_w or y < 0 or y + h > img_h:
        raise ValueError("Invalid bounding box")

    # crop image
    crop_img = cur_frame[y:y + h, x:x + w]
    crop_img = cv2.resize(crop_img, (224, 224))

    # Compute (stabilized) bbox center
    c_x = x + w / 2
    c_y = y + h / 2
    if homography is None:
        sc_x, sc_y = c_x, c_y
    else:
        sc_x, sc_y = homography.transform([c_x, c_y])

    # store cropped image
    get_blob('img').write(cv2.imencode('.jpg', crop_img)[1].tobytes())
    # store bbox center (row, col)
    get_blob('bc').write(np.array([sc_y, sc_x], dtype=np.float32).tobytes())
    # store grid
    get_blob('grid').write(n_grid.numpy().tobytes())
    # store bbox area and ratio
    get_blob('bbar').write(np.array([w*h, w/h], dtype=np.float32).tobytes())


def generate_feature_files(
        root_path, data_storage, make_video_id,
        out_detections_file, grid_num, stabilized,
):
    """Find source data in root_path (grandchild directories named "img1")
    and create a track feature file out_detections_file.  The track
    feature file holds a pickled dict of the form {key:
    [DetectionId]}, where the key identifies the track that the
    DetectionIds contain the derived data for.  Files are created via
    data_storage, a DataStorage, and make_video_id, which should be
    data_storage.video_id with its phase parameter specified.

    If the stabilized parameter is true, compute stabilized results
    (currently only replacing bbox centers).

    """
    # Dict mapping grandchild directories in root_path named "img1" to
    # a list of their files.  (All are Path objects.)
    img_dirs = (p for p in Path(root_path).glob('*/img1') if p.is_dir())
    img_dirs = {d: sorted(get_images(d)) for d in img_dirs}

    total_frames = sum(map(len, img_dirs.values()))
    pbar = tqdm(total=total_frames)

    compute_grid_features = Grid(
        grid_row=grid_num, grid_cols=grid_num,
        target_neighborhood_w=grid_num // 2,
    )

    tracks = {}
    for dirpath, filenames in img_dirs.items():
        # Maps track IDs to their number of occurrences
        gt_file_path = dirpath.parent / 'gt.kw18'

        frame_tracks = process_gt_file(gt_file_path)

        if stabilized:
            homographies = load_homographies(dirpath.parent / 'homog.txt')
            if min(len(filenames), len(frame_tracks)) > len(homographies):
                raise ValueError('All annotated frames must have a homography when stabilization is enabled')
        else:
            homographies = itertools.repeat(None)

        vid_name = dirpath.parent.name
        vid = make_video_id(vid_name)

        # for crop image and bbox center
        for frame_id, img_path, bb_info, homog in zip(
                itertools.count(), filenames, frame_tracks, homographies,
        ):
            if not bb_info:
                pbar.update()
                continue

            def filter_bb_info(img_w, img_h, bb_info):
                result = []
                for tid, bb in bb_info:
                    bb = restrict_bounding_box(img_w, img_h, bb)
                    if bb is not None:
                        result.append((tid, bb))
                return result

            cur_frame = cv2.imread(str(img_path))
            img_h, img_w = cur_frame.shape[:2]
            bb_info = filter_bb_info(img_w, img_h, bb_info)
            n_grids = compute_grid_features(
                (img_w, img_h), [bb for _, bb in bb_info],
            )

            # for each bbox
            for (track_id, bb), n_grid in zip(bb_info, n_grids):
                track_states = tracks.setdefault((vid, track_id), [])
                did = data_storage.detection_id(vid, track_id, frame_id)
                gb = functools.partial(data_storage.blob, did)
                create_bbox_files(
                    bb, cur_frame, gb, n_grid, homography=homog,
                )
                track_states.append((frame_id, did))

            pbar.update()
        pbar.update(max(0, len(filenames) - len(frame_tracks)))

    with open(out_detections_file, 'wb') as f:
        pickle.dump(tracks, f)


def create_parser():
    parser = argparse.ArgumentParser(description='processing kw18 file for generating training data')
    parser.add_argument('--root-path',
                        help='The root path contains all training data',
                        default='/home/bdong/HiDive_project/non-itar-training_files')
    parser.add_argument('--out-path',
                        help='The output path where stores all cropped images',
                        default='/home/bdong/HiDive_project/non-itar-training_files/cropped_labeled')
    parser.add_argument('--out-file-prefix',
                        help='output files prefix', default='out')
    parser.add_argument('--fix-seq-flag', type=bool,
                        help='fix_seq_flag=True:  generate fixed size seq for training and testing;\
                              fix_seq_flag=False: generate variable size seq for training and testing;',
                        default=False)
    parser.add_argument('--grid-num', type=int,
                        help='grid number', default=15)

    # The following two parameters are for reducing siamese training data.
    # The lower the more training data are generated
    parser.add_argument('--siamese-img-sample-rate', type=int,
                        help='sample rate of generating Siamese training data', default=8)
    parser.add_argument('--siamese-pos-sample-rate', type=int,
                        help='sample rate of generating positive Siamese training data', default=10)

    parser.add_argument('--siamese-training', dest='siamese_flag',
                        help='Generating required files for training Siamese model', action='store_true')
    parser.add_argument('--RNN-training', dest='siamese_flag',
                        help='Generating required files for training all RNN models', action='store_false')
    parser.set_defaults(siamese_flag=True)
    parser.add_argument('--stabilized', action='store_true',
                        help='Use homog.txt files in the training directories'
                        ' to produce training data using stabilized coordinates'
                        ' (currently only bbox centers and hence motion)')

    parser.add_argument('--aug-occlusion-mean-gap', type=float,
                        help='Average gap (in frames) between the start of augmented occlusions')
    parser.add_argument('--aug-occlusion-mean-length', type=float,
                        help='Mean length (in frames) of augmented occlusions')
    parser.add_argument('--aug-occlusion-stddev-length', type=float,
                        help='Standard deviation of the length (in frames) of augmented occlusions')
    parser.add_argument('--aug-occlusion-dist',
                        help='Distribution used for occlusion lengths')
    return parser


def process_train_or_test(tt, data_storage, args):
    print("Generating {} files...".format(tt))
    out_dets_file = '{}_{}_features.p'.format(args.out_file_prefix, tt)

    if args.siamese_flag:
        data_storage.create()

        def make_vid(name): return data_storage.video_id(name, tt)
        generate_feature_files(
            os.path.join(args.root_path, tt), data_storage, make_vid,
            out_dets_file, args.grid_num, stabilized=args.stabilized,
        )
        # generate training and testing files for Siamese network training
        siamese_tt_file_name = '{}_siamese_{}_set.p'.format(args.out_file_prefix, tt)
        generate_pairs_forSiamese(out_dets_file, siamese_tt_file_name,
                                  args.siamese_img_sample_rate, args.siamese_pos_sample_rate)
    else:
        aug_occlusion_args = (
            args.aug_occlusion_mean_gap, args.aug_occlusion_mean_length,
            args.aug_occlusion_stddev_length, args.aug_occlusion_dist,
        )
        if any(a is None for a in aug_occlusion_args):
            if not all(a is None for a in aug_occlusion_args):
                raise ValueError("All occlusion augmentation arguments must be provided if any is")
            aug_occlusion = None
        else:
            aug_occlusion = AugOcclusion(*aug_occlusion_args)

        # generate training and testing file for pre-training each component
        flag_str = 'F' if args.fix_seq_flag else 'V'

        out_seqs_file = '{}_{}_{}_set.p'.format(args.out_file_prefix, flag_str, tt)
        print('Generating sequences')
        generate_sequences(
            out_dets_file, out_seqs_file, time_seq_len=6,
            aug_occlusion=aug_occlusion, fixSize_flag=args.fix_seq_flag,
        )


if __name__ == '__main__':
    args = create_parser().parse_args()

    if args.siamese_img_sample_rate < 1:
        raise ValueError('siamese_img_sample_rate needs to be larger or equal to 1!')

    if args.siamese_pos_sample_rate < 1:
        raise ValueError('siamese_pos_sample_rate needs to be larger or equal to 1!')

    os.makedirs(args.out_path, exist_ok=True)
    with DataStorage(args.out_path) as data_storage:
        for tt in ['train', 'test']:
            process_train_or_test(tt, data_storage, args)
