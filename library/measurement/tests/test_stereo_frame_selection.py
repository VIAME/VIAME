"""What `stereo_frame_selection` guarantees beyond the golden replay.

`tests/golden/measurement`'s `calibration_pipeline` cases run this inside the
shipped pipeline and hold the calibration that comes out to the recording and
to ground truth. What is here is the pieces on their own: the frame grouping,
the correspondence filter, the extent matrix and the k-medians, on inputs
small enough to reason about by hand.

It replaces `tests/plugins/opencv/test_filter_stereo_feature_tracks.cxx`,
which asserted that three empty inputs gave three empty outputs.

Run just these:  ctest -R "unit:measurement"
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def selection():
    from viame.measurement import stereo_frame_selection
    return stereo_frame_selection


def track(identifier, states):
    """A feature track: `states` is a list of `(frame, x, y)`."""
    from kwiver.vital.types import FeatureD, FeatureTrackState, Track
    from kwiver.vital.types.descriptor import new_descriptor

    out = Track(identifier)
    descriptor = new_descriptor(1, "d")

    for frame, x, y in states:
        out.append(FeatureTrackState(
            frame, FeatureD(loc=np.array([float(x), float(y)])), descriptor))

    return out


def track_set(tracks):
    from kwiver.vital.types import FeatureTrackSet
    return FeatureTrackSet(tracks)


def landmark_map(points):
    """`{id: (x, y, z)}` as a landmark map."""
    from kwiver.vital.types import LandmarkD, SimpleLandmarkMap

    return SimpleLandmarkMap({
        identifier: LandmarkD(np.array([float(v) for v in point]))
        for identifier, point in points.items()})


# ----------------------------------------------------------------------------
# Grouping and correspondence
# ----------------------------------------------------------------------------

def test_empty_input_gives_empty_output(selection):
    result = selection.select_frames([], [], 10)

    assert len(result.world_pts) == 0
    assert len(result.frame_ids) == 0
    assert not result.is_usable()


def test_a_stereo_pair_pairs_by_track_id(selection):
    left = track_set([track(0, [(0, 10, 10), (1, 11, 10)]),
                      track(1, [(0, 20, 10), (1, 21, 10)])])
    right = track_set([track(0, [(0, 5, 10), (1, 6, 10)]),
                       track(1, [(0, 15, 10), (1, 16, 10)])])

    world = landmark_map({0: (0, 0, 0), 1: (30, 0, 0)})

    result = selection.select_frames([left, right], [world, world], 0)

    assert len(result.frame_ids) == 2
    assert result.frame_ids == [0, 1]
    assert result.is_usable()

    # Frame zero: both tracks, in track id order, with their world points
    assert result.image_pts[0][0].tolist() == [[10, 10], [20, 10]]
    assert result.image_pts[1][0].tolist() == [[5, 10], [15, 10]]
    assert result.world_pts[0].tolist() == [[0, 0, 0], [30, 0, 0]]


def test_a_frame_one_camera_missed_is_dropped(selection):
    """A stereo calibration needs the correspondence, so a frame where the
    two cameras saw different tracks is no use and is not half-used."""
    left = track_set([track(0, [(0, 10, 10), (1, 11, 10)]),
                      track(1, [(0, 20, 10), (1, 21, 10)])])
    # The right camera missed track 1 on frame 1
    right = track_set([track(0, [(0, 5, 10), (1, 6, 10)]),
                       track(1, [(0, 15, 10)])])

    world = landmark_map({0: (0, 0, 0), 1: (30, 0, 0)})

    result = selection.select_frames([left, right], [world, world], 0)

    assert result.frame_ids == [0]


def test_a_track_with_no_landmark_is_dropped(selection):
    left = track_set([track(0, [(0, 10, 10)]), track(1, [(0, 20, 10)])])
    right = track_set([track(0, [(0, 5, 10)]), track(1, [(0, 15, 10)])])

    # Only track 0 has a world point
    world = landmark_map({0: (0, 0, 0)})

    result = selection.select_frames([left, right], [world, world], 0)

    assert result.world_pts[0].tolist() == [[0, 0, 0]]


def test_landmark_maps_that_disagree_drop_the_track(selection):
    """Both maps have the id and give different points, so nothing is known
    about where the corner actually is."""
    left = track_set([track(0, [(0, 10, 10)])])
    right = track_set([track(0, [(0, 5, 10)])])

    result = selection.select_frames(
        [left, right],
        [landmark_map({0: (0, 0, 0)}), landmark_map({0: (1, 0, 0)})], 0)

    assert len(result.world_pts[0]) == 0


# ----------------------------------------------------------------------------
# The extent matrix
# ----------------------------------------------------------------------------

def test_the_extent_matrix_only_fills_the_matched_corner(selection):
    """Sixteen columns, four of them ever used.

    `world_point_corner_values` takes its bounds from the **first** world
    point of each frame rather than from all of them, so on a board whose
    point order is fixed all four "corners" are that one point and only the
    first extent is ever written. Reproduced from the C++; the recording of
    the whole pipeline is what says it does not matter in practice.
    """
    coordinates = selection.StereoPointCoordinates()
    coordinates.frame_ids = [0]
    coordinates.image_pts[0] = [np.array([[7.0, 8.0], [70.0, 80.0]],
                                         dtype=np.float32)]
    coordinates.image_pts[1] = [np.array([[3.0, 4.0], [30.0, 40.0]],
                                         dtype=np.float32)]
    coordinates.world_pts = [np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
                                      dtype=np.float32)]

    matrix = selection.frames_extents_matrix(coordinates)

    assert matrix.shape == (1, selection.EXTENT_SIZE)
    assert matrix[0, 0] == 7.0 and matrix[0, 1] == 8.0
    assert matrix[0, 8] == 3.0 and matrix[0, 9] == 4.0
    assert not matrix[0, 2:8].any()
    assert not matrix[0, 10:].any()


# ----------------------------------------------------------------------------
# k-medians
# ----------------------------------------------------------------------------

def test_kmedians_separates_two_obvious_groups(selection):
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                     [100.0, 100.0], [101.0, 100.0], [100.0, 101.0]],
                    dtype=np.float32)

    labels, centres = selection.kmedians(data, 2)

    assert centres.shape == (2, 2)
    # The three near the origin share a label, and so do the three far off
    assert len(set(labels[:3].tolist())) == 1
    assert len(set(labels[3:].tolist())) == 1
    assert labels[0] != labels[3]


def test_kmedians_leaves_no_cluster_empty(selection):
    """A median of no points is zero, which would drag the next iteration
    towards the origin, so an empty cluster is given the point furthest from
    the biggest one."""
    data = np.array([[0.0], [0.0], [0.0], [1.0], [50.0]], dtype=np.float32)

    labels, centres = selection.kmedians(data, 3)

    assert len(set(labels.tolist())) == 3


def test_the_median_of_an_even_count_is_the_mean_of_the_middle_two(selection):
    """Which is the C++'s rule, and not what `numpy.median` would give for
    an odd count offset by one."""
    assert selection._median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert selection._median([1.0, 2.0, 3.0]) == 2.0
    assert selection._median([]) == 0.0


def test_a_tie_goes_to_the_later_cluster(selection):
    """`<=` in the C++, not `<`. It decides nothing on real data and
    everything on a symmetric example, so it is pinned."""
    distances = np.array([[5.0, 5.0, 9.0]])
    assert selection._update_labels(distances).tolist() == [1]


def test_frame_selection_is_off_below_the_threshold(selection):
    coordinates = selection.StereoPointCoordinates()
    coordinates.frame_ids = [0, 1, 2]
    coordinates.world_pts = [np.zeros((1, 3), dtype=np.float32)] * 3
    coordinates.image_pts = [[np.zeros((1, 2), dtype=np.float32)] * 3] * 2

    assert len(selection.select_points_maximizing_variance(
        coordinates, 0).frame_ids) == 3
    assert len(selection.select_points_maximizing_variance(
        coordinates, 5).frame_ids) == 3
