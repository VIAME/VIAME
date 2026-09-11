# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Turning stereo feature tracks into calibration point sets.

`plugins/opencv/filter_stereo_feature_tracks.cxx` and `kmedians.cxx` in
python, per `lite-removals.md` section 2.4. Nothing here is OpenCV except
`cv2.kmeans`, which seeds the k-medians refinement.

What it does, in the order `select_frames` does it:

1. group the tracks of both cameras by frame, as a per-frame table indexed by
   camera and then by track id, with holes where a track was not seen;
2. drop the frames where the two cameras did not see the same set of tracks,
   since a stereo calibration needs the correspondence;
3. pair each frame's tracks with the world point of the same id, giving the
   2D and 3D point lists a calibration takes;
4. if a frame budget is set, cluster the frames by where the board's four
   corners landed and keep the frame nearest each cluster's centre -- which
   is a spread of board poses rather than the first N.

Step four is off by default (`frame_count_threshold` is 0), and
`tests/golden/measurement`'s `frames_6` variant is what turns it on so this
is not untested code.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The extent matrix has eight numbers per camera: the x and y of each of the
# board's four world-space corners, as they landed in that camera's image.
EXTENT_SIZE = 16
HALF_EXTENT_SIZE = EXTENT_SIZE // 2

# How close two world coordinates must be to be the same corner.
CORNER_TOLERANCE = 1e-6

# `cv::TermCriteria( EPS + COUNT, 50, 1.0 )` and five attempts, from the C++.
KMEANS_MAX_ITERATIONS = 50
KMEANS_EPSILON = 1.0
KMEANS_ATTEMPTS = 5

# When two centre sets count as unchanged, which is what ends the refinement.
CENTRE_TOLERANCE = 1e-9


class StereoPointCoordinates(object):
    """The 2D points of each camera and the 3D points they correspond to.

    `image_pts[camera][frame]` is an (n, 2) array and `world_pts[frame]` an
    (n, 3); `frame_ids[frame]` is the frame each came from.
    """

    def __init__(self):
        self.image_pts = [[], []]
        self.world_pts = []
        self.frame_ids = []

    def __len__(self):
        return len(self.frame_ids)

    def is_usable(self):
        """What the C++ checked before calibrating."""
        return (bool(self.image_pts[0]) and
                len(self.image_pts[0]) == len(self.world_pts) and
                len(self.image_pts[0]) == len(self.image_pts[1]))


def _states_by_track(feature_track_sets):
    """`group_by_frame_id`: a per-frame table of states, by camera and track.

    The shape is the C++'s: a list per frame, holding a list per camera,
    holding one slot per track id with `None` where that track was not seen
    in that frame. A frame with no state at all is dropped.
    """
    if not feature_track_sets:
        return []

    max_frame = -1
    max_track = -1

    for features in feature_track_sets:
        frames = list(features.all_frame_ids())
        tracks = [track.id for track in features.tracks()]

        if frames:
            max_frame = max(max_frame, max(frames))
        if tracks:
            max_track = max(max_track, max(tracks))

    if max_frame < 0 or max_track < 0:
        return []

    table = [None] * (max_frame + 1)

    for camera, features in enumerate(feature_track_sets):
        for track in features.tracks():
            for state in track:
                frame = state.frame_id

                if table[frame] is None:
                    table[frame] = [None] * len(feature_track_sets)

                if table[frame][camera] is None:
                    table[frame][camera] = [None] * (max_track + 1)

                table[frame][camera][track.id] = (track.id, state)

    return [frame for frame in table if frame is not None]


def _remove_frames_without_both_cameras(table):
    """`remove_frames_without_corresponding_left_right_match`.

    A single camera needs no correspondence; two need the same set of track
    ids in both, and a frame where they differ is dropped rather than
    half-used.
    """
    if all(len(frame) == 1 for frame in table):
        return table

    def ids(frame, camera):
        entries = frame[camera] or []
        return {entry[0] for entry in entries if entry is not None}

    return [frame for frame in table
            if len(frame) == 1 or ids(frame, 0) == ids(frame, 1)]


def points_from_frames(table, landmark_maps):
    """`StereoPointCoordinates::from_features`.

    A track is kept when both cameras saw it and a landmark of the same id
    exists -- and, if both landmark maps have one, when the two agree.
    """
    coordinates = StereoPointCoordinates()

    if not table or not landmark_maps:
        return coordinates

    left_landmarks = landmark_maps[0].landmarks()
    right_landmarks = (landmark_maps[1].landmarks()
                       if len(landmark_maps) > 1 else left_landmarks)

    for frame in table:
        first = frame[0] or []
        second = (frame[1] if len(frame) > 1 else frame[0]) or []

        if len(first) != len(second):
            continue

        left_points = []
        right_points = []
        world_points = []
        frame_ids = set()

        for entry1 in first:
            if entry1 is None:
                continue

            track_id, state1 = entry1

            for entry2 in second:
                if entry2 is None:
                    continue

                other_id, state2 = entry2

                if track_id != other_id:
                    continue

                in_left = track_id in left_landmarks
                in_right = track_id in right_landmarks

                if in_left:
                    left_world = np.asarray(left_landmarks[track_id].loc,
                                            dtype=np.float64)
                if in_right:
                    right_world = np.asarray(right_landmarks[track_id].loc,
                                             dtype=np.float64)

                if in_left and in_right:
                    found = bool(np.array_equal(left_world, right_world))
                elif in_left or in_right:
                    found = True
                else:
                    found = False

                if not found:
                    continue

                left_points.append(np.asarray(state1.feature.location,
                                              dtype=np.float64))
                right_points.append(np.asarray(state2.feature.location,
                                               dtype=np.float64))
                world_points.append(left_world if in_left else right_world)
                frame_ids.add(state1.frame_id)

        coordinates.frame_ids.append(sorted(frame_ids)[0] if frame_ids else 0)
        coordinates.image_pts[0].append(
            np.array(left_points, dtype=np.float32).reshape(-1, 2))
        coordinates.image_pts[1].append(
            np.array(right_points, dtype=np.float32).reshape(-1, 2))
        coordinates.world_pts.append(
            np.array(world_points, dtype=np.float32).reshape(-1, 3))

    return coordinates


def world_point_corner_values(world_pts):
    """The four corners of the board, from the bounds of the world points.

    Note what the C++ does and this keeps: the bounds come from the **first**
    point of each frame only, not from every point. On a board whose point
    order is the same in every frame -- which is what a chessboard detector
    gives -- that first point is the same corner each time, so the bounds are
    over one corner's positions rather than over the board. It works because
    `get_destination_extent` then matches world coordinates exactly and the
    board's own corners are among them.
    """
    if not world_pts:
        return []

    firsts = np.array([frame[0] for frame in world_pts if len(frame)],
                      dtype=np.float64)

    if not len(firsts):
        return []

    min_x, min_y = float(firsts[:, 0].min()), float(firsts[:, 1].min())
    max_x, max_y = float(firsts[:, 0].max()), float(firsts[:, 1].max())

    return [(min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y)]


def destination_extent(world_point, corners):
    """Which pair of extent columns a world point belongs in, or None."""
    for index, corner in enumerate(corners):
        if (abs(corner[0] - world_point[0]) < CORNER_TOLERANCE and
                abs(corner[1] - world_point[1]) < CORNER_TOLERANCE):
            return index * 2, index * 2 + 1

    return None


def frames_extents_matrix(coordinates):
    """`create_frames_extents_matrix`: one row per frame, sixteen columns."""
    frames = len(coordinates.world_pts)
    matrix = np.zeros((frames, EXTENT_SIZE), dtype=np.float32)
    corners = world_point_corner_values(coordinates.world_pts)

    if not corners:
        return matrix

    for index, frame in enumerate(coordinates.world_pts):
        for point in range(len(frame)):
            extent = destination_extent(frame[point], corners)

            if extent is None:
                continue

            x, y = extent
            matrix[index, x] = coordinates.image_pts[0][index][point][0]
            matrix[index, y] = coordinates.image_pts[0][index][point][1]
            matrix[index, x + HALF_EXTENT_SIZE] = \
                coordinates.image_pts[1][index][point][0]
            matrix[index, y + HALF_EXTENT_SIZE] = \
                coordinates.image_pts[1][index][point][1]

    return matrix


# ----------------------------------------------------------------------------
# k-medians
# ----------------------------------------------------------------------------

def _manhattan(data, centres):
    """Every point's L1 distance to every centre, as (points, centres)."""
    return np.abs(data[:, None, :] - centres[None, :, :]).sum(axis=-1)


def _update_labels(distances):
    """The nearest centre per point, ties going to the **last** one.

    `<=` in the C++, not `<`, which is what makes the later centre win.
    """
    best = np.full(distances.shape[0], 0, dtype=np.int32)
    lowest = np.full(distances.shape[0], np.finfo(np.float32).max,
                     dtype=np.float64)

    for cluster in range(distances.shape[1]):
        column = distances[:, cluster]
        take = column <= lowest
        best[take] = cluster
        lowest[take] = column[take]

    return best


def _median(values):
    """The C++'s median: the mean of the middle two for an even count."""
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = len(ordered)

    if count == 0:
        return 0.0

    if count % 2 == 0:
        return float((ordered[count // 2 - 1] + ordered[count // 2]) / 2.0)

    return float(ordered[count // 2])


def _update_medians(data, labels, clusters):
    centres = np.zeros((clusters, data.shape[1]), dtype=np.float32)

    for cluster in range(clusters):
        members = data[labels == cluster]

        for feature in range(data.shape[1]):
            centres[cluster, feature] = _median(
                members[:, feature] if len(members) else [])

    return centres


def find_closest_frame_to_centre(data, labels, centres, cluster):
    """The member of `cluster` nearest its centre.

    A cluster with no members gives index zero, which is what the C++'s
    uninitialised `i_selected` happens to be and what the empty-cluster
    repair relies on not happening.
    """
    best = 0
    lowest = np.inf

    for index in range(data.shape[0]):
        if labels[index] != cluster:
            continue

        distance = float(np.abs(data[index] - centres[cluster]).sum())

        if distance < lowest:
            best = index
            lowest = distance

    return best


def _find_furthest_frame_from_centre(data, labels, centres, cluster):
    best = 0
    highest = -np.inf

    for index in range(data.shape[0]):
        if labels[index] != cluster:
            continue

        distance = float(np.abs(data[index] - centres[cluster]).sum())

        if distance > highest:
            best = index
            highest = distance

    return best


def _repair_empty_clusters(data, labels, centres):
    """Give every empty cluster the point furthest from the biggest one.

    `make_sure_no_center_is_empty`. Without it a median of no points is zero,
    which drags the next iteration's assignment towards the origin.
    """
    clusters = centres.shape[0]

    while len(set(labels.tolist())) < clusters:
        empty = next(cluster for cluster in range(clusters)
                     if cluster not in set(labels.tolist()))

        counts = np.bincount(labels, minlength=clusters)
        biggest = int(np.argmax(counts))

        furthest = _find_furthest_frame_from_centre(
            data, labels, centres, biggest)

        labels[furthest] = empty
        centres = _update_medians(data, labels, clusters)

    return labels, centres


def kmedians(data, clusters):
    """`viame::kmedians`: k-means for the initial centres, then medians.

    The refinement runs until the centres stop moving, which is what the C++
    does; `cv2.kmeans` only provides the starting point.
    """
    import cv2

    _, labels, centres = cv2.kmeans(
        data, clusters, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
         KMEANS_MAX_ITERATIONS, KMEANS_EPSILON),
        KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS)

    labels = labels.reshape(-1).astype(np.int32)
    centres = np.asarray(centres, dtype=np.float32)

    while True:
        previous = centres.copy()

        distances = _manhattan(data.astype(np.float64),
                               centres.astype(np.float64))
        labels = _update_labels(distances)
        centres = _update_medians(data, labels, clusters)
        labels, centres = _repair_empty_clusters(data, labels, centres)

        if (previous.shape == centres.shape and
                np.abs(previous - centres).max() <= CENTRE_TOLERANCE):
            break

    return labels, centres


def select_points_maximizing_variance(coordinates, frame_count_threshold):
    """Keep one frame per cluster of board poses, or all of them."""
    frames = len(coordinates.frame_ids)

    if frame_count_threshold == 0 or frames <= frame_count_threshold:
        return coordinates

    matrix = frames_extents_matrix(coordinates)
    labels, centres = kmedians(matrix, int(frame_count_threshold))

    kept = sorted({find_closest_frame_to_centre(matrix, labels, centres,
                                                cluster)
                   for cluster in range(centres.shape[0])})

    selected = StereoPointCoordinates()

    for index in kept:
        selected.image_pts[0].append(coordinates.image_pts[0][index])
        selected.image_pts[1].append(coordinates.image_pts[1][index])
        selected.world_pts.append(coordinates.world_pts[index])
        selected.frame_ids.append(coordinates.frame_ids[index])

    return selected


def select_frames(feature_track_sets, landmark_maps, frame_count_threshold):
    """`filter_stereo_feature_tracks::select_frames`."""
    table = _states_by_track(feature_track_sets)
    table = _remove_frames_without_both_cameras(table)

    coordinates = points_from_frames(table, landmark_maps)

    return select_points_maximizing_variance(coordinates,
                                             frame_count_threshold)
