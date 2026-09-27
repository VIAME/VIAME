# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Exhaustive descriptor matching, which is what replaced FLANN and BFMatcher.

`cv2.BFMatcher` and `cv2.FlannBasedMatcher` are both gone from this tree, and
neither is reproduced: FLANN builds randomised KD-trees seeded from the clock,
so it cannot reproduce even itself, and BFMatcher is this with a different
spelling. Searching every candidate over a frame pair's descriptors is a matrix
multiply, and the answer is the true nearest neighbour, the same one every run.

See `ocv_flann_matcher.py` for the algorithm that wraps this, and
lite-findings.md 2.63 for what it measured.
"""

import numpy as np


def as_matrix(descriptor_set, binary):
    """A `vital::descriptor_set` as one (n, d) matrix.

    `binary` because a binary descriptor's bytes are a bit string rather than a
    number: the C++ bridge chose `CV_8U` against `CV_32F` by the descriptor's
    own type, which python cannot see, so the caller says instead.
    """
    if descriptor_set is None or descriptor_set.size() == 0:
        return None

    rows = [np.asarray(d.todoublearray())
            for d in descriptor_set.descriptors()]

    if not rows:
        return None

    return np.vstack(rows).astype(np.uint8 if binary else np.float32)


def nearest(query, train, k, binary, with_distance=False):
    """The indices of the `k` nearest rows of `train` for each row of `query`.

    With `with_distance`, returns `(indices, distances)`, the distances being
    **rooted** L2 for a float descriptor and plain Hamming counts for a binary
    one -- which is what `cv2.BFMatcher` reports and therefore what a ratio
    test's threshold is written against.

    Nearest by **squared** L2 for a float descriptor and by Hamming distance
    for a binary one, which are the two metrics `cv::FlannBasedMatcher` chooses
    between for the same reason. Squared rather than rooted because the order is
    all that is used and the root does not change it.

    Ties go to the lower index, which is `argsort`'s stable order and matches
    what a linear scan keeping a strict improvement would do.
    """
    count = train.shape[0]
    k = min(max(1, int(k)), count)

    if binary:
        # A byte at a time, so the whole cross product is not held at once:
        # unpacking 8 bits per byte makes the intermediate 8 times the input.
        bits_query = np.unpackbits(query, axis=1).astype(np.uint16)
        bits_train = np.unpackbits(train, axis=1).astype(np.uint16)
        distance = (bits_query[:, None, :] != bits_train[None, :, :]).sum(
            axis=2, dtype=np.int32)
    else:
        # |a - b|^2 = |a|^2 - 2ab + |b|^2, in float64 so that the subtraction
        # cannot cancel into a negative distance and reorder the neighbours.
        a = query.astype(np.float64)
        b = train.astype(np.float64)
        distance = ((a * a).sum(1)[:, None] - 2.0 * (a @ b.T) +
                    (b * b).sum(1)[None, :])

    rows = np.arange(distance.shape[0])[:, None]

    if k == 1:
        index = np.argmin(distance, axis=1)[:, None]
    else:
        # `argpartition` then sort just the k kept, rather than sorting every
        # column: the cross product is the expensive part and k is 1 or 2 here.
        partial = np.argpartition(distance, k - 1, axis=1)[:, :k]
        order = np.argsort(distance[rows, partial], axis=1, kind="stable")
        index = partial[rows, order]

    if not with_distance:
        return index

    found = distance[rows, index]

    # Rooted here and not before, because a ratio test compares distances and
    # the ratio of two squares is not the ratio of their roots. `cv2.BFMatcher`
    # reports the L2 distance, so a caller's 0.75 threshold means the rooted
    # one.
    return index, (found if binary else np.sqrt(found))


def match(query, train, cross_check=False, k=1, binary=False):
    """(query index, train index) pairs, one per query row at most.

    Without `cross_check`, every query row is matched to its nearest train row,
    in query order -- `cv::DescriptorMatcher::match`. With it, a pair survives
    only when the train row has the query row among **any** of its own `k`
    nearest, which is the rule the C++ `cross_check_match` used and is weaker
    than a strict mutual best.
    """
    if query is None or train is None or not len(query) or not len(train):
        return []

    if query.shape[1] != train.shape[1]:
        raise ValueError(
            "descriptor widths differ: {} against {}".format(
                query.shape[1], train.shape[1]))

    if not cross_check:
        return [(index, int(candidates[0])) for index, candidates
                in enumerate(nearest(query, train, 1, binary))]

    k = max(1, int(k))
    forward = nearest(query, train, k, binary)
    backward = nearest(train, query, k, binary)

    kept = []

    for index, candidates in enumerate(forward):
        for candidate in candidates:
            if index in backward[candidate]:
                kept.append((index, int(candidate)))
                break

    return kept


def ratio_match(query, train, ratio=0.75, binary=False):
    """(query, train) pairs surviving Lowe's ratio test.

    The two nearest train rows per query row, kept only when the nearer is
    closer than `ratio` times the second -- `knnMatch( ..., k=2 )` followed by
    `m[0].distance < ratio * m[1].distance`, which is how every caller in this
    tree used a brute force matcher.

    A query row with fewer than two candidates to compare is dropped, as the
    `len(m) == 2` guard in those callers dropped it.
    """
    if query is None or train is None or not len(query) or len(train) < 2:
        return []

    if query.shape[1] != train.shape[1]:
        raise ValueError(
            "descriptor widths differ: {} against {}".format(
                query.shape[1], train.shape[1]))

    index, distance = nearest(query, train, 2, binary, with_distance=True)
    keep = distance[:, 0] < float(ratio) * distance[:, 1]

    return [(int(q), int(index[q, 0])) for q in np.flatnonzero(keep)]
