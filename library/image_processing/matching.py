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

    The **ordering** is by squared L2 or by Hamming count, which are the two
    metrics `cv::FlannBasedMatcher` chooses between; the root is taken only on
    the distances that are returned, since it cannot change the order.

    Ties go to the lower index, which is what a linear scan keeping a strict
    improvement would do. The blocked loop below preserves that by sorting the
    carried indices ascending before each merge, so `argmin` -- which returns
    the first minimum -- resolves a tie towards the earlier descriptor.
    """
    query, train = np.asarray(query), np.asarray(train)
    if query.ndim != 2 or train.ndim != 2 or query.shape[1] != train.shape[1]:
        raise ValueError("descriptor matrices must have matching widths")
    count = train.shape[0]
    k = min(max(1, int(k)), count)
    if not count or not len(query):
        index = np.empty((len(query), k), dtype=np.int64)
        distance = np.empty(index.shape, dtype=np.float64)
        return (index, distance) if with_distance else index

    if binary:
        from viame.image_processing._features import nearest_binary
        index, found = nearest_binary(np.ascontiguousarray(query, dtype=np.uint8),
                                      np.ascontiguousarray(train, dtype=np.uint8), k)
        return (index, found) if with_distance else index

    # Bound temporary memory independently of the number of descriptors.
    # BLAS computes each block; only the best k distances survive it.
    index = np.empty((len(query), k), dtype=np.int64)
    found = np.empty((len(query), k), dtype=np.float64)
    train = train.astype(np.float64)
    train_norm = np.einsum('ij,ij->i', train, train)
    for start in range(0, len(query), 128):
        a = query[start:start + 128].astype(np.float64)
        norm = np.einsum('ij,ij->i', a, a)[:, None]
        best = np.full((len(a), k), np.inf)
        ids = np.full((len(a), k), count, dtype=np.int64)
        for offset in range(0, count, 1024):
            b = train[offset:offset + 1024]
            distance = norm - 2.0 * (a @ b.T) + train_norm[offset:offset + len(b)]
            np.maximum(distance, 0.0, out=distance)
            order = np.argsort(ids, axis=1, kind='stable')
            ids = np.take_along_axis(ids, order, axis=1)
            best = np.take_along_axis(best, order, axis=1)
            candidates = np.broadcast_to(np.arange(offset, offset + len(b)), distance.shape)
            candidates = np.concatenate((ids, candidates), axis=1)
            distances = np.concatenate((best, distance), axis=1)
            if k <= 4:
                # All candidate indices are ascending, so argmin resolves
                # ties by index without sorting the whole distance block.
                rows = np.arange(len(a))
                best, ids = np.empty((len(a), k)), np.empty((len(a), k), dtype=np.int64)
                for column in range(k):
                    chosen = np.argmin(distances, axis=1)
                    best[:, column] = distances[rows, chosen]
                    ids[:, column] = candidates[rows, chosen]
                    distances[rows, chosen] = np.inf
            else:
                order = np.argsort(distances, axis=1, kind='stable')[:, :k]
                best = np.take_along_axis(distances, order, axis=1)
                ids = np.take_along_axis(candidates, order, axis=1)
        index[start:start + len(a)] = ids
        found[start:start + len(a)] = np.sqrt(best)
    return (index, found) if with_distance else index


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
