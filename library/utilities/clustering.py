# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""k-means, for the callers that need a reproducible one.

**The seed is an argument, deliberately.** A k-means that draws from a global
generator gives a different answer depending on what else in the process has
drawn from it -- the trap finding 2.71 records for grabCut -- and a caller that
wants to repeat an augmentation needs the opposite. Passing the seed in makes
the result a function of its inputs and nothing else.

k-means++ for the initial centres, then Lloyd's iteration until the centres
stop moving or the iteration limit runs out.
"""

import numpy as np

__all__ = ["kmeans"]


def kmeans(samples, clusters, seed=0, max_iterations=100, epsilon=0.0,
           attempts=1):
    """`(compactness, labels, centres)` for `samples`, one row per sample.

    `compactness` is the total squared distance from each sample to its
    centre, and is what picking the best of several `attempts` is decided on.
    """
    samples = np.asarray(samples, dtype=np.float64)

    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    clusters = int(clusters)

    if clusters < 1 or clusters > len(samples):
        raise ValueError("{} clusters for {} samples".format(clusters,
                                                            len(samples)))

    best = None

    for attempt in range(max(int(attempts), 1)):
        generator = np.random.default_rng(int(seed) + attempt)
        centres = _plus_plus(samples, clusters, generator)

        for _step in range(max(int(max_iterations), 1)):
            labels = _assign(samples, centres)
            moved = 0.0

            for cluster in range(clusters):
                members = samples[labels == cluster]

                if not len(members):
                    continue

                middle = members.mean(axis=0)
                moved = max(moved,
                            float(((middle - centres[cluster]) ** 2).sum()))
                centres[cluster] = middle

            if moved <= float(epsilon) ** 2:
                break

        labels, nearest = _assign(samples, centres, with_distances=True)
        compactness = float(nearest.sum())

        if best is None or compactness < best[0]:
            best = (compactness, labels, centres)

    compactness, labels, centres = best

    return compactness, labels, centres.astype(np.float32)


def _distances(samples, centres):
    """Squared distance from every sample to every centre."""
    from scipy.spatial.distance import cdist
    return cdist(samples, centres, metric="sqeuclidean")


def _assign(samples, centres, with_distances=False):
    # Bound the distance workspace to about 8 MiB, independent of image size.
    chunk = max(1, (1024 * 1024) // len(centres))
    labels = np.empty(len(samples), dtype=np.int32)
    nearest = np.empty(len(samples), dtype=np.float64) if with_distances else None
    for start in range(0, len(samples), chunk):
        distances = _distances(samples[start:start + chunk], centres)
        chosen = np.argmin(distances, axis=1)
        labels[start:start + chunk] = chosen
        if with_distances:
            nearest[start:start + chunk] = distances[np.arange(len(chosen)), chosen]
    return (labels, nearest) if with_distances else labels


def _plus_plus(samples, clusters, generator):
    """k-means++ seeding: each centre drawn in proportion to its distance."""
    centres = np.empty((clusters, samples.shape[1]), dtype=np.float64)
    centres[0] = samples[generator.integers(len(samples))]

    nearest = np.full(len(samples), np.inf)
    for chosen in range(1, clusters):
        np.minimum(nearest, _distances(samples, centres[chosen - 1:chosen])[:, 0],
                   out=nearest)
        total = nearest.sum()

        if total <= 0.0:
            centres[chosen] = samples[generator.integers(len(samples))]
            continue

        centres[chosen] = samples[generator.choice(len(samples),
                                                   p=nearest / total)]

    return centres
