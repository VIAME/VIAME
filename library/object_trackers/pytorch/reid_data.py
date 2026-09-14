# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""The Re-ID training data BoT-SORT's and DeepSORT's trainers share.

Crops on disk one directory per track, the P-by-K batch sampler triplet loss
needs, and the frame bounds the sequence maps are checked against. The two
trainers each had an identical copy until P2-T07.

torch stays optional to import: without it the dataset and sampler subclass
`object`, so a build or a test that never trains can still import the
trainers.
"""

import random
from pathlib import Path

try:
    from torch.utils.data import Dataset as _TorchDataset, Sampler as _TorchSampler
except ImportError:
    _TorchDataset = object
    _TorchSampler = object


def _frame_bounds(track_sets):
    """Highest frame id each track set refers to, or None where it refers to
    none. build_sequence_maps checks its alignment against these, since the
    number of track sets and the number of image directories need not agree.
    """
    bounds = []

    for track_set in track_sets:
        highest = None

        if track_set is not None:
            for track in track_set.tracks():
                for state in track:
                    if state.detection() is None:
                        continue
                    if highest is None or state.frame_id > highest:
                        highest = state.frame_id

        bounds.append(highest)

    return bounds


class ReIDDataset(_TorchDataset):
    """Crops on disk, one directory per track."""

    def __init__(self, data_dir, transform=None):
        from PIL import Image  # noqa: F401  (kept local, see module note)

        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_to_idx = {}

        for idx, track_dir in enumerate(sorted(self.data_dir.iterdir())):
            if not track_dir.is_dir():
                continue

            self.label_to_idx[track_dir.name] = idx
            for img_path in track_dir.glob("*.jpg"):
                self.samples.append(str(img_path))
                self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class PKSampler(_TorchSampler):
    """Yield batches of P identities with K crops each.

    Triplet loss can only produce a gradient from an anchor that has both a
    positive (same track) and a negative (different track) in the same batch.
    Drawing crops uniformly at random gives a same-identity collision with
    probability roughly B^2 / 2N for a batch of B over N identities, which for a
    track dataset this size is only a few percent -- so nearly every batch was a
    no-op and the Re-ID model never actually learned. Sampling K crops from each
    of P identities guarantees every sample has a positive.
    """

    def __init__(self, labels, p, k, num_batches=None, same_sequence=0.7,
                 names=None):
        """
        Args:
            labels: the dataset's per sample label, an integer index
            names: label index -> identity name, as ReIDDataset.label_to_idx
                holds it the other way round. Without it the sampler cannot
                tell which clip an identity came from and simply draws
                globally, which is what it always did.
        """
        self.k = max(int(k), 2)
        self.same_sequence = same_sequence

        self.by_id = {}
        for idx, label in enumerate(labels):
            self.by_id.setdefault(label, []).append(idx)

        # A track with a single crop can never supply a positive pair
        self.ids = [i for i, idxs in self.by_id.items() if len(idxs) >= 2]
        self.p = max(min(int(p), len(self.ids)), 1)

        # Identities grouped by the clip they came from. Names are written
        # seq{seq:04d}_track{id:06d}, so the clip is the part before _track.
        self.by_sequence = {}

        if names:
            for identity in self.ids:
                name = names.get(identity)

                if name is None:
                    continue

                sequence = str(name).split("_track")[0]
                self.by_sequence.setdefault(sequence, []).append(identity)

        # Only clips that can fill a batch on their own are worth drawing
        # from, otherwise the batch is mostly topped up from elsewhere and
        # the point is lost
        self.rich_sequences = [s for s, ids in self.by_sequence.items()
                               if len(ids) >= self.p]

        if num_batches is None:
            num_batches = max(len(labels) // (self.p * self.k), 1)
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def _pick_identities(self):
        """The P identities for one batch.

        Drawn from a single clip most of the time. Sampling identities
        uniformly puts each one in a batch with fish from other clips, other
        water and other lighting, and batch-hard mining will happily satisfy
        the margin on those cues rather than on what the fish looks like. A
        negative from the same clip is the one that forces an appearance
        comparison. The rest of the time the draw is global, so the embedding
        still has to separate identities across clips.
        """
        if self.rich_sequences and random.random() < self.same_sequence:
            sequence = random.choice(self.rich_sequences)
            return random.sample(self.by_sequence[sequence], self.p)

        return random.sample(self.ids, self.p)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for track_id in self._pick_identities():
                pool = self.by_id[track_id]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    # Short track, repeat crops to fill its slot
                    batch.extend(random.choices(pool, k=self.k))
            yield batch
