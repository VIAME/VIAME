"""
SMQTK LSH Nearest Neighbor Index - Locality-Sensitive Hashing for NN search.
"""
import logging
import threading

import numpy as np

from .nn_index import NearestNeighborsIndex
from .hash_index import get_hash_index_impls
from .lsh_functors import get_lsh_functor_impls
from ..representation import (
    get_descriptor_index_impls,
    get_key_value_store_impls,
    DescriptorIndex,
    MemoryDescriptorIndex,
)
from ..utils import merge_dict
from ..utils.plugin import make_config, from_plugin_config, to_plugin_config
from ..utils.metrics import (
    euclidean_distance,
    cosine_distance,
    histogram_intersection_distance,
)


DISTANCE_FUNCTIONS = {
    "euclidean": euclidean_distance,
    "cosine": cosine_distance,
    "hik": histogram_intersection_distance,
}


class LSHNearestNeighborIndex(NearestNeighborsIndex):
    """
    Locality-Sensitive Hashing (LSH) based nearest-neighbor index.

    Uses a hash functor to convert descriptors to binary codes, then indexes
    those codes for efficient approximate nearest-neighbor search.
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(LSHNearestNeighborIndex, cls).get_default_config()
        c['lsh_functor'] = make_config(get_lsh_functor_impls())
        c['descriptor_index'] = make_config(get_descriptor_index_impls())
        c['hash_index'] = make_config(get_hash_index_impls())
        c['hash2uuids_kvstore'] = make_config(get_key_value_store_impls())
        c['distance_method'] = 'euclidean'
        c['read_only'] = False
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Resolve components from config
        lsh_functor = None
        if config_dict['lsh_functor'] and config_dict['lsh_functor']['type']:
            lsh_functor = from_plugin_config(config_dict['lsh_functor'],
                                             get_lsh_functor_impls())

        descriptor_index = None
        if config_dict['descriptor_index'] and \
                config_dict['descriptor_index']['type']:
            descriptor_index = from_plugin_config(
                config_dict['descriptor_index'],
                get_descriptor_index_impls()
            )

        hash_index = None
        if config_dict['hash_index'] and config_dict['hash_index']['type']:
            hash_index = from_plugin_config(config_dict['hash_index'],
                                            get_hash_index_impls())

        hash2uuids_kvstore = None
        if config_dict['hash2uuids_kvstore'] and \
                config_dict['hash2uuids_kvstore']['type']:
            hash2uuids_kvstore = from_plugin_config(
                config_dict['hash2uuids_kvstore'],
                get_key_value_store_impls()
            )

        return cls(
            lsh_functor=lsh_functor,
            descriptor_index=descriptor_index,
            hash_index=hash_index,
            hash2uuids_kvstore=hash2uuids_kvstore,
            distance_method=config_dict.get('distance_method', 'euclidean'),
            read_only=config_dict.get('read_only', False),
        )

    def __init__(self, lsh_functor, descriptor_index, hash_index=None,
                 hash2uuids_kvstore=None, distance_method='euclidean',
                 read_only=False):
        """
        Initialize LSH nearest-neighbor index.

        :param lsh_functor: Hash generation functor.
        :param descriptor_index: Index storing descriptors.
        :param hash_index: Optional index for hash codes.
        :param hash2uuids_kvstore: Key-value store mapping hash -> UUIDs.
        :param distance_method: Distance metric ('euclidean', 'cosine', 'hik').
        :param read_only: If True, index is read-only.
        """
        super(LSHNearestNeighborIndex, self).__init__()

        self.lsh_functor = lsh_functor
        self.descriptor_index = descriptor_index
        self.hash_index = hash_index
        self.hash2uuids_kvstore = hash2uuids_kvstore
        self.distance_method = distance_method
        self.read_only = read_only

        self._distance_func = DISTANCE_FUNCTIONS.get(distance_method)
        if self._distance_func is None:
            raise ValueError("Invalid distance method: %s" % distance_method)

        self._model_lock = threading.RLock()

    def get_config(self):
        c = {
            'distance_method': self.distance_method,
            'read_only': self.read_only,
        }

        if self.lsh_functor:
            c['lsh_functor'] = to_plugin_config(self.lsh_functor)
        else:
            c['lsh_functor'] = make_config(get_lsh_functor_impls())

        if self.descriptor_index:
            c['descriptor_index'] = to_plugin_config(self.descriptor_index)
        else:
            c['descriptor_index'] = make_config(get_descriptor_index_impls())

        if self.hash_index:
            c['hash_index'] = to_plugin_config(self.hash_index)
        else:
            c['hash_index'] = make_config(get_hash_index_impls())

        if self.hash2uuids_kvstore:
            c['hash2uuids_kvstore'] = to_plugin_config(self.hash2uuids_kvstore)
        else:
            c['hash2uuids_kvstore'] = make_config(get_key_value_store_impls())

        return c

    def count(self):
        """Return number of descriptors in this index."""
        if self.descriptor_index:
            return self.descriptor_index.count()
        return 0

    def _hash_to_int(self, h):
        """Convert a hash bit-vector to an integer key."""
        from ..utils.bit_utils import bit_vector_to_int_large
        return bit_vector_to_int_large(h)

    def _build_index(self, descriptors):
        """Build the index over the given descriptors."""
        with self._model_lock:
            if self.read_only:
                raise ValueError("Cannot build index in read-only mode.")

            descriptor_list = list(descriptors)
            if not descriptor_list:
                return

            self._log.debug("Building LSH index with %d descriptors",
                            len(descriptor_list))

            # Add to descriptor index
            self.descriptor_index.add_many_descriptors(descriptor_list)

            # Generate hashes and update mappings
            hash_vectors = []
            hash_to_uuids = {}

            for desc in descriptor_list:
                h = self.lsh_functor.get_hash(desc)
                hash_vectors.append(h)

                h_int = self._hash_to_int(h)
                if h_int not in hash_to_uuids:
                    # Check existing mapping
                    existing = self.hash2uuids_kvstore.get(h_int, set())
                    if not isinstance(existing, set):
                        existing = set(existing)
                    hash_to_uuids[h_int] = existing

                hash_to_uuids[h_int].add(desc.uuid())

            # Update KV store
            self.hash2uuids_kvstore.add_many(hash_to_uuids)

            # Update hash index
            if self.hash_index:
                self.hash_index.build_index(hash_vectors)

            self._log.debug("LSH index built successfully")

    def _update_index(self, descriptors):
        """Add descriptors to the index."""
        with self._model_lock:
            if self.read_only:
                raise ValueError("Cannot update index in read-only mode.")

            descriptor_list = list(descriptors)
            if not descriptor_list:
                return

            self._log.debug("Updating LSH index with %d descriptors",
                            len(descriptor_list))

            # Add to descriptor index
            self.descriptor_index.add_many_descriptors(descriptor_list)

            # Generate hashes and update mappings
            hash_vectors = []
            hash_to_uuids = {}

            for desc in descriptor_list:
                h = self.lsh_functor.get_hash(desc)
                hash_vectors.append(h)

                h_int = self._hash_to_int(h)
                if h_int not in hash_to_uuids:
                    existing = self.hash2uuids_kvstore.get(h_int, set())
                    if not isinstance(existing, set):
                        existing = set(existing)
                    hash_to_uuids[h_int] = existing

                hash_to_uuids[h_int].add(desc.uuid())

            # Update KV store
            self.hash2uuids_kvstore.add_many(hash_to_uuids)

            # Update hash index
            if self.hash_index:
                self.hash_index.update_index(hash_vectors)

    def _remove_from_index(self, uids):
        """Remove descriptors from the index by UID."""
        with self._model_lock:
            if self.read_only:
                raise ValueError("Cannot remove from index in read-only mode.")

            uid_list = list(uids)
            if not uid_list:
                return

            # Get descriptors and their hashes before removal
            for uid in uid_list:
                try:
                    desc = self.descriptor_index.get_descriptor(uid)
                    h = self.lsh_functor.get_hash(desc)
                    h_int = self._hash_to_int(h)

                    # Update hash-to-uuids mapping
                    existing = self.hash2uuids_kvstore.get(h_int, set())
                    if isinstance(existing, set) and uid in existing:
                        existing.discard(uid)
                        if existing:
                            self.hash2uuids_kvstore.add(h_int, existing)
                        else:
                            try:
                                self.hash2uuids_kvstore.remove(h_int)
                            except KeyError:
                                pass
                except KeyError:
                    pass

            # Remove from descriptor index
            self.descriptor_index.remove_many_descriptors(uid_list)

    def _nn(self, d, n=1):
        """Find n nearest neighbors to descriptor d."""
        with self._model_lock:
            # Get query vector
            if hasattr(d, 'vector'):
                q_vec = d.vector()
            else:
                q_vec = np.asarray(d)

            # Get query hash
            q_hash = self.lsh_functor.get_hash(q_vec)

            # Find candidate hashes
            if self.hash_index and self.hash_index.count() > 0:
                # Use hash index to find near hashes
                near_hashes, _ = self.hash_index.nn(q_hash, n * 10)
                candidate_uuids = set()
                for h in near_hashes:
                    h_int = self._hash_to_int(h)
                    uuids = self.hash2uuids_kvstore.get(h_int, set())
                    candidate_uuids.update(uuids)
            else:
                # Fall back to all descriptors
                candidate_uuids = set(self.descriptor_index.iterkeys())

            if not candidate_uuids:
                return [], []

            # Compute distances to candidates
            candidates = []
            for uid in candidate_uuids:
                try:
                    desc = self.descriptor_index.get_descriptor(uid)
                    c_vec = desc.vector()
                    dist = self._distance_func(q_vec, c_vec)
                    candidates.append((desc, float(dist)))
                except KeyError:
                    continue

            # Sort by distance and take top n
            candidates.sort(key=lambda x: x[1])
            candidates = candidates[:n]

            if not candidates:
                return [], []

            descriptors = [c[0] for c in candidates]
            distances = [c[1] for c in candidates]

            return descriptors, distances


NN_INDEX_CLASS = LSHNearestNeighborIndex


__all__ = [
    'LSHNearestNeighborIndex',
]
