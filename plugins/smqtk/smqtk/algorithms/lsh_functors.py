"""
SMQTK LSH Functors - Hash generation for locality-sensitive hashing.
"""
import abc
import logging

import numpy as np
import six

from . import SmqtkAlgorithm
from ..representation import get_data_element_impls
from ..utils import merge_dict
from ..utils.plugin import make_config, from_plugin_config, to_plugin_config


@six.add_metaclass(abc.ABCMeta)
class LshFunctor(SmqtkAlgorithm):
    """
    Locality-sensitive hashing functor interface.

    The aim is to generate hash codes (bit-vectors) such that similar items
    map to the same or similar hashes with high probability.
    """

    def __call__(self, descriptor):
        return self.get_hash(descriptor)

    @abc.abstractmethod
    def get_hash(self, descriptor):
        """
        Get the locality-sensitive hash code for the input descriptor.

        :param descriptor: Descriptor vector to hash.
        :type descriptor: numpy.ndarray[float]
        :return: Generated bit-vector as a numpy array of booleans.
        :rtype: numpy.ndarray[bool]
        """


class ItqFunctor(LshFunctor):
    """
    Iterative Quantization (ITQ) hash code generation functor.

    ITQ uses PCA followed by an iterative rotation optimization to produce
    binary hash codes from high-dimensional descriptor vectors.
    """

    @classmethod
    def is_usable(cls):
        return True

    @classmethod
    def get_default_config(cls):
        c = super(ItqFunctor, cls).get_default_config()
        c['mean_vec_cache'] = make_config(get_data_element_impls())
        c['rotation_cache'] = make_config(get_data_element_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(), config_dict)

        # Resolve cache elements from config
        mean_vec_cache = None
        if config_dict['mean_vec_cache'] \
                and config_dict['mean_vec_cache']['type']:
            mean_vec_cache = from_plugin_config(config_dict['mean_vec_cache'],
                                                get_data_element_impls())

        rotation_cache = None
        if config_dict['rotation_cache'] \
                and config_dict['rotation_cache']['type']:
            rotation_cache = from_plugin_config(config_dict['rotation_cache'],
                                                get_data_element_impls())

        return cls(
            mean_vec_cache=mean_vec_cache,
            rotation_cache=rotation_cache,
            bit_length=config_dict.get('bit_length', 8),
            itq_iterations=config_dict.get('itq_iterations', 50),
            normalize=config_dict.get('normalize', None),
            random_seed=config_dict.get('random_seed', None),
        )

    def __init__(self, mean_vec_cache=None, rotation_cache=None,
                 bit_length=8, itq_iterations=50, normalize=None,
                 random_seed=None):
        """
        Initialize ITQ functor.

        :param mean_vec_cache: Data element for mean vector cache.
        :param rotation_cache: Data element for rotation matrix cache.
        :param bit_length: Number of bits in generated hash codes.
        :param itq_iterations: Number of ITQ iterations.
        :param normalize: Normalization to apply ('unit', 'center', or None).
        :param random_seed: Random seed for reproducibility.
        """
        super(ItqFunctor, self).__init__()

        self.mean_vec_cache = mean_vec_cache
        self.rotation_cache = rotation_cache
        self.bit_length = bit_length
        self.itq_iterations = itq_iterations
        self.normalize = normalize
        self.random_seed = random_seed

        # Model state
        self.mean_vec = None
        self.rotation = None

        # Load from cache if available
        self._load_model()

    def _load_model(self):
        """Load model from cache elements if available."""
        if self.mean_vec_cache and not self.mean_vec_cache.is_empty():
            self.mean_vec = np.load(
                self.mean_vec_cache.to_buffered_reader()
            )

        if self.rotation_cache and not self.rotation_cache.is_empty():
            self.rotation = np.load(
                self.rotation_cache.to_buffered_reader()
            )

    def _save_model(self):
        """Save model to cache elements if configured."""
        if self.mean_vec_cache and self.mean_vec is not None:
            from six import BytesIO
            buff = BytesIO()
            np.save(buff, self.mean_vec)
            self.mean_vec_cache.set_bytes(buff.getvalue())

        if self.rotation_cache and self.rotation is not None:
            from six import BytesIO
            buff = BytesIO()
            np.save(buff, self.rotation)
            self.rotation_cache.set_bytes(buff.getvalue())

    def get_config(self):
        c = {
            'bit_length': self.bit_length,
            'itq_iterations': self.itq_iterations,
            'normalize': self.normalize,
            'random_seed': self.random_seed,
        }

        if self.mean_vec_cache:
            c['mean_vec_cache'] = to_plugin_config(self.mean_vec_cache)
        else:
            c['mean_vec_cache'] = make_config(get_data_element_impls())

        if self.rotation_cache:
            c['rotation_cache'] = to_plugin_config(self.rotation_cache)
        else:
            c['rotation_cache'] = make_config(get_data_element_impls())

        return c

    def has_model(self):
        """Check if this functor has a model loaded."""
        return self.mean_vec is not None and self.rotation is not None

    def fit(self, descriptors, use_multiprocessing=True):
        """
        Train the ITQ model on a set of descriptors.

        :param descriptors: Iterable of DescriptorElement instances.
        :param use_multiprocessing: Use multiprocessing for matrix creation.
        """
        from ..representation import elements_to_matrix

        self._log.info("Fitting ITQ model with bit_length=%d, iterations=%d",
                       self.bit_length, self.itq_iterations)

        # Convert descriptors to matrix
        descriptor_list = list(descriptors)
        if not descriptor_list:
            raise ValueError("No descriptors provided for training.")

        self._log.debug("Creating descriptor matrix from %d elements",
                        len(descriptor_list))
        d_matrix = elements_to_matrix(descriptor_list,
                                       use_multiprocessing=use_multiprocessing)

        # Normalize if configured
        if self.normalize == 'unit':
            norms = np.linalg.norm(d_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            d_matrix = d_matrix / norms

        # Compute and subtract mean
        self._log.debug("Computing mean vector")
        self.mean_vec = d_matrix.mean(axis=0)
        d_matrix = d_matrix - self.mean_vec

        # PCA to reduce to bit_length dimensions
        self._log.debug("Performing PCA")
        n_samples, n_features = d_matrix.shape
        n_components = min(self.bit_length, n_features, n_samples)

        # Use SVD for PCA
        _, s, Vt = np.linalg.svd(d_matrix, full_matrices=False)
        pca_components = Vt[:n_components]

        # Project data
        projected = np.dot(d_matrix, pca_components.T)

        # Pad if needed
        if projected.shape[1] < self.bit_length:
            padding = np.zeros((projected.shape[0],
                                self.bit_length - projected.shape[1]))
            projected = np.hstack([projected, padding])
            pca_components = np.vstack([
                pca_components,
                np.zeros((self.bit_length - pca_components.shape[0], n_features))
            ])

        # ITQ optimization
        self._log.debug("Running ITQ optimization")
        rotation = self._find_itq_rotation(projected, self.itq_iterations)

        # Combine PCA and ITQ rotation
        self.rotation = np.dot(rotation.T, pca_components)

        # Save model
        self._save_model()

        self._log.info("ITQ model fitting complete")

    def _find_itq_rotation(self, v, n_iter):
        """
        Find the optimal rotation matrix using ITQ.

        :param v: Projected data matrix (n_samples x n_bits).
        :param n_iter: Number of iterations.
        :return: Rotation matrix (n_bits x n_bits).
        """
        rng = np.random.RandomState(self.random_seed)

        n_bits = v.shape[1]

        # Initialize with random rotation
        r = np.linalg.qr(rng.randn(n_bits, n_bits))[0]

        for i in range(n_iter):
            # Project and binarize
            z = np.dot(v, r)
            b = np.sign(z)
            b[b == 0] = 1

            # Update rotation via SVD
            u, _, vt = np.linalg.svd(np.dot(b.T, v))
            r = np.dot(vt.T, u.T)

        return r

    def get_hash(self, descriptor):
        """
        Get the locality-sensitive hash code for the input descriptor.

        :param descriptor: Descriptor vector to hash.
        :type descriptor: numpy.ndarray[float]
        :return: Generated bit-vector as a numpy array of booleans.
        :rtype: numpy.ndarray[bool]
        """
        if not self.has_model():
            raise RuntimeError("ITQ model not trained. Call fit() first.")

        # Get vector from descriptor if it's a DescriptorElement
        if hasattr(descriptor, 'vector'):
            v = descriptor.vector()
        else:
            v = np.asarray(descriptor)

        # Ensure float type
        v = v.astype(np.float64)

        # Normalize if configured
        if self.normalize == 'unit':
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm

        # Subtract mean
        v = v - self.mean_vec

        # Project using combined rotation matrix
        projected = np.dot(self.rotation, v)

        # Binarize
        return projected >= 0


LSH_FUNCTOR_CLASS = ItqFunctor


def get_lsh_functor_impls(reload_modules=False):
    """Return available LshFunctor implementations."""
    return {
        'ItqFunctor': ItqFunctor,
    }


__all__ = [
    'LshFunctor',
    'ItqFunctor',
    'get_lsh_functor_impls',
]
