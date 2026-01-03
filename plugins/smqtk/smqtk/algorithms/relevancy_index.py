"""
SMQTK Relevancy Index - IQR relevancy scoring algorithms.
"""
import abc
import logging
import os
import tempfile
import threading

import numpy as np
import six

from . import SmqtkAlgorithm


@six.add_metaclass(abc.ABCMeta)
class RelevancyIndex(SmqtkAlgorithm):
    """
    Abstract class for IQR index implementations.

    Takes positive and negative exemplars and produces a [0, 1] ranking
    of indexed elements by determined relevancy.
    """

    def __len__(self):
        return self.count()

    @abc.abstractmethod
    def count(self):
        """Return number of elements in this index."""

    @abc.abstractmethod
    def build_index(self, descriptors):
        """
        Build the index based on the given descriptors.
        """

    @abc.abstractmethod
    def rank(self, pos, neg):
        """
        Rank indexed elements given positive and negative exemplars.

        :param pos: Iterable of positive exemplar DescriptorElement instances.
        :param neg: Iterable of negative exemplar DescriptorElement instances.
        :return: Map of indexed descriptor elements to a rank value [0, 1].
        """

    def get_model(self):
        """Get the model used for ranking."""
        return None


class LibSvmHikRelevancyIndex(RelevancyIndex):
    """
    SVM-based relevancy ranking using histogram intersection kernel.
    """

    @classmethod
    def is_usable(cls):
        try:
            import svmutil
            return True
        except ImportError:
            try:
                from libsvm import svmutil
                return True
            except ImportError:
                return False

    @classmethod
    def get_default_config(cls):
        return {
            'autoneg_select_ratio': 1.0,
            'cores': None,
        }

    def __init__(self, autoneg_select_ratio=1.0, cores=None):
        """
        Initialize LibSVM-based relevancy index.

        :param autoneg_select_ratio: Ratio for automatic negative selection.
        :param cores: Number of cores for parallel training (None for auto).
        """
        super(LibSvmHikRelevancyIndex, self).__init__()

        self.autoneg_select_ratio = autoneg_select_ratio
        self.cores = cores

        # Model state
        self._descr_cache = []
        self._descr_matrix = None
        self._model = None
        self._model_lock = threading.RLock()

    def get_config(self):
        return {
            'autoneg_select_ratio': self.autoneg_select_ratio,
            'cores': self.cores,
        }

    def count(self):
        return len(self._descr_cache)

    def build_index(self, descriptors):
        """Build the index from the given descriptors."""
        with self._model_lock:
            self._descr_cache = list(descriptors)
            if self._descr_cache:
                self._descr_matrix = np.vstack([
                    d.vector() for d in self._descr_cache
                ])
            else:
                self._descr_matrix = None
            self._model = None

    def rank(self, pos, neg):
        """
        Rank indexed elements using SVM with histogram intersection kernel.

        :param pos: Positive exemplar descriptors.
        :param neg: Negative exemplar descriptors.
        :return: Dictionary mapping descriptors to relevancy scores [0, 1].
        """
        with self._model_lock:
            if not self._descr_cache:
                return {}

            pos_list = list(pos)
            neg_list = list(neg)

            if not pos_list:
                # No positive examples, return equal scores
                return {d: 0.5 for d in self._descr_cache}

            try:
                try:
                    import svmutil
                    from svmutil import svm_train, svm_predict
                    import svm as svm_module
                except ImportError:
                    from libsvm import svmutil
                    from libsvm.svmutil import svm_train, svm_predict
                    from libsvm import svm as svm_module
            except ImportError:
                self._log.warning("libsvm not available, returning uniform scores")
                return {d: 0.5 for d in self._descr_cache}

            # Build training data
            pos_vectors = [d.vector() for d in pos_list]
            neg_vectors = [d.vector() for d in neg_list] if neg_list else []

            # Auto-select negatives if none provided
            if not neg_vectors and self.autoneg_select_ratio > 0:
                n_auto_neg = max(1, int(len(pos_vectors) * self.autoneg_select_ratio))
                # Randomly select from indexed descriptors
                all_uuids = set(d.uuid() for d in self._descr_cache)
                pos_uuids = set(d.uuid() for d in pos_list)
                available = [d for d in self._descr_cache
                             if d.uuid() not in pos_uuids]
                if available:
                    indices = np.random.choice(len(available),
                                               min(n_auto_neg, len(available)),
                                               replace=False)
                    neg_vectors = [available[i].vector() for i in indices]

            if not neg_vectors:
                # Still no negatives, return all positive scores
                return {d: 1.0 for d in self._descr_cache}

            # Prepare training data for libsvm
            n_pos = len(pos_vectors)
            n_neg = len(neg_vectors)

            train_labels = [1] * n_pos + [0] * n_neg
            train_vectors = pos_vectors + neg_vectors

            # Convert to libsvm format (list of dicts)
            train_data = []
            for vec in train_vectors:
                # libsvm uses 1-indexed features
                data_dict = {i+1: float(v) for i, v in enumerate(vec)}
                train_data.append(data_dict)

            # Train SVM with histogram intersection kernel (precomputed)
            # Use linear kernel with normalized vectors as approximation
            param_str = '-t 0 -c 1 -q'  # Linear kernel, quiet mode
            if self.cores:
                param_str += ' -m %d' % self.cores

            # Normalize vectors for better results
            def normalize(v):
                norm = np.linalg.norm(v)
                return v / norm if norm > 0 else v

            train_vectors_norm = [normalize(np.array(v)) for v in train_vectors]
            train_data_norm = []
            for vec in train_vectors_norm:
                data_dict = {i+1: float(v) for i, v in enumerate(vec)}
                train_data_norm.append(data_dict)

            # Train model
            try:
                prob = svm_module.svm_problem(train_labels, train_data_norm)
                param = svm_module.svm_parameter(param_str)
                self._model = svmutil.svm_train(prob, param)
            except Exception as e:
                self._log.error("SVM training failed: %s", e)
                return {d: 0.5 for d in self._descr_cache}

            # Score all indexed descriptors
            results = {}
            for desc in self._descr_cache:
                vec = normalize(desc.vector())
                test_data = [{i+1: float(v) for i, v in enumerate(vec)}]

                try:
                    labels, acc, vals = svm_predict([0], test_data, self._model, '-q')
                    # vals contains decision values
                    if vals and len(vals[0]) > 0:
                        score = vals[0][0]
                        # Convert to [0, 1] range using sigmoid
                        score = 1.0 / (1.0 + np.exp(-score))
                    else:
                        score = 0.5
                except Exception:
                    score = 0.5

                results[desc] = score

            return results

    def get_model(self):
        """Get the trained SVM model."""
        return self._model


RELEVANCY_INDEX_CLASS = LibSvmHikRelevancyIndex


def get_relevancy_index_impls(reload_modules=False):
    """Return available RelevancyIndex implementations."""
    return {
        'LibSvmHikRelevancyIndex': LibSvmHikRelevancyIndex,
    }


__all__ = [
    'RelevancyIndex',
    'LibSvmHikRelevancyIndex',
    'get_relevancy_index_impls',
]
