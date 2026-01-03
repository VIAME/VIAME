"""
SMQTK IQR Session - Interactive Query Refinement session management.
"""
import logging
import threading
import uuid

from ..representation import MemoryDescriptorIndex
from ..algorithms.relevancy_index import LibSvmHikRelevancyIndex


class IqrSession(object):
    """
    Encapsulates IQR session state.

    Manages positive and negative adjudications, working index, and
    relevancy ranking for interactive query refinement.
    """

    def __init__(self, pos_seed_neighbors=500, rel_index_config=None,
                 session_uid=None):
        """
        Initialize an IQR session.

        :param pos_seed_neighbors: Number of neighbors to seed from positives.
        :param rel_index_config: Configuration for relevancy index.
        :param session_uid: Optional unique identifier for this session.
        """
        self._log = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__])
        )

        self.pos_seed_neighbors = pos_seed_neighbors
        self.rel_index_config = rel_index_config or {}
        self.session_uid = session_uid or str(uuid.uuid4())

        # Adjudicated descriptors
        self.positive_descriptors = set()
        self.negative_descriptors = set()

        # External positive/negative examples (not from main index)
        self.external_positive_descriptors = set()
        self.external_negative_descriptors = set()

        # Working index - subset of descriptors to search
        self.working_index = MemoryDescriptorIndex()

        # Relevancy index for ranking
        self._rel_index = None

        # Results from last refinement
        self.results = {}

        # Feedback descriptors - samples for user to adjudicate
        self.feedback_map = {}

        # Thread lock for session state
        self._lock = threading.RLock()

    def reset(self):
        """Reset session state."""
        with self._lock:
            self.positive_descriptors.clear()
            self.negative_descriptors.clear()
            self.external_positive_descriptors.clear()
            self.external_negative_descriptors.clear()
            self.working_index = MemoryDescriptorIndex()
            self._rel_index = None
            self.results = {}
            self.feedback_map = {}

    @property
    def lock(self):
        """Return the session lock for external synchronization."""
        return self._lock

    def adjudicate(self, new_positives=None, new_negatives=None,
                   un_positives=None, un_negatives=None):
        """
        Update adjudications with new positive/negative descriptors.

        :param new_positives: New descriptors to mark as positive.
        :param new_negatives: New descriptors to mark as negative.
        :param un_positives: Descriptors to un-mark as positive.
        :param un_negatives: Descriptors to un-mark as negative.
        """
        with self._lock:
            new_positives = set(new_positives or [])
            new_negatives = set(new_negatives or [])
            un_positives = set(un_positives or [])
            un_negatives = set(un_negatives or [])

            # Remove from current sets
            self.positive_descriptors -= un_positives
            self.negative_descriptors -= un_negatives

            # Add new adjudications (ensure no overlap)
            self.positive_descriptors |= new_positives
            self.positive_descriptors -= new_negatives

            self.negative_descriptors |= new_negatives
            self.negative_descriptors -= new_positives

    def add_external_pos(self, descriptors):
        """Add external positive example descriptors."""
        with self._lock:
            self.external_positive_descriptors.update(descriptors)

    def add_external_neg(self, descriptors):
        """Add external negative example descriptors."""
        with self._lock:
            self.external_negative_descriptors.update(descriptors)

    def update_working_index(self, nn_index):
        """
        Update the working index using neighbors from positive exemplars.

        :param nn_index: Nearest neighbors index to query.
        """
        with self._lock:
            all_positives = (self.positive_descriptors |
                             self.external_positive_descriptors)

            if not all_positives:
                self._log.debug("No positive exemplars, clearing working index")
                self.working_index = MemoryDescriptorIndex()
                return

            # Collect candidate descriptors from NN queries
            candidates = set()
            for pos_d in all_positives:
                try:
                    neighbors, _ = nn_index.nn(pos_d, self.pos_seed_neighbors)
                    candidates.update(neighbors)
                except ValueError:
                    # Empty index
                    pass

            # Add positive and negative exemplars to candidates
            candidates.update(all_positives)
            candidates.update(self.negative_descriptors)
            candidates.update(self.external_negative_descriptors)

            # Build working index
            self.working_index = MemoryDescriptorIndex()
            self.working_index.add_many_descriptors(candidates)

            self._log.debug("Working index updated with %d descriptors",
                            self.working_index.count())

    def refine(self):
        """
        Refine the current working index based on adjudications.

        Uses SVM-based relevancy ranking to score descriptors.
        """
        with self._lock:
            if not self.working_index.count():
                self._log.warning("No descriptors in working index")
                self.results = {}
                return

            all_positives = (self.positive_descriptors |
                             self.external_positive_descriptors)
            all_negatives = (self.negative_descriptors |
                             self.external_negative_descriptors)

            if not all_positives:
                self._log.warning("No positive exemplars for refinement")
                self.results = {}
                return

            # Build relevancy index
            self._rel_index = LibSvmHikRelevancyIndex(
                **self.rel_index_config
            )
            self._rel_index.build_index(self.working_index.iterdescriptors())

            # Rank descriptors
            self.results = self._rel_index.rank(all_positives, all_negatives)

            self._log.debug("Refinement complete, %d results ranked",
                            len(self.results))

    def ordered_results(self):
        """
        Return results ordered by relevancy score (highest first).

        :return: List of (descriptor, score) tuples.
        """
        with self._lock:
            if not self.results:
                return []
            return sorted(self.results.items(), key=lambda x: x[1], reverse=True)

    def ordered_feedback(self, n=10):
        """
        Return descriptors for user feedback (most uncertain samples).

        :param n: Number of feedback samples to return.
        :return: List of (descriptor, score) tuples near decision boundary.
        """
        with self._lock:
            if not self.results:
                return []

            # Sort by distance from 0.5 (decision boundary)
            sorted_by_uncertainty = sorted(
                self.results.items(),
                key=lambda x: abs(x[1] - 0.5)
            )

            # Return most uncertain samples (excluding already adjudicated)
            adjudicated = self.positive_descriptors | self.negative_descriptors
            feedback = []
            for desc, score in sorted_by_uncertainty:
                if desc not in adjudicated:
                    feedback.append((desc, score))
                    if len(feedback) >= n:
                        break

            self.feedback_map = dict(feedback)
            return feedback

    def get_unadjudicated_relevancy(self):
        """
        Get relevancy scores for unadjudicated descriptors only.

        :return: Dictionary mapping unadjudicated descriptors to scores.
        """
        with self._lock:
            adjudicated = self.positive_descriptors | self.negative_descriptors
            return {d: s for d, s in self.results.items()
                    if d not in adjudicated}

    def get_relevancy_index(self):
        """Get the current relevancy index."""
        return self._rel_index

    def get_model(self):
        """Get the trained model from the relevancy index."""
        if self._rel_index:
            return self._rel_index.get_model()
        return None


__all__ = ['IqrSession']
