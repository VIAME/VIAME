import ubelt as ub
import torch.utils
import torch
import numpy as np


class MatchingSamplerPK(ub.NiceRepr, torch.utils.data.sampler.BatchSampler):
    """
    Samples random triples from a PCC-complient dataset

    Args:
        pccs (List[FrozenSet]):
            Groups of annotation-indices, where each group contains all annots
            with the same name (individual identity).
        p (int): number of individuals sampled per batch
        k (int): number of annots sampled per individual within a batch
        batch_size (int): if specified, k is adjusted to an appropriate length
        drop_last (bool): ignored
        num_batches (int): length of the loader
        rng (int | Random, default=None): random seed
        shuffle (bool): if False rng is ignored and getitem is deterministic

    TODO:
        Look at
        https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
        to see if we can try using all examples of a class before repeating
        them

    Example:
        >>> pccs = [(0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11), (12,)]
        >>> batch_sampler = self = MatchingSamplerPK(pccs, p=2, k=2, shuffle=True)
        >>> print('batch_sampler = {!r}'.format(batch_sampler))
        >>> for indices in batch_sampler:
        >>>     print('indices = {!r}'.format(indices))
    """
    def __init__(self, pccs, p=21, k=4, batch_size=None, drop_last=False,
                 rng=None, shuffle=False, num_batches=None, replace=True):
        import kwarray
        self.drop_last = drop_last
        self.replace = replace
        self.shuffle = shuffle
        self.pccs = pccs

        assert k > 1

        if replace is False:
            raise NotImplementedError(
                'We currently cant sample without replacement')

        if getattr(pccs, '__hasgraphid__', False):
            raise NotImplementedError('TODO: graphid API sampling')
        else:
            # Compute the total possible number of triplets
            # Its probably a huge number, but lets do it anyway.
            # --------
            # In the language of graphid (See Jon Crall's 2017 thesis)
            # The matching graph for MNIST is fully connected graph.  All
            # pairs of annotations with the same label have a positive edge
            # between them. All pairs of annotations with a different label
            # have a negative edge between them. There are no incomparable
            # edges. Therefore for each PCC, A the number of triples it can
            # contribute is the number of internal positive edges
            # ({len(A) choose 2}) times the number of outgoing negative edges.
            # ----
            # Each pair of positive examples could be a distinct triplet
            # For each of these any negative could be chosen
            # The number of distinct triples contributed by this PCC is the
            # product of num_pos_edges and num_neg_edges.
            import scipy  # NOQA
            try:
                from scipy.special import comb
            except ImportError:
                from scipy.misc import comb
            self.num_triples = 0
            self.num_pos_edges = 0
            default_num_batches = 0
            for pcc in ub.ProgIter(self.pccs, 'pccs',  enabled=0):
                num_pos_edges = comb(len(pcc), 2)
                if num_pos_edges > 0:
                    default_num_batches += len(pcc)
                other_pccs = [c for c in self.pccs if c is not pcc]
                num_neg_edges = sum(len(c) for c in other_pccs)
                self.num_triples += num_pos_edges * num_neg_edges
                self.num_pos_edges += num_pos_edges

        self.multitons = [pcc for pcc in self.pccs if len(pcc) > 1]

        p = min(len(self.multitons), p)
        k = min(max(len(p) for p in self.pccs), k)
        assert k > 1

        if batch_size is not None:
            p = batch_size // k

        batch_size = p * k

        if not num_batches:
            num_batches = default_num_batches

        self.batch_size = batch_size
        self.num_batches = num_batches
        self.p = p  # PCCs per batch
        self.k = k  # Items per PCC per batch
        self.rng = kwarray.ensure_rng(rng, api='python')

    def __nice__(self):
        return ('p={p}, k={k}, batch_size={batch_size}, '
                'len={num_batches}').format(**self.__dict__)

    def __iter__(self):
        for index in range(len(self)):
            indices = self[index]
            yield indices

    def __getitem__(self, index):
        if not self.shuffle:
            import kwarray
            self.rng = kwarray.ensure_rng(index, api='python')

        sub_pccs = self.rng.sample(self.multitons, self.p)

        groups = []
        for sub_pcc in sub_pccs:
            aids = self.rng.sample(sub_pcc, min(self.k, len(sub_pcc)))
            groups.append(aids)

        nhave = sum(map(len, groups))
        while nhave < self.batch_size:
            sub_pcc = self.rng.choice(self.pccs)
            aids = self.rng.sample(sub_pcc, min(self.k, len(sub_pcc)))
            groups.append(aids)
            nhave = sum(map(len, groups))
            overshoot = nhave - self.batch_size
            if overshoot:
                groups[-1] = groups[-1][:-overshoot]

        indices = sorted(ub.flatten(groups))
        return indices

    def __len__(self):
        return self.num_batches


class BalancedBatchSampler(
        ub.NiceRepr, torch.utils.data.sampler.BatchSampler):
    """
    A sampler for balancing classes amongst batches

    Args:
        index_to_label (List[int]): the label for each index in a dataset
        batch_size (int): number of dataset indexes for each batch
        num_batches (int | str, default='auto'): number of batches to generate
        quantile (float): interpolates between under and oversamling when
            num_batches='auto'. A value of 0 is pure undersampling, and a value
            of 1 is pure oversampling.
        shuffle (bool, default=False): if True randomize batch ordering
        drop_last (bool): unused, exists for compatibility
        rng (RandomState, default=None): random seed

    Example:
        >>> from .data.batch_samplers import *  # NOQA
        >>> from .data.batch_samplers import RingSampler  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> classes = ['class_{}'.format(i) for i in range(5)]
        >>> # Create a random class label for each item
        >>> index_to_label = rng.randint(0, len(classes), 100)
        >>> if 1:
        >>>     # Create a rare class
        >>>     index_to_label[0:3] = 42
        >>> quantile = 0.0
        >>> self = BalancedBatchSampler(index_to_label, batch_size=4, quantile=quantile, rng=0)
        >>> print('self.label_to_freq = {!r}'.format(self.label_to_freq))
        >>> indices = list(self)
        >>> print('indices = {!r}'.format(indices))
        >>> # Print the epoch / item label frequency per epoch
        >>> label_sequence = []
        >>> index_sequence = []
        >>> for item_indices in self:
        >>>     item_indices = np.array(item_indices)
        >>>     item_labels = index_to_label[item_indices]
        >>>     index_sequence.extend(item_indices)
        >>>     label_sequence.extend(item_labels)
        >>> label_hist = ub.dict_hist(label_sequence)
        >>> index_hist = ub.dict_hist(index_sequence)
        >>> label_hist = ub.sorted_vals(label_hist, reverse=True)
        >>> index_hist = ub.sorted_vals(index_hist, reverse=True)
        >>> index_hist = ub.dict_subset(index_hist, list(index_hist.keys())[0:5])
        >>> print('label_hist = {}'.format(ub.urepr(label_hist, nl=1)))
        >>> print('index_hist = {}'.format(ub.urepr(index_hist, nl=1)))
    """

    def __init__(self, index_to_label, batch_size=1, num_batches='auto',
                 quantile=0.5, shuffle=False, rng=None):
        import kwarray

        rng = kwarray.ensure_rng(rng, api='python')
        label_to_indices = kwarray.group_items(
            np.arange(len(index_to_label)), index_to_label)

        label_to_freq = ub.map_vals(len, label_to_indices)

        label_to_subsampler = {
            label: RingSampler(indices, shuffle=shuffle, rng=rng)
            for label, indices in label_to_indices.items()
        }

        self.label_to_freq = label_to_freq
        self.index_to_label = index_to_label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = rng
        self.label_to_indices = label_to_indices
        self.label_to_subsampler = label_to_subsampler

        if num_batches == 'auto':
            self.num_batches = self._auto_num_batches(quantile)
        else:
            self.num_batches = num_batches

        self.labels = list(self.label_to_indices.keys())

    def __nice__(self):
        return ub.urepr({
            'num_batches': self.num_batches,
            'batch_size': self.batch_size,
        }, nl=0)

    def _auto_num_batches(self, quantile):
        # Over / under sample each class depending on the balance factor
        label_freq = sorted(self.label_to_freq.values())
        # if 'idf':
        #     TODO: idf balancing
        #     N = len(self.index_to_label)
        #     label_to_idf = ub.map_vals(lambda x: N / x, self.label_to_freq)
        #     denom = sum(label_to_idf.values())
        #     label_to_prob = ub.map_vals(lambda x: x / denom, label_to_idf)
        # How many times will we sample each category?
        samples_per_label = np.quantile(label_freq, quantile)
        # Compute #items as seen per epoch, and #batches from that
        epoch_items = samples_per_label * len(label_freq)
        num_batches = max(1, int(round(epoch_items / self.batch_size)))
        return num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for index in range(self.num_batches):
            yield self[index]

    def __getitem__(self, index):
        # Choose a label for each item in the batch
        if not hasattr(self.rng, 'choices'):
            # python 3.5 support
            chosen_labels = [self.rng.choice(self.labels)
                             for _ in range(self.batch_size)]
        else:
            chosen_labels = self.rng.choices(self.labels, k=self.batch_size)
        # Count the number of items we need for each label
        label_freq = ub.dict_hist(chosen_labels)

        # Sample those indices
        batch_idxs = list(ub.flatten([
            self.label_to_subsampler[label].sample(num)
            for label, num in label_freq.items()
        ]))
        return batch_idxs

    def _balance_report(self, limit=None):
        # Print the epoch / item label frequency per epoch
        label_sequence = []
        index_sequence = []
        if limit is None:
            limit = self.num_batches
        for item_indices, _ in zip(self, range(limit)):
            item_indices = np.array(item_indices)
            item_labels = list(ub.take(self.index_to_label, item_indices))
            index_sequence.extend(item_indices)
            label_sequence.extend(ub.unique(item_labels))
        label_hist = ub.dict_hist(label_sequence)
        index_hist = ub.dict_hist(index_sequence)
        label_hist = ub.sorted_vals(label_hist, reverse=True)
        index_hist = ub.sorted_vals(index_hist, reverse=True)
        index_hist = ub.dict_subset(index_hist, list(index_hist.keys())[0:5])
        print('label_hist = {}'.format(ub.urepr(label_hist, nl=1)))
        print('index_hist = {}'.format(ub.urepr(index_hist, nl=1)))


class GroupedBalancedBatchSampler(ub.NiceRepr, torch.utils.data.sampler.BatchSampler):
    """
    Show items containing less frequent categories more often

    Args:
        index_to_labels (List[Listint]]): the labels for each index in a dataset
        batch_size (int): number of dataset indexes for each batch
        num_batches (int | str, default='auto'): number of batches to generate
        shuffle (bool, default=False): if True randomize batch ordering
        drop_last (bool): unused, exists for compatibility
        label_to_weight (dict, default=None):
            mapping from labels to user-specified weights
        rng (RandomState, default=None): random seed

    References:
        https://arxiv.org/pdf/1908.09492.pdf

    Example:
        >>> from .data.batch_samplers import *  # NOQA
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> classes = ['class_{}'.format(i) for i in range(10)]
        >>> # Create a set of random classes for each item
        >>> index_to_labels = [rng.randint(0, len(classes), rng.randint(10))
        >>>                   for _ in range(1000)]
        >>> # Create a rare class
        >>> index_to_labels[0][0] = 42
        >>> self = GroupedBalancedBatchSampler(index_to_labels, batch_size=4)
        >>> print('self.label_to_freq = {}'.format(ub.urepr(self.label_to_freq, nl=1)))
        >>> indices = list(self)
        >>> print('indices = {!r}'.format(indices))
        >>> # Print the epoch / item label frequency per epoch
        >>> label_sequence = []
        >>> index_sequence = []
        >>> for item_indices, _ in zip(self, range(1000)):
        >>>     item_indices = np.array(item_indices)
        >>>     item_labels = list(ub.flatten(ub.take(index_to_labels, item_indices)))
        >>>     index_sequence.extend(item_indices)
        >>>     label_sequence.extend(ub.unique(item_labels))
        >>> label_hist = ub.dict_hist(label_sequence)
        >>> index_hist = ub.dict_hist(index_sequence)
        >>> label_hist = ub.sorted_vals(label_hist, reverse=True)
        >>> index_hist = ub.sorted_vals(index_hist, reverse=True)
        >>> index_hist = ub.dict_subset(index_hist, list(index_hist.keys())[0:5])
        >>> print('label_hist = {}'.format(ub.urepr(label_hist, nl=1)))
        >>> print('index_hist = {}'.format(ub.urepr(index_hist, nl=1)))
    """

    def __init__(self, index_to_labels, batch_size=1, num_batches='auto',
                 label_to_weight=None, shuffle=False, rng=None):
        import kwarray

        rng = kwarray.ensure_rng(rng, api='python')
        label_to_indices = ub.ddict(set)

        flat_groups = []
        for index, item_labels in enumerate(index_to_labels):
            flat_groups.extend([index] * len(item_labels))
            for label in item_labels:
                label_to_indices[label].add(index)
        flat_labels = np.hstack(index_to_labels)
        label_to_freq = ub.dict_hist(flat_labels)

        # Use tf-idf based scheme to compute sample probabilities
        label_to_idf = {}
        label_to_tfidf = {}
        labels = sorted(set(flat_labels))
        for label in labels:
            # tf for each img, is the number of times the label appears
            index_to_tf = np.zeros(len(index_to_labels))
            for index, item_labels in enumerate(index_to_labels):
                index_to_tf[index] = (label == item_labels).sum()
            # idf is the #imgs / #imgs-with-label
            idf = len(index_to_tf) / (index_to_tf > 0).sum()
            if label_to_weight:
                idf = idf * label_to_weight[label]
            label_to_idf[label] = idf
            label_to_tfidf[label] = np.maximum(index_to_tf * idf, 1)
        index_to_weight = sum(label_to_tfidf.values())
        index_to_prob = index_to_weight / index_to_weight.sum()

        if 0:
            index_to_unique_labels = list(map(set, index_to_labels))
            unique_freq = ub.dict_hist(ub.flatten(index_to_unique_labels))
            tot = sum(unique_freq.values())
            unweighted_odds = ub.map_vals(lambda x: x / tot, unique_freq)

            label_to_indices = ub.ddict(set)
            for index, item_labels in enumerate(index_to_labels):
                for label in item_labels:
                    label_to_indices[label].add(index)
            ub.map_vals(len, label_to_indices)

            label_to_odds = ub.ddict(lambda: 0)
            for label, indices in label_to_indices.items():
                for idx in indices:
                    label_to_odds[label] += index_to_prob[idx]

            coi = {x for x, w in label_to_weight.items() if w > 0}
            coi_weighted = ub.dict_subset(label_to_odds, coi)
            coi_unweighted = ub.dict_subset(unweighted_odds, coi)
            print('coi_weighted = {}'.format(ub.urepr(coi_weighted, nl=1)))
            print('coi_unweighted = {}'.format(ub.urepr(coi_unweighted, nl=1)))

        self.index_to_prob = index_to_prob
        self.indices = np.arange(len(index_to_prob))

        if num_batches == 'auto':
            self.num_batches = self._auto_num_batches()
        else:
            self.num_batches = num_batches

        self.label_to_freq = label_to_freq
        self.index_to_labels = index_to_labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = kwarray.ensure_rng(rng, api='numpy')

    def __nice__(self):
        return ub.urepr({
            'num_batches': self.num_batches,
            'batch_size': self.batch_size,
            'label_to_freq': self.label_to_freq,
        }, nl=0)

    def _balance_report(self, limit=None):
        # Print the epoch / item label frequency per epoch
        label_sequence = []
        index_sequence = []
        if limit is None:
            limit = self.num_batches
        for item_indices, _ in zip(self, range(limit)):
            item_indices = np.array(item_indices)
            item_labels = list(ub.flatten(ub.take(self.index_to_labels, item_indices)))
            index_sequence.extend(item_indices)
            label_sequence.extend(ub.unique(item_labels))
        label_hist = ub.dict_hist(label_sequence)
        index_hist = ub.dict_hist(index_sequence)
        label_hist = ub.sorted_vals(label_hist, reverse=True)
        index_hist = ub.sorted_vals(index_hist, reverse=True)
        index_hist = ub.dict_subset(index_hist, list(index_hist.keys())[0:5])
        print('label_hist = {}'.format(ub.urepr(label_hist, nl=1)))
        print('index_hist = {}'.format(ub.urepr(index_hist, nl=1)))

    def _auto_num_batches(self):
        # The right way to calculate num samples would be using a generalized
        # solutions to the coupon collector problem, but in practice that
        # expected number of samples will be too large for imbalanced datasets.
        # Therefore we punt and simply use heuristics.
        num_batches = len(self.index_to_prob)
        # else:
        #     raise NotImplementedError(balance)
        # def nth_harmonic(n):
        #     """
        #     Example:
        #         >>> n = 10
        #         >>> want = float(sympy.harmonic(n))
        #         >>> got = nth_harmonic(n)
        #         >>> np.isclose(want, got)
        #     """
        #     return np.sum(1 / np.arange(1, n + 1))

        # def uniform_coupon_ev(n):
        #     ev = n * nth_harmonic(n)
        #     return ev

        # def uniform_coupon_ev_to_collect_k(n, k):
        #     i = np.arange(n)
        #     prob_new = (n - i + 1) / n
        #     ev_new = 1 / prob_new
        #     ev = np.sum(ev_new[0:k])
        #     return ev

        # n = 100
        # uniform_coupon_ev_to_collect_k(n, int(0.6 * n))
        # n / np.arange(1, n + 1)[::-1]
        # ev_uniform = uniform_coupon_ev(len(self.index_to_prob))
        return num_batches

    def __getitem__(self, index):
        # Hack, within each batch we are going to prevent replacement
        batch_idxs = self.rng.choice(
            self.indices, p=self.index_to_prob, replace=False,
            size=self.batch_size)
        return batch_idxs

    def __iter__(self):
        for index in range(self.num_batches):
            yield self[index]

    def __len__(self):
        return self.num_batches


class RingSampler(object):
    """
    Stateful sampling without replacement until all item are exhausted

    Example:
        >>> from .data.batch_samplers import RingSampler  # NOQA
        >>> self = RingSampler(list(range(1, 4)))
        >>> sampled_items = self.sample(7)
        >>> print('sampled_items = {!r}'.format(sampled_items))
        sampled_items = array([1, 2, 3, 1, 2, 3, 1])

        >>> self = RingSampler(list(range(1, 4)), rng=0, shuffle=True)
        >>> sampled_items = self.sample(7)
        >>> print('sampled_items = {!r}'.format(sampled_items))
        sampled_items = array([3, 2, 1, 1, 3, 2, 1])
    """
    def __init__(self, items, shuffle=False, rng=None):
        import kwarray
        if len(items) == 0:
            raise Exception('no items to sample')
        self.rng = kwarray.ensure_rng(rng)
        self.items = np.array(items)
        self.shuffle = shuffle
        self.indices = np.arange(len(items))
        self._pos = None
        self.refresh()

    def refresh(self):
        import kwarray
        self._pos = 0
        if self.shuffle:
            self.indices = kwarray.shuffle(self.indices, rng=self.rng)

    def sample_indices(self, size=None):
        """
        Sample indexes into the items array
        """
        n_need = size
        if size is None:
            n_need = 1
        n_total = len(self.indices)
        idx_accum = []
        while n_need > 0:
            # Take as many as we need or as many as we have
            n_avail = (n_total - self._pos)
            n_got = min(n_need, n_avail)
            n_need -= n_got

            idxs = self.indices[self._pos:self._pos + n_got]
            idx_accum.append(idxs.copy())

            # Update state, if we have exhausted all items, then refresh
            self._pos += n_got
            if self._pos == n_total:
                self.refresh()

        sampled_idxs = np.hstack(idx_accum)
        if size is None:
            sampled_idxs = sampled_idxs[0]
        return sampled_idxs

    def sample(self, size=None):
        """
        Sample items from the items array
        """
        sampled_idxs = self.sample_indices(size)
        sampled_items = self.items[sampled_idxs]
        return sampled_items


class PatchedBatchSampler(torch.utils.data.sampler.BatchSampler, ub.NiceRepr):
    """
    A modification of the standard torch BatchSampler that allows for
    specification of ``num_batches=auto``

    Example:
        >>> data_source = torch.arange(64)
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=None)
        >>> batch_size = 10
        >>> drop_last = False
        >>> num_batches = 'auto'
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, num_batches)
        >>> assert len(list(batch_sampler)) == 7 == len(batch_sampler)
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, 3)
        >>> assert len(list(batch_sampler)) == 3 == len(batch_sampler)
        >>> batch_sampler = PatchedBatchSampler(sampler, batch_size, drop_last, 1)
        >>> assert len(list(batch_sampler)) == 1 == len(batch_sampler)
    """
    def __init__(self, sampler, batch_size, drop_last, num_batches='auto'):
        super().__init__(sampler, batch_size, drop_last)
        self.num_batches = num_batches

    def __len__(self):
        if self.drop_last:
            max_num_batches = len(self.sampler) // self.batch_size  # type: ignore
        else:
            max_num_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore
        if self.num_batches == 'auto':
            num_batches = max_num_batches
        else:
            num_batches = min(max_num_batches, self.num_batches)
        return num_batches

    def __iter__(self):
        num_batches = len(self)
        for bx, batch in zip(range(num_batches), super().__iter__()):
            yield batch


class PatchedRandomSampler(torch.utils.data.sampler.Sampler, ub.NiceRepr):
    r"""
    A modification of the standard torch Sampler that allows specification of
    ``num_samples``.

    See: https://github.com/pytorch/pytorch/pull/39214

    Example:
        >>> data_source = torch.arange(10)
        >>> # with replacement
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=None)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=1)
        >>> assert len(sampler) == 1 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=5)
        >>> assert len(sampler) == 5 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=10)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=True, num_samples=15)
        >>> assert len(sampler) == 15 == len(list(sampler))
        >>> # without replacement
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=None)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=1)
        >>> assert len(sampler) == 1 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=5)
        >>> assert len(sampler) == 5 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=10)
        >>> assert len(sampler) == 10 == len(list(sampler))
        >>> sampler = PatchedRandomSampler(data_source, replacement=False, num_samples=15)
        >>> assert len(sampler) == 10 == len(list(sampler))
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            num = len(self.data_source)
        else:
            if self.replacement:
                num = self._num_samples
            else:
                num = min(self._num_samples, len(self.data_source))
        return num

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist()[:self.num_samples])

    def __len__(self):
        return self.num_samples


class SubsetSampler(torch.utils.data.sampler.Sampler, ub.NiceRepr):
    """
    Generates sample indices based on a specified order / subset

    Example:
        >>> indices = list(range(10))
        >>> assert indices == list(SubsetSampler(indices))
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            yield idx

    def __len__(self):
        return len(self.indices)
