import numpy as np
import torch
import torch.nn.functional as F
import itertools as it


def all_pairwise_distances(x, y=None, squared=False, approx=False):
    """
    Fast pairwise L2 squared distances between two sets of d-dimensional vectors

    Args:
        x (Tensor): an Nxd matrix
        y (Tensor, default=None): an optional Mxd matirx
        squared (bool, default=False): if True returns squared distances
        approx (bool, default=False): if True uses the quadratic distance approximation

    Returns:
        Tensor: dist: an NxM matrix where dist[i,j] is the square norm between
        x[i,:] and y[j,:] if y is not given then use 'y=x'.

        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    References:
        https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065

    SeeAlso:
        torch.nn.functional.pairwise_distance
        torch.nn.functional.pdist
        torch.norm(input[:, None] - input, dim=2, p=p)


    Example:
        >>> from .criterions.triplet import *
        >>> N, d = 5, 3
        >>> x = torch.rand(N, d)
        >>> dist = all_pairwise_distances(x)
        >>> assert dist.shape == (N, N)

        >>> a = x[None, :].expand(x.shape[0], -1, -1)
        >>> b = x[:, None].expand(-1, x.shape[0], -1)
        >>> real_dist = ((a - b) ** 2).sum(dim=2).sqrt_()
        >>> delta = (real_dist - dist.sqrt()).numpy()
        >>> assert delta.max() < 1e-5

    Example:
        >>> N, M, d = 5, 7, 3
        >>> x = torch.rand(N, d)
        >>> y = torch.rand(M, d)
        >>> dist = all_pairwise_distances(x, y)
        >>> assert dist.shape == (N, M)
    """
    if approx:
        return approx_pdist(x, y, squared=squared)
    else:
        return exact_pdist(x, y, squared=True)


def exact_pdist(x, y=None, squared=True):
    # Broadcast so all comparisons are NxN pairwise
    if y is None:
        y = x
    x_ = x[:, None, :]
    y_ = y[None, :, :]
    squared_dist = ((x_ - y_) ** 2).sum(dim=2)
    if squared:
        return squared_dist
    else:
        dist = squared_dist.sqrt_()
        return dist


def approx_pdist(x, y=None, squared=True):
    """
    This uses a quadratic expansion to approximate pairwise distances
    """
    x_norm = x.pow(2).sum(1).view(-1, 1)
    if y is None:
        y = x
        y_norm = x_norm.view(1, -1)
    else:
        y_norm = y.pow(2).sum(1).view(1, -1)
    yT = torch.transpose(y, 0, 1)
    xy = torch.mm(x, yT).mul_(-2.0)
    xy_norm = x_norm + y_norm
    squared_dist = xy_norm.add_(xy)
    squared_dist.clamp_(0, None)
    if squared:
        return squared_dist
    else:
        dist = squared_dist.sqrt_()
        return dist


def labels_to_adjacency_matrix(labels, symmetric=True, diagonal=True):
    """
    Construct an adjacency matrix of matching instances where `labels[i]` is
    the "name" or "identity" of the i-th item. The resulting matrix will have
    values adjm[i, j] == 1 if the i-th and j-th item have the same label and 0
    otherwise.

    Args:
        labels (ndarray): array of labels
        symmetric (bool, default=True): if False only the upper triangle of the
            matrix is populated.
        diagonal (bool, default=True): if False the diagonal is set to zero.

    Returns:
        ndarray: adjm : adjacency matrix

    Example:
        >>> labels = np.array([0, 0, 1, 1])
        >>> labels_to_adjacency_matrix(labels)
        array([[1, 1, 0, 0],
               [1, 1, 0, 0],
               [0, 0, 1, 1],
               [0, 0, 1, 1]], dtype=uint8)
        >>> labels_to_adjacency_matrix(labels, symmetric=False, diagonal=False)
        array([[0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1],
               [0, 0, 0, 0]], dtype=uint8)
    """
    import kwarray
    n = len(labels)
    adjm = np.zeros((n, n), dtype=np.uint8)
    unique_labels, groupxs = kwarray.group_indices(labels)
    pos_idxs = [(i, j) for g in groupxs for (i, j) in it.combinations(sorted(g), 2)]
    pos_multi_idxs = tuple(zip(*pos_idxs))
    adjm[pos_multi_idxs] = 1
    if symmetric:
        adjm += adjm.T
    if diagonal:
        np.fill_diagonal(adjm, 1)
    return adjm


class TripletLoss(torch.nn.TripletMarginLoss):
    """
    Triplet loss with either hard or soft margin

    Example:
        >>> dvecs = torch.randn(21, 128)
        >>> labels = (torch.randn(len(dvecs)) * 4).long()
        >>> info = TripletLoss().mine_negatives(dvecs, labels)
        >>> pos_dists, neg_dists = info['pos_dists'], info['neg_dists']
        >>> self = TripletLoss(soft=1)
        >>> loss = self(pos_dists, neg_dists)
        >>> loss_s = TripletLoss(soft=1, reduction='none')(pos_dists, neg_dists)
        >>> loss_h = TripletLoss(margin=1, reduction='none')(pos_dists, neg_dists)

    Ignore:
        >>> xdata = torch.linspace(-10, 10)
        >>> ydata = {
        >>>     'soft_margin[0]': F.softplus(0 + xdata).numpy(),
        >>>     'soft_margin[1]': F.softplus(1 + xdata).numpy(),
        >>>     'soft_margin[4]': F.softplus(4 + xdata).numpy(),
        >>>     'hard_margin[0]': (0 + xdata).clamp_(0, None).numpy(),
        >>>     'hard_margin[1]': (1 + xdata).clamp_(0, None).numpy(),
        >>>     'hard_margin[4]': (4 + xdata).clamp_(0, None).numpy(),
        >>> }
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.multi_plot(xdata.numpy(), ydata, fnum=1)
    """

    def __init__(self, margin=1.0, eps=1e-6, reduction='mean', soft=False):
        super(TripletLoss, self).__init__(margin=margin, eps=eps,
                                          reduction=reduction)
        self.soft = soft

    def mine_negatives(self, dvecs, labels, num=1, mode='hardest', eps=1e-9):
        """
        triplets =
             are a selection of anchor, positive, and negative annots
             chosen to be the hardest for each annotation (with a valid
             pos and neg partner) in the batch.

        Args:
            dvecs (Tensor): descriptor vectors for each item
            labels (Tensor): id-label for each descriptor vector
            num (int, default=1): number of negatives per positive combination
            mode (str, default='hardest'): method for selecting negatives
            eps (float, default=1e9): distance threshold for near duplicates

        Returns:
            Dict: info: containing neg_dists, pos_dists, triples, and dist

        CommandLine:
            xdoctest -m netharn.criterions.triplet TripletLoss.mine_negatives:1 --profile

        Example:
            >>> from .criterions.triplet import *
            >>> dvecs = torch.FloatTensor([
            ...     # Individual 1
            ...     [1.0, 0.0, 0.0, ],
            ...     [0.9, 0.1, 0.0, ],  # Looks like 2 [1]
            ...     # Individual 2
            ...     [0.0, 1.0, 0.0, ],
            ...     [0.0, 0.9, 0.1, ],  # Looks like 3 [3]
            ...     # Individual 3
            ...     [0.0, 0.0, 1.0, ],
            ...     [0.1, 0.0, 0.9, ],  # Looks like 1 [5]
            >>> ])
            >>> import itertools as it
            >>> labels = torch.LongTensor([0, 0, 1, 1, 2, 2])
            >>> num = 1
            >>> info = TripletLoss().mine_negatives(dvecs, labels, num)
            >>> print('info = {!r}'.format(info))
            >>> assert torch.all(info['pos_dists'] < info['neg_dists'])

        Example:
            >>> # xdoxctest: +SKIP
            >>> import itertools as it
            >>> p = 3
            >>> k = 10
            >>> d = p
            >>> mode = 'consistent'
            >>> mode = 'hardest'
            >>> for p, k in it.product(range(3, 13), range(2, 13)):
            >>>     d = p
            >>>     def make_individual_dvecs(i):
            >>>         vecs = torch.zeros((k, d))
            >>>         vecs[:, i] = torch.linspace(0.9, 1.0, k)
            >>>         return vecs
            >>>     dvecs = torch.cat([make_individual_dvecs(i) for i in range(p)], dim=0)
            >>>     labels = torch.LongTensor(np.hstack([[i] * k for i in range(p)]))
            >>>     num = 1
            >>>     info = TripletLoss().mine_negatives(dvecs, labels, num, mode=mode)
            >>>     if mode.startswith('hard'):
            >>>         assert torch.all(info['pos_dists'] < info['neg_dists'])
            >>>     base = k
            >>>     for a, p, n in info['triples']:
            >>>         x = a // base
            >>>         y = p // base
            >>>         z = n // base
            >>>         assert x == y, str([a, p, n])
            >>>         assert x != z, str([a, p, n])
        """
        import kwarray
        dist = all_pairwise_distances(dvecs, squared=True, approx=True)

        with torch.no_grad():
            labels_ = labels.numpy()

            symmetric = False
            pos_adjm = labels_to_adjacency_matrix(labels_, symmetric=symmetric,
                                                  diagonal=False)

            if symmetric:
                neg_adjm = 1 - pos_adjm
            else:
                neg_adjm = 1 - pos_adjm - pos_adjm.T
            np.fill_diagonal(neg_adjm, 0)

            # ignore near duplicates
            dist_ = dist.data.cpu().numpy()
            is_near_dup = np.where(dist_ < eps)
            neg_adjm[is_near_dup] = 0
            pos_adjm[is_near_dup] = 0

            # Filter out any anchor row that does not have both a positive and
            # a negative match
            flags = np.any(pos_adjm, axis=1) & np.any(neg_adjm, axis=1)

            anchors_idxs = np.where(flags)[0]
            pos_adjm_ = pos_adjm[flags].astype(bool)
            neg_adjm_ = neg_adjm[flags].astype(bool)

            if mode == 'hardest':
                # Order each anchors positives and negatives by increasing distance
                sortx_ = dist_[flags].argsort(axis=1)
                pos_cands_list = [x[m[x]] for x, m in zip(sortx_, pos_adjm_)]
                neg_cands_list = [x[n[x]] for x, n in zip(sortx_, neg_adjm_)]
                triples = []
                backup = []
                _iter = zip(anchors_idxs, pos_cands_list, neg_cands_list)
                for (anchor_idx, pos_cands, neg_cands) in _iter:
                    # Take `num` hardest negative pairs for each positive pair
                    num_ = min(len(neg_cands), len(pos_cands), num)
                    pos_idxs = pos_cands
                    neg_idxs = neg_cands[:num_]

                    anchor_dists = dist_[anchor_idx]

                    if True:
                        pos_dists = anchor_dists[pos_idxs]
                        neg_dists = anchor_dists[neg_idxs]

                        # Ignore any triple that satisfies the margin
                        # constraint
                        losses = pos_dists[:, None] - neg_dists[None, :] + self.margin
                        ilocs, jlocs = np.where(losses > 0)

                        if len(ilocs) > 0:
                            valid_pos_idxs = pos_idxs[ilocs].tolist()
                            valid_neg_idxs = neg_idxs[jlocs].tolist()

                            for pos_idx, neg_idx in zip(valid_pos_idxs, valid_neg_idxs):
                                triples.append((anchor_idx, pos_idx, neg_idx))
                        elif len(triples) == 0:
                            # Take everything because we might need a backup
                            for pos_idx, neg_idx in it.product(pos_idxs, neg_idxs):
                                backup.append((anchor_idx, pos_idx, neg_idx))
                    else:
                        for pos_idx, neg_idx in it.product(pos_idxs, neg_idxs):
                            # Only take items that will contribute positive loss
                            d_ap = anchor_dists[pos_idx]
                            d_an = anchor_dists[neg_idx]
                            loss = d_ap - d_an + self.margin
                            if loss > 0:
                                triples.append((anchor_idx, pos_idx, neg_idx))
                            elif len(triples) == 0:
                                backup.append((anchor_idx, pos_idx, neg_idx))

            elif mode == 'moderate':
                pos_cands_list = [np.where(m)[0].tolist() for m in pos_adjm_]
                neg_cands_list = [np.where(n)[0].tolist() for n in neg_adjm_]
                triples = []
                backup = []
                _iter = zip(anchors_idxs, pos_cands_list, neg_cands_list)
                for (anchor_idx, pos_cands, neg_cands) in _iter:
                    # Take `num` moderate negative pairs for each positive pair
                    # Only take items that will contribute positive loss
                    # but try not to take any that are too hard.
                    anchor_dists = dist_[anchor_idx]
                    neg_dists = anchor_dists[neg_cands]

                    for pos_idx in pos_cands:
                        pos_dist = anchor_dists[pos_idx]
                        losses = pos_dist - neg_dists + self.margin

                        # valid_negs = np.where((losses < margin) & (losses > 0))[0]
                        valid_negs = np.where(losses > 0)[0]
                        if len(valid_negs):
                            neg_idx = neg_cands[np.random.choice(valid_negs)]
                            triples.append((anchor_idx, pos_idx, neg_idx))
                        elif len(triples) == 0:
                            # We try to always return valid triples so, create
                            # a backup set in case we cant find any valid
                            # candidates
                            neg_idx = neg_cands[0]
                            backup.append((anchor_idx, pos_idx, neg_idx))

            elif mode == 'consistent':
                # Choose the same triples every time
                rng = kwarray.ensure_rng(0)
                pos_cands_list = [kwarray.shuffle(np.where(m)[0], rng=rng)
                                  for m in pos_adjm_]
                neg_cands_list = [kwarray.shuffle(np.where(n)[0], rng=rng)
                                  for n in neg_adjm_]
                triples = []
                _iter = zip(anchors_idxs, pos_cands_list, neg_cands_list)
                for (anchor_idx, pos_cands, neg_cands) in _iter:
                    num_ = min(len(neg_cands), len(pos_cands), num)
                    pos_idxs = pos_cands
                    for pos_idx in pos_idxs:
                        neg_idx = rng.choice(neg_cands)
                        triples.append((anchor_idx, pos_idx, neg_idx))
                rng.shuffle(triples)
            else:
                raise KeyError(mode)

            if len(triples) == 0:
                triples = backup
                if len(backup) == 0:
                    raise RuntimeError('unable to mine triples')

            triples = np.array(triples)
            A, P, N = triples.T

            if 0 and __debug__:
                if labels is not None:
                    for a, p, n in triples:
                        na_ = labels[a]
                        np_ = labels[p]
                        nn_ = labels[n]
                        assert na_ == np_
                        assert np_ != nn_

        # Note these distances are approximate distances, but they should be
        # good enough to backprop through (if not see alternate commented code).
        pos_dists = dist[A, P]
        neg_dists = dist[A, N]
        # pos_dists = (dvecs[A] - dvecs[P]).pow(2).sum(1)
        # neg_dists = (dvecs[A] - dvecs[N]).pow(2).sum(1)

        info = {
            'pos_dists': pos_dists,
            'neg_dists': neg_dists,
            'triples': triples,
            'dist': dist,
        }
        return info

    @classmethod
    def _hard_triples2(cls, dvecs, labels, margin=1):
        """
        Slow implementation of hard triples. Minimally modified from [1]

        References:
            ..[1] https://github.com/adambielski/siamese-triplet/blob/master/utils.py#L58
        """
        from itertools import combinations
        with torch.no_grad():
            embeddings = dvecs
            pdist = approx_pdist
            distance_matrix = pdist(embeddings)
            distance_matrix = distance_matrix.cpu()

            def semihard_negative(loss_values, margin=margin):
                semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
                return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

            def hardest_negative(loss_values):
                hard_negative = np.argmax(loss_values)
                return hard_negative if loss_values[hard_negative] > 0 else None

            negative_selection_fn = hardest_negative

            labels_ = labels.cpu().data.numpy()
            triplets = []

            for label in set(labels_):
                label_mask = (labels_ == label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue
                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
                anchor_positives = np.array(anchor_positives)

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                    loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + margin
                    loss_values = loss_values.data.cpu().numpy()
                    hard_negative = negative_selection_fn(loss_values)
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

            if len(triplets) == 0:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

            triplets = np.array(triplets)

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)

        info = {
            'pos_dists': ap_distances,
            'neg_dists': an_distances,
            'triples': triplets,
            'dist': distance_matrix,
        }
        return info

    def _softmargin(self, pos_dists, neg_dists):
        x = pos_dists - neg_dists
        loss = F.softplus(x + self.margin)  # log(1 + exp(x))
        return loss

    def _hardmargin(self, pos_dists, neg_dists):
        x = pos_dists - neg_dists
        # loss = (self.margin + x).clamp_(0, None)  # [margin + x]_{+}
        loss = F.relu(x + self.margin)  # [margin + x]_{+}
        return loss

    def forward(self, pos_dists, neg_dists):
        """
        Args:
            pos_dists (Tensor) [A x 1]: distance between the anchor and a positive
            neg_dists (Tensor) [A x 1]: distance between the anchor and a negative

        Notes:
            soft_triplet_loss = (1 / len(triplets)) * sum(log(1 + exp(dist[a, p] - dist[a, n])) for a, p, n in triplets)

        Example:
            >>> from .criterions.triplet import *
            >>> xbasis = ybasis = np.linspace(0, 5, 16)
            >>> pos_dists, neg_dists = map(torch.FloatTensor, np.meshgrid(xbasis, ybasis))
            >>> hard_loss = TripletLoss(reduction='none', soft=False).forward(pos_dists, neg_dists)
            >>> soft_loss = TripletLoss(reduction='none', soft=True).forward(pos_dists, neg_dists)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.plot_surface3d(pos_dists, neg_dists, hard_loss.numpy(), pnum=(1, 2, 1),
            >>>                       xlabel='d_pos', ylabel='d_neg', zlabel='hard-loss', contour=True, cmap='magma')
            >>> kwplot.plot_surface3d(pos_dists, neg_dists, soft_loss.numpy(), pnum=(1, 2, 2),
            >>>                       xlabel='d_pos', ylabel='d_neg', zlabel='hard-loss', contour=True, cmap='magma')
        """
        if self.soft:
            loss = self._softmargin(pos_dists, neg_dists)
        else:
            loss = self._hardmargin(pos_dists, neg_dists)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction != 'none':
            raise KeyError(self.reduction)
        return loss
