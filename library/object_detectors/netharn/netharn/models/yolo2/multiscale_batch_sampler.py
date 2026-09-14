import torch.utils.data.sampler as torch_sampler
import torch


class MultiScaleBatchSampler(torch_sampler.BatchSampler):
    """
    Indicies returned in the batch are tuples indicating data index and scale
    index. Requires that dataset has a `multi_scale_inp_size` attribute.

    Args:
        sampler (Sampler): Base sampler. Must have a data_source attribute.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        resample_freq (int): how often to change scales. if None, then
            only one scale is used.

    Example:
        >>> import torch.utils.data as torch_data
        >>> class DummyDatset(torch_data.Dataset):
        >>>     def __init__(self):
        >>>         super(DummyDatset, self).__init__()
        >>>         self.multi_scale_inp_size = [1, 2, 3, 4]
        >>>     def __len__(self):
        >>>         return 34
        >>> batch_size = 16
        >>> data_source = DummyDatset()
        >>> sampler1 = torch_sampler.RandomSampler(data_source)
        >>> sampler2 = torch_sampler.SequentialSampler(data_source)
        >>> rand = MultiScaleBatchSampler(sampler1, resample_freq=10)
        >>> seq = MultiScaleBatchSampler(sampler2, resample_freq=None)
        >>> rand_idxs = list(iter(rand))
        >>> seq_idxs = list(iter(seq))
        >>> assert len(rand_idxs[0]) == 16
        >>> assert len(rand_idxs[0][0]) == 2
        >>> assert len(rand_idxs[-1]) == 2
        >>> assert {len({x[1] for x in xs}) for xs in rand_idxs} == {1}
        >>> assert {x[1] for xs in seq_idxs for x in xs} == {None}
    """

    def __init__(self, sampler, batch_size=16, drop_last=False,
                 resample_freq=10):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_scales = len(sampler.data_source.multi_scale_inp_size)
        self.resample_freq = resample_freq

    def __iter__(self):
        batch = []
        if self.resample_freq:
            scale_index = int(torch.rand(1) * self.num_scales)
        else:
            scale_index = None

        for idx in self.sampler:
            batch.append((int(idx), scale_index))
            if len(batch) == self.batch_size:
                yield batch
                if self.resample_freq and idx % self.resample_freq == 0:
                    # choose a new scale index every `resample_freq` batches
                    scale_index = int(torch.rand(1) * self.num_scales)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.models.yolo2.multiscale_batch_sampler all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
