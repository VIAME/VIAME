"""
Deprecated. Ported to kwcoco
"""
import ubelt as ub
import functools


class ChannelSpec(ub.NiceRepr):
    """
    Parse and extract information about network input channel specs for
    early or late fusion networks.

    Notes:
        The pipe ('|') character represents an early-fused input stream, and
        order matters (it is non-communative).

        The comma (',') character separates different inputs streams/branches
        for a multi-stream/branch network which will be lated fused. Order does
        not matter

    TODO:
        - [ ] : normalize representations? e.g: rgb = r|g|b?
        - [ ] : rename to BandsSpec or SensorSpec?

    Example:
        >>> # Integer spec
        >>> ChannelSpec.coerce(3)
        <ChannelSpec(u0|u1|u2) ...>

        >>> # single mode spec
        >>> ChannelSpec.coerce('rgb')
        <ChannelSpec(rgb) ...>

        >>> # early fused input spec
        >>> ChannelSpec.coerce('rgb|disprity')
        <ChannelSpec(rgb|disprity) ...>

        >>> # late fused input spec
        >>> ChannelSpec.coerce('rgb,disprity')
        <ChannelSpec(rgb,disprity) ...>

        >>> # early and late fused input spec
        >>> ChannelSpec.coerce('rgb|ir,disprity')
        <ChannelSpec(rgb|ir,disprity) ...>

    Example:
        >>> from .data.channel_spec import *  # NOQA
        >>> self = ChannelSpec('gray')
        >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
        >>> self = ChannelSpec('rgb')
        >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity')
        >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
        >>> self = ChannelSpec('rgb|disparity,disparity')
        >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
        >>> self = ChannelSpec('rgb,disparity,flowx|flowy')
        >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))

    Example:
        >>> from .data.channel_spec import *  # NOQA
        >>> specs = [
        >>>     'rgb',              # and rgb input
        >>>     'rgb|disprity',     # rgb early fused with disparity
        >>>     'rgb,disprity',     # rgb early late with disparity
        >>>     'rgb|ir,disprity',  # rgb early fused with ir and late fused with disparity
        >>>     3,                  # 3 unknown channels
        >>> ]
        >>> for spec in specs:
        >>>     print('=======================')
        >>>     print('spec = {!r}'.format(spec))
        >>>     #
        >>>     self = ChannelSpec.coerce(spec)
        >>>     print('self = {!r}'.format(self))
        >>>     sizes = self.sizes()
        >>>     print('sizes = {!r}'.format(sizes))
        >>>     print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
        >>>     #
        >>>     item = self._demo_item((1, 1), rng=0)
        >>>     inputs = self.encode(item)
        >>>     components = self.decode(inputs)
        >>>     input_shapes = ub.map_vals(lambda x: x.shape, inputs)
        >>>     component_shapes = ub.map_vals(lambda x: x.shape, components)
        >>>     print('item = {}'.format(ub.urepr(item, precision=1)))
        >>>     print('inputs = {}'.format(ub.urepr(inputs, precision=1)))
        >>>     print('input_shapes = {}'.format(ub.urepr(input_shapes)))
        >>>     print('components = {}'.format(ub.urepr(components, precision=1)))
        >>>     print('component_shapes = {}'.format(ub.urepr(component_shapes, nl=1)))

    """

    _known = {
        'rgb': 'r|g|b'
    }

    _size_lut = {
        'rgb': 3,
    }

    def __init__(self, spec):
        # TODO: allow integer specs
        self.spec = spec
        self._info = {}

    def __nice__(self):
        return self.spec

    def __json__(self):
        return self.spec

    def __contains__(self, key):
        """
        Example:
            >>> 'disparity' in ChannelSpec('rgb,disparity,flowx|flowy')
            True
            >>> 'gray' in ChannelSpec('rgb,disparity,flowx|flowy')
            False
        """
        return key in self.unique()

    @property
    def info(self):
        self._info = {
            'spec': self.spec,
            'parsed': self.parse(),
            'unique': self.unique(),
            'normed': self.normalize(),
        }
        return self._info

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            self = data
            return self
        else:
            if isinstance(data, int):
                # we know the number of channels, but not their names
                spec = '|'.join(['u{}'.format(i) for i in range(data)])
            elif isinstance(data, str):
                spec = data
            else:
                raise TypeError(type(data))

            self = cls(spec)
            return self

    def parse(self):
        """
        Build internal representation
        """
        # commas break inputs into multiple streams
        stream_specs = self.spec.split(',')
        parsed = {ss: ss.split('|') for ss in stream_specs}
        return parsed

    def normalize(self):
        spec = self.spec
        stream_specs = spec.split(',')
        parsed = {ss: ss for ss in stream_specs}
        for k1 in parsed.keys():
            for k, v in self._known.items():
                parsed[k1] = parsed[k1].replace(k, v)
        parsed = {k: v.split('|') for k, v in parsed.items()}
        return parsed

    def keys(self):
        spec = self.spec
        stream_specs = spec.split(',')
        for spec in stream_specs:
            yield spec

    def streams(self):
        """
        Breaks this spec up into one spec for each early-fused input stream
        """
        streams = [self.__class__(spec) for spec in self.keys()]
        return streams

    def difference(self, other):
        """
        Set difference

        Example:
            >>> self = ChannelSpec('rgb|disparity,flowx|flowy')
            >>> other = ChannelSpec('rgb')
            >>> self.difference(other)
            >>> other = ChannelSpec('flowx')
            >>> self.difference(other)
        """
        assert len(list(other.keys())) == 1, 'can take diff with one stream'
        other_norm = ub.oset(ub.peek(other.normalize().values()))
        self_norm = self.normalize()

        new_streams = []
        for key, parts in self_norm.items():
            new_parts = ub.oset(parts) - ub.oset(other_norm)
            # shrink the representation of a complex r|g|b to an alias if
            # possible.
            # TODO: make this more efficient
            for alias, alias_spec in self._known.items():
                alias_parts = ub.oset(alias_spec.split('|'))
                index = subsequence_index(new_parts, alias_parts)
                if index is not None:
                    oset_delitem(new_parts, index)
                    oset_insert(new_parts, index.start, alias)
            new_stream = '|'.join(new_parts)
            new_streams.append(new_stream)
        new_spec = ','.join(new_streams)
        new = self.__class__(new_spec)
        return new

    def sizes(self):
        """
        Number of dimensions for each fused stream channel

        IE: The EARLY-FUSED channel sizes

        Example:
            >>> self = ChannelSpec('rgb|disparity,flowx|flowy')
            >>> self.sizes()
        """
        sizes = {
            key: sum(self._size_lut.get(part, 1) for part in vals)
            for key, vals in self.parse().items()
        }
        return sizes

    def unique(self):
        """
        Returns the unique channels that will need to be given or loaded
        """
        return set(ub.flatten(self.parse().values()))

    def _item_shapes(self, dims):
        """
        Expected shape for an input item

        Args:
            dims (Tuple[int, int]): the spatial dimension

        Returns:
            Dict[int, tuple]
        """
        item_shapes = {}
        parsed = self.parse()
        # normed = self.normalize()
        fused_keys = list(self.keys())
        for fused_key in fused_keys:
            components = parsed[fused_key]
            for mode_key in components:
                c = self._size_lut.get(mode_key, 1)
                shape = (c,) + tuple(dims)
                item_shapes[mode_key] = shape
        return item_shapes

    def _demo_item(self, dims=(4, 4), rng=None):
        """
        Create an input that satisfies this spec

        Returns:
            dict: an item like it might appear when its returned from the
                `__getitem__` method of a :class:`torch...Dataset`.

        Example:
            >>> dims = (1, 1)
            >>> ChannelSpec.coerce(3)._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('r|g|b|disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb|disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb,disaprity')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('rgb')._demo_item(dims, rng=0)
            >>> ChannelSpec.coerce('gray')._demo_item(dims, rng=0)
        """
        import torch
        import kwarray
        rng = kwarray.ensure_rng(rng)
        item_shapes = self._item_shapes(dims)
        item = {
            key: torch.from_numpy(rng.rand(*shape))
            for key, shape in item_shapes.items()
        }
        return item

    def encode(self, item, axis=0, impl=1):
        """
        Given a dictionary containing preloaded components of the network
        inputs, build a concatenated (fused) network representations of each
        input stream.

        Args:
            item (Dict[str, Tensor]): a batch item containing unfused parts.
                each key should be a single-stream (optionally early fused)
                channel key.
            axis (int, default=0): concatenation dimension

        Returns:
            Dict[str, Tensor]:
                mapping between input stream and its early fused tensor input.

        Example:
            >>> import torch
            >>> dims = (4, 4)
            >>> item = {
            >>>     'rgb': torch.rand(3, *dims),
            >>>     'disparity': torch.rand(1, *dims),
            >>>     'flowx': torch.rand(1, *dims),
            >>>     'flowy': torch.rand(1, *dims),
            >>> }
            >>> # Complex Case
            >>> self = ChannelSpec('rgb,disparity,rgb|disparity|flowx|flowy,flowx|flowy')
            >>> fused = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, fused)
            >>> print('input_shapes = {}'.format(ub.urepr(input_shapes, nl=1)))
            >>> # Simpler case
            >>> self = ChannelSpec('rgb|disparity')
            >>> fused = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, fused)
            >>> print('input_shapes = {}'.format(ub.urepr(input_shapes, nl=1)))

        Example:
            >>> # Case where we have to break up early fused data
            >>> from .data.channel_spec import *  # NOQA
            >>> import torch
            >>> dims = (40, 40)
            >>> item = {
            >>>     'rgb|disparity': torch.rand(4, *dims),
            >>>     'flowx': torch.rand(1, *dims),
            >>>     'flowy': torch.rand(1, *dims),
            >>> }
            >>> # Complex Case
            >>> self = ChannelSpec('rgb,disparity,rgb|disparity,rgb|disparity|flowx|flowy,flowx|flowy,flowx,disparity')
            >>> inputs = self.encode(item)
            >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
            >>> print('input_shapes = {}'.format(ub.urepr(input_shapes, nl=1)))

            >>> # xdoctest: +REQUIRES(--bench)
            >>> #self = ChannelSpec('rgb|disparity,flowx|flowy')
            >>> import timerit
            >>> ti = timerit.Timerit(100, bestof=10, verbose=2)
            >>> for timer in ti.reset('impl=simple'):
            >>>     with timer:
            >>>         inputs = self.encode(item, impl=0)
            >>> for timer in ti.reset('impl=minimize-concat'):
            >>>     with timer:
            >>>         inputs = self.encode(item, impl=1)

            import xdev
            _ = xdev.profile_now(self.encode)(item, impl=1)
        """
        import torch
        parsed = self.parse()
        # unique = self.unique()

        # TODO: This can be made much more efficient by determining if the
        # channels item can be directly translated to the result inputs. We
        # probably don't need to do the full decoding each and every time.

        if impl == 1:
            # Slightly more complex implementation that attempts to minimize
            # concat operations.
            item_keys = tuple(sorted(item.keys()))
            parsed_items = tuple(sorted([(k, tuple(v)) for k, v in parsed.items()]))
            new_fused_indices = _cached_single_fused_mapping(item_keys, parsed_items, axis=axis)

            fused = {}
            for key, idx_list in new_fused_indices.items():
                parts = [item[item_key][tuple(item_sl)] for item_key, item_sl in idx_list]
                if len(parts) == 1:
                    fused[key] = parts[0]
                else:
                    fused[key] = torch.cat(parts, dim=axis)
        elif impl == 0:
            # Simple implementation that always does the full break down of
            # item components.
            components = {}
            # Determine the layout of the channels in the input item
            key_specs = {key: ChannelSpec(key) for key in item.keys()}
            for key, spec in key_specs.items():
                decoded = spec.decode({key: item[key]}, axis=axis)
                for subkey, subval in decoded.items():
                    components[subkey] = subval

            fused = {}
            for key, parts in parsed.items():
                fused[key] = torch.cat([components[k] for k in parts], dim=axis)
        else:
            raise KeyError(impl)

        return fused

    def decode(self, inputs, axis=1):
        """
        break an early fused item into its components

        Args:
            inputs (Dict[str, Tensor]): dictionary of components

        Example:
            >>> import torch
            >>> dims = (4, 4)
            >>> components = {
            >>>     'rgb': torch.rand(3, *dims),
            >>>     'ir': torch.rand(1, *dims),
            >>> }
            >>> self = ChannelSpec('rgb|ir')
            >>> inputs = self.encode(components)
            >>> from .data import data_containers
            >>> item = {k: data_containers.ItemContainer(v, stack=True)
            >>>         for k, v in inputs.items()}
            >>> batch = data_containers.container_collate([item, item])
            >>> components = self.decode(batch)
        """
        parsed = self.parse()
        components = dict()
        for key, parts in parsed.items():
            idx1 = 0
            for part in parts:
                size = self._size_lut.get(part, 1)
                idx2 = idx1 + size
                fused = inputs[key]
                index = ([slice(None)] * axis + [slice(idx1, idx2)])
                component = fused[index]
                components[part] = component
                idx1 = idx2
        return components

    def component_indices(self, axis=2):
        """
        Look up component indices within fused streams

        Example:
            >>> import torch
            >>> dims = (4, 4)
            >>> inputs = ['flowx', 'flowy', 'disparity']
            >>> self = ChannelSpec('disparity,flowx|flowy')
            >>> component_indices = self.component_indices()
            >>> print('component_indices = {!r}'.format(component_indices))
        """
        parsed = self.parse()
        component_indices = dict()
        for key, parts in parsed.items():
            idx1 = 0
            for part in parts:
                size = self._size_lut.get(part, 1)
                idx2 = idx1 + size
                index = ([slice(None)] * axis + [slice(idx1, idx2)])
                idx1 = idx2
                component_indices[part] = (key, index)
        return component_indices


@functools.lru_cache(maxsize=None)
def _cached_single_fused_mapping(item_keys, parsed_items, axis=0):
    item_indices = {}
    for key in item_keys:
        key_idxs = _cached_single_stream_idxs(key, axis=axis)
        for subkey, subsl in key_idxs.items():
            item_indices[subkey] = subsl

    fused_indices = {}
    for key, parts in parsed_items:
        fused_indices[key] = [item_indices[k] for k in parts]

    new_fused_indices = {}
    for key, idx_list in fused_indices.items():
        # Determine which continguous slices can be merged into a
        # single slice
        prev_key = None
        prev_sl = None

        accepted = []
        accum = []
        for item_key, item_sl in idx_list:
            if prev_key == item_key:
                if prev_sl.stop == item_sl[-1].start and prev_sl.step == item_sl[-1].step:
                    accum.append((item_key, item_sl))
                    continue
            if accum:
                accepted.append(accum)
                accum = []
            prev_key = item_key
            prev_sl = item_sl[-1]
            accum.append((item_key, item_sl))
        if accum:
            accepted.append(accum)
            accum = []

        # Merge the accumulated contiguous slices
        new_idx_list = []
        for accum in accepted:
            if len(accum) > 1:
                item_key = accum[0][0]
                first = accum[0][1]
                last = accum[-1][1]
                new_sl = list(first)
                new_sl[-1] = slice(first[-1].start, last[-1].stop, last[-1].step)
                new_idx_list.append((item_key, new_sl))
            else:
                new_idx_list.append(accum[0])
        new_fused_indices[key] = new_idx_list
    return new_fused_indices


@functools.lru_cache(maxsize=None)
def _cached_single_stream_idxs(key, axis=0):
    """
    hack for speed

    axis = 0
    key = 'rgb|disparity'

    # xdoctest: +REQUIRES(--bench)
    import timerit
    ti = timerit.Timerit(100, bestof=10, verbose=2)
    for timer in ti.reset('time'):
        with timer:
            _cached_single_stream_idxs(key, axis=axis)
    for timer in ti.reset('time'):
        with timer:
            ChannelSpec(key).component_indices(axis=axis)
    """
    # concat operations.
    key_idxs = ChannelSpec(key).component_indices(axis=axis)
    return key_idxs


def subsequence_index(oset1, oset2):
    """
    Returns a slice into the first items indicating the position of
    the second items if they exist.

    This is a variant of the substring problem.

    Returns:
        None | slice

    Example:
        >>> oset1 = ub.oset([1, 2, 3, 4, 5, 6])
        >>> oset2 = ub.oset([2, 3, 4])
        >>> index = subsequence_index(oset1, oset2)
        >>> assert index

        >>> oset1 = ub.oset([1, 2, 3, 4, 5, 6])
        >>> oset2 = ub.oset([2, 4, 3])
        >>> index = subsequence_index(oset1, oset2)
        >>> assert not index
    """
    if len(oset2) == 0:
        base = 0
    else:
        item1 = oset2[0]
        try:
            base = oset1.index(item1)
        except (IndexError, KeyError):
            base = None

    index = None
    if base is not None:
        sl = slice(base, base + len(oset2))
        subset = oset1[sl]
        if subset == oset2:
            index = sl
    return index


def oset_insert(self, index, obj):
    """
    self = ub.oset()
    oset_insert(self, 0, 'a')
    oset_insert(self, 0, 'b')
    oset_insert(self, 0, 'c')
    oset_insert(self, 1, 'd')
    oset_insert(self, 2, 'e')
    oset_insert(self, 0, 'f')
    """
    if obj not in self:
        # Bump index of every item after the insert position
        for key in self.items[index:]:
            self.map[key] = self.map[key] + 1
        self.items.insert(index, obj)
        self.map[obj] = index


def oset_delitem(self, index):
    """
    for ubelt oset, todo contribute back to luminosoinsight

    >>> self = ub.oset([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> index = slice(3, 5)
    >>> oset_delitem(self, index)

    self = ub.oset(['r', 'g', 'b', 'disparity'])
    index = slice(0, 3)
    oset_delitem(self, index)

    """
    if isinstance(index, slice) and index == ub.orderedset.SLICE_ALL:
        self.clear()
    else:
        if ub.orderedset.is_iterable(index):
            to_remove = [self.items[i] for i in index]
        elif isinstance(index, slice) or hasattr(index, "__index__"):
            to_remove = self.items[index]
        else:
            raise TypeError("Don't know how to index an OrderedSet by %r" % index)

        if isinstance(to_remove, list):
            # Modified version of discard slightly more efficient for multiple
            # items
            remove_idxs = sorted([self.map[key] for key in to_remove], reverse=True)

            for key in to_remove:
                del self.map[key]

            for idx in remove_idxs:
                del self.items[idx]

            for k, v in self.map.items():
                # I think there is a more efficient way to do this?
                num_after = sum(v >= i for i in remove_idxs)
                if num_after:
                    self.map[k] = v - num_after
        else:
            self.discard(to_remove)

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/data/channel_spec.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
