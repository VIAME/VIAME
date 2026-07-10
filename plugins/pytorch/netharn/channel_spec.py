from viame.pytorch.netharn.data.channel_spec import ChannelSpec  # NOQA
# import ubelt as ub
# class ChannelSpec(ub.NiceRepr):
#     """
#     Parse and extract information about network input channel specs for
#     early or late fusion networks.

#     Notes:
#         The pipe ('|') character represents an early-fused input stream, and
#         order matters (it is non-communative).

#         The comma (',') character separates different inputs streams/branches
#         for a multi-stream/branch network which will be lated fused. Order does
#         not matter

#     TODO:
#         - [ ] : normalize representations? e.g: rgb = r|g|b?
#         - [ ] : rename to BandsSpec or SensorSpec?

#     Example:
#         >>> from .channel_spec import *  # NOQA
#         >>> self = ChannelSpec('gray')
#         >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
#         >>> self = ChannelSpec('rgb')
#         >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
#         >>> self = ChannelSpec('rgb|disparity')
#         >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
#         >>> self = ChannelSpec('rgb|disparity,disparity')
#         >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
#         >>> self = ChannelSpec('rgb,disparity,flowx|flowy')
#         >>> print('self.info = {}'.format(ub.urepr(self.info, nl=1)))
#     """

#     _known = {
#         'rgb': 'r|g|b'
#     }

#     _size_lut = {
#         'rgb': 3,
#     }

#     def __init__(self, spec):
#         self.spec = spec
#         self._info = {}

#     def __nice__(self):
#         return self.spec

#     def __json__(self):
#         return self.spec

#     def __contains__(self, key):
#         """
#         Example:
#             >>> 'disparity' in ChannelSpec('rgb,disparity,flowx|flowy')
#             True
#             >>> 'gray' in ChannelSpec('rgb,disparity,flowx|flowy')
#             False
#         """
#         return key in self.unique()

#     @property
#     def info(self):
#         self._info = {
#             'spec': self.spec,
#             'parsed': self.parse(),
#             'unique': self.unique(),
#             'normed': self.normalize(),
#         }
#         return self._info

#     @classmethod
#     def coerce(cls, data):
#         if isinstance(data, cls):
#             self = data
#         else:
#             self = cls(data)
#         return self

#     def parse(self):
#         """
#         Build internal representation
#         """
#         # commas break inputs into multiple streams
#         stream_specs = self.spec.split(',')
#         parsed = {ss: ss.split('|') for ss in stream_specs}
#         return parsed

#     def normalize(self):
#         spec = self.spec
#         stream_specs = spec.split(',')
#         parsed = {ss: ss for ss in stream_specs}
#         for k1 in parsed.keys():
#             for k, v in self._known.items():
#                 parsed[k1] = parsed[k1].replace(k, v)
#         parsed = {k: v.split('|') for k, v in parsed.items()}
#         return parsed

#     def keys(self):
#         spec = self.spec
#         stream_specs = spec.split(',')
#         for spec in stream_specs:
#             yield spec

#     def sizes(self):
#         """
#         Number of dimensions for each fused stream channel

#         Example:
#             >>> self = ChannelSpec('rgb|disparity,flowx|flowy')
#             >>> self.sizes()
#         """
#         sizes = {
#             key: sum(self._size_lut.get(part, 1) for part in vals)
#             for key, vals in self.parse().items()
#         }
#         return sizes

#     def unique(self):
#         """
#         Returns the unique channels that will need to be given or loaded
#         """
#         return set(ub.flatten(self.parse().values()))

#     def encode(self, item, axis=0):
#         """
#         Given a dictionary containing preloaded components of the network
#         inputs, build a concatenated network representations of each input
#         stream.

#         Args:
#             item (dict): a batch item
#             axis (int, default=0): concatenation dimension

#         Returns:
#             Dict[str, Tensor]: mapping between input stream and its early fused
#                 tensor input.

#         Example:
#             >>> import torch
#             >>> dims = (4, 4)
#             >>> item = {
#             >>>     'rgb': torch.rand(3, *dims),
#             >>>     'disparity': torch.rand(1, *dims),
#             >>>     'flowx': torch.rand(1, *dims),
#             >>>     'flowy': torch.rand(1, *dims),
#             >>> }
#             >>> # Complex Case
#             >>> self = ChannelSpec('rgb,disparity,rgb|disparity|flowx|flowy,flowx|flowy')
#             >>> inputs = self.encode(item)
#             >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
#             >>> print('input_shapes = {}'.format(ub.urepr(input_shapes, nl=1)))
#             >>> # Simpler case
#             >>> self = ChannelSpec('rgb|disparity')
#             >>> inputs = self.encode(item)
#             >>> input_shapes = ub.map_vals(lambda x: x.shape, inputs)
#             >>> print('input_shapes = {}'.format(ub.urepr(input_shapes, nl=1)))
#         """
#         import torch
#         inputs = dict()
#         parsed = self.parse()
#         unique = self.unique()
#         components = {k: item[k] for k in unique}
#         for key, parts in parsed.items():
#             inputs[key] = torch.cat([components[k] for k in parts], dim=axis)
#         return inputs

#     def decode(self, inputs, axis=1):
#         """
#         break an early fused item into its components

#         Example:
#             >>> import torch
#             >>> dims = (4, 4)
#             >>> components = {
#             >>>     'rgb': torch.rand(3, *dims),
#             >>>     'ir': torch.rand(1, *dims),
#             >>> }
#             >>> self = ChannelSpec('rgb|ir')
#             >>> inputs = self.encode(components)
#             >>> from bioharn import data_containers
#             >>> item = {k: data_containers.ItemContainer(v, stack=True)
#             >>>         for k, v in inputs.items()}
#             >>> batch = data_containers.container_collate([item, item])
#             >>> components = self.decode(batch)
#         """
#         parsed = self.parse()
#         components = dict()
#         for key, parts in parsed.items():
#             idx1 = 0
#             for part in parts:
#                 size = self._size_lut.get(part, 1)
#                 idx2 = idx1 + size
#                 fused = inputs[key]
#                 index = ([slice(None)] * axis + [slice(idx1, idx2)])
#                 component = fused[index]
#                 components[part] = component
#                 idx1 = idx2
#         return components
