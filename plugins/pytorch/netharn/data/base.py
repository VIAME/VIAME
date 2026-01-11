"""
DEPRECATE
"""
from torch.utils import data as torch_data


class DataMixin(object):
    def make_loader(self, *args, **kwargs):
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader
